"""In-memory fixtures for exercising the real tool loop without a provider or database."""

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel

from work_management_ai.agents.orchestrator.contracts import (
    ActiveConversationContext,
    OrchestratorInput,
)
from work_management_ai.agents.orchestrator.evaluators.tool_loop import (
    EVALUATOR_ID,
    ORCHESTRATOR_TOOL_LOOP_EVALUATORS,
)
from work_management_ai.agents.orchestrator.tool_loop import ToolLoopOrchestrator
from work_management_ai.agents.orchestrator.workflows.tool_loop_graph import WORKFLOW_ID
from work_management_ai.evaluation.context_exploration import CONTEXT_EXPLORATION_EVALUATORS
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.runtime.agent_registry import AgentRegistry
from work_management_ai.runtime.context_exploration import (
    ExplorationProjector,
)
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentId,
    ContextReference,
    JsonValue,
)
from work_management_ai.runtime.manifests import (
    ToolLoopAgentManifest,
    load_yaml_resource,
)
from work_management_ai.runtime.run_budget import (
    BudgetError,
    BudgetReceipt,
    ChildAllocation,
    RemainingBudget,
    UsageReservation,
)
from work_management_ai.runtime.session_history import (
    RevalidatedContext,
    SessionHistory,
)
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationReceipt,
    PayloadKind,
    SessionRecord,
    SettledInvocation,
)
from work_management_ai.runtime.skill_loading import (
    ApprovedPackageResources,
    SkillLoader,
    SkillLoaderBackend,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_catalog import CatalogAccess, ToolCatalog
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    BackendToolResult,
    PayloadReference,
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopCheckpoint,
    ToolLoopLimits,
    ToolLoopUsage,
    VersionPin,
)
from work_management_ai.runtime.tool_pipeline import PreparedTool, ToolPipeline
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context.contracts import (
    ContextReadObservation,
)

FIXTURE_TIME = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)


# --- In-memory durable fakes reusing real runtime ports -----------------------------------


class InMemorySessionLog:
    def __init__(self) -> None:
        self.records: list[SessionRecord] = []
        self.payloads: dict[UUID, JsonValue] = {}
        self.checkpoints: dict[UUID, ToolLoopCheckpoint] = {}
        self.invocations: dict[UUID, InvocationReceipt] = {}
        self.started: set[UUID] = set()

    async def initialize(self, scope: RunScope, state: ToolLoopCheckpoint) -> None:
        self.checkpoints[scope.orchestration_run_id] = state

    async def append(
        self, scope: RunScope, event: EventDraft, *, expected_sequence: int, lease: RunLease
    ) -> SessionRecord:
        assert len(self.records) == expected_sequence
        rec = SessionRecord(
            **event.model_dump(),
            scope=scope,
            sequence=expected_sequence + 1,
            occurred_at=FIXTURE_TIME,
        )
        self.records.append(rec)
        if event.kind == "tool.requested":
            assert event.invocation is not None and event.payload_ref is not None
            self.invocations[event.invocation.invocation_id] = InvocationReceipt(
                scope=scope,
                descriptor=event.invocation,
                requested_event_id=event.event_id,
                requested_sequence=rec.sequence,
                arguments_ref=event.payload_ref,
            )
        return rec

    async def read(self, scope: RunScope, *, after_sequence: int) -> tuple[SessionRecord, ...]:
        return tuple(r for r in self.records if r.scope == scope and r.sequence > after_sequence)

    async def checkpoint(
        self, scope: RunScope, state: ToolLoopCheckpoint, *, lease: RunLease
    ) -> None:
        self.checkpoints[scope.orchestration_run_id] = state

    async def load_checkpoint(self, scope: RunScope) -> ToolLoopCheckpoint | None:
        return self.checkpoints.get(scope.orchestration_run_id)

    async def claim(
        self, scope: RunScope, *, job_id: UUID, worker_id: str, lease_seconds: int
    ) -> RunLease:
        return RunLease(
            scope=scope,
            lease_id=uuid4(),
            job_id=job_id,
            worker_id=worker_id,
            fencing_token=1,
            expires_at=FIXTURE_TIME + timedelta(seconds=lease_seconds),
        )

    async def renew(self, scope: RunScope, lease: RunLease, *, lease_seconds: int) -> RunLease:
        return lease.model_copy(
            update={"expires_at": FIXTURE_TIME + timedelta(seconds=lease_seconds)}
        )

    async def get_invocation(
        self, scope: RunScope, invocation_id: UUID
    ) -> InvocationReceipt | None:
        rec = self.invocations.get(invocation_id)
        return rec if rec is not None and rec.scope == scope else None

    async def start_invocation(
        self, scope: RunScope, receipt: InvocationReceipt, *, lease: RunLease
    ) -> bool:
        iid = receipt.descriptor.invocation_id
        if iid in self.started:
            return False
        self.started.add(iid)
        return True

    async def settle_invocation(
        self,
        scope: RunScope,
        receipt: InvocationReceipt,
        value: SettledInvocation,
        *,
        lease: RunLease,
    ) -> SessionRecord:
        kind: Literal["run.waiting", "tool.failed", "tool.completed"] = (
            "run.waiting"
            if value.status == "DEFERRED"
            else "tool.failed"
            if value.status == "FAILED"
            else "tool.completed"
        )
        rec = await self.append(
            scope,
            EventDraft(
                event_id=uuid5(receipt.descriptor.invocation_id, f"settled:{value.status}"),
                kind=kind,
                invocation=receipt.descriptor,
                settlement=value,
                payload_ref=value.payload_ref,
            ),
            expected_sequence=len(self.records),
            lease=lease,
        )
        self.invocations[receipt.descriptor.invocation_id] = receipt.model_copy(
            update={"settlement": value, "settled_event_id": rec.event_id}
        )
        return rec

    async def put_payload(
        self,
        scope: RunScope,
        body: JsonValue,
        *,
        classification: Literal["RAW_CONTEXT"],
        kind: PayloadKind,
        lease: RunLease,
        ttl_seconds: int | None = None,
    ) -> PayloadReference:
        import hashlib

        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        pid = uuid4()
        self.payloads[pid] = body
        return PayloadReference(
            payload_id=pid,
            organization_id=scope.organization_id,
            fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
            expires_at=FIXTURE_TIME + timedelta(seconds=ttl_seconds or 3600),
        )

    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> JsonValue:
        if reference.organization_id != scope.organization_id:
            raise ValueError("TENANT_MISMATCH")
        return self.payloads[reference.payload_id]


class InMemoryBudget:
    def __init__(self, limits: ToolLoopLimits | None = None) -> None:
        self.limits = limits or ToolLoopLimits()
        self.root_calls = 0
        self.total_calls = 0
        self.tool_calls = 0
        self.handoffs = 0
        self.segments = 0
        self.reservations: list[UsageReservation] = []

    async def recover_tool(
        self,
        scope: RunScope,
        operation_id: UUID,
        *,
        lease: RunLease,
        interval_id: UUID,
        allocation_id: UUID | None,
    ) -> BudgetReceipt:
        for r in self.reservations:
            if r.operation_id == operation_id:
                return BudgetReceipt(scope=scope, lease=lease, reservation=r, replayed=True)
        raise BudgetError("NOT_FOUND")

    async def reserve(
        self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt:
        if reservation.kind == "INTERVAL":
            if self.segments >= self.limits.segments:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.segments += 1
        elif reservation.kind == "MODEL":
            if self.total_calls >= self.limits.total_model_requests or (
                reservation.allocation_id is None
                and self.root_calls >= self.limits.root_model_requests
            ):
                raise BudgetError("BUDGET_EXHAUSTED")
            self.total_calls += 1
            if reservation.allocation_id is None:
                self.root_calls += 1
        elif reservation.kind == "TOOL":
            if self.tool_calls >= self.limits.total_tool_calls:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.tool_calls += 1
        elif reservation.kind == "HANDOFF":
            if self.handoffs >= self.limits.handoffs:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.handoffs += 1
        self.reservations.append(reservation)
        return BudgetReceipt(scope=scope, lease=lease, reservation=reservation)

    async def allocate_child(
        self, scope: RunScope, allocation: ChildAllocation, *, lease: RunLease
    ) -> ChildAllocation:
        return allocation

    async def close_child(self, scope: RunScope, allocation_id: UUID, *, lease: RunLease) -> None:
        return None

    async def settle(
        self, scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
    ) -> None:
        return None

    async def remaining(self, scope: RunScope) -> RemainingBudget:
        return RemainingBudget(
            usage=ToolLoopUsage(
                root_model_requests=self.root_calls,
                total_model_requests=self.total_calls,
                total_tool_calls=self.tool_calls,
                handoffs=self.handoffs,
                segments=self.segments,
            ),
            available=ToolLoopUsage(
                root_model_requests=max(0, self.limits.root_model_requests - self.root_calls),
                total_model_requests=max(0, self.limits.total_model_requests - self.total_calls),
                total_tool_calls=max(0, self.limits.total_tool_calls - self.tool_calls),
                handoffs=max(0, self.limits.handoffs - self.handoffs),
                input_tokens=self.limits.input_tokens,
                output_tokens=self.limits.output_tokens,
                segments=max(0, self.limits.segments - self.segments),
                active_seconds=float(self.limits.segment_seconds),
            ),
        )


class AllowAllPolicy:
    async def pre_execute(
        self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
    ) -> PreparedTool:
        return PreparedTool(arguments=arguments, policy_version="policy-v1")

    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None:
        return None

    async def post_execute(self, context: ToolInvocationContext, result: BackendToolResult) -> None:
        return None

    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None:
        return None


class AllowContextAccess:
    async def revalidate(self, scope: RunScope, refs: tuple[Any, ...]) -> RevalidatedContext:
        return RevalidatedContext(allowed=True, references=refs, fresh=True)


class CountingApprovedResources(ApprovedPackageResources):
    def __init__(self, packages: dict[str, str]) -> None:
        super().__init__(packages)
        self.bodies_read = 0

    def read(self, manifest: Any, resource: Any) -> bytes:
        self.bodies_read += 1
        return super().read(manifest, resource)


class FixtureToolBackend:
    def __init__(
        self,
        handler: Callable[[ToolInvocationContext, BaseModel], Awaitable[BackendToolResult]],
        reconciler: Callable[
            [ToolInvocationContext, BaseModel], Awaitable[BackendToolResult | None]
        ]
        | None = None,
    ) -> None:
        self._handler = handler
        self._reconciler = reconciler
        self.calls: list[tuple[ToolInvocationContext, BaseModel]] = []
        self.reconcile_calls: list[tuple[ToolInvocationContext, BaseModel]] = []

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        self.calls.append((context, arguments))
        return await self._handler(context, arguments)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        self.reconcile_calls.append((context, arguments))
        if self._reconciler is None:
            return None
        return await self._reconciler(context, arguments)


class StaticAccessPort:
    def __init__(self, scope: RunScope, tools: frozenset[str], skills: frozenset[str]) -> None:
        self._facts = CatalogAccess(
            scope=scope,
            role="MANAGER",
            active_phase=5,
            enabled=True,
            permission_revision="perm-1",
            activation_revision="act-1",
            allowed_tools=tools,
            active_tools=tools,
            allowed_skills=skills,
        )

    def current(self, scope: RunScope) -> CatalogAccess:
        return self._facts


class ReadyPort:
    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        return True


def make_harness(
    *,
    responses: tuple[object, ...],
    backends: dict[str, Any] | None = None,
    limits: ToolLoopLimits | None = None,
    locale: Literal["vi", "en"] = "vi",
    message: str = "Xin chào!",
    policy: AllowAllPolicy | None = None,
    use_packaged_manifest: bool = False,
    model_ref: str = "mock:orchestrator-v2",
    specialists: tuple[tuple[AgentId, str], ...] = (
        (AgentId.WORK_INTELLIGENCE, "Answers questions from verified work records."),
        (AgentId.ASSIGNMENT, "Analyzes workload and prepares team recommendations."),
        (AgentId.PLANNING, "Prepares and revises plans for review."),
    ),
) -> tuple[
    ToolLoopOrchestrator,
    OrchestratorInput,
    RunScope,
    RunLease,
    MockModelGateway,
    InMemorySessionLog,
    CountingApprovedResources,
    ExplorationProjector,
]:
    org_id, member_id, conv_id, turn_id, run_id = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    scope = RunScope(
        organization_id=org_id,
        membership_id=member_id,
        conversation_id=conv_id,
        turn_id=turn_id,
        orchestration_run_id=run_id,
    )
    lease = RunLease(
        scope=scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="worker-t10",
        fencing_token=1,
        expires_at=FIXTURE_TIME + timedelta(seconds=120),
    )
    value = OrchestratorInput(
        orchestration_run_id=run_id,
        conversation_id=conv_id,
        turn_id=turn_id,
        message=message,
        locale=locale,
        actor=ActorReference(organization_id=org_id, membership_id=member_id),
        active_context=ActiveConversationContext(recent_messages=()),
    )

    skills = SkillRegistry.from_resources(
        (("work_management_ai.skills.explore_work_context", "skill.yaml"),)
    )
    tools = ToolRegistry.from_resources(
        (
            ("work_management_ai.tools.skills", "load.yaml"),
            ("work_management_ai.tools.context", "projects.v103.yaml"),
            ("work_management_ai.tools.context", "project.yaml"),
            ("work_management_ai.tools.context", "actor.yaml"),
            ("work_management_ai.tools.delegation", "delegate.v2.yaml"),
            ("work_management_ai.tools.automation", "tool.v102.yaml"),
        )
    )

    async def _noop_eval() -> object:
        return {"passed": True}

    agents = AgentRegistry(
        skill_registry=skills,
        tool_registry=tools,
        evaluator_ids=frozenset(),
        evaluator_bindings={
            **(
                CONTEXT_EXPLORATION_EVALUATORS
                if use_packaged_manifest
                else {"context_exploration@1": _noop_eval}
            ),
            **ORCHESTRATOR_TOOL_LOOP_EVALUATORS,
        },
    )
    manifest = ToolLoopAgentManifest.model_validate(
        {
            "schema_version": "3.0",
            "agent": {
                "id": "orchestrator",
                "name": "Orchestrator Tool Loop",
                "version": "2.0.0",
                "owner": "ai-runtime",
                "activation_phase": 2,
            },
            "capabilities": [
                "orchestration.plan",
                "orchestration.delegate",
                "orchestration.synthesize",
                "orchestration.explore",
            ],
            "contracts": {
                "input": "work_management_ai.agents.orchestrator.contracts.OrchestratorInput",
                "output": (
                    "work_management_ai.agents.orchestrator.tool_loop_contracts.OrchestratorOutputV2"
                ),
                "handoff": "work_management_ai.runtime.contracts.AgentHandoff",
            },
            "permissions": {
                "roles": ["ADMIN", "MANAGER", "EMPLOYEE"],
                "tenant_scope": "actor_membership",
                "risk_ceiling": "PROPOSAL_ONLY",
            },
            "runtime": {
                "workflow": WORKFLOW_ID,
                "checkpoint": "durable",
                "model_policy": "primary_orchestrator",
                "limits": (limits or ToolLoopLimits()).model_dump(),
            },
            "allowed_skills": ["explore_work_context@1"],
            "allowed_tools": [
                "load_skill@1",
                "explore_projects@1",
                "read_project_context@1",
                "read_actor_context@1",
                "delegate_agent@2",
            ],
            "approval": {"produced_writes": "NEVER", "can_self_approve": False},
            "fallback": {"strategy": "SAFE_MANUAL_FALLBACK"},
            "evaluators": [EVALUATOR_ID],
        }
    )
    if use_packaged_manifest:
        manifest = load_yaml_resource(
            "work_management_ai.agents.orchestrator", "agent.v2.yaml", ToolLoopAgentManifest
        )
    registered_agent = agents.register_tool_loop(manifest)
    allowed_tools = frozenset(manifest.allowed_tools)
    allowed_skills = frozenset(manifest.allowed_skills)
    catalog = ToolCatalog(
        agent=registered_agent,
        tools=tools,
        access=StaticAccessPort(scope, allowed_tools, allowed_skills),
        readiness=ReadyPort(),
    )
    resources = CountingApprovedResources(
        {"explore_work_context@1": "work_management_ai.skills.explore_work_context"}
    )
    skill_loader = SkillLoader(
        skills=skills,
        resources=resources,
        catalog=catalog,
        model_ref=model_ref,
    )
    store = InMemorySessionLog()
    history = SessionHistory(store, now=lambda: FIXTURE_TIME)
    budget = InMemoryBudget(limits=limits)

    all_backends: dict[str, Any] = {"load_skill@1": SkillLoaderBackend(skill_loader)}
    if backends:
        all_backends.update(backends)

    pipeline = ToolPipeline(
        store=store,
        policy=policy or AllowAllPolicy(),
        backends=all_backends,
        limits=limits or ToolLoopLimits(),
    )

    obs_by_id: dict[UUID, ContextReadObservation] = {}

    class _ObsSource:
        def completed(
            self, *, scope: RunScope, invocation_id: UUID
        ) -> ContextReadObservation | None:
            return obs_by_id.get(invocation_id)

    projector = ExplorationProjector(
        scope=scope,
        objective_ref=turn_id,
        sources=_ObsSource(),
        tool_pins=tuple(
            VersionPin(reference=p.reference, version=p.version, fingerprint=p.fingerprint)
            for p in registered_agent.tool_pins
        ),
        clock=lambda: FIXTURE_TIME,
    )

    # Wire sequence fixtures using stable step keys
    seq_key = f"orchestrator.tool_loop:{run_id}"
    gateway = MockModelGateway(
        fixtures={},
        model_ref=model_ref,
        tool_sequence_fixtures={seq_key: responses},
    )

    orchestrator = ToolLoopOrchestrator(
        gateway=gateway,
        catalog=catalog,
        pipeline=pipeline,
        store=store,
        history=history,
        access=AllowContextAccess(),
        budget=budget,
        skill_loader=skill_loader,
        projector=projector,
        observation_sink=obs_by_id,
        model_ref=model_ref,
        clock=lambda: FIXTURE_TIME,
        delegate_specialists=lambda _: specialists,
    )
    return orchestrator, value, scope, lease, gateway, store, resources, projector


def answer_fixture(
    text: str,
    *,
    outcome: Literal["ANSWER", "NEEDS_INPUT", "CAPABILITY_UNAVAILABLE"] = "ANSWER",
    artifact_state: Literal["NONE", "DRAFT_AVAILABLE", "PROPOSAL_PENDING", "APPLIED"] = "NONE",
    citations: list[dict[str, object]] | None = None,
    stop_reason: str = "ANSWER_READY",
) -> dict[str, object]:
    block: dict[str, object]
    if outcome == "NEEDS_INPUT":
        block = {
            "kind": "question",
            "text": text,
            "capability": None,
            "message_key": None,
            "citations": [],
        }
    elif citations:
        block = {
            "kind": "work_evidence",
            "text": text,
            "capability": None,
            "message_key": None,
            "citations": citations,
        }
    else:
        block = {
            "kind": "text",
            "text": text,
            "capability": None,
            "message_key": None,
            "citations": [],
        }
    return {
        "final": {
            "outcome": outcome,
            "summary": text,
            "blocks": [block],
            "artifact_state": artifact_state,
            "stop_reason": stop_reason,
        }
    }
