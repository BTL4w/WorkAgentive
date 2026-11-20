"""Task 9: delegate_agent contracts, schema filtering, guards, receipts and normalization."""

import json
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4, uuid5

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from work_management_ai.agents.work_intelligence.contracts import WorkIntelligenceInput
from work_management_ai.model_gateway.contracts import (
    ModelUsage,
    StructuredModelRequest,
    StructuredModelResponse,
)
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.openai_tools import OpenAINativeToolGateway
from work_management_ai.model_gateway.tool_contracts import ToolItem, ToolModelRequest
from work_management_ai.runtime.agent_registry import AgentRegistry
from work_management_ai.runtime.context_exploration import (
    ContextAccessDecision,
    ContextUseValidator,
    ExplorationProjector,
)
from work_management_ai.runtime.contracts import (
    AgentHandoff,
    AgentId,
    AgentResult,
    AgentRunStatus,
    InteractionOutcome,
    ProposedAction,
    RiskLevel,
)
from work_management_ai.runtime.delegation import (
    CanonicalWorkflowState,
    DelegationAdapter,
    DelegationError,
    DelegationToolBackend,
    filtered_delegate_tool_schema,
    normalize_specialist_result,
)
from work_management_ai.runtime.execution_engine import RecordedAgentRun
from work_management_ai.runtime.manifests import (
    ToolManifestV2,
    load_tool_manifest_resource,
)
from work_management_ai.runtime.run_budget import (
    BudgetedToolLoopGateway,
    BudgetError,
    BudgetReceipt,
    CandidateBudget,
    ChildAllocation,
    RemainingBudget,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_loop_contracts import (
    CompletedToolResult,
    DeferredToolResult,
    PayloadReference,
    RejectedToolResult,
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopLimits,
    ToolLoopUsage,
    VersionPin,
)
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    ReadCoverage,
)
from work_management_ai.tools.delegation.contracts import (
    DelegateAgentInput,
    DelegateAgentOutput,
    PlanningCreateDelegationInput,
    ReportingPrepareDelegationInput,
)

_NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)


def _scope(org_id: UUID | None = None, member_id: UUID | None = None) -> RunScope:
    return RunScope(
        organization_id=org_id or uuid4(),
        membership_id=member_id or uuid4(),
        conversation_id=uuid4(),
        turn_id=uuid4(),
        orchestration_run_id=uuid4(),
    )


def _lease(scope: RunScope) -> RunLease:
    return RunLease(
        scope=scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="worker-1",
        fencing_token=1,
        expires_at=_NOW + timedelta(seconds=120),
    )


def _context(
    scope: RunScope,
    args: DelegateAgentInput,
    *,
    lease: RunLease | None = None,
    version: str = "1.0.0",
    receipt_id: UUID | None = None,
) -> ToolInvocationContext:
    return ToolInvocationContext(
        scope=scope,
        lease=lease or _lease(scope),
        invocation_id=uuid4(),
        call_id="call-delegate-1",
        tool_reference="delegate_agent@1",
        tool_version=version,
        arguments_fingerprint="a" * 64,
        canonical_arguments=args,
        policy_version="policy-v1",
        receipt_id=receipt_id or uuid4(),
        budget_allocation_id=uuid4(),
        limits=ToolLoopLimits(),
    )


def _agent_registry(*, enable_retrieve: bool = False) -> AgentRegistry:
    skills = SkillRegistry.from_resources(
        (f"work_management_ai.skills.{name}", "skill.yaml")
        for name in (
            "analyze_workload",
            "answer_work_question",
            "compare_daily_update_evidence",
            "create_project_plan",
            "draft_management_report",
            "explain_verified_risk",
            "extract_daily_update",
            "recommend_project_team",
            "review_evidence_concerns",
            "revise_project_plan",
            "summarize_verified_project_metrics",
            "web_search_and_summarize",
        )
    )
    tools = ToolRegistry.from_resources(
        (
            ("work_management_ai.tools.work.read_resource", "tool.yaml"),
            ("work_management_ai.tools.work.read_my_tasks", "tool.yaml"),
            ("work_management_ai.tools.planning.manage_run", "tool.yaml"),
            ("work_management_ai.tools.assignment.manage_team", "tool.yaml"),
            ("work_management_ai.tools.assignment.read_workload", "tool.yaml"),
            ("work_management_ai.tools.assignment.assign_task", "tool.yaml"),
            ("work_management_ai.tools.daily_update", "tool.yaml"),
            ("work_management_ai.tools.risk", "tool.yaml"),
            ("work_management_ai.tools.reporting", "tool.yaml"),
            ("work_management_ai.tools.reporting", "propose.yaml"),
            ("work_management_ai.tools.reporting", "chat.yaml"),
            ("work_management_ai.tools.automation", "tool.yaml"),
            ("work_management_ai.tools.web", "search.yaml"),
            ("work_management_ai.tools.web", "read_pages.yaml"),
            ("work_management_ai.tools.web", "revalidate.yaml"),
        )
    )
    evaluators = frozenset(
        {
            "orchestrator_plan@1",
            "work_grounding@1",
            "planning_schema@1",
            "planning_invariants@1",
            "planning_grounding@1",
            "assignment_explanation@1",
            "assignment_policy@1",
            "daily_update_grounding@1",
            "risk_grounding@1",
            "reporting_numeric@1",
            "reporting_grounding@1",
            "retrieve_grounding@1",
        }
    )
    registry = AgentRegistry(skill_registry=skills, tool_registry=tools, evaluator_ids=evaluators)
    agent_packages = [
        "orchestrator",
        "work_intelligence",
        "planning",
        "assignment",
        "daily_update",
        "risk",
        "reporting",
    ]
    if enable_retrieve:
        agent_packages.append("retrieve")
    for pkg in agent_packages:
        registry.register_resource(f"work_management_ai.agents.{pkg}", "agent.yaml")
    return registry


class _FakeBudget:
    def __init__(
        self,
        *,
        handoffs_available: int = 6,
        total_model_available: int = 24,
        total_tools_available: int = 12,
    ) -> None:
        self.handoffs_available = handoffs_available
        self.total_model_available = total_model_available
        self.total_tools_available = total_tools_available
        self.reservations: list[UsageReservation] = []
        self.allocations: dict[UUID, ChildAllocation] = {}
        self.closed_allocations: set[UUID] = set()

    async def recover_tool(
        self,
        scope: RunScope,
        operation_id: UUID,
        *,
        lease: RunLease,
        interval_id: UUID,
        allocation_id: UUID | None,
    ) -> BudgetReceipt:
        raise BudgetError("NOT_FOUND")

    async def reserve(
        self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt:
        if reservation.kind == "HANDOFF":
            if self.handoffs_available <= 0:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.handoffs_available -= 1
        elif reservation.kind == "MODEL":
            if self.total_model_available <= 0:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.total_model_available -= 1
        elif reservation.kind == "TOOL":
            if self.total_tools_available <= 0:
                raise BudgetError("BUDGET_EXHAUSTED")
            self.total_tools_available -= 1
        self.reservations.append(reservation)
        return BudgetReceipt(scope=scope, lease=lease, reservation=reservation)

    async def allocate_child(
        self, scope: RunScope, allocation: ChildAllocation, *, lease: RunLease
    ) -> ChildAllocation:
        if self.total_model_available <= 0 and self.total_tools_available <= 0:
            raise BudgetError("BUDGET_EXHAUSTED")
        self.allocations[allocation.allocation_id] = allocation
        return allocation

    async def close_child(self, scope: RunScope, allocation_id: UUID, *, lease: RunLease) -> None:
        self.closed_allocations.add(allocation_id)

    async def settle(
        self, scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
    ) -> None:
        return None

    async def remaining(self, scope: RunScope) -> RemainingBudget:
        return RemainingBudget(
            usage=ToolLoopUsage(),
            available=ToolLoopUsage(
                root_model_requests=min(8, self.total_model_available),
                total_model_requests=self.total_model_available,
                total_tool_calls=self.total_tools_available,
                handoffs=self.handoffs_available,
                input_tokens=48000,
                output_tokens=8000,
                segments=4,
                active_seconds=180,
            ),
        )


class _RecordingRunner:
    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.handoffs: list[AgentHandoff] = []
        self.recorded_by_key: dict[str, RecordedAgentRun] = {}

    async def run_specialist(self, handoff: AgentHandoff) -> AgentResult:
        if handoff.target_agent_id is AgentId.WORK_INTELLIGENCE:
            WorkIntelligenceInput.model_validate(handoff.typed_input)
        self.handoffs.append(handoff)
        run_id = uuid4()
        self.recorded_by_key[handoff.idempotency_key] = RecordedAgentRun(
            id=run_id, status=self.result.status, replayed_result=self.result
        )
        return self.result

    async def lookup_agent_run(self, idempotency_key: str) -> RecordedAgentRun | None:
        return self.recorded_by_key.get(idempotency_key)


class _AlwaysAcceptContextAccess:
    async def check(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision:
        return ContextAccessDecision(
            outcome="ACCEPTED", actor_revision="actor-1", policy_revision="policy-1"
        )


def _projector_with_project(
    scope: RunScope, *, resolution: str = "UNIQUE", project_ids: tuple[UUID, ...] | None = None
) -> tuple[ExplorationProjector, tuple[ContextRef, ...]]:
    ids = project_ids or ((uuid4(),) if resolution == "UNIQUE" else (uuid4(), uuid4()))
    inv_id = uuid4()
    refs = tuple(
        ContextRef(
            ref_id=uuid4(),
            scope=scope,
            resource_type="project",
            resource_id=pid,
            version=1,
            fingerprint="b" * 64,
            observed_at=_NOW,
            invocation_id=inv_id,
            source="project_service",
        )
        for pid in ids
    )
    obs = ContextReadObservation(
        scope=scope,
        objective_ref=uuid4(),
        invocation_id=inv_id,
        tool_reference="explore_projects@1",
        tool_version="1.0.2",
        observed_at=_NOW,
        outcome="NEEDS_INPUT" if resolution == "AMBIGUOUS" else "OK",
        context_refs=refs,
        entities=(
            EntityResolution(
                query_fingerprint="c" * 64,
                resolution="AMBIGUOUS" if resolution == "AMBIGUOUS" else "UNIQUE",
                candidate_ref_ids=tuple(r.ref_id for r in refs),
                selected_ref_id=refs[0].ref_id if resolution == "UNIQUE" else None,
            ),
        ),
        facts=tuple(
            ContextFact(ref_id=r.ref_id, field="project.name", value="Sao Mai") for r in refs
        ),
        coverage=(
            ReadCoverage(
                requested=("overview",),
                returned=("overview",),
                status="COMPLETE",
                query_fingerprint="c" * 64,
            ),
        ),
    )
    sources = {inv_id: obs}

    class _Sources:
        def completed(
            self, *, scope: RunScope, invocation_id: UUID
        ) -> ContextReadObservation | None:
            return sources.get(invocation_id)

    projector = ExplorationProjector(
        scope=scope,
        objective_ref=obs.objective_ref,
        sources=_Sources(),
        tool_pins=(
            VersionPin(reference="explore_projects@1", version="1.0.2", fingerprint="f" * 64),
        ),
        clock=lambda: _NOW,
    )
    report = projector.empty()
    projector.apply(report, obs)
    return projector, refs


# --- Contract & schema validation tests ---------------------------------------------------


def test_delegate_manifest_is_schema_2_with_documented_nested_fields() -> None:
    manifest = load_tool_manifest_resource("work_management_ai.tools.delegation", "delegate.yaml")
    assert isinstance(manifest, ToolManifestV2)
    assert manifest.name == "delegate_agent"
    assert manifest.idempotency == "REQUIRED"
    assert manifest.audit == "REQUIRED"
    assert manifest.risk_level is RiskLevel.PROPOSAL_ONLY
    (schema,) = ToolRegistry([manifest]).candidate_schemas(("delegate_agent@1",))
    assert schema.wire_name == "delegate_agent"
    assert schema.description == manifest.description
    assert len(schema.description) >= 80
    props = cast(dict[str, dict[str, object]], schema.parameters["properties"])
    assert set(props) == {
        "agent_id",
        "capability",
        "objective",
        "context_ref_ids",
        "constraints",
        "expected_result",
        "input",
    }
    for field_schema in props.values():
        assert field_schema.get("description")


def test_self_delegation_capability_mismatch_and_crafted_authority_keys_are_rejected() -> None:
    # Self delegation to orchestrator
    with pytest.raises(ValidationError):
        DelegateAgentInput.model_validate(
            {
                "agent_id": "orchestrator",
                "capability": "orchestration.delegate",
                "objective": "Loop into self",
                "expected_result": "Anything",
                "input": {"capability": "work.answer_question", "question": "hi"},
            }
        )
    # Envelope capability != input.capability
    with pytest.raises(ValidationError):
        DelegateAgentInput.model_validate(
            {
                "agent_id": "planning",
                "capability": "planning.create",
                "objective": "Plan project",
                "expected_result": "Draft",
                "input": {
                    "capability": "assignment.analyze_workload",
                    "project_reference": "Sao Mai",
                },
            }
        )
    # Crafted authority keys in envelope or nested input
    for bad in (
        {"role": "ADMIN"},
        {"organization_id": str(uuid4())},
        {"approved": True},
        {"auto_approve": True},
    ):
        payload = {
            "agent_id": "assignment",
            "capability": "assignment.analyze_workload",
            "objective": "Check workload",
            "expected_result": "Workload summary",
            "input": {
                "capability": "assignment.analyze_workload",
                "project_reference": "Sao Mai",
            },
            **bad,
        }
        with pytest.raises(ValidationError):
            DelegateAgentInput.model_validate(payload)
        nested_payload = {
            "agent_id": "assignment",
            "capability": "assignment.analyze_workload",
            "objective": "Check workload",
            "expected_result": "Workload summary",
            "input": {
                "capability": "assignment.analyze_workload",
                "project_reference": "Sao Mai",
                **bad,
            },
        }
        with pytest.raises(ValidationError):
            DelegateAgentInput.model_validate(nested_payload)


def test_reporting_period_timezone_and_planning_anchor_dates_validation() -> None:
    valid_report = ReportingPrepareDelegationInput(
        capability="reporting.prepare_report",
        project_reference="Sao Mai",
        report_kind="weekly_status",
        period_start="2026-11-09",
        period_end="2026-11-15",
        timezone="Asia/Ho_Chi_Minh",
        locale="vi",
    )
    assert valid_report.timezone == "Asia/Ho_Chi_Minh"

    with pytest.raises(ValidationError):
        ReportingPrepareDelegationInput(
            capability="reporting.prepare_report",
            project_reference="Sao Mai",
            timezone="Mars/Olympus_Mons",
        )
    with pytest.raises(ValidationError):
        ReportingPrepareDelegationInput(
            capability="reporting.prepare_report",
            project_reference="Sao Mai",
            period_start="2026-11-20",
            period_end="2026-11-10",
            timezone="UTC",
        )

    valid_plan = PlanningCreateDelegationInput(
        capability="planning.create",
        goal="Launch AI training for 20 employees",
        anchor_start_date="2026-12-01",
        anchor_end_date="2026-12-21",
        duration_days=21,
        locale="vi",
    )
    assert valid_plan.anchor_start_date == "2026-12-01"

    with pytest.raises(ValidationError):
        PlanningCreateDelegationInput(
            capability="planning.create",
            goal="Launch AI training",
            anchor_start_date="not-a-date",
        )
    with pytest.raises(ValidationError):
        PlanningCreateDelegationInput(
            capability="planning.create",
            goal="Launch AI training",
            anchor_start_date="2026-12-21",
            anchor_end_date="2026-12-01",
        )


def test_filtered_delegate_schema_exposes_only_activated_capabilities() -> None:
    manifest = load_tool_manifest_resource("work_management_ai.tools.delegation", "delegate.yaml")
    (base_schema,) = ToolRegistry([manifest]).candidate_schemas(("delegate_agent@1",))
    filtered = filtered_delegate_tool_schema(
        base_schema,
        allowed_capabilities={
            (AgentId.ASSIGNMENT, "assignment.analyze_workload"),
            (AgentId.PLANNING, "planning.create"),
            (AgentId.WORK_INTELLIGENCE, "work.answer_question"),
        },
    )
    props = cast(dict[str, dict[str, object]], filtered.parameters["properties"])
    assert set(cast(list[str], props["agent_id"]["enum"])) == {"assignment", "work_intelligence"}
    assert set(cast(list[str], props["capability"]["enum"])) == {
        "assignment.analyze_workload",
        "work.answer_question",
    }
    raw = json.dumps(filtered.parameters)
    assert "retrieval.search_web" not in raw
    assert "reporting.prepare_report" not in raw
    assert "assignment.analyze_workload" in raw
    assert "planning.create" not in raw


@pytest.mark.asyncio
async def test_filtered_delegate_schema_reaches_actual_openai_sdk_wire_payload() -> None:
    manifest = load_tool_manifest_resource("work_management_ai.tools.delegation", "delegate.yaml")
    assert isinstance(manifest, ToolManifestV2)
    (base_schema,) = ToolRegistry([manifest]).candidate_schemas(("delegate_agent@1",))
    filtered = filtered_delegate_tool_schema(
        base_schema,
        allowed_capabilities={
            (AgentId.ASSIGNMENT, "assignment.analyze_workload"),
            (AgentId.RISK, "risk.explain"),
        },
    )
    captured: list[dict[str, object]] = []

    class _Final(BaseModel):
        text: str

    def respond(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-del-1",
                "object": "chat.completion",
                "created": 0,
                "model": "configured-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"text":"ok"}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        def factory(**kwargs: object) -> object:
            return ChatOpenAI(
                model=str(kwargs["model_name"]),
                api_key=SecretStr("fake"),
                http_async_client=client,
                use_responses_api=False,
                store=False,
                max_retries=0,
                max_completion_tokens=120,
            )

        adapter = OpenAINativeToolGateway(
            model_name="configured-model", api_key=SecretStr("fake"), chat_model_factory=factory
        )
        req = ToolModelRequest[_Final](
            invocation_key="delegate-wire-test-1",
            items=(ToolItem(kind="message", role="user", content="Check workload"),),
            tools=(filtered,),
            response_schema=_Final,
            max_output_tokens=120,
            timeout_seconds=30,
        )
        await adapter.generate_step(req)

    assert len(captured) == 1
    wire = cast(list[dict[str, object]], captured[0]["tools"])[0]
    fn = cast(dict[str, object], wire["function"])
    assert fn["name"] == "delegate_agent"
    assert fn["description"] == manifest.description
    assert fn["strict"] is True
    params = cast(dict[str, object], fn["parameters"])
    assert params["additionalProperties"] is False
    props = cast(dict[str, dict[str, object]], params["properties"])
    assert set(cast(list[str], params["required"])) == set(props)
    for field in props.values():
        assert field.get("description")
    defs = cast(dict[str, dict[str, object]], params.get("$defs", {}))
    assert defs
    for model_def in defs.values():
        assert model_def.get("additionalProperties") is False
        nested_props = cast(dict[str, dict[str, object]], model_def["properties"])
        assert set(cast(list[str], model_def["required"])) == set(nested_props)
        for nested_field in nested_props.values():
            assert nested_field.get("description")


# --- Runtime delegation guards, context validation, budget & reconciliation ---------------


@pytest.mark.asyncio
async def test_inactive_target_role_mismatch_and_exhausted_budget_deny_zero_child_runs() -> None:
    scope = _scope()
    lease = _lease(scope)
    registry = _agent_registry(enable_retrieve=False)
    runner = _RecordingRunner(
        AgentResult(
            agent_id=AgentId.ASSIGNMENT,
            agent_version="1.0.0",
            status=AgentRunStatus.COMPLETED,
            typed_output={"summary": "ok"},
            stop_reason="COMPLETED",
        )
    )
    tool_registry = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.yaml"),)
    )

    # 1) Inactive target (retrieve not registered / I3 OFF)
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        tool_registry=tool_registry,
        runner=runner,
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=uuid4(),
    )
    retrieve_args = DelegateAgentInput.model_validate(
        {
            "agent_id": "retrieve",
            "capability": "retrieval.search_web",
            "objective": "Search web",
            "expected_result": "Sources",
            "input": {"capability": "retrieval.search_web", "query": "ai news"},
        }
    )
    budget = _FakeBudget()
    with (
        candidate_budget(
            CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
        ),
        pytest.raises(DelegationError) as exc,
    ):
        await adapter.execute(_context(scope, retrieve_args, lease=lease), retrieve_args)
    assert exc.value.code == "DELEGATION_TARGET_UNAVAILABLE"
    assert runner.handoffs == []

    # 2) Role forbidden (EMPLOYEE cannot call planning.create)
    employee_adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        tool_registry=tool_registry,
        runner=runner,
        actor_role="EMPLOYEE",
        active_phase=5,
        parent_agent_run_id=uuid4(),
    )
    plan_args = DelegateAgentInput.model_validate(
        {
            "agent_id": "planning",
            "capability": "planning.create",
            "objective": "Plan project",
            "expected_result": "Draft",
            "input": {"capability": "planning.create", "goal": "Launch course"},
        }
    )
    with (
        candidate_budget(
            CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
        ),
        pytest.raises(DelegationError) as exc,
    ):
        await employee_adapter.execute(_context(scope, plan_args, lease=lease), plan_args)
    assert exc.value.code == "DELEGATION_ROLE_FORBIDDEN"
    assert runner.handoffs == []

    with pytest.raises(DelegationError) as unavailable:
        await adapter.execute(_context(scope, plan_args, lease=lease), plan_args)
    assert unavailable.value.code == "DELEGATION_COMPATIBILITY_UNAVAILABLE"
    assert runner.handoffs == []

    # 3) No remaining child quota -> zero child runs
    empty_budget = _FakeBudget(total_model_available=0, total_tools_available=0)
    work_args = DelegateAgentInput.model_validate(
        {
            "agent_id": "work_intelligence",
            "capability": "work.answer_question",
            "objective": "Check workload",
            "expected_result": "Workload",
            "input": {"capability": "work.answer_question", "question": "What is due today?"},
        }
    )
    with (
        candidate_budget(
            CandidateBudget(scope=scope, lease=lease, store=empty_budget, interval_id=uuid4())
        ),
        pytest.raises(DelegationError) as exc,
    ):
        await adapter.execute(_context(scope, work_args, lease=lease), work_args)
    assert exc.value.code == "DELEGATION_BUDGET_EXHAUSTED"
    assert runner.handoffs == []


@pytest.mark.asyncio
async def test_cross_tenant_and_ambiguous_context_refs_are_blocked_with_zero_child_runs() -> None:
    scope = _scope()
    lease = _lease(scope)
    registry = _agent_registry()
    tool_registry = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.yaml"),)
    )
    runner = _RecordingRunner(
        AgentResult(
            agent_id=AgentId.ASSIGNMENT,
            agent_version="1.0.0",
            status=AgentRunStatus.COMPLETED,
            typed_output={"summary": "ok"},
            stop_reason="COMPLETED",
        )
    )

    # Ambiguous exploration report: even selecting one candidate ref must return NEEDS_INPUT
    projector, refs = _projector_with_project(scope, resolution="AMBIGUOUS")
    validator = ContextUseValidator(projector=projector, access=_AlwaysAcceptContextAccess())
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        tool_registry=tool_registry,
        runner=runner,
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=uuid4(),
        context_validator=validator,
    )
    ambiguous_args = DelegateAgentInput.model_validate(
        {
            "agent_id": "assignment",
            "capability": "assignment.analyze_workload",
            "objective": "Check workload",
            "context_ref_ids": [refs[0].ref_id],
            "expected_result": "Workload",
            "input": {
                "capability": "assignment.analyze_workload",
                "project_reference": str(refs[0].resource_id),
            },
        }
    )
    budget = _FakeBudget()
    with (
        candidate_budget(
            CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
        ),
        pytest.raises(DelegationError) as exc,
    ):
        await adapter.execute(_context(scope, ambiguous_args, lease=lease), ambiguous_args)
    assert exc.value.code == "DELEGATION_CONTEXT_NEEDS_INPUT"
    assert runner.handoffs == []

    # Forged / cross-tenant ref ID not in current scope's report -> DENIED
    foreign_args = DelegateAgentInput.model_validate(
        {
            "agent_id": "assignment",
            "capability": "assignment.analyze_workload",
            "objective": "Check workload",
            "context_ref_ids": [uuid4()],
            "expected_result": "Workload",
            "input": {
                "capability": "assignment.analyze_workload",
                "project_reference": "foreign-project",
            },
        }
    )
    with (
        candidate_budget(
            CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
        ),
        pytest.raises(DelegationError) as exc,
    ):
        await adapter.execute(_context(scope, foreign_args, lease=lease), foreign_args)
    assert exc.value.code == "DELEGATION_CONTEXT_DENIED"
    assert runner.handoffs == []


@pytest.mark.asyncio
async def test_child_provider_calls_charge_parent_conserved_budget_allocation() -> None:
    scope = _scope()
    lease = _lease(scope)
    registry = _agent_registry()
    current = registry.resolve(AgentId.WORK_INTELLIGENCE, "1.0.0", 5)
    pin = VersionPin(
        reference="work_intelligence", version="1.0.0", fingerprint=current.fingerprint
    )
    tool_registry = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.yaml"),)
    )
    budget = _FakeBudget()

    class _ChildOut(BaseModel):
        answer: str

    seen: list[str] = []

    class _Model(MockModelGateway):
        async def generate_structured[OutputT: BaseModel](
            self, request: StructuredModelRequest[OutputT]
        ) -> StructuredModelResponse[OutputT]:
            seen.extend(message.content for message in request.messages)
            return await super().generate_structured(request)

    gateway = BudgetedToolLoopGateway(
        _Model(fixtures={"child-model-call": {"answer": "Verified answer"}})
    )

    class _BudgetedChildRunner:
        def __init__(self) -> None:
            self.handoffs: list[AgentHandoff] = []
            self.record: RecordedAgentRun | None = None

        async def run_specialist(self, handoff: AgentHandoff) -> AgentResult:
            self.handoffs.append(handoff)
            WorkIntelligenceInput.model_validate(handoff.typed_input)
            # Child calls structured model gateway under parent's child allocation_id.
            await gateway.generate_structured(
                StructuredModelRequest(
                    invocation_key="child-model-call",
                    messages=(),
                    output_schema=_ChildOut,
                    max_output_tokens=200,
                    timeout_seconds=15,
                )
            )
            result = AgentResult(
                agent_id=AgentId.WORK_INTELLIGENCE,
                agent_version="1.0.0",
                status=AgentRunStatus.COMPLETED,
                interaction_outcome=InteractionOutcome.RESULT_READY,
                typed_output={"summary": "Verified answer"},
                model_attempts_used=1,
                stop_reason="COMPLETED",
            )

            self.record = RecordedAgentRun(id=uuid4(), status=result.status, replayed_result=result)
            return result

        async def lookup_agent_run(self, idempotency_key: str) -> RecordedAgentRun | None:
            return self.record

    runner = _BudgetedChildRunner()
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=(pin,),
        tool_registry=tool_registry,
        runner=runner,
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=uuid4(),
    )
    args = DelegateAgentInput.model_validate(
        {
            "agent_id": "work_intelligence",
            "capability": "work.answer_question",
            "objective": "Answer project status",
            "expected_result": "Grounded status",
            "constraints": ["Include only incomplete tasks"],
            "input": {"capability": "work.answer_question", "question": "Status of Sao Mai?"},
        }
    )
    ctx = _context(scope, args, lease=lease)
    with candidate_budget(
        CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
    ):
        backend_result = await adapter.execute_backend(ctx, args)
    assert isinstance(backend_result.output, DelegateAgentOutput)
    assert json.loads(seen[-1])["delegated_objective"] == {
        "objective": "Answer project status",
        "expected_result": "Grounded status",
        "constraints": ["Include only incomplete tasks"],
    }
    for change, code in (
        ({"fingerprint": "0" * 64}, "DELEGATION_TARGET_PIN_MISMATCH"),
        ({"version": "9.0.0"}, "DELEGATION_TARGET_UNAVAILABLE"),
    ):
        bad_adapter = DelegationAdapter(
            agent_registry=registry,
            specialist_pins=(pin.model_copy(update=change),),
            runner=runner,
            actor_role="MANAGER",
            active_phase=5,
            parent_agent_run_id=uuid4(),
        )
        with pytest.raises(DelegationError) as exc:
            await bad_adapter.execute_backend(ctx, args)
        assert exc.value.code == code
    assert len(runner.handoffs) == 1
    assert not any(r.kind == "HANDOFF" for r in budget.reservations)
    model_res = [r for r in budget.reservations if r.kind == "MODEL"]
    assert len(model_res) == 1
    assert model_res[0].allocation_id == uuid5(ctx.receipt_id, "child-budget")
    assert uuid5(ctx.receipt_id, "child-budget") in budget.closed_allocations
    assert backend_result.output.typed_output == {"summary": "Verified answer"}


def test_normalize_specialist_result_distinguishes_draft_wait_workflow_and_unverified_legacy() -> (
    None
):
    scope = _scope()
    inv_id = uuid4()
    child_run_id = uuid4()
    payload_ref = PayloadReference(
        payload_id=uuid4(),
        organization_id=scope.organization_id,
        fingerprint="d" * 64,
        expires_at=_NOW + timedelta(days=7),
    )

    # 1) DRAFT_READY -> CompletedToolResult (NOT a wait)
    draft_result = AgentResult(
        agent_id=AgentId.PLANNING,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_HUMAN,
        interaction_outcome=InteractionOutcome.DRAFT_READY,
        typed_output={"draft_id": str(uuid4()), "draft_version": 1, "summary": "Draft ready"},
        stop_reason="DRAFT_READY",
    )
    outcome = normalize_specialist_result(
        scope,
        draft_result,
        invocation_id=inv_id,
        call_id="c1",
        child_run_id=child_run_id,
        payload_ref=payload_ref,
    )
    assert isinstance(outcome, CompletedToolResult)

    # 2) NEEDS_INPUT -> DeferredToolResult(kind="USER_INPUT")
    needs_input = AgentResult(
        agent_id=AgentId.PLANNING,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_INPUT,
        interaction_outcome=InteractionOutcome.NEEDS_INPUT,
        typed_output={"question": "Target launch date?"},
        stop_reason="NEEDS_INPUT",
    )
    wait_input = normalize_specialist_result(
        scope,
        needs_input,
        invocation_id=inv_id,
        call_id="c2",
        child_run_id=child_run_id,
    )
    assert isinstance(wait_input, DeferredToolResult)
    assert wait_input.wait_ref.kind == "USER_INPUT"

    # 3) ACTION_REVIEW_REQUIRED with versioned proposal -> HUMAN_DECISION wait
    proposal_id = uuid4()
    review_req = AgentResult(
        agent_id=AgentId.PLANNING,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_HUMAN,
        interaction_outcome=InteractionOutcome.ACTION_REVIEW_REQUIRED,
        typed_output={"proposal_id": str(proposal_id), "proposal_version": 2},
        proposed_actions=(
            ProposedAction(
                action_type="planning.approve",
                risk=RiskLevel.PROPOSAL_ONLY,
                requires_human_gate=True,
                reference_id=proposal_id,
            ),
        ),
        stop_reason="AWAITING_APPROVAL",
    )
    wait_human = normalize_specialist_result(
        scope,
        review_req,
        invocation_id=inv_id,
        call_id="c3",
        child_run_id=child_run_id,
    )
    assert isinstance(wait_human, DeferredToolResult)
    assert wait_human.wait_ref.kind == "HUMAN_DECISION"
    assert wait_human.wait_ref.resource_id == proposal_id
    assert wait_human.wait_ref.resource_version == 2

    # 4) Verified running workflow -> WORKFLOW_RESULT wait
    wf_id = uuid4()
    queued_result = AgentResult(
        agent_id=AgentId.PLANNING,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_HUMAN,
        typed_output={"workflow_run_id": str(wf_id)},
        stop_reason="WORKFLOW_QUEUED",
    )
    wait_wf = normalize_specialist_result(
        scope,
        queued_result,
        invocation_id=inv_id,
        call_id="c4",
        child_run_id=child_run_id,
        workflow_state=CanonicalWorkflowState(workflow_run_id=wf_id, status="RUNNING", version=1),
    )
    assert isinstance(wait_wf, DeferredToolResult)
    assert wait_wf.wait_ref.kind == "WORKFLOW_RESULT"
    assert wait_wf.wait_ref.resource_id == wf_id

    # 5) Legacy AWAITING_HUMAN WITHOUT verified proposal/workflow must NOT fabricate HUMAN_DECISION
    unverified_legacy = AgentResult(
        agent_id=AgentId.PLANNING,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_HUMAN,
        typed_output={"summary": "Legacy awaiting"},
        stop_reason="AWAITING_HUMAN",
    )
    rejected = normalize_specialist_result(
        scope,
        unverified_legacy,
        invocation_id=inv_id,
        call_id="c5",
        child_run_id=child_run_id,
    )
    assert isinstance(rejected, RejectedToolResult)
    assert rejected.safe_code == "DELEGATION_WAIT_UNVERIFIED"


@pytest.mark.asyncio
async def test_commit_before_parent_result_reconciles_same_child_run_without_duplicate() -> None:
    scope = _scope()
    lease = _lease(scope)
    registry = _agent_registry()
    tool_registry = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.yaml"),)
    )
    child_result = AgentResult(
        agent_id=AgentId.WORK_INTELLIGENCE,
        agent_version="1.0.0",
        status=AgentRunStatus.AWAITING_INPUT,
        interaction_outcome=InteractionOutcome.NEEDS_INPUT,
        typed_output={"question": "Which project?"},
        stop_reason="NEEDS_INPUT",
    )
    runner = _RecordingRunner(child_result)
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        tool_registry=tool_registry,
        runner=runner,
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=uuid4(),
    )
    backend = DelegationToolBackend(adapter)
    args = DelegateAgentInput.model_validate(
        {
            "agent_id": "work_intelligence",
            "capability": "work.answer_question",
            "objective": "Read project status",
            "expected_result": "Status",
            "input": {
                "capability": "work.answer_question",
                "question": "How is the course project doing?",
            },
        }
    )
    receipt_id = uuid4()
    ctx = _context(scope, args, lease=lease, receipt_id=receipt_id)
    budget = _FakeBudget()

    # First execution commits child run; imagine parent crashes before recording observation
    with candidate_budget(
        CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=uuid4())
    ):
        first = await backend.execute(ctx, args)
    assert len(runner.handoffs) == 1
    assert first.wait_ref is not None
    assert first.wait_ref.kind == "USER_INPUT"

    # Reconcile with same receipt_id must return same receipt without invoking runner again
    reconciled = await backend.reconcile(ctx, args)
    assert reconciled is not None
    assert reconciled.wait_ref == first.wait_ref
    assert len(runner.handoffs) == 1
