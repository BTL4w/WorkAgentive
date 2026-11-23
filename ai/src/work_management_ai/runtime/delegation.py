"""Candidate Specialist-as-tool delegation adapter, filtering, receipts and normalization.

Only the Orchestrator delegates; Specialists return AgentResult to the parent and
never delegate to peers or self. Server-bound identity, tenant scope, authorization,
context-use validation and conserved budget reservations are enforced in Python.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol, Union
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from work_management_ai.agents.assignment.contracts import (
    AssignmentAgentOutput,
    TeamRecommendationSnapshot,
)
from work_management_ai.agents.daily_update.contracts import DailyUpdateResult
from work_management_ai.model_gateway.tool_contracts import ToolSchema
from work_management_ai.runtime.activation import ActivationContext
from work_management_ai.runtime.agent_registry import AgentRegistry, AgentRegistryError
from work_management_ai.runtime.context_exploration import ContextUseValidator
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentBudget,
    AgentHandoff,
    AgentHandoffV2,
    AgentId,
    AgentResult,
    AgentRunStatus,
    ContextReference,
    DelegationBrief,
    GoalDirectedAgentResult,
    InteractionOutcome,
    JsonValue,
)
from work_management_ai.runtime.execution_engine import RecordedAgentRun
from work_management_ai.runtime.manifests import GoalDirectedAgentManifest
from work_management_ai.runtime.run_budget import (
    BudgetError,
    BudgetReceipt,
    CandidateBudget,
    ChildAllocation,
    UsageReservation,
    candidate_budget,
    current_candidate_budget,
)
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    CompletedToolResult,
    DeferredToolResult,
    PayloadReference,
    RejectedToolResult,
    RunScope,
    ToolInvocationContext,
    ToolLoopUsage,
    ToolOutcome,
    VersionPin,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry, ToolRegistryError
from work_management_ai.tools.context.contracts import ContextRef
from work_management_ai.tools.delegation.contracts import (
    AssignmentAnalyzeWorkloadDelegationInput,
    DailyUpdatePrepareDelegationInput,
    DelegateAgentInput,
    DelegateAgentInputV2,
    DelegateAgentOutput,
    DelegateAgentOutputV2,
    RiskExplainDelegationInput,
    WorkQuestionDelegationInput,
    WorkReadProjectDelegationInput,
)

_DELEGATE_TOOL_RESOURCE = (("work_management_ai.tools.delegation", "delegate.yaml"),)
_DELEGATE_V2_TOOL_RESOURCE = (("work_management_ai.tools.delegation", "delegate.v2.yaml"),)

_CAPABILITY_MODELS: dict[str, type[BaseModel]] = {
    "work.answer_question": WorkQuestionDelegationInput,
    "work.read_project": WorkReadProjectDelegationInput,
    "assignment.analyze_workload": AssignmentAnalyzeWorkloadDelegationInput,
    "risk.explain": RiskExplainDelegationInput,
    "daily_update.prepare": DailyUpdatePrepareDelegationInput,
}


# Candidate dispatch requires a compatible Specialist input and a recoverable
# effect boundary. The current verified paths are read-only.
_CANDIDATE_CAPABILITIES = frozenset(
    {
        (AgentId.WORK_INTELLIGENCE, "work.answer_question"),
        (AgentId.WORK_INTELLIGENCE, "work.read_project"),
        (AgentId.RISK, "risk.explain"),
        (AgentId.ASSIGNMENT, "assignment.analyze_workload"),
    }
)


_DAILY_UPDATE_CAPABILITY = (AgentId.DAILY_UPDATE, "daily_update.prepare")
_ASSIGNMENT_EFFECTS = frozenset(
    {"assignment.assign_task_explicitly", "assignment.recommend_team", "assignment.revise_team"}
)


class DelegationError(ToolPolicyError):
    """Typed fail-closed delegation error with stable safe code."""


class CanonicalWorkflowState(BaseModel):
    """Workflow status verified from canonical persistence, never legacy AWAITING_HUMAN."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workflow_run_id: UUID
    status: Literal[
        "QUEUED",
        "RUNNING",
        "NEEDS_INPUT",
        "DRAFT_AVAILABLE",
        "WAITING_FOR_DECISION",
        "COMPLETED",
        "FAILED",
    ]
    draft_id: UUID | None = None
    draft_version: int | None = Field(default=None, ge=1)
    proposal_id: UUID | None = None
    proposal_version: int | None = Field(default=None, ge=1)
    version: int = Field(default=1, ge=1, strict=True)


class SpecialistDelegationRunnerPort(Protocol):
    async def run_specialist(self, handoff: AgentHandoff) -> AgentResult: ...
    async def lookup_agent_run(self, idempotency_key: str) -> RecordedAgentRun | None: ...


class DailyUpdateDraftReceiptPort(Protocol):
    async def recover_draft(
        self, scope: RunScope, *, receipt_id: UUID, task_id: UUID, task_version: int
    ) -> DailyUpdateResult | None: ...


class GoalDailyUpdateDraftReceiptPort(Protocol):
    async def recover_goal_draft(
        self, scope: RunScope, *, receipt_id: UUID, task_id: UUID, task_version: int
    ) -> DailyUpdateResult | None: ...


class WorkflowWaitVerifierPort(Protocol):
    async def verify_workflow(
        self, scope: RunScope, workflow_run_id: UUID
    ) -> CanonicalWorkflowState | None: ...

    async def find_committed_planning_create(
        self, scope: RunScope, *, child_run_id: UUID, receipt_id: UUID, call_id: str
    ) -> UUID | None: ...


class ChildJobReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    resource_id: UUID
    resolved_action: Literal[
        "planning.create", "planning.resume", "planning.revise", "reporting.prepare_report"
    ]
    workflow_state: CanonicalWorkflowState | None = None


class ChildJobReceiptsPort(Protocol):
    async def recover_job(
        self,
        scope: RunScope,
        *,
        allocation_id: UUID,
        child_run_id: UUID,
        target: VersionPin,
    ) -> ChildJobReceipt | None: ...

    async def verify_reporting_job(
        self,
        scope: RunScope,
        generation_id: UUID,
    ) -> CanonicalWorkflowState | None: ...


def filtered_delegate_tool_schema(
    base_schema: ToolSchema,
    *,
    allowed_capabilities: Iterable[tuple[AgentId, str]],
    daily_update_ready: bool = False,
) -> ToolSchema:
    """Build a filtered strict Pydantic contract & ToolSchema for activated Specialists only."""

    supported = (
        _CANDIDATE_CAPABILITIES | {_DAILY_UPDATE_CAPABILITY}
        if daily_update_ready
        else _CANDIDATE_CAPABILITIES
    )
    allowed = frozenset(allowed_capabilities) & supported
    allowed_agents = tuple(sorted({agent.value for agent, _ in allowed}))
    allowed_caps = tuple(sorted({cap for _, cap in allowed}))
    if not allowed_agents or not allowed_caps:
        raise DelegationError("DELEGATION_CATALOG_EMPTY", stop_segment=True)

    branch_models: list[type[BaseModel]] = []
    for cap in allowed_caps:
        model = _CAPABILITY_MODELS[cap]
        if model not in branch_models:
            branch_models.append(model)
    literal_factory: Any = Literal
    union_factory: Any = Union
    agent_literal: Any = literal_factory[allowed_agents]
    cap_literal: Any = literal_factory[allowed_caps]
    input_type: Any = (
        branch_models[0] if len(branch_models) == 1 else union_factory[tuple(branch_models)]
    )
    allowed_set = frozenset((a.value, c) for a, c in allowed)

    def _check_pair(self: Any) -> Any:
        aid = self.agent_id
        cap = self.capability
        nested = self.input
        if getattr(nested, "capability", None) != cap or (aid, cap) not in allowed_set:
            raise ValueError("DELEGATION_CAPABILITY_MISMATCH")
        return self

    validator_map: dict[str, Any] = {"valid_pair": model_validator(mode="after")(_check_pair)}
    filtered_model: type[BaseModel] = create_model(
        "FilteredDelegateAgentInput",
        __config__=ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False),
        __validators__=validator_map,
        agent_id=(
            agent_literal,
            Field(description=DelegateAgentInput.model_fields["agent_id"].description),
        ),
        capability=(
            cap_literal,
            Field(description=DelegateAgentInput.model_fields["capability"].description),
        ),
        objective=(
            str,
            Field(
                min_length=1,
                max_length=4000,
                description=DelegateAgentInput.model_fields["objective"].description,
            ),
        ),
        context_ref_ids=(
            tuple[UUID, ...],
            Field(
                default=(),
                max_length=16,
                description=DelegateAgentInput.model_fields["context_ref_ids"].description,
            ),
        ),
        constraints=(
            tuple[str, ...],
            Field(
                default=(),
                max_length=16,
                description=DelegateAgentInput.model_fields["constraints"].description,
            ),
        ),
        expected_result=(
            str,
            Field(
                min_length=1,
                max_length=500,
                description=DelegateAgentInput.model_fields["expected_result"].description,
            ),
        ),
        input=(
            input_type,
            Field(description=DelegateAgentInput.model_fields["input"].description),
        ),
    )
    return ToolSchema(
        reference=base_schema.reference,
        description=base_schema.description,
        parameters=filtered_model.model_json_schema(),
        input_contract=filtered_model,
    )


def filtered_delegate_v2_tool_schema(
    base_schema: ToolSchema, *, allowed_specialists: Iterable[tuple[AgentId, str]]
) -> ToolSchema:
    """Publish the admitted Specialists without exposing their execution contracts."""
    summaries = dict(allowed_specialists)
    if base_schema.reference != "delegate_agent@2":
        raise DelegationError("DELEGATION_TOOL_MISMATCH", stop_segment=True)
    if not summaries or AgentId.ORCHESTRATOR in summaries:
        raise DelegationError("DELEGATION_CATALOG_INVALID", stop_segment=True)
    agents = sorted(summaries, key=lambda agent: agent.value)
    literal_factory: Any = Literal
    filtered_model = create_model(
        "ScopedDelegateAgentInputV2",
        __base__=DelegateAgentInputV2,
        agent_id=(
            literal_factory[tuple(agent.value for agent in agents)],
            Field(description=DelegateAgentInputV2.model_fields["agent_id"].description),
        ),
    )
    descriptions = "\n".join(f"- {agent.value}: {summaries[agent]}" for agent in agents)
    return ToolSchema(
        reference=base_schema.reference,
        description=f"{base_schema.description}\n\nEligible Specialists:\n{descriptions}",
        input_contract=filtered_model,
    )


def _uuid_field(data: dict[str, JsonValue], key: str) -> UUID | None:
    raw = data.get(key)
    if not isinstance(raw, str):
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _int_field(data: dict[str, JsonValue], key: str) -> int | None:
    raw = data.get(key)
    if type(raw) is int and raw >= 1:
        return raw
    return None


def _summary_text(result: AgentResult) -> str:
    for key in ("summary", "public_summary", "message", "answer", "explanation", "question"):
        val = result.typed_output.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:4000]
    return result.stop_reason


def _build_output(
    *, capability: str, child_run_id: UUID, result: AgentResult
) -> DelegateAgentOutput:
    data = result.typed_output
    return DelegateAgentOutput(
        agent_id=result.agent_id,
        agent_version=result.agent_version,
        capability=capability,
        child_agent_run_id=child_run_id,
        status=result.status,
        interaction_outcome=result.interaction_outcome,
        summary=_summary_text(result),
        typed_output=data,
        artifact_refs=result.evidence,
        evidence=result.evidence,
        proposed_actions=result.proposed_actions,
        requested_handoff=result.requested_handoff,
        workflow_run_id=_uuid_field(data, "workflow_run_id"),
        draft_id=_uuid_field(data, "draft_id"),
        draft_version=_int_field(data, "draft_version"),
        proposal_id=_uuid_field(data, "proposal_id"),
        proposal_version=_int_field(data, "proposal_version"),
        stop_reason=result.stop_reason,
        safe_error_code=result.safe_error_code,
        usage={
            "model_attempts_used": result.model_attempts_used,
            "iterations_used": result.iterations_used,
            "tool_calls_used": result.tool_calls_used,
        },
    )


def _normalize_to_parts(
    scope: RunScope,
    result: AgentResult,
    *,
    invocation_id: UUID,
    capability: str,
    child_run_id: UUID,
    workflow_state: CanonicalWorkflowState | None = None,
) -> tuple[DelegateAgentOutput | None, WaitReference | None, str | None]:
    """Return (output, wait_ref, rejected_code). Never conflates ack/wait/completed."""

    if (
        result.status in {AgentRunStatus.FAILED, AgentRunStatus.CANCELLED}
        or result.interaction_outcome is InteractionOutcome.FAILED
    ):
        return None, None, result.safe_error_code or "DELEGATION_CHILD_FAILED"

    # 1) Verified queued/running async workflow -> WORKFLOW_RESULT wait
    wf_id = _uuid_field(result.typed_output, "workflow_run_id")
    if wf_id is not None:
        if workflow_state is None or workflow_state.workflow_run_id != wf_id:
            return None, None, "DELEGATION_WAIT_UNVERIFIED"
        status = workflow_state.status
        if status == "FAILED":
            return None, None, "DELEGATION_WORKFLOW_FAILED"
        if status in {"QUEUED", "RUNNING", "NEEDS_INPUT"}:
            wait = WaitReference(
                wait_id=uuid5(
                    NAMESPACE_URL,
                    f"wait:workflow:{invocation_id}:{wf_id}:{status == 'NEEDS_INPUT'}",
                ),
                scope=scope,
                kind="USER_INPUT" if status == "NEEDS_INPUT" else "WORKFLOW_RESULT",
                source_invocation_id=invocation_id,
                resource_id=wf_id,
                resource_version=workflow_state.version,
            )
            return None, wait, None
        if status == "WAITING_FOR_DECISION":
            if workflow_state.proposal_id is None or workflow_state.proposal_version is None:
                return None, None, "DELEGATION_WAIT_UNVERIFIED"
            return (
                None,
                WaitReference(
                    wait_id=uuid5(
                        invocation_id,
                        f"decision:{workflow_state.proposal_id}:{workflow_state.proposal_version}",
                    ),
                    scope=scope,
                    kind="HUMAN_DECISION",
                    source_invocation_id=invocation_id,
                    resource_id=workflow_state.proposal_id,
                    resource_version=workflow_state.proposal_version,
                ),
                None,
            )
        data = {**result.typed_output, "workflow_status": status}
        interaction = InteractionOutcome.RESULT_READY
        if status == "DRAFT_AVAILABLE":
            if workflow_state.draft_id is None or workflow_state.draft_version is None:
                return None, None, "DELEGATION_WAIT_UNVERIFIED"
            interaction = InteractionOutcome.DRAFT_READY
            data.update(
                draft_id=str(workflow_state.draft_id), draft_version=workflow_state.draft_version
            )
        result = result.model_copy(
            update={
                "status": AgentRunStatus.COMPLETED,
                "typed_output": data,
                "interaction_outcome": interaction,
                "stop_reason": status,
            }
        )
    elif workflow_state is not None:
        return None, None, "DELEGATION_WORKFLOW_MISMATCH"

    # 2) Draft available or Result ready -> Completed observation (not a wait)
    if result.interaction_outcome in {
        InteractionOutcome.DRAFT_READY,
        InteractionOutcome.RESULT_READY,
    }:
        return (
            _build_output(capability=capability, child_run_id=child_run_id, result=result),
            None,
            None,
        )

    # 3) Explicit user input required -> USER_INPUT wait
    if (
        result.interaction_outcome is InteractionOutcome.NEEDS_INPUT
        or result.status is AgentRunStatus.AWAITING_INPUT
    ):
        wait = WaitReference(
            wait_id=uuid5(NAMESPACE_URL, f"wait:input:{scope.orchestration_run_id}:{child_run_id}"),
            scope=scope,
            kind="USER_INPUT",
            source_invocation_id=invocation_id,
        )
        return None, wait, None

    # 4) Explicit human decision required with exact versioned binding -> HUMAN_DECISION wait
    if result.interaction_outcome is InteractionOutcome.ACTION_REVIEW_REQUIRED:
        proposal_id = _uuid_field(result.typed_output, "proposal_id") or _uuid_field(
            result.typed_output, "draft_id"
        )
        proposal_version = _int_field(result.typed_output, "proposal_version") or _int_field(
            result.typed_output, "draft_version"
        )
        if isinstance(result, GoalDirectedAgentResult) and result.resolved_action in {
            "assignment.recommend_team",
            "assignment.revise_team",
        }:
            value = AssignmentAgentOutput.model_validate(result.typed_output)
            recommendation = TeamRecommendationSnapshot.model_validate(value.deterministic_result)
            if (
                recommendation.organization_id != scope.organization_id
                or recommendation.status != "PROPOSED"
            ):
                return None, None, "DELEGATION_WAIT_UNVERIFIED"
            proposal_id, proposal_version = recommendation.recommendation_id, recommendation.version
        if proposal_id is None or proposal_version is None:
            return None, None, "DELEGATION_WAIT_UNVERIFIED"
        wait = WaitReference(
            wait_id=uuid5(
                NAMESPACE_URL,
                f"wait:human:{scope.orchestration_run_id}:{proposal_id}:{proposal_version}",
            ),
            scope=scope,
            kind="HUMAN_DECISION",
            source_invocation_id=invocation_id,
            resource_id=proposal_id,
            resource_version=proposal_version,
        )
        return None, wait, None

    # 5) Legacy AWAITING_HUMAN without verified proposal/workflow must not fabricate a wait.
    if result.status is AgentRunStatus.AWAITING_HUMAN:
        return None, None, "DELEGATION_WAIT_UNVERIFIED"

    # 6) Standard COMPLETED specialist result
    if result.status is AgentRunStatus.COMPLETED:
        return (
            _build_output(capability=capability, child_run_id=child_run_id, result=result),
            None,
            None,
        )

    return None, None, "DELEGATION_STATUS_UNSUPPORTED"


def normalize_specialist_result(
    scope: RunScope,
    result: AgentResult,
    *,
    invocation_id: UUID,
    call_id: str,
    child_run_id: UUID,
    capability: str = "unknown",
    workflow_state: CanonicalWorkflowState | None = None,
    payload_ref: PayloadReference | None = None,
) -> ToolOutcome:
    """Normalize a Specialist AgentResult into a canonical ToolOutcome."""

    output, wait_ref, rejected_code = _normalize_to_parts(
        scope,
        result,
        invocation_id=invocation_id,
        capability=capability,
        child_run_id=child_run_id,
        workflow_state=workflow_state,
    )
    if rejected_code is not None:
        return RejectedToolResult(
            invocation_id=invocation_id,
            call_id=call_id,
            safe_code=rejected_code,
        )
    if wait_ref is not None:
        return DeferredToolResult(
            invocation_id=invocation_id,
            call_id=call_id,
            wait_ref=wait_ref,
        )
    assert output is not None
    if payload_ref is None:
        raise DelegationError("DELEGATION_PAYLOAD_REF_REQUIRED")
    return CompletedToolResult(
        invocation_id=invocation_id,
        call_id=call_id,
        payload_ref=payload_ref,
        evidence=output.evidence,
    )


def _to_context_reference(ref: ContextRef) -> ContextReference:
    return ContextReference(
        reference_id=ref.ref_id,
        organization_id=ref.scope.organization_id,
        resource_type=ref.resource_type,
        resource_id=ref.resource_id,
        version=ref.version,
        fingerprint=ref.fingerprint,
        observed_at=ref.observed_at,
        freshness_required=True,
    )


def _normalize_typed_input(args: DelegateAgentInput) -> dict[str, JsonValue]:
    """Adapt the candidate tool contract to the existing Specialist harnesses."""
    value = args.input
    if isinstance(value, WorkQuestionDelegationInput):
        return {"question": value.question, "requested_kind": None, "locale": value.locale}
    if isinstance(value, WorkReadProjectDelegationInput):
        return {
            "question": args.objective,
            "requested_kind": "PROJECT_DETAIL",
            "entity_reference": value.project_reference,
            "locale": value.locale,
        }
    if isinstance(value, RiskExplainDelegationInput):
        return {
            "task_reference": value.task_reference,
            "question": args.objective,
            "locale": value.locale,
        }
    if isinstance(value, AssignmentAnalyzeWorkloadDelegationInput):
        try:
            project_id = str(UUID(value.project_reference))
        except ValueError:
            raise DelegationError(
                "DELEGATION_PROJECT_REFERENCE_REQUIRED", stop_segment=False
            ) from None
        return {"operation": "ANALYZE_WORKLOAD", "project_id": project_id, "locale": value.locale}
    if isinstance(value, DailyUpdatePrepareDelegationInput):
        if (value.draft_id is None) != (value.draft_version is None):
            raise DelegationError("DELEGATION_DRAFT_VERSION_REQUIRED", stop_segment=False)
        return {
            "task_id": str(value.task_reference),
            "task_version": value.task_version,
            "text": value.update_text,
            "locale": value.locale,
            "draft_id": str(value.draft_id) if value.draft_id is not None else None,
            "draft_version": value.draft_version,
        }
    raise DelegationError("DELEGATION_COMPATIBILITY_UNAVAILABLE", stop_segment=False)


def _child_interval(allocation_id: UUID) -> UsageReservation:
    return UsageReservation(
        operation_id=uuid5(allocation_id, "interval"),
        kind="INTERVAL",
        allocation_id=allocation_id,
    )


@asynccontextmanager
async def _inline_child_budget(
    context: ToolInvocationContext,
    binding: CandidateBudget,
    allocation_id: UUID,
    child_budget: AgentBudget,
    brief: DelegationBrief,
    *,
    allow_job: bool = False,
) -> AsyncGenerator[None]:
    try:
        rem = await binding.store.remaining(context.scope)
        allowance = ToolLoopUsage(
            root_model_requests=0,
            total_model_requests=min(
                child_budget.max_model_attempts, rem.available.total_model_requests
            ),
            total_tool_calls=min(child_budget.max_tool_calls, rem.available.total_tool_calls),
            handoffs=0,
            input_tokens=min(child_budget.max_input_tokens, rem.available.input_tokens),
            output_tokens=min(child_budget.max_output_tokens, rem.available.output_tokens),
            segments=2 if allow_job else 1,
            active_seconds=float(context.limits.segment_seconds * (2 if allow_job else 1)),
        )
        if (
            (allowance.total_model_requests == 0 and allowance.total_tool_calls == 0)
            or rem.available.segments < allowance.segments
            or rem.available.active_seconds < allowance.active_seconds
        ):
            raise BudgetError("BUDGET_EXHAUSTED")
        await binding.store.allocate_child(
            context.scope,
            ChildAllocation(allocation_id=allocation_id, allowance=allowance),
            lease=context.lease,
        )
    except BudgetError:
        raise DelegationError("DELEGATION_BUDGET_EXHAUSTED", stop_segment=True) from None

    try:
        child_interval = await binding.store.reserve(
            context.scope,
            _child_interval(allocation_id),
            lease=context.lease,
        )
        try:
            child_binding = CandidateBudget(
                scope=context.scope,
                lease=context.lease,
                store=binding.store,
                interval_id=child_interval.reservation.operation_id,
                allocation_id=allocation_id,
                delegation_brief=brief,
                model_output_cap=allowance.output_tokens // max(1, allowance.total_model_requests),
            )
            with candidate_budget(child_binding):
                yield
        finally:
            await binding.store.settle(context.scope, child_interval, None)
    finally:
        await binding.store.close_child(context.scope, allocation_id, lease=context.lease)


async def _settle_replayed_child(context: ToolInvocationContext) -> None:
    binding = current_candidate_budget()
    if binding is None:
        return
    if binding.scope != context.scope or binding.lease != context.lease:
        raise DelegationError("DELEGATION_BUDGET_UNAVAILABLE", stop_segment=True)
    allocation_id = uuid5(context.receipt_id, "child-budget")
    await binding.store.settle(
        context.scope,
        BudgetReceipt(
            scope=context.scope,
            lease=context.lease,
            reservation=_child_interval(allocation_id),
            replayed=True,
        ),
        None,
    )
    await binding.store.close_child(context.scope, allocation_id, lease=context.lease)


def _check_tool_pin(context: ToolInvocationContext, tools: ToolRegistry, reference: str) -> None:
    if context.tool_reference != reference:
        raise DelegationError("DELEGATION_TOOL_MISMATCH", stop_segment=True)
    try:
        pinned = tools.resolve(reference).manifest.version
    except ToolRegistryError:
        raise DelegationError("DELEGATION_TOOL_MISMATCH", stop_segment=True) from None
    if context.tool_version != pinned:
        raise DelegationError("DELEGATION_VERSION_CONFLICT", stop_segment=True)


class DelegationAdapter:
    """Guarded Specialist-as-tool adapter reusing registry, context validation and budgets."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        specialist_pins: tuple[VersionPin, ...],
        tool_registry: ToolRegistry | None = None,
        runner: SpecialistDelegationRunnerPort,
        actor_role: Literal["ADMIN", "MANAGER", "EMPLOYEE"],
        active_phase: int,
        parent_agent_run_id: UUID,
        activation: ActivationContext | None = None,
        context_validator: ContextUseValidator | None = None,
        workflow_verifier: WorkflowWaitVerifierPort | None = None,
        daily_update_receipts: DailyUpdateDraftReceiptPort | None = None,
    ) -> None:
        self._agents = agent_registry
        self._specialist_pins = {pin.reference: pin for pin in specialist_pins}
        if len(self._specialist_pins) != len(specialist_pins):
            raise DelegationError("DELEGATION_TARGET_PIN_DUPLICATE", stop_segment=True)
        self._tools = (
            ToolRegistry.from_resources(_DELEGATE_TOOL_RESOURCE)
            if tool_registry is None
            else tool_registry
        )
        self._runner = runner
        self._actor_role = actor_role
        self._active_phase = active_phase
        self._parent_agent_run_id = parent_agent_run_id
        self._activation = activation
        self._context_validator = context_validator
        self._workflow_verifier = workflow_verifier
        self._daily_update_receipts = daily_update_receipts

    def _supported(self, args: DelegateAgentInput) -> bool:
        key = (args.agent_id, args.capability)
        return key in _CANDIDATE_CAPABILITIES or (
            key == _DAILY_UPDATE_CAPABILITY and self._daily_update_receipts is not None
        )

    def _resolve_target(self, args: DelegateAgentInput) -> tuple[str, AgentBudget]:
        if args.agent_id is AgentId.ORCHESTRATOR:
            raise DelegationError("DELEGATION_SELF_FORBIDDEN", stop_segment=True)
        pin = self._specialist_pins.get(args.agent_id.value)
        if pin is None:
            raise DelegationError("DELEGATION_TARGET_UNAVAILABLE", stop_segment=False)
        try:
            registered = self._agents.resolve(
                args.agent_id,
                pin.version,
                self._active_phase,
                activation=self._activation,
            )
        except AgentRegistryError:
            raise DelegationError("DELEGATION_TARGET_UNAVAILABLE", stop_segment=False) from None
        if registered.fingerprint != pin.fingerprint:
            raise DelegationError("DELEGATION_TARGET_PIN_MISMATCH", stop_segment=True)
        manifest = registered.manifest
        if args.capability not in manifest.capabilities:
            raise DelegationError("DELEGATION_CAPABILITY_MISMATCH", stop_segment=False)
        if self._actor_role not in manifest.permissions.roles:
            raise DelegationError("DELEGATION_ROLE_FORBIDDEN", stop_segment=False)
        rt = manifest.runtime
        budget = AgentBudget(
            max_model_attempts=rt.max_model_attempts,
            max_input_tokens=rt.max_input_tokens,
            max_output_tokens=rt.max_output_tokens,
            max_iterations=rt.max_iterations,
            max_tool_calls=rt.max_tool_calls,
            max_handoffs=0,
            max_replans=rt.max_replans,
            timeout_seconds=rt.timeout_seconds,
        )
        return manifest.agent.version, budget

    async def _validate_context_refs(
        self, context: ToolInvocationContext, args: DelegateAgentInput
    ) -> tuple[ContextReference, ...]:
        if not args.context_ref_ids:
            return ()
        if self._context_validator is None or self._context_validator.projector.current is None:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False)
        report = self._context_validator.projector.current
        known = {ref.ref_id: ref for ref in report.context_refs}
        resolved: list[ContextRef] = []
        for ref_id in args.context_ref_ids:
            ref = known.get(ref_id)
            if ref is None or ref.scope != context.scope:
                raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False)
            resolved.append(ref)
        validation = await self._context_validator.validate(
            context.scope,
            target_tool=f"delegate_agent:{args.capability}",
            input_fingerprint=context.arguments_fingerprint,
            refs=tuple(resolved),
        )
        if validation.outcome == "NEEDS_INPUT":
            raise DelegationError("DELEGATION_CONTEXT_NEEDS_INPUT", stop_segment=False)
        if validation.outcome != "ACCEPTED":
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False)
        return tuple(_to_context_reference(ref) for ref in resolved)

    async def _verify_workflow(
        self, scope: RunScope, result: AgentResult
    ) -> CanonicalWorkflowState | None:
        wf_id = _uuid_field(result.typed_output, "workflow_run_id")
        if wf_id is None or self._workflow_verifier is None:
            return None
        return await self._workflow_verifier.verify_workflow(scope, wf_id)

    def _idempotency_key(self, context: ToolInvocationContext) -> str:
        return f"delegate:{context.receipt_id}"

    async def _recover_daily_result(
        self,
        context: ToolInvocationContext,
        args: DelegateAgentInput,
        target_version: str,
        result: AgentResult | None,
    ) -> AgentResult | None:
        if self._daily_update_receipts is None or not isinstance(
            args.input, DailyUpdatePrepareDelegationInput
        ):
            return result
        if result is not None and result.status is not AgentRunStatus.AWAITING_HUMAN:
            return result
        value = args.input
        receipt = await self._daily_update_receipts.recover_draft(
            context.scope,
            receipt_id=context.receipt_id,
            task_id=value.task_reference,
            task_version=value.task_version,
        )
        if receipt is None:
            return result
        if result is not None:
            if DailyUpdateResult.model_validate(result.typed_output) != receipt:
                raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
            return result.model_copy(
                update={
                    "interaction_outcome": InteractionOutcome.ACTION_REVIEW_REQUIRED,
                }
            )
        return AgentResult(
            agent_id=args.agent_id,
            agent_version=target_version,
            status=AgentRunStatus.AWAITING_HUMAN,
            interaction_outcome=InteractionOutcome.ACTION_REVIEW_REQUIRED,
            typed_output=receipt.model_dump(mode="json"),
            stop_reason="DAILY_UPDATE_OWNER_REVIEW",
        )

    async def revalidate_wait(self, context: ToolInvocationContext, wait: WaitReference) -> None:
        if wait.kind != "HUMAN_DECISION":
            return
        args = DelegateAgentInput.model_validate(context.canonical_arguments.model_dump())
        if not isinstance(args.input, DailyUpdatePrepareDelegationInput):
            return
        target_version, _ = self._resolve_target(args)
        await self._validate_context_refs(context, args)
        current = await self._recover_daily_result(context, args, target_version, None)
        if current is None or (
            _uuid_field(current.typed_output, "draft_id"),
            _int_field(current.typed_output, "draft_version"),
        ) != (wait.resource_id, wait.resource_version):
            raise DelegationError("DELEGATION_WAIT_STALE", stop_segment=True)

    async def _to_backend_result(
        self,
        context: ToolInvocationContext,
        args: DelegateAgentInput,
        *,
        child_run_id: UUID,
        result: AgentResult,
    ) -> BackendToolResult:
        workflow_state = await self._verify_workflow(context.scope, result)
        output, wait_ref, rejected_code = _normalize_to_parts(
            context.scope,
            result,
            invocation_id=context.invocation_id,
            capability=args.capability,
            child_run_id=child_run_id,
            workflow_state=workflow_state,
        )
        if rejected_code is not None:
            raise DelegationError(rejected_code, stop_segment=False)
        if wait_ref is not None:
            return BackendToolResult(wait_ref=wait_ref, evidence=result.evidence)
        assert output is not None
        return BackendToolResult(output=output, evidence=output.evidence)

    async def execute_backend(
        self, context: ToolInvocationContext, value: DelegateAgentInput
    ) -> BackendToolResult:
        """Execute delegation and return BackendToolResult for ToolPipeline settlement."""

        args = DelegateAgentInput.model_validate(value.model_dump())
        _check_tool_pin(context, self._tools, "delegate_agent@1")
        target_version, child_budget = self._resolve_target(args)
        if not self._supported(args):
            raise DelegationError("DELEGATION_COMPATIBILITY_UNAVAILABLE", stop_segment=False)
        context_refs = await self._validate_context_refs(context, args)
        typed_input = _normalize_typed_input(args)

        binding = current_candidate_budget()
        if binding is None or binding.scope != context.scope or binding.lease != context.lease:
            raise DelegationError("DELEGATION_BUDGET_UNAVAILABLE", stop_segment=True)

        if binding.allocation_id is not None:
            raise DelegationError("DELEGATION_PEER_FORBIDDEN", stop_segment=True)

        # The pipeline reserves and settles the HANDOFF. The child allocation is
        # a separate ledger entry with an identity stable across replays.
        allocation_id = uuid5(context.receipt_id, "child-budget")
        idempotency_key = self._idempotency_key(context)
        handoff = AgentHandoff(
            orchestration_run_id=context.scope.orchestration_run_id,
            parent_agent_run_id=self._parent_agent_run_id,
            target_agent_id=args.agent_id,
            target_agent_version=target_version,
            capability=args.capability,
            objective=args.objective,
            typed_input=typed_input,
            context_references=context_refs,
            actor=ActorReference(
                organization_id=context.scope.organization_id,
                membership_id=context.scope.membership_id,
            ),
            budget=child_budget,
            step_id=f"delegate:{context.receipt_id}",
            idempotency_key=idempotency_key,
        )

        brief = DelegationBrief(
            objective=args.objective,
            constraints=args.constraints,
            expected_result=args.expected_result,
        )
        async with _inline_child_budget(context, binding, allocation_id, child_budget, brief):
            result = await self._runner.run_specialist(handoff)

        if (result.agent_id, result.agent_version) != (args.agent_id, target_version):
            raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        recorded = await self._runner.lookup_agent_run(idempotency_key)
        if recorded is None:
            raise DelegationError("DELEGATION_RECEIPT_REQUIRED", stop_segment=True)
        result = await self._recover_daily_result(context, args, target_version, result)
        assert result is not None
        return await self._to_backend_result(context, args, child_run_id=recorded.id, result=result)

    async def execute(
        self,
        context: ToolInvocationContext,
        value: DelegateAgentInput,
        *,
        payload_ref: PayloadReference | None = None,
    ) -> CompletedToolResult | DeferredToolResult:
        """Direct convenience interface returning CompletedToolResult | DeferredToolResult."""

        backend = await self.execute_backend(context, value)
        if backend.wait_ref is not None:
            return DeferredToolResult(
                invocation_id=context.invocation_id,
                call_id=context.call_id,
                wait_ref=backend.wait_ref,
            )
        if payload_ref is None:
            raise DelegationError("DELEGATION_PAYLOAD_REF_REQUIRED")
        return CompletedToolResult(
            invocation_id=context.invocation_id,
            call_id=context.call_id,
            payload_ref=payload_ref,
            evidence=backend.evidence,
        )

    async def reconcile(
        self, context: ToolInvocationContext, value: DelegateAgentInput
    ) -> BackendToolResult | None:
        """Recover an already-committed child AgentRun/workflow receipt without re-running."""

        args = DelegateAgentInput.model_validate(value.model_dump())
        _check_tool_pin(context, self._tools, "delegate_agent@1")
        target_version, _ = self._resolve_target(args)
        await self._validate_context_refs(context, args)
        recorded = await self._runner.lookup_agent_run(self._idempotency_key(context))
        if recorded is None:
            return None
        if recorded.target_agent_id is not None or recorded.replayed_result is None:
            pin = self._specialist_pins[args.agent_id.value]
            if (
                recorded.target_agent_id,
                recorded.target_agent_version,
                recorded.capability,
                recorded.manifest_fingerprint,
            ) != (args.agent_id, target_version, args.capability, pin.fingerprint):
                raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        result = recorded.replayed_result
        if result is not None and (result.agent_id, result.agent_version) != (
            args.agent_id,
            target_version,
        ):
            raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        result = await self._recover_daily_result(context, args, target_version, result)
        if result is None:
            if (
                args.capability != "planning.create"
                or recorded.status is not AgentRunStatus.RUNNING
                or self._workflow_verifier is None
            ):
                return None
            workflow_id = await self._workflow_verifier.find_committed_planning_create(
                context.scope,
                child_run_id=recorded.id,
                receipt_id=context.receipt_id,
                call_id=context.call_id,
            )
            if workflow_id is None:
                return None
            result = AgentResult(
                agent_id=args.agent_id,
                agent_version=target_version,
                status=AgentRunStatus.RUNNING,
                typed_output={"workflow_run_id": str(workflow_id)},
                stop_reason="DELEGATION_EFFECT_RECOVERED",
            )
        if self._supported(args):
            await _settle_replayed_child(context)
        return await self._to_backend_result(context, args, child_run_id=recorded.id, result=result)


class GoalSpecialistRunnerPort(Protocol):
    """Persist and recover goal-directed child results under the exact handoff identity."""

    async def run(self, handoff: AgentHandoffV2) -> tuple[UUID, GoalDirectedAgentResult]: ...

    async def reconcile(
        self, handoff: AgentHandoffV2
    ) -> tuple[UUID, GoalDirectedAgentResult] | None: ...


class DelegationReferencesPort(Protocol):
    """Resolve and recheck current, server-verified references within the specified run."""

    async def resolve(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]: ...

    async def revalidate(self, scope: RunScope, refs: tuple[ContextReference, ...]) -> None: ...

    def recorded(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]: ...


class VerifiedDelegationReferences:
    """Index provenance already established by the trusted turn and runtime."""

    def __init__(
        self,
        *,
        scope: RunScope,
        exploration: Iterable[ContextReference],
        active_context: Iterable[ContextReference],
        completed_specialists: Iterable[ContextReference],
        recheck: Callable[[RunScope, tuple[ContextReference, ...]], Awaitable[None]],
    ) -> None:
        self._scope = scope
        self._recheck = recheck
        self._refs: dict[UUID, ContextReference] = {}
        for source in (exploration, active_context, completed_specialists):
            for ref in source:
                known = self._refs.setdefault(ref.reference_id, ref)
                if ref.organization_id != scope.organization_id or ref != known:
                    raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)

    async def resolve(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]:
        refs = self.recorded(scope, reference_ids)
        await self.revalidate(scope, refs)
        return refs

    def recorded(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]:
        if scope != self._scope:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        try:
            return tuple(self._refs[ref_id] for ref_id in reference_ids)
        except KeyError:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False) from None

    async def revalidate(self, scope: RunScope, refs: tuple[ContextReference, ...]) -> None:
        if scope != self._scope or any(
            ref.organization_id != scope.organization_id for ref in refs
        ):
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        if refs:
            await self._recheck(scope, refs)


class DelegationAdapterV2:
    """Bind a goal to a ready Specialist under a pinned, run-scoped configuration."""

    def __init__(
        self,
        *,
        scope: RunScope,
        parent_agent_run_id: UUID,
        actor_role: Literal["ADMIN", "MANAGER", "EMPLOYEE"],
        active_phase: int,
        locale: Literal["vi", "en"],
        timezone: str,
        agent_registry: AgentRegistry,
        specialist_pins: tuple[VersionPin, ...],
        runners: Mapping[AgentId, GoalSpecialistRunnerPort],
        tool_registry: ToolRegistry | None = None,
        activation: ActivationContext | None = None,
        references: DelegationReferencesPort | None = None,
        workflow_verifier: WorkflowWaitVerifierPort | None = None,
        child_job_receipts: ChildJobReceiptsPort | None = None,
        daily_update_goal_receipts: GoalDailyUpdateDraftReceiptPort | None = None,
    ) -> None:
        self._scope = scope
        self._parent_id = parent_agent_run_id
        self._role = actor_role
        self._phase = active_phase
        self._locale: Literal["vi", "en"] = locale
        self._timezone = timezone
        self._registry = agent_registry
        self._tools = (
            ToolRegistry.from_resources(_DELEGATE_V2_TOOL_RESOURCE)
            if tool_registry is None
            else tool_registry
        )
        self._pins = {pin.reference: pin for pin in specialist_pins}
        if len(self._pins) != len(specialist_pins):
            raise DelegationError("DELEGATION_TARGET_PIN_DUPLICATE", stop_segment=True)
        self._runners = dict(runners)
        self._activation = activation
        self._references = references
        self._workflow_verifier = workflow_verifier
        self._child_jobs = child_job_receipts
        self._daily_goal_receipts = daily_update_goal_receipts

    def _target(
        self, agent_id: AgentId
    ) -> tuple[GoalDirectedAgentManifest, GoalSpecialistRunnerPort]:
        pin = self._pins.get(agent_id.value)
        runner = self._runners.get(agent_id)
        if (
            agent_id is AgentId.ORCHESTRATOR
            or pin is None
            or runner is None
            or (agent_id is AgentId.DAILY_UPDATE and self._daily_goal_receipts is None)
        ):
            raise DelegationError("DELEGATION_TARGET_UNAVAILABLE", stop_segment=False)
        try:
            target = self._registry.resolve(
                agent_id, pin.version, self._phase, activation=self._activation
            )
        except AgentRegistryError:
            raise DelegationError("DELEGATION_TARGET_UNAVAILABLE", stop_segment=False) from None
        if target.fingerprint != pin.fingerprint or not isinstance(
            target.manifest, GoalDirectedAgentManifest
        ):
            raise DelegationError("DELEGATION_TARGET_PIN_MISMATCH", stop_segment=True)
        if self._role not in target.manifest.permissions.roles:
            raise DelegationError("DELEGATION_ROLE_FORBIDDEN", stop_segment=False)
        return target.manifest, runner

    def available_specialists(self) -> tuple[tuple[AgentId, str], ...]:
        available: list[tuple[AgentId, str]] = []
        for agent_id in sorted(self._runners):
            try:
                manifest, _ = self._target(agent_id)
            except DelegationError as error:
                if error.stop_segment:
                    raise
                continue
            available.append((agent_id, manifest.delegation_summary))
        return tuple(available)

    async def _prepare(
        self,
        context: ToolInvocationContext,
        arguments: BaseModel,
        *,
        receipt_read: bool = False,
    ) -> tuple[AgentHandoffV2, GoalDirectedAgentManifest, GoalSpecialistRunnerPort]:
        if context.scope != self._scope or context.lease.scope != self._scope:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        _check_tool_pin(context, self._tools, "delegate_agent@2")
        args = DelegateAgentInputV2.model_validate(arguments.model_dump())
        if args != DelegateAgentInputV2.model_validate(context.canonical_arguments.model_dump()):
            raise DelegationError("TOOL_ARGUMENTS_CHANGED", stop_segment=True)
        manifest, runner = self._target(args.agent_id)
        refs: tuple[ContextReference, ...] = ()
        if args.context_ref_ids:
            if self._references is None:
                raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False)
            # A successful effect may advance the very version that authorized dispatch.
            if receipt_read and args.agent_id in {AgentId.ASSIGNMENT, AgentId.DAILY_UPDATE}:
                refs = self._references.recorded(context.scope, args.context_ref_ids)
            else:
                refs = await self._references.resolve(context.scope, args.context_ref_ids)
            if tuple(ref.reference_id for ref in refs) != args.context_ref_ids or any(
                ref.organization_id != context.scope.organization_id for ref in refs
            ):
                raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=False)
        rt = manifest.runtime
        handoff = AgentHandoffV2(
            orchestration_run_id=context.scope.orchestration_run_id,
            parent_agent_run_id=self._parent_id,
            target_agent_id=args.agent_id,
            target_agent_version=manifest.agent.version,
            brief=DelegationBrief(
                objective=args.objective,
                constraints=args.constraints,
                expected_result=args.expected_result,
            ),
            context_references=refs,
            actor=ActorReference(
                organization_id=context.scope.organization_id,
                membership_id=context.scope.membership_id,
            ),
            source_turn_ref=context.scope.turn_id,
            locale=self._locale,
            timezone=self._timezone,
            budget=AgentBudget(
                max_model_attempts=rt.max_model_attempts,
                max_input_tokens=rt.max_input_tokens,
                max_output_tokens=rt.max_output_tokens,
                max_iterations=rt.max_iterations,
                max_tool_calls=rt.max_tool_calls,
                max_replans=rt.max_replans,
                timeout_seconds=rt.timeout_seconds,
            ),
            step_id=f"delegate:{context.receipt_id}",
            idempotency_key=f"goal-delegate:{context.receipt_id}",
        )
        return handoff, manifest, runner

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        handoff, manifest, runner = await self._prepare(context, arguments)
        binding = current_candidate_budget()
        if binding is None or binding.scope != context.scope or binding.lease != context.lease:
            raise DelegationError("DELEGATION_BUDGET_UNAVAILABLE", stop_segment=True)
        if binding.allocation_id is not None:
            raise DelegationError("DELEGATION_PEER_FORBIDDEN", stop_segment=True)
        allocation_id = uuid5(context.receipt_id, "child-budget")
        async with _inline_child_budget(
            context,
            binding,
            allocation_id,
            handoff.budget,
            handoff.brief,
            allow_job=handoff.target_agent_id in {AgentId.PLANNING, AgentId.REPORTING},
        ):
            record = await runner.run(handoff)
        if handoff.target_agent_id is AgentId.DAILY_UPDATE:
            verified = await self._recover_goal_daily(context, handoff, record)
            if verified is None:
                raise DelegationError("DELEGATION_WAIT_UNVERIFIED", stop_segment=True)
            record = verified
        return await self._normalize(context, manifest, record)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        recovered = await self._recover(context, arguments)
        if recovered is not None and (
            recovered.wait_ref is None or recovered.wait_ref.kind != "WORKFLOW_RESULT"
        ):
            await _settle_replayed_child(context)
        return recovered

    async def _recover(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        handoff, manifest, runner = await self._prepare(context, arguments, receipt_read=True)
        record = await runner.reconcile(handoff)
        if handoff.target_agent_id is AgentId.DAILY_UPDATE:
            record = await self._recover_goal_daily(context, handoff, record)
        committed = None
        if record is None:
            child_id = uuid5(NAMESPACE_URL, f"agent-run:{handoff.idempotency_key}")
            committed = await self._read_child_job(context, handoff.target_agent_id, child_id)
            if committed is not None:
                key = (
                    "generation_id"
                    if handoff.target_agent_id is AgentId.REPORTING
                    else "workflow_run_id"
                )
                record = (
                    child_id,
                    GoalDirectedAgentResult(
                        agent_id=handoff.target_agent_id,
                        agent_version=handoff.target_agent_version,
                        status=AgentRunStatus.AWAITING_HUMAN,
                        typed_output={key: str(committed.resource_id)},
                        resolved_action=committed.resolved_action,
                        stop_reason="DELEGATION_JOB_RECOVERED",
                    ),
                )
        if record is None:
            return None
        if self._references is not None and (
            (
                handoff.target_agent_id is AgentId.ASSIGNMENT
                and record[1].resolved_action not in _ASSIGNMENT_EFFECTS
            )
            or (
                handoff.target_agent_id is AgentId.DAILY_UPDATE
                and record[1].status is not AgentRunStatus.AWAITING_HUMAN
            )
        ):
            await self._references.revalidate(context.scope, handoff.context_references)
        return await self._normalize(context, manifest, record, committed=committed)

    async def _recover_goal_daily(
        self,
        context: ToolInvocationContext,
        handoff: AgentHandoffV2,
        record: tuple[UUID, GoalDirectedAgentResult] | None,
    ) -> tuple[UUID, GoalDirectedAgentResult] | None:
        saved = record[1] if record is not None else None
        if saved is not None and saved.status not in {
            AgentRunStatus.AWAITING_HUMAN,
            AgentRunStatus.FAILED,
        }:
            return record
        failed = record if saved is not None and saved.status is AgentRunStatus.FAILED else None
        tasks = [
            ref for ref in handoff.context_references if ref.resource_type.casefold() == "task"
        ]
        if (
            self._daily_goal_receipts is None
            or len(tasks) != 1
            or (task_version := tasks[0].version) is None
        ):
            return failed
        receipt = await self._daily_goal_receipts.recover_goal_draft(
            context.scope,
            receipt_id=context.receipt_id,
            task_id=tasks[0].resource_id,
            task_version=task_version,
        )
        if receipt is None:
            return failed
        drafts = [
            ref
            for ref in handoff.context_references
            if ref.resource_type.casefold() == "daily_update_draft"
        ]
        if drafts and (
            len(drafts) != 1
            or drafts[0].resource_id != receipt.draft_id
            or drafts[0].version != receipt.draft_version - 1
        ):
            raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        if saved is not None and saved.status is AgentRunStatus.AWAITING_HUMAN:
            if (
                saved.resolved_action != "daily_update.prepare"
                or saved.interaction_outcome is not InteractionOutcome.ACTION_REVIEW_REQUIRED
                or DailyUpdateResult.model_validate(saved.typed_output) != receipt
            ):
                raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
            return record
        child_id = uuid5(NAMESPACE_URL, f"agent-run:{handoff.idempotency_key}")
        return child_id, GoalDirectedAgentResult(
            agent_id=AgentId.DAILY_UPDATE,
            agent_version=handoff.target_agent_version,
            status=AgentRunStatus.AWAITING_HUMAN,
            interaction_outcome=InteractionOutcome.ACTION_REVIEW_REQUIRED,
            typed_output=receipt.model_dump(mode="json"),
            stop_reason="DAILY_UPDATE_OWNER_REVIEW",
            resolved_action="daily_update.prepare",
        )

    async def _read_child_job(
        self, context: ToolInvocationContext, agent_id: AgentId, child_id: UUID
    ) -> ChildJobReceipt | None:
        if self._child_jobs is None or agent_id not in {AgentId.PLANNING, AgentId.REPORTING}:
            return None
        return await self._child_jobs.recover_job(
            context.scope,
            allocation_id=uuid5(context.receipt_id, "child-budget"),
            child_run_id=child_id,
            target=self._pins[agent_id.value],
        )

    async def revalidate_result(
        self, context: ToolInvocationContext, result: BackendToolResult
    ) -> None:
        refs = result.evidence
        if result.wait_ref is not None:
            await self.revalidate_wait(context, result.wait_ref)
        else:
            handoff, manifest, runner = await self._prepare(
                context, context.canonical_arguments, receipt_read=True
            )
            value = result.output
            if not isinstance(value, DelegateAgentOutputV2) or (
                (value.agent_id, value.agent_version) != (manifest.agent.id, manifest.agent.version)
                or value.resolved_action not in manifest.capabilities
            ):
                raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
            if value.agent_id in {AgentId.RISK, AgentId.ASSIGNMENT}:
                record = await runner.reconcile(handoff)
                if record is None or await self._normalize(context, manifest, record) != result:
                    raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
                if value.agent_id is AgentId.RISK or value.resolved_action in _ASSIGNMENT_EFFECTS:
                    return
            if self._references is not None and value.agent_id in {
                AgentId.ASSIGNMENT,
                AgentId.DAILY_UPDATE,
            }:
                await self._references.revalidate(context.scope, handoff.context_references)
            refs = tuple(dict.fromkeys((*refs, *value.evidence, *value.artifact_refs)))
        if refs:
            if self._references is None:
                raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
            await self._references.revalidate(context.scope, refs)

    async def revalidate_wait(self, context: ToolInvocationContext, wait: WaitReference) -> None:
        current = await self._recover(context, context.canonical_arguments)
        if current is None or current.wait_ref != wait:
            raise DelegationError("DELEGATION_WAIT_STALE", stop_segment=True)

    async def _normalize(
        self,
        context: ToolInvocationContext,
        manifest: GoalDirectedAgentManifest,
        record: tuple[UUID, GoalDirectedAgentResult],
        *,
        committed: ChildJobReceipt | None = None,
    ) -> BackendToolResult:
        child_id, result = record
        action = result.resolved_action
        if (result.agent_id, result.agent_version) != (manifest.agent.id, manifest.agent.version):
            raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        if action is not None and action not in manifest.capabilities:
            raise DelegationError("DELEGATION_CAPABILITY_MISMATCH", stop_segment=True)
        workflow_id = _uuid_field(result.typed_output, "workflow_run_id")
        generation_id = _uuid_field(result.typed_output, "generation_id")
        workflow = None
        if manifest.agent.id is AgentId.REPORTING and generation_id is not None:
            if self._child_jobs is None:
                raise DelegationError("DELEGATION_WAIT_UNVERIFIED", stop_segment=True)
            workflow = await self._child_jobs.verify_reporting_job(context.scope, generation_id)
            result = result.model_copy(
                update={
                    "typed_output": {**result.typed_output, "workflow_run_id": str(generation_id)},
                }
            )
        elif workflow_id is not None:
            if manifest.agent.id is AgentId.PLANNING and action in {
                "planning.create",
                "planning.resume",
                "planning.revise",
            }:
                if committed is None:
                    committed = await self._read_child_job(context, AgentId.PLANNING, child_id)
                if committed is None and action == "planning.revise":
                    raise DelegationError("DELEGATION_WAIT_UNVERIFIED", stop_segment=True)
                if committed is not None:
                    if (committed.resource_id, committed.resolved_action) != (workflow_id, action):
                        raise DelegationError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
                    workflow = committed.workflow_state
            if workflow is None and self._workflow_verifier is not None:
                workflow = await self._workflow_verifier.verify_workflow(context.scope, workflow_id)
        output, wait, failure = _normalize_to_parts(
            context.scope,
            result,
            invocation_id=context.invocation_id,
            capability=action or "unresolved",
            child_run_id=child_id,
            workflow_state=workflow,
        )
        if failure is not None:
            raise DelegationError(failure, stop_segment=False)
        if wait is not None:
            return BackendToolResult(wait_ref=wait, evidence=result.evidence)
        if output is None or action is None:
            raise DelegationError("DELEGATION_UNVERIFIED_RESULT", stop_segment=True)
        value = DelegateAgentOutputV2.model_validate(
            {**output.model_dump(exclude={"capability"}), "resolved_action": action}
        )
        return BackendToolResult(output=value, evidence=value.evidence)


class DelegationToolBackend:
    """ToolPipeline-compatible backend wrapper around DelegationAdapter."""

    def __init__(self, adapter: DelegationAdapter) -> None:
        self.adapter = adapter

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        args = DelegateAgentInput.model_validate(arguments.model_dump())
        return await self.adapter.execute_backend(context, args)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        args = DelegateAgentInput.model_validate(arguments.model_dump())
        return await self.adapter.reconcile(context, args)
