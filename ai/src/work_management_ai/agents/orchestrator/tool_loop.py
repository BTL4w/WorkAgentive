"""Native model-driven Orchestrator tool loop with durable recovery and verified finals."""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4, uuid5

from work_management_ai.agents.orchestrator.contracts import (
    OrchestratorInput,
    OrchestratorStatus,
)
from work_management_ai.agents.orchestrator.evaluators.tool_loop import (
    FinalVerificationError,
    verify_final_candidate,
)
from work_management_ai.agents.orchestrator.prompts.tool_loop_v1 import (
    PROMPT_VERSION,
    build_tool_loop_prompt,
)
from work_management_ai.agents.orchestrator.tool_loop_contracts import (
    ModelFinalResponse,
    ObservedArtifactState,
    OrchestratorExecutionResult,
    OrchestratorOutputV2,
    PublishedResponse,
    SuspendedWorkflow,
)
from work_management_ai.agents.orchestrator.workflows.tool_loop_graph import (
    ToolLoopGraph,
    ToolLoopGraphState,
)
from work_management_ai.model_gateway.errors import (
    ModelInvalidOutputError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from work_management_ai.model_gateway.tool_contracts import (
    NativeToolGateway,
    ToolCall,
    ToolItem,
    ToolModelRequest,
    ToolSchema,
)
from work_management_ai.runtime.context_exploration import ExplorationProjector
from work_management_ai.runtime.contracts import (
    ActivityResponseBlock,
    AgentId,
    ContextReference,
    InteractionOutcome,
    JsonValue,
    QuestionResponseBlock,
    ResponseBlock,
    SafeErrorResponseBlock,
    TextResponseBlock,
)
from work_management_ai.runtime.delegation import filtered_delegate_tool_schema
from work_management_ai.runtime.run_budget import (
    BudgetedToolLoopGateway,
    BudgetError,
    CandidateBudget,
    RunBudgetPort,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.session_history import (
    ContextAccessPort,
    ContextPayload,
    HistoryLimits,
    MessagePayload,
    SessionHistory,
    ToolArgumentsPayload,
    ToolResultPayload,
)
from work_management_ai.runtime.session_log import EventDraft, SessionLogPort
from work_management_ai.runtime.skill_loading import SkillLoader, SkillLoadingError
from work_management_ai.runtime.tool_catalog import PreparedToolSet, ToolCatalog, ToolCatalogError
from work_management_ai.runtime.tool_loop_contracts import (
    CompletedToolResult,
    DeferredToolResult,
    RejectedToolResult,
    RunLease,
    RunScope,
    ToolLoopCheckpoint,
    ToolLoopLimits,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import ToolPipeline
from work_management_ai.tools.context.contracts import ContextReadObservation
from work_management_ai.tools.delegation.contracts import DelegateAgentOutput

_CONTEXT_READ_TOOLS = frozenset(
    {"explore_projects@1", "read_project_context@1", "read_actor_context@1"}
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _strip_ephemeral(value: JsonValue) -> JsonValue:
    """Strip per-call timestamps/invocation IDs so identical repeated reads hash identically."""

    if isinstance(value, dict):
        return {
            k: _strip_ephemeral(v)
            for k, v in value.items()
            if k not in {"invocation_id", "observed_at", "child_agent_run_id"}
        }
    if isinstance(value, list):
        return [_strip_ephemeral(item) for item in value]
    return value


def _verified_context_summary(value: OrchestratorInput) -> dict[str, JsonValue] | None:
    """Bounded summary of already-verified active context (never dumps all Projects/profiles)."""

    ctx = value.active_context
    summary: dict[str, JsonValue] = {}
    if ctx.active_planning is not None:
        summary["active_planning"] = {
            "workflow_run_id": str(ctx.active_planning.workflow_run_id),
            "workflow_status": ctx.active_planning.workflow_status,
            "draft_id": str(ctx.active_planning.draft_id)
            if ctx.active_planning.draft_id is not None
            else None,
            "draft_version": ctx.active_planning.draft_version,
            "proposal_id": str(ctx.active_planning.proposal_id)
            if ctx.active_planning.proposal_id is not None
            else None,
            "proposal_version": ctx.active_planning.proposal_version,
        }
    if ctx.active_team is not None:
        summary["active_team"] = {
            "project_id": str(ctx.active_team.project_id),
            "recommendation_id": str(ctx.active_team.recommendation_id)
            if ctx.active_team.recommendation_id is not None
            else None,
            "recommendation_version": ctx.active_team.recommendation_version,
        }
    if ctx.exact_assignment is not None:
        summary["exact_assignment"] = {
            "project_id": str(ctx.exact_assignment.project_id),
            "task_id": str(ctx.exact_assignment.task_id),
            "task_version": ctx.exact_assignment.task_version,
        }
    return summary or None


def _empty_str_list() -> list[str]:
    return []


def _empty_evidence_list() -> list[ContextReference]:
    return []


def _empty_block_list() -> list[ResponseBlock]:
    return []


@dataclass
class _SegmentRun:
    value: OrchestratorInput
    scope: RunScope
    lease: RunLease
    interval_id: UUID
    checkpoint: ToolLoopCheckpoint
    admitted: PreparedToolSet | None = None
    step_items: tuple[ToolItem, ...] = ()
    step_tools: tuple[ToolSchema, ...] = ()
    model_refs: list[str] = field(default_factory=_empty_str_list)
    observed_evidence: list[ContextReference] = field(default_factory=_empty_evidence_list)
    observed_call_ids: list[str] = field(default_factory=_empty_str_list)
    draft_available: bool = False
    proposal_pending: bool = False
    applied_write: bool = False
    verified_blocks: list[ResponseBlock] = field(default_factory=_empty_block_list)
    last_progress_signature: tuple[str, str, str, str] | None = None
    last_invocation_id: UUID | None = None
    recovery_error: str | None = None

    def close_admitted(self) -> None:
        if self.admitted is not None:
            self.admitted.close()
            self.admitted = None


def _collect_context_evidence(run: _SegmentRun, obs: ContextReadObservation) -> None:
    for ref in obs.context_refs:
        run.observed_evidence.append(
            ContextReference(
                reference_id=ref.ref_id,
                organization_id=ref.scope.organization_id,
                resource_type=ref.resource_type,
                resource_id=ref.resource_id,
                version=ref.version,
                fingerprint=ref.fingerprint,
                observed_at=ref.observed_at,
            )
        )


def _fallback_response(
    *,
    code: str,
    stop_reason: str,
    model_refs: tuple[str, ...],
    call_ids: tuple[str, ...],
) -> PublishedResponse:
    return PublishedResponse(
        output=OrchestratorOutputV2(
            status=OrchestratorStatus.FAILED,
            blocks=(
                SafeErrorResponseBlock(
                    code=code,
                    message_key=f"assistant.errors.{code.lower()}",
                    manual_fallback="WORK_VIEW",
                ),
            ),
            stop_reason=stop_reason,
            model_refs=model_refs,
            evidence_call_ids=call_ids,
        )
    )


def _wait_result(
    *,
    wait_ref: WaitReference,
    log_cursor: int,
    model_refs: tuple[str, ...],
    call_ids: tuple[str, ...],
    locale: str,
) -> OrchestratorExecutionResult:
    if wait_ref.kind == "WORKFLOW_RESULT":
        return SuspendedWorkflow(
            wait_ref=wait_ref,
            public_activity=(
                ActivityResponseBlock(
                    label_key="assistant.activity.workflowRunning",
                    status="RUNNING",
                ),
            ),
            log_cursor=log_cursor,
        )
    if wait_ref.kind == "HUMAN_DECISION":
        text = (
            "Đề xuất đã sẵn sàng và đang chờ phê duyệt của Manager."
            if locale == "vi"
            else "A proposal is ready and awaiting Manager approval."
        )
        return PublishedResponse(
            output=OrchestratorOutputV2(
                status=OrchestratorStatus.AWAITING_HUMAN,
                blocks=(TextResponseBlock(text=text),),
                stop_reason="AWAITING_HUMAN_DECISION",
                model_refs=model_refs,
                evidence_call_ids=call_ids,
            ),
            wait_ref=wait_ref,
        )
    question = (
        "Cần thêm thông tin từ bạn để tiếp tục."
        if locale == "vi"
        else "More information is needed from you to continue."
    )
    return PublishedResponse(
        output=OrchestratorOutputV2(
            status=OrchestratorStatus.AWAITING_INPUT,
            blocks=(QuestionResponseBlock(question=question, response_context={}),),
            stop_reason="AWAITING_USER_INPUT",
            model_refs=model_refs,
            evidence_call_ids=call_ids,
        ),
        wait_ref=wait_ref,
    )


class ToolLoopOrchestrator:
    """Observation-driven native tool loop bound by registry, pipeline, history and budget."""

    def __init__(
        self,
        *,
        gateway: NativeToolGateway,
        catalog: ToolCatalog,
        pipeline: ToolPipeline,
        store: SessionLogPort,
        history: SessionHistory,
        access: ContextAccessPort,
        budget: RunBudgetPort,
        model_ref: str,
        skill_loader: SkillLoader | None = None,
        projector: ExplorationProjector | None = None,
        observation_sink: dict[UUID, ContextReadObservation] | None = None,
        driver_version: str = "tool_loop.v2",
        history_limits: HistoryLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        delegate_capabilities: Callable[[RunScope], Iterable[tuple[AgentId, str]]] | None = None,
    ) -> None:
        self.gateway = gateway
        self.catalog = catalog
        self.pipeline = pipeline
        self.store = store
        self.history = history
        self.access = access
        self.budget = budget
        self.model_ref = model_ref
        self.skill_loader = skill_loader
        self.projector = projector
        self.observation_sink = observation_sink
        self.driver_version = driver_version
        self.history_limits = history_limits or HistoryLimits()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.delegate_capabilities = delegate_capabilities
        self.limits: ToolLoopLimits = catalog.agent.manifest.runtime.limits

    async def run_segment(
        self,
        value: OrchestratorInput,
        *,
        scope: RunScope,
        checkpoint: ToolLoopCheckpoint | None,
        lease: RunLease,
    ) -> OrchestratorExecutionResult:
        """Execute one bounded Orchestrator segment from fresh turn or durable checkpoint."""

        if (
            lease.scope != scope
            or value.conversation_id != scope.conversation_id
            or value.turn_id != scope.turn_id
            or value.actor.organization_id != scope.organization_id
            or value.actor.membership_id != scope.membership_id
            or (
                value.orchestration_run_id is not None
                and value.orchestration_run_id != scope.orchestration_run_id
            )
        ):
            raise ValueError("ORCHESTRATOR_SCOPE_MISMATCH")

        interval_id = uuid4()
        try:
            interval_receipt = await self.budget.reserve(
                scope,
                UsageReservation(operation_id=interval_id, kind="INTERVAL"),
                lease=lease,
            )
        except BudgetError:
            return _fallback_response(
                code="BUDGET_EXHAUSTED",
                stop_reason="BUDGET_EXHAUSTED",
                model_refs=(),
                call_ids=(),
            )

        binding = CandidateBudget(
            scope=scope,
            lease=lease,
            store=self.budget,
            interval_id=interval_id,
        )
        try:
            with candidate_budget(binding):
                run = await self._initialize_or_recover(
                    value=value,
                    scope=scope,
                    checkpoint=checkpoint,
                    lease=lease,
                    interval_id=interval_id,
                )
                try:
                    graph = ToolLoopGraph(_GraphHandlers(self, run))
                    return await graph.run(
                        ToolLoopGraphState(
                            step=0,
                            repair_attempts=0,
                            transient_retries=0,
                            repair_hint=None,
                            pending_call=None,
                            last_outcome=None,
                            candidate_final=None,
                            fallback_code=None,
                            fallback_stop_reason=None,
                            result=None,
                            route="prepare",
                        ),
                        recursion_limit=max(32, self.limits.root_model_requests * 6 + 16),
                    )
                finally:
                    run.close_admitted()
        finally:
            await self.budget.settle(scope, interval_receipt, None)

    async def _initialize_or_recover(
        self,
        *,
        value: OrchestratorInput,
        scope: RunScope,
        checkpoint: ToolLoopCheckpoint | None,
        lease: RunLease,
        interval_id: UUID,
    ) -> _SegmentRun:
        existing = checkpoint or await self.store.load_checkpoint(scope)
        if existing is None:
            admitted = self.catalog.prepare(scope, model_ref=self.model_ref)
            try:
                catalog_fp = admitted.fingerprint
            finally:
                admitted.close()
            objective_ref = await self.store.put_payload(
                scope,
                MessagePayload(role="user", text=value.message).model_dump(mode="json"),
                classification="RAW_CONTEXT",
                kind="OBJECTIVE",
                lease=lease,
            )
            initial_state = ToolLoopCheckpoint(
                scope=scope,
                driver_version=self.driver_version,
                prompt_version=PROMPT_VERSION,
                catalog_fingerprint=catalog_fp,
                agent_version=self.catalog.agent.manifest.agent.version,
                manifest_fingerprint=self.catalog.agent.fingerprint,
                model_ref=self.model_ref,
                tool_versions=self.catalog.agent.tool_pins,
                skill_versions=(),
                sequence=0,
                segment_id=interval_id,
                objective_ref=objective_ref,
                payload_refs=(objective_ref,),
                limits=self.limits,
            )
            await self.store.initialize(scope, initial_state)
            rec = await self.store.append(
                scope,
                EventDraft(
                    event_id=uuid5(scope.orchestration_run_id, f"turn.started:{scope.turn_id}"),
                    kind="turn.started",
                    payload_ref=objective_ref,
                ),
                expected_sequence=0,
                lease=lease,
            )
            seq = rec.sequence
            verified_ctx = _verified_context_summary(value)
            if verified_ctx is not None:
                ctx_ref = await self.store.put_payload(
                    scope,
                    ContextPayload(category="CONTEXT", content=verified_ctx).model_dump(
                        mode="json"
                    ),
                    classification="RAW_CONTEXT",
                    kind="CONTEXT",
                    lease=lease,
                )
                ctx_rec = await self.store.append(
                    scope,
                    EventDraft(
                        event_id=uuid5(scope.orchestration_run_id, f"context.selected:{seq}"),
                        kind="context.selected",
                        payload_ref=ctx_ref,
                    ),
                    expected_sequence=seq,
                    lease=lease,
                )
                seq = ctx_rec.sequence
                initial_state = initial_state.model_copy(
                    update={"sequence": seq, "payload_refs": (*initial_state.payload_refs, ctx_ref)}
                )
            else:
                initial_state = initial_state.model_copy(update={"sequence": seq})
            await self.store.checkpoint(scope, initial_state, lease=lease)
            return _SegmentRun(
                value=value,
                scope=scope,
                lease=lease,
                interval_id=interval_id,
                checkpoint=initial_state,
            )

        # Recover existing run: replay canonical completed observations (including overflow reads).
        state = existing.model_copy(update={"segment_id": interval_id})
        run = _SegmentRun(
            value=value,
            scope=scope,
            lease=lease,
            interval_id=interval_id,
            checkpoint=state,
        )
        records = await self.store.read(scope, after_sequence=0)
        # Revalidate and check expiry BEFORE reading any historical payload during recovery.
        completed_refs = tuple(
            record.settlement.payload_ref
            for record in records
            if record.kind == "tool.completed"
            and record.settlement is not None
            and record.settlement.payload_ref is not None
        )
        if completed_refs:
            try:
                verdict = await self.access.revalidate(scope, completed_refs)
            except Exception:
                run.recovery_error = "CONTEXT_UNAVAILABLE"
                return run
            if not verdict.allowed:
                run.recovery_error = "CONTEXT_REVOKED"
                return run
            if not verdict.fresh:
                run.recovery_error = "CONTEXT_STALE"
                return run
            now = self.clock()
            if any(ref.expires_at <= now for ref in completed_refs):
                run.recovery_error = "CONTEXT_EXPIRED"
                return run

        report = self.projector.empty() if self.projector is not None else None
        for record in records:
            if (
                record.kind == "tool.completed"
                and record.invocation is not None
                and record.settlement is not None
                and record.settlement.status == "COMPLETED"
            ):
                if record.invocation.call_id not in run.observed_call_ids:
                    run.observed_call_ids.append(record.invocation.call_id)
                if record.settlement.payload_ref is not None:
                    try:
                        raw = await self.store.get_payload(scope, record.settlement.payload_ref)
                    except ValueError as exc:
                        run.recovery_error = (
                            "CONTEXT_EXPIRED"
                            if str(exc) == "CONTEXT_EXPIRED"
                            else "HISTORY_INVALID"
                        )
                        return run
                    except Exception:
                        run.recovery_error = "CONTEXT_UNAVAILABLE"
                        return run
                    payload = ToolResultPayload.model_validate(raw)
                    run.observed_evidence.extend(payload.references)
                    if record.invocation.tool_reference in _CONTEXT_READ_TOOLS:
                        obs = ContextReadObservation.model_validate(payload.output)
                        if self.observation_sink is not None:
                            self.observation_sink[obs.invocation_id] = obs
                        if self.projector is not None and report is not None:
                            report = self.projector.apply(report, obs)
                        _collect_context_evidence(run, obs)
                    elif record.invocation.tool_reference == "delegate_agent@1":
                        delegated = DelegateAgentOutput.model_validate(payload.output)
                        run.observed_evidence.extend(delegated.evidence)
                        if delegated.interaction_outcome is InteractionOutcome.DRAFT_READY or (
                            delegated.draft_id is not None and delegated.draft_version is not None
                        ):
                            run.draft_available = True
                        if (
                            delegated.proposal_id is not None
                            and delegated.proposal_version is not None
                        ):
                            run.proposal_pending = True
        return run


class _GraphHandlers:
    def __init__(self, orchestrator: ToolLoopOrchestrator, run: _SegmentRun) -> None:
        self._orch = orchestrator
        self._run = run

    async def prepare_step(self, state: ToolLoopGraphState) -> dict[str, object]:
        self._run.close_admitted()
        if self._run.recovery_error is not None:
            return {
                "fallback_code": self._run.recovery_error,
                "fallback_stop_reason": self._run.recovery_error,
                "route": "fallback",
            }
        try:
            admitted = self._orch.catalog.prepare(self._run.scope, model_ref=self._orch.model_ref)
        except ToolCatalogError as exc:
            return {
                "fallback_code": exc.code,
                "fallback_stop_reason": exc.code,
                "route": "fallback",
            }
        self._run.admitted = admitted

        # Build step tool schemas from real registry declarations; filter delegate_agent.
        schemas: list[ToolSchema] = []
        for schema in admitted.schemas:
            if (
                schema.reference == "delegate_agent@1"
                and self._orch.delegate_capabilities is not None
            ):
                allowed = tuple(self._orch.delegate_capabilities(self._run.scope))
                if not allowed:
                    continue
                schema = filtered_delegate_tool_schema(schema, allowed_capabilities=allowed)
            schemas.append(schema)
        self._run.step_tools = tuple(schemas)

        # Canonical Skill metadata comes strictly from pinned descriptors (0 body reads).
        skill_catalog = ()
        if self._orch.skill_loader is not None:
            try:
                skill_catalog = self._orch.skill_loader.catalog(self._run.scope)
            except SkillLoadingError as exc:
                return {
                    "fallback_code": exc.code,
                    "fallback_stop_reason": exc.code,
                    "route": "fallback",
                }

        records = await self._orch.store.read(self._run.scope, after_sequence=0)
        seq = records[-1].sequence if records else 0
        self._run.checkpoint = self._run.checkpoint.model_copy(
            update={"sequence": seq, "catalog_fingerprint": admitted.fingerprint}
        )
        projected = await self._orch.history.project(
            self._run.scope,
            records,
            access=self._orch.access,
            limits=self._orch.history_limits,
            checkpoint=self._run.checkpoint,
            skill_access=self._orch.skill_loader,
        )
        if projected.status == "WAITING" and projected.pending_wait is not None:
            return {
                "result": _wait_result(
                    wait_ref=projected.pending_wait,
                    log_cursor=seq,
                    model_refs=tuple(self._run.model_refs),
                    call_ids=tuple(self._run.observed_call_ids),
                    locale=self._run.value.locale,
                ),
                "route": "done",
            }
        if projected.status == "RECONCILE" and projected.pending_call is not None:
            receipt = await self._orch.store.get_invocation(
                self._run.scope, projected.pending_call.invocation_id
            )
            if receipt is None:
                return {
                    "fallback_code": "TOOL_RECONCILIATION_REQUIRED",
                    "fallback_stop_reason": "TOOL_RECONCILIATION_REQUIRED",
                    "route": "fallback",
                }
            if receipt.arguments_ref.expires_at <= self._orch.clock():
                return {
                    "fallback_code": "CONTEXT_EXPIRED",
                    "fallback_stop_reason": "CONTEXT_EXPIRED",
                    "route": "fallback",
                }
            try:
                raw_args = await self._orch.store.get_payload(
                    self._run.scope, receipt.arguments_ref
                )
                args_payload = ToolArgumentsPayload.model_validate(raw_args)
            except ValueError as exc:
                code = (
                    "CONTEXT_EXPIRED"
                    if str(exc) == "CONTEXT_EXPIRED"
                    else "TOOL_RECONCILIATION_REQUIRED"
                )
                return {
                    "fallback_code": code,
                    "fallback_stop_reason": code,
                    "route": "fallback",
                }
            except Exception:
                return {
                    "fallback_code": "TOOL_RECONCILIATION_REQUIRED",
                    "fallback_stop_reason": "TOOL_RECONCILIATION_REQUIRED",
                    "route": "fallback",
                }
            wire_name = projected.pending_call.tool_reference.rsplit("@", 1)[0].replace(".", "__")
            return {
                "pending_call": ToolCall(
                    call_id=projected.pending_call.call_id,
                    name=wire_name,
                    arguments=args_payload.arguments,
                ),
                "route": "execute_tool",
            }
        if projected.status != "READY":
            return {
                "fallback_code": projected.status,
                "fallback_stop_reason": projected.status,
                "route": "fallback",
            }

        self._run.step_items = (
            build_tool_loop_prompt(
                locale=self._run.value.locale,
                skills=skill_catalog,
                repair_hint=state["repair_hint"],
            ),
            *projected.items,
        )
        return {"route": "model"}

    async def call_model(self, state: ToolLoopGraphState) -> dict[str, object]:
        rem = await self._orch.budget.remaining(self._run.scope)
        if rem.available.root_model_requests <= 0 or rem.available.total_model_requests <= 0:
            return {
                "fallback_code": "BUDGET_EXHAUSTED",
                "fallback_stop_reason": "BUDGET_EXHAUSTED",
                "route": "fallback",
            }
        step = state["step"] + 1
        req = ToolModelRequest[ModelFinalResponse](
            invocation_key=f"orchestrator.tool_loop:{self._run.scope.orchestration_run_id}",
            items=self._run.step_items,
            tools=self._run.step_tools,
            response_schema=ModelFinalResponse,
            timeout_seconds=min(60.0, float(self._orch.limits.segment_seconds)),
            max_output_tokens=min(4000, self._orch.limits.output_tokens),
        )
        manual_receipt = None
        if not isinstance(self._orch.gateway, BudgetedToolLoopGateway):
            try:
                manual_receipt = await self._orch.budget.reserve(
                    self._run.scope,
                    UsageReservation(
                        kind="MODEL",
                        interval_id=self._run.interval_id,
                        input_tokens=1,
                        output_tokens=req.max_output_tokens,
                    ),
                    lease=self._run.lease,
                )
            except BudgetError:
                return {
                    "fallback_code": "BUDGET_EXHAUSTED",
                    "fallback_stop_reason": "BUDGET_EXHAUSTED",
                    "route": "fallback",
                }
        try:
            response = await self._orch.gateway.generate_step(req)
        except ModelInvalidOutputError:
            if manual_receipt is not None:
                await self._orch.budget.settle(self._run.scope, manual_receipt, None)
            if state["repair_attempts"] < self._orch.limits.repair_per_response:
                return {
                    "step": step,
                    "repair_attempts": state["repair_attempts"] + 1,
                    "repair_hint": (
                        "Return either exactly one valid permitted tool call or a valid "
                        "final structured response matching the schema."
                    ),
                    "route": "prepare",
                }
            return {
                "step": step,
                "fallback_code": "MODEL_INVALID_OUTPUT",
                "fallback_stop_reason": "MODEL_INVALID_OUTPUT",
                "route": "fallback",
            }
        except (ModelTimeoutError, ModelUnavailableError):
            if manual_receipt is not None:
                await self._orch.budget.settle(self._run.scope, manual_receipt, None)
            if state["transient_retries"] < self._orch.limits.retry_per_invocation:
                return {
                    "step": step,
                    "transient_retries": state["transient_retries"] + 1,
                    "route": "prepare",
                }
            return {
                "step": step,
                "fallback_code": "MODEL_UNAVAILABLE",
                "fallback_stop_reason": "MODEL_UNAVAILABLE",
                "route": "fallback",
            }
        except BudgetError:
            if manual_receipt is not None:
                await self._orch.budget.settle(self._run.scope, manual_receipt, None)
            return {
                "step": step,
                "fallback_code": "BUDGET_EXHAUSTED",
                "fallback_stop_reason": "BUDGET_EXHAUSTED",
                "route": "fallback",
            }

        if manual_receipt is not None:
            await self._orch.budget.settle(self._run.scope, manual_receipt, response.usage)
        if response.model_ref not in self._run.model_refs:
            self._run.model_refs.append(response.model_ref)

        # Commentary alongside a tool call is never final output.
        if response.tool_call is not None:
            return {
                "step": step,
                "repair_attempts": 0,
                "transient_retries": 0,
                "repair_hint": None,
                "pending_call": response.tool_call,
                "route": "execute_tool",
            }
        assert response.final is not None
        return {
            "step": step,
            "transient_retries": 0,
            "repair_hint": None,
            "candidate_final": response.final,
            "route": "verify_final",
        }

    async def execute_tool(self, state: ToolLoopGraphState) -> dict[str, object]:
        call = state["pending_call"]
        admitted = self._run.admitted
        if call is None or admitted is None:
            return {
                "fallback_code": "TOOL_RUNTIME_UNAVAILABLE",
                "fallback_stop_reason": "TOOL_RUNTIME_UNAVAILABLE",
                "route": "fallback",
            }
        # Enforce filtered step-level tool contract (e.g. excluded delegate_agent branches).
        step_schema = next((s for s in self._run.step_tools if s.wire_name == call.name), None)
        if step_schema is None:
            self._run.close_admitted()
            return {
                "pending_call": None,
                "fallback_code": "UNKNOWN_TOOL",
                "fallback_stop_reason": "UNKNOWN_TOOL",
                "route": "fallback",
            }
        if step_schema.input_contract is not None:
            try:
                step_schema.input_contract.model_validate_json(
                    json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False),
                    strict=True,
                )
            except Exception:
                self._run.close_admitted()
                return {
                    "pending_call": None,
                    "fallback_code": "TOOL_ARGUMENTS_INVALID",
                    "fallback_stop_reason": "TOOL_ARGUMENTS_INVALID",
                    "route": "fallback",
                }
        try:
            outcome = await self._orch.pipeline.execute(
                self._run.scope, call, admitted=admitted, lease=self._run.lease
            )
        finally:
            self._run.close_admitted()
        return {
            "pending_call": None,
            "last_outcome": outcome,
            "route": "observe_result",
        }

    async def observe_result(self, state: ToolLoopGraphState) -> dict[str, object]:
        outcome = state["last_outcome"]
        if outcome is None:
            return {
                "fallback_code": "TOOL_RUNTIME_UNAVAILABLE",
                "fallback_stop_reason": "TOOL_RUNTIME_UNAVAILABLE",
                "route": "fallback",
            }
        self._run.last_invocation_id = outcome.invocation_id
        records = await self._orch.store.read(self._run.scope, after_sequence=0)
        seq = records[-1].sequence if records else 0

        if isinstance(outcome, DeferredToolResult):
            updated = self._run.checkpoint.model_copy(
                update={"sequence": seq, "pending_wait": outcome.wait_ref}
            )
            self._run.checkpoint = updated
            await self._orch.store.checkpoint(self._run.scope, updated, lease=self._run.lease)
            return {
                "result": _wait_result(
                    wait_ref=outcome.wait_ref,
                    log_cursor=seq,
                    model_refs=tuple(self._run.model_refs),
                    call_ids=tuple(self._run.observed_call_ids),
                    locale=self._run.value.locale,
                ),
                "route": "done",
            }

        if isinstance(outcome, RejectedToolResult):
            if outcome.stop_segment:
                return {
                    "fallback_code": outcome.safe_code,
                    "fallback_stop_reason": outcome.safe_code,
                    "route": "fallback",
                }
            updated = self._run.checkpoint.model_copy(update={"sequence": seq})
            self._run.checkpoint = updated
            await self._orch.store.checkpoint(self._run.scope, updated, lease=self._run.lease)
            return {"last_outcome": None, "route": "prepare"}

        assert isinstance(outcome, CompletedToolResult)
        if outcome.call_id not in self._run.observed_call_ids:
            self._run.observed_call_ids.append(outcome.call_id)
        self._run.observed_evidence.extend(outcome.evidence)

        raw = await self._orch.store.get_payload(self._run.scope, outcome.payload_ref)
        payload = ToolResultPayload.model_validate(raw)
        self._run.observed_evidence.extend(payload.references)

        # Update loaded skill pins when load_skill completes.
        skill_pins = list(self._run.checkpoint.skill_versions)
        if payload.loaded_skill is not None and payload.loaded_skill not in skill_pins:
            skill_pins.append(payload.loaded_skill)

        # Inspect invocation metadata for context projection, artifact tracking & no-progress.
        receipt = await self._orch.store.get_invocation(self._run.scope, outcome.invocation_id)
        tool_ref = receipt.descriptor.tool_reference if receipt is not None else ""
        args_fp = receipt.descriptor.arguments_fingerprint if receipt is not None else ""

        if tool_ref in _CONTEXT_READ_TOOLS:
            obs = ContextReadObservation.model_validate(payload.output)
            _collect_context_evidence(self._run, obs)
            if self._orch.observation_sink is not None:
                self._orch.observation_sink[obs.invocation_id] = obs
            if self._orch.projector is not None:
                head = self._orch.projector.current or self._orch.projector.empty()
                self._orch.projector.apply(head, obs)
        elif tool_ref == "delegate_agent@1":
            delegated = DelegateAgentOutput.model_validate(payload.output)
            self._run.observed_evidence.extend(delegated.evidence)
            if delegated.interaction_outcome is InteractionOutcome.DRAFT_READY or (
                delegated.draft_id is not None and delegated.draft_version is not None
            ):
                self._run.draft_available = True
            if delegated.proposal_id is not None and delegated.proposal_version is not None:
                self._run.proposal_pending = True

        # No-progress guard: 2 identical (tool, args, resource_versions, semantic_result) calls.
        versions_fp = _digest(
            [(str(r.resource_id), r.version, r.fingerprint) for r in outcome.evidence]
        )
        result_fp = _digest(_strip_ephemeral(payload.output))
        signature = (tool_ref, args_fp, versions_fp, result_fp)
        if self._run.last_progress_signature == signature:
            return {
                "fallback_code": "NO_PROGRESS_LIMIT",
                "fallback_stop_reason": "NO_PROGRESS_LIMIT",
                "route": "fallback",
            }
        self._run.last_progress_signature = signature

        obs_event_id = uuid5(outcome.invocation_id, "observed")
        if any(r.event_id == obs_event_id for r in records):
            next_seq = seq
        else:
            obs_rec = await self._orch.store.append(
                self._run.scope,
                EventDraft(event_id=obs_event_id, kind="tool.observed"),
                expected_sequence=seq,
                lease=self._run.lease,
            )
            next_seq = obs_rec.sequence
        updated = self._run.checkpoint.model_copy(
            update={
                "sequence": next_seq,
                "skill_versions": tuple(skill_pins),
                "payload_refs": (*self._run.checkpoint.payload_refs, outcome.payload_ref),
            }
        )
        self._run.checkpoint = updated
        await self._orch.store.checkpoint(self._run.scope, updated, lease=self._run.lease)
        return {"last_outcome": None, "route": "prepare"}

    async def verify_final(self, state: ToolLoopGraphState) -> dict[str, object]:
        candidate = state["candidate_final"]
        if candidate is None:
            return {
                "fallback_code": "FINAL_VERIFICATION_FAILED",
                "fallback_stop_reason": "FINAL_VERIFICATION_FAILED",
                "route": "fallback",
            }
        artifacts = ObservedArtifactState(
            draft_available=self._run.draft_available,
            proposal_pending=self._run.proposal_pending,
            applied_write=self._run.applied_write,
            verified_blocks=tuple(self._run.verified_blocks),
        )
        try:
            published = verify_final_candidate(
                candidate,
                scope=self._run.scope,
                observed_evidence=tuple(self._run.observed_evidence),
                observed_call_ids=tuple(self._run.observed_call_ids),
                observed_artifacts=artifacts,
                model_refs=tuple(self._run.model_refs),
                source_invocation_id=self._run.last_invocation_id,
            )
        except FinalVerificationError as exc:
            if state["repair_attempts"] < self._orch.limits.repair_per_response:
                return {
                    "repair_attempts": state["repair_attempts"] + 1,
                    "candidate_final": None,
                    "repair_hint": (
                        f"Final output rejected ({exc.code}). Cite only observed evidence "
                        "and match artifact_state to verified tool outcomes."
                    ),
                    "route": "prepare",
                }
            return {
                "fallback_code": exc.code,
                "fallback_stop_reason": "FINAL_VERIFICATION_FAILED",
                "route": "fallback",
            }

        records = await self._orch.store.read(self._run.scope, after_sequence=0)
        seq = records[-1].sequence if records else 0
        msg_ref = await self._orch.store.put_payload(
            self._run.scope,
            MessagePayload(role="assistant", text=candidate.summary).model_dump(mode="json"),
            classification="RAW_CONTEXT",
            kind="MODEL_OUTPUT",
            lease=self._run.lease,
        )
        done_rec = await self._orch.store.append(
            self._run.scope,
            EventDraft(
                event_id=uuid5(self._run.scope.orchestration_run_id, f"turn.completed:{seq}"),
                kind="turn.completed",
                payload_ref=msg_ref,
            ),
            expected_sequence=seq,
            lease=self._run.lease,
        )
        updated = self._run.checkpoint.model_copy(
            update={
                "sequence": done_rec.sequence,
                "pending_wait": published.wait_ref,
                "payload_refs": (*self._run.checkpoint.payload_refs, msg_ref),
            }
        )
        self._run.checkpoint = updated
        await self._orch.store.checkpoint(self._run.scope, updated, lease=self._run.lease)
        return {"result": published, "route": "done"}

    async def fallback(self, state: ToolLoopGraphState) -> dict[str, object]:
        self._run.close_admitted()
        code = state["fallback_code"] or "ORCHESTRATOR_MANUAL_FALLBACK"
        stop = state["fallback_stop_reason"] or code
        return {
            "result": _fallback_response(
                code=code,
                stop_reason=stop,
                model_refs=tuple(self._run.model_refs),
                call_ids=tuple(self._run.observed_call_ids),
            ),
            "route": "done",
        }
