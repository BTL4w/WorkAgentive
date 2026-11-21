"""Backend delegation adapter with execution fencing and canonical workflow verification."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.database_models import (
    AssistantMessageModel,
    AssistantTurnModel,
    OrchestrationRunModel,
)
from app.modules.assistant.adapters.execution_recorder import (
    GOAL_HANDOFF_CAPABILITY,
    PostgreSQLExecutionRecorder,
)
from app.modules.assistant.adapters.orchestration_tools import ApplicationToolBackend
from app.modules.audit.adapters.database_models import AuditEventModel
from app.modules.audit.domain.events import AuditOutcome
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.planning_runs.adapters.database_models import (
    PlanningDraftModel,
    ProposalModel,
    WorkflowRunModel,
)
from app.modules.progress.adapters.assessment_repository import SqlAlchemyAssessmentRepository
from app.modules.progress.domain.daily_updates import DailyUpdateDraft, DailyUpdateError
from app.modules.progress.domain.evidence_support import AssessmentJobRef
from app.modules.work.adapters.database_models import IdempotencyRecordModel, IdempotencyState
from work_management_ai.agents.daily_update.contracts import DailyUpdateResult
from work_management_ai.runtime.contracts import (
    AgentHandoffV2,
    AgentResult,
    GoalDirectedAgentResult,
)
from work_management_ai.runtime.delegation import (
    CanonicalWorkflowState,
    DelegationAdapter,
    DelegationToolBackend,
)
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    RunScope,
    ToolInvocationContext,
    VersionPin,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError

_ALLOWED_WORKFLOW_STATUSES = {
    "QUEUED",
    "RUNNING",
    "NEEDS_INPUT",
    "DRAFT_AVAILABLE",
    "WAITING_FOR_DECISION",
    "COMPLETED",
    "FAILED",
}


class _ScopedReceiptReader:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        actor: AuthenticatedActor,
    ) -> None:
        self._sessions = session_factory
        self._actor = actor

    def _matches_actor(self, scope: RunScope) -> bool:
        return (scope.organization_id, scope.membership_id) == (
            self._actor.organization_id,
            self._actor.membership_id,
        )

    @asynccontextmanager
    async def _read(self) -> AsyncGenerator[AsyncSession]:
        async with self._sessions() as session, session.begin():
            await session.execute(text("SET LOCAL ROLE app_runtime"))
            await session.execute(
                text(
                    "SELECT set_config('app.organization_id', :org, true), "
                    "set_config('app.membership_id', :member, true)"
                ),
                {"org": str(self._actor.organization_id), "member": str(self._actor.membership_id)},
            )
            yield session


class PostgreSQLGoalSourceTurnReader(_ScopedReceiptReader):
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        actor: AuthenticatedActor,
        scope: RunScope,
    ) -> None:
        super().__init__(session_factory=session_factory, actor=actor)
        if not self._matches_actor(scope):
            raise ValueError("DELEGATION_CONTEXT_DENIED")
        self._scope = scope

    async def read(self, handoff: AgentHandoffV2) -> str:
        scope = self._scope
        if (
            handoff.actor.organization_id != scope.organization_id
            or handoff.actor.membership_id != scope.membership_id
            or handoff.orchestration_run_id != scope.orchestration_run_id
            or handoff.source_turn_ref != scope.turn_id
        ):
            raise ValueError("DELEGATION_CONTEXT_DENIED")
        async with self._read() as session:
            content = await session.scalar(
                select(AssistantMessageModel.content_blocks)
                .join(
                    AssistantTurnModel,
                    (AssistantTurnModel.organization_id == AssistantMessageModel.organization_id)
                    & (AssistantTurnModel.user_message_id == AssistantMessageModel.id),
                )
                .join(
                    OrchestrationRunModel,
                    (OrchestrationRunModel.organization_id == AssistantTurnModel.organization_id)
                    & (OrchestrationRunModel.turn_id == AssistantTurnModel.id),
                )
                .where(
                    AssistantTurnModel.organization_id == scope.organization_id,
                    AssistantTurnModel.id == scope.turn_id,
                    AssistantTurnModel.actor_membership_id == scope.membership_id,
                    AssistantTurnModel.conversation_id == scope.conversation_id,
                    AssistantMessageModel.conversation_id == scope.conversation_id,
                    AssistantMessageModel.created_by_membership_id == scope.membership_id,
                    AssistantMessageModel.role == "USER",
                    AssistantMessageModel.created_at > func.now() - text("interval '30 days'"),
                    OrchestrationRunModel.id == scope.orchestration_run_id,
                )
            )
        if content is None or len(content) != 1 or content[0].get("kind") != "text":
            raise ValueError("DELEGATION_CONTEXT_UNAVAILABLE")
        original = content[0].get("text")
        if not isinstance(original, str) or not original:
            raise ValueError("DELEGATION_CONTEXT_UNAVAILABLE")
        return original


class GoalSpecialistHarness(Protocol):
    async def run(self, handoff: AgentHandoffV2) -> AgentResult: ...


class PostgreSQLGoalSpecialistRunner:
    def __init__(
        self,
        *,
        recorder: PostgreSQLExecutionRecorder,
        harness: GoalSpecialistHarness,
        pin: VersionPin,
    ) -> None:
        self._recorder = recorder
        self._harness = harness
        self._pin = pin

    async def run(self, handoff: AgentHandoffV2) -> tuple[UUID, GoalDirectedAgentResult]:
        recorded = await self._recorder.start_agent_run(handoff)
        if recorded.existing:
            recovered = await self.reconcile(handoff)
            if recovered is None:
                raise ToolPolicyError("DELEGATION_RECEIPT_REQUIRED", stop_segment=True)
            return recovered
        result = await self._harness.run(handoff)
        if not isinstance(result, GoalDirectedAgentResult):
            raise ToolPolicyError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        await self._recorder.finish_agent_run(recorded.id, result)
        return recorded.id, result

    async def reconcile(
        self, handoff: AgentHandoffV2
    ) -> tuple[UUID, GoalDirectedAgentResult] | None:
        recorded = await self._recorder.lookup_agent_run(handoff.idempotency_key)
        if recorded is None:
            return None
        if (
            recorded.target_agent_id is not handoff.target_agent_id
            or recorded.target_agent_version != handoff.target_agent_version
            or recorded.capability != GOAL_HANDOFF_CAPABILITY
            or recorded.manifest_fingerprint != self._pin.fingerprint
            or self._pin.reference != handoff.target_agent_id.value
            or self._pin.version != handoff.target_agent_version
        ):
            raise ToolPolicyError("DELEGATION_RECEIPT_MISMATCH", stop_segment=True)
        result = recorded.replayed_result
        return (recorded.id, result) if isinstance(result, GoalDirectedAgentResult) else None


class PostgreSQLDailyUpdateReceiptReader(_ScopedReceiptReader):
    """Read an exact draft from its application receipt after a partial child run."""

    async def recover_draft(
        self, scope: RunScope, *, receipt_id: UUID, task_id: UUID, task_version: int
    ) -> DailyUpdateResult | None:
        if not self._matches_actor(scope):
            return None
        keys = {
            str(uuid5(NAMESPACE_URL, f"daily:delegate:{receipt_id}:{step}")): step
            for step in ("DRAFT", "ASSESS")
        }
        async with self._read() as session:
            repository = SqlAlchemyAssessmentRepository(session, self._actor)
            try:
                await repository.authenticate()
                rows = await session.execute(
                    select(IdempotencyRecordModel).where(
                        IdempotencyRecordModel.organization_id == scope.organization_id,
                        IdempotencyRecordModel.actor_membership_id == scope.membership_id,
                        IdempotencyRecordModel.operation.in_(
                            ["daily_update.draft", "daily_update.revise", "daily_update.assess"]
                        ),
                        IdempotencyRecordModel.idempotency_key.in_(keys),
                        IdempotencyRecordModel.state == IdempotencyState.COMPLETED,
                        IdempotencyRecordModel.expires_at > func.now(),
                    )
                )
                bodies: dict[str, object] = {}
                for saved in rows.scalars():
                    step = keys[saved.idempotency_key]
                    if step in bodies or (saved.operation == "daily_update.assess") != (
                        step == "ASSESS"
                    ):
                        return None
                    bodies[step] = saved.response_body
                if "DRAFT" not in bodies or "ASSESS" not in bodies:
                    return None
                receipt = DailyUpdateDraft.model_validate(bodies["DRAFT"])
                assessment = AssessmentJobRef.model_validate(bodies["ASSESS"])
                draft = await repository.draft(receipt.id)
                current = await repository.context(
                    task_id, draft.reporting_timezone, datetime.now(UTC)
                )
                row = await repository.latest_row(draft)
                binding, _ = await repository.binding_for(draft)
            except DailyUpdateError:
                return None
            if (
                draft.version != receipt.version
                or draft.confirmed_update_id is not None
                or len(draft.items) != 1
                or draft.items[0].task_id != task_id
                or draft.items[0].expected_task_version != task_version
                or current.task_version != task_version
                or current.progress_version != draft.items[0].expected_progress_version
                or assessment.draft_id != draft.id
                or row is None
                or row.id != assessment.id
                or row.state == "PENDING"
                or row.binding != binding
            ):
                return None
            return DailyUpdateResult(
                draft_id=draft.id,
                draft_version=draft.version,
                task_id=task_id,
                task_version=task_version,
                assessment_id=assessment.id,
            )


class PostgreSQLWorkflowWaitVerifier(_ScopedReceiptReader):
    """Verify async workflow wait state from canonical workflow_runs rows under RLS."""

    async def find_committed_planning_create(
        self, scope: RunScope, *, child_run_id: UUID, receipt_id: UUID, call_id: str
    ) -> UUID | None:
        if not self._matches_actor(scope):
            return None
        # The audit row and workflow are committed in the same application
        # transaction, before the separate child-result transaction.
        async with self._read() as session:
            return (
                await session.execute(
                    select(AuditEventModel.resource_id)
                    .where(
                        AuditEventModel.organization_id == scope.organization_id,
                        AuditEventModel.actor_membership_id == scope.membership_id,
                        AuditEventModel.outcome == AuditOutcome.SUCCEEDED,
                        AuditEventModel.action == "planning_run.created",
                        AuditEventModel.resource_type == "workflow_run",
                        AuditEventModel.request_id.in_(
                            (
                                f"assistant-tool:{child_run_id}:delegate:{receipt_id}:manage:1",
                                f"assistant-tool:{child_run_id}:delegate:{call_id}:manage:1",
                            )
                        ),
                        AuditEventModel.idempotency_key
                        == f"assistant:{scope.turn_id}:planning:create",
                    )
                    .distinct()
                )
            ).scalar_one_or_none()

    async def verify_workflow(
        self, scope: RunScope, workflow_run_id: UUID
    ) -> CanonicalWorkflowState | None:
        if not self._matches_actor(scope):
            return None
        async with self._read() as session:
            row = await session.scalar(
                select(WorkflowRunModel).where(
                    WorkflowRunModel.organization_id == scope.organization_id,
                    WorkflowRunModel.id == workflow_run_id,
                    WorkflowRunModel.requested_by_membership_id == scope.membership_id,
                )
            )
            if row is None or row.status not in _ALLOWED_WORKFLOW_STATUSES:
                return None
            draft = None
            proposal = None
            if row.status == "DRAFT_AVAILABLE":
                draft = (
                    await session.execute(
                        select(
                            PlanningDraftModel.id, PlanningDraftModel.current_version_number
                        ).where(
                            PlanningDraftModel.organization_id == scope.organization_id,
                            PlanningDraftModel.workflow_run_id == workflow_run_id,
                            PlanningDraftModel.created_by_membership_id == scope.membership_id,
                        )
                    )
                ).one_or_none()
            if row.status == "WAITING_FOR_DECISION":
                proposal = (
                    await session.execute(
                        select(ProposalModel.id, ProposalModel.current_version_number).where(
                            ProposalModel.organization_id == scope.organization_id,
                            ProposalModel.workflow_run_id == workflow_run_id,
                            ProposalModel.status == "READY_FOR_DECISION",
                            ProposalModel.approval_id.is_not(None),
                        )
                    )
                ).one_or_none()
            return CanonicalWorkflowState(
                workflow_run_id=row.id,
                status=cast(
                    Literal[
                        "QUEUED",
                        "RUNNING",
                        "NEEDS_INPUT",
                        "DRAFT_AVAILABLE",
                        "WAITING_FOR_DECISION",
                        "COMPLETED",
                        "FAILED",
                    ],
                    row.status,
                ),
                version=row.version,
                draft_id=draft[0] if draft is not None else None,
                draft_version=draft[1] if draft is not None else None,
                proposal_id=proposal[0] if proposal is not None else None,
                proposal_version=proposal[1] if proposal is not None else None,
            )


class BackendDelegationTool:
    """Fenced candidate tool backend bridging delegate_agent to Specialists and receipts."""

    def __init__(self, *, actor: AuthenticatedActor, adapter: DelegationAdapter) -> None:
        self._adapter = adapter
        inner = DelegationToolBackend(adapter)
        self._backend = ApplicationToolBackend(
            actor=actor,
            execute=inner.execute,
            reconcile=inner.reconcile,
        )

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        return await self._backend.execute(context, arguments)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        return await self._backend.reconcile(context, arguments)

    async def revalidate_wait(self, context: ToolInvocationContext, wait: WaitReference) -> None:
        await self._adapter.revalidate_wait(context, wait)
