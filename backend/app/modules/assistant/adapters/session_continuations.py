"""PostgreSQL continuation ownership, wait admission and canonical workflow routing."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Literal
from uuid import UUID, uuid5

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.context_continuations import resolve_input
from app.modules.assistant.adapters.continuation_models import OrchestrationWaitModel
from app.modules.assistant.adapters.database_models import (
    AgentRunModel,
    AssistantConversationModel,
    AssistantEventModel,
    AssistantJobModel,
    AssistantMessageModel,
    AssistantTurnModel,
    OrchestrationRunModel,
)
from app.modules.assistant.adapters.repository import agent_run_domain, orchestration_domain
from app.modules.assistant.adapters.session_models import OrchestrationRuntimeStateModel
from app.modules.assistant.adapters.session_repository import PostgreSQLSessionRepository
from app.modules.assistant.adapters.session_usage_models import OrchestrationChildJobModel
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.continuation_service import ContinuationService
from app.modules.assistant.application.ports import LinkedWorkflowEvent
from app.modules.assistant.application.session_service import SessionService
from app.modules.assistant.domain.continuation import (
    TransferReceipt,
    WaitDraft,
    WaitRecord,
    WakeDecision,
)
from app.modules.assistant.domain.models import OrchestrationRunStatus
from app.modules.assistant.domain.session import SessionError, SessionEvent
from app.modules.feedback.adapters.payload_repository import live_payload
from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthRepository
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.planning_runs.adapters.database_models import (
    ApprovalModel,
    ProposalModel,
    WorkflowEventModel,
    WorkflowJobModel,
    WorkflowRunModel,
)
from app.modules.planning_runs.domain.models import WorkflowEvent
from app.modules.reporting.adapters.usage_models import ReportGenerationJobModel
from work_management_ai.runtime.context_exploration import ContextUseAccessPort
from work_management_ai.runtime.tool_loop_contracts import (
    RunScope,
    ToolLoopCheckpoint,
    WaitReference,
)

_WORKFLOW_OUTCOMES = {
    "workflow.failed": "FAILED",
    "workflow.needs_input": "NEEDS_INPUT",
    "planning.draft.created": "DRAFT_AVAILABLE",
    "planning.draft.updated": "DRAFT_AVAILABLE",
    "planning.draft.conflicted": "DRAFT_AVAILABLE",
    "proposal.ready": "WAITING_FOR_DECISION",
    "workflow.completed": "COMPLETED",
}


def continuation_service(
    sessions: async_sessionmaker[AsyncSession],
    *,
    context_access: ContextUseAccessPort | None = None,
) -> ContinuationService:
    factory = PostgreSQLAssistantTransactionFactory(sessions)

    @asynccontextmanager
    async def transaction(actor: AuthenticatedActor) -> AsyncGenerator[PostgreSQLContinuations]:
        async with factory(actor) as tx:
            yield PostgreSQLContinuations(tx.session, context_access=context_access)
            await tx.commit()

    return ContinuationService(transaction)


def _record(row: OrchestrationWaitModel) -> WaitRecord:
    return WaitRecord(
        reference=WaitReference.model_validate(
            {
                "wait_id": row.wait_id,
                "scope": RunScope(
                    organization_id=row.organization_id,
                    membership_id=row.membership_id,
                    conversation_id=row.conversation_id,
                    turn_id=row.source_turn_id,
                    orchestration_run_id=row.source_run_id,
                ),
                "source_invocation_id": row.source_invocation_id,
                "kind": row.kind,
                "resource_id": row.resource_id,
                "resource_version": row.resource_version,
                "source_fingerprint": row.source_fingerprint,
                "policy_fingerprint": row.policy_fingerprint,
            }
        ),
        owner_run_id=row.owner_run_id,
        owner_turn_id=row.owner_turn_id,
        version=row.version,
    )


class PostgreSQLContinuations:
    def __init__(
        self, session: AsyncSession, *, context_access: ContextUseAccessPort | None = None
    ) -> None:
        self._session = session
        self._context_access = context_access
        self._journal = PostgreSQLSessionRepository(session)

    @asynccontextmanager
    async def _as_owner(
        self, organization_id: UUID, membership_id: UUID
    ) -> AsyncGenerator[AuthenticatedActor | None]:
        previous = await self._session.scalar(
            text("SELECT current_setting('app.membership_id', true)")
        )
        actor = await SqlAlchemyAuthRepository(self._session).find_current_actor_by_membership(
            organization_id=organization_id, membership_id=membership_id
        )
        if actor is None:
            yield None
            return
        await self._session.execute(
            text("SELECT set_config('app.membership_id', :member, true)"),
            {"member": str(membership_id)},
        )
        try:
            yield actor
        finally:
            await self._session.execute(
                text("SELECT set_config('app.membership_id', :member, true)"),
                {"member": previous or ""},
            )

    async def route_linked_workflow_event(
        self, item: LinkedWorkflowEvent, membership_id: UUID
    ) -> Literal["LEGACY", "HANDLED", "PENDING"]:
        """Keep the projection cursor until the exact child job can be reconciled."""
        async with self._as_owner(item.event.organization_id, membership_id) as actor:
            if actor is None:
                return "LEGACY"
            scope = session_scope(
                RunScope(
                    organization_id=actor.organization_id,
                    membership_id=membership_id,
                    conversation_id=item.conversation_id,
                    turn_id=item.turn_id,
                    orchestration_run_id=item.agent_run.orchestration_run_id,
                )
            )
            if not await self._journal.authorize(scope, actor):
                return "LEGACY"
            if await self._journal.state(scope) is None:
                return "LEGACY"
            if await self._planning_job_pending(item):
                return "PENDING"
            await self.receive_event(actor, item)
            return "HANDLED"

    async def _planning_job_pending(self, item: LinkedWorkflowEvent) -> bool:
        wait, child, job = OrchestrationWaitModel, OrchestrationChildJobModel, WorkflowJobModel
        return (
            await self._session.scalar(
                select(job.id)
                .join(
                    child,
                    (child.organization_id == job.organization_id)
                    & (child.workflow_job_id == job.id),
                )
                .join(
                    wait,
                    (wait.organization_id == child.organization_id)
                    & (wait.source_run_id == child.orchestration_run_id)
                    & (wait.child_allocation_id == child.allocation_id),
                )
                .where(
                    wait.organization_id == item.event.organization_id,
                    wait.source_run_id == item.agent_run.orchestration_run_id,
                    wait.source_turn_id == item.turn_id,
                    wait.resource_id == item.event.workflow_run_id,
                    wait.kind == "WORKFLOW_RESULT",
                    wait.closed.is_(False),
                    wait.wake_job_id.is_(None),
                    job.status.in_(("QUEUED", "RUNNING")),
                )
                .limit(1)
            )
            is not None
        )

    async def route_report_completion(self, job: ReportGenerationJobModel) -> None:
        async with self._as_owner(job.organization_id, job.requester_membership_id) as actor:
            if actor is not None:
                await self.receive_report_completion(actor, job.id)

    async def _wait(
        self, actor: AuthenticatedActor, wait_id: UUID, *, lock: bool = True
    ) -> OrchestrationWaitModel | None:
        row = OrchestrationWaitModel
        query = select(row).where(
            row.organization_id == actor.organization_id,
            row.membership_id == actor.membership_id,
            row.wait_id == wait_id,
        )
        return await self._session.scalar(query.with_for_update() if lock else query)

    async def _run(
        self, actor: AuthenticatedActor, run_id: UUID, conversation_id: UUID
    ) -> tuple[OrchestrationRunModel, AssistantTurnModel]:
        run, turn = OrchestrationRunModel, AssistantTurnModel
        row = (
            (
                await self._session.execute(
                    select(run, turn)
                    .join(
                        turn,
                        (turn.organization_id == run.organization_id) & (turn.id == run.turn_id),
                    )
                    .where(
                        run.organization_id == actor.organization_id,
                        run.id == run_id,
                        run.actor_membership_id == actor.membership_id,
                        turn.actor_membership_id == actor.membership_id,
                        turn.conversation_id == conversation_id,
                    )
                    .with_for_update(of=(run, turn))
                )
            )
            .tuples()
            .one_or_none()
        )
        if row is None:
            raise SessionError("CONTINUATION_NOT_FOUND")
        return row

    async def open_wait(self, actor: AuthenticatedActor, value: WaitDraft) -> WaitRecord:
        reference = value.reference
        scope = session_scope(reference.scope)
        if not await self._journal.authorize(scope, actor):
            raise SessionError("CONTINUATION_NOT_FOUND")
        run, turn = await self._run(actor, scope.orchestration_run_id, scope.conversation_id)
        previous = await self._wait(actor, reference.wait_id, lock=False)
        if previous is not None:
            if _record(previous).reference != reference:
                raise SessionError("CONTINUATION_ID_CONFLICT")
            return _record(previous)
        if run.status != OrchestrationRunStatus.RUNNING:
            raise SessionError("CONTINUATION_SOURCE_NOT_RUNNING")
        state = await SessionService.locked(self._journal, scope, session_lease(value.lease))
        if state.checkpoint_payload_id is None or state.checkpoint_sequence != state.sequence:
            raise SessionError("CONTINUATION_CHECKPOINT_REQUIRED")
        payload = await self._journal.payload(scope, state.checkpoint_payload_id)
        if payload is None or not await self._journal.payload_live(scope, payload.id):
            raise SessionError("CONTEXT_EXPIRED")
        checkpoint = ToolLoopCheckpoint.model_validate(payload.body)
        if checkpoint.pending_wait != reference:
            raise SessionError("CONTINUATION_WAIT_MISMATCH")
        if reference.kind != "USER_INPUT" and reference.resource_id is None:
            raise SessionError("CONTINUATION_RESOURCE_REQUIRED")

        job = await self._session.scalar(
            select(AssistantJobModel)
            .where(
                AssistantJobModel.organization_id == actor.organization_id,
                AssistantJobModel.id == value.lease.job_id,
                AssistantJobModel.orchestration_run_id == run.id,
                AssistantJobModel.requester_membership_id == actor.membership_id,
                AssistantJobModel.status == "RUNNING",
                AssistantJobModel.locked_by == value.lease.worker_id,
            )
            .with_for_update()
        )
        now = await self._journal.now()
        if job is None or job.lease_until is None or job.lease_until <= now:
            raise SessionError("LEASE_LOST")
        active = await self._open_owner(actor, run.id)
        if active is not None:
            admitted_followup = (
                job.job_type == "assistant.turn.execute"
                and active.source_run_id != run.id
                and active.kind in {"USER_INPUT", "HUMAN_DECISION"}
            )
            if active.wake_job_id != job.id and not admitted_followup:
                raise SessionError("CONTINUATION_ALREADY_WAITING")
            active.closed = True
            await self._session.flush()
        allocation = uuid5(reference.source_invocation_id, "child-budget")
        child = await self._session.get(
            OrchestrationChildJobModel, (scope.organization_id, run.id, allocation)
        )
        if reference.policy_fingerprint is not None or (
            reference.source_fingerprint is not None
            and (child is None or child.generation_job_id is None)
        ):
            raise SessionError("CONTINUATION_FINGERPRINT_UNSUPPORTED")
        if reference.source_fingerprint is not None and child is not None:
            generation = await self._session.scalar(
                select(ReportGenerationJobModel.snapshot_hash).where(
                    ReportGenerationJobModel.organization_id == scope.organization_id,
                    ReportGenerationJobModel.id == child.generation_job_id,
                    ReportGenerationJobModel.requester_membership_id == actor.membership_id,
                )
            )
            if generation != reference.source_fingerprint:
                raise SessionError("CONTINUATION_WAIT_MISMATCH")
        row = OrchestrationWaitModel(
            organization_id=actor.organization_id,
            wait_id=reference.wait_id,
            source_run_id=run.id,
            source_turn_id=turn.id,
            source_invocation_id=reference.source_invocation_id,
            child_allocation_id=child.allocation_id if child is not None else None,
            membership_id=actor.membership_id,
            conversation_id=scope.conversation_id,
            owner_run_id=run.id,
            owner_turn_id=turn.id,
            version=1,
            kind=reference.kind,
            resource_id=reference.resource_id,
            resource_version=reference.resource_version,
            source_fingerprint=reference.source_fingerprint,
            policy_fingerprint=reference.policy_fingerprint,
            closed=False,
        )
        self._session.add(row)
        if reference.kind != "WORKFLOW_RESULT":
            status = "AWAITING_INPUT" if reference.kind == "USER_INPUT" else "AWAITING_HUMAN"
            run.status = turn.status = status
        run.checkpoint = {**run.checkpoint, "continuation_protocol": "2.0"}
        run.updated_at = turn.updated_at = job.updated_at = now
        job.status = "COMPLETED"
        job.locked_by = None
        job.lease_until = None
        await self._journal.save_state(replace(state, lease=None))
        await self._activity(row, "assistant.run.waiting.v1", turn.id, run.id)
        await self._session.flush()
        if child is not None and child.generation_job_id is not None:
            await self.receive_report_completion(actor, child.generation_job_id)
        elif row.kind in {"WORKFLOW_RESULT", "HUMAN_DECISION"}:
            await self._catch_up_workflow(actor, row)
        return _record(row)

    async def _catch_up_workflow(
        self, actor: AuthenticatedActor, row: OrchestrationWaitModel
    ) -> None:
        """A committed outcome may predate wait registration."""
        workflow_id = row.resource_id
        if row.kind == "HUMAN_DECISION":
            workflow_id = await self._session.scalar(
                select(ProposalModel.workflow_run_id).where(
                    ProposalModel.organization_id == row.organization_id,
                    ProposalModel.id == row.resource_id,
                )
            )
        if workflow_id is None:
            return
        a, e = AgentRunModel, WorkflowEventModel
        candidates = (
            (
                await self._session.execute(
                    select(a, e)
                    .join(
                        e,
                        (e.organization_id == a.organization_id)
                        & (e.workflow_run_id == a.workflow_run_id),
                    )
                    .where(
                        a.organization_id == row.organization_id,
                        a.orchestration_run_id == row.source_run_id,
                        e.workflow_run_id == workflow_id,
                        e.event_type.in_(_WORKFLOW_OUTCOMES),
                        live_payload(e, "workflow_events"),
                    )
                    .order_by(e.sequence.desc(), e.id)
                    .limit(16)
                )
            )
            .tuples()
            .all()
        )
        for agent, event in candidates:
            linked = LinkedWorkflowEvent(
                agent_run=agent_run_domain(agent),
                turn_id=row.source_turn_id,
                conversation_id=row.conversation_id,
                event=WorkflowEvent(
                    id=event.id,
                    organization_id=event.organization_id,
                    workflow_run_id=event.workflow_run_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    public_payload=event.public_payload,
                    created_at=event.created_at,
                ),
            )
            if (await self.receive_event(actor, linked)).outcome != "IGNORED":
                return

    async def receive_report_completion(
        self, actor: AuthenticatedActor, generation_id: UUID
    ) -> WakeDecision:
        row, binding, job = (
            OrchestrationWaitModel,
            OrchestrationChildJobModel,
            ReportGenerationJobModel,
        )
        binding_row = await self._session.scalar(
            select(binding).where(
                binding.organization_id == actor.organization_id,
                binding.generation_job_id == generation_id,
            )
        )
        if binding_row is None:
            return WakeDecision("IGNORED")
        # Serialize registration and completion even when the wait row is not visible yet.
        state = await self._session.scalar(
            select(OrchestrationRuntimeStateModel)
            .where(
                OrchestrationRuntimeStateModel.organization_id == actor.organization_id,
                OrchestrationRuntimeStateModel.orchestration_run_id
                == binding_row.orchestration_run_id,
                OrchestrationRuntimeStateModel.membership_id == actor.membership_id,
            )
            .with_for_update()
        )
        if state is None:
            return WakeDecision("IGNORED")
        pair = (
            (
                await self._session.execute(
                    select(row, job)
                    .join(
                        binding,
                        (binding.organization_id == row.organization_id)
                        & (binding.orchestration_run_id == row.source_run_id)
                        & (binding.allocation_id == row.child_allocation_id),
                    )
                    .join(
                        job,
                        (job.organization_id == binding.organization_id)
                        & (job.id == binding.generation_job_id),
                    )
                    .where(
                        row.organization_id == actor.organization_id,
                        row.membership_id == actor.membership_id,
                        row.kind == "WORKFLOW_RESULT",
                        row.closed.is_(False),
                        row.resource_id == generation_id,
                        job.id == generation_id,
                        job.requester_membership_id == actor.membership_id,
                        job.state.in_(("AWAITING_REVIEW", "AI_UNAVAILABLE", "FAILED")),
                    )
                    .with_for_update(of=row)
                )
            )
            .tuples()
            .one_or_none()
        )
        if pair is None:
            return WakeDecision("IGNORED")
        wait, canonical_job = pair
        scope = session_scope(_record(wait).reference.scope)
        if not await self._journal.authorize(scope, actor):
            return WakeDecision("IGNORED")
        if (
            wait.resource_version is None
            or canonical_job.expected_report_version != wait.resource_version
            or (
                wait.source_fingerprint is not None
                and wait.source_fingerprint != canonical_job.snapshot_hash
            )
            or wait.policy_fingerprint is not None
        ):
            return WakeDecision("IGNORED")
        if wait.wake_job_id is not None:
            return WakeDecision("DUPLICATE", wait.wake_job_id)
        return await self._wake(wait, None, canonical_job.state)

    async def _open_owner(
        self, actor: AuthenticatedActor, run_id: UUID
    ) -> OrchestrationWaitModel | None:
        return await self._session.scalar(
            select(OrchestrationWaitModel)
            .where(
                OrchestrationWaitModel.organization_id == actor.organization_id,
                OrchestrationWaitModel.membership_id == actor.membership_id,
                OrchestrationWaitModel.owner_run_id == run_id,
                OrchestrationWaitModel.closed.is_(False),
            )
            .with_for_update()
        )

    async def transfer(
        self,
        actor: AuthenticatedActor,
        *,
        wait_id: UUID,
        expected_version: int,
        target_run_id: UUID,
    ) -> TransferReceipt:
        row = await self._wait(actor, wait_id)
        if row is None or not await self._journal.authorize(
            session_scope(_record(row).reference.scope), actor
        ):
            raise SessionError("CONTINUATION_NOT_FOUND")
        if (
            row.owner_run_id == target_run_id
            and row.version == expected_version + 1
            and row.source_run_id != target_run_id
        ):
            source = await self._session.scalar(
                select(AssistantEventModel.orchestration_run_id).where(
                    AssistantEventModel.organization_id == row.organization_id,
                    AssistantEventModel.conversation_id == row.conversation_id,
                    AssistantEventModel.id
                    == uuid5(row.wait_id, f"assistant.run.transferred.v1:{row.version}"),
                )
            )
            if source is not None:
                return TransferReceipt(_record(row), source)
        if row.version != expected_version or row.closed or row.wake_job_id is not None:
            raise SessionError("CONTINUATION_VERSION_CONFLICT")
        source, source_turn = await self._run(actor, row.owner_run_id, row.conversation_id)
        target, target_turn = await self._run(actor, target_run_id, row.conversation_id)
        if target.id == source.id or target.status != "QUEUED":
            raise SessionError("CONTINUATION_TARGET_INVALID")
        accepted = await self._assert_later_user_turn(source_turn, target_turn)
        if row.kind == "USER_INPUT":
            binding = await resolve_input(
                self._journal,
                _record(row).reference,
                user_turn_id=target_turn.id,
                content_blocks=accepted.content_blocks,
                access=self._context_access,
            )
            row.context_binding = binding.model_dump(mode="json") if binding is not None else None
        if await self._open_owner(actor, target.id) is not None:
            raise SessionError("CONTINUATION_TARGET_INVALID")
        target_job = await self._target_job(actor, target.id)
        now = await self._journal.now()
        transferred = orchestration_domain(source).mark_continuation_transferred(row.kind, now)
        source.status = transferred.status.value
        source.stop_reason = transferred.stop_reason
        source.completed_at = transferred.completed_at
        source.updated_at = source_turn.updated_at = now
        source_turn.status = "COMPLETED"
        if row.kind == "WORKFLOW_RESULT":
            target_job.status = "COMPLETED"
            target_job.updated_at = now
            target.status = target_turn.status = "RUNNING"
            target.started_at = target.updated_at = target_turn.updated_at = now
        row.owner_run_id, row.owner_turn_id = target.id, target_turn.id
        row.version += 1
        await self._activity(row, "assistant.run.transferred.v1", source_turn.id, source.id)
        return TransferReceipt(_record(row), source.id)

    async def _assert_later_user_turn(
        self, source: AssistantTurnModel, target: AssistantTurnModel
    ) -> AssistantMessageModel:
        message = AssistantMessageModel
        messages = {
            item.id: item
            for item in (
                await self._session.scalars(
                    select(message).where(
                        message.organization_id == target.organization_id,
                        message.conversation_id == target.conversation_id,
                        message.id.in_((source.user_message_id, target.user_message_id)),
                        message.role == "USER",
                        message.created_by_membership_id == target.actor_membership_id,
                    )
                )
            ).all()
        }
        previous, accepted = (
            messages.get(source.user_message_id),
            messages.get(target.user_message_id),
        )
        if previous is None or accepted is None or accepted.sequence <= previous.sequence:
            raise SessionError("CONTINUATION_TARGET_INVALID")
        return accepted

    async def _target_job(self, actor: AuthenticatedActor, run_id: UUID) -> AssistantJobModel:
        job = await self._session.scalar(
            select(AssistantJobModel)
            .where(
                AssistantJobModel.organization_id == actor.organization_id,
                AssistantJobModel.orchestration_run_id == run_id,
                AssistantJobModel.job_type == "assistant.turn.execute",
                AssistantJobModel.status == "QUEUED",
            )
            .with_for_update()
        )
        if job is None:
            raise SessionError("CONTINUATION_TARGET_INVALID")
        return job

    async def _activity(
        self, row: OrchestrationWaitModel, event_type: str, turn_id: UUID, run_id: UUID
    ) -> None:
        conversation = await self._session.scalar(
            select(AssistantConversationModel)
            .where(
                AssistantConversationModel.organization_id == row.organization_id,
                AssistantConversationModel.id == row.conversation_id,
                AssistantConversationModel.owner_membership_id == row.membership_id,
            )
            .with_for_update()
        )
        if conversation is None:
            raise SessionError("CONTINUATION_NOT_FOUND")
        now = await self._journal.now()
        conversation.last_event_sequence += 1
        conversation.version += 1
        conversation.updated_at = now
        self._session.add(
            AssistantEventModel(
                id=uuid5(row.wait_id, f"{event_type}:{row.version}"),
                organization_id=row.organization_id,
                conversation_id=row.conversation_id,
                sequence=conversation.last_event_sequence,
                event_type=event_type,
                public_payload={
                    "turn_id": str(turn_id),
                    "wait_kind": row.kind,
                    "owner_turn_id": str(row.owner_turn_id),
                },
                turn_id=turn_id,
                orchestration_run_id=run_id,
                dedupe_key=f"{event_type}:{row.wait_id}:{row.version}",
                occurred_at=now,
            )
        )

    async def receive_event(
        self, actor: AuthenticatedActor, event: LinkedWorkflowEvent
    ) -> WakeDecision:
        linked = event
        event_value = linked.event
        source_run_id = linked.agent_run.orchestration_run_id
        rows = (
            await self._session.scalars(
                select(OrchestrationWaitModel)
                .where(
                    OrchestrationWaitModel.organization_id == actor.organization_id,
                    OrchestrationWaitModel.membership_id == actor.membership_id,
                    OrchestrationWaitModel.source_run_id == source_run_id,
                    OrchestrationWaitModel.source_turn_id == linked.turn_id,
                    OrchestrationWaitModel.conversation_id == linked.conversation_id,
                    OrchestrationWaitModel.closed.is_(False),
                    OrchestrationWaitModel.kind.in_(("WORKFLOW_RESULT", "HUMAN_DECISION")),
                )
                .order_by(OrchestrationWaitModel.wait_id)
                .with_for_update()
            )
        ).all()
        for row in rows:
            scope = session_scope(_record(row).reference.scope)
            if not await self._journal.authorize(scope, actor):
                continue
            if row.workflow_event_id == event_value.id:
                return WakeDecision("DUPLICATE", row.wake_job_id)
            if row.wake_job_id is not None:
                continue
            outcome = await self._workflow_outcome(actor, row, linked)
            if outcome is not None:
                return await self._wake(row, event_value.id, outcome)
        return WakeDecision("IGNORED")

    async def _workflow_outcome(
        self, actor: AuthenticatedActor, row: OrchestrationWaitModel, linked: LinkedWorkflowEvent
    ) -> str | None:
        e, w, a = WorkflowEventModel, WorkflowRunModel, AgentRunModel
        persisted = (
            (
                await self._session.execute(
                    select(e, w)
                    .join(w, (w.organization_id == e.organization_id) & (w.id == e.workflow_run_id))
                    .join(a, (a.organization_id == w.organization_id) & (a.workflow_run_id == w.id))
                    .where(
                        e.organization_id == actor.organization_id,
                        e.id == linked.event.id,
                        e.sequence == linked.event.sequence,
                        e.workflow_run_id == linked.event.workflow_run_id,
                        w.requested_by_membership_id == actor.membership_id,
                        a.id == linked.agent_run.id,
                        a.orchestration_run_id == row.source_run_id,
                        live_payload(e, "workflow_events"),
                        live_payload(a, "agent_runs"),
                    )
                )
            )
            .tuples()
            .one_or_none()
        )
        if persisted is None:
            return None
        event, workflow = persisted
        if row.source_fingerprint is not None or row.policy_fingerprint is not None:
            return None
        if row.kind == "HUMAN_DECISION":
            if workflow.status != "COMPLETED":
                return None
            return await self._decision_outcome(row, event)
        if row.child_allocation_id is not None:
            binding = await self._session.get(
                OrchestrationChildJobModel,
                (row.organization_id, row.source_run_id, row.child_allocation_id),
            )
            if binding is None or binding.workflow_job_id is None:
                return None
            job = await self._session.scalar(
                select(WorkflowJobModel).where(
                    WorkflowJobModel.organization_id == row.organization_id,
                    WorkflowJobModel.id == binding.workflow_job_id,
                    WorkflowJobModel.workflow_run_id == workflow.id,
                )
            )
            if job is None or job.status not in {"COMPLETED", "FAILED"}:
                return None
        if (
            row.resource_id != workflow.id
            or row.resource_version is None
            or workflow.version < row.resource_version
        ):
            return None
        if _WORKFLOW_OUTCOMES.get(event.event_type) != workflow.status:
            return None
        has_newer = await self._session.scalar(
            select(e.id)
            .where(
                e.organization_id == row.organization_id,
                e.workflow_run_id == workflow.id,
                e.sequence > event.sequence,
                e.event_type.in_(_WORKFLOW_OUTCOMES),
            )
            .limit(1)
        )
        return workflow.status if has_newer is None else None

    async def _decision_outcome(
        self, row: OrchestrationWaitModel, event: WorkflowEventModel
    ) -> str | None:
        if event.event_type != "workflow.completed":
            return None
        payload = event.public_payload
        if str(row.resource_id) != payload.get(
            "proposal_id"
        ) or row.resource_version != payload.get("proposal_version"):
            return None
        decision = {"APPROVE": "APPROVED", "REJECT": "REJECTED"}.get(str(payload.get("decision")))
        if decision is None:
            return None
        p, a = ProposalModel, ApprovalModel
        approval_id = await self._session.scalar(
            select(a.id)
            .join(
                p,
                (p.organization_id == a.organization_id)
                & (p.id == a.proposal_id)
                & (p.approval_id == a.id),
            )
            .where(
                p.organization_id == row.organization_id,
                p.id == row.resource_id,
                p.workflow_run_id == event.workflow_run_id,
                p.current_version_number == row.resource_version,
                p.status == decision,
                a.proposal_version_number == row.resource_version,
                a.status == decision,
                a.decided_by_membership_id.is_not(None),
            )
        )
        if approval_id is None or str(approval_id) != payload.get("approval_id"):
            return None
        return decision

    async def _wake(
        self, row: OrchestrationWaitModel, event_id: UUID | None, outcome: str
    ) -> WakeDecision:
        job_id = uuid5(row.wait_id, "continuation-wake")
        now = await self._journal.now()
        self._session.add(
            AssistantJobModel(
                id=job_id,
                organization_id=row.organization_id,
                conversation_id=row.conversation_id,
                turn_id=row.owner_turn_id,
                orchestration_run_id=row.owner_run_id,
                requester_membership_id=row.membership_id,
                job_type="assistant.turn.resume",
                status="QUEUED",
                payload={
                    "wait_id": str(row.wait_id),
                    "owner_version": row.version,
                    "source_invocation_id": str(row.source_invocation_id),
                    "workflow_event_id": str(event_id) if event_id is not None else None,
                    "generation_id": str(row.resource_id) if event_id is None else None,
                    "outcome": outcome,
                },
                available_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await self._session.flush()
        row.workflow_event_id = event_id
        row.wake_job_id = job_id
        source = session_scope(_record(row).reference.scope)
        state = await self._journal.state(source, lock=True)
        if state is None:
            raise SessionError("SESSION_NOT_INITIALIZED")
        await SessionService.append_locked(
            self._journal,
            state,
            SessionEvent(
                event_id=uuid5(row.wait_id, "canonical-completion"),
                kind="continuation.received",
                causation_id=event_id or row.resource_id,
                correlation_id=row.wait_id,
            ),
            state.sequence,
        )
        await self._activity(
            row, "assistant.run.wake_queued.v1", row.owner_turn_id, row.owner_run_id
        )
        return WakeDecision("QUEUED", job_id)
