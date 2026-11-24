"""SQLAlchemy private journal adapter; every operation is transaction/tenant scoped."""

from dataclasses import asdict, fields
from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.database_models import (
    AssistantConversationModel,
    AssistantJobModel,
    AssistantTurnModel,
    OrchestrationRunModel,
)
from app.modules.assistant.adapters.session_models import (
    OrchestrationRuntimeStateModel as StateRow,
)
from app.modules.assistant.adapters.session_models import (
    OrchestrationSessionEventModel as EventRow,
)
from app.modules.assistant.adapters.session_models import (
    OrchestrationSessionPayloadModel as PayloadRow,
)
from app.modules.assistant.adapters.session_usage_models import OrchestrationUsageModel
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.domain.session import (
    RuntimeState,
    SessionError,
    SessionEvent,
    SessionLease,
    SessionPayload,
    SessionPins,
    SessionScope,
    StoredSessionEvent,
)
from app.modules.assistant.domain.session_usage import POOL_ID
from app.modules.feedback.adapters.payload_models import RetentionPayloadModel
from app.modules.feedback.adapters.payload_repository import live_payload
from app.modules.identity.adapters.database_models import UserModel
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.adapters.database_models import MembershipModel


class PostgreSQLSessionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def authorize(
        self, scope: SessionScope, actor: AuthenticatedActor, *, lock: bool = False
    ) -> bool:
        r, t, c, m, u = (
            OrchestrationRunModel,
            AssistantTurnModel,
            AssistantConversationModel,
            MembershipModel,
            UserModel,
        )
        statement = (
            select(r.id)
            .join(t, (r.organization_id == t.organization_id) & (r.turn_id == t.id))
            .join(c, (t.organization_id == c.organization_id) & (t.conversation_id == c.id))
            .join(m, (m.organization_id == r.organization_id) & (m.id == r.actor_membership_id))
            .join(u, u.id == m.user_id)
            .where(
                r.organization_id == scope.organization_id,
                r.id == scope.orchestration_run_id,
                r.turn_id == scope.turn_id,
                r.actor_membership_id == scope.membership_id,
                t.actor_membership_id == scope.membership_id,
                c.id == scope.conversation_id,
                c.owner_membership_id == scope.membership_id,
                m.is_active.is_(True),
                u.is_active.is_(True),
                m.user_id == actor.user_id,
            )
        )
        if lock:
            statement = statement.with_for_update(of=r)
        return await self.session.scalar(statement) is not None

    async def now(self) -> datetime:
        value = await self.session.scalar(select(func.clock_timestamp()))
        assert isinstance(value, datetime)
        return value

    async def state(self, scope: SessionScope, *, lock: bool = False) -> RuntimeState | None:
        query = select(StateRow).where(
            StateRow.organization_id == scope.organization_id,
            StateRow.orchestration_run_id == scope.orchestration_run_id,
        )
        if lock:
            query = query.with_for_update()
        row = await self.session.scalar(query)
        if row is None:
            return None
        pins = SessionPins(**{f.name: getattr(row, f.name) for f in fields(SessionPins)})
        lease = None
        if row.lease_id is not None:
            assert (
                row.lease_job_id is not None
                and row.lease_owner is not None
                and row.lease_until is not None
            )
            lease = SessionLease(
                scope,
                row.lease_id,
                row.lease_job_id,
                row.lease_owner,
                row.fence,
                row.lease_until,
                row.lease_job_attempt,
            )
        return RuntimeState(
            scope,
            pins,
            row.sequence,
            row.checkpoint_sequence,
            row.checkpoint_payload_id,
            lease,
            row.fence,
        )

    async def create_state(self, state: RuntimeState) -> None:
        self.session.add(StateRow(**asdict(state.scope), **asdict(state.pins)))
        await self.session.flush()

    async def initialize_budget(self, scope: SessionScope, limits: dict[str, int]) -> None:
        from app.modules.assistant.application.session_service import fingerprint

        self.session.add(
            OrchestrationUsageModel(
                organization_id=scope.organization_id,
                orchestration_run_id=scope.orchestration_run_id,
                operation_id=POOL_ID,
                kind="POOL",
                amount=limits,
                fingerprint=fingerprint(limits),
                fence=0,
                started_at=await self.now(),
                closed=False,
            )
        )
        await self.session.flush()

    async def save_state(self, state: RuntimeState) -> None:
        row = await self.session.scalar(
            select(StateRow).where(
                StateRow.organization_id == state.scope.organization_id,
                StateRow.orchestration_run_id == state.scope.orchestration_run_id,
            )
        )
        if row is None:
            raise SessionError("SESSION_NOT_FOUND")
        row.sequence = state.sequence
        row.checkpoint_sequence = state.checkpoint_sequence
        row.checkpoint_payload_id = state.checkpoint_payload_id
        row.fence = state.fence
        lease = state.lease
        row.lease_id = lease.lease_id if lease else None
        row.lease_job_id = lease.job_id if lease else None
        row.lease_job_attempt = lease.job_attempt if lease else None
        row.lease_owner = lease.worker_id if lease else None
        row.lease_until = lease.expires_at if lease else None
        await self.session.flush()

    async def job_matches(self, scope: SessionScope, job_id: UUID) -> bool:
        j = AssistantJobModel
        return (
            await self.session.scalar(
                select(j.id).where(
                    j.organization_id == scope.organization_id,
                    j.id == job_id,
                    j.orchestration_run_id == scope.orchestration_run_id,
                    j.turn_id == scope.turn_id,
                    j.conversation_id == scope.conversation_id,
                    j.requester_membership_id == scope.membership_id,
                    j.status.in_(("QUEUED", "RUNNING")),
                )
            )
            is not None
        )

    async def lock_job_expiry(
        self, scope: SessionScope, job_id: UUID, worker_id: str, attempt: int
    ) -> datetime | None:
        job = AssistantJobModel
        return await self.session.scalar(
            select(job.lease_until)
            .where(
                job.organization_id == scope.organization_id,
                job.id == job_id,
                job.orchestration_run_id == scope.orchestration_run_id,
                job.turn_id == scope.turn_id,
                job.conversation_id == scope.conversation_id,
                job.requester_membership_id == scope.membership_id,
                job.status == "RUNNING",
                job.locked_by == worker_id,
                job.attempt_count == attempt,
            )
            .with_for_update()
        )

    async def put_payload(self, payload: SessionPayload) -> None:
        self.session.add(PayloadRow(**asdict(payload)))
        await self.session.flush()

    async def payload(self, scope: SessionScope, payload_id: UUID) -> SessionPayload | None:
        row = await self.session.scalar(
            select(PayloadRow).where(
                PayloadRow.organization_id == scope.organization_id,
                PayloadRow.orchestration_run_id == scope.orchestration_run_id,
                PayloadRow.id == payload_id,
            )
        )
        return (
            None
            if row is None
            else SessionPayload(**{f.name: getattr(row, f.name) for f in fields(SessionPayload)})
        )

    async def retention_deadline(self, scope: SessionScope) -> datetime:
        value = await self.session.scalar(
            select(RetentionPayloadModel.expires_at).where(
                RetentionPayloadModel.organization_id == scope.organization_id,
                RetentionPayloadModel.storage_kind == "orchestration_runs",
                RetentionPayloadModel.owner_id == scope.orchestration_run_id,
                RetentionPayloadModel.purged_at.is_(None),
            )
        )
        if value is None or value <= await self.now():
            raise SessionError("CONTEXT_EXPIRED")
        return value

    async def payload_live(self, scope: SessionScope, payload_id: UUID) -> bool:
        return (
            await self.session.scalar(
                select(PayloadRow.id).where(
                    PayloadRow.organization_id == scope.organization_id,
                    PayloadRow.orchestration_run_id == scope.orchestration_run_id,
                    PayloadRow.id == payload_id,
                    PayloadRow.expires_at > func.clock_timestamp(),
                    live_payload(PayloadRow, "orchestration_session_payloads"),
                )
            )
            is not None
        )

    @staticmethod
    def _event(scope: SessionScope, row: EventRow) -> StoredSessionEvent:
        return StoredSessionEvent(
            scope,
            row.sequence,
            row.occurred_at,
            SessionEvent(**{f.name: getattr(row, f.name) for f in fields(SessionEvent)}),
            row.fingerprint,
        )

    async def event(self, scope: SessionScope, event_id: UUID) -> StoredSessionEvent | None:
        row = await self.session.scalar(
            select(EventRow).where(
                EventRow.organization_id == scope.organization_id,
                EventRow.orchestration_run_id == scope.orchestration_run_id,
                EventRow.event_id == event_id,
            )
        )
        return None if row is None else self._event(scope, row)

    async def events(
        self, scope: SessionScope, *, after: int = 0, invocation_id: UUID | None = None
    ) -> tuple[StoredSessionEvent, ...]:
        query = (
            select(EventRow)
            .where(
                EventRow.organization_id == scope.organization_id,
                EventRow.orchestration_run_id == scope.orchestration_run_id,
                EventRow.sequence > after,
            )
            .order_by(EventRow.sequence)
        )
        if invocation_id is not None:
            query = query.where(EventRow.invocation_id == invocation_id)
        return tuple(self._event(scope, row) for row in (await self.session.scalars(query)).all())

    async def insert_event(self, event: StoredSessionEvent) -> None:
        self.session.add(
            EventRow(
                organization_id=event.scope.organization_id,
                orchestration_run_id=event.scope.orchestration_run_id,
                sequence=event.sequence,
                occurred_at=event.occurred_at,
                fingerprint=event.fingerprint,
                **asdict(event.event),
            )
        )
        await self.session.flush()


class PostgreSQLSessionTransaction:
    def __init__(self, factory: PostgreSQLAssistantTransactionFactory, actor: AuthenticatedActor):
        self._inner = factory(actor)
        self.repository = PostgreSQLSessionRepository(self._inner.session)

    async def __aenter__(self) -> Self:
        try:
            await self._inner.__aenter__()
        except SQLAlchemyError:
            raise SessionError("SESSION_STORAGE_FAILED") from None
        return self

    async def commit(self) -> None:
        await self._inner.commit()

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            await self._inner.__aexit__(exc_type, exc_val, exc_tb)
        except SQLAlchemyError:
            raise SessionError("SESSION_STORAGE_FAILED") from None
        if isinstance(exc_val, SQLAlchemyError):
            raise SessionError("SESSION_STORAGE_FAILED") from None


class PostgreSQLSessionTransactionFactory:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self._factory = PostgreSQLAssistantTransactionFactory(sessions)

    def __call__(self, actor: AuthenticatedActor) -> PostgreSQLSessionTransaction:
        return PostgreSQLSessionTransaction(self._factory, actor)
