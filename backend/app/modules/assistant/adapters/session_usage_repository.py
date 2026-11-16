"""SQL storage only. Application service owns the lease+ledger transaction."""

from dataclasses import asdict, fields
from typing import Self
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.session_repository import (
    PostgreSQLSessionRepository,
    PostgreSQLSessionTransaction,
)
from app.modules.assistant.adapters.session_usage_models import OrchestrationUsageModel as Row
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.domain.session import SessionError, SessionScope
from app.modules.assistant.domain.session_usage import UsageEntry
from app.modules.identity.domain.auth import AuthenticatedActor


class PostgreSQLUsageRepository(PostgreSQLSessionRepository):
    async def usage_entries(self, scope: SessionScope) -> tuple[UsageEntry, ...]:
        rows = (
            await self.session.scalars(
                select(Row).where(
                    Row.organization_id == scope.organization_id,
                    Row.orchestration_run_id == scope.orchestration_run_id,
                )
            )
        ).all()
        return tuple(
            UsageEntry(**{field.name: getattr(row, field.name) for field in fields(UsageEntry)})
            for row in rows
        )

    async def insert_usage(self, scope: SessionScope, entry: UsageEntry) -> None:
        self.session.add(
            Row(
                organization_id=scope.organization_id,
                orchestration_run_id=scope.orchestration_run_id,
                **asdict(entry),
            )
        )
        await self.session.flush()

    async def _row(self, scope: SessionScope, operation_id: UUID) -> Row:
        row = await self.session.scalar(
            select(Row).where(
                Row.organization_id == scope.organization_id,
                Row.orchestration_run_id == scope.orchestration_run_id,
                Row.operation_id == operation_id,
            )
        )
        if row is None:
            raise SessionError("BUDGET_OPERATION_NOT_FOUND")
        return row

    async def settle_usage(self, scope: SessionScope, entry: UsageEntry) -> None:
        row = await self._row(scope, entry.operation_id)
        row.settled, row.settlement_fingerprint, row.closed = (
            entry.settled,
            entry.settlement_fingerprint,
            True,
        )
        await self.session.flush()

    async def close_allocation(self, scope: SessionScope, operation_id: UUID) -> None:
        row = await self._row(scope, operation_id)
        row.closed = True
        await self.session.flush()


class PostgreSQLUsageTransaction(PostgreSQLSessionTransaction):
    def __init__(self, factory: PostgreSQLAssistantTransactionFactory, actor: AuthenticatedActor):
        super().__init__(factory, actor)
        self.repository = PostgreSQLUsageRepository(self.repository.session)

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        return self


class PostgreSQLUsageTransactionFactory:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self._factory = PostgreSQLAssistantTransactionFactory(sessions)

    def __call__(self, actor: AuthenticatedActor) -> PostgreSQLUsageTransaction:
        return PostgreSQLUsageTransaction(self._factory, actor)
