"""Application-owned durable budget storage; same transaction as lease fencing."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from app.modules.assistant.application.session_ports import SessionRepository
from app.modules.assistant.domain.session import SessionScope
from app.modules.assistant.domain.session_usage import UsageEntry
from app.modules.identity.domain.auth import AuthenticatedActor


class UsageRepository(SessionRepository, Protocol):
    async def usage_entries(self, scope: SessionScope) -> tuple[UsageEntry, ...]: ...
    async def insert_usage(self, scope: SessionScope, entry: UsageEntry) -> None: ...
    async def settle_usage(self, scope: SessionScope, entry: UsageEntry) -> None: ...
    async def close_allocation(self, scope: SessionScope, operation_id: UUID) -> None: ...


class UsageTransaction(Protocol):
    @property
    def repository(self) -> UsageRepository: ...
    async def commit(self) -> None: ...


class UsageTransactionFactory(Protocol):
    def __call__(
        self, actor: AuthenticatedActor
    ) -> AbstractAsyncContextManager[UsageTransaction]: ...
