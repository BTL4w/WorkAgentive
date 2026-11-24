"""Application transaction boundary for trusted continuation commands."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from app.modules.assistant.application.ports import LinkedWorkflowEvent
from app.modules.assistant.domain.continuation import (
    TransferReceipt,
    WaitDraft,
    WaitRecord,
    WakeDecision,
)
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.domain.auth import AuthenticatedActor


class ContinuationTransaction(Protocol):
    async def open_wait(self, actor: AuthenticatedActor, value: WaitDraft) -> WaitRecord: ...

    async def receive_event(
        self, actor: AuthenticatedActor, event: LinkedWorkflowEvent
    ) -> WakeDecision: ...

    async def transfer(
        self,
        actor: AuthenticatedActor,
        *,
        wait_id: UUID,
        expected_version: int,
        target_run_id: UUID,
    ) -> TransferReceipt: ...


class ContinuationTransactions(Protocol):
    def __call__(
        self, actor: AuthenticatedActor
    ) -> AbstractAsyncContextManager[ContinuationTransaction]: ...


class ContinuationService:
    def __init__(self, transactions: ContinuationTransactions) -> None:
        self._transactions = transactions

    async def open_wait(self, actor: AuthenticatedActor, value: WaitDraft) -> WaitRecord:
        scope = value.reference.scope
        if (scope.organization_id, scope.membership_id) != (
            actor.organization_id,
            actor.membership_id,
        ) or value.lease.scope != scope:
            raise SessionError("CONTINUATION_NOT_FOUND")
        async with self._transactions(actor) as transaction:
            return await transaction.open_wait(actor, value)

    async def receive_event(
        self, actor: AuthenticatedActor, event: LinkedWorkflowEvent
    ) -> WakeDecision:
        if event.event.organization_id != actor.organization_id:
            return WakeDecision("IGNORED")
        async with self._transactions(actor) as transaction:
            return await transaction.receive_event(actor, event)

    async def transfer(
        self,
        actor: AuthenticatedActor,
        *,
        wait_id: UUID,
        expected_version: int,
        target_run_id: UUID,
    ) -> TransferReceipt:
        """Claim a wait after trusted routing has bound the user's follow-up to it."""
        if type(expected_version) is not int or expected_version < 1:
            raise SessionError("CONTINUATION_VERSION_CONFLICT")
        async with self._transactions(actor) as transaction:
            return await transaction.transfer(
                actor,
                wait_id=wait_id,
                expected_version=expected_version,
                target_run_id=target_run_id,
            )
