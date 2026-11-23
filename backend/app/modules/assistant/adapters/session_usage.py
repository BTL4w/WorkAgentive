"""AI budget port to the application ledger; models never authorize allowance."""

import math
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.session_usage_repository import (
    PostgreSQLUsageTransactionFactory,
)
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.application.session_service import fingerprint
from app.modules.assistant.application.session_usage_service import RunUsageService
from app.modules.assistant.domain.session import SessionError, SessionLease
from app.modules.assistant.domain.session_usage import DIMENSIONS, JobClaim
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.runtime.run_budget import (
    BudgetReceipt,
    ChildAllocation,
    ChildJobClaim,
    JobBudgetReceipt,
    RemainingBudget,
    UsageReservation,
)
from work_management_ai.runtime.tool_loop_contracts import RunLease, RunScope, ToolLoopUsage


def _vector(usage: ToolLoopUsage) -> dict[str, int]:
    return {
        key: math.ceil(usage.active_seconds * 1000)
        if key == "active_milliseconds"
        else int(getattr(usage, key))
        for key in DIMENSIONS
    }


def _usage(value: dict[str, int]) -> ToolLoopUsage:
    return ToolLoopUsage(
        **{key: number for key, number in value.items() if key != "active_milliseconds"},
        active_seconds=value["active_milliseconds"] / 1000,
    )


def _claim(value: ChildJobClaim) -> JobClaim:
    return JobClaim(value.kind, value.job_id, value.worker_id, value.attempt, value.lease_until)


class PostgreSQLRunBudget:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], actor: AuthenticatedActor):
        self._transactions = PostgreSQLUsageTransactionFactory(sessions)
        self._actor = actor
        self.service = RunUsageService(self._transactions, actor)

    async def job_scope(self, claim: ChildJobClaim) -> tuple[RunScope, UUID] | None:
        async with self._transactions(self._actor) as transaction:
            bound = await transaction.repository.bound_job(self._actor, claim.kind, claim.job_id)
        if bound is None:
            return None
        scope, allocation_id = bound
        return RunScope(
            organization_id=scope.organization_id,
            membership_id=scope.membership_id,
            conversation_id=scope.conversation_id,
            turn_id=scope.turn_id,
            orchestration_run_id=scope.orchestration_run_id,
        ), allocation_id

    async def close_for_job(
        self,
        scope: RunScope,
        allocation_id: UUID,
        claim: ChildJobClaim,
    ) -> None:
        await self.service.close_job(session_scope(scope), allocation_id, _claim(claim))

    async def recover_tool(
        self,
        scope: RunScope,
        operation_id: UUID,
        *,
        lease: RunLease,
        interval_id: UUID,
        allocation_id: UUID | None,
    ) -> BudgetReceipt:
        async with self.service.transaction(session_scope(scope), session_lease(lease)) as repo:
            entries = await repo.usage_entries(session_scope(scope))
            entry = next((e for e in entries if e.operation_id == operation_id), None)
            if entry is None or entry.kind not in {"TOOL", "HANDOFF"}:
                raise SessionError("EFFECT_RESERVATION_INVALID")
            if entry.allocation_id != allocation_id:
                raise SessionError("BUDGET_SCOPE_MISMATCH")
            active_seconds = await self.service.active(
                repo,
                entries,
                interval_id,
                allocation_id,
                session_lease(lease),
            )
            state = await repo.state(session_scope(scope))
            assert state is not None and state.lease is not None
            seconds = (state.lease.expires_at - await repo.now()).total_seconds()
            if seconds <= 0:
                raise SessionError("LEASE_LOST")
            return BudgetReceipt(
                scope=scope,
                lease=lease,
                replayed=True,
                seconds_remaining=min(seconds, active_seconds),
                reservation=UsageReservation(
                    operation_id=operation_id,
                    kind="TOOL" if entry.kind == "TOOL" else "HANDOFF",
                    allocation_id=entry.allocation_id,
                    interval_id=entry.interval_id,
                ),
            )

    async def reserve(
        self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt:
        reservation = UsageReservation.model_validate(reservation)
        replayed, seconds = await self._reserve(scope, reservation, session_lease(lease))
        return BudgetReceipt(
            scope=scope,
            lease=lease,
            reservation=reservation,
            replayed=replayed,
            seconds_remaining=seconds,
        )

    async def bind_job(
        self,
        scope: RunScope,
        allocation_id: UUID,
        kind: Literal["PLANNING", "REPORTING"],
        job_id: UUID,
        *,
        lease: RunLease,
    ) -> None:
        await self.service.bind_job(
            session_scope(scope), session_lease(lease), allocation_id, kind, job_id
        )

    async def reserve_for_job(
        self,
        scope: RunScope,
        allocation_id: UUID,
        *,
        job_claim_token: ChildJobClaim,
        reservation: UsageReservation,
    ) -> JobBudgetReceipt:
        claim = ChildJobClaim.model_validate(job_claim_token)
        reservation = UsageReservation.model_validate(reservation)
        if reservation.allocation_id != allocation_id:
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        replayed, seconds = await self._reserve(scope, reservation, _claim(claim))
        return JobBudgetReceipt(
            scope=scope,
            claim=claim,
            reservation=reservation,
            replayed=replayed,
            seconds_remaining=seconds,
        )

    async def _reserve(
        self,
        scope: RunScope,
        reservation: UsageReservation,
        authority: SessionLease | JobClaim,
    ) -> tuple[bool, float]:
        return await self.service.reserve(
            session_scope(scope),
            authority,
            operation_id=reservation.operation_id,
            kind=reservation.kind,
            allocation_id=reservation.allocation_id,
            interval_id=reservation.interval_id,
            inputs=reservation.input_tokens,
            outputs=reservation.output_tokens,
            retry_of=reservation.retry_of,
            repair_of=reservation.repair_of,
        )

    async def allocate_child(
        self, scope: RunScope, allocation: ChildAllocation, *, lease: RunLease
    ) -> ChildAllocation:
        allocation = ChildAllocation.model_validate(allocation)
        await self.service.allocate(
            session_scope(scope),
            session_lease(lease),
            allocation.allocation_id,
            _vector(allocation.allowance),
        )
        return allocation

    async def close_child(self, scope: RunScope, allocation_id: UUID, *, lease: RunLease) -> None:
        await self.service.close_child(session_scope(scope), session_lease(lease), allocation_id)

    async def settle(
        self, scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
    ) -> None:
        if receipt.scope != scope:
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        await self._settle(scope, session_lease(receipt.lease), receipt.reservation, usage)

    async def settle_for_job(
        self,
        scope: RunScope,
        receipt: JobBudgetReceipt,
        usage: ModelUsage | None,
    ) -> None:
        if receipt.scope != scope:
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        await self._settle(scope, _claim(receipt.claim), receipt.reservation, usage)

    async def _settle(
        self,
        scope: RunScope,
        authority: SessionLease | JobClaim,
        request: UsageReservation,
        usage: ModelUsage | None,
    ) -> None:
        identity = fingerprint(
            dict(
                kind=request.kind,
                allocation_id=request.allocation_id,
                interval_id=request.interval_id,
                inputs=request.input_tokens,
                outputs=request.output_tokens,
                retry_of=request.retry_of,
                repair_of=request.repair_of,
            )
        )
        await self.service.settle(
            session_scope(scope),
            authority,
            request.operation_id,
            identity,
            (usage.input_tokens, usage.output_tokens) if usage else None,
            allocation_id=request.allocation_id,
        )

    async def remaining(self, scope: RunScope) -> RemainingBudget:
        consumed, available = await self.service.remaining(session_scope(scope))
        # Root availability is also constrained by unallocated aggregate requests.
        available["root_model_requests"] = min(
            available["root_model_requests"], available["total_model_requests"]
        )
        return RemainingBudget(usage=_usage(consumed), available=_usage(available))
