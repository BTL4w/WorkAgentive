"""AI budget port to the application ledger; models never authorize allowance."""

import math
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.session_usage_repository import (
    PostgreSQLUsageTransactionFactory,
)
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.application.session_service import fingerprint
from app.modules.assistant.application.session_usage_service import RunUsageService
from app.modules.assistant.domain.session import SessionError
from app.modules.assistant.domain.session_usage import DIMENSIONS
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.runtime.run_budget import (
    BudgetReceipt,
    ChildAllocation,
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


class PostgreSQLRunBudget:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], actor: AuthenticatedActor):
        self.service = RunUsageService(PostgreSQLUsageTransactionFactory(sessions), actor)

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
        replayed, seconds = await self.service.reserve(
            session_scope(scope),
            session_lease(lease),
            operation_id=reservation.operation_id,
            kind=reservation.kind,
            allocation_id=reservation.allocation_id,
            interval_id=reservation.interval_id,
            inputs=reservation.input_tokens,
            outputs=reservation.output_tokens,
            retry_of=reservation.retry_of,
            repair_of=reservation.repair_of,
        )
        return BudgetReceipt(
            scope=scope,
            lease=lease,
            reservation=reservation,
            replayed=replayed,
            seconds_remaining=seconds,
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
        request = receipt.reservation
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
            session_lease(receipt.lease),
            request.operation_id,
            identity,
            (usage.input_tokens, usage.output_tokens) if usage else None,
        )

    async def remaining(self, scope: RunScope) -> RemainingBudget:
        consumed, available = await self.service.remaining(session_scope(scope))
        # Root availability is also constrained by unallocated aggregate requests.
        available["root_model_requests"] = min(
            available["root_model_requests"], available["total_model_requests"]
        )
        return RemainingBudget(usage=_usage(consumed), available=_usage(available))
