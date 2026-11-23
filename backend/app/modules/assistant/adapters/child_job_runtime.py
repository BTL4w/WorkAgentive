"""Apply a durable child allocation to a claimed Planning or Reporting job."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.execution_fence import ExecutionFenceContext, WorkflowJobFence, execution_fence
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import session_scope
from app.modules.assistant.domain.session_usage import JobClaim
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.run_budget import (
    ChildJobBudget,
    ChildJobClaim,
    UsageReservation,
    child_job_budget,
)


@asynccontextmanager
async def execute_child_job(
    sessions: async_sessionmaker[AsyncSession],
    actor: AuthenticatedActor,
    claim: ChildJobClaim,
) -> AsyncGenerator[None]:
    store = PostgreSQLRunBudget(sessions, actor)
    binding = await store.job_scope(claim)
    if binding is None:
        yield
        return
    scope, allocation_id = binding
    try:
        interval = await store.reserve_for_job(
            scope,
            allocation_id,
            job_claim_token=claim,
            reservation=UsageReservation(kind="INTERVAL", allocation_id=allocation_id),
        )
        interval_id = interval.reservation.operation_id
        operation_id = uuid4()
        try:
            effect = await store.reserve_for_job(
                scope,
                allocation_id,
                job_claim_token=claim,
                reservation=UsageReservation(
                    operation_id=operation_id,
                    kind="TOOL",
                    allocation_id=allocation_id,
                    interval_id=interval_id,
                ),
            )
            fence = ExecutionFenceContext(
                actor=actor,
                scope=session_scope(scope),
                lease=None,
                operation_id=operation_id,
                interval_id=interval_id,
                workflow=WorkflowJobFence(
                    allocation_id,
                    JobClaim(
                        claim.kind, claim.job_id, claim.worker_id, claim.attempt, claim.lease_until
                    ),
                ),
            )
            try:
                with (
                    execution_fence(fence),
                    child_job_budget(
                        ChildJobBudget(scope, allocation_id, claim, store, interval_id)
                    ),
                ):
                    yield
            finally:
                await store.settle_for_job(scope, effect, None)
        finally:
            await store.settle_for_job(scope, interval, None)
    finally:
        await store.close_for_job(scope, allocation_id, claim)
