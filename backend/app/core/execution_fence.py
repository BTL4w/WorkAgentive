"""Candidate effect checks inside the caller's short application transaction.

This fence supplies neither business permission nor approval. Services still own
validation, version checks, receipts, outbox and audit in that same transaction.
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.assistant.adapters.session_usage_repository import PostgreSQLUsageRepository
from app.modules.assistant.application.session_usage_service import RunUsageService
from app.modules.assistant.domain.session import SessionError, SessionLease, SessionScope
from app.modules.assistant.domain.session_usage import ChildJobKind, JobClaim
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.run_budget import current_candidate_budget


@dataclass(frozen=True, slots=True)
class WorkflowJobFence:
    allocation_id: UUID
    claim: JobClaim


@dataclass(frozen=True, slots=True)
class ExecutionFenceContext:
    actor: AuthenticatedActor
    scope: SessionScope
    lease: SessionLease | None
    operation_id: UUID
    workflow: WorkflowJobFence | None = None
    interval_id: UUID | None = None
    allow_job_enqueue: bool = False


_CURRENT: ContextVar[ExecutionFenceContext | None] = ContextVar(
    "candidate_effect_fence", default=None
)


@contextmanager
def execution_fence(context: ExecutionFenceContext) -> Generator[None]:
    """Trusted adapter binding; queue workers must rehydrate it from persisted IDs."""
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)


async def assert_execution_fence(
    session: AsyncSession, context: ExecutionFenceContext | None = None
) -> None:
    context = context or _CURRENT.get()
    if context is None:
        return  # authorized manual flows are independent of candidate execution
    scope, actor, lease = context.scope, context.actor, context.lease
    if not session.in_transaction():
        raise SessionError("EFFECT_TRANSACTION_REQUIRED")
    if (scope.organization_id, scope.membership_id) != (
        actor.organization_id,
        actor.membership_id,
    ):
        raise SessionError("SESSION_NOT_FOUND")
    tenant, member = (
        await session.execute(
            text(
                "SELECT current_setting('app.organization_id',true), "
                "current_setting('app.membership_id',true)"
            )
        )
    ).one()
    if (tenant, member) != (str(scope.organization_id), str(scope.membership_id)):
        raise SessionError("SESSION_NOT_FOUND")
    repo = PostgreSQLUsageRepository(session)
    child = context.workflow
    if child is None:
        if lease is None or lease.scope != scope:
            raise SessionError("SESSION_NOT_FOUND")
        await RunUsageService.guard_effect(
            repo, scope, lease, context.operation_id, interval_id=context.interval_id
        )
    else:
        # Match enqueue's membership -> runtime state -> job lock order.
        await repo.lock_job_actor(scope, actor)
        if lease is not None or await repo.state(scope, lock=True) is None:
            raise SessionError("EFFECT_FENCE_CONTEXT_INVALID")
        await repo.guard_job(scope, child.allocation_id, child.claim)
        entries = await repo.usage_entries(scope)
        allocation = RunUsageService.entry(entries, child.allocation_id)
        entry = RunUsageService.entry(entries, context.operation_id)
        if (
            allocation.kind != "ALLOCATION"
            or allocation.closed
            or entry.kind != "TOOL"
            or entry.closed
            or not entry.job_owned
            or entry.fence != child.claim.attempt
            or entry.allocation_id != child.allocation_id
            or entry.interval_id != context.interval_id
        ):
            raise SessionError("EFFECT_RESERVATION_INVALID")
        await RunUsageService.active(
            repo, entries, entry.interval_id, child.allocation_id, child.claim
        )
        await repo.retention_deadline(scope)
    if not await repo.authorize(scope, actor):
        raise SessionError("SESSION_NOT_FOUND")
    active = await session.scalar(
        text(
            "SELECT status NOT IN ('COMPLETED','FAILED') FROM "
            "orchestration_runs WHERE organization_id=:org AND id=:run"
        ),
        {"org": scope.organization_id, "run": scope.orchestration_run_id},
    )
    if active is not True:
        raise SessionError("EXECUTION_CANCELLED")


async def bind_candidate_job(session: AsyncSession, kind: ChildJobKind, job_id: UUID) -> None:
    """Link a queued job to the child allocation inside its enqueue transaction."""
    context = _CURRENT.get()
    if context is None or not context.allow_job_enqueue:
        return
    budget = current_candidate_budget()
    if (
        context.workflow is not None
        or budget is None
        or budget.allocation_id is None
        or (budget.scope.organization_id, budget.scope.orchestration_run_id)
        != (context.scope.organization_id, context.scope.orchestration_run_id)
    ):
        raise SessionError("BUDGET_SCOPE_MISMATCH")
    await assert_execution_fence(session, context)
    await RunUsageService.bind_job_under_lock(
        PostgreSQLUsageRepository(session), context.scope, budget.allocation_id, kind, job_id
    )


async def assert_child_job_effect_fence(session: AsyncSession) -> None:
    context = _CURRENT.get()
    if context is not None and context.workflow is not None:
        await assert_execution_fence(session, context)


def child_job_actor(organization_id: UUID) -> AuthenticatedActor | None:
    context = _CURRENT.get()
    if context is None or context.workflow is None:
        return None
    if context.scope.organization_id != organization_id:
        raise SessionError("SESSION_NOT_FOUND")
    return context.actor


@asynccontextmanager
async def fenced_effect(
    session: AsyncSession, context: ExecutionFenceContext | None = None
) -> AsyncGenerator[None]:
    """Bracket only short DB work; leave this block before the service commits.

    Never hold these locks over model calls, network I/O or human waits.
    """
    await assert_execution_fence(session, context)
    yield
    await assert_execution_fence(session, context)
