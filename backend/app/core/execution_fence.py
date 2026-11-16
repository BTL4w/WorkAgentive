"""Candidate effect checks inside the caller's short application transaction.

This fence supplies neither business permission nor approval. Services still own
validation, version checks, receipts, outbox and audit in that same transaction.
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.assistant.adapters.session_usage_repository import PostgreSQLUsageRepository
from app.modules.assistant.application.session_usage_service import RunUsageService
from app.modules.assistant.domain.session import SessionError, SessionLease, SessionScope
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.planning_runs.adapters.database_models import WorkflowJobModel, WorkflowRunModel


@dataclass(frozen=True, slots=True)
class WorkflowJobFence:
    job_id: UUID
    workflow_run_id: UUID
    worker_id: str
    attempt_count: int
    lease_until: datetime
    allocation_id: UUID


@dataclass(frozen=True, slots=True)
class ExecutionFenceContext:
    actor: AuthenticatedActor
    scope: SessionScope
    lease: SessionLease
    operation_id: UUID
    workflow: WorkflowJobFence | None = None
    interval_id: UUID | None = None


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
    if lease.scope != scope or (scope.organization_id, scope.membership_id) != (
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
    # Lock order: runtime state -> job -> effect. Locks live until application commit.
    if context.workflow is None:
        await RunUsageService.guard_effect(
            repo, scope, lease, context.operation_id, interval_id=context.interval_id
        )
    else:
        if await repo.state(scope, lock=True) is None:
            raise SessionError("SESSION_NOT_INITIALIZED")
        child = context.workflow
        job = await session.scalar(
            select(WorkflowJobModel)
            .where(
                WorkflowJobModel.organization_id == scope.organization_id,
                WorkflowJobModel.id == child.job_id,
                WorkflowJobModel.workflow_run_id == child.workflow_run_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        now = await repo.now()
        if (
            job is None
            or job.status != "RUNNING"
            or job.locked_by_worker_id != child.worker_id
            or job.attempt_count != child.attempt_count
            or job.lease_until != child.lease_until
            or job.lease_until is None
            or job.lease_until <= now
        ):
            raise SessionError("WORKFLOW_LEASE_LOST")
        workflow = await session.scalar(
            select(WorkflowRunModel).where(
                WorkflowRunModel.organization_id == scope.organization_id,
                WorkflowRunModel.id == child.workflow_run_id,
                WorkflowRunModel.requested_by_membership_id == scope.membership_id,
            )
        )
        if workflow is None or workflow.status in {"COMPLETED", "FAILED"}:
            raise SessionError("WORKFLOW_LEASE_LOST")
        entries = await repo.usage_entries(scope)
        allocation = RunUsageService.entry(entries, child.allocation_id)
        entry = RunUsageService.entry(entries, context.operation_id)
        if (
            allocation.kind != "ALLOCATION"
            or allocation.closed
            or entry.kind not in {"TOOL", "HANDOFF"}
            or entry.closed
            or entry.allocation_id != allocation.operation_id
        ):
            raise SessionError("EFFECT_RESERVATION_INVALID")
        await RunUsageService.active(repo, entries, entry.interval_id, entry.allocation_id, lease)
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
