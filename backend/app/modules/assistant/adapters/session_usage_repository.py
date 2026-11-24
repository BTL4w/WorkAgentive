"""SQL storage only. Application service owns the lease+ledger transaction."""

from dataclasses import asdict, fields
from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import select, text, true
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.database_models import AssistantTurnModel, OrchestrationRunModel
from app.modules.assistant.adapters.session_repository import (
    PostgreSQLSessionRepository,
    PostgreSQLSessionTransaction,
)
from app.modules.assistant.adapters.session_usage_models import OrchestrationChildJobModel
from app.modules.assistant.adapters.session_usage_models import OrchestrationUsageModel as Row
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.domain.session import SessionError, SessionScope
from app.modules.assistant.domain.session_usage import ChildJobKind, JobClaim, UsageEntry
from app.modules.identity.adapters.database_models import UserModel
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.adapters.database_models import MembershipModel
from app.modules.organization.domain.roles import MembershipRole
from app.modules.planning_runs.adapters.database_models import WorkflowJobModel, WorkflowRunModel
from app.modules.reporting.adapters.usage_models import ReportGenerationJobModel


class PostgreSQLUsageRepository(PostgreSQLSessionRepository):
    async def lock_job_actor(self, scope: SessionScope, actor: AuthenticatedActor) -> None:
        if actor.role not in {MembershipRole.ADMIN, MembershipRole.MANAGER}:
            raise SessionError("SESSION_NOT_FOUND")
        active = await self.session.scalar(
            text("SELECT public.lock_active_membership(:org,:member)"),
            {"org": scope.organization_id, "member": scope.membership_id},
        )
        # Read the role on a fresh statement after acquiring the membership lock.
        if (
            active is not True
            or await self.session.scalar(
                select(MembershipModel.id)
                .join(UserModel, UserModel.id == MembershipModel.user_id)
                .where(
                    MembershipModel.organization_id == scope.organization_id,
                    MembershipModel.id == scope.membership_id,
                    MembershipModel.user_id == actor.user_id,
                    MembershipModel.role == actor.role,
                    UserModel.is_active.is_(True),
                )
            )
            is None
        ):
            raise SessionError("SESSION_NOT_FOUND")

    async def bound_job(
        self,
        actor: AuthenticatedActor,
        kind: ChildJobKind,
        job_id: UUID,
    ) -> tuple[SessionScope, UUID] | None:
        job_column = (
            OrchestrationChildJobModel.workflow_job_id
            if kind == "PLANNING"
            else OrchestrationChildJobModel.generation_job_id
        )
        if kind == "PLANNING":
            requester = (
                select(WorkflowRunModel.requested_by_membership_id)
                .join(
                    WorkflowJobModel,
                    (WorkflowJobModel.organization_id == WorkflowRunModel.organization_id)
                    & (WorkflowJobModel.workflow_run_id == WorkflowRunModel.id),
                )
                .where(
                    WorkflowJobModel.organization_id == actor.organization_id,
                    WorkflowJobModel.id == job_id,
                )
            )
        else:
            requester = select(ReportGenerationJobModel.requester_membership_id).where(
                ReportGenerationJobModel.organization_id == actor.organization_id,
                ReportGenerationJobModel.id == job_id,
            )
        binding = (
            select(
                OrchestrationChildJobModel.allocation_id,
                OrchestrationRunModel.id.label("run_id"),
                AssistantTurnModel.id.label("turn_id"),
                AssistantTurnModel.conversation_id,
            )
            .select_from(OrchestrationChildJobModel)
            .join(
                OrchestrationRunModel,
                (
                    OrchestrationRunModel.organization_id
                    == OrchestrationChildJobModel.organization_id
                )
                & (OrchestrationRunModel.id == OrchestrationChildJobModel.orchestration_run_id),
            )
            .join(
                AssistantTurnModel,
                (AssistantTurnModel.organization_id == OrchestrationRunModel.organization_id)
                & (AssistantTurnModel.id == OrchestrationRunModel.turn_id),
            )
            .where(
                OrchestrationChildJobModel.organization_id == actor.organization_id,
                OrchestrationRunModel.actor_membership_id == actor.membership_id,
                AssistantTurnModel.actor_membership_id == actor.membership_id,
                job_column == job_id,
            )
            .subquery()
        )
        row = (
            await self.session.execute(
                select(
                    binding.c.allocation_id,
                    binding.c.run_id,
                    binding.c.turn_id,
                    binding.c.conversation_id,
                )
                .select_from(MembershipModel)
                .join(UserModel, UserModel.id == MembershipModel.user_id)
                .outerjoin(binding, true())
                .where(
                    MembershipModel.organization_id == actor.organization_id,
                    MembershipModel.id == actor.membership_id,
                    MembershipModel.id == requester.scalar_subquery(),
                    MembershipModel.user_id == actor.user_id,
                    MembershipModel.role == actor.role,
                    MembershipModel.is_active.is_(True),
                    UserModel.is_active.is_(True),
                )
            )
        ).one_or_none()
        if row is None:
            raise SessionError("SESSION_NOT_FOUND")
        allocation_id, run_id, turn_id, conversation_id = row
        if allocation_id is None:
            return None
        return SessionScope(
            actor.organization_id,
            actor.membership_id,
            conversation_id,
            turn_id,
            run_id,
        ), allocation_id

    async def _job_link(
        self, scope: SessionScope, allocation_id: UUID
    ) -> OrchestrationChildJobModel | None:
        return await self.session.scalar(
            select(OrchestrationChildJobModel).where(
                OrchestrationChildJobModel.organization_id == scope.organization_id,
                OrchestrationChildJobModel.orchestration_run_id == scope.orchestration_run_id,
                OrchestrationChildJobModel.allocation_id == allocation_id,
            )
        )

    async def has_job(self, scope: SessionScope, allocation_id: UUID) -> bool:
        return await self._job_link(scope, allocation_id) is not None

    async def _planning_job(self, scope: SessionScope, job_id: UUID) -> WorkflowJobModel | None:
        return await self.session.scalar(
            select(WorkflowJobModel)
            .join(
                WorkflowRunModel,
                (WorkflowRunModel.organization_id == WorkflowJobModel.organization_id)
                & (WorkflowRunModel.id == WorkflowJobModel.workflow_run_id),
            )
            .where(
                WorkflowJobModel.organization_id == scope.organization_id,
                WorkflowJobModel.id == job_id,
                WorkflowRunModel.requested_by_membership_id == scope.membership_id,
            )
            .with_for_update(of=WorkflowJobModel)
            .execution_options(populate_existing=True)
        )

    async def _reporting_job(
        self, scope: SessionScope, job_id: UUID
    ) -> ReportGenerationJobModel | None:
        return await self.session.scalar(
            select(ReportGenerationJobModel)
            .where(
                ReportGenerationJobModel.organization_id == scope.organization_id,
                ReportGenerationJobModel.id == job_id,
                ReportGenerationJobModel.requester_membership_id == scope.membership_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    async def bind_job(
        self,
        scope: SessionScope,
        allocation_id: UUID,
        kind: ChildJobKind,
        job_id: UUID,
    ) -> None:
        planning = kind == "PLANNING"
        job = (
            await self._planning_job(scope, job_id)
            if planning
            else await self._reporting_job(scope, job_id)
        )
        if job is None:
            raise SessionError("CHILD_JOB_NOT_FOUND")
        workflow_job_id, generation_job_id = (job_id, None) if planning else (None, job_id)
        link = await self._job_link(scope, allocation_id)
        if link is not None:
            if (link.workflow_job_id, link.generation_job_id) != (
                workflow_job_id,
                generation_job_id,
            ):
                raise SessionError("CHILD_JOB_BINDING_CONFLICT")
            return
        self.session.add(
            OrchestrationChildJobModel(
                organization_id=scope.organization_id,
                orchestration_run_id=scope.orchestration_run_id,
                allocation_id=allocation_id,
                workflow_job_id=workflow_job_id,
                generation_job_id=generation_job_id,
            )
        )
        await self.session.flush()

    async def guard_job(
        self, scope: SessionScope, allocation_id: UUID, claim: JobClaim
    ) -> datetime:
        link = await self._job_link(scope, allocation_id)
        if link is None:
            raise SessionError("WORKFLOW_LEASE_LOST")
        if claim.kind == "PLANNING" and link.workflow_job_id == claim.job_id:
            planning = await self._planning_job(scope, claim.job_id)
            state = (
                (
                    planning.status,
                    planning.locked_by_worker_id,
                    planning.attempt_count,
                    planning.lease_until,
                )
                if planning is not None
                else None
            )
        elif claim.kind == "REPORTING" and link.generation_job_id == claim.job_id:
            reporting = await self._reporting_job(scope, claim.job_id)
            state = (
                (reporting.state, reporting.lease_owner, reporting.fence, reporting.lease_until)
                if reporting is not None
                else None
            )
        else:
            state = None
        if state is None:
            raise SessionError("WORKFLOW_LEASE_LOST")
        status, worker_id, attempt, expires_at = state
        if (
            (status, worker_id, attempt) != ("RUNNING", claim.worker_id, claim.attempt)
            or expires_at is None
            or expires_at <= await self.now()
            or expires_at < claim.lease_until
            # The Reporting heartbeat extends the deadline under the same worker/fence.
            or (claim.kind == "PLANNING" and expires_at != claim.lease_until)
        ):
            raise SessionError("WORKFLOW_LEASE_LOST")
        await self.session.execute(
            text(
                "SELECT set_config('app.orchestration_child_job',:job,true), "
                "set_config('app.orchestration_child_worker',:worker,true), "
                "set_config('app.orchestration_child_attempt',:attempt,true)"
            ),
            {"job": str(claim.job_id), "worker": claim.worker_id, "attempt": str(claim.attempt)},
        )
        active = await self.session.scalar(
            text("SELECT orchestration_child_claim_current(:org,:run,:allocation,:attempt)"),
            {
                "org": scope.organization_id,
                "run": scope.orchestration_run_id,
                "allocation": allocation_id,
                "attempt": claim.attempt,
            },
        )
        if active is not True:
            raise SessionError("EXECUTION_CANCELLED")
        return expires_at

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

    async def update_interval_fence(
        self, scope: SessionScope, operation_id: UUID, fence: int
    ) -> None:
        row = await self._row(scope, operation_id)
        row.fence = fence
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
