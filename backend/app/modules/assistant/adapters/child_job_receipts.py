"""Read canonical job bindings after a parent or Specialist interruption."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.database_models import AgentRunModel
from app.modules.assistant.adapters.session_usage_models import OrchestrationChildJobModel
from app.modules.assistant.adapters.session_usage_repository import (
    PostgreSQLUsageTransactionFactory,
)
from app.modules.assistant.adapters.tool_loop_recorder import session_scope
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.planning_runs.adapters.database_models import WorkflowJobModel, WorkflowRunModel
from app.modules.reporting.adapters.database_models import ReportModel
from app.modules.reporting.adapters.usage_models import ReportGenerationJobModel
from work_management_ai.runtime.delegation import CanonicalWorkflowState, ChildJobReceipt
from work_management_ai.runtime.tool_loop_contracts import RunScope, VersionPin

_PLANNING_ACTIONS = {
    "planning.start": "planning.create",
    "planning.resume": "planning.resume",
    "proposal.ai_revise": "planning.revise",
}


class PostgreSQLChildJobReceipts:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        actor: AuthenticatedActor,
    ) -> None:
        self._transactions = PostgreSQLUsageTransactionFactory(sessions)
        self._actor = actor

    async def recover_job(
        self,
        scope: RunScope,
        *,
        allocation_id: UUID,
        child_run_id: UUID,
        target: VersionPin,
    ) -> ChildJobReceipt | None:
        async with self._transactions(self._actor) as transaction:
            repo = transaction.repository
            if not await repo.authorize(session_scope(scope), self._actor):
                return None
            session = repo.session
            matching_child = await session.scalar(
                select(AgentRunModel.id).where(
                    AgentRunModel.organization_id == scope.organization_id,
                    AgentRunModel.orchestration_run_id == scope.orchestration_run_id,
                    AgentRunModel.id == child_run_id,
                    AgentRunModel.agent_id == target.reference,
                    AgentRunModel.agent_version == target.version,
                    AgentRunModel.manifest_fingerprint == target.fingerprint,
                )
            )
            if matching_child is None:
                return None
            binding = await session.scalar(
                select(OrchestrationChildJobModel).where(
                    OrchestrationChildJobModel.organization_id == scope.organization_id,
                    OrchestrationChildJobModel.orchestration_run_id == scope.orchestration_run_id,
                    OrchestrationChildJobModel.allocation_id == allocation_id,
                )
            )
            if binding is None:
                return None
            if target.reference == "reporting" and binding.generation_job_id is not None:
                return ChildJobReceipt(
                    resource_id=binding.generation_job_id,
                    resolved_action="reporting.prepare_report",
                )
            if target.reference != "planning" or binding.workflow_job_id is None:
                return None
            row = (
                await session.execute(
                    select(WorkflowJobModel.workflow_run_id, WorkflowJobModel.job_type)
                    .join(
                        WorkflowRunModel,
                        (WorkflowRunModel.organization_id == WorkflowJobModel.organization_id)
                        & (WorkflowRunModel.id == WorkflowJobModel.workflow_run_id),
                    )
                    .where(
                        WorkflowJobModel.organization_id == scope.organization_id,
                        WorkflowJobModel.id == binding.workflow_job_id,
                        WorkflowRunModel.requested_by_membership_id == scope.membership_id,
                    )
                )
            ).one_or_none()
            if row is None or row.job_type not in _PLANNING_ACTIONS:
                return None
            return ChildJobReceipt.model_validate(
                {
                    "resource_id": row.workflow_run_id,
                    "resolved_action": _PLANNING_ACTIONS[row.job_type],
                }
            )

    async def verify_reporting_job(
        self,
        scope: RunScope,
        generation_id: UUID,
    ) -> CanonicalWorkflowState | None:
        async with self._transactions(self._actor) as transaction:
            repo = transaction.repository
            if not await repo.authorize(session_scope(scope), self._actor):
                return None
            row = (
                await repo.session.execute(
                    select(ReportGenerationJobModel, ReportModel.version)
                    .join(
                        OrchestrationChildJobModel,
                        (
                            OrchestrationChildJobModel.organization_id
                            == ReportGenerationJobModel.organization_id
                        )
                        & (
                            OrchestrationChildJobModel.generation_job_id
                            == ReportGenerationJobModel.id
                        ),
                    )
                    .join(
                        ReportModel,
                        (ReportModel.organization_id == ReportGenerationJobModel.organization_id)
                        & (ReportModel.id == ReportGenerationJobModel.report_id),
                    )
                    .where(
                        ReportGenerationJobModel.organization_id == scope.organization_id,
                        ReportGenerationJobModel.requester_membership_id == scope.membership_id,
                        ReportGenerationJobModel.id == generation_id,
                        OrchestrationChildJobModel.orchestration_run_id
                        == scope.orchestration_run_id,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        job, version = row
        if job.state in {"QUEUED", "RUNNING"}:
            status = "QUEUED" if job.state == "QUEUED" else "RUNNING"
            return CanonicalWorkflowState(
                workflow_run_id=generation_id,
                status=status,
                version=job.expected_report_version,
            )
        if job.state == "AWAITING_REVIEW" and job.proposed_version_id is not None:
            return CanonicalWorkflowState(
                workflow_run_id=generation_id,
                status="WAITING_FOR_DECISION",
                version=version,
                proposal_id=job.proposed_version_id,
                proposal_version=version,
            )
        return CanonicalWorkflowState(
            workflow_run_id=generation_id, status="FAILED", version=version
        )
