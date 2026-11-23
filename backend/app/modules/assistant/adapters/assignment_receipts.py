"""Recover exact, actor-scoped Assignment effects from canonical transactions."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.assignment_tools import AssignmentApplicationService
from app.modules.identity.adapters.database_models import UserModel
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.adapters.database_models import MembershipModel
from app.modules.organization.domain.roles import MembershipRole
from app.modules.work.adapters.database_models import (
    IdempotencyRecordModel,
    IdempotencyState,
    TaskModel,
)
from app.modules.work.planning.assignment.adapters.recommendation_repository import (
    SqlAlchemyRecommendationRepository,
    fingerprint,
)
from app.modules.work.planning.assignment.adapters.repository import (
    SqlAlchemyExplicitAssignmentRepository,
    SqlAlchemyTeamRequirementRepository,
)
from app.modules.work.planning.assignment.application.assignment_service import (
    ExplicitAssignmentCommand,
    ExplicitAssignmentResult,
)
from app.modules.work.planning.assignment.application.recommendation_service import (
    CreateRecommendationCommand,
    ReviseRecommendationCommand,
)
from app.modules.work.planning.assignment.application.requirement_service import (
    DeriveRequirementsCommand,
    TeamRequirementError,
    TeamRequirementStatus,
    canonical_requirement_command,
)
from app.modules.work.planning.assignment.domain.recommendations import (
    CandidateOverride,
    RecommendationError,
    RecommendationVersion,
)
from work_management_ai.agents.assignment.contracts import (
    AssignmentAgentOutput,
    AssignmentOperation,
    ExplicitAssignmentSnapshot,
    ProjectWorkloadSnapshot,
    TeamRecommendationSnapshot,
    TeamRequirementsPendingSnapshot,
    explicit_goal_assignment_key,
)
from work_management_ai.runtime.contracts import (
    AgentHandoffV2,
    AgentId,
    AgentRunStatus,
    ContextReference,
    GoalDirectedAgentResult,
    InteractionOutcome,
    ProposedAction,
    RiskLevel,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError

_TEAM_CREATE = "team_recommendation.CreateRecommendationCommand"
_TEAM_REVISE = "team_recommendation.ReviseRecommendationCommand"
_REQUIREMENTS_DERIVE = "team_requirement.derive"
_EXPLICIT = TypeAdapter(ExplicitAssignmentResult)
_RECOMMENDATION = TypeAdapter(RecommendationVersion)


class _RequirementReceipt(BaseModel):
    id: UUID
    version: int = Field(ge=1)


class PostgreSQLAssignmentReceiptReader:
    def __init__(
        self, *, session_factory: async_sessionmaker[AsyncSession], actor: AuthenticatedActor
    ) -> None:
        self._sessions = session_factory
        self._actor = actor

    @staticmethod
    def _pin(handoff: AgentHandoffV2, resource_type: str) -> ContextReference | None:
        values = tuple(
            ref for ref in handoff.context_references if ref.resource_type.upper() == resource_type
        )
        if len(values) > 1:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        return values[0] if values else None

    @asynccontextmanager
    async def _read(self, handoff: AgentHandoffV2) -> AsyncGenerator[AsyncSession]:
        actor = self._actor
        if (
            handoff.target_agent_id is not AgentId.ASSIGNMENT
            or handoff.actor.organization_id != actor.organization_id
            or handoff.actor.membership_id != actor.membership_id
            or actor.role not in {MembershipRole.ADMIN, MembershipRole.MANAGER}
            or any(
                ref.organization_id != actor.organization_id for ref in handoff.context_references
            )
        ):
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_SCOPE_INVALID")
        async with self._sessions() as session, session.begin():
            await session.execute(text("SET LOCAL ROLE app_runtime"))
            await session.execute(
                text(
                    "SELECT set_config('app.organization_id', :org, true), "
                    "set_config('app.membership_id', :member, true)"
                ),
                {"org": str(actor.organization_id), "member": str(actor.membership_id)},
            )
            active = await session.scalar(
                text("SELECT public.lock_active_membership(:org, :member)"),
                {"org": actor.organization_id, "member": actor.membership_id},
            )
            current_actor = await session.scalar(
                select(MembershipModel.id)
                .join(UserModel, UserModel.id == MembershipModel.user_id)
                .where(
                    MembershipModel.organization_id == actor.organization_id,
                    MembershipModel.id == actor.membership_id,
                    MembershipModel.user_id == actor.user_id,
                    MembershipModel.role == actor.role,
                    MembershipModel.is_active.is_(True),
                    UserModel.is_active.is_(True),
                )
            )
            if active is not True or current_actor is None:
                raise ToolPolicyError("ASSIGNMENT_RECEIPT_SCOPE_INVALID")
            yield session

    async def recover_goal(self, handoff: AgentHandoffV2) -> GoalDirectedAgentResult | None:
        actor = self._actor
        base = handoff.idempotency_key
        task_refs = [
            ref for ref in handoff.context_references if ref.resource_type.upper() == "TASK"
        ]
        member_refs = [
            ref for ref in handoff.context_references if ref.resource_type.upper() == "MEMBERSHIP"
        ]
        assignment_key = (
            explicit_goal_assignment_key(handoff.source_turn_ref)
            if len(task_refs) == len(member_refs) == 1
            else None
        )
        keys = (
            f"{base}:assignment.manage_team",
            f"{base}:assignment.manage_team:derive-requirements",
        )
        if assignment_key is not None:
            keys += (assignment_key,)
        async with self._read(handoff) as session:
            records = (
                await session.scalars(
                    select(IdempotencyRecordModel).where(
                        IdempotencyRecordModel.organization_id == actor.organization_id,
                        IdempotencyRecordModel.actor_membership_id == actor.membership_id,
                        IdempotencyRecordModel.state == IdempotencyState.COMPLETED,
                        IdempotencyRecordModel.expires_at > func.now(),
                        IdempotencyRecordModel.idempotency_key.in_(keys),
                    )
                )
            ).all()
            if not records:
                return None
            if len(records) != 1:
                raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
            record = records[0]
            try:
                if record.idempotency_key == assignment_key:
                    snapshot = await self._assignment(session, handoff, record)
                    return self._result(
                        handoff,
                        AssignmentOperation.ASSIGN_TASK_EXPLICITLY,
                        "assignment.assign_task_explicitly",
                        snapshot,
                    )
                if record.idempotency_key == f"{base}:assignment.manage_team:derive-requirements":
                    snapshot = await self._requirements(session, handoff, record)
                    return self._result(
                        handoff,
                        AssignmentOperation.RECOMMEND_TEAM,
                        "assignment.recommend_team",
                        snapshot,
                    )
                recommendation, operation, action = await self._recommendation(
                    session, handoff, record
                )
                return self._result(handoff, operation, action, recommendation)
            except (ValidationError, TeamRequirementError, RecommendationError) as error:
                raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID") from error

    async def _assignment(
        self, session: AsyncSession, handoff: AgentHandoffV2, record: IdempotencyRecordModel
    ) -> ExplicitAssignmentSnapshot:
        task, member, project = (
            self._pin(handoff, "TASK"),
            self._pin(handoff, "MEMBERSHIP"),
            self._pin(handoff, "PROJECT"),
        )
        if task is None or task.version is None or member is None:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        command = ExplicitAssignmentCommand(
            actor=self._actor,
            task_id=task.resource_id,
            expected_task_version=task.version,
            assignee_membership_id=member.resource_id,
            request_id=f"assistant:{record.idempotency_key}",
            idempotency_key=record.idempotency_key,
            once_per_turn=True,
        )
        saved = _EXPLICIT.validate_python(record.response_body)
        if (
            record.operation != command.idempotency_operation
            or record.request_fingerprint
            != SqlAlchemyExplicitAssignmentRepository.fingerprint(command)
            or saved.task.organization_id != self._actor.organization_id
            or saved.task.id != task.resource_id
            or saved.task.version != task.version + 1
            or saved.task.assignee_membership_id != member.resource_id
            or (project is not None and saved.task.project_id != project.resource_id)
        ):
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        visible = await session.scalar(
            select(TaskModel.id)
            .join(
                MembershipModel,
                (MembershipModel.organization_id == TaskModel.organization_id)
                & (MembershipModel.id == TaskModel.assignee_membership_id),
            )
            .join(UserModel, UserModel.id == MembershipModel.user_id)
            .where(
                TaskModel.organization_id == self._actor.organization_id,
                TaskModel.id == saved.task.id,
                TaskModel.project_id == saved.task.project_id,
                TaskModel.version == saved.task.version,
                TaskModel.assignee_membership_id == member.resource_id,
                MembershipModel.is_active.is_(True),
                UserModel.is_active.is_(True),
            )
        )
        if visible is None:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_STALE")
        return AssignmentApplicationService.assignment_snapshot(
            saved, observed_at=saved.task.updated_at
        )

    async def _requirements(
        self, session: AsyncSession, handoff: AgentHandoffV2, record: IdempotencyRecordModel
    ) -> TeamRequirementsPendingSnapshot:
        project = self._pin(handoff, "PROJECT")
        if project is None or record.operation != _REQUIREMENTS_DERIVE:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        command = DeriveRequirementsCommand(
            actor=self._actor,
            project_id=project.resource_id,
            request_id=f"assistant:{record.idempotency_key}",
            idempotency_key=record.idempotency_key,
        )
        if record.request_fingerprint != canonical_requirement_command(command):
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        receipt = _RequirementReceipt.model_validate(record.response_body)
        requirements = await SqlAlchemyTeamRequirementRepository(session).read_operation_receipt(
            self._actor, project_id=project.resource_id, set_id=receipt.id, version=receipt.version
        )
        if requirements is None:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        reasons = tuple(sorted({item.reason for item in requirements.incomplete_items}))
        return TeamRequirementsPendingSnapshot(
            organization_id=self._actor.organization_id,
            project_id=project.resource_id,
            requirement_set_id=requirements.id,
            requirement_version=requirements.version,
            reason_codes=reasons or ("MANAGER_CONFIRMATION_REQUIRED",),
            observed_at=record.created_at,
        )

    async def _recommendation(
        self, session: AsyncSession, handoff: AgentHandoffV2, record: IdempotencyRecordModel
    ) -> tuple[TeamRecommendationSnapshot, AssignmentOperation, str]:
        project = self._pin(handoff, "PROJECT")
        if project is None:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        saved = _RECOMMENDATION.validate_python(record.response_body)
        command: CreateRecommendationCommand | ReviseRecommendationCommand
        if record.operation == _TEAM_CREATE and saved.version == 1:
            operation, action = AssignmentOperation.RECOMMEND_TEAM, "assignment.recommend_team"
            command = CreateRecommendationCommand(
                actor=self._actor,
                request_id=f"assistant:{record.idempotency_key}",
                idempotency_key=record.idempotency_key,
                project_id=project.resource_id,
                requirement_set_id=saved.requirement_set_id,
                requirement_version=saved.requirement_version,
                policy_version=saved.policy_version,
            )
        elif record.operation == _TEAM_REVISE:
            recommendation = self._pin(handoff, "TEAM_RECOMMENDATION")
            if (
                recommendation is None
                or recommendation.version is None
                or recommendation.resource_id != saved.recommendation_id
                or recommendation.version + 1 != saved.version
            ):
                raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
            operation, action = AssignmentOperation.REVISE_TEAM, "assignment.revise_team"
            command = ReviseRecommendationCommand(
                actor=self._actor,
                request_id=f"assistant:{record.idempotency_key}",
                idempotency_key=record.idempotency_key,
                recommendation_id=saved.recommendation_id,
                expected_version=recommendation.version,
                expected_project_id=project.resource_id,
                overrides=tuple(
                    CandidateOverride(
                        item.requirement_id,
                        item.membership_id,
                        item.override_reason,
                        item.allocated_effort_hours,
                    )
                    for item in saved.selections
                ),
            )
        else:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        if record.request_fingerprint != fingerprint(command):
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        canonical = await SqlAlchemyRecommendationRepository(session).read_operation_receipt(
            self._actor,
            project_id=project.resource_id,
            recommendation_id=saved.recommendation_id,
            version=saved.version,
        )
        if canonical != replace(saved, status=canonical.status):
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        return (
            AssignmentApplicationService.recommendation_snapshot(
                organization_id=self._actor.organization_id,
                project_id=project.resource_id,
                version=canonical,
                observed_at=record.created_at,
            ),
            operation,
            action,
        )

    @staticmethod
    def _result(
        handoff: AgentHandoffV2,
        operation: AssignmentOperation,
        action: str,
        snapshot: ExplicitAssignmentSnapshot
        | TeamRecommendationSnapshot
        | TeamRequirementsPendingSnapshot,
    ) -> GoalDirectedAgentResult:
        proposed_actions: tuple[ProposedAction, ...] = ()
        status, outcome = AgentRunStatus.COMPLETED, InteractionOutcome.RESULT_READY
        if isinstance(snapshot, TeamRecommendationSnapshot):
            kind, resource_id, version = (
                "TEAM_RECOMMENDATION",
                snapshot.recommendation_id,
                snapshot.version,
            )
            evidence_id = f"team-recommendation:{resource_id}:v{version}"
            if snapshot.status == "PROPOSED":
                status, outcome = (
                    AgentRunStatus.AWAITING_HUMAN,
                    InteractionOutcome.ACTION_REVIEW_REQUIRED,
                )
                proposed_actions = (
                    ProposedAction(
                        action_type="assignment.team_proposal_review",
                        risk=RiskLevel.PROPOSAL_ONLY,
                        requires_human_gate=True,
                        reference_id=resource_id,
                    ),
                )
        elif isinstance(snapshot, TeamRequirementsPendingSnapshot):
            kind, resource_id, version = (
                "TEAM_REQUIREMENTS",
                snapshot.requirement_set_id,
                snapshot.requirement_version,
            )
            evidence_id = f"team-requirements:{resource_id}:v{version}"
            outcome = InteractionOutcome.DRAFT_READY
        else:
            kind, resource_id, version = "TASK", snapshot.task_id, snapshot.task_version
            evidence_id = f"task-assignment:{resource_id}:v{version}"
        return GoalDirectedAgentResult(
            agent_id=AgentId.ASSIGNMENT,
            agent_version=handoff.target_agent_version,
            status=status,
            interaction_outcome=outcome,
            resolved_action=action,
            typed_output=AssignmentAgentOutput(
                operation=operation,
                deterministic_result=snapshot.model_dump(mode="json"),
                explanation_status="UNAVAILABLE"
                if isinstance(snapshot, TeamRecommendationSnapshot)
                else "NOT_REQUESTED",
            ).model_dump(mode="json"),
            proposed_actions=proposed_actions,
            evidence=(
                ContextReference(
                    reference_id=uuid5(NAMESPACE_URL, evidence_id),
                    organization_id=snapshot.organization_id,
                    resource_type=kind,
                    resource_id=resource_id,
                    version=version,
                    observed_at=snapshot.observed_at,
                ),
            ),
            stop_reason="ASSIGNMENT_EFFECT_RECOVERED",
        )

    async def _verify_existing_requirements(
        self, handoff: AgentHandoffV2, draft: TeamRequirementsPendingSnapshot
    ) -> None:
        async with self._read(handoff) as session:
            requirements = await SqlAlchemyTeamRequirementRepository(
                session
            ).read_operation_receipt(
                self._actor,
                project_id=draft.project_id,
                set_id=draft.requirement_set_id,
                version=draft.requirement_version,
            )
        if requirements is None:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_MISSING")
        reasons = tuple(sorted({item.reason for item in requirements.incomplete_items}))
        if not reasons and requirements.status is not TeamRequirementStatus.CONFIRMED:
            reasons = ("MANAGER_CONFIRMATION_REQUIRED",)
        if draft.reason_codes != reasons:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")

    async def verify_goal(
        self, handoff: AgentHandoffV2, saved: GoalDirectedAgentResult
    ) -> GoalDirectedAgentResult:
        canonical = await self.recover_goal(handoff)
        if canonical is None:
            if saved.interaction_outcome in {
                InteractionOutcome.FAILED,
                InteractionOutcome.NEEDS_INPUT,
            }:
                return saved
            # Existing requirements may already need review without writing anything.
            try:
                data = AssignmentAgentOutput.model_validate(saved.typed_output)
                project = self._pin(handoff, "PROJECT")
                if saved.proposed_actions or any(
                    ref.organization_id != self._actor.organization_id for ref in saved.evidence
                ):
                    raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
                if (
                    saved.resolved_action == "assignment.analyze_workload"
                    and data.operation is AssignmentOperation.ANALYZE_WORKLOAD
                    and saved.interaction_outcome is InteractionOutcome.RESULT_READY
                ):
                    snapshot = ProjectWorkloadSnapshot.model_validate(data.deterministic_result)
                    if (
                        snapshot.organization_id == self._actor.organization_id
                        and project is not None
                        and snapshot.project_id == project.resource_id
                    ):
                        return saved
                if (
                    saved.resolved_action == "assignment.recommend_team"
                    and data.operation is AssignmentOperation.RECOMMEND_TEAM
                    and saved.interaction_outcome is InteractionOutcome.DRAFT_READY
                ):
                    draft = TeamRequirementsPendingSnapshot.model_validate(
                        data.deterministic_result
                    )
                    if (
                        draft.organization_id == self._actor.organization_id
                        and project is not None
                        and draft.project_id == project.resource_id
                    ):
                        await self._verify_existing_requirements(handoff, draft)
                        return saved
            except (ValidationError, TeamRequirementError) as error:
                raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID") from error
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_MISSING")
        if saved.interaction_outcome in {InteractionOutcome.FAILED, InteractionOutcome.NEEDS_INPUT}:
            return canonical
        if saved.resolved_action != canonical.resolved_action:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        try:
            claimed = AssignmentAgentOutput.model_validate(saved.typed_output)
            actual = AssignmentAgentOutput.model_validate(canonical.typed_output)
        except ValidationError as error:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID") from error
        claimed_result = claimed.deterministic_result.copy()
        actual_result = actual.deterministic_result.copy()
        for data in (claimed_result, actual_result):
            data.pop("observed_at", None)
            if isinstance(actual_result.get("recommendation_id"), str):
                data.pop("status", None)
        if claimed.operation is not actual.operation or claimed_result != actual_result:
            raise ToolPolicyError("ASSIGNMENT_RECEIPT_BINDING_INVALID")
        return canonical.model_copy(
            update={
                "typed_output": {
                    **actual.model_dump(mode="json"),
                    "explanation": claimed.model_dump(mode="json")["explanation"],
                    "explanation_status": claimed.explanation_status,
                },
                "iterations_used": saved.iterations_used,
                "model_attempts_used": saved.model_attempts_used,
                "tool_calls_used": saved.tool_calls_used,
                "verifier_results": saved.verifier_results,
            }
        )
