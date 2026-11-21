"""Strict model-facing input/output contracts for candidate Specialist delegation."""

from datetime import date
from typing import Literal, Self
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from work_management_ai.runtime.contracts import (
    AgentId,
    AgentRunStatus,
    ContextReference,
    InteractionOutcome,
    JsonValue,
    ProposedAction,
    RequestedHandoff,
)


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("DATE_ISO_INVALID") from exc
    if parsed.isoformat() != value:
        raise ValueError("DATE_ISO_INVALID")
    return value


def _iana_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("TIMEZONE_INVALID") from exc
    return value


# --- Capability-specific typed business inputs (no server authority fields) ---------------


class WorkQuestionDelegationInput(_StrictContract):
    capability: Literal["work.answer_question"] = Field(
        description="Capability discriminator for answering grounded work questions."
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's work question to answer from authorized Projects and Tasks.",
    )

    locale: Literal["vi", "en"] = Field(default="vi", description="Language of the result.")


class WorkReadProjectDelegationInput(_StrictContract):
    capability: Literal["work.read_project"] = Field(
        description="Capability discriminator for reading a detailed authorized Project view."
    )
    project_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Resolved Project UUID or exact search reference.",
    )

    locale: Literal["vi", "en"] = Field(default="vi", description="Language of the result.")


class PlanningCreateDelegationInput(_StrictContract):
    capability: Literal["planning.create"] = Field(
        description="Capability discriminator for creating a new Project plan draft or proposal."
    )
    goal: str = Field(
        min_length=1,
        max_length=2000,
        description="Business goal and scope for the new Project plan.",
    )
    anchor_start_date: str | None = Field(
        default=None,
        description="Optional ISO start date (YYYY-MM-DD) stated by the user.",
    )
    anchor_end_date: str | None = Field(
        default=None,
        description="Optional ISO target end date (YYYY-MM-DD) stated by the user.",
    )
    duration_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        strict=True,
        description="Optional requested duration in days when explicitly supplied by the user.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the planning draft or clarification.",
    )

    @field_validator("anchor_start_date", "anchor_end_date")
    @classmethod
    def valid_dates(cls, value: str | None) -> str | None:
        return _iso_date(value)

    @model_validator(mode="after")
    def valid_date_order(self) -> Self:
        if (
            self.anchor_start_date is not None
            and self.anchor_end_date is not None
            and self.anchor_start_date > self.anchor_end_date
        ):
            raise ValueError("PLANNING_ANCHOR_DATE_RANGE_INVALID")
        return self


class PlanningResumeDelegationInput(_StrictContract):
    capability: Literal["planning.resume"] = Field(
        description="Capability discriminator for continuing an active Planning workflow."
    )
    workflow_run_id: UUID = Field(
        description="Identifier of the existing Planning workflow run to continue."
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="New user clarification or adjustment for the active Planning workflow.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the planning response.",
    )


class PlanningReviseDelegationInput(_StrictContract):
    capability: Literal["planning.revise"] = Field(
        description="Capability discriminator for revising one exact Planning proposal version."
    )
    proposal_id: UUID = Field(description="Identifier of the immutable Planning proposal.")
    expected_version: int = Field(
        ge=1,
        strict=True,
        description="Exact current version of the Planning proposal to revise.",
    )
    instruction: str = Field(
        min_length=1,
        max_length=2000,
        description="Requested changes to apply in the new proposal revision.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the revised proposal.",
    )


class PlanningExplainDelegationInput(_StrictContract):
    capability: Literal["planning.explain"] = Field(
        description="Capability discriminator for explaining an existing Planning draft or run."
    )
    workflow_run_id: UUID | None = Field(
        default=None,
        description="Optional Planning workflow run identifier to explain.",
    )
    draft_id: UUID | None = Field(
        default=None,
        description="Optional Planning draft identifier to explain.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the explanation.",
    )


class AssignmentAnalyzeWorkloadDelegationInput(_StrictContract):
    capability: Literal["assignment.analyze_workload"] = Field(
        description="Capability discriminator for deterministic Project workload analysis."
    )
    project_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Resolved Project UUID or verified reference whose workload is analyzed.",
    )

    locale: Literal["vi", "en"] = Field(default="vi", description="Language of the result.")


class AssignmentRecommendTeamDelegationInput(_StrictContract):
    capability: Literal["assignment.recommend_team"] = Field(
        description="Capability discriminator for creating a Project Team recommendation proposal."
    )
    project_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Resolved Project UUID or verified reference for team staffing.",
    )
    instruction: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional staffing preference or constraint stated by the Manager.",
    )


class AssignmentReviseTeamDelegationInput(_StrictContract):
    capability: Literal["assignment.revise_team"] = Field(
        description="Capability discriminator for revising an existing Team recommendation."
    )
    project_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Resolved Project UUID or verified reference for the recommendation.",
    )
    recommendation_id: UUID = Field(
        description="Identifier of the existing Team recommendation to revise."
    )
    expected_version: int = Field(
        ge=1,
        strict=True,
        description="Exact current version of the Team recommendation.",
    )
    instruction: str = Field(
        min_length=1,
        max_length=1000,
        description="Requested staffing changes for the new recommendation revision.",
    )


class AssignmentExplicitTaskDelegationInput(_StrictContract):
    capability: Literal["assignment.assign_task_explicitly"] = Field(
        description="Capability discriminator for an explicit Manager task assignment command."
    )
    task_id: UUID = Field(description="Exact Task UUID to assign.")
    assignee_membership_id: UUID = Field(
        description="Exact organization membership UUID of the target assignee."
    )
    expected_task_version: int = Field(
        ge=1,
        strict=True,
        description="Exact current version of the Task for optimistic concurrency.",
    )


class DailyUpdatePrepareDelegationInput(_StrictContract):
    capability: Literal["daily_update.prepare"] = Field(
        description="Capability discriminator for extracting an owner-reviewed Daily Update draft."
    )
    task_reference: UUID = Field(description="Exact UUID of the owner-visible Task.")
    task_version: int = Field(
        ge=1, strict=True, description="Exact version of the owner-visible Task."
    )
    locale: Literal["vi", "en"] = Field(default="vi", description="Language of the result.")
    update_text: str = Field(
        min_length=1,
        max_length=4000,
        description="Raw progress report supplied by the Task owner.",
    )
    draft_id: UUID | None = Field(
        default=None,
        description="Optional existing Daily Update draft UUID when revising a draft.",
    )
    draft_version: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Exact current version of the draft when draft_id is provided.",
    )


class RiskExplainDelegationInput(_StrictContract):
    capability: Literal["risk.explain"] = Field(
        description="Capability discriminator for explaining persisted verified Task risk."
    )
    task_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Task UUID or title reference whose persisted risk should be explained.",
    )
    expected_fingerprint: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional verified risk context fingerprint when already observed.",
    )

    locale: Literal["vi", "en"] = Field(default="vi", description="Language of the result.")


class ReportingPrepareDelegationInput(_StrictContract):
    capability: Literal["reporting.prepare_report"] = Field(
        description="Capability discriminator for preparing a grounded management report."
    )
    project_reference: str = Field(
        min_length=1,
        max_length=200,
        description="Resolved Project UUID or verified search reference to report on.",
    )
    report_kind: Literal["weekly_status", "executive_summary", "project_status"] = Field(
        default="weekly_status",
        description="Kind of management report requested by the Manager.",
    )
    period_start: str | None = Field(
        default=None,
        description="Optional inclusive ISO start date (YYYY-MM-DD) for the reporting window.",
    )
    period_end: str | None = Field(
        default=None,
        description="Optional inclusive ISO end date (YYYY-MM-DD) for the reporting window.",
    )
    timezone: str = Field(
        default="UTC",
        min_length=1,
        max_length=64,
        description="Valid IANA timezone name (e.g. Asia/Ho_Chi_Minh, UTC) for the report window.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the management report.",
    )

    @field_validator("period_start", "period_end")
    @classmethod
    def valid_dates(cls, value: str | None) -> str | None:
        return _iso_date(value)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        return _iana_timezone(value)

    @model_validator(mode="after")
    def valid_period_order(self) -> Self:
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_start > self.period_end
        ):
            raise ValueError("REPORTING_PERIOD_RANGE_INVALID")
        return self


class ReportingExplainSnapshotDelegationInput(_StrictContract):
    capability: Literal["reporting.explain_snapshot"] = Field(
        description="Capability discriminator for explaining an immutable verified report snapshot."
    )
    snapshot_id: UUID = Field(description="Exact immutable report snapshot UUID.")
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Specific question about metrics or status in the verified snapshot.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the explanation.",
    )


class RetrieveWebDelegationInput(_StrictContract):
    capability: Literal["retrieval.search_web", "retrieval.summarize_web"] = Field(
        description="Capability discriminator for read-only public web search or summarization."
    )
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Public web research query requested by the user.",
    )
    locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Output language locale for the grounded web summary.",
    )


type DelegationBranchInput = (
    WorkQuestionDelegationInput
    | WorkReadProjectDelegationInput
    | PlanningCreateDelegationInput
    | PlanningResumeDelegationInput
    | PlanningReviseDelegationInput
    | PlanningExplainDelegationInput
    | AssignmentAnalyzeWorkloadDelegationInput
    | AssignmentRecommendTeamDelegationInput
    | AssignmentReviseTeamDelegationInput
    | AssignmentExplicitTaskDelegationInput
    | DailyUpdatePrepareDelegationInput
    | RiskExplainDelegationInput
    | ReportingPrepareDelegationInput
    | ReportingExplainSnapshotDelegationInput
    | RetrieveWebDelegationInput
)

AGENT_CAPABILITY_PAIRS: frozenset[tuple[AgentId, str]] = frozenset(
    {
        (AgentId.WORK_INTELLIGENCE, "work.answer_question"),
        (AgentId.WORK_INTELLIGENCE, "work.read_project"),
        (AgentId.PLANNING, "planning.create"),
        (AgentId.PLANNING, "planning.resume"),
        (AgentId.PLANNING, "planning.revise"),
        (AgentId.PLANNING, "planning.explain"),
        (AgentId.ASSIGNMENT, "assignment.analyze_workload"),
        (AgentId.ASSIGNMENT, "assignment.recommend_team"),
        (AgentId.ASSIGNMENT, "assignment.revise_team"),
        (AgentId.ASSIGNMENT, "assignment.assign_task_explicitly"),
        (AgentId.DAILY_UPDATE, "daily_update.prepare"),
        (AgentId.RISK, "risk.explain"),
        (AgentId.REPORTING, "reporting.prepare_report"),
        (AgentId.REPORTING, "reporting.explain_snapshot"),
        (AgentId.RETRIEVE, "retrieval.search_web"),
        (AgentId.RETRIEVE, "retrieval.summarize_web"),
    }
)


class DelegateAgentInput(_StrictContract):
    agent_id: AgentId = Field(
        description="Activated target Specialist agent identifier; never orchestrator."
    )
    capability: str = Field(
        min_length=1,
        max_length=100,
        description="Exact active capability boundary owned by the target Specialist.",
    )
    objective: str = Field(
        min_length=1,
        max_length=4000,
        description="Clear goal for the Specialist to accomplish within the user's request.",
    )
    context_ref_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=16,
        description=(
            "Authorized ContextRef UUIDs from prior exploration reads; never grants rights."
        ),
    )
    constraints: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description=(
            "Task constraints for the Specialist; cannot relax permissions or approval gates."
        ),
    )
    expected_result: str = Field(
        min_length=1,
        max_length=500,
        description="Short description of the outcome expected back from the Specialist.",
    )
    input: DelegationBranchInput = Field(
        description=(
            "Capability-specific typed business parameters matching the selected capability."
        ),
    )

    @field_validator("objective", "expected_result")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DELEGATION_TEXT_EMPTY")
        return value

    @field_validator("constraints")
    @classmethod
    def valid_constraints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or len(item) > 500 for item in values):
            raise ValueError("DELEGATION_CONSTRAINT_INVALID")
        return values

    @field_validator("context_ref_ids")
    @classmethod
    def unique_context_refs(cls, values: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if len(set(values)) != len(values):
            raise ValueError("DELEGATION_DUPLICATE_CONTEXT_REF")
        return values

    @model_validator(mode="after")
    def valid_target_and_branch(self) -> Self:
        if self.agent_id is AgentId.ORCHESTRATOR:
            raise ValueError("DELEGATION_SELF_FORBIDDEN")
        if self.input.capability != self.capability:
            raise ValueError("DELEGATION_CAPABILITY_MISMATCH")
        if (self.agent_id, self.capability) not in AGENT_CAPABILITY_PAIRS:
            raise ValueError("DELEGATION_CAPABILITY_MISMATCH")
        return self


class DelegateAgentInputV2(_StrictContract):
    agent_id: AgentId = Field(description="Target Specialist from the offered list.")
    objective: str = Field(
        min_length=1,
        max_length=4000,
        description="Goal for the Specialist to accomplish within the user's request.",
    )
    context_ref_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=16,
        description="Verified context reference IDs from this run. References never grant rights.",
    )
    constraints: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="User and task constraints within existing permission and approval gates.",
    )
    expected_result: str = Field(
        min_length=1,
        max_length=500,
        description="Short description of the expected result.",
    )

    @field_validator("objective", "expected_result")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        return DelegateAgentInput.non_blank_text(value)

    @field_validator("constraints")
    @classmethod
    def valid_constraints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return DelegateAgentInput.valid_constraints(values)

    @field_validator("context_ref_ids")
    @classmethod
    def unique_context_refs(cls, values: tuple[UUID, ...]) -> tuple[UUID, ...]:
        return DelegateAgentInput.unique_context_refs(values)

    @model_validator(mode="after")
    def valid_target(self) -> Self:
        if self.agent_id == AgentId.ORCHESTRATOR:
            raise ValueError("DELEGATION_SELF_FORBIDDEN")
        return self


class _DelegateAgentOutputFields(_StrictContract):
    agent_id: AgentId
    agent_version: str = Field(min_length=1, max_length=64)
    child_agent_run_id: UUID
    status: AgentRunStatus
    interaction_outcome: InteractionOutcome | None = None
    summary: str = Field(default="", max_length=4000)
    typed_output: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_refs: tuple[ContextReference, ...] = ()
    evidence: tuple[ContextReference, ...] = ()
    proposed_actions: tuple[ProposedAction, ...] = ()
    requested_handoff: RequestedHandoff | None = None
    workflow_run_id: UUID | None = None
    draft_id: UUID | None = None
    draft_version: int | None = Field(default=None, ge=1, strict=True)
    proposal_id: UUID | None = None
    proposal_version: int | None = Field(default=None, ge=1, strict=True)
    stop_reason: str = Field(min_length=1, max_length=100)
    safe_error_code: str | None = Field(default=None, min_length=1, max_length=100)
    usage: dict[str, int] = Field(default_factory=dict)


class DelegateAgentOutput(_DelegateAgentOutputFields):
    """Normalized observation returned to the Orchestrator after Specialist execution."""

    capability: str = Field(min_length=1, max_length=100)


class DelegateAgentOutputV2(_DelegateAgentOutputFields):
    """Normalized observation from a goal-directed Specialist."""

    resolved_action: str | None = Field(default=None, min_length=1, max_length=100)
