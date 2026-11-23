"""Native Orchestrator output, retaining the four existing public statuses."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
from work_management_ai.runtime.contracts import ActivityResponseBlock, ResponseBlock
from work_management_ai.runtime.tool_loop_contracts import WaitReference


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class ModelCitation(_Contract):
    """Strict model-facing citation; runtime verifies every reference against observed evidence."""

    evidence_id: str = Field(
        min_length=1,
        max_length=128,
        description="Exact observed evidence or ContextRef identifier.",
    )
    resource_type: str = Field(
        min_length=1,
        max_length=100,
        description="Observed resource type (for example project, task, or actor).",
    )
    resource_id: UUID = Field(description="Exact observed resource UUID.")
    version: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Exact observed resource version when present, or null.",
    )


class ModelFinalBlock(_Contract):
    """Strict-schema-compatible final block projected by runtime into public ResponseBlocks."""

    kind: Literal["text", "work_evidence", "question", "capability_unavailable"] = Field(
        description="Public block kind to emit after runtime verification."
    )
    text: str | None = Field(
        default=None,
        max_length=20_000,
        description=(
            "Visible answer text, evidence summary, or clarification question. "
            "Questions may contain at most 4,000 characters."
        ),
    )
    capability: str | None = Field(
        default=None,
        max_length=100,
        description="Unavailable capability identifier when kind is capability_unavailable.",
    )
    message_key: str | None = Field(
        default=None,
        max_length=200,
        description="Localized message key when kind is capability_unavailable.",
    )
    citations: tuple[ModelCitation, ...] = Field(
        default=(),
        max_length=16,
        description="Observed evidence citations when kind is work_evidence.",
    )

    @model_validator(mode="after")
    def valid_block_shape(self) -> Self:
        if self.kind in {"text", "question"}:
            if not self.text or not self.text.strip():
                raise ValueError("FINAL_BLOCK_TEXT_REQUIRED")
            if self.kind == "question" and len(self.text) > 4_000:
                raise ValueError("FINAL_QUESTION_TOO_LONG")
            if self.citations or self.capability is not None or self.message_key is not None:
                raise ValueError("FINAL_BLOCK_FIELDS_UNSUPPORTED")
        elif self.kind == "work_evidence":
            if not self.text or not self.text.strip() or not self.citations:
                raise ValueError("FINAL_EVIDENCE_BLOCK_INCOMPLETE")
            if self.capability is not None or self.message_key is not None:
                raise ValueError("FINAL_BLOCK_FIELDS_UNSUPPORTED")
        elif self.kind == "capability_unavailable":
            if (
                not self.capability
                or not self.capability.strip()
                or not self.message_key
                or not self.message_key.strip()
            ):
                raise ValueError("FINAL_UNAVAILABLE_BLOCK_INCOMPLETE")
            if self.citations:
                raise ValueError("FINAL_BLOCK_FIELDS_UNSUPPORTED")
        return self


class ModelFinalResponse(_Contract):
    """Bounded model-facing final schema compatible with strict provider JSON Schema."""

    outcome: Literal["ANSWER", "NEEDS_INPUT", "CAPABILITY_UNAVAILABLE"] = Field(
        description=(
            "ANSWER includes a text or work_evidence block. NEEDS_INPUT requires a question "
            "block. CAPABILITY_UNAVAILABLE requires a capability_unavailable block."
        )
    )
    summary: str = Field(
        min_length=1,
        max_length=4_000,
        description="Short visible summary matching the emitted blocks.",
    )
    blocks: tuple[ModelFinalBlock, ...] = Field(
        min_length=1,
        max_length=8,
        description="Ordered visible response blocks for the user.",
    )
    artifact_state: Literal["NONE", "DRAFT_AVAILABLE", "PROPOSAL_PENDING", "APPLIED"] = Field(
        description=(
            "Observed artifact lifecycle state: NONE, DRAFT_AVAILABLE, PROPOSAL_PENDING, "
            "or APPLIED. Never claim APPLIED unless a verified write actually committed."
        )
    )
    stop_reason: str = Field(
        min_length=1,
        max_length=100,
        description="Short machine-readable stop reason.",
    )

    @model_validator(mode="after")
    def non_blank_summary(self) -> Self:
        if not self.summary.strip() or not self.stop_reason.strip():
            raise ValueError("FINAL_RESPONSE_TEXT_EMPTY")
        return self


class ObservedArtifactState(_Contract):
    """Runtime-observed artifact states from canonical completed tool outputs only."""

    draft_available: bool = False
    proposal_pending: bool = False
    applied_write: bool = False
    verified_blocks: tuple[ResponseBlock, ...] = ()


class OrchestratorFinalResponse(_Contract):
    """Model-authored candidate. Publication still requires runtime verification."""

    status: OrchestratorStatus
    blocks: tuple[ResponseBlock, ...] = Field(min_length=1, max_length=16)
    stop_reason: str = Field(min_length=1, max_length=100)


class OrchestratorOutputV2(OrchestratorFinalResponse):
    schema_version: Literal["2.0"] = "2.0"
    model_refs: tuple[str, ...] = ()
    evidence_call_ids: tuple[str, ...] = ()


class PublishedResponse(_Contract):
    kind: Literal["PUBLISHED"] = "PUBLISHED"
    output: OrchestratorOutputV2
    wait_ref: WaitReference | None = None

    @model_validator(mode="after")
    def public_wait_matches_status(self) -> Self:
        expected = {
            OrchestratorStatus.AWAITING_INPUT: "USER_INPUT",
            OrchestratorStatus.AWAITING_HUMAN: "HUMAN_DECISION",
        }.get(self.output.status)
        if expected is None:
            if self.wait_ref is not None:
                raise ValueError("TERMINAL_RESPONSE_CANNOT_WAIT")
        elif self.wait_ref is None or self.wait_ref.kind != expected:
            raise ValueError("PUBLIC_STATUS_REQUIRES_MATCHING_WAIT")
        return self


class SuspendedWorkflow(_Contract):
    kind: Literal["SUSPENDED"] = "SUSPENDED"
    wait_ref: WaitReference
    public_activity: tuple[ActivityResponseBlock, ...]
    log_cursor: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def workflow_wait_only(self) -> Self:
        if self.wait_ref.kind != "WORKFLOW_RESULT":
            raise ValueError("SUSPENSION_REQUIRES_WORKFLOW_WAIT")
        return self


type OrchestratorExecutionResult = Annotated[
    PublishedResponse | SuspendedWorkflow, Field(discriminator="kind")
]
