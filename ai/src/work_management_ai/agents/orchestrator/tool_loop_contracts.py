"""Native Orchestrator output, retaining the four existing public statuses."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
from work_management_ai.runtime.contracts import ActivityResponseBlock, ResponseBlock
from work_management_ai.runtime.tool_loop_contracts import WaitReference


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


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
