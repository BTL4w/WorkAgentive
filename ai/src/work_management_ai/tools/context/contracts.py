"""Typed service facts, untrusted content, coverage and runtime provenance."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from work_management_ai.runtime.tool_loop_contracts import RunScope


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


type DataOutcome = Literal["OK", "PARTIAL", "NEEDS_INPUT", "UNAVAILABLE", "STALE"]
type Resolution = Literal[
    "UNIQUE", "AMBIGUOUS", "NOT_FOUND_OR_NOT_ACCESSIBLE", "UNAVAILABLE", "NOT_REQUESTED"
]
type UseOutcome = Literal["ACCEPTED", "NEEDS_INPUT", "REFRESH_REQUIRED", "DENIED"]


class ExploreProjectsInput(Contract):
    query: str = Field(
        min_length=1,
        max_length=160,
        description="Project name or search phrase supplied by the user or available context.",
    )
    cursor: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Copy the cursor from the preceding page for the same query and page size; "
            "null for the first page."
        ),
    )
    page_size: int = Field(
        default=10,
        ge=1,
        le=20,
        strict=True,
        description=(
            "Maximum candidates per page, from 1 to 20. Use 10 unless another size is needed."
        ),
    )


class ReadProjectInput(Contract):
    reference: UUID = Field(
        description=(
            "The Project UUID. From discovery, copy resource_id from the confirmed Project's "
            "context_refs entry. ref_id identifies the evidence record."
        ),
    )
    sections: tuple[str, ...] = Field(
        default=("overview",),
        min_length=1,
        max_length=8,
        description=(
            "Use ['overview'] for the Project overview. Other section names produce "
            "NOT_SUPPORTED gaps."
        ),
    )


class ReadActorInput(Contract):
    sections: tuple[str, ...] = Field(
        default=("identity",),
        min_length=1,
        max_length=8,
        description=(
            "Use ['identity'] for the current actor's display name and organization name."
        ),
    )


class ContextRef(Contract):
    ref_id: UUID
    scope: RunScope
    resource_type: Literal["project", "actor"]
    resource_id: UUID
    version: int | None = Field(default=None, ge=1, strict=True)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    observed_at: AwareDatetime
    invocation_id: UUID | None = None
    source: Literal["project_service", "current_actor_service"] = "project_service"


class ContextFact(Contract):
    ref_id: UUID
    field: Literal[
        "project.name", "project.version", "actor.display_name", "actor.organization_name"
    ]
    value: str | int = Field(union_mode="left_to_right")


class SourceContent(Contract):
    ref_id: UUID
    field: Literal["project.description"] = "project.description"
    text: str = Field(max_length=2048)
    trust: Literal["UNTRUSTED_SOURCE_CONTENT"] = "UNTRUSTED_SOURCE_CONTENT"
    truncated: bool = False


class EntityResolution(Contract):
    query_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    resolution: Resolution
    candidate_ref_ids: tuple[UUID, ...] = Field(default=(), max_length=20)
    selected_ref_id: UUID | None = None

    @model_validator(mode="after")
    def selected_is_unique(self) -> Self:
        if self.selected_ref_id is not None and (
            self.resolution != "UNIQUE" or self.candidate_ref_ids != (self.selected_ref_id,)
        ):
            raise ValueError("INVALID_ENTITY_SELECTION")
        return self


class ReadCoverage(Contract):
    requested: tuple[str, ...] = Field(default=(), max_length=8)
    returned: tuple[str, ...] = Field(default=(), max_length=8)
    status: Literal["COMPLETE", "PARTIAL", "NOT_REQUESTED"]
    query_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    cursor: str | None = Field(default=None, max_length=2048)
    truncated: bool = False


class ContextGap(Contract):
    code: Literal["NOT_SUPPORTED", "PAGINATION", "TRUNCATED", "UNAVAILABLE", "STALE", "MISSING"]
    section: str = Field(default="context", min_length=1, max_length=64)


class ContextReadObservation(Contract):
    schema_version: Literal["1.0"] = "1.0"
    scope: RunScope
    objective_ref: UUID
    invocation_id: UUID
    tool_reference: Literal["explore_projects@1", "read_project_context@1", "read_actor_context@1"]
    tool_version: str = Field(
        pattern=(
            r"^1\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
        ),
        max_length=64,
    )
    observed_at: AwareDatetime
    outcome: DataOutcome
    context_refs: tuple[ContextRef, ...] = Field(default=(), max_length=20)
    entities: tuple[EntityResolution, ...] = Field(default=(), max_length=1)
    facts: tuple[ContextFact, ...] = Field(default=(), max_length=40)
    source_content_refs: tuple[SourceContent, ...] = Field(default=(), max_length=1)
    coverage: tuple[ReadCoverage, ...] = Field(default=(), max_length=1)
    gaps: tuple[ContextGap, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def provenance(self) -> Self:
        refs = tuple(
            r.model_copy(update={"invocation_id": self.invocation_id})
            if r.invocation_id is None
            else r
            for r in self.context_refs
        )
        ids = {r.ref_id for r in refs}
        if len(ids) != len(refs) or any(
            r.scope != self.scope or r.invocation_id != self.invocation_id for r in refs
        ):
            raise ValueError("CONTEXT_SCOPE_MISMATCH")
        if any(f.ref_id not in ids for f in (*self.facts, *self.source_content_refs)) or any(
            not set(e.candidate_ref_ids) <= ids for e in self.entities
        ):
            raise ValueError("CONTEXT_EVIDENCE_MISSING")
        object.__setattr__(self, "context_refs", refs)
        return self


class ExplorationReport(Contract):
    schema_version: Literal["1.0"] = "1.0"
    scope: RunScope
    report_id: UUID
    objective_ref: UUID
    revision: int = Field(default=0, ge=0)
    source_invocation_ids: tuple[UUID, ...] = Field(default=(), max_length=16)
    context_refs: tuple[ContextRef, ...] = Field(default=(), max_length=32)
    entities: tuple[EntityResolution, ...] = ()
    facts: tuple[ContextFact, ...] = ()
    source_content_refs: tuple[SourceContent, ...] = ()
    coverage: tuple[ReadCoverage, ...] = ()
    gaps: tuple[ContextGap, ...] = Field(default=(), max_length=16)
    selected_context_refs: tuple[UUID, ...] = Field(default=(), max_length=32)
    status: Literal["COMPLETE", "PARTIAL", "NOT_REQUESTED"] = "NOT_REQUESTED"
    truncated: bool = False
    built_at: AwareDatetime
    snapshot_sequence: int = Field(default=0, ge=0)


class ContextUseValidation(Contract):
    scope: RunScope
    target_tool: str = Field(min_length=1, max_length=100)
    input_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    actor_revision: str
    policy_revision: str
    context_versions: tuple[ContextRef, ...]
    outcome: UseOutcome
