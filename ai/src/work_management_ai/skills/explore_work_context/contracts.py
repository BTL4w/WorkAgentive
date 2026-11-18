"""Inputs name context needs; output is the runtime-owned report."""

from uuid import UUID

from pydantic import Field

from work_management_ai.tools.context.contracts import Contract, ExplorationReport


class ExploreWorkContextInput(Contract):
    objective_ref: UUID
    objective: str = Field(min_length=1, max_length=4000)
    known_context_ref_ids: tuple[UUID, ...] = Field(default=(), max_length=32)
    unresolved_entities: tuple[str, ...] = Field(default=(), max_length=8)
    questions: tuple[str, ...] = Field(default=(), max_length=8)


ExploreWorkContextOutput = ExplorationReport
