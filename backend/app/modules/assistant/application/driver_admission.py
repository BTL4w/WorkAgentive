"""Admission boundary for durable Assistant driver pins."""

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.tool_loop_contracts import VersionPin


class PinnedDriver(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    driver: Literal["tool_loop"] = "tool_loop"
    driver_version: str = Field(min_length=1, max_length=64)
    agent_version: str = Field(min_length=1, max_length=64)
    manifest_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_version: str = Field(min_length=1, max_length=64)
    workflow_id: str = Field(min_length=1, max_length=100)
    contract_version: Literal["2.0"] = "2.0"
    model_provider: Literal["mock", "openai"]
    model_ref: str = Field(min_length=1, max_length=200)
    tool_versions: tuple[VersionPin, ...]
    specialist_versions: tuple[VersionPin, ...]
    skill_versions: tuple[VersionPin, ...]
    skill_bundle_fingerprints: tuple[VersionPin, ...]


@dataclass(frozen=True, slots=True)
class ContinuationAdmission:
    wait_id: UUID
    expected_version: int
    pin: PinnedDriver


class DriverAdmissionPort(Protocol):
    def select(
        self,
        actor: AuthenticatedActor,
        *,
        verified_action: bool,
        continuation_source: PinnedDriver | Literal["legacy"] | None,
    ) -> PinnedDriver | None: ...
