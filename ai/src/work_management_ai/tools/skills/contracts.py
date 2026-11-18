"""Selectors carry identity only; skill text cannot grant execution permission."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from work_management_ai.runtime.tool_loop_contracts import VersionPin


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class LoadSkillInput(_Contract):
    skill_ref: str = Field(
        pattern=r"^[a-z][a-z0-9_.-]*@[1-9][0-9]*$",
        max_length=100,
        description=(
            "Copy the exact name@major from the catalog's reference field, including underscores "
            "and version."
        ),
    )
    resource_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]{0,63}$",
        description=(
            "Use null for instructions. For a companion, copy its ID from the already-loaded "
            "Skill's resource list."
        ),
    )


class SkillResourceInfo(_Contract):
    id: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1, le=16384)
    media_type: Literal["text/plain", "text/markdown", "application/json"]


class SkillMetadata(_Contract):
    reference: str
    version: str
    name: str
    description: str
    triggers: tuple[str, ...]
    manifest_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    instructions_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_kind: Literal["INSTRUCTIONS"] = "INSTRUCTIONS"


class LoadedSkill(_Contract):
    reference: str
    version: str
    manifest_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    body_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    resource_id: str | None
    content: str
    media_type: Literal["text/plain", "text/markdown", "application/json"]
    resources: tuple[SkillResourceInfo, ...] = Field(max_length=16)
    effective_tools: tuple[str, ...] = Field(max_length=6)
    pin: VersionPin

    @field_validator("content")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 16384:
            raise ValueError("SKILL_RESOURCE_TOO_LARGE")
        if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
            raise ValueError("SKILL_RESOURCE_ENCODING")
        return value
