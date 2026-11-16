"""Immutable exact-major Skill manifest registry."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel

from work_management_ai.runtime.contracts import RiskLevel
from work_management_ai.runtime.manifests import (
    SkillManifest,
    SkillManifestV2,
    canonical_manifest_fingerprint,
    resolve_contract,
)


class SkillRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredSkill:
    reference: str
    manifest: SkillManifest | SkillManifestV2
    fingerprint: str


class SkillRegistry:
    def __init__(self, manifests: Iterable[SkillManifest | SkillManifestV2] = ()) -> None:
        entries: dict[str, RegisteredSkill] = {}
        for manifest in manifests:
            if isinstance(manifest, SkillManifestV2):
                manifest = SkillManifestV2.model_validate(manifest.model_dump())
            if (manifest.risk_level is RiskLevel.READ_ONLY and manifest.approval != "NONE") or (
                manifest.risk_level is RiskLevel.PROPOSAL_ONLY and manifest.approval != "ALWAYS"
            ):
                raise SkillRegistryError("SKILL_APPROVAL_INCOMPATIBLE")
            for contract_path in (manifest.input_contract, manifest.output_contract):
                try:
                    contract = resolve_contract(contract_path)
                    if isinstance(manifest, SkillManifestV2) and (
                        not isinstance(contract, type) or not issubclass(contract, BaseModel)
                    ):
                        raise ValueError("SKILL_CONTRACT_INVALID")
                except (AttributeError, ImportError, ValueError) as exc:
                    raise SkillRegistryError("SKILL_CONTRACT_INVALID") from exc
            major = manifest.version.split(".", maxsplit=1)[0]
            reference = f"{manifest.name}@{major}"
            if reference in entries:
                raise SkillRegistryError("SKILL_MAJOR_ALREADY_REGISTERED")
            entries[reference] = RegisteredSkill(
                reference=reference,
                manifest=manifest,
                fingerprint=canonical_manifest_fingerprint(manifest),
            )
        self._entries: Mapping[str, RegisteredSkill] = MappingProxyType(entries)

    def resolve(self, reference: str) -> RegisteredSkill:
        try:
            return self._entries[reference]
        except KeyError as exc:
            raise SkillRegistryError("SKILL_NOT_REGISTERED") from exc
