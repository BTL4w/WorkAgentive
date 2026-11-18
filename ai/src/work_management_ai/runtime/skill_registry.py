"""Immutable exact-major Skill manifest registry."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ValidationError

from work_management_ai.runtime.contracts import RiskLevel
from work_management_ai.runtime.manifests import (
    SkillManifest,
    SkillManifestV2,
    canonical_manifest_fingerprint,
    load_skill_manifest_resource,
    resolve_contract,
)
from work_management_ai.runtime.skill_bundle import (
    SkillBundleDescriptor,
    SkillBundleError,
    bundle_name_for,
    read_skill_bundle,
)


class SkillRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredSkill:
    reference: str
    manifest: SkillManifest | SkillManifestV2
    fingerprint: str
    # Canonical SKILL.md metadata indexed at trusted registration. The manifest
    # fingerprint already pins the instructions digest, so it is not re-hashed here.
    bundle: SkillBundleDescriptor | None = None


def _verified_bundle(
    manifest: SkillManifest | SkillManifestV2, descriptor: SkillBundleDescriptor
) -> SkillBundleDescriptor:
    if not isinstance(manifest, SkillManifestV2):
        raise SkillRegistryError("SKILL_BUNDLE_MANIFEST_INCOMPATIBLE")
    try:
        descriptor = SkillBundleDescriptor.model_validate(descriptor.model_dump())
    except ValidationError as exc:
        raise SkillRegistryError("SKILL_BUNDLE_INVALID") from exc
    if descriptor.name != bundle_name_for(manifest.name):
        raise SkillRegistryError("SKILL_BUNDLE_IDENTITY_MISMATCH")
    if descriptor.instructions != manifest.instructions or descriptor.resources != (
        manifest.resources
    ):
        raise SkillRegistryError("SKILL_BUNDLE_RESOURCE_MISMATCH")
    # Re-derive from the approved bytes so a forged descriptor cannot inject metadata.
    try:
        derived = read_skill_bundle(
            package=descriptor.package,
            instructions=manifest.instructions,
            resources=manifest.resources,
        )
    except SkillBundleError as exc:
        raise SkillRegistryError(exc.code) from exc
    if derived != descriptor:
        raise SkillRegistryError("SKILL_BUNDLE_DESCRIPTOR_MISMATCH")
    if derived.description != manifest.description:
        raise SkillRegistryError("SKILL_BUNDLE_DESCRIPTION_MISMATCH")
    return derived


class SkillRegistry:
    @classmethod
    def from_resources(cls, resources: Iterable[tuple[str, str]]) -> "SkillRegistry":
        manifests: list[SkillManifest | SkillManifestV2] = []
        bundles: dict[str, SkillBundleDescriptor] = {}
        for package, resource in resources:
            manifest = load_skill_manifest_resource(package, resource)
            manifests.append(manifest)
            if isinstance(manifest, SkillManifestV2):
                try:
                    descriptor = read_skill_bundle(
                        package=package,
                        instructions=manifest.instructions,
                        resources=manifest.resources,
                    )
                except SkillBundleError as exc:
                    raise SkillRegistryError(exc.code) from exc
                reference = f"{manifest.name}@{manifest.version.split('.', maxsplit=1)[0]}"
                bundles[reference] = descriptor
        return cls(manifests, bundles=bundles)

    def __init__(
        self,
        manifests: Iterable[SkillManifest | SkillManifestV2] = (),
        *,
        bundles: Mapping[str, SkillBundleDescriptor] | None = None,
    ) -> None:
        pending = dict(bundles or {})
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
            descriptor = pending.pop(reference, None)
            entries[reference] = RegisteredSkill(
                reference=reference,
                manifest=manifest,
                fingerprint=canonical_manifest_fingerprint(manifest),
                bundle=None if descriptor is None else _verified_bundle(manifest, descriptor),
            )
        if pending:
            raise SkillRegistryError("SKILL_BUNDLE_NOT_REGISTERED")
        self._entries: Mapping[str, RegisteredSkill] = MappingProxyType(entries)

    def resolve(self, reference: str) -> RegisteredSkill:
        try:
            return self._entries[reference]
        except KeyError as exc:
            raise SkillRegistryError("SKILL_NOT_REGISTERED") from exc
