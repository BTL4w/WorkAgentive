"""Lazy reads from approved package identities, never paths supplied by a model."""

import hashlib
import json
import re
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel

from work_management_ai.runtime.contracts import AgentId
from work_management_ai.runtime.manifests import SkillManifestV2, SkillResourceManifest
from work_management_ai.runtime.session_history import RevalidatedSkills
from work_management_ai.runtime.skill_bundle import (
    SkillBundleError,
    bundle_name_for,
    parse_skill_markdown,
)
from work_management_ai.runtime.skill_registry import (
    RegisteredSkill,
    SkillRegistry,
    SkillRegistryError,
)
from work_management_ai.runtime.tool_catalog import ToolCatalog, ToolCatalogError
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    RunScope,
    ToolInvocationContext,
    VersionPin,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.tools.skills.contracts import (
    LoadedSkill,
    LoadSkillInput,
    SkillMetadata,
    SkillResourceInfo,
)


class SkillLoadingError(ToolPolicyError):
    pass


class ApprovedPackageResources:
    def __init__(self, packages: Mapping[str, str]):
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", package)
            for package in packages.values()
        ):
            raise SkillLoadingError("SKILL_PACKAGE_INVALID")
        self._packages = MappingProxyType(dict(packages))

    def _path(self, manifest: SkillManifestV2, resource: SkillResourceManifest) -> Traversable:
        # Revalidate even trusted registrations before any filesystem access.
        resource = SkillResourceManifest.model_validate(resource.model_dump())
        reference = f"{manifest.name}@{manifest.version.split('.')[0]}"
        package = self._packages.get(reference)
        if package is None:
            raise SkillLoadingError("SKILL_PACKAGE_NOT_APPROVED")
        try:
            root = files(package)
            path = root.joinpath(*resource.path.split("/"))
            if not isinstance(root, Path):
                raise SkillLoadingError("SKILL_RESOURCE_PATH_INVALID")
            part = root
            for name in resource.path.split("/"):
                part = part / name
                if part.is_symlink():
                    raise SkillLoadingError("SKILL_RESOURCE_PATH_INVALID")
            if not part.resolve().is_relative_to(root.resolve()):
                raise SkillLoadingError("SKILL_RESOURCE_PATH_INVALID")
            if not path.is_file():
                raise SkillLoadingError("SKILL_RESOURCE_UNAVAILABLE")
            return path
        except SkillLoadingError:
            raise
        except (OSError, ImportError, ValueError, TypeError):
            raise SkillLoadingError("SKILL_RESOURCE_UNAVAILABLE") from None

    def available(self, manifest: SkillManifestV2) -> bool:
        for resource in (manifest.instructions, *manifest.resources):
            self._path(manifest, resource)
        return True

    def read(self, manifest: SkillManifestV2, resource: SkillResourceManifest) -> bytes:
        path = self._path(manifest, resource)
        try:
            with path.open("rb") as handle:
                body = handle.read(16385)
        except OSError:
            raise SkillLoadingError("SKILL_RESOURCE_UNAVAILABLE") from None
        if len(body) > 16384:
            raise SkillLoadingError("SKILL_RESOURCE_TOO_LARGE")
        if len(body) != resource.size_bytes or hashlib.sha256(body).hexdigest() != resource.sha256:
            raise SkillLoadingError("SKILL_RESOURCE_MISMATCH")
        return body


def _pin(skill: RegisteredSkill, manifest: SkillManifestV2) -> VersionPin:
    digest = hashlib.sha256(
        f"{skill.fingerprint}:{manifest.instructions.sha256}".encode()
    ).hexdigest()
    return VersionPin(reference=skill.reference, version=manifest.version, fingerprint=digest)


class SkillLoader:
    def __init__(
        self,
        *,
        skills: SkillRegistry,
        resources: ApprovedPackageResources,
        catalog: ToolCatalog,
        model_ref: str,
        known_loaded: Mapping[RunScope, tuple[VersionPin, ...]] | None = None,
    ):
        self.skills, self.resources, self.tools, self.model_ref = (
            skills,
            resources,
            catalog,
            model_ref,
        )
        # Operational pins only. Every use independently revalidates current access.
        self._loaded = {
            scope: {p.reference: p for p in pins} for scope, pins in (known_loaded or {}).items()
        }

    def _skill(self, scope: RunScope, reference: str) -> tuple[RegisteredSkill, SkillManifestV2]:
        facts = self.tools.context(scope)
        if (
            reference not in self.tools.agent.manifest.allowed_skills
            or reference not in facts.allowed_skills
        ):
            raise SkillLoadingError("SKILL_NOT_AVAILABLE", stop_segment=True)
        try:
            entry = self.skills.resolve(reference)
        except SkillRegistryError:
            raise SkillLoadingError("SKILL_NOT_AVAILABLE") from None
        if not isinstance(entry.manifest, SkillManifestV2):
            raise SkillLoadingError("SKILL_MANIFEST_INCOMPATIBLE")
        manifest = entry.manifest
        dependency = VersionPin(
            reference=reference, version=manifest.version, fingerprint=entry.fingerprint
        )
        if dependency not in self.tools.agent.skill_pins:
            raise SkillLoadingError("SKILL_VERSION_CONFLICT", stop_segment=True)
        if entry.bundle is None and manifest.instructions.path != "SKILL.md":
            raise SkillLoadingError("SKILL_BUNDLE_DESCRIPTOR_REQUIRED", stop_segment=True)
        if AgentId.ORCHESTRATOR not in manifest.runnable_by_agents:
            raise SkillLoadingError("SKILL_NOT_AVAILABLE", stop_segment=True)
        expected = _pin(entry, manifest)
        prior = self._loaded.get(scope, {}).get(reference)
        if prior is not None and prior != expected:
            raise SkillLoadingError("SKILL_VERSION_CONFLICT", stop_segment=True)
        self.resources.available(manifest)
        return entry, manifest

    async def revalidate_skills(
        self, scope: RunScope, pins: tuple[VersionPin, ...]
    ) -> RevalidatedSkills:
        try:
            for pin in pins:
                entry, manifest = self._skill(scope, pin.reference)
                if pin != _pin(entry, manifest):
                    return RevalidatedSkills(allowed=True, pins=pins, fresh=False)
        except SkillLoadingError as exc:
            if exc.code == "SKILL_NOT_AVAILABLE":
                return RevalidatedSkills(allowed=False, pins=pins)
            if exc.code == "SKILL_VERSION_CONFLICT":
                return RevalidatedSkills(allowed=True, pins=pins, fresh=False)
            raise
        return RevalidatedSkills(allowed=True, pins=pins)

    def revalidate_loaded(
        self, scope: RunScope, value: LoadSkillInput, output: LoadedSkill
    ) -> None:
        entry, manifest = self._skill(scope, value.skill_ref)
        resource = (
            manifest.instructions
            if value.resource_id is None
            else next((r for r in manifest.resources if r.id == value.resource_id), None)
        )
        expected_index = tuple(
            SkillResourceInfo(**r.model_dump(exclude={"path"})) for r in manifest.resources
        )
        if (
            resource is None
            or output.reference != value.skill_ref
            or output.resource_id != value.resource_id
            or output.version != manifest.version
            or output.manifest_fingerprint != entry.fingerprint
            or output.pin != _pin(entry, manifest)
            or output.media_type != resource.media_type
            or output.resources != expected_index
        ):
            raise SkillLoadingError("SKILL_VERSION_CONFLICT", stop_segment=True)
        expected_digest = resource.sha256
        if value.resource_id is None and entry.bundle is not None:
            expected = self._content(entry, manifest, resource)
            expected_digest = hashlib.sha256(expected.encode()).hexdigest()
        if (
            output.body_sha256 != expected_digest
            or hashlib.sha256(output.content.encode()).hexdigest() != expected_digest
        ):
            raise SkillLoadingError("SKILL_VERSION_CONFLICT", stop_segment=True)
        current = self.tools.prepare(scope, model_ref=self.model_ref)
        try:
            current.resolve("load_skill")
            effective = set(manifest.allowed_tools) & {t.reference for t in current.schemas}
            if not set(output.effective_tools).issubset(effective):
                raise SkillLoadingError("SKILL_CONTEXT_STALE", stop_segment=True)
        finally:
            current.close()

    def catalog(self, scope: RunScope) -> tuple[SkillMetadata, ...]:
        try:
            admitted = self.tools.prepare(scope, model_ref=self.model_ref)
            try:
                if "load_skill@1" not in {v.reference for v in admitted.schemas}:
                    return ()
                facts = self.tools.context(scope)
                entries: list[SkillMetadata] = []
                for reference in sorted(
                    set(self.tools.agent.manifest.allowed_skills) & facts.allowed_skills
                ):
                    entry, manifest = self._skill(scope, reference)
                    entries.append(
                        SkillMetadata(
                            reference=reference,
                            version=manifest.version,
                            name=entry.bundle.name
                            if entry.bundle
                            else bundle_name_for(manifest.name),
                            description=(
                                entry.bundle.description if entry.bundle else manifest.description
                            ),
                            triggers=manifest.triggers,
                            manifest_fingerprint=entry.fingerprint,
                            instructions_sha256=manifest.instructions.sha256,
                        )
                    )
                admitted.resolve("load_skill")
                return tuple(entries)
            finally:
                admitted.close()
        except ToolCatalogError:
            raise SkillLoadingError("SKILL_CATALOG_UNAVAILABLE", stop_segment=True) from None

    def _content(
        self, entry: RegisteredSkill, manifest: SkillManifestV2, resource: SkillResourceManifest
    ) -> str:
        try:
            text = self.resources.read(manifest, resource).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise SkillLoadingError("SKILL_RESOURCE_ENCODING") from None
        if any(ord(c) < 32 and c not in "\r\n\t" for c in text):
            raise SkillLoadingError("SKILL_RESOURCE_ENCODING")
        if resource.media_type == "application/json":
            try:
                json.loads(text)
            except (ValueError, RecursionError):
                raise SkillLoadingError("SKILL_RESOURCE_INVALID") from None
        if resource == manifest.instructions and entry.bundle is not None:
            try:
                frontmatter, text = parse_skill_markdown(text)
            except SkillBundleError as exc:
                raise SkillLoadingError(exc.code) from None
            if (frontmatter.name, frontmatter.description) != (
                entry.bundle.name,
                entry.bundle.description,
            ):
                raise SkillLoadingError("SKILL_BUNDLE_DESCRIPTOR_MISMATCH")
        return text

    def load(self, scope: RunScope, value: LoadSkillInput) -> LoadedSkill:
        value = LoadSkillInput.model_validate(value)
        try:
            admitted = self.tools.prepare(scope, model_ref=self.model_ref)
            try:
                admitted.resolve("load_skill")
                entry, manifest = self._skill(scope, value.skill_ref)
                pin = _pin(entry, manifest)
                if value.resource_id is not None:
                    resource = next(
                        (r for r in manifest.resources if r.id == value.resource_id), None
                    )
                    if resource is None:
                        raise SkillLoadingError("SKILL_RESOURCE_NOT_DECLARED")
                    if self._loaded.get(scope, {}).get(value.skill_ref) != pin:
                        raise SkillLoadingError("SKILL_INSTRUCTIONS_NOT_LOADED")
                else:
                    resource = manifest.instructions
                text = self._content(entry, manifest, resource)
                effective = tuple(
                    sorted(set(manifest.allowed_tools) & {v.reference for v in admitted.schemas})
                )
                # Recheck access after the read; package pins never override revocation.
                current, _ = self._skill(scope, value.skill_ref)
                if current != entry:
                    raise SkillLoadingError("SKILL_VERSION_CONFLICT", stop_segment=True)
                admitted.resolve("load_skill")
                output = LoadedSkill(
                    reference=entry.reference,
                    version=manifest.version,
                    manifest_fingerprint=entry.fingerprint,
                    body_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    resource_id=value.resource_id,
                    content=text,
                    media_type=resource.media_type,
                    resources=tuple(
                        SkillResourceInfo(**r.model_dump(exclude={"path"}))
                        for r in manifest.resources
                    ),
                    effective_tools=effective,
                    pin=pin,
                )
                if value.resource_id is None:
                    self._loaded.setdefault(scope, {})[entry.reference] = pin
                return output
            finally:
                admitted.close()
        except ToolCatalogError:
            raise SkillLoadingError("SKILL_CATALOG_UNAVAILABLE", stop_segment=True) from None


class SkillLoaderBackend:
    """Pure read backend for Task6; pinned package bytes are content-addressed data."""

    def __init__(self, loader: SkillLoader):
        self.loader = loader

    def _check(self, context: ToolInvocationContext) -> None:
        if context.tool_reference != "load_skill@1" or not any(
            pin.reference == context.tool_reference and pin.version == context.tool_version
            for pin in self.loader.tools.agent.tool_pins
        ):
            raise SkillLoadingError("SKILL_TOOL_IDENTITY_INVALID", stop_segment=True)

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        self._check(context)
        value = LoadSkillInput.model_validate(arguments.model_dump())
        output = self.loader.load(context.scope, value)
        return BackendToolResult(output=output, loaded_skill=output.pin)

    async def revalidate_result(
        self, context: ToolInvocationContext, result: BackendToolResult
    ) -> None:
        self._check(context)
        if not isinstance(result.output, LoadedSkill):
            raise SkillLoadingError("SKILL_RESULT_INVALID", stop_segment=True)
        value = LoadSkillInput.model_validate(context.canonical_arguments.model_dump())
        self.loader.revalidate_loaded(context.scope, value, result.output)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        # Reconstruct only the same immutable, approved package read under current access.
        # This adapter never executes skills/scripts or creates business effects.
        return await self.execute(context, arguments)
