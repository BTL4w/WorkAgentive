"""Canonical SKILL.md frontmatter parsing for trusted, approved Skill bundles.

Agent Skills metadata (``name``/``description``) is authored only in the bundle's
``SKILL.md``. Governance (version, owner, permissions, risk, approval, allowlists,
hashes) stays in the trusted sidecar manifest; frontmatter can never supply it.
Package identities and resource paths come from trusted registration only; this
module never accepts a path chosen by a model, user or Project note.
"""

import hashlib
import json
import re
from collections.abc import Hashable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from work_management_ai.runtime.manifests import SKILL_RESOURCE_MAX_BYTES, SkillResourceManifest

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NAME_MAX: Final = 64
_DESCRIPTION_MAX: Final = 1024
_PACKAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_INSTRUCTIONS_FILE: Final = "SKILL.md"
_FRONTMATTER_FIELDS: Final = frozenset({"name", "description"})


class SkillBundleError(ValueError):
    """Typed fail-closed error; ``str(error)`` is the stable code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def bundle_name_for(skill_name: str) -> str:
    """Map an internal skill identity (``explore_work_context``) to its bundle name."""

    return skill_name.replace("_", "-")


def _validate_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _NAME_MAX
        or _NAME.fullmatch(value) is None
    ):
        raise SkillBundleError("SKILL_NAME_INVALID")
    return value


def _validate_description(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _DESCRIPTION_MAX
        or not value.strip()
        or _has_control_characters(value)
    ):
        raise SkillBundleError("SKILL_DESCRIPTION_INVALID")
    return value


def _has_control_characters(text: str) -> bool:
    return any(ord(c) < 32 and c not in "\r\n\t" for c in text)


class _BundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SkillFrontmatter(_BundleModel):
    """The only portable metadata accepted from SKILL.md frontmatter."""

    name: str
    description: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _validate_name(value)

    @field_validator("description")
    @classmethod
    def valid_description(cls, value: str) -> str:
        return _validate_description(value)


class SkillBundleDescriptor(_BundleModel):
    """Pinned catalog metadata for one approved bundle; carries no instruction body."""

    package: str
    bundle: str
    name: str
    description: str
    instructions: SkillResourceManifest
    resources: tuple[SkillResourceManifest, ...] = Field(default=(), max_length=16)

    @field_validator("package")
    @classmethod
    def valid_package(cls, value: str) -> str:
        if _PACKAGE.fullmatch(value) is None:
            raise SkillBundleError("SKILL_BUNDLE_PACKAGE_INVALID")
        return value

    @field_validator("bundle", "name")
    @classmethod
    def valid_bundle_name(cls, value: str) -> str:
        return _validate_name(value)

    @field_validator("description")
    @classmethod
    def valid_description(cls, value: str) -> str:
        return _validate_description(value)

    @model_validator(mode="after")
    def canonical_layout(self) -> Self:
        if self.name != self.bundle:
            raise SkillBundleError("SKILL_NAME_FOLDER_MISMATCH")
        if _check_layout(self.instructions, self.resources) != self.bundle:
            raise SkillBundleError("SKILL_BUNDLE_LAYOUT_INVALID")
        return self


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader (no custom/python tags) that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        key = cast(object, loader.construct_object(key_node, deep=True))  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(key, Hashable) or key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, "duplicate or unhashable key", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,  # pyright: ignore[reportArgumentType]
)


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.split("\n")
    if lines[0].removesuffix("\r") != "---":
        raise SkillBundleError("SKILL_FRONTMATTER_MISSING")
    for index in range(1, len(lines)):
        if lines[index].removesuffix("\r") == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    raise SkillBundleError("SKILL_FRONTMATTER_MISSING")


def parse_skill_markdown(text: str) -> tuple[SkillFrontmatter, str]:
    """Parse complete SKILL.md text into canonical frontmatter and non-empty body."""

    if _has_control_characters(text):
        raise SkillBundleError("SKILL_RESOURCE_ENCODING")
    raw_frontmatter, body = _split_frontmatter(text)
    try:
        loaded: object = yaml.load(raw_frontmatter, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, RecursionError, TypeError, ValueError):
        raise SkillBundleError("SKILL_FRONTMATTER_INVALID") from None
    if not isinstance(loaded, dict) or not loaded:
        raise SkillBundleError("SKILL_FRONTMATTER_INVALID")
    data: dict[str, object] = {}
    for key, value in cast(dict[object, object], loaded).items():
        if not isinstance(key, str):
            raise SkillBundleError("SKILL_FRONTMATTER_INVALID")
        data[key] = value
    # Frontmatter cannot declare tools, permissions, activation, risk or approval.
    if data.keys() - _FRONTMATTER_FIELDS:
        raise SkillBundleError("SKILL_FRONTMATTER_FIELD_UNSUPPORTED")
    if data.keys() != _FRONTMATTER_FIELDS:
        raise SkillBundleError("SKILL_FRONTMATTER_FIELD_MISSING")
    name = _validate_name(data["name"])
    description = _validate_description(data["description"])
    if not body.strip():
        raise SkillBundleError("SKILL_INSTRUCTIONS_EMPTY")
    return SkillFrontmatter(name=name, description=description), body


def _revalidate(resource: SkillResourceManifest) -> SkillResourceManifest:
    # Revalidate even trusted objects (model_construct bypass) before any filesystem access.
    try:
        return SkillResourceManifest.model_validate(resource.model_dump())
    except ValidationError as exc:
        if any(error["loc"][:1] == ("path",) for error in exc.errors()):
            raise SkillBundleError("SKILL_RESOURCE_PATH_INVALID") from None
        raise SkillBundleError("SKILL_RESOURCE_INVALID") from None


def _check_layout(
    instructions: SkillResourceManifest, resources: tuple[SkillResourceManifest, ...]
) -> str:
    parts = instructions.path.split("/")
    if (
        len(parts) != 2
        or parts[1] != _INSTRUCTIONS_FILE
        or _NAME.fullmatch(parts[0]) is None
        or len(parts[0]) > _NAME_MAX
        or instructions.id != "instructions"
        or instructions.media_type != "text/markdown"
    ):
        raise SkillBundleError("SKILL_BUNDLE_LAYOUT_INVALID")
    bundle = parts[0]
    entries = (instructions, *resources)
    if (
        len(resources) > 16
        or any(not r.path.startswith(f"{bundle}/") for r in resources)
        or len({r.id for r in entries}) != len(entries)
        or len({r.path for r in entries}) != len(entries)
    ):
        raise SkillBundleError("SKILL_BUNDLE_LAYOUT_INVALID")
    return bundle


def _approved_file(root: Traversable, relative: str) -> Traversable:
    parts = relative.split("/")
    try:
        # Namespace/virtual roots do not expose a single filesystem boundary
        # that this reader can verify. Reject them before joining or reading.
        if not isinstance(root, Path):
            raise SkillBundleError("SKILL_RESOURCE_PATH_INVALID")
        part = root
        for name in parts:
            part = part / name
            if part.is_symlink():
                raise SkillBundleError("SKILL_RESOURCE_PATH_INVALID")
        if not part.resolve().is_relative_to(root.resolve()):
            raise SkillBundleError("SKILL_RESOURCE_PATH_INVALID")
        path = root.joinpath(*parts)
        if not path.is_file():
            raise SkillBundleError("SKILL_RESOURCE_UNAVAILABLE")
        return path
    except SkillBundleError:
        raise
    except (OSError, ValueError, TypeError):
        raise SkillBundleError("SKILL_RESOURCE_UNAVAILABLE") from None


def _read_verified_text(root: Traversable, resource: SkillResourceManifest) -> str:
    path = _approved_file(root, resource.path)
    try:
        with path.open("rb") as handle:
            body = handle.read(SKILL_RESOURCE_MAX_BYTES + 1)
    except OSError:
        raise SkillBundleError("SKILL_RESOURCE_UNAVAILABLE") from None
    # Never accept a truncated prefix: the whole resource must fit the limit.
    if len(body) > SKILL_RESOURCE_MAX_BYTES:
        raise SkillBundleError("SKILL_RESOURCE_TOO_LARGE")
    if len(body) != resource.size_bytes or hashlib.sha256(body).hexdigest() != resource.sha256:
        raise SkillBundleError("SKILL_RESOURCE_MISMATCH")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise SkillBundleError("SKILL_RESOURCE_ENCODING") from None
    if _has_control_characters(text):
        raise SkillBundleError("SKILL_RESOURCE_ENCODING")
    if resource.media_type == "application/json":
        try:
            json.loads(text)
        except (ValueError, RecursionError):
            raise SkillBundleError("SKILL_RESOURCE_INVALID") from None
    return text


def read_skill_bundle(
    *,
    package: str,
    instructions: SkillResourceManifest,
    resources: tuple[SkillResourceManifest, ...] = (),
) -> SkillBundleDescriptor:
    """Trusted registration/build step: index one approved bundle's canonical metadata.

    ``package`` and every resource path must come from trusted registration. The
    package must expose a filesystem Path root; namespace/virtual roots are not
    supported because their containment cannot be verified by this reader.
    The whole instructions resource is verified against its approved size/SHA-256 and
    parsed; companion resources are verified but not parsed. The returned
    descriptor holds metadata only, so model-step catalogs never reread bodies.
    """

    if _PACKAGE.fullmatch(package) is None:
        raise SkillBundleError("SKILL_BUNDLE_PACKAGE_INVALID")
    instructions = _revalidate(instructions)
    resources = tuple(_revalidate(resource) for resource in resources)
    bundle = _check_layout(instructions, resources)
    try:
        root = files(package)
    except (ImportError, TypeError, ValueError):
        raise SkillBundleError("SKILL_RESOURCE_UNAVAILABLE") from None
    frontmatter, _ = parse_skill_markdown(_read_verified_text(root, instructions))
    if frontmatter.name != bundle:
        raise SkillBundleError("SKILL_NAME_FOLDER_MISMATCH")
    for resource in resources:
        _read_verified_text(root, resource)
    try:
        return SkillBundleDescriptor(
            package=package,
            bundle=bundle,
            name=frontmatter.name,
            description=frontmatter.description,
            instructions=instructions,
            resources=resources,
        )
    except ValidationError:
        raise SkillBundleError("SKILL_BUNDLE_INVALID") from None
