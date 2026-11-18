"""Immutable exact-major Tool manifest registry."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ValidationError

from work_management_ai.model_gateway.tool_contracts import ToolSchema
from work_management_ai.runtime.contracts import JsonValue, RiskLevel
from work_management_ai.runtime.manifests import (
    ToolManifest,
    ToolManifestV2,
    canonical_manifest_fingerprint,
    load_tool_manifest_resource,
    resolve_contract,
)
from work_management_ai.runtime.tool_loop_contracts import VersionPin


class ToolRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    reference: str
    manifest: ToolManifest | ToolManifestV2
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CandidateToolDefinition:
    schema: ToolSchema
    output_contract: type[BaseModel]
    pin: VersionPin


def _check_parameter_descriptions(schema: dict[str, JsonValue]) -> None:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for parameter in properties.values():
            if not isinstance(parameter, dict):
                raise ToolRegistryError("TOOL_PARAMETER_DESCRIPTION_REQUIRED")
            description = parameter.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ToolRegistryError("TOOL_PARAMETER_DESCRIPTION_REQUIRED")
            _check_parameter_descriptions(parameter)
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        for definition in definitions.values():
            if isinstance(definition, dict):
                _check_parameter_descriptions(definition)
    items = schema.get("items")
    if isinstance(items, dict):
        _check_parameter_descriptions(items)
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    _check_parameter_descriptions(branch)


class ToolRegistry:
    @classmethod
    def from_resources(cls, resources: Iterable[tuple[str, str]]) -> "ToolRegistry":
        return cls(
            load_tool_manifest_resource(package, resource) for package, resource in resources
        )

    def __init__(self, manifests: Iterable[ToolManifest | ToolManifestV2] = ()) -> None:
        entries: dict[str, RegisteredTool] = {}
        for manifest in manifests:
            if isinstance(manifest, ToolManifestV2):
                manifest = ToolManifestV2.model_validate(manifest.model_dump())
            if manifest.risk_level is RiskLevel.EXPLICIT_WRITE and (
                manifest.idempotency != "REQUIRED"
                or manifest.audit != "REQUIRED"
                or "EMPLOYEE" in manifest.roles
                or not manifest.roles
            ):
                raise ToolRegistryError("TOOL_EXPLICIT_WRITE_POLICY_INVALID")
            for contract_path in (manifest.input_contract, manifest.output_contract):
                try:
                    resolve_contract(contract_path)
                except (AttributeError, ImportError, ValueError) as exc:
                    raise ToolRegistryError("TOOL_CONTRACT_INVALID") from exc
            major = manifest.version.split(".", maxsplit=1)[0]
            reference = f"{manifest.name}@{major}"
            if reference in entries:
                raise ToolRegistryError("TOOL_MAJOR_ALREADY_REGISTERED")
            entries[reference] = RegisteredTool(
                reference=reference,
                manifest=manifest,
                fingerprint=canonical_manifest_fingerprint(manifest),
            )
        self._entries: Mapping[str, RegisteredTool] = MappingProxyType(entries)

    def resolve(self, reference: str) -> RegisteredTool:
        try:
            return self._entries[reference]
        except KeyError as exc:
            raise ToolRegistryError("TOOL_NOT_REGISTERED") from exc

    def candidate_schemas(self, references: tuple[str, ...]) -> tuple[ToolSchema, ...]:
        return tuple(definition.schema for definition in self.candidate_definitions(references))

    def candidate_definitions(
        self, references: tuple[str, ...]
    ) -> tuple[CandidateToolDefinition, ...]:
        """Strict typed native declarations; legacy registry parsing is unchanged."""
        result: list[CandidateToolDefinition] = []
        aliases: set[str] = set()
        for reference in sorted(references):
            registered = self.resolve(reference)
            if not isinstance(registered.manifest, ToolManifestV2):
                raise ToolRegistryError("TOOL_MANIFEST_SCHEMA_INCOMPATIBLE")
            input_type = resolve_contract(registered.manifest.input_contract)
            output_type = resolve_contract(registered.manifest.output_contract)
            if not (
                isinstance(input_type, type)
                and issubclass(input_type, BaseModel)
                and isinstance(output_type, type)
                and issubclass(output_type, BaseModel)
            ):
                raise ToolRegistryError("TOOL_CONTRACT_INVALID")
            try:
                schema = ToolSchema(
                    reference=reference,
                    description=registered.manifest.description,
                    input_contract=input_type,
                )
            except ValidationError:
                raise ToolRegistryError("TOOL_DECLARATION_INVALID") from None
            _check_parameter_descriptions(schema.parameters)
            if schema.wire_name in aliases:
                raise ToolRegistryError("TOOL_ALIAS_COLLISION")
            aliases.add(schema.wire_name)
            payload = json.dumps(
                {
                    "manifest": registered.fingerprint,
                    "input": schema.parameters,
                    "output": output_type.model_json_schema(),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            result.append(
                CandidateToolDefinition(
                    schema=schema,
                    output_contract=output_type,
                    pin=VersionPin(
                        reference=reference,
                        version=registered.manifest.version,
                        fingerprint=hashlib.sha256(payload).hexdigest(),
                    ),
                )
            )
        return tuple(result)
