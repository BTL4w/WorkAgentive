"""Immutable exact-major Tool manifest registry."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel

from work_management_ai.model_gateway.tool_contracts import ToolSchema
from work_management_ai.runtime.contracts import RiskLevel
from work_management_ai.runtime.manifests import (
    ToolManifest,
    canonical_manifest_fingerprint,
    resolve_contract,
)


class ToolRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    reference: str
    manifest: ToolManifest
    fingerprint: str


class ToolRegistry:
    def __init__(self, manifests: Iterable[ToolManifest] = ()) -> None:
        entries: dict[str, RegisteredTool] = {}
        for manifest in manifests:
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
        """Strict typed native declarations; legacy registry parsing is unchanged."""
        result: list[ToolSchema] = []
        aliases: set[str] = set()
        for reference in sorted(references):
            registered = self.resolve(reference)
            input_type = resolve_contract(registered.manifest.input_contract)
            output_type = resolve_contract(registered.manifest.output_contract)
            if not (
                isinstance(input_type, type)
                and issubclass(input_type, BaseModel)
                and isinstance(output_type, type)
                and issubclass(output_type, BaseModel)
            ):
                raise ToolRegistryError("TOOL_CONTRACT_INVALID")
            schema = ToolSchema(
                reference=reference, description=registered.manifest.name, input_contract=input_type
            )
            if schema.wire_name in aliases:
                raise ToolRegistryError("TOOL_ALIAS_COLLISION")
            aliases.add(schema.wire_name)
            result.append(schema)
        return tuple(result)
