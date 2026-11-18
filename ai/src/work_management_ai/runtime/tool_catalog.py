"""Scoped step declarations. Every selection rechecks trusted current facts.

This module performs no authentication or database reads. Composition must supply
current authenticated facts; Task6 application guards remain mandatory at dispatch.
There is no cache of effective permissions shared between scopes or revisions.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from work_management_ai.model_gateway.tool_contracts import ToolSchema
from work_management_ai.runtime.agent_registry import RegisteredToolLoopAgent
from work_management_ai.runtime.contracts import RiskLevel
from work_management_ai.runtime.tool_loop_contracts import AdmittedTool, RunScope
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry, ToolRegistryError

_RISKS = {RiskLevel.READ_ONLY: 0, RiskLevel.PROPOSAL_ONLY: 1, RiskLevel.EXPLICIT_WRITE: 2}


class ToolCatalogError(ToolPolicyError):
    """Safe preparation/revocation failure; no raw source errors."""

    def __init__(self, code: str, *, stop_segment: bool = True):
        super().__init__(code, stop_segment=stop_segment)


@dataclass(frozen=True, slots=True)
class CatalogAccess:
    scope: RunScope
    role: Literal["ADMIN", "MANAGER", "EMPLOYEE"]
    active_phase: int
    enabled: bool
    permission_revision: str
    activation_revision: str
    allowed_tools: frozenset[str]
    active_tools: frozenset[str]
    allowed_skills: frozenset[str]


class CatalogAccessPort(Protocol):
    def current(self, scope: RunScope) -> CatalogAccess:
        """Return current authenticated scope/access, never model/client claims."""
        ...


class ToolReadinessPort(Protocol):
    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        """True only for a registered, ready handler and permitted model route."""
        ...


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _copy(tool: AdmittedTool) -> AdmittedTool:
    return replace(tool, schema=tool.schema.model_copy(deep=True))


class PreparedToolSet:
    def __init__(
        self, tools: tuple[AdmittedTool, ...], fingerprint: str, revalidate: Callable[[str], bool]
    ):
        self._tools = {tool.schema.wire_name: _copy(tool) for tool in tools}
        self._fingerprint, self._revalidate, self._closed = fingerprint, revalidate, False

    @property
    def schemas(self) -> tuple[ToolSchema, ...]:
        if self._closed:
            return ()
        return tuple(v.schema.model_copy(deep=True) for v in self._tools.values())

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def resolve(self, wire_name: str) -> AdmittedTool:
        tool = self._tools.get(wire_name)
        if tool is None:
            raise ToolCatalogError("UNKNOWN_TOOL", stop_segment=False)
        if self._closed or not self._revalidate(tool.reference):
            raise ToolCatalogError("TOOL_ADMISSION_STALE")
        return _copy(tool)

    def close(self) -> None:
        self._closed = True


class ToolCatalog:
    def __init__(
        self,
        *,
        agent: RegisteredToolLoopAgent,
        tools: ToolRegistry,
        access: CatalogAccessPort,
        readiness: ToolReadinessPort,
    ):
        self.agent, self.tools, self.access, self.readiness = agent, tools, access, readiness

    def context(self, scope: RunScope) -> CatalogAccess:
        try:
            facts = self.access.current(scope)
        except Exception:
            raise ToolCatalogError("TOOL_CATALOG_UNAVAILABLE") from None
        manifest = self.agent.manifest
        if facts.scope != scope:
            raise ToolCatalogError("TOOL_CATALOG_SCOPE_MISMATCH")
        if (
            facts.enabled is not True
            or facts.active_phase < manifest.agent.activation_phase
            or facts.role not in manifest.permissions.roles
            or not facts.permission_revision
            or not facts.activation_revision
        ):
            raise ToolCatalogError("TOOL_CATALOG_INACTIVE")
        return facts

    def _available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        try:
            result = self.readiness.available(scope, reference, model_ref)
            if type(result) is not bool:
                raise ValueError("INVALID_READINESS_RESULT")
            return result
        except Exception:
            raise ToolCatalogError("TOOL_CATALOG_UNAVAILABLE") from None

    def prepare(self, scope: RunScope, *, model_ref: str) -> PreparedToolSet:
        if not model_ref or len(model_ref) > 200:
            raise ToolCatalogError("TOOL_MODEL_ROUTE_INVALID")
        facts = self.context(scope)
        manifest = self.agent.manifest
        try:
            definitions = self.tools.candidate_definitions(manifest.allowed_tools)
        except (ToolRegistryError, ValueError, ImportError, AttributeError):
            raise ToolCatalogError("TOOL_CATALOG_INVALID") from None
        admitted: list[AdmittedTool] = []
        for definition in definitions:
            schema = definition.schema
            metadata = self.tools.resolve(schema.reference).manifest
            if definition.pin not in self.agent.tool_pins:
                raise ToolCatalogError("TOOL_VERSION_CONFLICT")
            if (
                _RISKS[metadata.risk_level] > _RISKS[manifest.permissions.risk_ceiling]
                or metadata.max_attempts > 2
            ):
                raise ToolCatalogError("TOOL_CATALOG_INCOMPATIBLE")
            if (
                schema.reference not in facts.active_tools
                or schema.reference not in facts.allowed_tools
                or (metadata.roles and facts.role not in metadata.roles)
            ):
                continue
            if not self._available(scope, schema.reference, model_ref):
                continue
            assert schema.input_contract is not None
            admitted.append(
                AdmittedTool(
                    schema=schema,
                    input_contract=schema.input_contract,
                    output_contract=definition.output_contract,
                    version=definition.pin.version,
                    fingerprint=definition.pin.fingerprint,
                    risk=metadata.risk_level,
                    timeout_seconds=metadata.timeout_seconds,
                    retry_limit=metadata.max_attempts - 1,
                    idempotency="REQUIRED" if metadata.idempotency == "REQUIRED" else "NONE",
                    audit="NONE" if metadata.audit == "NONE" else "REQUIRED",
                )
            )
        if self.context(scope) != facts:
            raise ToolCatalogError("TOOL_CATALOG_CHANGED")
        fingerprint = _digest(
            {
                "scope": scope.model_dump(mode="json"),
                "role": facts.role,
                "permission_revision": facts.permission_revision,
                "activation_revision": facts.activation_revision,
                "agent": self.agent.fingerprint,
                "skill_dependencies": [p.model_dump(mode="json") for p in self.agent.skill_pins],
                "tool_dependencies": [p.model_dump(mode="json") for p in self.agent.tool_pins],
                "model_ref": model_ref,
                "tools": [(v.reference, v.version, v.fingerprint) for v in admitted],
            }
        )

        def revalidate(reference: str) -> bool:
            return (
                self.context(scope) == facts
                and reference in facts.allowed_tools
                and reference in facts.active_tools
                and self._available(scope, reference, model_ref)
            )

        return PreparedToolSet(tuple(admitted), fingerprint, revalidate)
