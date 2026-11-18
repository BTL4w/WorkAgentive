"""Step schemas never turn a cached grant into enduring execution authority."""

from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, ValidationError

from work_management_ai.runtime.agent_registry import AgentRegistry, AgentRegistryError
from work_management_ai.runtime.contracts import AgentId
from work_management_ai.runtime.manifests import (
    AgentManifest,
    SkillManifest,
    ToolLoopAgentManifest,
    canonical_manifest_fingerprint,
    load_yaml_resource,
)
from work_management_ai.runtime.tool_catalog import ToolCatalog, ToolCatalogError
from work_management_ai.runtime.tool_registry import ToolRegistry, ToolRegistryError
from work_management_ai.tools.context import contracts as context_contracts

from .catalog_fakes import FIXTURE_EVALUATOR_BINDINGS, CatalogFixture, candidate_manifest, load_tool


def test_stable_metadata_schema_copy_and_closed_selection():
    f = CatalogFixture()
    admitted = f.catalog.prepare(f.scope, model_ref="configured-model")
    again = f.catalog.prepare(f.scope, model_ref="configured-model")
    assert [s.reference for s in admitted.schemas] == ["load_skill@1", "read_actor_context@1"]
    assert admitted.fingerprint == again.fingerprint
    schema = admitted.schemas[0]
    schema.parameters.clear()
    assert "properties" in admitted.schemas[0].parameters
    tool = admitted.resolve("load_skill")
    tool.schema.parameters.clear()
    assert "properties" in admitted.resolve("load_skill").schema.parameters
    admitted.close()
    with pytest.raises(ToolCatalogError):
        admitted.resolve("load_skill")
    assert again.resolve("load_skill").reference == "load_skill@1"


def test_revocation_invalidates_old_selection_and_filters_new_schema():
    f = CatalogFixture()
    previous = f.catalog.prepare(f.scope, model_ref="same-model")
    f.revoke("read_actor_context@1")
    assert "read_actor_context@1" not in {
        s.reference for s in f.catalog.prepare(f.scope, model_ref="same-model").schemas
    }
    with pytest.raises(ToolCatalogError):
        previous.resolve("read_actor_context")


def test_role_activation_and_readiness_are_intersected():
    f = CatalogFixture()
    f.access.value = replace(f.access.value, role="EMPLOYEE")
    assert [s.reference for s in f.catalog.prepare(f.scope, model_ref="m").schemas] == [
        "load_skill@1"
    ]
    f.access.value = replace(f.access.value, active_tools=frozenset())
    assert f.catalog.prepare(f.scope, model_ref="m").schemas == ()
    f.access.value = replace(f.access.value, enabled=False)
    with pytest.raises(ToolCatalogError):
        f.catalog.prepare(f.scope, model_ref="m")


def test_backend_preparation_error_is_not_false_absence():
    f = CatalogFixture()
    f.readiness.error = True
    with pytest.raises(ToolCatalogError, match="TOOL_CATALOG_UNAVAILABLE"):
        f.catalog.prepare(f.scope, model_ref="m")
    f.readiness.error = False
    f.access.error = True
    with pytest.raises(ToolCatalogError, match="TOOL_CATALOG_UNAVAILABLE"):
        f.catalog.prepare(f.scope, model_ref="m")


def test_scope_revision_and_model_bind_catalog_identity():
    f = CatalogFixture()
    first = f.catalog.prepare(f.scope, model_ref="m")
    with pytest.raises(ToolCatalogError):
        f.catalog.prepare(f.scope.model_copy(update={"organization_id": uuid4()}), model_ref="m")
    assert f.catalog.prepare(f.scope, model_ref="other-model").fingerprint != first.fingerprint
    other_scope = f.scope.model_copy(update={"membership_id": uuid4()})
    f.access.value = replace(f.access.value, scope=other_scope)
    assert f.catalog.prepare(other_scope, model_ref="m").fingerprint != first.fingerprint
    with pytest.raises(ToolCatalogError):
        first.resolve("load_skill")
    f.access.value = replace(f.access.value, permission_revision="p2")
    second = f.catalog.prepare(other_scope, model_ref="m")
    f.access.value = replace(f.access.value, activation_revision="a2")
    assert second.fingerprint != f.catalog.prepare(other_scope, model_ref="m").fingerprint


def test_wire_alias_collision_is_rejected_before_selection():
    original = load_tool()
    registry = ToolRegistry(
        [
            original.model_copy(update={"name": "scope.read"}),
            original.model_copy(update={"name": "scope__read"}),
        ]
    )
    with pytest.raises(ToolRegistryError, match="TOOL_ALIAS_COLLISION"):
        registry.candidate_schemas(("scope.read@1", "scope__read@1"))


def test_tool_catalog_rejects_same_major_replacement_after_agent_registration():
    f = CatalogFixture()
    prior = f.tools.resolve("load_skill@1").manifest
    replacement = ToolRegistry(
        [
            prior.model_copy(update={"version": "1.0.3"}),
            f.tools.resolve("read_actor_context@1").manifest,
        ]
    )
    catalog = ToolCatalog(agent=f.agent, tools=replacement, access=f.access, readiness=f.readiness)
    with pytest.raises(ToolCatalogError, match="TOOL_VERSION_CONFLICT"):
        catalog.prepare(f.scope, model_ref="m")


def test_tool_catalog_rejects_contract_drift_at_same_manifest_version(
    monkeypatch: pytest.MonkeyPatch,
):
    class ChangedActorInput(BaseModel):
        sections: tuple[str, ...] = Field(description="Altered declaration for the same tool.")

    f = CatalogFixture()
    monkeypatch.setattr(context_contracts, "ReadActorInput", ChangedActorInput)
    with pytest.raises(ToolCatalogError, match="TOOL_VERSION_CONFLICT"):
        f.catalog.prepare(f.scope, model_ref="m")


def test_candidate_agent_is_orchestrator_only_and_opt_in():
    f = CatalogFixture()
    with pytest.raises(AgentRegistryError):
        f.registry.resolve_tool_loop("2.0.0", active_phase=5)
    assert f.registry.resolve_tool_loop("2.0.0", active_phase=5, enabled=True) == f.agent
    with pytest.raises(AgentRegistryError):
        f.registry.resolve_tool_loop("2.0.0", active_phase=1, enabled=True)
    data = candidate_manifest().model_dump(mode="json")
    data["agent"]["id"] = AgentId.PLANNING
    with pytest.raises(ValidationError):
        ToolLoopAgentManifest.model_validate(data)
    data = candidate_manifest().model_dump(mode="json")
    data["runtime"]["limits"]["root_model_requests"] = 9
    with pytest.raises(ValidationError):
        ToolLoopAgentManifest.model_validate(data)
    with pytest.raises(AgentRegistryError, match="AGENT_TOOL_NOT_REGISTERED"):
        f.registry.register_tool_loop(candidate_manifest(tools=("delegate_agent@1",)))


def test_legacy_manifest_fingerprints_remain_identical():
    agent = load_yaml_resource(
        "work_management_ai.agents.orchestrator", "agent.yaml", AgentManifest
    )
    skill = load_yaml_resource("runtime", "legacy_analyze_workload.yaml", SkillManifest)
    assert (
        canonical_manifest_fingerprint(agent)
        == "34510dc199279165f220e1bd1d4c7ff0d3148157d7844d2511391b62ac16038f"
    )
    assert (
        canonical_manifest_fingerprint(skill)
        == "3bef158f4dbe17202b0cde06b0795d988fc32c08acc554eb68f3410be9d83ee5"
    )


def test_candidate_does_not_declare_unknown_capability_or_legacy_retry_cap():
    data = candidate_manifest().model_dump(mode="json")
    data["capabilities"] = ["admin.grant_permission"]
    with pytest.raises(ValidationError):
        ToolLoopAgentManifest.model_validate(data)
    data = candidate_manifest().model_dump(mode="json")
    data["runtime"]["max_model_attempts"] = 8
    with pytest.raises(ValidationError):
        ToolLoopAgentManifest.model_validate(data)


def test_closed_set_has_no_callable_schema_surface():
    f = CatalogFixture()
    selected = f.catalog.prepare(f.scope, model_ref="m")
    selected.close()
    assert selected.schemas == ()


def test_candidate_registration_rejects_legacy_skill_without_resource_contract():
    from work_management_ai.runtime.manifests import SkillManifest
    from work_management_ai.runtime.skill_registry import SkillRegistry

    # A valid legacy skill stays valid, but is insufficient for a native package loader.
    legacy = SkillManifest.model_validate(
        {
            "schema_version": "1.0",
            "name": "fixture_context",
            "version": "1.0.0",
            "owner": "tests",
            "description": "Legacy bodyless declaration",
            "runnable_by_agents": ["orchestrator"],
            "allowed_tools": ["read_actor_context@1"],
            "risk_level": "READ_ONLY",
            "input_contract": "work_management_ai.runtime.contracts.AgentHandoff",
            "output_contract": "work_management_ai.runtime.contracts.AgentResult",
            "evaluators": ["fixture@1"],
        }
    )
    with pytest.raises(AgentRegistryError, match="AGENT_SKILL_SCHEMA_INCOMPATIBLE"):
        CatalogFixture(skill_registry=SkillRegistry([legacy]))


def test_candidate_rejects_three_attempt_tool_policy():
    f = CatalogFixture()
    tools = ToolRegistry(
        [
            load_tool().model_copy(update={"max_attempts": 3}),
            f.tools.resolve("read_actor_context@1").manifest,
        ]
    )
    agent = AgentRegistry(
        skill_registry=f.skills,
        tool_registry=tools,
        evaluator_ids=frozenset({"orchestrator_plan@1"}),
        evaluator_bindings=FIXTURE_EVALUATOR_BINDINGS,
    ).register_tool_loop(f.agent.manifest)
    catalog = ToolCatalog(agent=agent, tools=tools, access=f.access, readiness=f.readiness)
    with pytest.raises(ToolCatalogError, match="TOOL_CATALOG_INCOMPATIBLE"):
        catalog.prepare(f.scope, model_ref="m")


@pytest.mark.parametrize(
    "contract",
    [
        "work_management_ai.runtime.contracts.RiskLevel",
        "work_management_ai.runtime.manifests.canonical_manifest_fingerprint",
    ],
)
def test_candidate_agent_contract_must_be_typed_model(contract: str):
    f = CatalogFixture()
    data = candidate_manifest().model_dump(mode="json")
    data["agent"]["version"] = "2.0.1"
    data["contracts"]["input"] = contract
    with pytest.raises(AgentRegistryError, match="AGENT_CONTRACT_INVALID"):
        f.registry.register_tool_loop(ToolLoopAgentManifest.model_validate(data))
