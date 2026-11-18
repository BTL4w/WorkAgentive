"""The candidate model receives documented parameters from registered contracts."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from work_management_ai.runtime.manifests import (
    ToolManifest,
    ToolManifestV2,
    canonical_manifest_fingerprint,
    load_tool_manifest_resource,
)
from work_management_ai.runtime.tool_registry import ToolRegistry, ToolRegistryError
from work_management_ai.tools.context import contracts

from .catalog_fakes import load_tool


@pytest.mark.parametrize(
    ("resource", "name", "parameters"),
    [
        ("projects.yaml", "explore_projects", {"query", "cursor", "page_size"}),
        ("project.yaml", "read_project_context", {"reference", "sections"}),
        ("actor.yaml", "read_actor_context", {"sections"}),
        ("load.yaml", "load_skill", {"skill_ref", "resource_id"}),
    ],
)
def test_context_resources_export_model_facing_tool_definitions(
    resource: str, name: str, parameters: set[str]
) -> None:
    package = "skills" if name == "load_skill" else "context"
    manifest = load_tool_manifest_resource(f"work_management_ai.tools.{package}", resource)
    assert isinstance(manifest, ToolManifestV2)
    (schema,) = ToolRegistry([manifest]).candidate_schemas((f"{name}@1",))
    assert schema.wire_name == name
    assert schema.description == manifest.description
    properties = schema.parameters["properties"]
    assert isinstance(properties, dict)
    assert properties.keys() == parameters
    for field in properties.values():
        assert isinstance(field, dict)
        assert field.get("description")


@pytest.mark.parametrize("description", [None, "", "  ", "Load Skill", "load_skill@1", "x" * 4001])
def test_candidate_definition_requires_meaningful_description(description: str | None) -> None:
    data = load_tool().model_dump()
    if description is None:
        del data["description"]
    else:
        data["description"] = description
    with pytest.raises(ValidationError):
        ToolManifestV2.model_validate(data)


def test_legacy_tool_remains_resolvable_and_is_excluded_from_candidate_definitions() -> None:
    manifest = load_tool_manifest_resource("work_management_ai.tools.automation", "tool.yaml")
    assert isinstance(manifest, ToolManifest)
    registry = ToolRegistry([manifest])
    assert registry.resolve("automation.preview@1").fingerprint == (
        "c640c62628d470401f7869476210a9e8f46ab452710cf93a2e1fcb5cc0c2669b"
    )
    assert (
        canonical_manifest_fingerprint(manifest)
        == registry.resolve("automation.preview@1").fingerprint
    )
    with pytest.raises(ToolRegistryError, match="TOOL_MANIFEST_SCHEMA_INCOMPATIBLE"):
        registry.candidate_schemas(("automation.preview@1",))


@pytest.mark.parametrize("side", ["input_contract", "output_contract"])
def test_candidate_definition_rejects_contract_that_is_not_a_model(side: str) -> None:
    manifest = load_tool().model_copy(
        update={side: "work_management_ai.runtime.contracts.RiskLevel"}
    )
    with pytest.raises(ToolRegistryError, match="TOOL_CONTRACT_INVALID"):
        ToolRegistry([manifest]).candidate_schemas(("load_skill@1",))


class UndocumentedItem(BaseModel):
    text: str


class NestedInput(BaseModel):
    items: list[UndocumentedItem] = Field(description="Items to examine.")


def test_candidate_definition_requires_descriptions_for_nested_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contracts, "NestedInput", NestedInput, raising=False)
    manifest = load_tool().model_copy(
        update={"input_contract": "work_management_ai.tools.context.contracts.NestedInput"}
    )
    with pytest.raises(ToolRegistryError, match="TOOL_PARAMETER_DESCRIPTION_REQUIRED"):
        ToolRegistry([manifest]).candidate_schemas(("load_skill@1",))
