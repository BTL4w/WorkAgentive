"""Explore package is a bounded bilingual playbook, never authority."""

import hashlib
from importlib.resources import files

import pytest
from pydantic import ValidationError

from work_management_ai.runtime.manifests import SkillManifestV2, load_yaml_resource
from work_management_ai.skills.explore_work_context.contracts import ExploreWorkContextInput
from work_management_ai.tools.context.contracts import ExploreProjectsInput, ReadActorInput


def test_package_has_exact_allowlist_and_hashed_bilingual_instructions():
    pkg = "work_management_ai.skills.explore_work_context"
    manifest = load_yaml_resource(pkg, "skill.yaml", SkillManifestV2)
    body = files(pkg).joinpath(manifest.instructions.path).read_bytes()
    assert manifest.instructions.sha256 == hashlib.sha256(body).hexdigest()
    assert manifest.instructions.size_bytes == len(body)
    assert set(manifest.allowed_tools) == {
        "explore_projects@1",
        "read_project_context@1",
        "read_actor_context@1",
    }
    assert manifest.runnable_by_agents == ("orchestrator",)
    text = body.decode()
    for token in [
        "vi:",
        "en:",
        "NOT_SUPPORTED",
        "PARTIAL",
        "AMBIGUOUS",
        "source",
        "hello",
        "chào",
        "Specialist",
    ]:
        assert token in text
    assert ExploreWorkContextInput.model_fields["objective_ref"]


def test_actor_input_cannot_name_someone_else_or_grant_scope():
    with pytest.raises(ValidationError):
        ReadActorInput.model_validate({"membership_id": "bad"})
    with pytest.raises(ValidationError):
        ExploreProjectsInput(query="Project", page_size=21)


def test_read_tool_manifests_register_typed_read_only_surface():
    from work_management_ai.runtime.manifests import ToolManifest
    from work_management_ai.runtime.skill_registry import SkillRegistry
    from work_management_ai.runtime.tool_registry import ToolRegistry

    registry = ToolRegistry(
        tuple(
            load_yaml_resource("work_management_ai.tools.context", f"{name}.yaml", ToolManifest)
            for name in ("projects", "project", "actor")
        )
    )
    schemas = registry.candidate_schemas(
        ("explore_projects@1", "read_project_context@1", "read_actor_context@1")
    )
    assert len(schemas) == 3
    entry = SkillRegistry.from_resources(
        (("work_management_ai.skills.explore_work_context", "skill.yaml"),)
    ).resolve("explore_work_context@1")
    assert entry.manifest.version == "1.0.1"
    assert entry.bundle is not None
    assert entry.bundle.name == "explore-work-context"
