"""Exercise the packaged Skill artifacts through the verified reader."""

from importlib.resources import files

import pytest

from work_management_ai.runtime.manifests import SkillManifestV2, load_skill_manifest_resource
from work_management_ai.runtime.skill_bundle import read_skill_bundle, read_skill_instructions
from work_management_ai.runtime.skill_registry import SkillRegistry

_SKILLS = (
    "analyze_workload",
    "answer_work_question",
    "compare_daily_update_evidence",
    "create_project_plan",
    "draft_management_report",
    "explain_verified_risk",
    "explore_work_context",
    "extract_daily_update",
    "recommend_project_team",
    "review_evidence_concerns",
    "revise_project_plan",
    "summarize_verified_project_metrics",
    "web_search_and_summarize",
)
_ROOT = "work_management_ai.skills"


def test_skill_inventory_matches_packaged_sidecars() -> None:
    assert {
        entry.name for entry in files(_ROOT).iterdir() if entry.joinpath("skill.yaml").is_file()
    } == set(_SKILLS)


@pytest.mark.parametrize("name", _SKILLS)
def test_packaged_skill_loads_verified_canonical_bundle(name: str) -> None:
    package = f"{_ROOT}.{name}"
    manifest = load_skill_manifest_resource(package, "skill.yaml")
    assert isinstance(manifest, SkillManifestV2)
    descriptor = read_skill_bundle(
        package=package, instructions=manifest.instructions, resources=manifest.resources
    )
    bundle_name = name.replace("_", "-")
    assert descriptor.name == bundle_name
    assert descriptor.description == manifest.description
    assert descriptor.instructions.path == f"{bundle_name}/SKILL.md"
    root = files(package)
    assert sorted(
        f"{entry.name}/SKILL.md" for entry in root.iterdir() if entry.joinpath("SKILL.md").is_file()
    ) == [descriptor.instructions.path]
    assert not root.joinpath("SKILL.md").is_file()
    entry = SkillRegistry.from_resources(((package, "skill.yaml"),)).resolve(f"{name}@1")
    assert entry.bundle == descriptor


@pytest.mark.parametrize("name", _SKILLS)
def test_packaged_instructions_do_not_repeat_frontmatter(name: str) -> None:
    package = f"{_ROOT}.{name}"
    manifest = load_skill_manifest_resource(package, "skill.yaml")
    assert isinstance(manifest, SkillManifestV2)
    instructions = read_skill_instructions(package, manifest)
    assert not instructions.lstrip().startswith("---\nname:")
