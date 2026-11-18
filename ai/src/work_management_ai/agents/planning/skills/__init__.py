"""Progressive Planning Skill loader."""

from functools import cached_property

from work_management_ai.runtime.manifests import SkillManifestV2
from work_management_ai.runtime.skill_bundle import read_skill_instructions
from work_management_ai.runtime.skill_registry import SkillRegistry

_PACKAGES = {
    "create_project_plan@1": "work_management_ai.skills.create_project_plan",
    "revise_project_plan@1": "work_management_ai.skills.revise_project_plan",
}


class PlanningSkillLoader:
    @cached_property
    def _skills(self) -> SkillRegistry:
        return SkillRegistry.from_resources(
            (package, "skill.yaml") for package in _PACKAGES.values()
        )

    def catalog(self) -> tuple[tuple[str, str], ...]:
        catalog: list[tuple[str, str]] = []
        for reference in _PACKAGES:
            entry = self._skills.resolve(reference)
            if entry.bundle is None:
                raise ValueError("SKILL_MANIFEST_INCOMPATIBLE")
            catalog.append((reference, entry.bundle.description))
        return tuple(catalog)

    def load(self, reference: str) -> tuple[SkillManifestV2, str]:
        package = _PACKAGES.get(reference)
        if package is None:
            raise ValueError("PLANNING_SKILL_NOT_ALLOWED")
        manifest = self._skills.resolve(reference).manifest
        if not isinstance(manifest, SkillManifestV2):
            raise ValueError("SKILL_MANIFEST_INCOMPATIBLE")
        return manifest, read_skill_instructions(package, manifest)


__all__ = ["PlanningSkillLoader"]
