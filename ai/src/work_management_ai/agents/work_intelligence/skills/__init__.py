"""Progressive loader for the Work Intelligence Skill."""

from functools import cached_property

from work_management_ai.runtime.manifests import SkillManifestV2
from work_management_ai.runtime.skill_bundle import read_skill_instructions
from work_management_ai.runtime.skill_registry import RegisteredSkill, SkillRegistry

_PACKAGE = "work_management_ai.skills.answer_work_question"
_REFERENCE = "answer_work_question@1"


class AnswerWorkQuestionSkillLoader:
    @cached_property
    def _skill(self) -> RegisteredSkill:
        return SkillRegistry.from_resources(((_PACKAGE, "skill.yaml"),)).resolve(_REFERENCE)

    def catalog(self) -> tuple[tuple[str, str], ...]:
        if self._skill.bundle is None:
            raise ValueError("SKILL_MANIFEST_INCOMPATIBLE")
        return ((_REFERENCE, self._skill.bundle.description),)

    def load(self, reference: str) -> tuple[SkillManifestV2, str]:
        if reference != _REFERENCE:
            raise ValueError("SKILL_NOT_ALLOWED")
        manifest = self._skill.manifest
        if not isinstance(manifest, SkillManifestV2):
            raise ValueError("SKILL_MANIFEST_INCOMPATIBLE")
        return manifest, read_skill_instructions(_PACKAGE, manifest)


__all__ = ["AnswerWorkQuestionSkillLoader"]
