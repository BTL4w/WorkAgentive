"""Authenticated test policy facts, not a production source of authority."""

from dataclasses import replace
from uuid import uuid4

from work_management_ai.runtime.agent_registry import AgentRegistry, RegisteredToolLoopAgent
from work_management_ai.runtime.manifests import (
    AgentManifest,
    ToolLoopAgentManifest,
    ToolManifestV2,
    load_tool_manifest_resource,
    load_yaml_resource,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_catalog import CatalogAccess, ToolCatalog
from work_management_ai.runtime.tool_loop_contracts import RunScope
from work_management_ai.runtime.tool_registry import ToolRegistry


async def fixture_evaluator() -> object:
    return "test binding"


FIXTURE_EVALUATOR_BINDINGS = {
    "orchestrator_plan@1": fixture_evaluator,
    "fixture@1": fixture_evaluator,
}


def candidate_manifest(
    *, tools: tuple[str, ...] = ("load_skill@1",), skills: tuple[str, ...] = ()
) -> ToolLoopAgentManifest:
    value = load_yaml_resource(
        "work_management_ai.agents.orchestrator", "agent.yaml", AgentManifest
    ).model_dump(mode="json")
    value.update(
        schema_version="3.0",
        runtime={
            "workflow": "tool_loop.v1",
            "checkpoint": "durable",
            "model_policy": "native_tools",
            "limits": {},
        },
        allowed_tools=tools,
        allowed_skills=skills,
    )
    value["agent"]["version"] = "2.0.0"
    return ToolLoopAgentManifest.model_validate(value)


def load_tool() -> ToolManifestV2:
    manifest = load_tool_manifest_resource("work_management_ai.tools.skills", "load.yaml")
    assert isinstance(manifest, ToolManifestV2)
    return manifest


class Access:
    def __init__(self, scope: RunScope):
        self.value = CatalogAccess(
            scope=scope,
            role="MANAGER",
            active_phase=5,
            enabled=True,
            permission_revision="p1",
            activation_revision="a1",
            allowed_tools=frozenset({"load_skill@1", "read_actor_context@1"}),
            active_tools=frozenset({"load_skill@1", "read_actor_context@1"}),
            allowed_skills=frozenset({"fixture_context@1"}),
        )
        self.error = False

    def current(self, scope: RunScope) -> CatalogAccess:
        if self.error:
            raise RuntimeError("private account error")
        return self.value


class Readiness:
    def __init__(self):
        self.unavailable: set[str] = set()
        self.error = False

    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        if self.error:
            raise RuntimeError("private provider error")
        return reference not in self.unavailable


class CatalogFixture:
    def __init__(self, *, skill_registry: SkillRegistry | None = None):
        self.scope = RunScope(**{name: uuid4() for name in RunScope.model_fields})
        self.access, self.readiness = Access(self.scope), Readiness()
        loader = load_tool()
        reader = load_tool_manifest_resource("work_management_ai.tools.context", "actor.yaml")
        assert isinstance(reader, ToolManifestV2)
        reader = reader.model_copy(update={"roles": ("ADMIN", "MANAGER")})
        self.tools = ToolRegistry([loader, reader])
        self.skills = skill_registry or SkillRegistry()
        self.registry = AgentRegistry(
            skill_registry=self.skills,
            tool_registry=self.tools,
            evaluator_ids=frozenset({"orchestrator_plan@1", "fixture@1"}),
            evaluator_bindings=FIXTURE_EVALUATOR_BINDINGS,
        )
        self.agent: RegisteredToolLoopAgent = self.registry.register_tool_loop(
            candidate_manifest(
                tools=("load_skill@1", "read_actor_context@1"),
                skills=("fixture_context@1",) if skill_registry else (),
            )
        )
        self.catalog = ToolCatalog(
            agent=self.agent, tools=self.tools, access=self.access, readiness=self.readiness
        )

    def revoke(self, reference: str):
        self.access.value = replace(
            self.access.value,
            allowed_tools=self.access.value.allowed_tools - {reference},
            permission_revision="p2",
        )
