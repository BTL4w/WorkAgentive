"""Exercise the v2 skill backend with the shipped Explore bundle."""

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

from work_management_ai.runtime.agent_registry import AgentRegistry, CandidateEvaluator
from work_management_ai.runtime.manifests import (
    AgentManifest,
    ToolLoopAgentManifest,
    load_yaml_resource,
)
from work_management_ai.runtime.skill_loading import (
    ApprovedPackageResources,
    SkillLoader,
    SkillLoaderBackend,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_catalog import CatalogAccess, ToolCatalog
from work_management_ai.runtime.tool_loop_contracts import (
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopLimits,
)
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.skills.contracts import LoadSkillInput

SKILL = "explore_work_context@1"
LOAD_TOOL = "load_skill@1"


class SkillAccess:
    def __init__(self, scope: RunScope, tools: tuple[str, ...]):
        self.scope = scope
        self.tools = frozenset(tools)
        self.revoked = False

    def current(self, scope: RunScope) -> CatalogAccess:
        return CatalogAccess(
            scope=self.scope,
            role="EMPLOYEE",
            active_phase=5,
            enabled=True,
            permission_revision="revoked" if self.revoked else "fixture",
            activation_revision="fixture",
            allowed_tools=self.tools,
            active_tools=self.tools,
            allowed_skills=frozenset() if self.revoked else frozenset({SKILL}),
        )

    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        return scope == self.scope and reference in self.tools


class SkillToolFixture:
    def __init__(
        self,
        skills: SkillRegistry,
        tools: ToolRegistry,
        evaluators: Mapping[str, CandidateEvaluator],
        context_references: tuple[str, ...],
    ) -> None:
        manifest = load_yaml_resource(
            "work_management_ai.agents.orchestrator", "agent.yaml", AgentManifest
        ).model_dump(mode="json")
        manifest["agent"]["version"] = "2.0.0"
        manifest.update(
            schema_version="3.0",
            runtime={
                "workflow": "tool_loop.v1",
                "checkpoint": "durable",
                "model_policy": "native_tools",
            },
            allowed_tools=(LOAD_TOOL, *context_references),
            allowed_skills=(SKILL,),
            evaluators=tuple(evaluators),
        )
        self.agent = AgentRegistry(
            skill_registry=skills,
            tool_registry=tools,
            evaluator_ids=frozenset(),
            evaluator_bindings=evaluators,
        ).register_tool_loop(ToolLoopAgentManifest.model_validate(manifest))
        self.skills = skills
        self.tools = tools
        self.version = tools.resolve(LOAD_TOOL).manifest.version
        bundle = skills.resolve(SKILL).bundle
        assert bundle is not None
        self.resources = ApprovedPackageResources({SKILL: bundle.package})

    def bind(self, scope: RunScope) -> tuple[SkillAccess, SkillLoaderBackend]:
        access = SkillAccess(scope, self.agent.manifest.allowed_tools)
        catalog = ToolCatalog(agent=self.agent, tools=self.tools, access=access, readiness=access)
        return access, SkillLoaderBackend(
            SkillLoader(
                skills=self.skills,
                resources=self.resources,
                catalog=catalog,
                model_ref="offline-fixture",
            )
        )

    def invocation(self, scope: RunScope, args: LoadSkillInput) -> ToolInvocationContext:
        return ToolInvocationContext(
            scope=scope,
            lease=RunLease(
                scope=scope,
                lease_id=uuid4(),
                job_id=uuid4(),
                worker_id="fixture",
                fencing_token=1,
                expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            ),
            invocation_id=uuid4(),
            call_id="fixture-call",
            tool_reference=LOAD_TOOL,
            tool_version=self.version,
            arguments_fingerprint="f" * 64,
            canonical_arguments=args,
            policy_version="fixture",
            receipt_id=uuid4(),
            budget_allocation_id=uuid4(),
            limits=ToolLoopLimits(),
        )
