"""Opt-in hosted decisions over synthetic facts and packaged tool declarations."""

import asyncio
import json
import os
from dataclasses import replace
from time import perf_counter
from typing import Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel, SecretStr

from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
from work_management_ai.agents.orchestrator.prompts.tool_loop_v1 import PROMPT_VERSION
from work_management_ai.agents.orchestrator.tool_loop_contracts import PublishedResponse
from work_management_ai.evaluation.orchestrator_tool_loop import (
    REQUIRED_CASES,
    RecordingGateway,
    ToolLoopCase,
    Trial,
)
from work_management_ai.evaluation.tool_loop_harness import (
    FIXTURE_TIME,
    FixtureToolBackend,
    make_harness,
)
from work_management_ai.model_gateway.openai_tools import OpenAINativeToolGateway
from work_management_ai.model_gateway.tool_contracts import ToolModelRequest, ToolModelResponse
from work_management_ai.runtime.contracts import AgentId
from work_management_ai.runtime.manifests import (
    GoalDirectedAgentManifest,
    canonical_manifest_fingerprint,
    load_yaml_resource,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_loop_contracts import BackendToolResult, ToolInvocationContext
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    ExploreProjectsInputV103,
    ReadCoverage,
    SourceContent,
)

type Locale = Literal["vi", "en"]
type Probe = Literal["greeting", "ambiguous", "32h", "48h", "skill-overview"]


class BoundedNativeGateway(RecordingGateway):
    """Keep provider output and request bounds explicit for ten synthetic trials."""

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        bounded = request.model_copy(
            update={"max_output_tokens": min(request.max_output_tokens, 1200)}
        )
        return await super().generate_step(bounded)


def _case(locale: Locale, probe: Probe) -> ToolLoopCase:
    scenario = (
        "greeting"
        if probe == "greeting"
        else "workload"
        if probe in {"32h", "48h"}
        else "ambiguity"
    )
    case = next(c for c in REQUIRED_CASES if c.locale == locale and c.scenario == scenario)
    if probe != "skill-overview":
        return case
    text = (
        "Dùng kỹ năng khám phá ngữ cảnh công việc để xác minh và tóm tắt dự án Sao Mai."
        if locale == "vi"
        else "Use the work context exploration skill to verify and summarize Project Sao Mai."
    )
    return replace(case, prompt=text)


async def _project_overview(call: ToolInvocationContext, args: BaseModel) -> BackendToolResult:
    if not isinstance(args, ExploreProjectsInputV103) or not args.include_overview_if_unique:
        raise ValueError("COMBINED_OVERVIEW_NOT_REQUESTED")
    if "sao mai" not in args.query.casefold():
        raise ValueError("UNEXPECTED_SYNTHETIC_PROJECT")
    ref_id = uuid4()
    return BackendToolResult(
        output=ContextReadObservation(
            scope=call.scope,
            objective_ref=call.scope.turn_id,
            invocation_id=call.invocation_id,
            tool_reference="explore_projects@1",
            tool_version=call.tool_version,
            observed_at=FIXTURE_TIME,
            outcome="OK",
            context_refs=(
                ContextRef(
                    ref_id=ref_id,
                    scope=call.scope,
                    resource_type="project",
                    resource_id=uuid4(),
                    version=3,
                    fingerprint="a" * 64,
                    observed_at=FIXTURE_TIME,
                    source="project_service",
                ),
            ),
            entities=(
                EntityResolution(
                    query_fingerprint="b" * 64,
                    resolution="UNIQUE",
                    candidate_ref_ids=(ref_id,),
                    selected_ref_id=ref_id,
                ),
            ),
            facts=(
                ContextFact(ref_id=ref_id, field="project.name", value="Sao Mai"),
                ContextFact(ref_id=ref_id, field="project.version", value=3),
            ),
            source_content_refs=(SourceContent(ref_id=ref_id, text="Synthetic launch plan"),),
            coverage=(
                ReadCoverage(
                    requested=("overview",),
                    returned=("overview",),
                    status="COMPLETE",
                    query_fingerprint="b" * 64,
                ),
            ),
        )
    )


@pytest.mark.asyncio
async def test_synthetic_unique_overview_fixture_runs_through_tool_pipeline() -> None:
    steps: tuple[dict[str, object], ...] = (
        {
            "tool_call": {
                "call_id": "overview-test",
                "name": "explore_projects",
                "arguments": {"query": "Sao Mai", "include_overview_if_unique": True},
            },
        },
        {
            "final": {
                "outcome": "ANSWER",
                "summary": "Synthetic launch plan",
                "blocks": [{"kind": "text", "text": "Synthetic launch plan", "citations": []}],
                "artifact_state": "NONE",
                "stop_reason": "SYNTHETIC_OVERVIEW",
            },
        },
    )
    orch, value, scope, lease, gateway, log, _, _ = make_harness(
        responses=steps,
        use_packaged_manifest=True,
        backends={"explore_projects@1": FixtureToolBackend(_project_overview)},
    )
    result = await orch.run_segment(value, scope=scope, checkpoint=None, lease=lease)
    assert isinstance(result, PublishedResponse)
    assert result.output.status is OrchestratorStatus.COMPLETED
    assert sum(event.kind == "tool.completed" for event in log.records) == 1
    gateway.assert_complete()


@pytest.fixture(scope="module")
def hosted_settings() -> tuple[str, SecretStr]:
    model = os.getenv("APP_AI_MODEL", "").strip()
    credential = os.getenv("APP_OPENAI_API_KEY", "").strip()
    if not model or not credential or os.getenv("APP_AI_PROVIDER") != "openai":
        pytest.fail("Pilot requires configured OpenAI provider, main model and credential.")
    if any(os.getenv(name) for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_PROXY")):
        pytest.fail("Pilot requires the default OpenAI endpoint and no OpenAI proxy override.")
    if any(
        os.getenv(name, "").casefold() in ("1", "true", "yes", "on")
        for name in (
            "APP_LANGSMITH_TRACING",
            "LANGSMITH_TRACING",
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING",
            "LANGCHAIN_TRACING_V2",
        )
    ):
        pytest.fail("Pilot requires automatic tracing to be disabled.")
    return model, SecretStr(credential)


@pytest.mark.asyncio
@pytest.mark.live_provider
@pytest.mark.skipif(
    os.getenv("RUN_ORCHESTRATOR_HOSTED_PILOT") != "1",
    reason="hosted Orchestrator pilot is opt-in",
)
@pytest.mark.parametrize("locale", ["vi", "en"])
@pytest.mark.parametrize("probe", ["greeting", "ambiguous", "32h", "48h", "skill-overview"])
async def test_hosted_native_tool_loop(
    locale: Locale, probe: Probe, hosted_settings: tuple[str, SecretStr]
) -> None:
    model, credential = hosted_settings
    gateway = BoundedNativeGateway(OpenAINativeToolGateway(model_name=model, api_key=credential))
    case = _case(locale, probe)
    trial = Trial(case, probe if probe in ("ambiguous", "32h", "48h") else "default")
    specialist_manifests = tuple(
        load_yaml_resource(
            f"work_management_ai.agents.{agent.value}",
            "agent.v2.yaml",
            GoalDirectedAgentManifest,
        )
        for agent in (AgentId.WORK_INTELLIGENCE, AgentId.ASSIGNMENT, AgentId.PLANNING)
    )
    orch, value, scope, lease, _, log, resources, _ = make_harness(
        responses=(),
        use_packaged_manifest=True,
        model_ref=model,
        locale=locale,
        message=case.prompt,
        specialists=tuple(
            (manifest.agent.id, manifest.delegation_summary) for manifest in specialist_manifests
        ),
        backends={
            "delegate_agent@2": FixtureToolBackend(trial.delegate),
            "explore_projects@1": FixtureToolBackend(
                _project_overview if probe == "skill-overview" else trial.explore
            ),
        },
    )
    orch.gateway = gateway
    started = perf_counter()
    async with asyncio.timeout(180):
        result = await orch.run_segment(value, scope=scope, checkpoint=None, lease=lease)
    called = tuple(
        event.invocation.tool_reference
        for event in log.records
        if event.kind == "tool.requested" and event.invocation is not None
    )
    skill = SkillRegistry.from_resources(
        (("work_management_ai.skills.explore_work_context", "skill.yaml"),)
    ).resolve("explore_work_context@1")
    assert skill.bundle is not None
    registered = orch.catalog.agent
    record = {
        "probe": probe,
        "locale": locale,
        "model": model,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "input_tokens": [entry.input_tokens if entry else None for entry in gateway.responses],
        "output_tokens": [entry.output_tokens if entry else None for entry in gateway.responses],
        "model_requests": len(gateway.requests),
        "tool_refs": called,
        "specialists": trial.specialists,
        "skill_body_reads": resources.bodies_read,
        "wait_kind": (result.wait_ref.kind if result.wait_ref else None),
        "status": result.output.status.value
        if isinstance(result, PublishedResponse)
        else "SUSPENDED",
        "agent_manifest": registered.fingerprint,
        "specialist_manifests": {
            manifest.agent.id.value: canonical_manifest_fingerprint(manifest)
            for manifest in specialist_manifests
        },
        "prompt_version": PROMPT_VERSION,
        "tools": [pin.model_dump(mode="json") for pin in registered.tool_pins],
        "tool_declarations": sorted(
            {
                (tool.reference, canonical_manifest_fingerprint(tool))
                for request in gateway.requests
                for tool in request.tools
            }
        ),
        "skill_descriptor": canonical_manifest_fingerprint(skill.bundle),
        "skill_body": skill.bundle.instructions.sha256,
        "skill_resources": {resource.path: resource.sha256 for resource in skill.bundle.resources},
    }
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    assert isinstance(result, PublishedResponse)
    if probe == "greeting":
        assert called == () and resources.bodies_read == 0
    if probe == "ambiguous":
        assert result.wait_ref is not None and result.wait_ref.kind == "USER_INPUT"
        assert "explore_projects@1" in called and not trial.specialists
    if probe in {"32h", "48h"}:
        assert trial.specialists == (
            ["assignment", "planning"] if probe == "48h" else ["assignment"]
        )
        assert any(
            item.kind == "tool_result" and f"{probe}/40h" in str(item.content)
            for request in gateway.requests[1:]
            for item in request.items
        )
    if probe == "48h":
        assert result.wait_ref is not None and result.wait_ref.kind == "HUMAN_DECISION"
        assert result.wait_ref.resource_id == trial.expected_resource_id
    if probe == "skill-overview":
        assert resources.bodies_read == 1
        assert called.count("explore_projects@1") == 1 and "read_project_context@1" not in called
    if probe in {"greeting", "32h", "skill-overview"}:
        assert result.output.status is OrchestratorStatus.COMPLETED
