"""Opt-in hosted decisions over synthetic facts and packaged tool declarations."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Literal, cast
from uuid import uuid4

import pytest
from dotenv import dotenv_values
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI
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


@pytest.mark.parametrize(
    ("configured_model", "expected"),
    [("gpt-5-mini", "openai/gpt-5-mini"), ("openai/gpt-5-mini", "openai/gpt-5-mini")],
)
def test_openrouter_settings_use_configured_model_and_routing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured_model: str, expected: str
) -> None:
    for name in (*_OPENAI_OVERRIDE_VARS, *_TRACING_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_AI_MODEL", configured_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-key")
    monkeypatch.delenv("APP_OPENAI_API_KEY", raising=False)
    model, key = _openrouter_settings(tmp_path / ".env")
    assert model == expected
    assert key == SecretStr("fixture-openrouter-key")
    chat = _create_openrouter_chat_model(
        model_name=model,
        api_key=key,
        timeout_seconds=5,
        max_output_tokens=500,
        use_responses_api=False,
    )
    assert isinstance(chat, ChatOpenAI)
    assert chat.openai_api_base == "https://openrouter.ai/api/v1"
    assert chat.model_name == expected
    assert chat.extra_body == {"provider": {"require_parameters": True}}
    payload = chat._get_request_payload(  # pyright: ignore[reportPrivateUsage]
        "fixture", parallel_tool_calls=False
    )
    assert payload["max_tokens"] == 500
    assert "max_completion_tokens" not in payload
    assert "parallel_tool_calls" not in payload


def test_openrouter_settings_loads_local_alias_without_exporting_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "APP_AI_MODEL",
        "OPENROUTER_API_KEY",
        "OPEN_ROUTER_API",
        *_OPENAI_OVERRIDE_VARS,
        *_TRACING_VARS,
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('APP_AI_MODEL="gpt-5-mini"\nOPEN_ROUTER_API="fixture-local-key"\n')
    assert _openrouter_settings(env_file) == ("openai/gpt-5-mini", SecretStr("fixture-local-key"))
    assert "OPEN_ROUTER_API" not in os.environ
    assert "OPENROUTER_API_KEY" not in os.environ


_OPENAI_OVERRIDE_VARS = ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_PROXY")
_TRACING_VARS = (
    "APP_LANGSMITH_TRACING",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)


def _openrouter_settings(env_file: Path) -> tuple[str, SecretStr]:
    local = dotenv_values(env_file, interpolate=False) if env_file.is_file() else {}
    model = os.getenv("APP_AI_MODEL", "").strip() or (local.get("APP_AI_MODEL") or "").strip()
    credential = (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPEN_ROUTER_API", "").strip()
        or (local.get("OPENROUTER_API_KEY") or "").strip()
        or (local.get("OPEN_ROUTER_API") or "").strip()
    )
    if not model or not credential:
        pytest.fail("Set APP_AI_MODEL and either OPENROUTER_API_KEY or OPEN_ROUTER_API.")
    if any(os.getenv(name) for name in _OPENAI_OVERRIDE_VARS):
        pytest.fail("Pilot uses the official OpenRouter endpoint without OpenAI overrides.")
    if any(os.getenv(name, "").casefold() in ("1", "true", "yes", "on") for name in _TRACING_VARS):
        pytest.fail("Pilot requires automatic tracing to be disabled.")
    return model if "/" in model else f"openai/{model}", SecretStr(credential)


class _OpenRouterChatOpenAI(ChatOpenAI):
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        payload = cast(
            dict[str, object],
            super()._get_request_payload(input_, stop=stop, **kwargs),  # pyright: ignore[reportUnknownMemberType]
        )
        # OpenRouter's request filter expects max_tokens. The native gateway still
        # validates that the model returns at most one tool call per step.
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload.pop("parallel_tool_calls", None)
        return payload


def _create_openrouter_chat_model(
    *,
    model_name: str,
    api_key: SecretStr,
    timeout_seconds: float,
    max_output_tokens: int,
    use_responses_api: bool,
) -> _OpenRouterChatOpenAI:
    return _OpenRouterChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_seconds,
        max_retries=0,
        max_completion_tokens=max_output_tokens,
        use_responses_api=use_responses_api,
        extra_body={"provider": {"require_parameters": True}},
    )


@pytest.fixture(scope="module")
def hosted_settings() -> tuple[str, SecretStr]:
    return _openrouter_settings(Path(__file__).resolve().parents[2] / ".env")


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
    gateway = BoundedNativeGateway(
        OpenAINativeToolGateway(
            model_name=model,
            api_key=credential,
            chat_model_factory=_create_openrouter_chat_model,
        )
    )
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
        result = await orch.run_segment(
            value,
            scope=scope,
            checkpoint=None,
            lease=lease,
            verified_context_refs=(trial.ref(scope),) if probe in {"32h", "48h"} else (),
        )
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
        "provider": "openrouter",
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
        # Loading and verifying the same pinned artifact can both read the body.
        # The pilot measures the model's tool calls, not package I/O.
        assert called.count("load_skill@1") == 1
        assert called.count("explore_projects@1") == 1 and "read_project_context@1" not in called
    if probe in {"greeting", "32h", "skill-overview"}:
        assert result.output.status is OrchestratorStatus.COMPLETED
