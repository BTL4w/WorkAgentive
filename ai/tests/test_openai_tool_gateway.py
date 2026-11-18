"""Hosted native adapter contracts using stubs and the installed SDK offline."""

import json
import logging
import traceback
from typing import cast

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.errors import (
    ModelInvalidOutputError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from work_management_ai.model_gateway.openai_tools import OpenAINativeToolGateway
from work_management_ai.model_gateway.tool_contracts import (
    ToolCall,
    ToolItem,
    ToolModelRequest,
    ToolSchema,
)
from work_management_ai.runtime.manifests import ToolManifestV2, load_tool_manifest_resource
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context import contracts as context_contracts


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


class ProjectRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_reference: str
    section: str | None = None


def request() -> ToolModelRequest[Answer]:
    return ToolModelRequest[Answer](
        invocation_key="turn-1.step-2",
        items=(ToolItem(kind="message", role="user", content="Đọc mục tiêu dự án đào tạo."),),
        tools=(
            ToolSchema(
                reference="automation.preview@1",
                description="Read preview",
                input_contract=ProjectRead,
            ),
        ),
        response_schema=Answer,
        timeout_seconds=5,
        max_output_tokens=120,
    )


class StubChat:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.binding: dict[str, object] = {}
        self.messages: list[BaseMessage] | None = None
        self.calls = 0

    def bind_tools(
        self,
        tools: list[dict[str, object]],
        *,
        strict: bool,
        parallel_tool_calls: bool,
        response_format: dict[str, object],
    ) -> "StubChat":
        self.binding = dict(
            tools=tools,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
        )
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> object:
        self.calls += 1
        self.messages = messages
        logging.getLogger("openai._base_client").debug("private-sdk-body")
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def gateway(outcome: object) -> tuple[OpenAINativeToolGateway, StubChat, dict[str, object]]:
    stub = StubChat(outcome)
    config: dict[str, object] = {}

    def factory(**kwargs: object) -> StubChat:
        config.update(kwargs)
        return stub

    return (
        OpenAINativeToolGateway(
            model_name="configured-model", api_key=SecretStr("fake-key"), chat_model_factory=factory
        ),
        stub,
        config,
    )


def final_message() -> AIMessage:
    return AIMessage(
        content='{"text":"20 nhân viên"}',
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    )


@pytest.mark.asyncio
async def test_tool_result_roundtrip_uses_call_id() -> None:
    adapter, stub, config = gateway(final_message())
    call = ToolCall(
        call_id="call-7",
        name="automation__preview",
        arguments={"project_reference": "project-a", "section": None},
    )
    req = request().model_copy(
        update={
            "items": (
                *request().items,
                ToolItem(kind="tool_call", call=call),
                ToolItem(kind="tool_result", call_id="call-7", content={"goal": "20 employees"}),
            )
        }
    )
    result = await adapter.generate_step(req)
    assert result.final == Answer(text="20 nhân viên")
    assert result.usage == ModelUsage(20, 5)
    assert result.model_ref == "openai:configured-model"
    assert adapter.model_name == config["model_name"] == "configured-model"
    assert config["max_output_tokens"] == 120
    messages = stub.messages
    assert messages is not None
    assert isinstance(messages[-2], AIMessage)
    assert messages[-2].tool_calls[0]["id"] == "call-7"
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == "call-7"
    assert json.loads(str(messages[-1].content)) == {"goal": "20 employees"}


@pytest.mark.asyncio
async def test_parallel_disabled_and_strict_schema() -> None:
    adapter, stub, _ = gateway(final_message())
    req = request()
    before = req.tools[0].model_copy(deep=True)
    await adapter.generate_step(req)
    assert stub.binding["parallel_tool_calls"] is False
    assert stub.binding["strict"] is True
    tools = cast(list[dict[str, object]], stub.binding["tools"])
    function = cast(dict[str, object], tools[0]["function"])
    assert function["name"] == "automation__preview"
    assert function["strict"] is True
    schema = cast(dict[str, object], function["parameters"])
    assert schema["required"] == ["project_reference", "section"]
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    alternatives = cast(list[object], properties["section"]["anyOf"])
    assert {"type": "null"} in alternatives
    assert "default" not in properties["section"]
    assert req.tools[0] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content", ["", "Here is my answer", '{"wrong":1}', '{"text":""}', '{"text":"ok","extra":true}']
)
async def test_final_schema_validated(content: str) -> None:
    adapter, _, _ = gateway(AIMessage(content=content))
    with pytest.raises(ModelInvalidOutputError):
        await adapter.generate_step(request())


@pytest.mark.asyncio
async def test_one_native_call_and_commentary_excludes_provider_reasoning() -> None:
    response = AIMessage(
        content=[
            {"type": "reasoning", "text": "private-reasoning"},
            {"type": "text", "text": "Tôi sẽ đọc dữ liệu."},
        ],
        tool_calls=[
            {
                "id": "call-1",
                "name": "automation__preview",
                "args": {"project_reference": "project-a", "section": None},
            }
        ],
        additional_kwargs={
            "reasoning_content": "private-reasoning",
            "system_prompt": "private-system",
        },
        response_metadata={"finish_reason": "tool_calls"},
    )
    adapter, _, _ = gateway(response)
    result = await adapter.generate_step(request())
    assert result.tool_call is not None
    assert result.tool_call.call_id == "call-1"
    assert result.commentary == "Tôi sẽ đọc dữ liệu."
    assert "private-" not in result.model_dump_json()
    assert result.final is None


INVALID_CALLS: list[object] = [
    AIMessage(content="", tool_calls=[{"id": "c1", "name": "unknown", "args": {}}]),
    AIMessage(
        content="", tool_calls=[{"id": "c1", "name": "automation__preview", "args": {"wrong": 1}}]
    ),
    AIMessage(
        content="",
        tool_calls=[
            {"id": "c1", "name": "automation__preview", "args": {}},
            {"id": "c2", "name": "automation__preview", "args": {}},
        ],
    ),
    AIMessage(
        content='{"text":"final"}',
        tool_calls=[
            {
                "id": "c1",
                "name": "automation__preview",
                "args": {"project_reference": "p", "section": None},
            }
        ],
    ),
    AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "c1", "name": "automation__preview", "args": "{", "error": "private error"}
        ],
    ),
    AIMessage(content='{"text":"partial"}', response_metadata={"finish_reason": "length"}),
    AIMessage(content=[{"type": "refusal", "refusal": "private refusal"}]),
    AIMessage(content='{"text":"ok"}', response_metadata={"finish_reason": "tool_calls"}),
    AIMessage(
        content="",
        response_metadata={"finish_reason": "stop"},
        tool_calls=[
            {
                "id": "c1",
                "name": "automation__preview",
                "args": {"project_reference": "p", "section": None},
            }
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", INVALID_CALLS)
async def test_invalid_native_response_never_returns_dispatchable_call(outcome: object) -> None:
    adapter, _, _ = gateway(outcome)
    with pytest.raises(ModelInvalidOutputError):
        await adapter.generate_step(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("private-provider-key-and-context"),
        ModelUnavailableError("private-provider-key-and-context"),
    ],
)
async def test_provider_failure_is_redacted(
    caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    adapter, stub, _ = gateway(error)
    with caplog.at_level(logging.DEBUG), pytest.raises(ModelUnavailableError) as caught:
        await adapter.generate_step(request())
    formatted = "".join(traceback.format_exception(caught.value))
    assert "private-provider-key-and-context" not in formatted
    assert "private-sdk-body" not in caplog.text
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_provider_timeout_is_normalized() -> None:
    adapter, _, _ = gateway(TimeoutError("private timeout"))
    with pytest.raises(ModelTimeoutError):
        await adapter.generate_step(request())


@pytest.mark.asyncio
async def test_unusable_model_fails_closed() -> None:
    attempts: list[str] = []

    def factory(**kwargs: object) -> StubChat:
        attempts.append(str(kwargs["model_name"]))
        raise NotImplementedError("private model not supported")

    adapter = OpenAINativeToolGateway(
        model_name="configured-model", api_key=SecretStr("fake"), chat_model_factory=factory
    )
    with pytest.raises(ModelUnavailableError):
        await adapter.generate_step(request())
    assert attempts == ["configured-model"]


@pytest.mark.asyncio
async def test_unbounded_schema_fails_before_provider_creation() -> None:
    class Unbounded(BaseModel):
        values: dict[str, str]

    adapter, stub, config = gateway(final_message())
    with pytest.raises(ModelUnavailableError):
        await adapter.generate_step(
            ToolModelRequest[Unbounded](
                invocation_key="unsupported",
                items=request().items,
                tools=request().tools,
                response_schema=Unbounded,
            )
        )
    assert not config and stub.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("with_call", [False, True])
async def test_installed_sdk_sends_strict_stateless_payload_offline(
    with_call: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    class DocumentedSection(BaseModel):
        key: str = Field(description="Section key from the available catalog.")

    class DocumentedInput(BaseModel):
        sections: list[DocumentedSection] = Field(description="Sections to inspect.")

    monkeypatch.setattr(context_contracts, "DocumentedInput", DocumentedInput, raising=False)
    actor = load_tool_manifest_resource("work_management_ai.tools.context", "actor.yaml")
    assert isinstance(actor, ToolManifestV2)
    nested = actor.model_copy(
        update={
            "name": "test_documented_nested",
            "input_contract": "work_management_ai.tools.context.contracts.DocumentedInput",
        }
    )
    registry = ToolRegistry([actor, nested])
    schemas = registry.candidate_schemas(("read_actor_context@1", "test_documented_nested@1"))

    def respond(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/chat/completions")
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "configured-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if with_call else "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-next",
                                    "type": "function",
                                    "function": {
                                        "name": "automation__preview",
                                        "arguments": json.dumps(
                                            {"project_reference": "project-a", "section": None}
                                        ),
                                    },
                                }
                            ],
                        }
                        if with_call
                        else {"role": "assistant", "content": '{"text":"ok"}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        def factory(**kwargs: object) -> object:
            return ChatOpenAI(
                model=str(kwargs["model_name"]),
                api_key=SecretStr("fake"),
                http_async_client=client,
                use_responses_api=False,
                store=False,
                max_retries=0,
                max_completion_tokens=120,
            )

        adapter = OpenAINativeToolGateway(
            model_name="configured-model", api_key=SecretStr("fake"), chat_model_factory=factory
        )
        previous = ToolCall(
            call_id="call-previous",
            name="automation__preview",
            arguments={"project_reference": "project-a", "section": None},
        )
        req = request().model_copy(
            update={
                "tools": (*request().tools, *schemas),
                "items": (
                    *request().items,
                    ToolItem(kind="tool_call", call=previous),
                    ToolItem(
                        kind="tool_result",
                        call_id="call-previous",
                        content={"goal": "20 employees"},
                    ),
                ),
            }
        )
        result = await adapter.generate_step(req)
        if with_call:
            assert result.tool_call is not None and result.tool_call.call_id == "call-next"
        else:
            assert result.final == Answer(text="ok")
    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == "configured-model"
    assert payload["parallel_tool_calls"] is False
    assert payload["store"] is False
    assert "previous_response_id" not in payload
    assert "reasoning" not in payload
    assert payload["max_completion_tokens"] == 120
    functions = {
        function["name"]: function
        for row in cast(list[dict[str, object]], payload["tools"])
        for function in (cast(dict[str, object], row["function"]),)
    }
    actor_wire = functions["read_actor_context"]
    assert actor_wire["description"] == actor.description
    assert actor_wire["strict"] is True
    actor_parameters = cast(dict[str, object], actor_wire["parameters"])
    assert actor_parameters["required"] == ["sections"]
    assert actor_parameters["additionalProperties"] is False
    actor_fields = cast(dict[str, dict[str, object]], actor_parameters["properties"])
    assert actor_fields["sections"]["description"] == (
        context_contracts.ReadActorInput.model_fields["sections"].description
    )
    nested_wire = cast(dict[str, object], functions["test_documented_nested"]["parameters"])
    nested_defs = cast(dict[str, dict[str, object]], nested_wire["$defs"])
    nested_object = nested_defs["DocumentedSection"]
    assert nested_object["required"] == ["key"]
    assert nested_object["additionalProperties"] is False
    nested_fields = cast(dict[str, dict[str, object]], nested_object["properties"])
    assert nested_fields["key"]["description"] == "Section key from the available catalog."
    messages = cast(list[dict[str, object]], payload["messages"])
    calls = cast(list[dict[str, object]], messages[-2]["tool_calls"])
    assert calls[0]["id"] == messages[-1]["tool_call_id"] == "call-previous"
    assert json.loads(str(messages[-1]["content"])) == {"goal": "20 employees"}
    fmt = payload["response_format"]
    assert isinstance(fmt, dict)
    assert fmt["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_default_factory_preserves_model_and_disables_provider_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import work_management_ai.model_gateway.openai_tools as module

    config: dict[str, object] = {}

    def create(**kwargs: object) -> StubChat:
        config.update(kwargs)
        return StubChat(final_message())

    monkeypatch.setattr(module, "ChatOpenAI", create)
    adapter = OpenAINativeToolGateway(model_name="configured-model", api_key=SecretStr("fake"))
    await adapter.generate_step(request())
    assert config["model"] == "configured-model"
    assert config["max_retries"] == 0
    assert config["store"] is False
    assert config["use_responses_api"] is False
    assert config["max_completion_tokens"] == 120
    assert config["timeout"] == 5


@pytest.mark.asyncio
async def test_deadline_cancels_provider_await() -> None:
    import asyncio

    cancelled: list[bool] = []

    class Slow(StubChat):
        async def ainvoke(self, messages: list[BaseMessage]) -> object:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    def factory(**kwargs: object) -> Slow:
        return Slow(None)

    adapter = OpenAINativeToolGateway(
        model_name="configured-model",
        api_key=SecretStr("fake"),
        chat_model_factory=factory,
    )
    with pytest.raises(ModelTimeoutError):
        await adapter.generate_step(request().model_copy(update={"timeout_seconds": 0.01}))
    assert cancelled == [True]


@pytest.mark.asyncio
async def test_current_public_block_union_is_not_silently_weakened_for_provider() -> None:
    from work_management_ai.agents.orchestrator.tool_loop_contracts import OrchestratorFinalResponse

    adapter, stub, config = gateway(final_message())
    req = ToolModelRequest[OrchestratorFinalResponse](
        invocation_key="public-schema",
        items=(),
        tools=(),
        response_schema=OrchestratorFinalResponse,
    )
    with pytest.raises(ModelUnavailableError):
        await adapter.generate_step(req)
    assert not config and stub.calls == 0


@pytest.mark.asyncio
async def test_text_observation_is_not_double_json_encoded() -> None:
    adapter, stub, _ = gateway(final_message())
    call = ToolCall(
        call_id="text-read",
        name="automation__preview",
        arguments={"project_reference": "p", "section": None},
    )
    req = request().model_copy(
        update={
            "items": (
                ToolItem(kind="tool_call", call=call),
                ToolItem(
                    kind="tool_result", call_id="text-read", content="Không tìm thấy dữ liệu."
                ),
            )
        }
    )
    await adapter.generate_step(req)
    assert stub.messages is not None
    assert stub.messages[-1].content == "Không tìm thấy dữ liệu."
