"""Stateless hosted native tool steps, isolated from the structured v1 gateway."""

import asyncio
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.errors import (
    ModelInvalidInputError,
    ModelInvalidOutputError,
    ModelUnavailableError,
    normalize_model_error,
)
from work_management_ai.model_gateway.openai import private_provider_request
from work_management_ai.model_gateway.tool_contracts import (
    ToolCall,
    ToolItem,
    ToolModelRequest,
    ToolModelResponse,
)


class _Runnable(Protocol):
    async def ainvoke(self, messages: list[BaseMessage]) -> object: ...


class _ChatModel(Protocol):
    def bind_tools(
        self,
        tools: list[dict[str, object]],
        *,
        strict: bool,
        parallel_tool_calls: bool,
        response_format: dict[str, object],
    ) -> _Runnable: ...


def _create_chat_model(
    *,
    model_name: str,
    api_key: SecretStr,
    timeout_seconds: float,
    max_output_tokens: int,
    use_responses_api: bool,
) -> object:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
        max_completion_tokens=max_output_tokens,
        use_responses_api=use_responses_api,
        store=False,
    )


_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "$defs",
        "$ref",
        "title",
        "description",
        "default",
        "enum",
        "const",
        "format",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
    }
)


def _strict_schema(schema: Mapping[str, object], *, root: bool = False) -> dict[str, object]:
    """Compile the supported subset without silently removing validation rules.

    Missing optional nullable fields become required nullable fields on the wire.
    Open maps/unknown keywords cannot be faithfully represented and fail closed.
    Pydantic validation still applies after the provider responds.
    """

    if set(schema) - _SCHEMA_KEYS or (root and schema.get("type") != "object"):
        raise ValueError("unsupported native schema")
    if not any(key in schema for key in ("type", "$ref", "anyOf")):
        raise ValueError("unbounded native schema")
    output = {k: deepcopy(v) for k, v in schema.items() if k != "default"}
    for key in ("$defs", "properties"):
        if key not in schema:
            continue
        value = schema[key]
        if not isinstance(value, dict):
            raise ValueError("invalid schema map")
        compiled: dict[str, object] = {}
        for name, child in cast(dict[str, object], value).items():
            if not isinstance(child, dict):
                raise ValueError("invalid nested schema")
            compiled[name] = _strict_schema(cast(dict[str, object], child))
        output[key] = compiled
    if schema.get("type") == "object":
        if schema.get("additionalProperties", False) is not False:
            raise ValueError("open dictionaries are not supported")
        properties = cast(dict[str, object], output.get("properties", {}))
        output["properties"] = properties
        output["required"] = list(properties)
        output["additionalProperties"] = False
    if "items" in schema:
        items = schema["items"]
        if not isinstance(items, dict):
            raise ValueError("invalid array schema")
        output["items"] = _strict_schema(cast(dict[str, object], items))
    if "anyOf" in schema:
        alternatives = schema["anyOf"]
        if not isinstance(alternatives, list):
            raise ValueError("invalid union schema")
        branches: list[dict[str, object]] = []
        for branch in cast(list[object], alternatives):
            if not isinstance(branch, dict):
                raise ValueError("invalid union branch")
            branches.append(_strict_schema(cast(dict[str, object], branch)))
        output["anyOf"] = branches
    if "$ref" in schema and not str(schema["$ref"]).startswith("#/$defs/"):
        raise ValueError("external schema references are not supported")
    return output


def _messages(items: tuple[ToolItem, ...]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for item in items:
        if item.kind == "message":
            assert isinstance(item.content, str)
            constructors = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
            assert item.role is not None
            result.append(constructors[item.role](content=item.content))
        elif item.kind == "tool_call":
            assert item.call is not None
            result.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": item.call.call_id,
                            "name": item.call.name,
                            "args": item.call.arguments,
                        }
                    ],
                )
            )
        else:
            assert item.call_id is not None
            result.append(
                ToolMessage(
                    content=item.content
                    if isinstance(item.content, str)
                    else json.dumps(item.content, ensure_ascii=False, allow_nan=False),
                    tool_call_id=item.call_id,
                )
            )
    return result


def _usage(raw: AIMessage) -> ModelUsage | None:
    metadata = raw.usage_metadata
    if metadata is None:
        return None
    return ModelUsage(metadata["input_tokens"], metadata["output_tokens"])


def _visible_text(raw: AIMessage) -> str:
    if isinstance(raw.content, str):
        return raw.content
    text: list[str] = []
    for block in raw.content:
        if isinstance(block, str):
            text.append(block)
        elif block.get("type") in {"text", "output_text"}:
            value = block.get("text")
            if not isinstance(value, str):
                raise ValueError("invalid visible text")
            text.append(value)
        elif block.get("type") != "reasoning":
            raise ValueError("unsupported response content")
    return "\n".join(text)


class OpenAINativeToolGateway:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: SecretStr,
        chat_model_factory: Callable[..., object] = _create_chat_model,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._factory = chat_model_factory

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_step[OutputT: BaseModel](
        self,
        request: ToolModelRequest[OutputT],
    ) -> ToolModelResponse[OutputT]:
        try:
            request = type(request).model_validate(request)
        except (ValidationError, ValueError):
            raise ModelInvalidInputError("native model request is invalid") from None
        try:
            if not self._model_name.strip():
                raise ValueError("missing configured model")
            final_schema = _strict_schema(request.response_schema.model_json_schema(), root=True)
            declarations: list[dict[str, object]] = []
            for schema in request.tools:
                if schema.input_contract is None:
                    raise ValueError("registered input contract required")
                declarations.append(
                    {
                        "type": "function",
                        "function": {
                            "name": schema.wire_name,
                            "description": schema.description,
                            "strict": True,
                            "parameters": _strict_schema(schema.parameters, root=True),
                        },
                    }
                )
            messages = _messages(request.items)
        except (TypeError, ValueError, RecursionError):
            raise ModelUnavailableError("native tool protocol is unsupported") from None
        try:
            with private_provider_request():
                chat = cast(
                    _ChatModel,
                    self._factory(
                        model_name=self._model_name,
                        api_key=self._api_key,
                        timeout_seconds=request.timeout_seconds,
                        max_output_tokens=request.max_output_tokens,
                        use_responses_api=False,
                    ),
                )
                runnable = chat.bind_tools(
                    declarations,
                    strict=True,
                    parallel_tool_calls=False,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "orchestrator_final",
                            "strict": True,
                            "schema": final_schema,
                        },
                    },
                )
                async with asyncio.timeout(request.timeout_seconds):
                    raw = await runnable.ainvoke(messages)
        except ValidationError:
            raise ModelInvalidOutputError("model output failed schema validation") from None
        except Exception as error:
            normalized = normalize_model_error(error)
            raise type(normalized)("model provider request failed") from None

        usage: ModelUsage | None = None
        try:
            if not isinstance(raw, AIMessage):
                raise ValueError("invalid native response")
            usage = _usage(raw)
            finish = raw.response_metadata.get("finish_reason")
            if (
                finish not in (None, "stop", "tool_calls")
                or raw.invalid_tool_calls
                or raw.additional_kwargs.get("refusal")
            ):
                raise ValueError("incomplete or unusable native response")
            if finish is not None and (finish == "tool_calls") != bool(raw.tool_calls):
                raise ValueError("native finish state contradicts response")
            text = _visible_text(raw)
            if raw.tool_calls:
                if len(raw.tool_calls) != 1:
                    raise ValueError("exactly one call is supported")
                native_call = raw.tool_calls[0]
                call = ToolCall.model_validate(
                    {
                        "call_id": native_call["id"],
                        "name": native_call["name"],
                        "arguments": native_call["args"],
                    }
                )
                schema = next((s for s in request.tools if s.wire_name == call.name), None)
                if (
                    schema is None
                    or schema.input_contract is None
                    or any(
                        item.call is not None and item.call.call_id == call.call_id
                        for item in request.items
                    )
                ):
                    raise ValueError("unadmitted tool or reused call ID")
                schema.input_contract.model_validate(call.arguments)
                if text.strip():
                    try:
                        request.response_schema.model_validate_json(text)
                    except (ValidationError, ValueError):
                        pass
                    else:
                        raise ValueError("call and final are mutually exclusive")
                return ToolModelResponse[OutputT](
                    tool_call=call,
                    commentary=text or None,
                    model_ref=f"openai:{self._model_name}",
                    usage=usage,
                )
            final = request.response_schema.model_validate_json(text)
            return ToolModelResponse[OutputT](
                final=final, model_ref=f"openai:{self._model_name}", usage=usage
            )
        except (TypeError, ValueError, KeyError):
            raise ModelInvalidOutputError(
                "model output failed schema validation", usage=usage
            ) from None
