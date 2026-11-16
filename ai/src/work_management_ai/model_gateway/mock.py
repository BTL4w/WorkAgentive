"""Deterministic Model Gateway used by local development and automated tests."""

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import cast

from pydantic import BaseModel, ValidationError

from work_management_ai.model_gateway.contracts import (
    ModelUsage,
    StructuredModelRequest,
    StructuredModelResponse,
    validate_request,
)
from work_management_ai.model_gateway.errors import (
    ModelInvalidOutputError,
    ModelUnavailableError,
    normalize_model_error,
)
from work_management_ai.model_gateway.tool_contracts import (
    ToolModelRequest,
    ToolModelResponse,
)


class MockModelGateway:
    """Select immutable fixtures deterministically by invocation key."""

    def __init__(
        self,
        *,
        fixtures: Mapping[str, object],
        model_ref: str = "mock:planning-v1",
        usage_fixtures: Mapping[str, ModelUsage] | None = None,
        sequence_fixtures: Mapping[str, tuple[object, ...]] | None = None,
        tool_sequence_fixtures: Mapping[str, tuple[object, ...]] | None = None,
    ) -> None:
        self._fixtures = MappingProxyType(deepcopy(dict(fixtures)))
        self._sequences = {
            key: list(deepcopy(values)) for key, values in (sequence_fixtures or {}).items()
        }
        self._model_ref = model_ref
        self._usage_fixtures = MappingProxyType(dict(usage_fixtures or {}))
        self._tool_sequences = {
            key: list(deepcopy(values)) for key, values in (tool_sequence_fixtures or {}).items()
        }
        self._step_requests: list[ToolModelRequest[BaseModel]] = []

    @property
    def step_requests(self) -> tuple[ToolModelRequest[BaseModel], ...]:
        """Isolated captures of actual requests, not scripted expected history."""

        return tuple(request.model_copy(deep=True) for request in self._step_requests)

    def assert_complete(self) -> None:
        """Fail when the caller stopped before consuming all scripted steps."""

        remaining = sum(len(values) for values in self._tool_sequences.values())
        remaining += sum(len(values) for values in self._sequences.values())
        if remaining:
            raise AssertionError(f"{remaining} scripted model responses remain")

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        """Consume one native response; this mock never dispatches a tool."""

        # Revalidate even if a test has used model_copy(update=...) to change it.
        request = type(request).model_validate(request)
        self._step_requests.append(
            ToolModelRequest[BaseModel](
                invocation_key=request.invocation_key,
                items=request.items,
                tools=request.tools,
                response_schema=request.response_schema,
                timeout_seconds=request.timeout_seconds,
                max_output_tokens=request.max_output_tokens,
            ).model_copy(deep=True)
        )
        if request.invocation_key in self._tool_sequences:
            sequence = self._tool_sequences[request.invocation_key]
            if not sequence:
                raise ModelUnavailableError("model fixture sequence exhausted")
            fixture = sequence.pop(0)
        elif request.invocation_key in self._fixtures:
            fixture = deepcopy(self._fixtures[request.invocation_key])
        else:
            raise ModelUnavailableError("model fixture unavailable")
        if isinstance(fixture, Exception):
            raise normalize_model_error(fixture) from fixture
        try:
            if not isinstance(fixture, dict):
                raise ValueError("native fixture must be a response object")
            payload = dict(cast(dict[str, object], fixture))
            if payload.get("final") is not None:
                payload["final"] = request.response_schema.model_validate(payload["final"])
            response = ToolModelResponse[OutputT].model_validate(
                {
                    **payload,
                    "model_ref": self._model_ref,
                    "usage": self._usage_fixtures.get(request.invocation_key),
                }
            )
            if response.tool_call is not None:
                call = response.tool_call
                schema = next((s for s in request.tools if s.wire_name == call.name), None)
                if schema is None or any(
                    item.call is not None and item.call.call_id == call.call_id
                    for item in request.items
                ):
                    raise ValueError("tool is unadmitted or call ID is duplicated")
                if schema.input_contract is not None:
                    schema.input_contract.model_validate(call.arguments)
        except (TypeError, ValidationError, ValueError) as error:
            raise ModelInvalidOutputError(
                "model output failed schema validation",
                usage=self._usage_fixtures.get(request.invocation_key),
            ) from error
        return response

    async def generate_structured[StructuredOutputT: BaseModel](
        self,
        request: StructuredModelRequest[StructuredOutputT],
    ) -> StructuredModelResponse[StructuredOutputT]:
        """Return a typed fixture or a normalized deterministic failure."""

        validate_request(request)
        if request.invocation_key in self._sequences:
            sequence = self._sequences[request.invocation_key]
            if not sequence:
                raise ModelUnavailableError("model fixture sequence exhausted")
            fixture = sequence.pop(0)
        elif request.invocation_key in self._fixtures:
            fixture = self._fixtures[request.invocation_key]
        else:
            raise ModelUnavailableError("model fixture unavailable")
        if isinstance(fixture, Exception):
            raise normalize_model_error(fixture) from fixture

        try:
            parsed = request.output_schema.model_validate(fixture)
        except (TypeError, ValidationError, ValueError) as error:
            raise ModelInvalidOutputError(
                "model output failed schema validation",
                usage=self._usage_fixtures.get(request.invocation_key),
            ) from error

        return StructuredModelResponse(
            parsed=parsed,
            model_ref=self._model_ref,
            usage=self._usage_fixtures.get(request.invocation_key),
        )
