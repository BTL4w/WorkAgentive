"""Provider-neutral native tool steps; separate from the structured v1 API."""

from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.runtime.contracts import JsonValue


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class ToolSchema(_Contract):
    """Registry-owned identity and declaration; loading does not confer access.

    Wire aliases are transport-only. Audit and dispatch use ``reference``.
    Contract types never cross the provider boundary.
    """

    reference: str = Field(pattern=r"^[a-z][a-z0-9_.]*@[1-9][0-9]*$", max_length=100)
    description: str = Field(min_length=1, max_length=4000)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    input_contract: type[BaseModel] | None = Field(default=None, exclude=True)

    @property
    def wire_name(self) -> str:
        return self.reference.rsplit("@", 1)[0].replace(".", "__")

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        if len(self.wire_name) > 64:
            raise ValueError("TOOL_WIRE_NAME_TOO_LONG")
        if self.input_contract is not None:
            schema = self.input_contract.model_json_schema()
            if self.parameters and self.parameters != schema:
                raise ValueError("TOOL_SCHEMA_CONTRACT_MISMATCH")
            object.__setattr__(self, "parameters", schema)
        return self


class ToolCall(_Contract):
    call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    arguments: dict[str, JsonValue]


class ToolItem(_Contract):
    """Typed instruction, conversation message, assistant call or paired tool observation."""

    kind: Literal["message", "tool_call", "tool_result"]
    role: Literal["system", "developer", "user", "assistant"] | None = None
    content: JsonValue = None
    call: ToolCall | None = None
    call_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.kind == "message":
            valid = (
                self.role is not None
                and isinstance(self.content, str)
                and self.call is None
                and self.call_id is None
            )
        elif self.kind == "tool_call":
            valid = (
                self.call is not None
                and self.call_id is None
                and self.role is None
                and self.content is None
            )
        else:
            valid = self.call_id is not None and self.call is None and self.role is None
        if not valid:
            raise ValueError("TOOL_HISTORY_ITEM_INVALID")
        return self


def validate_history(items: tuple[ToolItem, ...]) -> None:
    """Every historical call settles once, in order, before a new model step."""

    seen: set[str] = set()
    pending: str | None = None
    for item in items:
        if item.kind == "tool_call":
            assert item.call is not None
            if pending is not None or item.call.call_id in seen:
                raise ValueError("TOOL_CALL_DUPLICATE_OR_UNSETTLED")
            pending = item.call.call_id
            seen.add(pending)
        elif item.kind == "tool_result":
            if pending is None or item.call_id != pending:
                raise ValueError("TOOL_RESULT_UNPAIRED")
            pending = None
        elif pending is not None:
            raise ValueError("TOOL_CALL_UNSETTLED")
    if pending is not None:
        raise ValueError("TOOL_CALL_UNSETTLED")


class ToolModelRequest[OutputT: BaseModel](_Contract):
    invocation_key: str = Field(min_length=1, max_length=255)
    items: tuple[ToolItem, ...]
    tools: tuple[ToolSchema, ...]
    response_schema: type[OutputT] = Field(exclude=True)
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=4000, gt=0, le=8000, strict=True)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        validate_history(self.items)
        names = [tool.wire_name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("TOOL_WIRE_NAME_COLLISION")
        return self


class ToolModelResponse[OutputT: BaseModel](_Contract):
    commentary: str | None = Field(default=None, max_length=4000)
    tool_call: ToolCall | None = None
    final: OutputT | None = None
    model_ref: str = Field(min_length=1, max_length=200)
    usage: ModelUsage | None = None
    finish: Literal["TOOL_CALL", "FINAL"] | None = None

    @model_validator(mode="after")
    def exactly_one_decision(self) -> Self:
        if (self.tool_call is None) == (self.final is None):
            raise ValueError("MODEL_STEP_REQUIRES_ONE_DECISION")
        finish = "TOOL_CALL" if self.tool_call is not None else "FINAL"
        if self.finish is not None and self.finish != finish:
            raise ValueError("MODEL_STEP_FINISH_MISMATCH")
        object.__setattr__(self, "finish", finish)
        return self


class NativeToolGateway(Protocol):
    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]: ...
