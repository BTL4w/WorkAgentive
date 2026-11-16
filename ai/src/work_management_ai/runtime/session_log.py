"""Private ordered journal port. Event headers carry IDs, never raw model context."""

from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from work_management_ai.runtime.contracts import JsonValue
from work_management_ai.runtime.tool_loop_contracts import (
    PayloadReference,
    RunLease,
    RunScope,
    ToolLoopCheckpoint,
    WaitReference,
)

type EventKind = Literal[
    "turn.started",
    "context.selected",
    "model.requested",
    "model.completed",
    "model.failed",
    "tool.requested",
    "tool.authorized",
    "tool.denied",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "tool.observed",
    "run.waiting",
    "continuation.received",
    "run.resumed",
    "turn.completed",
    "run.cancelled",
    "run.failed",
]
type PayloadKind = Literal[
    "OBJECTIVE",
    "CONTEXT",
    "TOOL_ARGUMENTS",
    "TOOL_RESULT",
    "MODEL_OUTPUT",
    "SKILL_BODY",
    "WORKING_PLAN",
    "CHECKPOINT",
]


class InvocationStartedError(ValueError):
    """A concurrent executor owns dispatch; rejection must not close its receipt."""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class InvocationDescriptor(_Contract):
    invocation_id: UUID
    call_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    tool_reference: str = Field(pattern=r"^[a-z][a-z0-9_.]*@[1-9][0-9]*$", max_length=100)
    tool_version: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    arguments_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class SettledInvocation(_Contract):
    status: Literal["COMPLETED", "DEFERRED", "REJECTED", "FAILED"]
    payload_ref: PayloadReference | None = None
    wait_ref: WaitReference | None = None
    safe_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,99}$")

    @model_validator(mode="after")
    def result_shape(self) -> Self:
        if self.status == "COMPLETED":
            valid = (
                self.payload_ref is not None and self.wait_ref is None and self.safe_code is None
            )
        elif self.status == "DEFERRED":
            valid = (
                self.wait_ref is not None and self.payload_ref is None and self.safe_code is None
            )
        else:
            valid = (
                self.safe_code is not None and self.payload_ref is None and self.wait_ref is None
            )
        if not valid:
            raise ValueError("SETTLEMENT_SHAPE_INVALID")
        return self


class EventDraft(_Contract):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID
    kind: EventKind
    safe_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,99}$")
    payload_ref: PayloadReference | None = None
    invocation: InvocationDescriptor | None = None
    settlement: SettledInvocation | None = None
    agent_run_id: UUID | None = None
    causation_id: UUID | None = None
    correlation_id: UUID | None = None

    @model_validator(mode="after")
    def invocation_shape(self) -> Self:
        if self.safe_code is not None and (
            self.kind != "tool.denied" or self.settlement is not None
        ):
            raise ValueError("DIAGNOSTIC_EVENT_INVALID")
        if (
            self.kind in {"tool.requested", "tool.completed", "tool.failed"}
            and self.invocation is None
        ):
            raise ValueError("INVOCATION_REQUIRED")
        if self.kind in {"tool.completed", "tool.failed"} and self.settlement is None:
            raise ValueError("SETTLEMENT_REQUIRED")
        if self.settlement is not None and self.invocation is None:
            raise ValueError("SETTLEMENT_INVOCATION_REQUIRED")
        if self.invocation is not None:
            if self.kind == "tool.requested":
                if self.settlement is not None or self.payload_ref is None:
                    raise ValueError("TOOL_REQUEST_PAYLOAD_REQUIRED")
                if self.payload_ref.fingerprint != self.invocation.arguments_fingerprint:
                    raise ValueError("TOOL_ARGUMENT_FINGERPRINT_MISMATCH")
            elif self.settlement is not None:
                expected = (
                    "run.waiting"
                    if self.settlement.status == "DEFERRED"
                    else "tool.failed"
                    if self.settlement.status == "FAILED"
                    else "tool.completed"
                )
                if self.kind != expected or self.payload_ref != self.settlement.payload_ref:
                    raise ValueError("SETTLEMENT_EVENT_MISMATCH")
        return self


class SessionRecord(EventDraft):
    scope: RunScope
    sequence: int = Field(ge=1, strict=True)
    occurred_at: AwareDatetime


class InvocationReceipt(_Contract):
    scope: RunScope
    descriptor: InvocationDescriptor
    requested_event_id: UUID
    requested_sequence: int = Field(ge=1, strict=True)
    arguments_ref: PayloadReference
    settlement: SettledInvocation | None = None
    settled_event_id: UUID | None = None


class SessionLogPort(Protocol):
    async def initialize(self, scope: RunScope, state: ToolLoopCheckpoint) -> None: ...
    async def append(
        self, scope: RunScope, event: EventDraft, *, expected_sequence: int, lease: RunLease
    ) -> SessionRecord: ...
    async def read(self, scope: RunScope, *, after_sequence: int) -> tuple[SessionRecord, ...]: ...
    async def checkpoint(
        self, scope: RunScope, state: ToolLoopCheckpoint, *, lease: RunLease
    ) -> None: ...
    async def load_checkpoint(self, scope: RunScope) -> ToolLoopCheckpoint | None: ...
    async def claim(
        self, scope: RunScope, *, job_id: UUID, worker_id: str, lease_seconds: int
    ) -> RunLease: ...
    async def renew(self, scope: RunScope, lease: RunLease, *, lease_seconds: int) -> RunLease: ...
    async def get_invocation(
        self, scope: RunScope, invocation_id: UUID
    ) -> InvocationReceipt | None: ...
    async def start_invocation(
        self, scope: RunScope, receipt: InvocationReceipt, *, lease: RunLease
    ) -> bool:
        """Atomically claim first dispatch; False permits reconciliation only."""
        ...

    async def settle_invocation(
        self,
        scope: RunScope,
        receipt: InvocationReceipt,
        value: SettledInvocation,
        *,
        lease: RunLease,
    ) -> SessionRecord: ...
    async def put_payload(
        self,
        scope: RunScope,
        body: JsonValue,
        *,
        classification: Literal["RAW_CONTEXT"],
        kind: PayloadKind,
        lease: RunLease,
        ttl_seconds: int | None = None,
    ) -> PayloadReference: ...
    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> JsonValue: ...
