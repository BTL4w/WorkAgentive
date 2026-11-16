"""In-memory contract fakes, not a production session log or authorization layer."""

from datetime import datetime, timedelta

from pydantic import BaseModel

from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopCheckpoint,
)

ScriptedToolGateway = MockModelGateway


class FakeClock:
    def __init__(self, initial: datetime) -> None:
        if initial.tzinfo is None or initial.utcoffset() is None:
            raise ValueError("aware clock required")
        self._now = initial

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        if seconds < 0:
            raise ValueError("clock cannot move backwards")
        self._now += timedelta(seconds=seconds)


class FakeCheckpointStore:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self._checkpoints: dict[RunScope, ToolLoopCheckpoint] = {}
        self._fences: dict[RunScope, int] = {}

    def save(self, checkpoint: ToolLoopCheckpoint, *, lease: RunLease) -> None:
        if (
            lease.scope != checkpoint.scope
            or lease.expires_at <= self.clock.now()
            or lease.fencing_token < self._fences.get(checkpoint.scope, 0)
        ):
            raise ValueError("FAKE_LEASE_REJECTED")
        self._fences[checkpoint.scope] = lease.fencing_token
        self._checkpoints[checkpoint.scope] = checkpoint.model_copy(deep=True)

    def load(self, scope: RunScope) -> ToolLoopCheckpoint | None:
        checkpoint = self._checkpoints.get(scope)
        return checkpoint.model_copy(deep=True) if checkpoint is not None else None


class FakeToolBackend:
    def __init__(self, result: BackendToolResult) -> None:
        self._result = result.model_copy(deep=True)
        self.calls: list[tuple[ToolInvocationContext, BaseModel]] = []

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        self.calls.append((context.model_copy(deep=True), arguments.model_copy(deep=True)))
        return self._result.model_copy(deep=True)
