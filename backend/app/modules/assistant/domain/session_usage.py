"""Provider-independent, integer budget arithmetic. No authority in model usage."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from app.modules.assistant.domain.session import SessionError

DIMENSIONS = (
    "root_model_requests",
    "total_model_requests",
    "total_tool_calls",
    "handoffs",
    "input_tokens",
    "output_tokens",
    "segments",
    "active_milliseconds",
)
CAPS = dict(zip(DIMENSIONS, (8, 24, 12, 6, 48000, 8000, 4, 720000), strict=True))
POOL_ID = UUID(int=0)

type ChildJobKind = Literal["PLANNING", "REPORTING"]


@dataclass(frozen=True, slots=True)
class JobClaim:
    kind: ChildJobKind
    job_id: UUID
    worker_id: str
    attempt: int
    lease_until: datetime

    @property
    def fencing_token(self) -> int:
        return self.attempt


def vector(**values: int) -> dict[str, int]:
    return {key: values.get(key, 0) for key in DIMENSIONS}


def validate_vector(value: dict[str, int]) -> None:
    if set(value) != set(DIMENSIONS) or any(type(v) is not int or v < 0 for v in value.values()):
        raise SessionError("BUDGET_VECTOR_INVALID")


def add(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left[key] + right[key] for key in DIMENSIONS}


def subtract(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    value = {key: left[key] - right[key] for key in DIMENSIONS}
    if any(v < 0 for v in value.values()):
        raise SessionError("BUDGET_EXHAUSTED")
    return value


def limits_from_checkpoint(body: object) -> dict[str, int]:
    if not isinstance(body, dict):
        raise SessionError("BUDGET_LIMITS_INVALID")
    mapping = cast(dict[str, object], body)
    limits = mapping.get("limits")
    usage = mapping.get("usage")
    if (
        not isinstance(limits, dict)
        or not isinstance(usage, dict)
        or any(cast(dict[str, object], usage).values())
    ):
        raise SessionError("BUDGET_INITIAL_USAGE_INVALID")
    limits = cast(dict[str, object], limits)
    result: dict[str, int] = {}
    for key, maximum in CAPS.items():
        raw = limits.get("active_seconds" if key == "active_milliseconds" else key)
        if type(raw) is not int:
            raise SessionError("BUDGET_LIMITS_INVALID")
        value = raw * 1000 if key == "active_milliseconds" else raw
        if not 0 <= value <= maximum:
            raise SessionError("BUDGET_LIMITS_INVALID")
        result[key] = value
    for key, maximum in (
        ("segment_seconds", 180),
        ("retry_per_invocation", 1),
        ("repair_per_response", 1),
    ):
        value = limits.get(key)
        if type(value) is not int or not 0 <= value <= maximum:
            raise SessionError("BUDGET_LIMITS_INVALID")
        result[key] = value
    if (
        result["segment_seconds"] == 0
        or result["root_model_requests"] > result["total_model_requests"]
    ):
        raise SessionError("BUDGET_LIMITS_INVALID")
    return result


@dataclass(frozen=True)
class UsageEntry:
    operation_id: UUID
    kind: str
    amount: dict[str, int]
    fingerprint: str
    fence: int
    started_at: datetime
    allocation_id: UUID | None = None
    interval_id: UUID | None = None
    attempt_parent: UUID | None = None
    attempt_kind: str | None = None
    settled: dict[str, int] | None = None
    settlement_fingerprint: str | None = None
    closed: bool = False
    job_owned: bool = False

    @property
    def charge(self) -> dict[str, int]:
        return self.settled if self.settled is not None else self.amount


def balances(entries: tuple[UsageEntry, ...]) -> tuple[dict[str, int], dict[str, int]]:
    pool = next((e for e in entries if e.kind == "POOL"), None)
    if pool is None:
        raise SessionError("BUDGET_NOT_INITIALIZED")
    consumed = vector()
    held = vector()
    for entry in entries:
        if entry.kind not in {"POOL", "ALLOCATION"}:
            consumed = add(consumed, entry.charge)
    for allocation in entries:
        if allocation.kind == "ALLOCATION" and not allocation.closed:
            spent = vector()
            for entry in entries:
                if entry.allocation_id == allocation.operation_id:
                    spent = add(spent, entry.charge)
            held = add(held, subtract(allocation.amount, spent))
    available = subtract({key: pool.amount[key] for key in DIMENSIONS}, add(consumed, held))
    return consumed, available
