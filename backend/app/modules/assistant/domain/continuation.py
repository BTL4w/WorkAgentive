"""Durable continuation references and ownership receipts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from work_management_ai.runtime.tool_loop_contracts import RunLease, WaitReference


@dataclass(frozen=True, slots=True)
class WaitDraft:
    reference: "WaitReference"
    lease: "RunLease"


@dataclass(frozen=True, slots=True)
class WaitRecord:
    reference: "WaitReference"
    owner_run_id: UUID
    owner_turn_id: UUID
    version: int


@dataclass(frozen=True, slots=True)
class TransferReceipt:
    wait: WaitRecord
    previous_run_id: UUID


@dataclass(frozen=True, slots=True)
class WakeDecision:
    outcome: Literal["IGNORED", "QUEUED", "DUPLICATE"]
    job_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class WakeReceipt:
    wait: WaitRecord
    outcome: str
