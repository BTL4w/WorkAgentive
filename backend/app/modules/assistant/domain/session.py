"""Application-owned private journal records; no AI/provider or database imports."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


class SessionError(ValueError):
    """A safe error code, never provider text or private payload content."""


@dataclass(frozen=True, slots=True)
class SessionScope:
    organization_id: UUID
    membership_id: UUID
    conversation_id: UUID
    turn_id: UUID
    orchestration_run_id: UUID


@dataclass(frozen=True, slots=True)
class SessionPins:
    driver_version: str
    prompt_version: str
    catalog_fingerprint: str
    agent_version: str
    manifest_fingerprint: str | None
    model_ref: str | None
    schema_version: str = "2.0"
    driver: str = "tool_loop"


@dataclass(frozen=True, slots=True)
class SessionLease:
    scope: SessionScope
    lease_id: UUID
    job_id: UUID
    worker_id: str
    fencing_token: int
    expires_at: datetime
    job_attempt: int | None = None


@dataclass(frozen=True, slots=True)
class SessionPayload:
    id: UUID
    organization_id: UUID
    orchestration_run_id: UUID
    kind: str
    classification: str
    body: object
    fingerprint: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class SessionEvent:
    event_id: UUID
    kind: str
    payload_id: UUID | None = None
    invocation_id: UUID | None = None
    call_id: str | None = None
    tool_reference: str | None = None
    tool_version: str | None = None
    arguments_fingerprint: str | None = None
    settlement_status: str | None = None
    wait_id: UUID | None = None
    wait_kind: str | None = None
    resource_id: UUID | None = None
    resource_version: int | None = None
    policy_fingerprint: str | None = None
    source_fingerprint: str | None = None
    safe_code: str | None = None
    agent_run_id: UUID | None = None
    causation_id: UUID | None = None
    correlation_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class StoredSessionEvent:
    scope: SessionScope
    sequence: int
    occurred_at: datetime
    event: SessionEvent
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RuntimeState:
    scope: SessionScope
    pins: SessionPins
    sequence: int = 0
    checkpoint_sequence: int = 0
    checkpoint_payload_id: UUID | None = None
    lease: SessionLease | None = None
    fence: int = 0
