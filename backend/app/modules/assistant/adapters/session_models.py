"""Tenant/run scoped journal headers, expiring payloads and mutable recovery cursor."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class OrchestrationRuntimeStateModel(Base):
    __tablename__ = "orchestration_runtime_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "turn_id", "membership_id"],
            [
                "orchestration_runs.organization_id",
                "orchestration_runs.id",
                "orchestration_runs.turn_id",
                "orchestration_runs.actor_membership_id",
            ],
            name="fk_session_state_run_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "turn_id", "conversation_id", "membership_id"],
            [
                "assistant_turns.organization_id",
                "assistant_turns.id",
                "assistant_turns.conversation_id",
                "assistant_turns.actor_membership_id",
            ],
            name="fk_session_state_turn_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "lease_job_id", "orchestration_run_id"],
            [
                "assistant_jobs.organization_id",
                "assistant_jobs.id",
                "assistant_jobs.orchestration_run_id",
            ],
            name="fk_session_state_job_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "checkpoint_payload_id"],
            [
                "orchestration_session_payloads.organization_id",
                "orchestration_session_payloads.orchestration_run_id",
                "orchestration_session_payloads.id",
            ],
            name="fk_session_checkpoint_payload",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint("driver='tool_loop' AND schema_version='2.0'", name="driver"),
        CheckConstraint(
            "sequence>=0 AND checkpoint_sequence>=0 AND checkpoint_sequence<=sequence AND fence>=0",
            name="cursor",
        ),
        CheckConstraint(
            (
                "(lease_id IS NULL AND lease_job_id IS NULL AND lease_owner IS NULL "
                "AND lease_until IS NULL) OR (lease_id IS NOT NULL AND lease_job_id "
                "IS NOT NULL AND lease_owner IS NOT NULL AND lease_until IS NOT NULL)"
            ),
            name="lease_shape",
        ),
    )
    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    orchestration_run_id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID]
    turn_id: Mapped[UUID]
    membership_id: Mapped[UUID]
    driver: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[str] = mapped_column(String(16))
    driver_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    catalog_fingerprint: Mapped[str] = mapped_column(String(128))
    agent_version: Mapped[str] = mapped_column(String(64))
    manifest_fingerprint: Mapped[str | None] = mapped_column(String(128))
    model_ref: Mapped[str | None] = mapped_column(String(200))
    sequence: Mapped[int] = mapped_column(Integer, server_default="0")
    checkpoint_sequence: Mapped[int] = mapped_column(Integer, server_default="0")
    checkpoint_payload_id: Mapped[UUID | None]
    lease_id: Mapped[UUID | None]
    lease_job_id: Mapped[UUID | None]
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fence: Mapped[int] = mapped_column(Integer, server_default="0")


class OrchestrationSessionPayloadModel(Base):
    __tablename__ = "orchestration_session_payloads"
    __table_args__ = (
        UniqueConstraint("organization_id", "orchestration_run_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_session_payload_run",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            (
                "classification='RAW_CONTEXT' AND expires_at>created_at AND "
                "expires_at<=created_at+interval '30 days'"
            ),
            name="retention",
        ),
        CheckConstraint(
            (
                "kind IN ('OBJECTIVE','CONTEXT','TOOL_ARGUMENTS','TOOL_RESULT',"
                "'MODEL_OUTPUT','SKILL_BODY','WORKING_PLAN','CHECKPOINT')"
            ),
            name="kind",
        ),
        CheckConstraint("fingerprint ~ '^[a-f0-9]{64}$'", name="fingerprint"),
        Index("ix_session_payload_expiry", "organization_id", "expires_at", "id"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    orchestration_run_id: Mapped[UUID]
    kind: Mapped[str] = mapped_column(String(32))
    classification: Mapped[str] = mapped_column(String(24))
    body: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))
    fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OrchestrationSessionEventModel(Base):
    __tablename__ = "orchestration_session_events"
    __table_args__ = (
        UniqueConstraint("organization_id", "orchestration_run_id", "event_id"),
        UniqueConstraint("organization_id", "orchestration_run_id", "sequence"),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_session_event_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "payload_id"],
            [
                "orchestration_session_payloads.organization_id",
                "orchestration_session_payloads.orchestration_run_id",
                "orchestration_session_payloads.id",
            ],
            name="fk_session_event_payload",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "agent_run_id"],
            ["agent_runs.organization_id", "agent_runs.id"],
            name="fk_session_event_agent",
            ondelete="RESTRICT",
        ),
        CheckConstraint("sequence>=1", name="sequence"),
        CheckConstraint("schema_version='1.0' AND fingerprint ~ '^[a-f0-9]{64}$'", name="envelope"),
        CheckConstraint(
            (
                "kind IN ('turn.started','context.selected','model.requested',"
                "'model.completed','model.failed','tool.requested','tool.authorized',"
                "'tool.denied','tool.started','tool.completed','tool.failed',"
                "'tool.observed','run.waiting','continuation.received','run.resumed',"
                "'turn.completed','run.cancelled','run.failed')"
            ),
            name="kind",
        ),
        CheckConstraint(
            "safe_code IS NULL OR safe_code ~ '^[A-Z][A-Z0-9_]{0,99}$'", name="safe_code"
        ),
        CheckConstraint(
            (
                "settlement_status IS NULL OR settlement_status IN ('COMPLETED',"
                "'DEFERRED','REJECTED','FAILED')"
            ),
            name="settlement",
        ),
        Index(
            "ix_session_event_invocation",
            "organization_id",
            "orchestration_run_id",
            "invocation_id",
            "sequence",
        ),
        Index(
            "uq_session_invocation_request",
            "organization_id",
            "orchestration_run_id",
            "invocation_id",
            unique=True,
            postgresql_where=text("kind='tool.requested'"),
        ),
        Index(
            "uq_session_invocation_settle",
            "organization_id",
            "orchestration_run_id",
            "invocation_id",
            unique=True,
            postgresql_where=text("settlement_status IS NOT NULL"),
        ),
        Index(
            "uq_session_call",
            "organization_id",
            "orchestration_run_id",
            "call_id",
            unique=True,
            postgresql_where=text("kind='tool.requested'"),
        ),
    )
    event_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    orchestration_run_id: Mapped[UUID]
    sequence: Mapped[int] = mapped_column(Integer)
    schema_version: Mapped[str] = mapped_column(String(16), server_default="1.0")
    kind: Mapped[str] = mapped_column(String(64))
    fingerprint: Mapped[str] = mapped_column(String(64))
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    payload_id: Mapped[UUID | None]
    invocation_id: Mapped[UUID | None]
    call_id: Mapped[str | None] = mapped_column(String(128))
    tool_reference: Mapped[str | None] = mapped_column(String(100))
    tool_version: Mapped[str | None] = mapped_column(String(64))
    arguments_fingerprint: Mapped[str | None] = mapped_column(String(64))
    settlement_status: Mapped[str | None] = mapped_column(String(16))
    wait_id: Mapped[UUID | None]
    wait_kind: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[UUID | None]
    resource_version: Mapped[int | None]
    policy_fingerprint: Mapped[str | None] = mapped_column(String(128))
    source_fingerprint: Mapped[str | None] = mapped_column(String(128))
    safe_code: Mapped[str | None] = mapped_column(String(100))
    agent_run_id: Mapped[UUID | None]
    causation_id: Mapped[UUID | None]
    correlation_id: Mapped[UUID | None]
