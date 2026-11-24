"""SQL model for run-scoped continuation ownership and wake receipts."""

from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class OrchestrationWaitModel(Base):
    __tablename__ = "orchestration_waits"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "source_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_orchestration_wait_source",
        ),
        ForeignKeyConstraint(
            ["organization_id", "source_run_id", "child_allocation_id"],
            [
                "orchestration_child_jobs.organization_id",
                "orchestration_child_jobs.orchestration_run_id",
                "orchestration_child_jobs.allocation_id",
            ],
            name="fk_orchestration_wait_child",
        ),
        ForeignKeyConstraint(
            ["organization_id", "owner_run_id", "owner_turn_id", "membership_id"],
            [
                "orchestration_runs.organization_id",
                "orchestration_runs.id",
                "orchestration_runs.turn_id",
                "orchestration_runs.actor_membership_id",
            ],
            name="fk_orchestration_wait_owner_run",
        ),
        ForeignKeyConstraint(
            ["organization_id", "owner_turn_id", "conversation_id", "membership_id"],
            [
                "assistant_turns.organization_id",
                "assistant_turns.id",
                "assistant_turns.conversation_id",
                "assistant_turns.actor_membership_id",
            ],
            name="fk_orchestration_wait_owner_turn",
        ),
        ForeignKeyConstraint(
            ["organization_id", "wake_job_id", "owner_run_id"],
            [
                "assistant_jobs.organization_id",
                "assistant_jobs.id",
                "assistant_jobs.orchestration_run_id",
            ],
            name="fk_orchestration_wait_job",
        ),
        ForeignKeyConstraint(
            ["organization_id", "workflow_event_id"],
            ["workflow_events.organization_id", "workflow_events.id"],
            name="fk_orchestration_wait_event",
        ),
        CheckConstraint("kind IN ('USER_INPUT','WORKFLOW_RESULT','HUMAN_DECISION')", name="kind"),
        CheckConstraint(
            "(resource_id IS NULL AND resource_version IS NULL) OR "
            "(resource_id IS NOT NULL AND resource_version IS NOT NULL AND resource_version >= 1)",
            name="resource",
        ),
        CheckConstraint("version >= 1", name="version"),
        Index(
            "ix_orchestration_waits_source",
            "organization_id",
            "source_run_id",
            "source_invocation_id",
        ),
        Index(
            "uq_orchestration_waits_open_owner",
            "organization_id",
            "owner_run_id",
            unique=True,
            postgresql_where=text("closed IS FALSE"),
        ),
    )

    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    wait_id: Mapped[UUID] = mapped_column(primary_key=True)
    source_run_id: Mapped[UUID]
    source_turn_id: Mapped[UUID]
    source_invocation_id: Mapped[UUID]
    child_allocation_id: Mapped[UUID | None]
    membership_id: Mapped[UUID]
    conversation_id: Mapped[UUID]
    owner_run_id: Mapped[UUID]
    owner_turn_id: Mapped[UUID]
    version: Mapped[int]
    kind: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[UUID | None]
    resource_version: Mapped[int | None]
    source_fingerprint: Mapped[str | None] = mapped_column(String(128))
    policy_fingerprint: Mapped[str | None] = mapped_column(String(128))
    workflow_event_id: Mapped[UUID | None]
    wake_job_id: Mapped[UUID | None]
    context_binding: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    closed: Mapped[bool]
