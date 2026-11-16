"""Durable budget ledger. All IDs and foreign keys include tenant and run."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class OrchestrationUsageModel(Base):
    __tablename__ = "orchestration_usage"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_usage_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "allocation_id"],
            [
                "orchestration_usage.organization_id",
                "orchestration_usage.orchestration_run_id",
                "orchestration_usage.operation_id",
            ],
            name="fk_usage_allocation",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "interval_id"],
            [
                "orchestration_usage.organization_id",
                "orchestration_usage.orchestration_run_id",
                "orchestration_usage.operation_id",
            ],
            name="fk_usage_interval",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "attempt_parent"],
            [
                "orchestration_usage.organization_id",
                "orchestration_usage.orchestration_run_id",
                "orchestration_usage.operation_id",
            ],
            name="fk_usage_attempt",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "kind IN ('POOL','ALLOCATION','MODEL','TOOL','HANDOFF','INTERVAL')", name="kind"
        ),
        CheckConstraint("fence>=0 AND (kind='POOL' OR fence>0)", name="fence"),
        CheckConstraint(
            "(attempt_parent IS NULL AND attempt_kind IS NULL) OR (attempt_parent IS NOT NULL "
            "AND attempt_kind IN ('RETRY','REPAIR') AND kind='MODEL')",
            name="attempt",
        ),
        CheckConstraint(
            "(kind IN ('POOL','ALLOCATION','INTERVAL') AND interval_id IS NULL) OR (kind IN "
            "('MODEL','TOOL','HANDOFF') AND interval_id IS NOT NULL)",
            name="interval",
        ),
        Index(
            "uq_usage_pool",
            "organization_id",
            "orchestration_run_id",
            unique=True,
            postgresql_where=text("kind='POOL'"),
        ),
        Index(
            "uq_usage_attempt",
            "organization_id",
            "orchestration_run_id",
            "attempt_parent",
            "attempt_kind",
            unique=True,
            postgresql_where=text("attempt_parent IS NOT NULL"),
        ),
    )
    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    orchestration_run_id: Mapped[UUID] = mapped_column(primary_key=True)
    operation_id: Mapped[UUID] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    amount: Mapped[dict[str, int]] = mapped_column(JSONB)
    fingerprint: Mapped[str] = mapped_column(String(64))
    fence: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    allocation_id: Mapped[UUID | None]
    interval_id: Mapped[UUID | None]
    attempt_parent: Mapped[UUID | None]
    attempt_kind: Mapped[str | None] = mapped_column(String(8))
    settled: Mapped[dict[str, int] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    settlement_fingerprint: Mapped[str | None] = mapped_column(String(64))
    closed: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
