"""orchestrator_continuations

Revision ID: 20261116_orch_waits
Revises: 20261122_orch_child_jobs
Create Date: 2026-11-24 06:29:36.363171
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261116_orch_waits"
down_revision: str | Sequence[str] | None = "20261122_orch_child_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""

    op.create_table(
        "orchestration_waits",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("wait_id", sa.Uuid(), nullable=False),
        sa.Column("source_run_id", sa.Uuid(), nullable=False),
        sa.Column("source_turn_id", sa.Uuid(), nullable=False),
        sa.Column("source_invocation_id", sa.Uuid(), nullable=False),
        sa.Column("membership_id", sa.Uuid(), nullable=False),
        sa.Column("child_allocation_id", sa.Uuid(), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("owner_run_id", sa.Uuid(), nullable=False),
        sa.Column("owner_turn_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("resource_version", sa.Integer(), nullable=True),
        sa.Column("source_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("policy_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("workflow_event_id", sa.Uuid(), nullable=True),
        sa.Column("wake_job_id", sa.Uuid(), nullable=True),
        sa.Column("context_binding", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("closed", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('USER_INPUT','WORKFLOW_RESULT','HUMAN_DECISION')",
            name=op.f("ck_orchestration_waits_kind"),
        ),
        sa.CheckConstraint(
            "(resource_id IS NULL AND resource_version IS NULL) OR "
            "(resource_id IS NOT NULL AND resource_version IS NOT NULL AND resource_version >= 1)",
            name=op.f("ck_orchestration_waits_resource"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_orchestration_waits_version")),
        sa.ForeignKeyConstraint(
            ["organization_id", "owner_run_id", "owner_turn_id", "membership_id"],
            [
                "orchestration_runs.organization_id",
                "orchestration_runs.id",
                "orchestration_runs.turn_id",
                "orchestration_runs.actor_membership_id",
            ],
            name="fk_orchestration_wait_owner_run",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "owner_turn_id", "conversation_id", "membership_id"],
            [
                "assistant_turns.organization_id",
                "assistant_turns.id",
                "assistant_turns.conversation_id",
                "assistant_turns.actor_membership_id",
            ],
            name="fk_orchestration_wait_owner_turn",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "source_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_orchestration_wait_source",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "wake_job_id", "owner_run_id"],
            [
                "assistant_jobs.organization_id",
                "assistant_jobs.id",
                "assistant_jobs.orchestration_run_id",
            ],
            name="fk_orchestration_wait_job",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "workflow_event_id"],
            ["workflow_events.organization_id", "workflow_events.id"],
            name="fk_orchestration_wait_event",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "source_run_id", "child_allocation_id"],
            [
                "orchestration_child_jobs.organization_id",
                "orchestration_child_jobs.orchestration_run_id",
                "orchestration_child_jobs.allocation_id",
            ],
            name="fk_orchestration_wait_child",
        ),
        sa.PrimaryKeyConstraint("organization_id", "wait_id", name=op.f("pk_orchestration_waits")),
    )
    op.create_index(
        "ix_orchestration_waits_source",
        "orchestration_waits",
        ["organization_id", "source_run_id", "source_invocation_id"],
        unique=False,
    )
    op.create_index(
        "uq_orchestration_waits_open_owner",
        "orchestration_waits",
        ["organization_id", "owner_run_id"],
        unique=True,
        postgresql_where=sa.text("closed IS FALSE"),
    )
    op.drop_constraint(
        op.f("uq_assistant_jobs_organization_id_orchestration_run_id_job_type"),
        "assistant_jobs",
        type_="unique",
    )
    op.create_index(
        "uq_assistant_jobs_run_initial",
        "assistant_jobs",
        ["organization_id", "orchestration_run_id", "job_type"],
        unique=True,
        postgresql_where=sa.text("job_type <> 'assistant.turn.resume'"),
    )
    op.drop_constraint(op.f("ck_assistant_jobs_job_type"), "assistant_jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_assistant_jobs_job_type"),
        "assistant_jobs",
        "job_type IN ('assistant.turn.execute', 'assistant.conversation.title', "
        "'assistant.turn.resume')",
    )
    owner = """organization_id=NULLIF(current_setting('app.organization_id', true), '')::uuid
        AND membership_id=NULLIF(current_setting('app.membership_id', true), '')::uuid
        AND EXISTS (
          SELECT 1 FROM orchestration_runtime_states s
          WHERE s.organization_id=orchestration_waits.organization_id
            AND s.orchestration_run_id=orchestration_waits.source_run_id
            AND s.turn_id=orchestration_waits.source_turn_id
            AND s.conversation_id=orchestration_waits.conversation_id
            AND s.membership_id=orchestration_waits.membership_id
        )"""
    op.execute("ALTER TABLE orchestration_waits OWNER TO migration_owner")
    op.execute("ALTER TABLE orchestration_waits ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE orchestration_waits FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY private_wait_owner ON orchestration_waits TO app_runtime "
        f"USING ({owner}) WITH CHECK ({owner})"
    )
    op.execute("GRANT SELECT, INSERT ON orchestration_waits TO app_runtime")
    op.execute(
        "GRANT UPDATE (owner_run_id, owner_turn_id, version, workflow_event_id, "
        "wake_job_id, closed, context_binding) ON orchestration_waits TO app_runtime"
    )

    _child_claim_current(allow_transferred=True)


def downgrade() -> None:
    """Reverse the schema change when the revision supports it."""
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM assistant_jobs WHERE job_type='assistant.turn.resume')
            OR EXISTS (SELECT 1 FROM orchestration_waits)
        THEN RAISE EXCEPTION 'Continuation records require an explicit recovery before downgrade';
        END IF;
    END $$""")
    op.drop_constraint(op.f("ck_assistant_jobs_job_type"), "assistant_jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_assistant_jobs_job_type"),
        "assistant_jobs",
        "job_type IN ('assistant.turn.execute', 'assistant.conversation.title')",
    )
    op.drop_index(
        "uq_assistant_jobs_run_initial",
        table_name="assistant_jobs",
        postgresql_where=sa.text("job_type <> 'assistant.turn.resume'"),
    )
    op.create_unique_constraint(
        op.f("uq_assistant_jobs_organization_id_orchestration_run_id_job_type"),
        "assistant_jobs",
        ["organization_id", "orchestration_run_id", "job_type"],
        postgresql_nulls_not_distinct=False,
    )
    op.drop_index(
        "uq_orchestration_waits_open_owner",
        table_name="orchestration_waits",
        postgresql_where=sa.text("closed IS FALSE"),
    )
    op.drop_index("ix_orchestration_waits_source", table_name="orchestration_waits")
    _child_claim_current(allow_transferred=False)
    op.drop_table("orchestration_waits")


def _child_claim_current(*, allow_transferred: bool) -> None:
    active = "r.status NOT IN ('COMPLETED','FAILED')"
    eligible = active
    if allow_transferred:
        eligible += """ OR (r.status='COMPLETED' AND r.stop_reason='CONTINUATION_TRANSFERRED'
          AND EXISTS (
            SELECT 1 FROM orchestration_waits wait
            WHERE wait.organization_id=b.organization_id
              AND wait.source_run_id=b.orchestration_run_id
              AND wait.child_allocation_id=b.allocation_id
              AND wait.kind='WORKFLOW_RESULT'
              AND wait.owner_run_id<>wait.source_run_id
          ))"""
    op.execute(f"""
      CREATE OR REPLACE FUNCTION orchestration_child_claim_current(
        org_id uuid, orchestration_id uuid, allocation uuid, expected_attempt integer
      ) RETURNS boolean LANGUAGE sql AS $$
        SELECT EXISTS (
          SELECT 1 FROM orchestration_child_jobs b
          JOIN orchestration_runs r ON r.organization_id=b.organization_id
            AND r.id=b.orchestration_run_id
          LEFT JOIN workflow_jobs j
            ON j.organization_id=b.organization_id AND j.id=b.workflow_job_id
          LEFT JOIN workflow_runs w
            ON w.organization_id=j.organization_id AND w.id=j.workflow_run_id
          LEFT JOIN report_generation_jobs g ON g.organization_id=b.organization_id
            AND g.id=b.generation_job_id
          WHERE b.organization_id=org_id AND b.orchestration_run_id=orchestration_id
            AND b.allocation_id=allocation
            AND ({eligible})
            AND r.actor_membership_id=NULLIF(current_setting('app.membership_id',true),'')::uuid
            AND COALESCE(j.id,g.id)=
                NULLIF(current_setting('app.orchestration_child_job',true),'')::uuid
            AND COALESCE(j.locked_by_worker_id,g.lease_owner)=
                current_setting('app.orchestration_child_worker',true)
            AND COALESCE(j.attempt_count,g.fence)=
                NULLIF(current_setting('app.orchestration_child_attempt',true),'')::integer
            AND (expected_attempt IS NULL OR COALESCE(j.attempt_count,g.fence)=expected_attempt)
            AND COALESCE(j.lease_until,g.lease_until)>clock_timestamp()
            AND COALESCE(j.status,g.state)='RUNNING'
            AND COALESCE(w.requested_by_membership_id,g.requester_membership_id)=
                r.actor_membership_id
        );
      $$;
    """)
