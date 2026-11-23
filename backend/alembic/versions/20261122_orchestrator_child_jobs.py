"""Pin async child jobs to conserved orchestration allocations.

Revision ID: 20261122_orch_child_jobs
Revises: 20261116_orch_retention
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261122_orch_child_jobs"
down_revision: str | Sequence[str] | None = "20261116_orch_retention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "orchestration_usage",
        sa.Column("job_owned", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_check_constraint(
        op.f("ck_orchestration_usage_job_owned"),
        "orchestration_usage",
        "NOT job_owned OR (allocation_id IS NOT NULL AND kind IN ('MODEL','TOOL','INTERVAL'))",
    )
    op.create_table(
        "orchestration_child_jobs",
        sa.Column("organization_id", sa.Uuid(), primary_key=True),
        sa.Column("orchestration_run_id", sa.Uuid(), primary_key=True),
        sa.Column("allocation_id", sa.Uuid(), primary_key=True),
        sa.Column("workflow_job_id", sa.Uuid(), nullable=True),
        sa.Column("generation_job_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "(workflow_job_id IS NULL) <> (generation_job_id IS NULL)",
            name=op.f("ck_orchestration_child_jobs_job"),
        ),
        sa.UniqueConstraint("organization_id", "workflow_job_id"),
        sa.UniqueConstraint("organization_id", "generation_job_id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "allocation_id"],
            [
                "orchestration_usage.organization_id",
                "orchestration_usage.orchestration_run_id",
                "orchestration_usage.operation_id",
            ],
            name="fk_orch_child_allocation",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "workflow_job_id"],
            ["workflow_jobs.organization_id", "workflow_jobs.id"],
            name="fk_orch_child_workflow_job",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "generation_job_id"],
            ["report_generation_jobs.organization_id", "report_generation_jobs.id"],
            name="fk_orch_child_generation_job",
        ),
    )
    op.execute("""
      ALTER TABLE orchestration_child_jobs ENABLE ROW LEVEL SECURITY;
      ALTER TABLE orchestration_child_jobs FORCE ROW LEVEL SECURITY;
      CREATE POLICY orchestration_child_job_owner ON orchestration_child_jobs TO app_runtime
        USING (EXISTS(SELECT 1 FROM orchestration_usage a
          WHERE a.organization_id=orchestration_child_jobs.organization_id
            AND a.orchestration_run_id=orchestration_child_jobs.orchestration_run_id
            AND a.operation_id=orchestration_child_jobs.allocation_id AND a.kind='ALLOCATION'))
        WITH CHECK (EXISTS(SELECT 1 FROM orchestration_usage a
          JOIN orchestration_runtime_states s USING (organization_id,orchestration_run_id)
          WHERE a.organization_id=orchestration_child_jobs.organization_id
            AND a.orchestration_run_id=orchestration_child_jobs.orchestration_run_id
            AND a.operation_id=orchestration_child_jobs.allocation_id AND a.kind='ALLOCATION'
            AND NOT a.closed AND s.lease_id IS NOT NULL AND s.lease_until>clock_timestamp()));
      GRANT SELECT, INSERT ON orchestration_child_jobs TO app_runtime;
    """)
    op.execute("""
      CREATE FUNCTION orchestration_child_claim_current(
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
            AND b.allocation_id=allocation AND r.status NOT IN ('COMPLETED','FAILED')
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
    op.execute("""
      CREATE OR REPLACE FUNCTION guard_orchestration_usage() RETURNS trigger
      LANGUAGE plpgsql AS $$
      DECLARE state orchestration_runtime_states%ROWTYPE;
              item record;
              allocation uuid;
              job_write boolean;
              bound boolean;
      BEGIN
        IF current_user <> 'app_runtime' THEN RETURN NEW; END IF;
        SELECT * INTO state FROM orchestration_runtime_states
          WHERE organization_id=NEW.organization_id AND
            orchestration_run_id=NEW.orchestration_run_id
          FOR UPDATE;
        IF NOT FOUND THEN RAISE EXCEPTION 'BUDGET_NOT_INITIALIZED'; END IF;
        allocation := CASE WHEN NEW.kind='ALLOCATION' THEN NEW.operation_id
                           ELSE NEW.allocation_id END;
        SELECT EXISTS(SELECT 1 FROM orchestration_child_jobs b
          WHERE b.organization_id=NEW.organization_id
            AND b.orchestration_run_id=NEW.orchestration_run_id
            AND b.allocation_id=allocation) INTO bound;
        job_write := NEW.job_owned OR (TG_OP='UPDATE' AND NEW.kind='ALLOCATION' AND bound);
        IF job_write AND NOT orchestration_child_claim_current(
          NEW.organization_id, NEW.orchestration_run_id, allocation,
          CASE WHEN NEW.job_owned THEN NEW.fence ELSE NULL END
        ) THEN RAISE EXCEPTION 'WORKFLOW_LEASE_LOST'; END IF;
        IF TG_OP='INSERT' THEN
          IF NEW.kind='POOL' THEN
            IF NEW.fence<>0 OR state.lease_id IS NOT NULL THEN RAISE EXCEPTION
              'BUDGET_POOL_INVALID'; END IF;
          ELSIF NOT job_write AND (NEW.fence<>state.fence OR
            state.lease_until<=clock_timestamp() OR state.lease_id IS NULL) THEN
            RAISE EXCEPTION 'LEASE_LOST';
          END IF;
          IF bound AND NOT job_write THEN RAISE EXCEPTION 'ALLOCATION_OWNED_BY_JOB'; END IF;
          IF NEW.allocation_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM orchestration_usage
             WHERE organization_id=NEW.organization_id AND
               orchestration_run_id=NEW.orchestration_run_id
               AND operation_id=NEW.allocation_id AND kind='ALLOCATION' AND NOT closed) THEN
            RAISE EXCEPTION 'ALLOCATION_INVALID';
          END IF;
          IF NEW.interval_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM orchestration_usage
             WHERE organization_id=NEW.organization_id AND
               orchestration_run_id=NEW.orchestration_run_id
               AND operation_id=NEW.interval_id AND kind='INTERVAL' AND NOT closed
               AND allocation_id IS NOT DISTINCT FROM NEW.allocation_id AND fence=NEW.fence
               AND job_owned=NEW.job_owned
               AND started_at + (amount->>'active_milliseconds')::bigint *
                 interval '1 millisecond'>clock_timestamp()) THEN
            RAISE EXCEPTION 'ACTIVE_INTERVAL_INVALID';
          END IF;
        ELSE
          IF OLD.closed OR NEW.kind='POOL' OR NOT NEW.closed OR
            (NOT job_write AND
              (state.lease_id IS NULL OR state.lease_until<=clock_timestamp())) THEN
            RAISE EXCEPTION 'BUDGET_SETTLEMENT_IMMUTABLE';
          END IF;
          IF (to_jsonb(NEW)-ARRAY['settled','settlement_fingerprint','closed']) IS DISTINCT FROM
             (to_jsonb(OLD)-ARRAY['settled','settlement_fingerprint','closed']) THEN
            RAISE EXCEPTION 'BUDGET_IDENTITY_IMMUTABLE';
          END IF;
          IF NEW.kind<>'ALLOCATION' AND (NEW.settled IS NULL OR
            NEW.settlement_fingerprint IS NULL) THEN
            RAISE EXCEPTION 'BUDGET_SETTLEMENT_INVALID';
          END IF;
        END IF;
        FOR item IN SELECT * FROM jsonb_each(NEW.amount) LOOP
          IF jsonb_typeof(item.value)<>'number' OR item.value::text !~ '^[0-9]+$' THEN
            RAISE EXCEPTION 'BUDGET_VECTOR_INVALID';
          END IF;
        END LOOP;
        IF NEW.settled IS NOT NULL THEN
          IF (SELECT array_agg(key ORDER BY key) FROM jsonb_each(NEW.settled)) IS DISTINCT FROM
             (SELECT array_agg(key ORDER BY key) FROM jsonb_each(NEW.amount))
               THEN RAISE EXCEPTION 'BUDGET_VECTOR_INVALID'; END IF;
          FOR item IN SELECT * FROM jsonb_each(NEW.settled) LOOP
            IF jsonb_typeof(item.value)<>'number' OR item.value::text !~ '^[0-9]+$'
               OR (item.value::text)::numeric>(NEW.amount->>item.key)::numeric
               OR (item.key NOT IN
                 ('input_tokens','output_tokens','active_milliseconds') AND
                 item.value<>NEW.amount->item.key) THEN
              RAISE EXCEPTION 'BUDGET_SETTLEMENT_INVALID';
            END IF;
          END LOOP;
        END IF;
        RETURN NEW;
      END $$;
    """)


def downgrade() -> None:
    raise RuntimeError("Forward-only child allocation ledger; retain additive schema on rollback")
