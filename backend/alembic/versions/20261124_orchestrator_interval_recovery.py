"""Allow a reclaimed Assistant job to resume its original active interval.

Revision ID: 20261124_orch_interval
Revises: 20261116_orch_waits
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261124_orch_interval"
down_revision: str | Sequence[str] | None = "20261116_orch_waits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "orchestration_runtime_states", sa.Column("lease_job_attempt", sa.Integer(), nullable=True)
    )
    op.execute("GRANT UPDATE (lease_job_attempt) ON orchestration_runtime_states TO app_runtime")
    op.execute("GRANT UPDATE (fence) ON orchestration_usage TO app_runtime")
    op.execute("""
      CREATE OR REPLACE FUNCTION protect_session_state() RETURNS trigger LANGUAGE plpgsql
      SET search_path=pg_catalog,public AS $$
      DECLARE mutable_columns text[] := ARRAY[
        'sequence','checkpoint_sequence','checkpoint_payload_id','lease_id','lease_job_id',
        'lease_job_attempt','lease_owner','lease_until','fence'
      ];
      BEGIN
        IF (to_jsonb(NEW)-mutable_columns) IS DISTINCT FROM (to_jsonb(OLD)-mutable_columns)
          OR NEW.fence<OLD.fence
        THEN RAISE EXCEPTION 'runtime pins are immutable'; END IF;
        IF NEW.sequence<>(SELECT COALESCE(max(sequence),0) FROM orchestration_session_events
          WHERE organization_id=NEW.organization_id
            AND orchestration_run_id=NEW.orchestration_run_id)
        THEN RAISE EXCEPTION 'runtime cursor must match journal'; END IF;
        IF current_user='app_runtime' AND NEW.lease_id IS NOT NULL
          AND (NEW.lease_job_attempt IS NOT NULL OR OLD.lease_job_attempt IS NOT NULL)
          AND ROW(NEW.lease_id,NEW.lease_job_id,NEW.lease_owner,NEW.lease_job_attempt,
                  NEW.lease_until,NEW.fence) IS DISTINCT FROM
              ROW(OLD.lease_id,OLD.lease_job_id,OLD.lease_owner,OLD.lease_job_attempt,
                  OLD.lease_until,OLD.fence)
          AND (NEW.lease_job_attempt IS NULL OR NOT EXISTS (
            SELECT 1 FROM assistant_jobs j
            WHERE j.organization_id=NEW.organization_id
              AND j.orchestration_run_id=NEW.orchestration_run_id AND j.id=NEW.lease_job_id
              AND j.requester_membership_id=NEW.membership_id AND j.status='RUNNING'
              AND j.locked_by=NEW.lease_owner AND j.attempt_count=NEW.lease_job_attempt
              AND j.lease_until>=NEW.lease_until AND j.lease_until>clock_timestamp()
          ))
        THEN RAISE EXCEPTION 'ASSISTANT_JOB_CLAIM_INVALID'; END IF;
        RETURN NEW;
      END $$;
    """)
    op.execute("""
      CREATE FUNCTION guard_orchestration_interval_takeover() RETURNS trigger
      LANGUAGE plpgsql AS $$
      DECLARE state orchestration_runtime_states%ROWTYPE;
      BEGIN
        IF current_user <> 'app_runtime' THEN RETURN NEW; END IF;
        SELECT * INTO state FROM orchestration_runtime_states
          WHERE organization_id=OLD.organization_id
            AND orchestration_run_id=OLD.orchestration_run_id
          FOR UPDATE;
        IF NOT FOUND OR OLD.kind <> 'INTERVAL' OR OLD.closed OR OLD.job_owned
          OR OLD.allocation_id IS NOT NULL OR state.lease_id IS NULL
          OR state.lease_until <= clock_timestamp() OR NEW.fence <> state.fence
          OR NEW.fence <= OLD.fence
          OR OLD.started_at + (OLD.amount->>'active_milliseconds')::bigint *
               interval '1 millisecond' <= clock_timestamp()
          OR (to_jsonb(NEW)-'fence') IS DISTINCT FROM (to_jsonb(OLD)-'fence')
          OR NOT EXISTS (
              SELECT 1 FROM assistant_jobs j
              WHERE j.organization_id=state.organization_id
                AND j.orchestration_run_id=state.orchestration_run_id
                AND j.id=state.lease_job_id AND j.requester_membership_id=state.membership_id
                AND j.job_type IN ('assistant.turn.execute','assistant.turn.resume')
                AND j.status='RUNNING' AND j.locked_by=state.lease_owner
                AND j.attempt_count=state.lease_job_attempt
                AND j.attempt_count > 1 AND j.lease_until > clock_timestamp()
                AND OLD.operation_id=substring(
                  encode(
                    sha256(convert_to('assistant-root-interval:' || j.id::text, 'UTF8')), 'hex'
                  ),
                  1, 32)::uuid
          ) THEN
            RAISE EXCEPTION 'ACTIVE_INTERVAL_TAKEOVER_DENIED';
        END IF;
        RETURN NEW;
      END $$;
      DROP TRIGGER usage_guard ON orchestration_usage;
      CREATE TRIGGER usage_guard_insert BEFORE INSERT ON orchestration_usage
        FOR EACH ROW EXECUTE FUNCTION guard_orchestration_usage();
      CREATE TRIGGER usage_guard_update BEFORE UPDATE ON orchestration_usage
        FOR EACH ROW WHEN (OLD.fence IS NOT DISTINCT FROM NEW.fence)
        EXECUTE FUNCTION guard_orchestration_usage();
      CREATE TRIGGER usage_guard_interval_takeover BEFORE UPDATE ON orchestration_usage
        FOR EACH ROW WHEN (OLD.fence IS DISTINCT FROM NEW.fence)
        EXECUTE FUNCTION guard_orchestration_interval_takeover();
    """)


def downgrade() -> None:
    raise RuntimeError("Forward-only attempt fencing. Keep the additive schema on rollback.")
