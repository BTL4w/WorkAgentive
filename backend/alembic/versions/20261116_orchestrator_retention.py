"""Fence loaded v2 workers when private context is purged; retain safe headers.

Revision ID: 20261116_orch_retention
Revises: 20261116_orch_usage
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261116_orch_retention"
down_revision: str | Sequence[str] | None = "20261116_orch_usage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    tenant = "organization_id=NULLIF(current_setting('app.organization_id',true),'')::uuid"
    for table in ("orchestration_runtime_states", "orchestration_session_events"):
        op.execute(f"GRANT SELECT ON {table} TO app_retention")
        op.execute(
            f"CREATE POLICY session_retention_headers ON {table} FOR SELECT "
            f"TO app_retention USING ({tenant})"
        )
    op.execute(
        "GRANT UPDATE (lease_id,lease_job_id,lease_owner,lease_until,fence) "
        "ON orchestration_runtime_states TO app_retention"
    )
    op.execute(
        "CREATE POLICY session_retention_fence ON orchestration_runtime_states "
        f"FOR UPDATE TO app_retention USING ({tenant}) WITH CHECK ({tenant})"
    )
    op.execute("""
      CREATE FUNCTION guard_session_retention_fence() RETURNS trigger
      LANGUAGE plpgsql SET search_path=pg_catalog,public AS $$
      BEGIN
        IF current_user='app_retention' THEN
          IF (to_jsonb(NEW)-ARRAY['lease_id','lease_job_id','lease_owner','lease_until','fence'])
             IS DISTINCT FROM
             (to_jsonb(OLD)-ARRAY['lease_id','lease_job_id','lease_owner','lease_until','fence'])
             OR NEW.fence<>OLD.fence+1 OR NEW.lease_id IS NOT NULL
             OR NEW.lease_job_id IS NOT NULL OR NEW.lease_owner IS NOT NULL
             OR NEW.lease_until IS NOT NULL OR NOT EXISTS(
                SELECT 1 FROM ai_retention_payloads p
                WHERE p.organization_id=OLD.organization_id AND p.purged_at IS NULL AND
                ((p.storage_kind='orchestration_runs' AND p.owner_id=OLD.orchestration_run_id)
                 OR (p.storage_kind='orchestration_session_payloads'
                     AND p.run_id=OLD.orchestration_run_id))
                AND LEAST(p.expires_at,p.created_at+make_interval(days=>LEAST(30,
                  COALESCE(nullif(current_setting('app.ai_raw_retention_days',true),'')::int,30))))
                  <=COALESCE(nullif(current_setting('app.retention_now',true),'')::timestamptz,
                             clock_timestamp())
             ) THEN RAISE EXCEPTION 'RETENTION_FENCE_LOST'; END IF;
        END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER session_retention_fence_guard BEFORE UPDATE ON orchestration_runtime_states
        FOR EACH ROW EXECUTE FUNCTION guard_session_retention_fence();
    """)


def downgrade() -> None:
    raise RuntimeError("Keep additive retention fencing on rollback; disable candidate admission")
