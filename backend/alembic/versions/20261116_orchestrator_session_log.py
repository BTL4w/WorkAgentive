"""orchestrator_session_log

Revision ID: 20261116_orch_session
Revises: f9128a14d001
Create Date: 2026-11-16 16:43:52.826198
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261116_orch_session"
down_revision: str | Sequence[str] | None = "f9128a14d001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""
    op.create_unique_constraint(
        "uq_session_job_scope", "assistant_jobs", ["organization_id", "id", "orchestration_run_id"]
    )
    op.create_unique_constraint(
        "uq_session_turn_scope",
        "assistant_turns",
        ["organization_id", "id", "conversation_id", "actor_membership_id"],
    )
    op.create_unique_constraint(
        "uq_session_run_scope",
        "orchestration_runs",
        ["organization_id", "id", "turn_id", "actor_membership_id"],
    )
    op.create_table(
        "orchestration_runtime_states",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_run_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("membership_id", sa.Uuid(), nullable=False),
        sa.Column("driver", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("driver_version", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("catalog_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("agent_version", sa.String(length=64), nullable=False),
        sa.Column("manifest_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("model_ref", sa.String(length=200), nullable=True),
        sa.Column("sequence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("checkpoint_sequence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("checkpoint_payload_id", sa.Uuid(), nullable=True),
        sa.Column("lease_id", sa.Uuid(), nullable=True),
        sa.Column("lease_job_id", sa.Uuid(), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fence", sa.Integer(), server_default="0", nullable=False),
        sa.CheckConstraint(
            "driver='tool_loop' AND schema_version='2.0'",
            name=op.f("ck_orchestration_runtime_states_driver"),
        ),
        sa.CheckConstraint(
            (
                "(lease_id IS NULL AND lease_job_id IS NULL AND lease_owner IS NULL "
                "AND lease_until IS NULL) OR (lease_id IS NOT NULL AND lease_job_id "
                "IS NOT NULL AND lease_owner IS NOT NULL AND lease_until IS NOT NULL)"
            ),
            name=op.f("ck_orchestration_runtime_states_lease_shape"),
        ),
        sa.CheckConstraint(
            "sequence>=0 AND checkpoint_sequence>=0 AND checkpoint_sequence<=sequence AND fence>=0",
            name=op.f("ck_orchestration_runtime_states_cursor"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "lease_job_id", "orchestration_run_id"],
            [
                "assistant_jobs.organization_id",
                "assistant_jobs.id",
                "assistant_jobs.orchestration_run_id",
            ],
            name="fk_session_state_job_scope",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
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
        sa.ForeignKeyConstraint(
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
        sa.PrimaryKeyConstraint(
            "organization_id", "orchestration_run_id", name=op.f("pk_orchestration_runtime_states")
        ),
    )
    op.create_table(
        "orchestration_session_payloads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_run_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("classification", sa.String(length=24), nullable=False),
        sa.Column(
            "body", postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True
        ),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            (
                "classification='RAW_CONTEXT' AND expires_at>created_at AND "
                "expires_at<=created_at+interval '30 days'"
            ),
            name=op.f("ck_orchestration_session_payloads_retention"),
        ),
        sa.CheckConstraint(
            "fingerprint ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_orchestration_session_payloads_fingerprint"),
        ),
        sa.CheckConstraint(
            (
                "kind IN ('OBJECTIVE','CONTEXT','TOOL_ARGUMENTS','TOOL_RESULT',"
                "'MODEL_OUTPUT','SKILL_BODY','WORKING_PLAN','CHECKPOINT')"
            ),
            name=op.f("ck_orchestration_session_payloads_kind"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_session_payload_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "organization_id", "id", name=op.f("pk_orchestration_session_payloads")
        ),
        sa.UniqueConstraint(
            "organization_id",
            "orchestration_run_id",
            "id",
            name=op.f("uq_orchestration_session_payloads_organization_id_orchestration_run_id_id"),
        ),
    )
    op.create_index(
        "ix_session_payload_expiry",
        "orchestration_session_payloads",
        ["organization_id", "expires_at", "id"],
        unique=False,
    )
    op.create_table(
        "orchestration_session_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("orchestration_run_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=16), server_default="1.0", nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("payload_id", sa.Uuid(), nullable=True),
        sa.Column("invocation_id", sa.Uuid(), nullable=True),
        sa.Column("call_id", sa.String(length=128), nullable=True),
        sa.Column("tool_reference", sa.String(length=100), nullable=True),
        sa.Column("tool_version", sa.String(length=64), nullable=True),
        sa.Column("arguments_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("settlement_status", sa.String(length=16), nullable=True),
        sa.Column("wait_id", sa.Uuid(), nullable=True),
        sa.Column("wait_kind", sa.String(length=32), nullable=True),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("resource_version", sa.Integer(), nullable=True),
        sa.Column("policy_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("source_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("safe_code", sa.String(length=100), nullable=True),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column("causation_id", sa.Uuid(), nullable=True),
        sa.Column("correlation_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            (
                "kind IN ('turn.started','context.selected','model.requested',"
                "'model.completed','model.failed','tool.requested','tool.authorized',"
                "'tool.denied','tool.started','tool.completed','tool.failed',"
                "'tool.observed','run.waiting','continuation.received','run.resumed',"
                "'turn.completed','run.cancelled','run.failed')"
            ),
            name=op.f("ck_orchestration_session_events_kind"),
        ),
        sa.CheckConstraint(
            "safe_code IS NULL OR safe_code ~ '^[A-Z][A-Z0-9_]{0,99}$'",
            name=op.f("ck_orchestration_session_events_safe_code"),
        ),
        sa.CheckConstraint(
            "schema_version='1.0' AND fingerprint ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_orchestration_session_events_envelope"),
        ),
        sa.CheckConstraint(
            (
                "settlement_status IS NULL OR settlement_status IN ('COMPLETED',"
                "'DEFERRED','REJECTED','FAILED')"
            ),
            name=op.f("ck_orchestration_session_events_settlement"),
        ),
        sa.CheckConstraint("sequence>=1", name=op.f("ck_orchestration_session_events_sequence")),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_run_id"],
            ["agent_runs.organization_id", "agent_runs.id"],
            name="fk_session_event_agent",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id", "payload_id"],
            [
                "orchestration_session_payloads.organization_id",
                "orchestration_session_payloads.orchestration_run_id",
                "orchestration_session_payloads.id",
            ],
            name="fk_session_event_payload",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "orchestration_run_id"],
            [
                "orchestration_runtime_states.organization_id",
                "orchestration_runtime_states.orchestration_run_id",
            ],
            name="fk_session_event_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "organization_id", "event_id", name=op.f("pk_orchestration_session_events")
        ),
        sa.UniqueConstraint(
            "organization_id",
            "orchestration_run_id",
            "event_id",
            name=op.f(
                "uq_orchestration_session_events_organization_id_orchestration_run_id_event_id"
            ),
        ),
        sa.UniqueConstraint(
            "organization_id",
            "orchestration_run_id",
            "sequence",
            name=op.f(
                "uq_orchestration_session_events_organization_id_orchestration_run_id_sequence"
            ),
        ),
    )
    op.create_index(
        "ix_session_event_invocation",
        "orchestration_session_events",
        ["organization_id", "orchestration_run_id", "invocation_id", "sequence"],
        unique=False,
    )
    op.create_index(
        "uq_session_call",
        "orchestration_session_events",
        ["organization_id", "orchestration_run_id", "call_id"],
        unique=True,
        postgresql_where=sa.text("kind='tool.requested'"),
    )
    op.create_index(
        "uq_session_invocation_request",
        "orchestration_session_events",
        ["organization_id", "orchestration_run_id", "invocation_id"],
        unique=True,
        postgresql_where=sa.text("kind='tool.requested'"),
    )
    op.create_index(
        "uq_session_invocation_settle",
        "orchestration_session_events",
        ["organization_id", "orchestration_run_id", "invocation_id"],
        unique=True,
        postgresql_where=sa.text("settlement_status IS NOT NULL"),
    )
    op.create_foreign_key(
        "fk_session_checkpoint_payload",
        "orchestration_runtime_states",
        "orchestration_session_payloads",
        ["organization_id", "orchestration_run_id", "checkpoint_payload_id"],
        ["organization_id", "orchestration_run_id", "id"],
        deferrable=True,
        initially="DEFERRED",
    )
    _security()


def _security() -> None:
    owner = """organization_id=NULLIF(current_setting('app.organization_id',true),'')::uuid
      AND EXISTS(SELECT 1 FROM orchestration_runs r
        JOIN assistant_turns t ON t.organization_id=r.organization_id AND t.id=r.turn_id
        JOIN assistant_conversations c ON c.organization_id=t.organization_id
          AND c.id=t.conversation_id
        JOIN memberships m ON m.organization_id=r.organization_id AND m.id=r.actor_membership_id
        JOIN users u ON u.id=m.user_id
        WHERE r.organization_id={table}.organization_id AND r.id={table}.orchestration_run_id
          AND c.owner_membership_id=NULLIF(current_setting('app.membership_id',true),'')::uuid
          AND t.actor_membership_id=c.owner_membership_id
          AND r.actor_membership_id=c.owner_membership_id
          AND m.is_active AND u.is_active)"""
    for table in (
        "orchestration_runtime_states",
        "orchestration_session_payloads",
        "orchestration_session_events",
    ):
        op.execute(f"ALTER TABLE {table} OWNER TO migration_owner")
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        policy = owner.format(table=table)
        op.execute(
            f"CREATE POLICY private_owner ON {table} TO app_runtime "
            f"USING ({policy}) WITH CHECK ({policy})"
        )
        op.execute(f"GRANT SELECT,INSERT ON {table} TO app_runtime")
        op.execute(f"REVOKE UPDATE,DELETE ON {table} FROM app_runtime")
    op.execute(
        "GRANT UPDATE (sequence,checkpoint_sequence,checkpoint_payload_id,"
        "lease_id,lease_job_id,lease_owner,lease_until,fence) ON "
        "orchestration_runtime_states TO app_runtime"
    )
    op.execute("GRANT SELECT,UPDATE(body) ON orchestration_session_payloads TO app_retention")
    op.execute(
        "CREATE POLICY session_payload_retention ON "
        "orchestration_session_payloads TO app_retention USING "
        "(organization_id=NULLIF(current_setting('app.organization_id',true),"
        "'')::uuid) WITH CHECK "
        "(organization_id=NULLIF(current_setting('app.organization_id',true),"
        "'')::uuid)"
    )
    op.execute("""CREATE FUNCTION protect_session_event() RETURNS trigger LANGUAGE plpgsql
    SET search_path=pg_catalog,public AS $$
    BEGIN RAISE EXCEPTION 'session event headers are append-only'; END $$""")
    op.execute(
        "CREATE TRIGGER session_event_immutable BEFORE UPDATE OR DELETE ON "
        "orchestration_session_events FOR EACH ROW EXECUTE FUNCTION "
        "protect_session_event()"
    )
    op.execute("""CREATE FUNCTION protect_session_state() RETURNS trigger LANGUAGE plpgsql
    SET search_path=pg_catalog,public AS $$
    BEGIN
      IF (to_jsonb(NEW)-ARRAY['sequence','checkpoint_sequence','checkpoint_payload_id',
             'lease_id','lease_job_id','lease_owner','lease_until','fence']) IS DISTINCT FROM
         (to_jsonb(OLD)-ARRAY['sequence','checkpoint_sequence','checkpoint_payload_id',
             'lease_id','lease_job_id','lease_owner','lease_until','fence']) OR NEW.fence<OLD.fence
      THEN RAISE EXCEPTION 'runtime pins are immutable'; END IF;
      IF NEW.sequence<>(SELECT COALESCE(max(sequence),0) FROM orchestration_session_events
        WHERE organization_id=NEW.organization_id AND orchestration_run_id=NEW.orchestration_run_id)
      THEN RAISE EXCEPTION 'runtime cursor must match journal'; END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER session_state_immutable BEFORE UPDATE ON "
        "orchestration_runtime_states FOR EACH ROW EXECUTE FUNCTION "
        "protect_session_state()"
    )
    op.execute("""CREATE FUNCTION register_session_payload() RETURNS trigger LANGUAGE plpgsql
    SET search_path=pg_catalog,public AS $$
    DECLARE parent ai_retention_payloads; deadline timestamptz;
    BEGIN
      SELECT * INTO parent FROM ai_retention_payloads WHERE organization_id=NEW.organization_id
        AND storage_kind='orchestration_runs' AND owner_id=NEW.orchestration_run_id;
      deadline:=LEAST(parent.expires_at,parent.created_at+make_interval(days=>LEAST(30,COALESCE(nullif(current_setting('app.ai_raw_retention_days',true),'')::int,30))));
      IF parent.owner_id IS NULL OR parent.purged_at IS NOT NULL
         OR NEW.classification<>'RAW_CONTEXT'
         OR NEW.expires_at>deadline OR NEW.expires_at<=clock_timestamp()
      THEN RAISE EXCEPTION 'CONTEXT_EXPIRED'; END IF;
      INSERT INTO ai_retention_payloads(organization_id,storage_kind,owner_id,
        classification,created_at,expires_at,run_id)
        VALUES(NEW.organization_id,TG_TABLE_NAME,NEW.id,'RAW_CONTEXT',parent.created_at,NEW.expires_at,NEW.orchestration_run_id);
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER register_session_payload_retention AFTER INSERT ON "
        "orchestration_session_payloads FOR EACH ROW EXECUTE FUNCTION "
        "register_session_payload()"
    )
    op.execute(
        "CREATE TRIGGER a_guard_session_payload_retention BEFORE UPDATE ON "
        "orchestration_session_payloads FOR EACH ROW EXECUTE FUNCTION "
        "guard_ai_payload('body','{\"body\": null}')"
    )
    op.execute("""CREATE FUNCTION protect_session_payload() RETURNS trigger LANGUAGE plpgsql
    SET search_path=pg_catalog,public AS $$ BEGIN
      IF TG_OP='UPDATE' AND current_user='app_retention' AND NEW.body IS NULL AND
        (to_jsonb(NEW)-'body') IS NOT DISTINCT FROM (to_jsonb(OLD)-'body')
      THEN RETURN NEW; END IF;
      RAISE EXCEPTION 'session payload immutable except retention scrub';
    END $$""")
    op.execute(
        "CREATE TRIGGER session_payload_immutable BEFORE UPDATE OR DELETE ON "
        "orchestration_session_payloads FOR EACH ROW EXECUTE FUNCTION "
        "protect_session_payload()"
    )


def downgrade() -> None:
    raise RuntimeError(
        "Disable candidate admission for rollback; keep additive journal "
        "schema. Destructive removal requires a separately reviewed "
        "migration."
    )
