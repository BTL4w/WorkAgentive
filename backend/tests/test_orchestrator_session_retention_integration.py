"""Private projection and worker fencing survive payload expiry, not raw copies."""

from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.modules.assistant.domain.session import SessionError
from app.modules.feedback.adapters.payload_repository import RetentionTransactions
from app.modules.feedback.application.retention_service import RetentionService
from tests.test_orchestrator_session_log_integration import Journal, pytestmark
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.runtime.session_log import EventDraft

__all__ = ["journal", "pytestmark"]


@pytest.mark.asyncio
async def test_expiry_tombstone_preserves_audit(journal: Journal) -> None:
    engine, sessions, _, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope,
        {"kind": "MESSAGE", "role": "user", "text": "private warehouse context"},
        classification="RAW_CONTEXT",
        kind="OBJECTIVE",
        ttl_seconds=1,
        lease=lease,
    )
    await recorder.append(
        scope,
        EventDraft(event_id=uuid4(), kind="turn.started", payload_ref=ref),
        expected_sequence=0,
        lease=lease,
    )
    async with engine.connect() as conn:
        before = await conn.scalar(
            text("SELECT count(*) FROM audit_events WHERE organization_id=:org"),
            {"org": scope.organization_id},
        )
    result = await RetentionService(RetentionTransactions(sessions)).purge_once(
        organization_id=scope.organization_id, now=datetime.now(UTC) + timedelta(seconds=2)
    )
    assert result.purged == 1
    with pytest.raises(SessionError, match="CONTEXT_EXPIRED"):
        await recorder.get_payload(scope, ref)
    records = await recorder.read(scope, after_sequence=0)
    assert len(records) == 1 and records[0].payload_ref == ref
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT body FROM orchestration_session_payloads WHERE id=:id"),
                {"id": ref.payload_id},
            )
            is None
        )
        assert (
            await conn.scalar(
                text("SELECT count(*) FROM audit_events WHERE organization_id=:org"),
                {"org": scope.organization_id},
            )
            == before + 1
        )
        assert (
            await conn.scalar(
                text(
                    "SELECT count(*) FROM outbox_events WHERE organization_id=:org AND "
                    "event_type='ai.retention.completed.v1'"
                ),
                {"org": scope.organization_id},
            )
            == 1
        )


@pytest.mark.asyncio
async def test_loaded_worker_cannot_write_after_individual_payload_purge(journal: Journal) -> None:
    _, sessions, _, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope,
        {"raw": "loaded before purge"},
        classification="RAW_CONTEXT",
        kind="CONTEXT",
        ttl_seconds=1,
        lease=lease,
    )
    loaded = await recorder.get_payload(scope, ref)
    await RetentionService(RetentionTransactions(sessions)).purge_once(
        organization_id=scope.organization_id, now=datetime.now(UTC) + timedelta(seconds=2)
    )
    with pytest.raises(SessionError, match="LEASE_LOST"):
        await recorder.put_payload(
            scope, loaded, classification="RAW_CONTEXT", kind="TOOL_RESULT", lease=lease
        )
    with pytest.raises(SessionError, match="LEASE_LOST"):
        await recorder.append(
            scope,
            EventDraft(event_id=uuid4(), kind="model.completed"),
            expected_sequence=0,
            lease=lease,
        )


@pytest.mark.asyncio
async def test_raw_payload_cannot_be_relabelled_or_exceed_30_days(journal: Journal) -> None:
    _, _, _, recorder, scope, lease, _, _ = journal
    with pytest.raises(ValueError):
        await recorder.put_payload(
            scope,
            {},
            classification="RAW_CONTEXT",
            kind="CONTEXT",
            ttl_seconds=31 * 86400,
            lease=lease,
        )
    # Redacted trace ceiling is tested separately by the existing retention policy
    # suite; session model/tool payload storage intentionally accepts RAW only.
    with pytest.raises(ValueError):
        await recorder.put_payload(
            scope,
            {},
            classification=cast(Literal["RAW_CONTEXT"], "REDACTED_TRACE"),
            kind="CONTEXT",
            lease=lease,
        )


@pytest.mark.asyncio
async def test_purge_fences_loaded_budget_receipt(journal: Journal) -> None:
    from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
    from work_management_ai.model_gateway.contracts import ModelUsage
    from work_management_ai.runtime.run_budget import UsageReservation

    _, sessions, actor, recorder, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await store.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease)
    receipt = await store.reserve(
        scope,
        UsageReservation(
            kind="MODEL",
            interval_id=active.reservation.operation_id,
            input_tokens=100,
            output_tokens=20,
        ),
        lease=lease,
    )
    await recorder.put_payload(
        scope,
        {"raw": "loaded"},
        classification="RAW_CONTEXT",
        kind="CONTEXT",
        ttl_seconds=1,
        lease=lease,
    )
    await RetentionService(RetentionTransactions(sessions)).purge_once(
        organization_id=scope.organization_id, now=datetime.now(UTC) + timedelta(seconds=2)
    )
    with pytest.raises(SessionError, match="LEASE_LOST"):
        await store.settle(scope, receipt, ModelUsage(input_tokens=10, output_tokens=1))
    assert (await store.remaining(scope)).usage.input_tokens == 100


@pytest.mark.asyncio
async def test_retention_role_cannot_fence_live_or_other_tenant_run(journal: Journal) -> None:
    from sqlalchemy.exc import DBAPIError

    _, sessions, _, _, scope, _, _, _ = journal
    transactions = RetentionTransactions(sessions)
    with pytest.raises(DBAPIError, match="RETENTION_FENCE_LOST"):
        async with transactions(scope.organization_id) as repo:
            await repo.session.execute(
                text(
                    "UPDATE orchestration_runtime_states SET "
                    "fence=fence+1,lease_id=NULL,lease_job_id=NULL,lease_owner=NULL,lease_until=NULL"
                )
            )
    async with transactions(uuid4()) as repo:
        assert (
            await repo.session.scalar(text("SELECT count(*) FROM orchestration_runtime_states"))
            == 0
        )
        result = await repo.session.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "fence=fence+1,lease_id=NULL,lease_job_id=NULL,lease_owner=NULL,lease_until=NULL"
            )
        )
        assert result.rowcount == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cleanup_skips_active_fenced_transaction(journal: Journal) -> None:
    from app.modules.assistant.adapters.session_usage_repository import (
        PostgreSQLUsageTransactionFactory,
    )
    from app.modules.assistant.adapters.tool_loop_recorder import session_scope

    _, sessions, actor, recorder, scope, lease, _, _ = journal
    await recorder.put_payload(
        scope, {}, classification="RAW_CONTEXT", kind="CONTEXT", ttl_seconds=1, lease=lease
    )
    cleaner = RetentionService(RetentionTransactions(sessions))
    now = datetime.now(UTC) + timedelta(seconds=2)
    async with PostgreSQLUsageTransactionFactory(sessions)(actor) as tx:
        assert await tx.repository.state(session_scope(scope), lock=True) is not None
        assert (
            await cleaner.purge_once(organization_id=scope.organization_id, now=now)
        ).purged == 0
    assert (await cleaner.purge_once(organization_id=scope.organization_id, now=now)).purged == 1


@pytest.mark.asyncio
async def test_retention_role_cannot_fence_again_from_old_tombstone(journal: Journal) -> None:
    from sqlalchemy.exc import DBAPIError

    _, sessions, _, recorder, scope, lease, _, _ = journal
    await recorder.put_payload(
        scope, {}, classification="RAW_CONTEXT", kind="CONTEXT", ttl_seconds=1, lease=lease
    )
    now = datetime.now(UTC) + timedelta(seconds=2)
    transactions = RetentionTransactions(sessions)
    await RetentionService(transactions).purge_once(organization_id=scope.organization_id, now=now)
    with pytest.raises(DBAPIError, match="RETENTION_FENCE_LOST"):
        async with transactions(scope.organization_id) as repo:
            await repo.session.execute(
                text("SELECT set_config('app.retention_now',:now,true)"), {"now": now.isoformat()}
            )
            await repo.session.execute(
                text(
                    "UPDATE orchestration_runtime_states SET fence=fence+1,"
                    "lease_id=NULL,lease_job_id=NULL,lease_owner=NULL,lease_until=NULL"
                )
            )
