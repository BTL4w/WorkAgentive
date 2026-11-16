"""Real PostgreSQL private journal tests, including runtime-role RLS and fencing."""

import asyncio
import os
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.database import create_database_engine, create_session_factory
from app.modules.assistant.adapters.session_repository import PostgreSQLSessionTransactionFactory
from app.modules.assistant.adapters.tool_loop_recorder import PostgreSQLToolLoopRecorder
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.session_service import SessionService
from app.modules.assistant.domain.models import (
    AssistantConversation,
    AssistantEvent,
    AssistantJob,
    AssistantMessage,
    AssistantTurn,
    MessageRole,
    OrchestrationRun,
)
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.domain.roles import MembershipRole
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationDescriptor,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import RunLease, RunScope, ToolLoopCheckpoint

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


async def seed_members(connection: AsyncConnection, pairs: tuple[tuple[UUID, UUID], ...]) -> None:
    for organization_id, membership_id in pairs:
        user_id = uuid4()
        await connection.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, 'Assistant')"),
            {"id": organization_id, "slug": f"assistant-{organization_id.hex}"},
        )
        await connection.execute(
            text(
                "INSERT INTO users "
                "(id, email_normalized, email_display, display_name, password_hash) "
                "VALUES (:id, :email, :email, 'Assistant', 'hash')"
            ),
            {"id": user_id, "email": f"{user_id.hex}@example.test"},
        )
        await connection.execute(
            text(
                "INSERT INTO memberships (id, organization_id, user_id, role) "
                "VALUES (:id, :org, :user, 'MANAGER')"
            ),
            {"id": membership_id, "org": organization_id, "user": user_id},
        )


def turn_graph(
    *, conversation: AssistantConversation, actor: AuthenticatedActor
) -> tuple[AssistantMessage, AssistantTurn, OrchestrationRun, AssistantJob, AssistantEvent]:
    message_id, turn_id, run_id = uuid4(), uuid4(), uuid4()
    now = datetime.now(UTC)
    message = AssistantMessage(
        id=message_id,
        organization_id=actor.organization_id,
        conversation_id=conversation.id,
        sequence=1,
        role=MessageRole.USER,
        content_blocks=({"kind": "text", "text": "Plan a customer event"},),
        created_by_membership_id=actor.membership_id,
        turn_id=turn_id,
        dedupe_key="message-1",
        created_at=now,
    )
    turn = AssistantTurn.create(
        id=turn_id,
        organization_id=actor.organization_id,
        conversation_id=conversation.id,
        user_message_id=message_id,
        actor_membership_id=actor.membership_id,
        objective="Plan a customer event",
        locale="en",
        now=now,
    )
    run = OrchestrationRun.create(
        id=run_id,
        organization_id=actor.organization_id,
        turn_id=turn_id,
        orchestrator_version="1.0.0",
        orchestrator_fingerprint="manifest-sha",
        execution_plan={"steps": []},
        budget={"max_iterations": 8},
        now=now,
    )
    job = AssistantJob.create(
        organization_id=actor.organization_id,
        conversation_id=conversation.id,
        turn_id=turn_id,
        orchestration_run_id=run_id,
        requester_membership_id=actor.membership_id,
        payload={"turn_id": str(turn_id)},
        now=now,
    )
    event = AssistantEvent(
        id=uuid4(),
        organization_id=actor.organization_id,
        conversation_id=conversation.id,
        sequence=1,
        event_type="assistant.turn.queued.v1",
        public_payload={"turn_id": str(turn_id), "status": "QUEUED"},
        turn_id=turn_id,
        orchestration_run_id=run_id,
        dedupe_key="turn-queued-1",
        occurred_at=now,
    )
    return message, turn, run, job, event


type Journal = tuple[
    AsyncEngine,
    async_sessionmaker[AsyncSession],
    AuthenticatedActor,
    PostgreSQLToolLoopRecorder,
    RunScope,
    RunLease,
    ToolLoopCheckpoint,
    UUID,
]


@pytest_asyncio.fixture
async def journal() -> AsyncGenerator[Journal]:
    engine = create_database_engine(Settings(environment="test"))
    sessions = create_session_factory(engine)
    org, member = uuid4(), uuid4()
    async with engine.begin() as connection:
        await seed_members(connection, ((org, member),))
        user = await connection.scalar(
            text("SELECT user_id FROM memberships WHERE id=:id"), {"id": member}
        )
    assert isinstance(user, UUID)
    actor = AuthenticatedActor(
        user_id=user,
        email="session@example.test",
        display_name="Minh",
        membership_id=member,
        organization_id=org,
        organization_name="Synthetic",
        role=MembershipRole.MANAGER,
    )
    factory = PostgreSQLAssistantTransactionFactory(sessions)
    conversation = AssistantConversation.create(
        organization_id=org, owner_membership_id=member, locale="vi"
    )
    async with factory(actor) as tx:
        await tx.repository.create_conversation_mutation(
            actor=actor,
            conversation=conversation,
            request_id="create",
            idempotency_key="create",
            request_fingerprint="create",
        )
        await tx.commit()
    message, turn, run, job, event = turn_graph(conversation=conversation, actor=actor)
    async with factory(actor) as tx:
        await tx.repository.submit_message_mutation(
            actor=actor,
            message=message,
            turn=turn,
            run=run,
            job=job,
            event=event,
            request_id="turn",
            idempotency_key="turn",
            request_fingerprint="turn",
        )
        await tx.commit()
    scope = RunScope(
        organization_id=org,
        membership_id=member,
        conversation_id=conversation.id,
        turn_id=turn.id,
        orchestration_run_id=run.id,
    )
    service = SessionService(PostgreSQLSessionTransactionFactory(sessions))
    recorder = PostgreSQLToolLoopRecorder(service=service, actor=actor)
    checkpoint = ToolLoopCheckpoint(
        scope=scope,
        driver_version="2.0.0",
        prompt_version="tool-loop.v1",
        catalog_fingerprint="a" * 64,
    )
    assert await recorder.load_checkpoint(scope) is None  # legacy run unchanged until explicit init
    await recorder.initialize(scope, checkpoint)
    lease = await recorder.claim(scope, job_id=job.id, worker_id="worker-a", lease_seconds=60)
    try:
        yield engine, sessions, actor, recorder, scope, lease, checkpoint, job.id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sequence_conflict_and_dedupe(journal: Journal) -> None:
    _, _, _, recorder, scope, lease, _, _ = journal
    event = EventDraft(event_id=uuid4(), kind="turn.started")
    first = await recorder.append(scope, event, expected_sequence=0, lease=lease)
    assert first.sequence == 1
    assert await recorder.append(scope, event, expected_sequence=0, lease=lease) == first
    with pytest.raises(SessionError, match="SEQUENCE_CONFLICT"):
        await recorder.append(
            scope,
            EventDraft(event_id=uuid4(), kind="model.requested"),
            expected_sequence=0,
            lease=lease,
        )
    with pytest.raises(SessionError, match="EVENT_ID_CONFLICT"):
        await recorder.append(
            scope, event.model_copy(update={"kind": "run.failed"}), expected_sequence=1, lease=lease
        )
    assert len(await recorder.read(scope, after_sequence=0)) == 1


@pytest.mark.asyncio
async def test_headers_append_only(journal: Journal) -> None:
    _, sessions, actor, recorder, scope, lease, _, _ = journal
    await recorder.append(
        scope, EventDraft(event_id=uuid4(), kind="turn.started"), expected_sequence=0, lease=lease
    )
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as tx:
        role = await tx.session.execute(
            text("SELECT current_user,rolbypassrls FROM pg_roles WHERE rolname=current_user")
        )
        assert role.one() == ("app_runtime", False)
        for operation in (
            "UPDATE orchestration_session_events SET kind='run.failed'",
            "DELETE FROM orchestration_session_events",
        ):
            with pytest.raises(DBAPIError):
                async with tx.session.begin_nested():
                    await tx.session.execute(text(operation))


@pytest.mark.asyncio
async def test_cross_tenant_event_payload_and_fk_denied(journal: Journal) -> None:
    engine, sessions, actor, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope,
        {"goal": "private training goal"},
        classification="RAW_CONTEXT",
        kind="CONTEXT",
        lease=lease,
    )
    await recorder.append(
        scope,
        EventDraft(event_id=uuid4(), kind="context.selected", payload_ref=ref),
        expected_sequence=0,
        lease=lease,
    )
    org, member = uuid4(), uuid4()
    async with engine.begin() as connection:
        await seed_members(connection, ((org, member),))
        user = await connection.scalar(
            text("SELECT user_id FROM memberships WHERE id=:id"), {"id": member}
        )
    assert isinstance(user, UUID)
    other = replace(actor, user_id=user, organization_id=org, membership_id=member)
    async with PostgreSQLAssistantTransactionFactory(sessions)(other) as tx:
        for table in (
            "orchestration_session_events",
            "orchestration_session_payloads",
            "orchestration_runtime_states",
        ):
            assert await tx.session.scalar(text(f"SELECT count(*) FROM {table}")) == 0
    foreign = PostgreSQLToolLoopRecorder(service=recorder.service, actor=other)
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await foreign.get_payload(scope, ref)
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await foreign.read(scope, after_sequence=0)
    async with engine.connect() as connection:
        with pytest.raises(DBAPIError):
            await connection.execute(
                text(
                    "INSERT INTO orchestration_session_payloads (id,organization_id,"
                    "orchestration_run_id,kind,classification,body,fingerprint,"
                    "created_at,expires_at) VALUES (:id,:org,:run,'CONTEXT',"
                    "'RAW_CONTEXT','{}',:fp,now(),now()+interval '1 day')"
                ),
                {"id": uuid4(), "org": org, "run": scope.orchestration_run_id, "fp": "a" * 64},
            )
        await connection.rollback()


@pytest.mark.asyncio
async def test_private_log_not_public_sse(journal: Journal) -> None:
    engine, _, _, recorder, scope, lease, _, _ = journal
    async with engine.connect() as conn:
        before = await conn.scalar(
            text("SELECT count(*) FROM assistant_events WHERE orchestration_run_id=:run"),
            {"run": scope.orchestration_run_id},
        )
    payload = await recorder.put_payload(
        scope,
        {"text": "private-context-marker"},
        classification="RAW_CONTEXT",
        kind="CONTEXT",
        lease=lease,
    )
    await recorder.append(
        scope,
        EventDraft(event_id=uuid4(), kind="context.selected", payload_ref=payload),
        expected_sequence=0,
        lease=lease,
    )
    async with engine.connect() as conn:
        after = await conn.scalar(
            text("SELECT count(*) FROM assistant_events WHERE orchestration_run_id=:run"),
            {"run": scope.orchestration_run_id},
        )
        assert after == before
        header = await conn.scalar(
            text(
                "SELECT row_to_json(e)::text FROM orchestration_session_events e "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
        assert "private-context-marker" not in header


@pytest.mark.asyncio
async def test_driver_pin_survives_flag_change(journal: Journal) -> None:
    _, _, actor, recorder, scope, lease, checkpoint, _ = journal
    record = await recorder.append(
        scope, EventDraft(event_id=uuid4(), kind="turn.started"), expected_sequence=0, lease=lease
    )
    updated = checkpoint.model_copy(update={"sequence": record.sequence})
    await recorder.checkpoint(scope, updated, lease=lease)
    restarted = PostgreSQLToolLoopRecorder(service=recorder.service, actor=actor)
    assert await restarted.load_checkpoint(scope) == updated
    with pytest.raises(SessionError, match="PIN_CONFLICT"):
        await restarted.initialize(scope, checkpoint.model_copy(update={"driver_version": "3.0.0"}))
    restored = await restarted.load_checkpoint(scope)
    assert restored is not None and restored.driver_version == "2.0.0"


@pytest.mark.asyncio
async def test_settlement_replays_without_duplicate_and_updates_cursor(journal: Journal) -> None:
    _, _, _, recorder, scope, lease, checkpoint, _ = journal
    invocation = uuid4()
    args = await recorder.put_payload(
        scope,
        {"project_ref": "project-a"},
        classification="RAW_CONTEXT",
        kind="TOOL_ARGUMENTS",
        lease=lease,
    )
    descriptor = InvocationDescriptor(
        invocation_id=invocation,
        call_id="c1",
        tool_reference="read_project_context@1",
        tool_version="1.0.0",
        arguments_fingerprint=args.fingerprint,
    )
    await recorder.append(
        scope,
        EventDraft(
            event_id=uuid4(), kind="tool.requested", invocation=descriptor, payload_ref=args
        ),
        expected_sequence=0,
        lease=lease,
    )
    receipt = await recorder.get_invocation(scope, invocation)
    assert receipt is not None and receipt.settlement is None
    result = await recorder.put_payload(
        scope,
        {"goal": "20 employees"},
        classification="RAW_CONTEXT",
        kind="TOOL_RESULT",
        lease=lease,
    )
    value = SettledInvocation(status="COMPLETED", payload_ref=result)
    settled = await recorder.settle_invocation(scope, receipt, value, lease=lease)
    assert settled.sequence == 2
    assert await recorder.settle_invocation(scope, receipt, value, lease=lease) == settled
    restored = await recorder.get_invocation(scope, invocation)
    assert restored is not None and restored.settlement == value
    with pytest.raises(SessionError, match="INVOCATION_CONFLICT"):
        await recorder.settle_invocation(
            scope, receipt, SettledInvocation(status="FAILED", safe_code="TIMEOUT"), lease=lease
        )
    await recorder.checkpoint(scope, checkpoint.model_copy(update={"sequence": 2}), lease=lease)
    restored_checkpoint = await recorder.load_checkpoint(scope)
    assert restored_checkpoint is not None and restored_checkpoint.sequence == 2


@pytest.mark.asyncio
async def test_active_lease_excludes_other_worker_and_stale_executor(journal: Journal) -> None:
    engine, _, _, recorder, scope, lease, _, job = journal
    with pytest.raises(SessionError, match="LEASE_BUSY"):
        await recorder.claim(scope, job_id=job, worker_id="worker-b", lease_seconds=60)
    renewed = await recorder.renew(scope, lease, lease_seconds=90)
    assert renewed.fencing_token == lease.fencing_token
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=now()-interval "
                "'1 second' WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
    takeover = await recorder.claim(scope, job_id=job, worker_id="worker-b", lease_seconds=60)
    assert takeover.fencing_token > lease.fencing_token
    with pytest.raises(SessionError, match="LEASE_LOST"):
        await recorder.append(
            scope,
            EventDraft(event_id=uuid4(), kind="turn.started"),
            expected_sequence=0,
            lease=lease,
        )


@pytest.mark.asyncio
async def test_concurrent_appends_have_one_sequence_winner(journal: Journal) -> None:
    _, _, _, recorder, scope, lease, _, _ = journal
    results = await asyncio.gather(
        *(
            recorder.append(
                scope,
                EventDraft(event_id=uuid4(), kind="model.requested"),
                expected_sequence=0,
                lease=lease,
            )
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(r, SessionError) for r in results) == 1
    assert len(await recorder.read(scope, after_sequence=0)) == 1


@pytest.mark.asyncio
async def test_expired_payload_not_replayed_and_retention_purges_only_body(
    journal: Journal,
) -> None:
    engine, sessions, _, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope,
        {"text": "private-expiring"},
        classification="RAW_CONTEXT",
        kind="CONTEXT",
        ttl_seconds=1,
        lease=lease,
    )
    await recorder.append(
        scope,
        EventDraft(event_id=uuid4(), kind="context.selected", payload_ref=ref),
        expected_sequence=0,
        lease=lease,
    )
    await asyncio.sleep(1.05)
    with pytest.raises(SessionError, match="CONTEXT_EXPIRED"):
        await recorder.get_payload(scope, ref)
    from app.modules.feedback.adapters.payload_repository import RetentionTransactions
    from app.modules.feedback.application.retention_service import RetentionService

    await RetentionService(RetentionTransactions(sessions)).purge_once(
        organization_id=scope.organization_id, now=datetime.now(UTC), limit=100
    )
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
                text(
                    "SELECT count(*) FROM orchestration_session_events WHERE "
                    "orchestration_run_id=:run"
                ),
                {"run": scope.orchestration_run_id},
            )
            == 1
        )


@pytest.mark.asyncio
async def test_revoked_membership_and_other_owner_cannot_read(journal: Journal) -> None:
    engine, _, actor, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope, {"text": "private"}, classification="RAW_CONTEXT", kind="CONTEXT", lease=lease
    )
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET is_active=false WHERE id=:id"), {"id": actor.membership_id}
        )
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await recorder.get_payload(scope, ref)


@pytest.mark.asyncio
async def test_deferred_ack_can_complete_after_restart_without_fake_success(
    journal: Journal,
) -> None:
    _, _, actor, recorder, scope, lease, _, _ = journal
    from work_management_ai.runtime.tool_loop_contracts import WaitReference

    invocation = uuid4()
    args = await recorder.put_payload(
        scope,
        {"goal": "training"},
        classification="RAW_CONTEXT",
        kind="TOOL_ARGUMENTS",
        lease=lease,
    )
    descriptor = InvocationDescriptor(
        invocation_id=invocation,
        call_id="child-1",
        tool_reference="delegate_agent@1",
        tool_version="1.0.0",
        arguments_fingerprint=args.fingerprint,
    )
    await recorder.append(
        scope,
        EventDraft(
            event_id=uuid4(), kind="tool.requested", invocation=descriptor, payload_ref=args
        ),
        expected_sequence=0,
        lease=lease,
    )
    receipt = await recorder.get_invocation(scope, invocation)
    assert receipt is not None
    wait = WaitReference(
        wait_id=uuid4(), scope=scope, kind="WORKFLOW_RESULT", source_invocation_id=invocation
    )
    deferred = SettledInvocation(status="DEFERRED", wait_ref=wait)
    waiting = await recorder.settle_invocation(scope, receipt, deferred, lease=lease)
    assert waiting.kind == "run.waiting" and waiting.settlement == deferred
    restarted = PostgreSQLToolLoopRecorder(service=recorder.service, actor=actor)
    pending = await restarted.get_invocation(scope, invocation)
    assert pending is not None and pending.settlement == deferred
    ref = await restarted.put_payload(
        scope,
        {"draft_ref": "draft-a"},
        classification="RAW_CONTEXT",
        kind="TOOL_RESULT",
        lease=lease,
    )
    value = SettledInvocation(status="COMPLETED", payload_ref=ref)
    completed = await restarted.settle_invocation(scope, pending, value, lease=lease)
    assert completed.sequence == 3 and completed.kind == "tool.completed"
    assert await restarted.settle_invocation(scope, pending, value, lease=lease) == completed
    canonical = await restarted.get_invocation(scope, invocation)
    assert canonical is not None and canonical.settlement == value


@pytest.mark.asyncio
async def test_checkpoint_retry_keeps_payload_identity(journal: Journal) -> None:
    engine, _, _, recorder, scope, lease, checkpoint, _ = journal
    await recorder.checkpoint(scope, checkpoint, lease=lease)
    async with engine.connect() as conn:
        before = await conn.scalar(
            text(
                "SELECT checkpoint_payload_id FROM orchestration_runtime_states "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
    await recorder.checkpoint(scope, checkpoint, lease=lease)
    async with engine.connect() as conn:
        after = await conn.scalar(
            text(
                "SELECT checkpoint_payload_id FROM orchestration_runtime_states "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
    assert after == before


@pytest.mark.asyncio
async def test_same_tenant_other_owner_and_missing_context_are_denied(journal: Journal) -> None:
    engine, sessions, actor, recorder, scope, lease, _, _ = journal
    await recorder.append(
        scope, EventDraft(event_id=uuid4(), kind="turn.started"), expected_sequence=0, lease=lease
    )
    user, member = uuid4(), uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO users(id,email_normalized,email_display,display_name,"
                "password_hash) VALUES(:id,:email,:email,'Other','hash')"
            ),
            {"id": user, "email": f"{user.hex}@example.test"},
        )
        await conn.execute(
            text(
                "INSERT INTO memberships(id,organization_id,user_id,role) VALUES(:id,"
                ":org,:user,'EMPLOYEE')"
            ),
            {"id": member, "org": scope.organization_id, "user": user},
        )
    other = replace(actor, user_id=user, membership_id=member, role=MembershipRole.EMPLOYEE)
    async with PostgreSQLAssistantTransactionFactory(sessions)(other) as tx:
        for table in (
            "orchestration_session_events",
            "orchestration_session_payloads",
            "orchestration_runtime_states",
        ):
            assert await tx.session.scalar(text(f"SELECT count(*) FROM {table}")) == 0
    async with sessions() as session, session.begin():
        await session.execute(text("SET LOCAL ROLE app_runtime"))
        assert await session.scalar(text("SELECT count(*) FROM orchestration_session_events")) == 0
    async with engine.connect() as conn:
        flags = await conn.execute(
            text(
                "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE "
                "relname IN ('orchestration_session_events',"
                "'orchestration_session_payloads','orchestration_runtime_states')"
            )
        )
        assert flags.all() == [(True, True)] * 3
    foreign = PostgreSQLToolLoopRecorder(service=recorder.service, actor=other)
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await foreign.load_checkpoint(scope)


@pytest.mark.asyncio
async def test_payload_reference_and_lifetime_cannot_be_forged(journal: Journal) -> None:
    engine, _, _, recorder, scope, lease, _, _ = journal
    ref = await recorder.put_payload(
        scope, {"value": "private"}, classification="RAW_CONTEXT", kind="CONTEXT", lease=lease
    )
    with pytest.raises(SessionError, match="PAYLOAD_REFERENCE_MISMATCH"):
        await recorder.get_payload(scope, ref.model_copy(update={"fingerprint": "b" * 64}))
    with pytest.raises(SessionError, match="PAYLOAD_TTL_INVALID"):
        await recorder.put_payload(
            scope,
            {"value": "x"},
            classification="RAW_CONTEXT",
            kind="CONTEXT",
            ttl_seconds=31 * 86400,
            lease=lease,
        )
    async with engine.connect() as conn:
        deadline = await conn.scalar(
            text(
                "SELECT expires_at FROM ai_retention_payloads WHERE "
                "organization_id=:org AND storage_kind='orchestration_runs' AND "
                "owner_id=:run"
            ),
            {"org": scope.organization_id, "run": scope.orchestration_run_id},
        )
        assert ref.expires_at <= deadline


@pytest.mark.asyncio
async def test_json_null_payload_roundtrips_without_becoming_expired(journal: Journal) -> None:
    _, _, _, recorder, scope, lease, _, _ = journal
    reference = await recorder.put_payload(
        scope, None, classification="RAW_CONTEXT", kind="CONTEXT", lease=lease
    )
    assert await recorder.get_payload(scope, reference) is None


@pytest.mark.asyncio
async def test_storage_errors_do_not_expose_private_sql_values(journal: Journal) -> None:
    _, sessions, actor, _, _, _, _, _ = journal
    import traceback

    with pytest.raises(SessionError, match="SESSION_STORAGE_FAILED") as caught:
        async with PostgreSQLSessionTransactionFactory(sessions)(actor) as tx:
            await tx.repository.session.execute(
                text("SELECT CAST(:private AS integer)"), {"private": "private-sql-value"}
            )
    assert "private-sql-value" not in "".join(traceback.format_exception(caught.value))
