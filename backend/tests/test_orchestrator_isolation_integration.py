"""Exercise the runtime role and public event service under different actors."""

import os
from dataclasses import replace
from typing import Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from app.modules.assistant.adapters.tool_loop_recorder import PostgreSQLToolLoopRecorder
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.event_service import AssistantEventService
from app.modules.assistant.application.service import ResourceNotFoundError
from app.modules.assistant.domain.session import SessionError
from tests.test_orchestrator_session_log_integration import Journal, seed_members
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.runtime.session_log import EventDraft

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL"),
]


@pytest.mark.parametrize("viewer", ["foreign", "same-tenant", "revoked"])
async def test_private_journal_and_event_cursor_reauthorize_viewer(
    journal: Journal, viewer: Literal["foreign", "same-tenant", "revoked"]
) -> None:
    engine, sessions, owner, recorder, scope, lease, _, _ = journal
    private = "private-prompt-and-provider-error-marker"
    payload = await recorder.put_payload(
        scope, {"text": private}, classification="RAW_CONTEXT", kind="CONTEXT", lease=lease
    )
    await recorder.append(
        scope,
        EventDraft(event_id=uuid4(), kind="context.selected", payload_ref=payload),
        expected_sequence=0,
        lease=lease,
    )
    assert await recorder.get_payload(scope, payload) == {"text": private}
    transactions = PostgreSQLAssistantTransactionFactory(sessions)
    events = AssistantEventService(transaction_factory=transactions)
    frames = await events.replay(
        actor=owner, conversation_id=scope.conversation_id, after_sequence=0
    )
    assert frames and private not in "".join(frames)
    outsider = owner
    async with engine.begin() as connection:
        if viewer == "revoked":
            await connection.execute(
                text("UPDATE memberships SET is_active=false WHERE id=:id"),
                {"id": owner.membership_id},
            )
        else:
            org, member = uuid4(), uuid4()
            await seed_members(connection, ((org, member),))
            if viewer == "same-tenant":
                org = owner.organization_id
                await connection.execute(
                    text("UPDATE memberships SET organization_id=:org WHERE id=:id"),
                    {"org": org, "id": member},
                )
            user = await connection.scalar(
                text("SELECT user_id FROM memberships WHERE id=:id"), {"id": member}
            )
            assert isinstance(user, UUID)
            outsider = replace(owner, organization_id=org, membership_id=member, user_id=user)
    denied = PostgreSQLToolLoopRecorder(service=recorder.service, actor=outsider)
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await denied.get_payload(scope, payload)
    with pytest.raises(ResourceNotFoundError):
        await events.authorize(actor=outsider, conversation_id=scope.conversation_id)
    async with transactions(outsider) as transaction:
        for table in (
            "orchestration_runtime_states",
            "orchestration_session_events",
            "orchestration_session_payloads",
            "orchestration_usage",
        ):
            assert (
                await transaction.session.scalar(
                    text(f"SELECT count(*) FROM {table} WHERE orchestration_run_id=:run"),
                    {"run": scope.orchestration_run_id},
                )
                == 0
            )
