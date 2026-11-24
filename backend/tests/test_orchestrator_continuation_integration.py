"""Continuation owner and fencing tests against PostgreSQL."""

import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.ports import LinkedWorkflowEvent
from app.modules.assistant.application.projection_service import AssistantProjectionService
from app.modules.assistant.domain.models import AgentRun, AssistantConversation
from app.modules.assistant.domain.session import SessionError
from app.modules.planning_runs.adapters.database_models import (
    ApprovalModel,
    ProposalModel,
    ProposalVersionModel,
    WorkflowEventModel,
    WorkflowRunModel,
)
from app.modules.planning_runs.domain.models import WorkflowEvent
from tests.test_orchestrator_session_log_integration import Journal, seed_members, turn_graph
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.runtime.context_exploration import ContextAccessDecision
from work_management_ai.runtime.session_history import ToolResultPayload
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationDescriptor,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import RunScope, VersionPin, WaitReference
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    UseOutcome,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


async def submit_target(seed: Journal, message_text: str = "Plan a customer event") -> UUID:
    _, sessions, actor, _, scope, _, _, _ = seed
    conversation = replace(
        AssistantConversation.create(
            organization_id=actor.organization_id,
            owner_membership_id=actor.membership_id,
            locale="en",
        ),
        id=scope.conversation_id,
    )
    message, turn, run, job, event = turn_graph(conversation=conversation, actor=actor)
    key = str(turn.id)
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as tx:
        await tx.repository.submit_message_mutation(
            actor=actor,
            message=replace(
                message, dedupe_key=key, content_blocks=({"kind": "text", "text": message_text},)
            ),
            turn=turn,
            run=run,
            job=job,
            event=replace(event, dedupe_key=key),
            request_id=key,
            idempotency_key=key,
            request_fingerprint=key,
        )
        await tx.commit()
    return run.id


async def prepare_wait(
    seed: Journal, *, workflow_id: UUID | None = None, proposal_id: UUID | None = None
) -> WaitReference:
    _, _, _, _, scope, _, _, _ = seed
    reference = WaitReference(
        wait_id=uuid4(),
        scope=scope,
        kind="HUMAN_DECISION"
        if proposal_id is not None
        else ("WORKFLOW_RESULT" if workflow_id is not None else "USER_INPUT"),
        resource_id=proposal_id or workflow_id,
        resource_version=1 if proposal_id or workflow_id else None,
        source_invocation_id=uuid4(),
    )
    await suspend_fixture_job(seed, reference)
    return reference


async def suspend_fixture_job(
    seed: Journal, reference: WaitReference, *, tool_pins: tuple[VersionPin, ...] = ()
) -> None:
    engine, _, _, recorder, scope, lease, checkpoint, job_id = seed
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE orchestration_runs SET status='RUNNING' WHERE id=:id"),
            {"id": scope.orchestration_run_id},
        )
        await connection.execute(
            text("UPDATE assistant_turns SET status='RUNNING' WHERE id=:id"),
            {"id": scope.turn_id},
        )
        await connection.execute(
            text(
                "UPDATE assistant_jobs SET status='RUNNING', locked_by=:worker, "
                "lease_until=:until, attempt_count=1 WHERE id=:id"
            ),
            {"id": job_id, "worker": lease.worker_id, "until": lease.expires_at},
        )
    records = await recorder.read(scope, after_sequence=0)
    sequence = records[-1].sequence if records else 0
    await recorder.checkpoint(
        scope,
        checkpoint.model_copy(
            update={"pending_wait": reference, "sequence": sequence, "tool_versions": tool_pins}
        ),
        lease=lease,
    )


async def test_two_concurrent_targets_leave_one_owner_and_fence_the_source(journal: Journal):
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    first, second = await submit_target(journal), await submit_target(journal)
    reference = await prepare_wait(journal, proposal_id=uuid4())
    service = continuation_service(sessions)
    opened = await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    assert opened.version == 1
    assert opened.owner_run_id == scope.orchestration_run_id
    assert await service.open_wait(actor, WaitDraft(reference=reference, lease=lease)) == opened

    receipts = await asyncio.gather(
        service.transfer(actor, wait_id=reference.wait_id, expected_version=1, target_run_id=first),
        service.transfer(
            actor, wait_id=reference.wait_id, expected_version=1, target_run_id=second
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(receipt, BaseException) for receipt in receipts) == 1
    assert (
        sum(
            isinstance(receipt, SessionError) and str(receipt) == "CONTINUATION_VERSION_CONFLICT"
            for receipt in receipts
        )
        == 1
    )
    winner = next(receipt for receipt in receipts if not isinstance(receipt, BaseException))
    replay = await service.transfer(
        actor,
        wait_id=reference.wait_id,
        expected_version=1,
        target_run_id=winner.wait.owner_run_id,
    )
    assert replay == winner
    async with engine.connect() as connection:
        owners = (
            (
                await connection.execute(
                    text("SELECT owner_run_id FROM orchestration_waits WHERE wait_id=:id"),
                    {"id": reference.wait_id},
                )
            )
            .scalars()
            .all()
        )
        assert len(owners) == 1 and owners[0] in {first, second}
        result = (
            await connection.execute(
                text("SELECT status,stop_reason FROM orchestration_runs WHERE id=:id"),
                {"id": scope.orchestration_run_id},
            )
        ).one()
        assert tuple(result) == ("COMPLETED", "CONTINUATION_TRANSFERRED")
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await recorder.claim(scope, job_id=job_id, worker_id=lease.worker_id, lease_seconds=60)


async def test_wait_owner_cannot_be_read_or_transferred_from_another_tenant(journal: Journal):
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, _, _, lease, _, _ = journal
    reference = await prepare_wait(journal)
    service = continuation_service(sessions)
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    other_org, other_member = uuid4(), uuid4()
    async with engine.begin() as connection:
        await seed_members(connection, ((other_org, other_member),))
        other_user = await connection.scalar(
            text("SELECT user_id FROM memberships WHERE id=:id"), {"id": other_member}
        )
    assert isinstance(other_user, UUID)
    outsider = replace(
        actor, organization_id=other_org, membership_id=other_member, user_id=other_user
    )
    async with PostgreSQLAssistantTransactionFactory(sessions)(outsider) as tx:
        assert await tx.session.scalar(text("SELECT count(*) FROM orchestration_waits")) == 0
    with pytest.raises(SessionError, match="CONTINUATION_NOT_FOUND"):
        await service.transfer(
            outsider, wait_id=reference.wait_id, expected_version=1, target_run_id=uuid4()
        )


async def seed_workflow_events(
    seed: Journal, workflow_id: UUID, *, event_type: str = "workflow.failed"
) -> tuple[LinkedWorkflowEvent, ...]:
    _, sessions, actor, _, scope, _, _, _ = seed
    child = AgentRun.create(
        organization_id=actor.organization_id,
        orchestration_run_id=scope.orchestration_run_id,
        agent_id="planning",
        agent_version="1.1.0",
        manifest_fingerprint="planning-sha",
        capability="planning.create",
        typed_input={},
        budget={},
        workflow_run_id=workflow_id,
    )
    events = tuple(
        WorkflowEvent(
            id=uuid4(),
            organization_id=actor.organization_id,
            workflow_run_id=workflow_id,
            sequence=index,
            event_type=name,
            public_payload={},
            created_at=datetime.now(UTC),
        )
        for index, name in enumerate((event_type, event_type), start=1)
    )
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as tx:
        tx.session.add(
            WorkflowRunModel(
                id=workflow_id,
                organization_id=actor.organization_id,
                requested_by_membership_id=actor.membership_id,
                status="FAILED",
                version=2,
                workflow_name="project_planning",
                workflow_version="1",
                verifier_version="1",
                input_goal_text="Fixture",
            )
        )
        await tx.session.flush()
        await tx.repository.append_agent_run(run=child)
        for event in events:
            tx.session.add(
                WorkflowEventModel(
                    id=event.id,
                    organization_id=event.organization_id,
                    workflow_run_id=workflow_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    public_payload={},
                )
            )
        await tx.commit()
    return tuple(
        LinkedWorkflowEvent(child, scope.turn_id, scope.conversation_id, event) for event in events
    )


async def test_terminal_event_during_transfer_queues_only_one_job_for_current_owner(
    journal: Journal,
):
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import TransferReceipt, WaitDraft

    engine, sessions, actor, _, scope, lease, _, _ = journal
    target = await submit_target(journal)
    workflow_id = uuid4()
    reference = await prepare_wait(journal, workflow_id=workflow_id)
    service = continuation_service(sessions)
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    older, newer = await seed_workflow_events(journal, workflow_id)
    assert (await service.receive_event(actor, older)).outcome == "IGNORED"
    spoof = replace(newer, event=replace(newer.event, id=uuid4()))
    assert (await service.receive_event(actor, spoof)).outcome == "IGNORED"

    receipt, wake, replay = await asyncio.gather(
        service.transfer(
            actor, wait_id=reference.wait_id, expected_version=1, target_run_id=target
        ),
        service.receive_event(actor, newer),
        service.receive_event(actor, newer),
        return_exceptions=True,
    )
    assert not isinstance(wake, BaseException) and not isinstance(replay, BaseException)
    assert {wake.outcome, replay.outcome} == {"QUEUED", "DUPLICATE"}
    owner = target if isinstance(receipt, TransferReceipt) else scope.orchestration_run_id
    if isinstance(receipt, BaseException):
        assert isinstance(receipt, SessionError)
        assert str(receipt) == "CONTINUATION_VERSION_CONFLICT"
    async with engine.connect() as connection:
        jobs = (
            (
                await connection.execute(
                    text(
                        "SELECT orchestration_run_id FROM assistant_jobs "
                        "WHERE organization_id=:org AND job_type='assistant.turn.resume'"
                    ),
                    {"org": actor.organization_id},
                )
            )
            .scalars()
            .all()
        )
    assert jobs == [owner]
    assert replay.job_id == wake.job_id


async def test_completion_before_wait_is_recovered_and_legacy_projection_does_not_finish_owner(
    journal: Journal,
):
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, _, scope, lease, _, _ = journal
    workflow_id = uuid4()
    reference = await prepare_wait(journal, workflow_id=workflow_id)
    await seed_workflow_events(journal, workflow_id)
    projected = AssistantProjectionService(
        transaction_factory=PostgreSQLAssistantTransactionFactory(sessions)
    )
    assert await projected.project_once(organization_id=actor.organization_id) == 2
    service = continuation_service(sessions)
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    async with engine.connect() as connection:
        run_status = await connection.scalar(
            text("SELECT status FROM orchestration_runs WHERE id=:id"),
            {"id": scope.orchestration_run_id},
        )
        wake_jobs = await connection.scalar(
            text(
                "SELECT count(*) FROM assistant_jobs WHERE orchestration_run_id=:id "
                "AND job_type='assistant.turn.resume'"
            ),
            {"id": scope.orchestration_run_id},
        )
    assert run_status == "RUNNING"
    assert wake_jobs == 1


@pytest.mark.parametrize("current_version", [1, 2])
async def test_rejection_wake_requires_canonical_current_proposal_and_approval(
    journal: Journal,
    current_version: int,
):
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, _, scope, lease, _, _ = journal
    workflow_id, proposal_id, approval_id = uuid4(), uuid4(), uuid4()
    reference = await prepare_wait(journal, proposal_id=proposal_id)
    _, linked = await seed_workflow_events(journal, workflow_id)
    service = continuation_service(sessions)
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    e = replace(
        linked.event,
        id=uuid4(),
        sequence=3,
        event_type="workflow.completed",
        public_payload={
            "proposal_id": str(proposal_id),
            "proposal_version": 1,
            "approval_id": str(approval_id),
            "decision": "REJECT",
        },
    )
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as tx:
        row = ProposalModel(
            id=proposal_id,
            organization_id=actor.organization_id,
            workflow_run_id=workflow_id,
            current_version_number=current_version,
            status="REJECTED",
        )
        tx.session.add(row)
        await tx.session.flush()
        tx.session.add_all(
            [
                ProposalVersionModel(
                    id=uuid4(),
                    organization_id=actor.organization_id,
                    proposal_id=proposal_id,
                    version_number=version,
                    created_by_membership_id=actor.membership_id,
                    content={},
                )
                for version in range(1, current_version + 1)
            ]
        )
        await tx.session.flush()
        approval = ApprovalModel(
            id=approval_id,
            organization_id=actor.organization_id,
            proposal_id=proposal_id,
            proposal_version_number=1,
            status="PENDING",
        )
        tx.session.add(approval)
        await tx.session.flush()
        approval.status = "REJECTED"
        approval.decided_at = datetime.now(UTC)
        approval.decided_by_membership_id = actor.membership_id
        approval.version = 2
        row.approval_id = approval_id
        await tx.session.execute(
            text("UPDATE workflow_runs SET status='COMPLETED' WHERE id=:id"),
            {"id": workflow_id},
        )
        tx.session.add(
            WorkflowEventModel(
                id=e.id,
                organization_id=actor.organization_id,
                workflow_run_id=workflow_id,
                sequence=e.sequence,
                event_type=e.event_type,
                public_payload=e.public_payload,
            )
        )
        await tx.commit()
    assert (await service.receive_event(actor, replace(linked, event=e))).outcome == (
        "QUEUED" if current_version == 1 else "IGNORED"
    )
    async with engine.connect() as connection:
        payload = await connection.scalar(
            text(
                "SELECT payload FROM assistant_jobs WHERE orchestration_run_id=:id "
                "AND job_type='assistant.turn.resume'"
            ),
            {"id": scope.orchestration_run_id},
        )
    if current_version == 1:
        assert payload["outcome"] == "REJECTED"
    else:
        assert payload is None


async def test_only_an_authorized_candidate_chosen_in_the_user_turn_creates_a_binding(
    journal: Journal,
) -> None:
    from app.modules.assistant.adapters.context_continuations import PostgreSQLUserClarifications
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    _, sessions, actor, recorder, scope, lease, _, _ = journal
    current_outcome: UseOutcome = "REFRESH_REQUIRED"

    class Access:
        async def check(
            self,
            scope: RunScope,
            *,
            target_tool: str,
            input_fingerprint: str,
            refs: tuple[ContextRef, ...],
        ) -> ContextAccessDecision:
            return ContextAccessDecision(
                outcome=current_outcome, actor_revision="r1", policy_revision="p1"
            )

    selected, other = tuple(
        ContextRef(
            ref_id=uuid4(),
            scope=scope,
            resource_type="project",
            resource_id=uuid4(),
            version=1,
            fingerprint="a" * 64,
            observed_at=datetime.now(UTC),
        )
        for _ in range(2)
    )
    invocation = uuid4()
    pin = VersionPin(reference="explore_projects@1", version="1.0.3", fingerprint="pin")
    discovery = ContextReadObservation(
        scope=scope,
        objective_ref=uuid4(),
        invocation_id=invocation,
        tool_reference="explore_projects@1",
        tool_version=pin.version,
        outcome="OK",
        observed_at=datetime.now(UTC),
        context_refs=(selected, other),
        entities=(
            EntityResolution(
                query_fingerprint="b" * 64,
                resolution="AMBIGUOUS",
                candidate_ref_ids=(selected.ref_id, other.ref_id),
            ),
        ),
        facts=(
            ContextFact(ref_id=selected.ref_id, field="project.name", value="Project Alpha"),
            ContextFact(ref_id=other.ref_id, field="project.name", value="Project Alpine"),
        ),
    )
    arguments = await recorder.put_payload(
        scope,
        {"arguments": {}},
        classification="RAW_CONTEXT",
        kind="TOOL_ARGUMENTS",
        lease=lease,
    )
    await recorder.append(
        scope,
        EventDraft(
            event_id=uuid4(),
            kind="tool.requested",
            payload_ref=arguments,
            invocation=InvocationDescriptor(
                invocation_id=invocation,
                call_id="explore",
                tool_reference=pin.reference,
                tool_version=pin.version,
                arguments_fingerprint=arguments.fingerprint,
            ),
        ),
        expected_sequence=0,
        lease=lease,
    )
    receipt = await recorder.get_invocation(scope, invocation)
    assert receipt is not None
    assert await recorder.start_invocation(scope, receipt, lease=lease)
    payload = await recorder.put_payload(
        scope,
        ToolResultPayload(output=discovery.model_dump(mode="json")).model_dump(mode="json"),
        classification="RAW_CONTEXT",
        kind="TOOL_RESULT",
        lease=lease,
    )
    await recorder.settle_invocation(
        scope,
        receipt,
        SettledInvocation(status="COMPLETED", payload_ref=payload),
        lease=lease,
    )
    reference = WaitReference(
        wait_id=uuid4(),
        scope=scope,
        kind="USER_INPUT",
        source_invocation_id=invocation,
    )
    unrelated = await submit_target(journal, "Write a report instead")
    target = await submit_target(journal, "Project Alpha")
    await suspend_fixture_job(journal, reference, tool_pins=(pin,))
    service = continuation_service(sessions, context_access=Access())
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    for turn, error in (
        (unrelated, "CONTINUATION_TARGET_UNRELATED"),
        (target, "CONTINUATION_CONTEXT_REFRESH_REQUIRED"),
    ):
        with pytest.raises(SessionError, match=error):
            await service.transfer(
                actor,
                wait_id=reference.wait_id,
                expected_version=1,
                target_run_id=turn,
            )
    current_outcome = "ACCEPTED"
    receipt_transfer = await service.transfer(
        actor,
        wait_id=reference.wait_id,
        expected_version=1,
        target_run_id=target,
    )
    reply = receipt_transfer.wait.owner_turn_id
    reader = PostgreSQLUserClarifications(sessions, actor)
    binding = await reader.accepted(
        source_scope=scope,
        user_turn_id=reply,
        source_invocation_id=invocation,
    )
    assert binding is not None
    assert (binding.resource_id, binding.resource_version, binding.source_tool_pin) == (
        selected.resource_id,
        1,
        pin,
    )
    assert (
        await reader.accepted(
            source_scope=scope,
            user_turn_id=scope.turn_id,
            source_invocation_id=invocation,
        )
        is None
    )


async def test_revoked_owner_does_not_route_transferred_source_to_legacy_projection(
    journal: Journal,
) -> None:
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, _, scope, lease, _, _ = journal
    workflow_id = uuid4()
    target = await submit_target(journal)
    reference = await prepare_wait(journal, workflow_id=workflow_id)
    service = continuation_service(sessions)
    await service.open_wait(actor, WaitDraft(reference=reference, lease=lease))
    await service.transfer(
        actor,
        wait_id=reference.wait_id,
        expected_version=1,
        target_run_id=target,
    )
    await seed_workflow_events(journal, workflow_id, event_type="workflow.needs_input")
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE memberships SET is_active=false WHERE id=:id"),
            {"id": actor.membership_id},
        )
    projection = AssistantProjectionService(
        transaction_factory=PostgreSQLAssistantTransactionFactory(sessions)
    )
    await projection.project_once(organization_id=actor.organization_id)
    async with engine.connect() as connection:
        result = (
            await connection.execute(
                text("SELECT status,stop_reason FROM orchestration_runs WHERE id=:id"),
                {"id": scope.orchestration_run_id},
            )
        ).one()
        jobs = await connection.scalar(
            text(
                "SELECT count(*) FROM assistant_jobs WHERE job_type='assistant.turn.resume' "
                "AND organization_id=:org"
            ),
            {"org": actor.organization_id},
        )
    assert tuple(result) == ("COMPLETED", "CONTINUATION_TRANSFERRED")
    assert jobs == 0


async def test_canonical_checkpoint_admits_only_one_candidate_wait(journal: Journal) -> None:
    from app.modules.assistant.adapters.session_continuations import continuation_service
    from app.modules.assistant.domain.continuation import WaitDraft

    engine, sessions, actor, _, scope, lease, _, _ = journal
    reference = await prepare_wait(journal, workflow_id=uuid4())
    unexpected = reference.model_copy(update={"wait_id": uuid4()})
    service = continuation_service(sessions)
    correct, incorrect = await asyncio.gather(
        service.open_wait(actor, WaitDraft(reference=reference, lease=lease)),
        service.open_wait(actor, WaitDraft(reference=unexpected, lease=lease)),
        return_exceptions=True,
    )
    assert not isinstance(correct, BaseException) and correct.reference == reference
    assert isinstance(incorrect, SessionError)
    assert str(incorrect) in {"CONTINUATION_WAIT_MISMATCH", "LEASE_LOST"}
    async with engine.connect() as connection:
        stored = await connection.scalar(
            text("SELECT wait_id FROM orchestration_waits WHERE source_run_id=:run"),
            {"run": scope.orchestration_run_id},
        )
    assert stored == reference.wait_id
