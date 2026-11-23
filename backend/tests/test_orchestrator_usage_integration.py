"""Real atomic conservation and tenant/lease boundaries; synthetic providers only."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from app.core.execution_fence import ExecutionFenceContext, WorkflowJobFence, execution_fence
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import session_scope
from app.modules.assistant.domain.session import SessionError
from app.modules.assistant.domain.session_usage import JobClaim
from app.modules.planning_runs.adapters.transaction import PostgreSQLPlanningRunTransactionFactory
from tests.test_orchestrator_session_log_integration import Journal
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.runtime.run_budget import (
    BudgetReceipt,
    ChildAllocation,
    ChildJobClaim,
    UsageReservation,
)
from work_management_ai.runtime.tool_loop_contracts import RunLease, RunScope, ToolLoopUsage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


foreign_journal = journal


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_job_budget_survives_parent_expiry_and_denies_reclaimed_attempt(
    journal: Journal,
    foreign_journal: Journal,
    interrupted: bool,
):
    engine, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    allocation = ChildAllocation(
        allocation_id=uuid4(),
        allowance=ToolLoopUsage(
            total_model_requests=2,
            total_tool_calls=1,
            input_tokens=500,
            output_tokens=200,
            segments=1,
            active_seconds=180,
        ),
    )
    await store.allocate_child(scope, allocation, lease=lease)
    job_id, workflow_id = uuid4(), uuid4()
    deadline = datetime.now(UTC) + timedelta(minutes=3)
    async with engine.begin() as conn:
        ids = {
            "org": scope.organization_id,
            "actor": scope.membership_id,
            "workflow": workflow_id,
            "job": job_id,
            "deadline": deadline,
        }
        await conn.execute(
            text(
                "INSERT INTO workflow_runs (id,organization_id,requested_by_membership_id,status,"
                "workflow_name,workflow_version,verifier_version,input_goal_text) "
                "VALUES (:workflow,:org,:actor,'RUNNING','project_planning','1.0.0','1.0.0','Test')"
            ),
            ids,
        )
        await conn.execute(
            text(
                "INSERT INTO workflow_jobs (id,organization_id,workflow_run_id,job_type,status,"
                "payload,attempt_count,locked_by_worker_id,lease_until) "
                "VALUES (:job,:org,:workflow,'test','RUNNING','{}',1,'worker',:deadline)"
            ),
            ids,
        )
    await store.bind_job(scope, allocation.allocation_id, "PLANNING", job_id, lease=lease)
    await store.close_child(scope, allocation.allocation_id, lease=lease)
    _, foreign_sessions, foreign_actor, _, _, _, _, _ = foreign_journal
    async with foreign_sessions() as foreign, foreign.begin():
        await foreign.execute(text("SET LOCAL ROLE app_runtime"))
        await foreign.execute(
            text(
                "SELECT set_config('app.organization_id',:org,true), "
                "set_config('app.membership_id',:actor,true)"
            ),
            {"org": str(foreign_actor.organization_id), "actor": str(foreign_actor.membership_id)},
        )
        assert (
            await foreign.scalar(
                text("SELECT count(*) FROM orchestration_child_jobs WHERE organization_id=:org"),
                {"org": scope.organization_id},
            )
            == 0
        )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states "
                "SET lease_until=clock_timestamp()-interval '1 second' "
                "WHERE organization_id=:org AND orchestration_run_id=:run"
            ),
            {"org": scope.organization_id, "run": scope.orchestration_run_id},
        )
    claim = ChildJobClaim(
        kind="PLANNING",
        job_id=job_id,
        worker_id="worker",
        attempt=1,
        lease_until=deadline,
    )
    assert await store.job_scope(claim) == (scope, allocation.allocation_id)
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await PostgreSQLRunBudget(foreign_sessions, foreign_actor).job_scope(claim)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET is_active=false WHERE organization_id=:org AND id=:actor"),
            ids,
        )
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await store.job_scope(claim)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET is_active=true WHERE organization_id=:org AND id=:actor"),
            ids,
        )
    active = await store.reserve_for_job(
        scope,
        allocation.allocation_id,
        job_claim_token=claim,
        reservation=UsageReservation(kind="INTERVAL", allocation_id=allocation.allocation_id),
    )
    receipt = await store.reserve_for_job(
        scope,
        allocation.allocation_id,
        job_claim_token=claim,
        reservation=UsageReservation(
            kind="MODEL",
            allocation_id=allocation.allocation_id,
            interval_id=active.reservation.operation_id,
            input_tokens=30,
            output_tokens=20,
        ),
    )
    await store.settle_for_job(scope, receipt, ModelUsage(input_tokens=20, output_tokens=10))
    effect = await store.reserve_for_job(
        scope,
        allocation.allocation_id,
        job_claim_token=claim,
        reservation=UsageReservation(
            kind="TOOL",
            allocation_id=allocation.allocation_id,
            interval_id=active.reservation.operation_id,
        ),
    )
    fence = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=None,
        operation_id=effect.reservation.operation_id,
        interval_id=active.reservation.operation_id,
        workflow=WorkflowJobFence(
            allocation_id=allocation.allocation_id,
            claim=JobClaim("PLANNING", job_id, "worker", 1, deadline),
        ),
    )
    transactions = PostgreSQLPlanningRunTransactionFactory(sessions)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET role='EMPLOYEE' WHERE organization_id=:org AND id=:actor"),
            ids,
        )
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"), execution_fence(fence):
        async with transactions(actor) as transaction:
            await transaction.commit()
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET role='MANAGER' WHERE organization_id=:org AND id=:actor"),
            ids,
        )
    with execution_fence(fence):
        async with transactions(actor) as transaction:
            await transaction.session.execute(
                text(
                    "UPDATE workflow_runs SET updated_at=clock_timestamp() "
                    "WHERE organization_id=:org AND id=:workflow"
                ),
                ids,
            )
            await transaction.commit()
    await store.settle_for_job(scope, effect, None)
    if not interrupted:
        await store.settle_for_job(scope, active, None)
        await store.close_for_job(scope, allocation.allocation_id, claim)
        assert (await store.remaining(scope)).available.total_model_requests == 23
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE workflow_jobs SET attempt_count=2 WHERE organization_id=:org AND id=:job"),
            ids,
        )
    if interrupted:
        reclaimed = claim.model_copy(update={"attempt": 2})
        await store.close_for_job(scope, allocation.allocation_id, reclaimed)
        remaining = await store.remaining(scope)
        assert remaining.available.total_model_requests == 23
        assert remaining.usage.active_seconds == 180
    with pytest.raises(SessionError, match="WORKFLOW_LEASE_LOST"), execution_fence(fence):
        async with transactions(actor) as transaction:
            await transaction.commit()
    with pytest.raises(SessionError, match="WORKFLOW_LEASE_LOST"):
        await store.settle_for_job(scope, receipt, ModelUsage(input_tokens=20, output_tokens=10))
    with pytest.raises(SessionError, match="WORKFLOW_LEASE_LOST"):
        await store.reserve_for_job(
            scope,
            allocation.allocation_id,
            job_claim_token=claim,
            reservation=UsageReservation(kind="INTERVAL", allocation_id=allocation.allocation_id),
        )
    assert (await store.remaining(scope)).usage.total_model_requests == 1


async def interval(
    store: PostgreSQLRunBudget, scope: RunScope, lease: RunLease, allocation_id: UUID | None = None
) -> BudgetReceipt:
    return await store.reserve(
        scope, UsageReservation(kind="INTERVAL", allocation_id=allocation_id), lease=lease
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "limit"), [("MODEL", 8), ("TOOL", 12), ("HANDOFF", 6)])
async def test_counter_denied_before_backend(
    journal: Journal, kind: Literal["MODEL", "TOOL", "HANDOFF"], limit: int
):
    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    request = UsageReservation(
        kind=kind,
        interval_id=active.reservation.operation_id,
        input_tokens=1 if kind == "MODEL" else 0,
        output_tokens=1 if kind == "MODEL" else 0,
    )
    for _ in range(limit):
        await store.reserve(
            scope, request.model_copy(update={"operation_id": uuid4()}), lease=lease
        )
    with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
        await store.reserve(
            scope, request.model_copy(update={"operation_id": uuid4()}), lease=lease
        )
    resumed = await PostgreSQLRunBudget(sessions, actor).remaining(scope)
    assert resumed.usage == (await store.remaining(scope)).usage


@pytest.mark.asyncio
async def test_segment_five_denied_and_waits_stop_clock(journal: Journal):
    engine, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    for _ in range(4):
        receipt = await interval(store, scope, lease)
        await store.settle(scope, receipt, None)
    before = (await store.remaining(scope)).usage
    await asyncio.sleep(0.02)
    assert (await store.remaining(scope)).usage == before
    assert before.segments == 4 and before.active_seconds < 5
    with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
        await interval(store, scope, lease)
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT count(*) FROM orchestration_usage WHERE organization_id=:org"),
                {"org": scope.organization_id},
            )
            == 5
        )


@pytest.mark.asyncio
async def test_concurrent_reservation_tokens_and_replay(journal: Journal):
    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    requests = [
        UsageReservation(
            kind="MODEL",
            interval_id=active.reservation.operation_id,
            input_tokens=24000,
            output_tokens=4000,
        )
        for _ in range(4)
    ]
    results = await asyncio.gather(
        *(store.reserve(scope, r, lease=lease) for r in requests), return_exceptions=True
    )
    receipts = [r for r in results if not isinstance(r, BaseException)]
    assert len(receipts) == 2
    persisted = (await store.remaining(scope)).usage
    assert (persisted.input_tokens, persisted.output_tokens) == (48000, 8000)
    receipt = receipts[0]
    replay = await store.reserve(scope, receipt.reservation, lease=lease)
    assert replay.replayed
    assert (await store.remaining(scope)).usage == persisted
    await store.settle(scope, receipt, ModelUsage(input_tokens=100, output_tokens=20))
    await store.settle(scope, receipt, ModelUsage(input_tokens=100, output_tokens=20))
    assert (await store.remaining(scope)).usage.input_tokens == 24100
    with pytest.raises(SessionError, match="SETTLEMENT_CONFLICT"):
        await store.settle(scope, receipt, ModelUsage(input_tokens=1, output_tokens=1))


@pytest.mark.asyncio
async def test_child_pool_conservation_and_aggregate_request_25(journal: Journal):
    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    allocation = ChildAllocation(
        allocation_id=uuid4(),
        allowance=ToolLoopUsage(
            total_model_requests=24,
            input_tokens=24000,
            output_tokens=4000,
            segments=1,
            active_seconds=180,
        ),
    )
    await store.allocate_child(scope, allocation, lease=lease)
    remaining = await store.remaining(scope)
    assert remaining.available.total_model_requests == 0
    assert remaining.usage.total_model_requests == 0
    with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
        await store.allocate_child(
            scope, allocation.model_copy(update={"allocation_id": uuid4()}), lease=lease
        )
    active = await interval(store, scope, lease, allocation.allocation_id)
    for _ in range(24):
        receipt = await store.reserve(
            scope,
            UsageReservation(
                kind="MODEL",
                allocation_id=allocation.allocation_id,
                interval_id=active.reservation.operation_id,
                input_tokens=1,
                output_tokens=1,
            ),
            lease=lease,
        )
        await store.settle(scope, receipt, None)
    with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
        await store.reserve(
            scope,
            UsageReservation(
                kind="MODEL",
                allocation_id=allocation.allocation_id,
                interval_id=active.reservation.operation_id,
                input_tokens=1,
                output_tokens=1,
            ),
            lease=lease,
        )
    await store.settle(scope, active, None)
    await store.close_child(scope, allocation.allocation_id, lease=lease)
    remaining = await store.remaining(scope)
    assert remaining.usage.total_model_requests == 24
    assert remaining.available.input_tokens == 48000 - 24
    assert await store.allocate_child(scope, allocation, lease=lease) == allocation
    with pytest.raises(SessionError, match="ALLOCATION_CLOSED"):
        await interval(store, scope, lease, allocation.allocation_id)


@pytest.mark.asyncio
async def test_expired_interval_and_stale_worker_cannot_reserve_or_settle(journal: Journal):
    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_usage SET started_at=clock_timestamp()-interval '181 "
                "seconds' WHERE operation_id=:id"
            ),
            {"id": active.reservation.operation_id},
        )
    with pytest.raises(SessionError, match="ACTIVE_INTERVAL_EXPIRED"):
        await store.reserve(
            scope,
            UsageReservation(kind="TOOL", interval_id=active.reservation.operation_id),
            lease=lease,
        )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=clock_timestamp()-interval"
                " '1 second' WHERE organization_id=:org AND orchestration_run_id=:run"
            ),
            {"org": scope.organization_id, "run": scope.orchestration_run_id},
        )
    fresh = await recorder.claim(scope, job_id=job_id, worker_id="worker-b", lease_seconds=60)
    assert fresh.fencing_token > lease.fencing_token
    for operation in (
        store.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease),
        store.settle(scope, active, None),
    ):
        with pytest.raises(SessionError, match="LEASE_LOST"):
            await operation
    replay = await store.reserve(scope, active.reservation, lease=fresh)
    await store.settle(scope, replay, None)  # lost worker's clock cannot refund its reservation
    assert (await store.remaining(scope)).usage.active_seconds == 180


@pytest.mark.asyncio
async def test_cross_tenant_and_rls(journal: Journal):
    _, sessions, actor, _, scope, _, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    foreign = scope.model_copy(update={"organization_id": uuid4()})
    with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
        await store.remaining(foreign)
    async with sessions() as session:
        await session.execute(text("SET LOCAL ROLE app_runtime"))
        await session.execute(
            text(
                "SELECT set_config('app.organization_id', :org, true), "
                "set_config('app.membership_id', :member, true)"
            ),
            {"org": str(uuid4()), "member": str(actor.membership_id)},
        )
        assert list((await session.execute(text("SELECT * FROM orchestration_usage"))).all()) == []


@pytest.mark.asyncio
async def test_retry_repair_bounds_and_clock_cannot_close_pending(journal: Journal) -> None:
    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    base = UsageReservation(
        kind="MODEL", interval_id=active.reservation.operation_id, input_tokens=20, output_tokens=10
    )
    receipt = await store.reserve(scope, base, lease=lease)
    with pytest.raises(SessionError, match="INTERVAL_OPERATIONS_PENDING"):
        await store.settle(scope, active, None)
    await store.settle(scope, receipt, None)
    retry = await store.reserve(
        scope,
        base.model_copy(update={"operation_id": uuid4(), "retry_of": base.operation_id}),
        lease=lease,
    )
    await store.settle(scope, retry, None)
    with pytest.raises(SessionError, match="BUDGET_ATTEMPT_EXHAUSTED"):
        await store.reserve(
            scope,
            base.model_copy(update={"operation_id": uuid4(), "retry_of": base.operation_id}),
            lease=lease,
        )
    with pytest.raises(SessionError, match="BUDGET_ATTEMPT_EXHAUSTED"):
        await store.reserve(
            scope,
            base.model_copy(
                update={"operation_id": uuid4(), "retry_of": retry.reservation.operation_id}
            ),
            lease=lease,
        )
    repair = await store.reserve(
        scope,
        base.model_copy(update={"operation_id": uuid4(), "repair_of": base.operation_id}),
        lease=lease,
    )
    await store.settle(scope, repair, None)
    with pytest.raises(SessionError, match="BUDGET_ATTEMPT_EXHAUSTED"):
        await store.reserve(
            scope,
            base.model_copy(update={"operation_id": uuid4(), "repair_of": base.operation_id}),
            lease=lease,
        )
    assert (await store.remaining(scope)).usage.input_tokens == 60
    await store.settle(scope, active, None)


@pytest.mark.asyncio
async def test_full_active_budget_and_child_resume_identity(journal: Journal) -> None:
    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    store = PostgreSQLRunBudget(sessions, actor)
    allocation = ChildAllocation(
        allocation_id=uuid4(),
        allowance=ToolLoopUsage(
            total_model_requests=2,
            input_tokens=100,
            output_tokens=100,
            segments=2,
            active_seconds=360,
        ),
    )
    await store.allocate_child(scope, allocation, lease=lease)
    first = await interval(store, scope, lease, allocation.allocation_id)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=clock_timestamp()-interval"
                " '1 second' WHERE organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    fresh = await recorder.claim(scope, job_id=job_id, worker_id="resume", lease_seconds=60)
    await store.allocate_child(scope, allocation, lease=fresh)
    resumed = await interval(store, scope, fresh, allocation.allocation_id)
    assert resumed.reservation.allocation_id == first.reservation.allocation_id
    # Four executing worker intervals, including crashed ones, exhaust 720s.
    await interval(store, scope, fresh)
    await interval(store, scope, fresh)
    assert (await store.remaining(scope)).usage.active_seconds == 720
    with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
        await interval(store, scope, fresh)


@pytest.mark.asyncio
async def test_effect_fence_lives_in_same_transaction_as_write(journal: Journal) -> None:
    from app.modules.assistant.adapters.session_usage_repository import (
        PostgreSQLUsageTransactionFactory,
    )
    from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
    from app.modules.assistant.application.session_usage_service import RunUsageService

    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    tool = await store.reserve(
        scope,
        UsageReservation(kind="TOOL", interval_id=active.reservation.operation_id),
        lease=lease,
    )
    factory = PostgreSQLUsageTransactionFactory(sessions)
    async with factory(actor) as tx:
        await RunUsageService.guard_effect(
            tx.repository, session_scope(scope), session_lease(lease), tool.reservation.operation_id
        )
        # Operational synthetic write, not an approval bypass/business mutation.
        await tx.repository.session.execute(
            text(
                "UPDATE assistant_jobs SET attempt_count=attempt_count+1 WHERE "
                "organization_id=:org AND id=:id"
            ),
            {"org": scope.organization_id, "id": job_id},
        )
        await tx.commit()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=clock_timestamp()-interval"
                " '1 second' WHERE organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    await recorder.claim(scope, job_id=job_id, worker_id="takeover", lease_seconds=60)
    with pytest.raises(SessionError, match="LEASE_LOST"):
        async with factory(actor) as tx:
            await RunUsageService.guard_effect(
                tx.repository,
                session_scope(scope),
                session_lease(lease),
                tool.reservation.operation_id,
            )
            await tx.repository.session.execute(
                text(
                    "UPDATE assistant_jobs SET attempt_count=attempt_count+1 WHERE "
                    "organization_id=:org AND id=:id"
                ),
                {"org": scope.organization_id, "id": job_id},
            )
            await tx.commit()
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT attempt_count FROM assistant_jobs WHERE id=:id"), {"id": job_id}
            )
            == 1
        )


@pytest.mark.asyncio
async def test_recording_boundary_charges_tool_and_handoff_once(journal: Journal) -> None:
    from work_management_ai.runtime.run_budget import CandidateBudget, candidate_budget
    from work_management_ai.runtime.session_log import EventDraft, InvocationDescriptor

    _, sessions, actor, recorder, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    payload = await recorder.put_payload(
        scope, {}, classification="RAW_CONTEXT", kind="TOOL_ARGUMENTS", lease=lease
    )
    descriptor = InvocationDescriptor(
        invocation_id=uuid4(),
        call_id="call-1",
        tool_reference="delegate_agent@1",
        tool_version="1.0.0",
        arguments_fingerprint=payload.fingerprint,
    )
    event = EventDraft(
        event_id=uuid4(), kind="tool.requested", invocation=descriptor, payload_ref=payload
    )
    context = CandidateBudget(
        scope=scope, lease=lease, store=store, interval_id=active.reservation.operation_id
    )
    with candidate_budget(context):
        await recorder.append(scope, event, expected_sequence=0, lease=lease)
        await recorder.append(scope, event, expected_sequence=0, lease=lease)
    usage = (await store.remaining(scope)).usage
    assert usage.total_tool_calls == usage.handoffs == 1


@pytest.mark.asyncio
async def test_real_budget_wrapper_denies_ninth_provider_call(journal: Journal) -> None:
    from pydantic import BaseModel

    from work_management_ai.model_gateway.mock import MockModelGateway
    from work_management_ai.model_gateway.tool_contracts import ToolItem, ToolModelRequest
    from work_management_ai.runtime.run_budget import (
        BudgetedToolLoopGateway,
        CandidateBudget,
        candidate_budget,
    )

    class Answer(BaseModel):
        text: str

    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    provider = MockModelGateway(
        fixtures={"p": {"final": {"text": "ok"}}},
        usage_fixtures={"p": ModelUsage(input_tokens=10, output_tokens=1)},
    )
    gateway = BudgetedToolLoopGateway(provider, native=provider)
    request = ToolModelRequest(
        invocation_key="p",
        items=(ToolItem(kind="message", role="user", content="hi"),),
        tools=(),
        response_schema=Answer,
        max_output_tokens=100,
    )
    context = CandidateBudget(
        scope=scope, lease=lease, store=store, interval_id=active.reservation.operation_id
    )
    with candidate_budget(context):
        for _ in range(8):
            await gateway.generate_step(request)
        with pytest.raises(SessionError, match="BUDGET_EXHAUSTED"):
            await gateway.generate_step(request)
    assert len(provider.step_requests) == 8


@pytest.mark.asyncio
async def test_usage_identity_and_settlement_are_immutable(journal: Journal) -> None:
    from sqlalchemy.exc import DBAPIError

    from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory

    _, sessions, actor, _, scope, lease, _, _ = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    await store.settle(scope, active, None)
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as tx:
        role = (
            await tx.session.execute(
                text("SELECT current_user,rolbypassrls FROM pg_roles WHERE rolname=current_user")
            )
        ).one()
        assert role == ("app_runtime", False)
        for sql in (
            "DELETE FROM orchestration_usage",
            "UPDATE orchestration_usage SET amount='{}'::jsonb",
            "UPDATE orchestration_usage SET closed=false",
            "UPDATE orchestration_usage SET settled='{}'::jsonb",
        ):
            with pytest.raises(DBAPIError):
                async with tx.session.begin_nested():
                    await tx.session.execute(text(sql))


@pytest.mark.asyncio
async def test_expiry_before_commit_rolls_back_effect(journal: Journal) -> None:
    from app.modules.assistant.adapters.session_usage_repository import PostgreSQLUsageRepository
    from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
    from app.modules.assistant.application.session_usage_service import RunUsageService

    engine, sessions, actor, _, scope, lease, _, job_id = journal
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    tool = await store.reserve(
        scope,
        UsageReservation(kind="TOOL", interval_id=active.reservation.operation_id),
        lease=lease,
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states "
                "SET lease_until=clock_timestamp()+interval '150 milliseconds' "
                "WHERE organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    with pytest.raises(SessionError, match="LEASE_LOST"):
        async with store.service.effect_transaction(
            session_scope(scope), session_lease(lease), tool.reservation.operation_id
        ) as repo:
            assert isinstance(repo, PostgreSQLUsageRepository)
            await RunUsageService.guard_effect(
                repo, session_scope(scope), session_lease(lease), tool.reservation.operation_id
            )
            await repo.session.execute(
                text("UPDATE assistant_jobs SET attempt_count=1 WHERE id=:id"), {"id": job_id}
            )
            await asyncio.sleep(0.2)
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT attempt_count FROM assistant_jobs WHERE id=:id"), {"id": job_id}
            )
            == 0
        )


@pytest.mark.asyncio
async def test_preledger_run_cannot_silently_reset_budget(journal: Journal) -> None:
    engine, sessions, actor, recorder, scope, lease, checkpoint, _ = journal
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM orchestration_usage WHERE organization_id=:org"),
            {"org": scope.organization_id},
        )
    await recorder.initialize(scope, checkpoint)
    store = PostgreSQLRunBudget(sessions, actor)
    with pytest.raises(SessionError, match="BUDGET_NOT_INITIALIZED"):
        await interval(store, scope, lease)


@pytest.mark.asyncio
async def test_usage_foreign_parent_fk_even_without_rls(
    journal: Journal, foreign_journal: Journal
) -> None:
    """A real foreign allocation cannot bind to this run, even for a DB owner."""
    from sqlalchemy.exc import DBAPIError

    engine, sessions, actor, _, scope, lease, _, _ = journal
    _, other_sessions, other_actor, _, other_scope, other_lease, _, _ = foreign_journal
    assert scope.organization_id != other_scope.organization_id
    foreign_id = uuid4()
    await PostgreSQLRunBudget(other_sessions, other_actor).allocate_child(
        other_scope,
        ChildAllocation(allocation_id=foreign_id, allowance=ToolLoopUsage(total_model_requests=1)),
        lease=other_lease,
    )
    store = PostgreSQLRunBudget(sessions, actor)
    active = await interval(store, scope, lease)
    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO orchestration_usage "
                    "(organization_id,orchestration_run_id,operation_id,kind,amount,fingerprint,"
                    "fence,started_at,allocation_id,interval_id,closed) "
                    "SELECT organization_id,orchestration_run_id,:op,'TOOL',amount,fingerprint,"
                    "fence,started_at,:foreign,:active,false FROM orchestration_usage "
                    "WHERE organization_id=:org AND operation_id=:active"
                ),
                {
                    "op": uuid4(),
                    "foreign": foreign_id,
                    "active": active.reservation.operation_id,
                    "org": scope.organization_id,
                },
            )
