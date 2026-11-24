"""Prove that a stale worker cannot commit a real Planning mutation."""

import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.core.execution_fence import ExecutionFenceContext, execution_fence
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.planning_runs.adapters.ai_runtime import PlanningAIRuntime
from app.modules.planning_runs.adapters.transaction import PostgreSQLPlanningRunTransactionFactory
from app.modules.planning_runs.application.run_service import PlanningRunService
from tests.test_orchestrator_session_log_integration import Journal
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.runtime.run_budget import UsageReservation

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL"),
]


async def test_new_fence_denies_old_planning_worker_and_manual_path_remains_available(
    journal: Journal,
) -> None:
    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    ledger = PostgreSQLRunBudget(sessions, actor)
    clock = await ledger.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease)
    reservation = await ledger.reserve(
        scope,
        UsageReservation(kind="TOOL", interval_id=clock.reservation.operation_id),
        lease=lease,
    )
    stale = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease),
        operation_id=reservation.reservation.operation_id,
        interval_id=clock.reservation.operation_id,
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE orchestration_runtime_states "
                "SET lease_until=clock_timestamp()-interval '1 second' "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
    fresh = await recorder.claim(scope, job_id=job_id, worker_id="replacement", lease_seconds=60)
    assert fresh.fencing_token > lease.fencing_token
    service = PlanningRunService(
        transaction_factory=PostgreSQLPlanningRunTransactionFactory(sessions),
        runtime=PlanningAIRuntime(progressive_enabled=True),
    )
    operation_id = str(uuid4())

    async def create():
        return await service.create_planning_run(
            actor=actor,
            message="Plan a customer event",
            locale="en",
            request_id=operation_id,
            idempotency_key=operation_id,
        )

    with execution_fence(stale), pytest.raises(ValueError, match="LEASE_LOST"):
        await create()
    async with engine.connect() as connection:
        for table in ("workflow_runs", "workflow_jobs", "proposals"):
            assert (
                await connection.scalar(
                    text(f"SELECT count(*) FROM {table} WHERE organization_id=:org"),
                    {"org": scope.organization_id},
                )
                == 0
            )
    committed = await create()
    replay = await create()
    assert replay.run.id == committed.run.id


@pytest.mark.parametrize("invalid", ["foreign-operation", "expiry", "mixed-update"])
async def test_runtime_role_cannot_move_an_ineligible_interval(
    journal: Journal, invalid: str
) -> None:
    from sqlalchemy.exc import DBAPIError

    from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
    from app.modules.assistant.domain.session_usage import assistant_job_interval

    engine, sessions, actor, recorder, scope, lease, _, job_id = journal
    interval_id = uuid4() if invalid == "foreign-operation" else assistant_job_interval(job_id)
    ledger = PostgreSQLRunBudget(sessions, actor)
    await ledger.reserve(
        scope, UsageReservation(kind="INTERVAL", operation_id=interval_id), lease=lease
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE assistant_jobs SET attempt_count=2,locked_by='replacement',"
                "status='RUNNING',lease_until=clock_timestamp()+interval '60 seconds' WHERE id=:id"
            ),
            {"id": job_id},
        )
        await connection.execute(
            text(
                "UPDATE orchestration_runtime_states "
                "SET lease_until=clock_timestamp()-interval '1 second' "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
        if invalid == "expiry":
            await connection.execute(
                text(
                    "UPDATE orchestration_usage SET started_at=clock_timestamp()-interval '1 hour' "
                    "WHERE orchestration_run_id=:run AND operation_id=:interval"
                ),
                {"run": scope.orchestration_run_id, "interval": interval_id},
            )
    fresh = await recorder.claim(
        scope, job_id=job_id, worker_id="replacement", lease_seconds=60, job_attempt=2
    )
    async with PostgreSQLAssistantTransactionFactory(sessions)(actor) as transaction:
        with pytest.raises(DBAPIError, match="ACTIVE_INTERVAL_TAKEOVER_DENIED"):
            await transaction.session.execute(
                text(
                    "UPDATE orchestration_usage SET fence=:fence"
                    + (", closed=true" if invalid == "mixed-update" else "")
                    + " WHERE orchestration_run_id=:run AND operation_id=:interval"
                ),
                {
                    "fence": fresh.fencing_token,
                    "run": scope.orchestration_run_id,
                    "interval": interval_id,
                },
            )
    if invalid == "expiry":
        with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
            await ledger.reserve(
                scope, UsageReservation(kind="INTERVAL", operation_id=interval_id), lease=fresh
            )


async def test_same_worker_reclaim_fences_previous_assistant_attempt_even_with_live_journal(
    journal: Journal,
) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlalchemy.exc import DBAPIError

    from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
    from app.modules.assistant.domain.session import SessionError

    engine, sessions, actor, recorder, scope, original, _, job_id = journal
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=now()-interval '1 second' "
                "WHERE orchestration_run_id=:run"
            ),
            {"run": scope.orchestration_run_id},
        )
    transactions = PostgreSQLAssistantTransactionFactory(sessions)
    worker_id = original.worker_id

    async def claim_job() -> int:
        now = datetime.now(UTC)
        async with transactions(scope.organization_id) as transaction:
            job = await transaction.repository.claim_job(
                organization_id=scope.organization_id,
                worker_id=worker_id,
                now=now,
                lease_until=now + timedelta(seconds=60),
            )
            assert job is not None
            await transaction.commit()
        return job.attempt_count

    first = await claim_job()
    old = await recorder.claim(
        scope, job_id=job_id, worker_id=worker_id, lease_seconds=60, job_attempt=first
    )
    async with transactions(actor) as transaction:
        with pytest.raises(DBAPIError, match="ASSISTANT_JOB_CLAIM_INVALID"):
            await transaction.session.execute(
                text(
                    "UPDATE orchestration_runtime_states SET lease_owner='unclaimed-worker' "
                    "WHERE orchestration_run_id=:run"
                ),
                {"run": scope.orchestration_run_id},
            )
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE assistant_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"),
            {"id": job_id},
        )
    second = await claim_job()
    assert second == first + 1
    with pytest.raises(SessionError, match="LEASE_LOST"):
        await recorder.put_payload(
            scope, {"text": "old attempt"}, classification="RAW_CONTEXT", kind="CONTEXT", lease=old
        )
    current = await recorder.claim(
        scope, job_id=job_id, worker_id=worker_id, lease_seconds=60, job_attempt=second
    )
    assert current.fencing_token > old.fencing_token
    assert current.job_attempt == second
