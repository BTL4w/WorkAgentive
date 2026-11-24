"""Exercise recovery of the production candidate worker using PostgreSQL receipts."""

import os
from typing import Any, Literal

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.modules.assistant.adapters.session_continuations import PostgreSQLContinuations
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import PostgreSQLToolLoopRecorder
from app.modules.assistant.application.job_service import AssistantJobService
from app.modules.assistant.domain.models import AssistantJob
from app.modules.planning_runs.application.run_service import PlanningRunService
from tests.test_orchestrator_tool_loop_worker_integration import run_candidate_turn
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.runtime.run_budget import BudgetReceipt
from work_management_ai.runtime.tool_loop_contracts import RunScope

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL"),
]

Boundary = Literal[
    "before-model-decision-commit",
    "after-tool-request",
    "after-planning-commit",
    "after-deferred-result",
    "before-wait-commit",
]


class WorkerInterrupted(BaseException):
    """Skip the normal job exception handler, as when the process disappears."""


@pytest.mark.parametrize(
    "boundary",
    [
        "before-model-decision-commit",
        "after-tool-request",
        "after-planning-commit",
        "after-deferred-result",
        "before-wait-commit",
    ],
)
async def test_planning_worker_recovers_from_durable_boundary(
    boundary: Boundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def interrupt(
        engine: AsyncEngine, job: AssistantJob, worker: AssistantJobService, errors: list[Exception]
    ) -> None:
        with monkeypatch.context() as patch:
            original_budget_settle = PostgreSQLRunBudget.settle

            async def settle_budget(
                ledger: PostgreSQLRunBudget,
                scope: RunScope,
                receipt: BudgetReceipt,
                usage: ModelUsage | None,
            ) -> None:
                # A vanished process cannot run its in-memory finally block.
                if receipt.reservation.kind != "INTERVAL" or boundary == "before-wait-commit":
                    await original_budget_settle(ledger, scope, receipt, usage)

            patch.setattr(PostgreSQLRunBudget, "settle", settle_budget)
            if boundary == "before-model-decision-commit":
                original_put = PostgreSQLToolLoopRecorder.put_payload

                async def put(*args: Any, **kwargs: Any) -> Any:
                    if kwargs["kind"] == "TOOL_ARGUMENTS":
                        raise WorkerInterrupted
                    return await original_put(*args, **kwargs)

                patch.setattr(PostgreSQLToolLoopRecorder, "put_payload", put)
            elif boundary == "after-tool-request":
                original_append = PostgreSQLToolLoopRecorder.append

                async def append(*args: Any, **kwargs: Any) -> Any:
                    record = await original_append(*args, **kwargs)
                    if record.kind == "tool.requested":
                        raise WorkerInterrupted
                    return record

                patch.setattr(PostgreSQLToolLoopRecorder, "append", append)
            elif boundary == "after-planning-commit":
                original_create = PlanningRunService.create_planning_run

                async def create(*args: Any, **kwargs: Any) -> Any:
                    await original_create(*args, **kwargs)
                    raise WorkerInterrupted

                patch.setattr(PlanningRunService, "create_planning_run", create)
            elif boundary == "after-deferred-result":
                original_settle = PostgreSQLToolLoopRecorder.settle_invocation

                async def settle(*args: Any, **kwargs: Any) -> Any:
                    record = await original_settle(*args, **kwargs)
                    if record.kind == "run.waiting":
                        raise WorkerInterrupted
                    return record

                patch.setattr(PostgreSQLToolLoopRecorder, "settle_invocation", settle)
            else:
                original_open = PostgreSQLContinuations.open_wait

                async def open_wait(*args: Any, **kwargs: Any) -> Any:
                    await original_open(*args, **kwargs)
                    raise WorkerInterrupted

                patch.setattr(PostgreSQLContinuations, "open_wait", open_wait)

            with pytest.RaisesGroup(WorkerInterrupted, allow_unwrapped=True):
                await worker.run_once(
                    worker_id="crashed-worker", organization_id=job.organization_id
                )
                if errors:
                    raise errors[0]

        async with engine.begin() as connection:
            current = await connection.scalar(
                text("SELECT status FROM assistant_jobs WHERE id=:id"), {"id": job.id}
            )
            assert current == "RUNNING"
            for table, field in (
                ("assistant_jobs", "id"),
                ("orchestration_runtime_states", "orchestration_run_id"),
            ):
                await connection.execute(
                    text(
                        f"UPDATE {table} SET lease_until=clock_timestamp()-interval '1 second' "
                        f"WHERE organization_id=:org AND {field}=:id"
                    ),
                    {
                        "id": job.id if table == "assistant_jobs" else job.orchestration_run_id,
                        "org": job.organization_id,
                    },
                )

    await run_candidate_turn(
        planning=True,
        interrupt=interrupt,
        lost_model_response=boundary == "before-model-decision-commit",
    )


async def test_verified_response_survives_interruption_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder

    async def interrupt(
        engine: AsyncEngine, organization_id: UUID, run_id: UUID, worker: AssistantJobService
    ) -> None:
        with monkeypatch.context() as patch:

            async def stop(*args: Any, **kwargs: Any) -> Any:
                raise WorkerInterrupted

            patch.setattr(PostgreSQLExecutionRecorder, "finish_orchestrator_run", stop)
            with pytest.raises(WorkerInterrupted):
                await worker.run_once(worker_id="crashed-resume", organization_id=organization_id)
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE assistant_jobs SET lease_until=now()-interval '1 second' "
                    "WHERE orchestration_run_id=:run AND job_type='assistant.turn.resume'"
                ),
                {"run": run_id},
            )
            await connection.execute(
                text(
                    "UPDATE orchestration_runtime_states SET lease_until=now()-interval '1 second' "
                    "WHERE orchestration_run_id=:run"
                ),
                {"run": run_id},
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM assistant_events WHERE orchestration_run_id=:run "
                        "AND event_type='assistant.turn.response.v1'"
                    ),
                    {"run": run_id},
                )
                == 0
            )

    await run_candidate_turn(planning=True, interrupt_resume=interrupt)
