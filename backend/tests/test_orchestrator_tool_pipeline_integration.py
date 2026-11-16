"""Real journal/ledger/transaction proof; effects are synthetic job metadata only."""

import asyncio
import os
from typing import Literal
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from app.core.execution_fence import ExecutionFenceContext, assert_execution_fence, fenced_effect
from app.modules.assistant.adapters.orchestration_tools import ApplicationToolBackend
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.domain.session import SessionError
from tests.test_orchestrator_session_log_integration import Journal
from tests.test_orchestrator_session_log_integration import journal as journal
from work_management_ai.model_gateway.tool_contracts import ToolCall, ToolSchema
from work_management_ai.runtime.contracts import ContextReference, JsonValue
from work_management_ai.runtime.run_budget import (
    CandidateBudget,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.session_log import EventDraft, InvocationDescriptor, PayloadKind
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    BackendToolResult,
    PayloadReference,
    RunLease,
    RunScope,
    ToolInvocationContext,
)
from work_management_ai.runtime.tool_pipeline import PreparedTool, ToolPipeline, ToolPolicyError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str


class Catalog:
    tool = AdmittedTool(
        schema=ToolSchema(
            reference="read_project_context@1",
            description="Synthetic boundary",
            input_contract=Input,
        ),
        input_contract=Input,
        output_contract=Output,
        version="1.0.0",
        fingerprint="synthetic",
    )
    schemas = (tool.schema,)
    fingerprint = "synthetic"

    def resolve(self, wire_name: str) -> AdmittedTool:
        if wire_name != "read_project_context":
            raise KeyError(wire_name)
        return self.tool

    def close(self) -> None:
        pass


class Policy:
    def __init__(self):
        self.fail_post = False
        self.deny = False
        self.denied: list[str] = []

    async def pre_execute(
        self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
    ) -> PreparedTool:
        return PreparedTool(arguments=arguments, policy_version="test-policy-1")

    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None:
        if self.deny:
            raise ToolPolicyError("PERMISSION_DENIED", stop_segment=True)

    async def post_execute(self, context: ToolInvocationContext, result: BackendToolResult) -> None:
        if self.fail_post:
            raise ValueError("private raw error")

    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None:
        self.denied.append(safe_code)


class SyntheticService:
    """Application-style transaction + receipt; never a production business handler."""

    def __init__(self, journal: Journal):
        _, self.sessions, self.actor, _, self.scope, _, _, self.job_id = journal
        self.transactions = PostgreSQLAssistantTransactionFactory(self.sessions)

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        async with self.transactions(self.actor) as tx:
            async with fenced_effect(tx.session):
                await tx.session.execute(
                    text(
                        "UPDATE assistant_jobs SET attempt_count=attempt_count+1, "
                        "safe_error_code=:receipt WHERE organization_id=:org AND id=:id"
                    ),
                    {
                        "org": self.scope.organization_id,
                        "id": self.job_id,
                        "receipt": context.arguments_fingerprint,
                    },
                )
            await tx.commit()
        return BackendToolResult(output=Output(text="committed"))

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        async with self.transactions(self.actor) as tx:
            value = await tx.session.scalar(
                text(
                    "SELECT safe_error_code FROM assistant_jobs WHERE "
                    "organization_id=:org AND id=:id"
                ),
                {"org": self.scope.organization_id, "id": self.job_id},
            )
            if value == context.arguments_fingerprint:
                return BackendToolResult(output=Output(text="committed"))
        return None


async def setup(j: Journal):
    _, sessions, actor, recorder, scope, lease, _, _ = j
    budget = PostgreSQLRunBudget(sessions, actor)
    clock = await budget.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease)
    binding = CandidateBudget(
        scope=scope, lease=lease, store=budget, interval_id=clock.reservation.operation_id
    )
    policy, service = Policy(), SyntheticService(j)
    backend = ApplicationToolBackend(
        actor=actor, execute=service.execute, reconcile=service.reconcile
    )
    pipeline = ToolPipeline(
        store=recorder, policy=policy, backends={Catalog.tool.reference: backend}
    )
    return binding, policy, pipeline


@pytest.mark.asyncio
async def test_committed_effect_post_failure_reconciles_after_pipeline_restart(journal: Journal):
    engine, _, _, recorder, scope, lease, _, job_id = journal
    binding, policy, pipeline = await setup(journal)
    call = ToolCall(call_id="call-1", name="read_project_context", arguments={"query": "status"})
    with candidate_budget(binding):
        policy.fail_post = True
        failed = await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
        assert failed.kind == "REJECTED" and failed.safe_code == "TOOL_POSTCONDITION_FAILED"
        policy.fail_post = False
        restarted = ToolPipeline(store=recorder, policy=policy, backends=pipeline.backends)
        done = await restarted.execute(scope, call, admitted=Catalog(), lease=lease)
        assert done.kind == "COMPLETED"
        assert await restarted.execute(scope, call, admitted=Catalog(), lease=lease) == done
        policy.deny = True
        denied = await restarted.execute(scope, call, admitted=Catalog(), lease=lease)
        assert denied.kind == "REJECTED" and denied.safe_code == "PERMISSION_DENIED"
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT attempt_count FROM assistant_jobs WHERE id=:id"), {"id": job_id}
            )
            == 1
        )
    records = await recorder.read(scope, after_sequence=0)
    assert sum(r.kind == "tool.started" for r in records) == 1
    assert sum(r.kind == "tool.completed" for r in records) == 1
    assert "private raw error" not in str(records)
    assert (await binding.store.remaining(scope)).usage.total_tool_calls == 1


@pytest.mark.asyncio
async def test_stale_fence_cannot_write_and_unscoped_manual_transaction_still_works(
    journal: Journal,
):
    engine, _, actor, recorder, scope, lease, _, job_id = journal
    binding, _, _ = await setup(journal)
    operation = await binding.store.reserve(
        scope, UsageReservation(kind="TOOL", interval_id=binding.interval_id), lease=lease
    )
    context = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease),
        operation_id=operation.reservation.operation_id,
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "lease_until=clock_timestamp()-interval '1 second' WHERE "
                "organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    await recorder.claim(scope, job_id=job_id, worker_id="new", lease_seconds=60)
    service = SyntheticService(journal)
    with pytest.raises(SessionError, match="LEASE_LOST"):
        async with service.transactions(actor) as tx:
            await assert_execution_fence(tx.session, context)
            await tx.session.execute(
                text("UPDATE assistant_jobs SET attempt_count=1 WHERE id=:id"), {"id": job_id}
            )
            await tx.commit()
    async with service.transactions(actor) as tx:
        await assert_execution_fence(tx.session)  # no candidate context: manual path
        await tx.session.execute(
            text("UPDATE assistant_jobs SET attempt_count=1 WHERE id=:id"), {"id": job_id}
        )
        await tx.commit()


@pytest.mark.asyncio
async def test_atomic_dispatch_claim_has_one_winner(journal: Journal):
    _, _, _, recorder, scope, lease, _, _ = journal
    payload = await recorder.put_payload(
        scope, {}, classification="RAW_CONTEXT", kind="TOOL_ARGUMENTS", lease=lease
    )
    descriptor = InvocationDescriptor(
        invocation_id=uuid4(),
        call_id="race",
        tool_reference="read_project_context@1",
        tool_version="1.0.0",
        arguments_fingerprint=payload.fingerprint,
    )
    await recorder.append(
        scope,
        EventDraft(
            event_id=uuid4(), kind="tool.requested", invocation=descriptor, payload_ref=payload
        ),
        expected_sequence=0,
        lease=lease,
    )
    receipt = await recorder.get_invocation(scope, descriptor.invocation_id)
    assert receipt is not None
    values = await asyncio.gather(
        *(recorder.start_invocation(scope, receipt, lease=lease) for _ in range(2))
    )
    assert sorted(values) == [False, True]


@pytest.mark.asyncio
async def test_completed_result_replay_after_new_lease_does_not_reset_budget(journal: Journal):
    engine, _, _, recorder, scope, lease, _, job_id = journal
    binding, _, pipeline = await setup(journal)
    call = ToolCall(call_id="resumed", name="read_project_context", arguments={"query": "status"})
    with candidate_budget(binding):
        first = await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
        assert first.kind == "COMPLETED"
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "lease_until=clock_timestamp()-interval '1 second' WHERE "
                "organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    fresh = await recorder.claim(scope, job_id=job_id, worker_id="resume", lease_seconds=60)
    interval = await binding.store.reserve(scope, UsageReservation(kind="INTERVAL"), lease=fresh)
    resumed = CandidateBudget(
        scope=scope, lease=fresh, store=binding.store, interval_id=interval.reservation.operation_id
    )
    with candidate_budget(resumed):
        result = await pipeline.execute(scope, call, admitted=Catalog(), lease=fresh)
    assert result == first
    assert (await binding.store.remaining(scope)).usage.total_tool_calls == 1


@pytest.mark.asyncio
async def test_transaction_holds_fence_until_commit_and_expiry_rolls_back(journal: Journal):
    engine, _, actor, _, scope, lease, _, job_id = journal
    binding, _, _ = await setup(journal)
    tool = await binding.store.reserve(
        scope, UsageReservation(kind="TOOL", interval_id=binding.interval_id), lease=lease
    )
    context = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease),
        operation_id=tool.reservation.operation_id,
    )
    service = SyntheticService(journal)
    async with service.transactions(actor) as tx:
        await assert_execution_fence(tx.session, context)

        async def try_takeover():
            async with engine.begin() as conn:
                await conn.execute(text("SET LOCAL lock_timeout='80ms'"))
                await conn.execute(
                    text(
                        "UPDATE orchestration_runtime_states SET lease_owner='race' WHERE "
                        "organization_id=:org"
                    ),
                    {"org": scope.organization_id},
                )

        from sqlalchemy.exc import DBAPIError

        with pytest.raises(DBAPIError, match="lock timeout"):
            await try_takeover()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "lease_until=clock_timestamp()+interval '120 milliseconds' WHERE "
                "organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    with pytest.raises(SessionError, match="LEASE_LOST"):
        async with service.transactions(actor) as tx:
            async with fenced_effect(tx.session, context):
                await tx.session.execute(
                    text("UPDATE assistant_jobs SET attempt_count=1 WHERE id=:id"), {"id": job_id}
                )
                await asyncio.sleep(0.16)
            await tx.commit()
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT attempt_count FROM assistant_jobs WHERE id=:id"), {"id": job_id}
            )
            == 0
        )


@pytest.mark.asyncio
async def test_cross_actor_and_cross_tenant_fence_rejected(journal: Journal):
    from dataclasses import replace

    _, _, actor, _, scope, lease, _, _ = journal
    binding, _, _ = await setup(journal)
    tool = await binding.store.reserve(
        scope, UsageReservation(kind="TOOL", interval_id=binding.interval_id), lease=lease
    )
    service = SyntheticService(journal)
    base = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease),
        operation_id=tool.reservation.operation_id,
    )
    for forged in (
        replace(base, scope=replace(base.scope, membership_id=uuid4())),
        replace(base, scope=replace(base.scope, organization_id=uuid4())),
    ):
        with pytest.raises(SessionError, match="SESSION_NOT_FOUND"):
            async with service.transactions(actor) as tx:
                await assert_execution_fence(tx.session, forged)


@pytest.mark.asyncio
async def test_child_fence_uses_workflow_lease_and_allocation_while_parent_suspended(
    journal: Journal,
):
    from datetime import UTC, datetime, timedelta

    from app.core.execution_fence import WorkflowJobFence
    from work_management_ai.runtime.run_budget import ChildAllocation
    from work_management_ai.runtime.tool_loop_contracts import ToolLoopUsage

    engine, sessions, actor, _, scope, lease, _, job_id = journal
    budget = PostgreSQLRunBudget(sessions, actor)
    allocation = ChildAllocation(
        allocation_id=uuid4(),
        allowance=ToolLoopUsage(total_tool_calls=2, segments=1, active_seconds=180),
    )
    await budget.allocate_child(scope, allocation, lease=lease)
    interval = await budget.reserve(
        scope,
        UsageReservation(kind="INTERVAL", allocation_id=allocation.allocation_id),
        lease=lease,
    )
    tool = await budget.reserve(
        scope,
        UsageReservation(
            kind="TOOL",
            allocation_id=allocation.allocation_id,
            interval_id=interval.reservation.operation_id,
        ),
        lease=lease,
    )
    workflow_id, child_id = uuid4(), uuid4()
    until = datetime.now(UTC) + timedelta(seconds=50)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO workflow_runs "
                "(id,organization_id,requested_by_membership_id,status,workflow_name"
                ",workflow_version,verifier_version,input_goal_text) VALUES "
                "(:id,:org,:member,'RUNNING','synthetic','1','1','synthetic')"
            ),
            {"id": workflow_id, "org": scope.organization_id, "member": scope.membership_id},
        )
        await conn.execute(
            text(
                "INSERT INTO workflow_jobs "
                "(id,organization_id,workflow_run_id,job_type,status,payload,locked_"
                "by_worker_id,lease_until,attempt_count) VALUES "
                "(:id,:org,:workflow,'synthetic','RUNNING','{}','child-worker',:unti"
                "l,1)"
            ),
            {"id": child_id, "org": scope.organization_id, "workflow": workflow_id, "until": until},
        )
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "lease_id=NULL,lease_owner=NULL,lease_job_id=NULL,lease_until=NULL "
                "WHERE organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    child = WorkflowJobFence(
        job_id=child_id,
        workflow_run_id=workflow_id,
        worker_id="child-worker",
        attempt_count=1,
        lease_until=until,
        allocation_id=allocation.allocation_id,
    )
    context = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease),
        operation_id=tool.reservation.operation_id,
        workflow=child,
    )
    service = SyntheticService(journal)
    async with service.transactions(actor) as tx:
        async with fenced_effect(tx.session, context):
            await tx.session.execute(
                text("UPDATE assistant_jobs SET attempt_count=1 WHERE id=:id"), {"id": job_id}
            )
        await tx.commit()
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE workflow_jobs SET attempt_count=2 WHERE id=:id"), {"id": child_id}
        )
    with pytest.raises(SessionError, match="WORKFLOW_LEASE_LOST"):
        async with service.transactions(actor) as tx:
            await assert_execution_fence(tx.session, context)


@pytest.mark.asyncio
async def test_membership_revoked_after_commit_cannot_replay_private_result(journal: Journal):
    engine, _, actor, _, scope, lease, _, _ = journal
    binding, _, pipeline = await setup(journal)
    call = ToolCall(call_id="revoked", name="read_project_context", arguments={"query": "status"})
    with candidate_budget(binding):
        assert (
            await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
        ).kind == "COMPLETED"
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE memberships SET is_active=false WHERE id=:id"),
                {"id": actor.membership_id},
            )
        denied = await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
        assert denied.kind == "REJECTED" and denied.stop_segment
        assert "committed" not in denied.model_dump_json()


@pytest.mark.asyncio
async def test_concurrent_denial_cannot_close_started_invocation(journal: Journal):
    _, _, _, recorder, scope, lease, _, _ = journal
    binding, _, pipeline = await setup(journal)
    entered, release = asyncio.Event(), asyncio.Event()
    original = pipeline.backends[Catalog.tool.reference]

    class PausedBackend:
        async def execute(
            self, context: ToolInvocationContext, arguments: BaseModel
        ) -> BackendToolResult:
            entered.set()
            await release.wait()
            return await original.execute(context, arguments)

        async def reconcile(
            self, context: ToolInvocationContext, arguments: BaseModel
        ) -> BackendToolResult | None:
            return await original.reconcile(context, arguments)

    pipeline.backends[Catalog.tool.reference] = PausedBackend()
    denied_policy = Policy()
    denied_policy.deny = True
    other = ToolPipeline(store=recorder, policy=denied_policy, backends=pipeline.backends)
    call = ToolCall(call_id="shared", name="read_project_context", arguments={"query": "status"})
    with candidate_budget(binding):
        first = asyncio.create_task(pipeline.execute(scope, call, admitted=Catalog(), lease=lease))
        await entered.wait()
        try:
            rejected = await other.execute(scope, call, admitted=Catalog(), lease=lease)
            assert rejected.kind == "REJECTED"
        finally:
            release.set()
        result = await first
        assert result.kind == "COMPLETED"
        assert await pipeline.execute(scope, call, admitted=Catalog(), lease=lease) == result


@pytest.mark.asyncio
async def test_crash_before_dispatch_resumes_first_execution_on_new_interval(journal: Journal):
    engine, _, _, recorder, scope, lease, _, job_id = journal
    binding, policy, pipeline = await setup(journal)
    original_guard = policy.guard

    async def interrupted(
        context: ToolInvocationContext, tool: AdmittedTool, evidence: tuple[ContextReference, ...]
    ) -> None:
        raise asyncio.CancelledError

    policy.guard = interrupted
    call = ToolCall(
        call_id="before-dispatch", name="read_project_context", arguments={"query": "status"}
    )
    with candidate_budget(binding), pytest.raises(asyncio.CancelledError):
        await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
    assert [r.kind for r in await recorder.read(scope, after_sequence=0)] == ["tool.requested"]
    policy.guard = original_guard
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET "
                "lease_until=clock_timestamp()-interval '1 second' WHERE "
                "organization_id=:org"
            ),
            {"org": scope.organization_id},
        )
    fresh = await recorder.claim(scope, job_id=job_id, worker_id="resume-first", lease_seconds=60)
    clock = await binding.store.reserve(scope, UsageReservation(kind="INTERVAL"), lease=fresh)
    resumed = CandidateBudget(
        scope=scope, lease=fresh, store=binding.store, interval_id=clock.reservation.operation_id
    )
    with candidate_budget(resumed):
        result = await pipeline.execute(scope, call, admitted=Catalog(), lease=fresh)
    assert result.kind == "COMPLETED"
    assert (await binding.store.remaining(scope)).usage.total_tool_calls == 1


@pytest.mark.asyncio
async def test_expired_active_interval_cannot_recover_with_live_lease(journal: Journal):
    engine, _, _, _, scope, lease, _, _ = journal
    binding, _, pipeline = await setup(journal)
    call = ToolCall(
        call_id="expired-clock", name="read_project_context", arguments={"query": "status"}
    )
    with candidate_budget(binding):
        assert (
            await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
        ).kind == "COMPLETED"
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE orchestration_usage SET "
                    "started_at=clock_timestamp()-interval '181 seconds' WHERE "
                    "operation_id=:id"
                ),
                {"id": binding.interval_id},
            )
        result = await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
    assert result.kind == "REJECTED" and result.stop_segment


@pytest.mark.asyncio
async def test_original_arguments_expiry_between_dispatch_and_effect_blocks_write(journal: Journal):
    engine, _, _, recorder, scope, lease, _, job_id = journal
    binding, _, pipeline = await setup(journal)
    original = pipeline.backends[Catalog.tool.reference]
    expired = asyncio.Event()
    original_put = recorder.put_payload

    async def short_ttl(
        scope: RunScope,
        body: JsonValue,
        *,
        classification: Literal["RAW_CONTEXT"],
        kind: PayloadKind,
        lease: RunLease,
        ttl_seconds: int | None = None,
    ) -> PayloadReference:
        return await original_put(
            scope,
            body,
            classification=classification,
            kind=kind,
            lease=lease,
            ttl_seconds=2 if kind == "TOOL_ARGUMENTS" else ttl_seconds,
        )

    recorder.put_payload = short_ttl

    class ExpiredBackend:
        async def execute(
            self, context: ToolInvocationContext, arguments: BaseModel
        ) -> BackendToolResult:
            receipt = await recorder.get_invocation(scope, context.invocation_id)
            assert receipt is not None
            await asyncio.sleep(2.05)  # actual TTL; immutable source metadata remains untouched
            expired.set()
            return await original.execute(context, arguments)

        async def reconcile(
            self, context: ToolInvocationContext, arguments: BaseModel
        ) -> BackendToolResult | None:
            return await original.reconcile(context, arguments)

    pipeline.backends[Catalog.tool.reference] = ExpiredBackend()
    call = ToolCall(
        call_id="expiry-race", name="read_project_context", arguments={"query": "status"}
    )
    with candidate_budget(binding):
        result = await pipeline.execute(scope, call, admitted=Catalog(), lease=lease)
    assert expired.is_set()
    assert result.kind == "REJECTED" and result.stop_segment
    async with engine.connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT attempt_count FROM assistant_jobs WHERE id=:id"), {"id": job_id}
            )
            == 0
        )


@pytest.mark.asyncio
async def test_recovery_timeout_capped_by_current_segment_remaining_time(journal: Journal):
    engine, _, _, _, scope, lease, _, _ = journal
    binding, _, _ = await setup(journal)
    operation = await binding.store.reserve(
        scope, UsageReservation(kind="TOOL", interval_id=binding.interval_id), lease=lease
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_usage SET "
                "started_at=clock_timestamp()-interval '179 seconds' WHERE "
                "operation_id=:id"
            ),
            {"id": binding.interval_id},
        )
    recovered = await binding.store.recover_tool(
        scope,
        operation.reservation.operation_id,
        lease=lease,
        interval_id=binding.interval_id,
        allocation_id=None,
    )
    assert recovered.replayed and 0 < recovered.seconds_remaining <= 1
    assert (await binding.store.remaining(scope)).usage.total_tool_calls == 1
