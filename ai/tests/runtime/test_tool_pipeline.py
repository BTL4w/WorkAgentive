"""Fail closed before dispatch, and reconcile ambiguous committed effects."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from work_management_ai.model_gateway.tool_contracts import ToolCall, ToolSchema
from work_management_ai.runtime.contracts import ContextReference, JsonValue
from work_management_ai.runtime.run_budget import CandidateBudget, candidate_budget
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationReceipt,
    PayloadKind,
    SessionRecord,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    BackendToolResult,
    PayloadReference,
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopCheckpoint,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import PreparedTool, ToolPipeline, ToolPolicyError

from .test_tool_loop_budget import Budget
from .test_tool_loop_budget import binding as binding


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str


class Catalog:
    def __init__(self):
        self.tool = AdmittedTool(
            schema=ToolSchema(
                reference="read_project_context@1", description="Read", input_contract=Input
            ),
            input_contract=Input,
            output_contract=Output,
            version="1.0.0",
            fingerprint="tool-sha",
        )
        self.closed = False

    @property
    def schemas(self):
        return (self.tool.schema,)

    @property
    def fingerprint(self):
        return "catalog"

    def resolve(self, wire_name: str) -> AdmittedTool:
        if self.closed or wire_name != self.tool.schema.wire_name:
            raise KeyError(wire_name)
        return self.tool

    def close(self):
        self.closed = True


class Log:
    def __init__(self):
        self.events: list[SessionRecord] = []
        self.payloads: dict[UUID, JsonValue] = {}
        self.expired = False

    async def initialize(self, scope: RunScope, state: ToolLoopCheckpoint):
        pass

    async def checkpoint(self, scope: RunScope, state: ToolLoopCheckpoint, *, lease: RunLease):
        pass

    async def load_checkpoint(self, scope: RunScope) -> ToolLoopCheckpoint | None:
        return None

    async def claim(
        self, scope: RunScope, *, job_id: UUID, worker_id: str, lease_seconds: int
    ) -> RunLease:
        raise NotImplementedError

    async def renew(self, scope: RunScope, lease: RunLease, *, lease_seconds: int) -> RunLease:
        return lease

    async def read(self, scope: RunScope, *, after_sequence: int):
        return tuple(v for v in self.events if v.sequence > after_sequence)

    async def append(
        self, scope: RunScope, event: EventDraft, *, expected_sequence: int, lease: RunLease
    ):
        old = next((v for v in self.events if v.event_id == event.event_id), None)
        if old:
            return old
        assert expected_sequence == len(self.events)
        value = SessionRecord(
            **event.model_dump(),
            scope=scope,
            sequence=len(self.events) + 1,
            occurred_at=datetime.now(UTC),
        )
        self.events.append(value)
        return value

    async def start_invocation(
        self, scope: RunScope, receipt: InvocationReceipt, *, lease: RunLease
    ) -> bool:
        if any(
            v.kind == "tool.started" and v.invocation == receipt.descriptor for v in self.events
        ):
            return False
        await self.append(
            scope,
            EventDraft(event_id=uuid4(), kind="tool.started", invocation=receipt.descriptor),
            expected_sequence=len(self.events),
            lease=lease,
        )
        return True

    async def put_payload(
        self,
        scope: RunScope,
        body: JsonValue,
        *,
        classification: Literal["RAW_CONTEXT"],
        kind: PayloadKind,
        lease: RunLease,
        ttl_seconds: int | None = None,
    ):
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ref = PayloadReference(
            payload_id=uuid4(),
            organization_id=scope.organization_id,
            fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        self.payloads[ref.payload_id] = json.loads(encoded)
        return ref

    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> JsonValue:
        if self.expired:
            raise ValueError("CONTEXT_EXPIRED")
        return self.payloads[reference.payload_id]

    async def get_invocation(
        self, scope: RunScope, invocation_id: UUID
    ) -> InvocationReceipt | None:
        events = [
            v for v in self.events if v.invocation and v.invocation.invocation_id == invocation_id
        ]
        requested = next((v for v in events if v.kind == "tool.requested"), None)
        if requested is None:
            return None
        assert requested.invocation and requested.payload_ref
        last = next((v for v in reversed(events) if v.settlement), None)
        return InvocationReceipt(
            scope=scope,
            descriptor=requested.invocation,
            requested_event_id=requested.event_id,
            requested_sequence=requested.sequence,
            arguments_ref=requested.payload_ref,
            settlement=last.settlement if last else None,
        )

    async def settle_invocation(
        self,
        scope: RunScope,
        receipt: InvocationReceipt,
        value: SettledInvocation,
        *,
        lease: RunLease,
    ):
        old = await self.get_invocation(scope, receipt.descriptor.invocation_id)
        if old and old.settlement:
            assert old.settlement == value
        return await self.append(
            scope,
            EventDraft(
                event_id=uuid4(),
                kind="run.waiting"
                if value.status == "DEFERRED"
                else "tool.failed"
                if value.status == "FAILED"
                else "tool.completed",
                invocation=receipt.descriptor,
                payload_ref=value.payload_ref,
                settlement=value,
            ),
            expected_sequence=len(self.events),
            lease=lease,
        )


class Policy:
    def __init__(self):
        self.stages: list[str] = []
        self.deny = self.mutate = self.post_fail = False
        self.audit: list[str] = []
        self.version = "policy-1"

    async def pre_execute(
        self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
    ) -> PreparedTool:
        self.stages.append("pre")
        return PreparedTool(
            arguments=Input.model_validate(arguments.model_dump()), policy_version=self.version
        )

    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None:
        self.stages.append("guards")
        if self.deny:
            raise ToolPolicyError("PERMISSION_DENIED", stop_segment=True)
        if self.mutate:
            object.__setattr__(context.canonical_arguments, "query", "switched")

    async def post_execute(self, context: ToolInvocationContext, result: BackendToolResult) -> None:
        self.stages.append("post")
        if self.post_fail:
            raise RuntimeError("private provider details")

    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None:
        self.audit.append(safe_code)


class Backend:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.effects = 0
        self.saved: BackendToolResult | None = None
        self.mode = "success"

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        return self.saved

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        self.policy.stages.append("execute")
        self.effects += 1
        if self.mode == "timeout":
            raise TimeoutError("private timeout")
        if self.mode == "cancel":
            raise asyncio.CancelledError
        if self.mode == "wait":
            self.saved = BackendToolResult(
                wait_ref=WaitReference(
                    wait_id=uuid4(),
                    scope=context.scope,
                    kind="WORKFLOW_RESULT",
                    source_invocation_id=context.invocation_id,
                )
            )
        else:
            self.saved = BackendToolResult(output=Output(text="verified"))
        return self.saved


class Fixture:
    def __init__(self):
        self.log, self.catalog, self.policy = Log(), Catalog(), Policy()
        self.backend = Backend(self.policy)
        self.pipeline = ToolPipeline(
            store=self.log, policy=self.policy, backends={self.catalog.tool.reference: self.backend}
        )
        self.call = ToolCall(
            call_id="call-1", name="read_project_context", arguments={"query": "status"}
        )


@pytest.mark.asyncio
async def test_stage_order_and_canonical_replay(binding: tuple[CandidateBudget, Budget]):
    ctx, _ = binding
    f = Fixture()
    with candidate_budget(ctx):
        first = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert first.kind == "COMPLETED"
        assert f.policy.stages == ["pre", "guards", "execute", "post", "guards"]
        replay = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert replay == first and f.backend.effects == 1
    assert [v.kind for v in f.log.events] == [
        "tool.requested",
        "tool.authorized",
        "tool.started",
        "tool.completed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,code", [("deny", "PERMISSION_DENIED"), ("mutate", "TOOL_ARGUMENTS_CHANGED")]
)
async def test_deny_and_argument_mutation_never_dispatch(
    binding: tuple[CandidateBudget, Budget], mode: str, code: str
):
    ctx, _ = binding
    f = Fixture()
    setattr(f.policy, mode, True)
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.safe_code == code and result.stop_segment
    assert f.backend.effects == 0 and code in f.policy.audit
    assert f.log.events[-1].settlement is not None


@pytest.mark.asyncio
async def test_cached_result_rechecks_permission_and_expiry(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()
    with candidate_budget(ctx):
        await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        f.policy.deny = True
        denied = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert denied.kind == "REJECTED" and denied.safe_code == "PERMISSION_DENIED"
        f.policy.deny = False
        f.log.expired = True
        expired = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert expired.kind == "REJECTED" and expired.stop_segment and f.backend.effects == 1


@pytest.mark.asyncio
async def test_post_failure_reconciles_committed_receipt_without_second_effect(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()
    f.policy.post_fail = True
    with candidate_budget(ctx):
        failed = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert failed.kind == "REJECTED" and failed.safe_code == "TOOL_POSTCONDITION_FAILED"
        f.policy.post_fail = False
        replay = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert replay.kind == "COMPLETED" and f.backend.effects == 1
    assert "private provider details" not in str(f.log.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
async def test_unknown_effect_never_blindly_retries(
    binding: tuple[CandidateBudget, Budget], mode: str
):
    ctx, _ = binding
    f = Fixture()
    f.backend.mode = mode
    with candidate_budget(ctx):
        if mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        else:
            result = await f.pipeline.execute(
                ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease
            )
            assert result.kind == "REJECTED" and result.stop_segment
        replay = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert replay.kind == "REJECTED" and replay.safe_code == "TOOL_RECONCILIATION_REQUIRED"
    assert f.backend.effects == 1


@pytest.mark.asyncio
async def test_wait_and_payload_conflict(binding: tuple[CandidateBudget, Budget]):
    ctx, _ = binding
    f = Fixture()
    f.backend.mode = "wait"
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert result.kind == "DEFERRED" and f.log.events[-1].kind == "run.waiting"
        conflict = await f.pipeline.execute(
            ctx.scope,
            f.call.model_copy(update={"arguments": {"query": "other"}}),
            admitted=f.catalog,
            lease=ctx.lease,
        )
    assert (
        conflict.kind == "REJECTED"
        and conflict.safe_code == "INVOCATION_CONFLICT"
        and f.backend.effects == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"query": 1},
        {"query": "ok", "organization_id": str(uuid4())},
        {"query": "ok", "approved": True},
    ],
)
async def test_strict_inputs_and_reserved_authority(
    binding: tuple[CandidateBudget, Budget], arguments: dict[str, JsonValue]
):
    ctx, _ = binding
    f = Fixture()
    with candidate_budget(ctx):
        result = await f.pipeline.execute(
            ctx.scope,
            f.call.model_copy(update={"arguments": arguments}),
            admitted=f.catalog,
            lease=ctx.lease,
        )
    assert result.kind == "REJECTED" and f.backend.effects == 0


@pytest.mark.asyncio
async def test_missing_budget_closed_catalog_and_exhaustion(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, budget = binding
    f = Fixture()
    result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.stop_segment
    with candidate_budget(ctx):
        f.catalog.close()
        unknown = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert unknown.kind == "REJECTED" and unknown.safe_code == "UNKNOWN_TOOL"
        f.catalog.closed = False
        budget.reject = True
        denied = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert denied.kind == "REJECTED" and denied.stop_segment
    assert f.backend.effects == 0


@pytest.mark.asyncio
async def test_post_failure_has_durable_safe_diagnostic(binding: tuple[CandidateBudget, Budget]):
    ctx, _ = binding
    f = Fixture()
    f.policy.post_fail = True
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.stop_segment
    assert f.log.events[-1].kind == "tool.denied"
    assert f.log.events[-1].safe_code == "TOOL_POSTCONDITION_FAILED"
    assert all(v.settlement is None for v in f.log.events)


@pytest.mark.asyncio
async def test_replay_does_not_rebind_changed_policy_version(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()
    with candidate_budget(ctx):
        first = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert first.kind == "COMPLETED"
        f.policy.version = "policy-2"
        changed = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert changed.kind == "REJECTED" and changed.safe_code == "INVOCATION_CONFLICT"
    assert f.backend.effects == 1


@pytest.mark.asyncio
async def test_uuid_output_replay_uses_json_contract(binding: tuple[CandidateBudget, Budget]):
    from dataclasses import replace

    class ResourceOutput(BaseModel):
        resource_id: UUID

    ctx, _ = binding
    f = Fixture()
    f.catalog.tool = replace(f.catalog.tool, output_contract=ResourceOutput)
    original_execute = f.backend.execute

    async def resource(context: ToolInvocationContext, arguments: BaseModel) -> BackendToolResult:
        await original_execute(context, arguments)
        return BackendToolResult(
            output=ResourceOutput(resource_id=UUID("00000000-0000-0000-0000-000000000001"))
        )

    f.backend.execute = resource
    with candidate_budget(ctx):
        first = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert first.kind == "COMPLETED"
        replay = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert replay == first and f.backend.effects == 1


@pytest.mark.asyncio
async def test_pre_hook_can_suspend_for_input_without_dispatch(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()

    async def missing(scope: RunScope, tool: AdmittedTool, arguments: BaseModel) -> PreparedTool:
        return PreparedTool(arguments=arguments, policy_version="1", decision="NEEDS_INPUT")

    f.policy.pre_execute = missing
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "DEFERRED" and result.wait_ref.kind == "USER_INPUT"
    assert f.backend.effects == 0
    assert f.log.events[-1].kind == "run.waiting"


@pytest.mark.asyncio
async def test_admission_closed_during_execution_never_publishes_result(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()
    original = f.backend.execute

    async def closed(context: ToolInvocationContext, arguments: BaseModel) -> BackendToolResult:
        result = await original(context, arguments)
        f.catalog.close()
        return result

    f.backend.execute = closed
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.stop_segment
    assert not any(r.kind == "tool.completed" for r in f.log.events)


@pytest.mark.asyncio
async def test_foreign_prepared_references_stop_before_journaling(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()

    async def foreign(scope: RunScope, tool: AdmittedTool, arguments: BaseModel) -> PreparedTool:
        return PreparedTool(
            arguments=arguments,
            policy_version="1",
            resource_versions=(
                ContextReference(
                    reference_id=uuid4(),
                    organization_id=uuid4(),
                    resource_id=uuid4(),
                    resource_type="project",
                    observed_at=datetime.now(UTC),
                ),
            ),
        )

    f.policy.pre_execute = foreign
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert (
        result.kind == "REJECTED" and result.safe_code == "TENANT_MISMATCH" and result.stop_segment
    )
    assert f.backend.effects == 0 and f.log.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "foreign", "hidden"])
async def test_output_evidence_and_hidden_fields_never_publish(
    binding: tuple[CandidateBudget, Budget], mode: str
):
    from dataclasses import replace

    class Details(BaseModel):
        data: dict[str, JsonValue]

    ctx, _ = binding
    f = Fixture()
    if mode == "hidden":
        f.catalog.tool = replace(f.catalog.tool, output_contract=Details)

    async def unsafe(context: ToolInvocationContext, arguments: BaseModel) -> BackendToolResult:
        if mode == "invalid":
            return BackendToolResult(output=Input(query="wrong schema"))
        if mode == "hidden":
            return BackendToolResult(output=Details(data={"system_prompt": "private"}))
        return BackendToolResult(
            output=Output(text="foreign"),
            evidence=(
                ContextReference(
                    reference_id=uuid4(),
                    organization_id=uuid4(),
                    resource_id=uuid4(),
                    resource_type="project",
                    observed_at=datetime.now(UTC),
                ),
            ),
        )

    f.backend.execute = unsafe
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.stop_segment
    assert not any(r.kind == "tool.completed" for r in f.log.events)
    assert "private" not in str(f.log.payloads)


@pytest.mark.asyncio
async def test_timeout_is_enforced_by_pipeline(binding: tuple[CandidateBudget, Budget]):
    from work_management_ai.runtime.run_budget import BudgetReceipt, UsageReservation

    ctx, budget = binding
    f = Fixture()
    reserve = budget.reserve

    async def short(
        scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt:
        return (await reserve(scope, reservation, lease=lease)).model_copy(
            update={"seconds_remaining": 0.01}
        )

    budget.reserve = short

    async def stuck(context: ToolInvocationContext, arguments: BaseModel) -> BackendToolResult:
        f.backend.effects += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    f.backend.execute = stuck
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
        assert (
            result.kind == "REJECTED" and result.safe_code == "TOOL_TIMEOUT" and result.stop_segment
        )
        replay = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert replay.kind == "REJECTED" and f.backend.effects == 1


@pytest.mark.asyncio
async def test_post_dispatch_policy_failure_always_stops_pending_invocation(
    binding: tuple[CandidateBudget, Budget],
):
    ctx, _ = binding
    f = Fixture()

    async def stale(context: ToolInvocationContext, result: BackendToolResult) -> None:
        raise ToolPolicyError("STALE_CONTEXT")

    f.policy.post_execute = stale
    with candidate_budget(ctx):
        result = await f.pipeline.execute(ctx.scope, f.call, admitted=f.catalog, lease=ctx.lease)
    assert result.kind == "REJECTED" and result.stop_segment
    assert all(r.settlement is None for r in f.log.events)
