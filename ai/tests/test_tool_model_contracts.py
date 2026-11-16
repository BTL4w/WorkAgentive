"""Native tool boundaries reject ambiguous execution and preserve replay pairing."""

from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.tool_contracts import (
    ToolCall,
    ToolItem,
    ToolModelRequest,
    ToolModelResponse,
    ToolSchema,
    validate_history,
)
from work_management_ai.runtime.contracts import JsonValue
from work_management_ai.runtime.tool_loop_contracts import (
    RunScope,
    ToolLoopCheckpoint,
    ToolLoopLimits,
)


class Answer(BaseModel):
    text: str


def request() -> ToolModelRequest[Answer]:
    return ToolModelRequest[Answer](
        invocation_key="turn.1", items=(), tools=(), response_schema=Answer
    )


def test_one_call_or_typed_final() -> None:
    call = ToolCall(call_id="call1", name="read_project_context", arguments={})
    assert ToolModelResponse[Answer](tool_call=call, model_ref="mock:v1").finish == "TOOL_CALL"
    assert (
        ToolModelResponse[Answer](final=Answer(text="hello"), model_ref="mock:v1").finish == "FINAL"
    )
    for value in (
        {},
        {"tool_call": call, "final": Answer(text="wrong")},
        {"tool_calls": [call, call]},
    ):
        with pytest.raises(ValidationError):
            ToolModelResponse[Answer].model_validate({"model_ref": "mock:v1", **value})


def test_commentary_is_not_final() -> None:
    with pytest.raises(ValidationError):
        ToolModelResponse[Answer](commentary="working", model_ref="mock:v1")


def test_wrong_call_pair_or_duplicate_id_rejected() -> None:
    call = ToolCall(call_id="c1", name="read_project_context", arguments={})
    with pytest.raises(ValueError):
        validate_history((ToolItem(kind="tool_result", call_id="unknown", content="{}"),))
    with pytest.raises(ValueError):
        validate_history(
            (ToolItem(kind="tool_call", call=call), ToolItem(kind="tool_call", call=call))
        )
    validate_history(
        (
            ToolItem(kind="tool_call", call=call),
            ToolItem(kind="tool_result", call_id="c1", content="{}"),
        )
    )


def test_scope_and_checkpoint_no_authority_fields() -> None:
    scope = RunScope(
        organization_id=uuid4(),
        membership_id=uuid4(),
        conversation_id=uuid4(),
        turn_id=uuid4(),
        orchestration_run_id=uuid4(),
    )
    checkpoint = ToolLoopCheckpoint(
        scope=scope, driver_version="2.0.0", prompt_version="v1", catalog_fingerprint="abc"
    )
    assert checkpoint.sequence == 0
    with pytest.raises(ValidationError):
        ToolLoopCheckpoint.model_validate(checkpoint.model_dump() | {"approved": True})
    with pytest.raises(ValidationError):
        ToolLoopCheckpoint.model_validate(checkpoint.model_dump() | {"raw_prompt": "secret"})
    limits = ToolLoopLimits()
    assert (limits.root_model_requests, limits.total_model_requests, limits.total_tool_calls) == (
        8,
        24,
        12,
    )
    with pytest.raises(ValidationError):
        ToolLoopLimits(root_model_requests=9)


def test_wire_alias_collision_rejected() -> None:
    schema = ToolSchema(reference="automation.preview@1", description="Preview", parameters={})
    assert schema.wire_name == "automation__preview"
    with pytest.raises(ValidationError):
        ToolModelRequest[Answer](
            invocation_key="x",
            items=(),
            tools=(
                schema,
                ToolSchema(reference="automation__preview@1", description="Other", parameters={}),
            ),
            response_schema=Answer,
        )


@pytest.mark.asyncio
async def test_native_mock_validates_final_and_isolates_sequences() -> None:
    fixtures = {"turn.1": ({"final": {"text": "hello"}},)}
    first = MockModelGateway(fixtures={}, tool_sequence_fixtures=fixtures)
    second = MockModelGateway(fixtures={}, tool_sequence_fixtures=fixtures)
    assert (await first.generate_step(request())).final == Answer(text="hello")
    assert (await second.generate_step(request())).final == Answer(text="hello")
    assert first.step_requests[0].invocation_key == "turn.1"
    from work_management_ai.model_gateway.errors import ModelInvalidOutputError

    bad = MockModelGateway(fixtures={"turn.1": {"final": {"wrong": "field"}}})
    with pytest.raises(ModelInvalidOutputError):
        await bad.generate_step(request())


def test_history_requires_complete_ordered_pairs() -> None:
    call = ToolCall(call_id="c1", name="read_project_context", arguments={})
    start = ToolItem(kind="tool_call", call=call)
    result = ToolItem(kind="tool_result", call_id="c1", content={"version": 7})
    for items in ((start,), (result, start), (start, result, result)):
        with pytest.raises(ValueError):
            validate_history(items)
    with pytest.raises(ValidationError):
        ToolItem(kind="tool_result", call=call, call_id="c1", content={})
    validate_history((start, result, ToolItem(kind="message", role="user", content="Tiếp tục")))


@pytest.mark.asyncio
async def test_mock_captures_observation_and_detects_unconsumed_steps() -> None:
    from work_management_ai.model_gateway.errors import ModelUnavailableError

    call: dict[str, object] = {
        "tool_call": {"call_id": "c1", "name": "read_project_context", "arguments": {}}
    }
    script: dict[str, tuple[object, ...]] = {
        "run-a": (call, {"final": {"text": "20 people"}}),
        "run-b": ({"final": {"text": "Hi"}},),
    }
    gateway = MockModelGateway(fixtures={}, tool_sequence_fixtures=script)
    schema = ToolSchema(reference="read_project_context@1", description="Read", parameters={})
    first = ToolModelRequest[Answer](
        invocation_key="run-a", items=(), tools=(schema,), response_schema=Answer
    )
    with pytest.raises(AssertionError):
        gateway.assert_complete()
    response = await gateway.generate_step(first)
    assert response.tool_call is not None
    observation: dict[str, JsonValue] = {"people": 20}
    followup = ToolModelRequest[Answer](
        invocation_key="run-a",
        items=(
            ToolItem(kind="tool_call", call=response.tool_call),
            ToolItem(kind="tool_result", call_id="c1", content=observation),
        ),
        tools=(schema,),
        response_schema=Answer,
    )
    assert (await gateway.generate_step(followup)).final == Answer(text="20 people")
    assert (
        await gateway.generate_step(first.model_copy(update={"invocation_key": "run-b"}))
    ).final == Answer(text="Hi")
    gateway.assert_complete()
    with pytest.raises(ModelUnavailableError):
        await gateway.generate_step(first)
    assert gateway.step_requests[1].items[1].content == {"people": 20}
    observation["people"] = 99
    captured = gateway.step_requests
    content = captured[1].items[1].content
    assert isinstance(content, dict)
    content["people"] = 100
    assert gateway.step_requests[1].items[1].content == {"people": 20}
    assert script["run-a"][0] == call


INVALID_RESPONSES: list[dict[str, object]] = [
    {"commentary": "working"},
    {"tool_calls": [{"call_id": "c1", "name": "read_project_context", "arguments": {}}]},
    {"tool_call": {"call_id": "c1", "name": "unknown", "arguments": {}}},
    {"tool_call": {"call_id": "c1", "name": "read_project_context", "arguments": {"wrong": 1}}},
    {"final": {"text": "ok"}, "finish": "TOOL_CALL"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", INVALID_RESPONSES)
async def test_invalid_output_is_not_a_dispatchable_response(fixture: dict[str, object]) -> None:
    from work_management_ai.model_gateway.errors import ModelInvalidOutputError

    gateway = MockModelGateway(fixtures={"turn.1": fixture})
    from pydantic import ConfigDict

    class NoArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")

    schema = ToolSchema(
        reference="read_project_context@1", description="Read", input_contract=NoArguments
    )
    with pytest.raises(ModelInvalidOutputError):
        await gateway.generate_step(request().model_copy(update={"tools": (schema,)}))


def test_public_v2_preserves_statuses_and_internal_suspend_is_distinct() -> None:
    from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
    from work_management_ai.agents.orchestrator.tool_loop_contracts import (
        OrchestratorFinalResponse,
        OrchestratorOutputV2,
        PublishedResponse,
        SuspendedWorkflow,
    )
    from work_management_ai.runtime.contracts import TextResponseBlock
    from work_management_ai.runtime.tool_loop_contracts import WaitReference

    final = OrchestratorFinalResponse(
        status=OrchestratorStatus.COMPLETED,
        blocks=(TextResponseBlock(text="Xin chào"),),
        stop_reason="ANSWER_READY",
    )
    output = OrchestratorOutputV2(**final.model_dump())
    assert output.schema_version == "2.0"
    assert PublishedResponse(output=output).output.blocks == final.blocks
    wait = WaitReference(
        wait_id=uuid4(), scope=scope(), kind="WORKFLOW_RESULT", source_invocation_id=uuid4()
    )
    suspended = SuspendedWorkflow(wait_ref=wait, public_activity=(), log_cursor=2)
    assert "status" not in suspended.model_dump()
    with pytest.raises(ValidationError):
        PublishedResponse(output=output.model_copy(update={"status": "AWAITING_INPUT"}))
    with pytest.raises(ValidationError):
        SuspendedWorkflow(
            wait_ref=wait.model_copy(update={"kind": "HUMAN_DECISION"}),
            public_activity=(),
            log_cursor=2,
        )


def scope() -> RunScope:
    return RunScope(
        organization_id=uuid4(),
        membership_id=uuid4(),
        conversation_id=uuid4(),
        turn_id=uuid4(),
        orchestration_run_id=uuid4(),
    )


def test_v1_message_contract_is_unchanged() -> None:
    from work_management_ai.model_gateway.contracts import ModelMessage

    assert ModelMessage(role="user", content="hi").images == ()
    with pytest.raises(TypeError):
        ModelMessage(role="tool", content="{}", call_id="c1")  # type: ignore[call-arg, arg-type]


def test_checkpoint_roundtrip_preserves_usage_versions_and_tenant_binding() -> None:
    from datetime import UTC, datetime, timedelta

    from work_management_ai.runtime.tool_loop_contracts import PayloadReference, ToolLoopUsage

    run_scope = scope()
    payload = PayloadReference(
        payload_id=uuid4(),
        organization_id=run_scope.organization_id,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        fingerprint="a" * 64,
    )
    checkpoint = ToolLoopCheckpoint(
        scope=run_scope,
        driver_version="2.0.0",
        prompt_version="v1",
        catalog_fingerprint="abc",
        objective_ref=payload,
        usage=ToolLoopUsage(root_model_requests=2, total_model_requests=3, total_tool_calls=1),
    )
    restored = ToolLoopCheckpoint.model_validate_json(checkpoint.model_dump_json())
    assert restored == checkpoint
    updates: tuple[dict[str, object], ...] = (
        {"role": "ADMIN"},
        {"approved": True},
        {"raw_context": {}},
        {"thought": "private"},
    )
    for update in updates:
        with pytest.raises(ValidationError):
            ToolLoopCheckpoint.model_validate(checkpoint.model_dump() | update)
    with pytest.raises(ValidationError):
        ToolLoopCheckpoint.model_validate(checkpoint.model_dump() | {"scope": scope()})
    with pytest.raises(ValidationError):
        ToolLoopCheckpoint.model_validate(
            checkpoint.model_dump()
            | {"usage": {"root_model_requests": 9, "total_model_requests": 9}}
        )


@pytest.mark.asyncio
async def test_native_and_legacy_script_queues_are_independent() -> None:
    from work_management_ai.model_gateway.contracts import StructuredModelRequest

    gateway = MockModelGateway(
        fixtures={},
        sequence_fixtures={"turn.1": ({"text": "legacy"},)},
        tool_sequence_fixtures={"turn.1": ({"final": {"text": "native"}},)},
    )
    legacy = StructuredModelRequest(
        invocation_key="turn.1", messages=(), output_schema=Answer, timeout_seconds=5
    )
    assert (await gateway.generate_step(request())).final == Answer(text="native")
    assert (await gateway.generate_structured(legacy)).parsed == Answer(text="legacy")
    gateway.assert_complete()


@pytest.mark.asyncio
async def test_shared_tool_boundaries_with_scoped_fakes() -> None:
    from datetime import UTC, datetime, timedelta

    from runtime.tool_loop_fakes import (
        FakeCheckpointStore,
        FakeClock,
        FakeToolBackend,
        ScriptedToolGateway,
    )

    from work_management_ai.runtime.contracts import RiskLevel
    from work_management_ai.runtime.tool_loop_contracts import (
        AdmittedTool,
        BackendToolResult,
        RunLease,
        ToolInvocationContext,
    )

    clock = FakeClock(datetime(2026, 11, 16, tzinfo=UTC))
    run_scope = scope()
    lease = RunLease(
        scope=run_scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="test",
        fencing_token=1,
        expires_at=clock.now() + timedelta(seconds=60),
    )
    checkpoint = ToolLoopCheckpoint(
        scope=run_scope, driver_version="2.0.0", prompt_version="v1", catalog_fingerprint="abc"
    )
    store = FakeCheckpointStore(clock)
    store.save(checkpoint, lease=lease)
    assert store.load(run_scope) == checkpoint
    assert store.load(scope()) is None
    with pytest.raises(ValueError):
        store.save(checkpoint, lease=lease.model_copy(update={"scope": scope()}))
    arguments = Answer(text="project-a")
    context = ToolInvocationContext(
        scope=run_scope,
        lease=lease,
        invocation_id=uuid4(),
        call_id="c1",
        tool_reference="read_project_context@1",
        tool_version="1.0.0",
        arguments_fingerprint="a" * 64,
        canonical_arguments=arguments,
        policy_version="1",
        receipt_id=uuid4(),
        budget_allocation_id=uuid4(),
        limits=ToolLoopLimits(),
    )
    backend = FakeToolBackend(BackendToolResult(output=Answer(text="20 people")))
    result = await backend.execute(context, arguments)
    assert isinstance(result.output, Answer)
    assert result.model_dump()["output"] == {"text": "20 people"}
    assert backend.calls[0][0].scope == run_scope
    admitted = AdmittedTool(
        schema=ToolSchema(
            reference="read_project_context@1", description="Read", input_contract=Answer
        ),
        input_contract=Answer,
        output_contract=Answer,
        version="1.0.0",
        fingerprint="abc",
        risk=RiskLevel.READ_ONLY,
    )
    assert admitted.reference == "read_project_context@1"
    clock.advance(61)
    with pytest.raises(ValueError):
        store.save(checkpoint, lease=lease)
    assert ScriptedToolGateway is MockModelGateway


@pytest.mark.parametrize(
    "updates",
    [
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 0},
        {"max_output_tokens": 8001},
        {"max_output_tokens": True},
    ],
)
def test_invalid_native_request_limits_fail_before_model(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToolModelRequest[Answer].model_validate(
            {
                "invocation_key": "x",
                "items": (),
                "tools": (),
                "response_schema": Answer,
                **updates,
            }
        )


@pytest.mark.asyncio
async def test_mock_rejects_reused_call_id_and_records_normalized_usage() -> None:
    from work_management_ai.model_gateway.contracts import ModelUsage
    from work_management_ai.model_gateway.errors import ModelInvalidOutputError

    call = ToolCall(call_id="c1", name="read_project_context", arguments={})
    gateway = MockModelGateway(
        fixtures={"turn.1": {"tool_call": call.model_dump()}},
        usage_fixtures={"turn.1": ModelUsage(10, 2)},
    )
    schema = ToolSchema(reference="read_project_context@1", description="Read")
    req = request().model_copy(update={"tools": (schema,)})
    assert (await gateway.generate_step(req)).usage == ModelUsage(10, 2)
    with pytest.raises(ModelInvalidOutputError) as caught:
        await gateway.generate_step(
            req.model_copy(
                update={
                    "items": (
                        ToolItem(kind="tool_call", call=call),
                        ToolItem(kind="tool_result", call_id="c1", content={}),
                    )
                }
            )
        )
    assert caught.value.usage == ModelUsage(10, 2)
