"""Hub-only dispatch and permission checks on persisted card history and SSE."""

import json
import os
from typing import cast
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.agent_runtime import build_agent_registry
from app.modules.assistant.adapters.risk_tools import RiskBlockProjector
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.event_service import AssistantEventService
from app.modules.assistant.application.service import AssistantService, PlanningSnapshotPort
from app.modules.identity.domain.auth import AuthenticatedActor
from tests.test_daily_update_api_integration import seed_task
from tests.test_evidence_api_integration import Harness, harness
from tests.test_orchestrator_session_log_integration import Journal, journal
from tests.test_phase4_context_permissions import reader
from tests.test_risk_api_integration import setup
from work_management_ai.agents.orchestrator.contracts import (
    ActiveConversationContext,
    OrchestratorInput,
)
from work_management_ai.agents.orchestrator.harness import OrchestratorHarness
from work_management_ai.agents.risk.harness import RiskHarness
from work_management_ai.agents.risk.tests.test_harness import Actors, Tools, handoff
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.runtime.contracts import ActorReference, AgentHandoff, AgentId, AgentResult
from work_management_ai.runtime.policy_guard import PolicyGuard

__all__ = ["harness", "journal"]


def test_risk_registry_limits_inactive_agents_and_peer_handoffs():
    registry, tools = build_agent_registry()
    with pytest.raises(ValueError, match="AGENT_PHASE_INACTIVE"):
        registry.resolve(AgentId.RISK, "1.0.0", 3)
    manifest = registry.resolve(AgentId.RISK, "1.0.0", 4).manifest
    assert manifest.runtime.max_iterations == 4
    assert manifest.runtime.max_tool_calls == 4
    assert manifest.runtime.max_model_attempts == 1
    assert manifest.runtime.timeout_seconds == 90
    assert manifest.allowed_tools == ("risk.read@1",)
    assert tools.resolve("risk.read@1").manifest.risk_level.value == "READ_ONLY"


@pytest.mark.asyncio
async def test_orchestrator_renders_verified_risk_card_without_resynthesis_or_plan_write():
    tools = Tools()
    specialist = RiskHarness(
        model_gateway=MockModelGateway(fixtures={}), tool_executor=tools, actor_resolver=Actors()
    )
    calls: list[AgentHandoff] = []

    class Specialists:
        async def run_specialist(self, handoff: AgentHandoff) -> AgentResult:
            calls.append(handoff)
            return await specialist.run(handoff)

    registry, _ = build_agent_registry()
    fixture = {
        "objectives": ["Explain risk"],
        "steps": [
            dict(
                step_id="risk",
                target_agent_id="risk",
                target_agent_version="1.0.0",
                capability="risk.explain",
                objective="Explain risk",
                typed_input={"task_reference": str(tools.task)},
                depends_on=[],
                mode="READ_ONLY",
            )
        ],
        "unavailable_capabilities": [],
        "response_language": "vi",
    }
    result = await OrchestratorHarness(
        model_gateway=MockModelGateway(fixtures={"orchestrator.vi.plan": fixture}),
        registry=registry,
        policy_guard=PolicyGuard(),
        actor_resolver=Actors(),
        specialists=Specialists(),
    ).run_turn(
        OrchestratorInput(
            conversation_id=uuid4(),
            turn_id=uuid4(),
            message="Giải thích rủi ro",
            locale="vi",
            actor=handoff(tools).actor,
            active_context=ActiveConversationContext(recent_messages=()),
        )
    )
    assert result.status.value == "COMPLETED"
    assert result.blocks[0].kind == "risk"
    assert result.blocks[0].model_dump()["content"]["score"] == "92"
    assert len(calls) == 1 and calls[0].target_agent_id is AgentId.RISK


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL")
@pytest.mark.asyncio
async def test_get_history_and_sse_remove_card_after_manager_permission_is_revoked(
    harness: Harness,
):
    task = UUID(await seed_task(harness))
    app, risks, _, manager = await setup(harness)
    await risks.request_refresh(manager, task, str(uuid4()))
    assert await risks.run_once(manager)
    reads = reader(harness)
    context = await reads.read(manager, task)
    card = dict(
        kind="risk",
        task_id=str(task),
        fingerprint=context.fingerprint,
        content={
            **context.model_dump(mode="json"),
            "explanation": {
                "observation_explanations": [],
                "limitations": [],
                "recommendations": [],
                "replan_requested": False,
            },
            "fallback": True,
        },
    )
    sessions = async_sessionmaker(
        bind=harness.connection,
        class_=AsyncSession,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    transactions = PostgreSQLAssistantTransactionFactory(sessions)
    projector = RiskBlockProjector(reads)
    service = AssistantService(
        transaction_factory=transactions,
        planning_snapshot=cast(PlanningSnapshotPort, object()),
        block_projector=projector,
        orchestrator_version="1.0.0",
        orchestrator_fingerprint="a" * 64,
    )
    events = AssistantEventService(transaction_factory=transactions, block_projector=projector)
    app.state.assistant_service = service
    created = await service.create_conversation(
        actor=manager,
        locale="vi",
        title=None,
        request_id=str(uuid4()),
        idempotency_key=str(uuid4()),
    )
    conversation = created.conversation
    await harness.sql(
        "INSERT INTO assistant_messages "
        "(id,organization_id,conversation_id,sequence,role,content_blocks,created_at) "
        "VALUES (:id,:org,:conv,1,'ASSISTANT',CAST(:blocks AS jsonb),now())",
        {
            "id": uuid4(),
            "org": manager.organization_id,
            "conv": conversation.id,
            "blocks": json.dumps([card]),
        },
    )
    await harness.sql(
        "INSERT INTO assistant_events "
        "(id,organization_id,conversation_id,sequence,event_type,public_payload,occurred_at) "
        "VALUES (:id,:org,:conv,2,'test.risk',CAST(:payload AS jsonb),now())",
        {
            "id": uuid4(),
            "org": manager.organization_id,
            "conv": conversation.id,
            "payload": json.dumps({"blocks": [card]}),
        },
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        before = await client.get(f"/api/v1/ai/conversations/{conversation.id}")
        assert before.status_code == 200, before.text
        assert before.json()["messages"][0]["content_blocks"][0]["kind"] == "risk"
        assert "83" in "\n".join(
            await events.replay(actor=manager, conversation_id=conversation.id, after_sequence=1)
        )
        await harness.sql(
            "UPDATE memberships SET role='EMPLOYEE' WHERE id=:id", {"id": manager.membership_id}
        )
        after = await client.get(f"/api/v1/ai/conversations/{conversation.id}")
        assert after.status_code == 200, after.text
        assert after.json()["messages"][0]["content_blocks"][0]["kind"] == "safe_error"
        replay = "\n".join(
            await events.replay(actor=manager, conversation_id=conversation.id, after_sequence=1)
        )
        assert "RISK_CONTEXT_UNAVAILABLE" in replay
        assert "83" not in replay and "Review deadline" not in replay


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL")
@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["legacy", "goal", "ambiguous"])
async def test_generated_advice_and_limitations_preserve_snapshot_and_public_card(
    harness: Harness, case: str
):
    from app.modules.assistant.adapters.agent_runtime import CurrentAgentActorResolver
    from app.modules.assistant.adapters.risk_tools import RiskToolAdapter
    from app.modules.work.adapters.task_repository import SqlAlchemyTaskTransactionFactory
    from app.modules.work.application.task_service import TaskService
    from work_management_ai.runtime.contracts import (
        AgentHandoffV2,
        AgentRunStatus,
        DelegationBrief,
        GoalDirectedAgentResult,
        InteractionOutcome,
    )

    task = UUID(await seed_task(harness))
    if case == "ambiguous":
        await seed_task(harness)
    _, risks, _, manager = await setup(harness)
    await risks.request_refresh(manager, task, str(uuid4()))
    assert await risks.run_once(manager)
    reads = reader(harness)
    context = await reads.read(manager, task)

    class CurrentActors:
        async def resolve(
            self, *, organization_id: UUID, membership_id: UUID
        ) -> AuthenticatedActor | None:
            assert (
                organization_id == manager.organization_id
                and membership_id == manager.membership_id
            )
            return manager

    sessions = async_sessionmaker(
        bind=harness.connection,
        class_=AsyncSession,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    tools = RiskToolAdapter(
        actors=CurrentActors(),
        tasks=TaskService(SqlAlchemyTaskTransactionFactory(sessions)),
        reads=reads,
    )
    source = context.observations[0].source_ids[0]
    output = dict(
        observation_explanations=[
            dict(
                text="Review the recorded deadline",
                observation_ids=[context.observations[0].id],
                source_ids=[source],
                assertions=[
                    dict(
                        source_id=source,
                        field="title",
                        value=next(
                            s.values["title"] for s in context.permitted_sources if s.id == source
                        ),
                    )
                ],
            )
        ],
        limitations=["An additional advisory limitation"],
        recommendations=["A new advisory action"],
        replan_requested=False,
    )
    legacy = handoff(Tools()).model_copy(
        update={
            "actor": manager_ref(manager),
            "typed_input": {
                "task_reference": str(task),
                "locale": "vi",
                "question": "Explain risk",
            },
        }
    )
    value = (
        legacy
        if case == "legacy"
        else AgentHandoffV2.model_validate(
            {
                **legacy.model_dump(
                    exclude={"schema_version", "typed_input", "objective", "capability"}
                ),
                "target_agent_version": "1.1.0",
                "source_turn_ref": uuid4(),
                "locale": "vi",
                "timezone": "UTC",
                "brief": DelegationBrief(
                    objective="Rủi ro của task Prepare là gì?", expected_result="Verified risk"
                ),
            }
        )
    )
    result = await RiskHarness(
        model_gateway=MockModelGateway(fixtures={"risk.vi.explain": output}),
        tool_executor=tools,
        actor_resolver=CurrentAgentActorResolver(CurrentActors()),
    ).run(value)
    if case == "ambiguous":
        assert isinstance(result, GoalDirectedAgentResult)
        assert result.status is AgentRunStatus.AWAITING_INPUT
        assert result.interaction_outcome is InteractionOutcome.NEEDS_INPUT
        assert result.resolved_action is None and result.tool_calls_used == 1
        return
    assert result.status is AgentRunStatus.COMPLETED
    card = dict(
        kind="risk", task_id=str(task), fingerprint=context.fingerprint, content=result.typed_output
    )
    projected = await RiskBlockProjector(reads).project(manager, card)
    assert projected["kind"] == "risk"
    assert projected["content"]["score"] == "83"
    assert projected["content"]["recommendations"] == list(context.recommendations)
    assert projected["content"]["explanation"]["recommendations"] == ["A new advisory action"]
    assert projected["content"]["explanation"]["limitations"] == [
        "An additional advisory limitation"
    ]


def manager_ref(actor: "AuthenticatedActor") -> "ActorReference":
    from work_management_ai.runtime.contracts import ActorReference

    return ActorReference(organization_id=actor.organization_id, membership_id=actor.membership_id)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL")
@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["pending", "finalized"])
@pytest.mark.parametrize("change", ["permission", "fingerprint", "missing_pin"])
async def test_goal_risk_receipts_recheck_authorized_snapshot(
    journal: Journal, replay: str, change: str
) -> None:
    from app.modules.assistant.adapters.agent_runtime import CurrentAgentActorResolver
    from app.modules.assistant.adapters.delegation_tools import PostgreSQLGoalSpecialistRunner
    from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
    from app.modules.assistant.adapters.risk_tools import RiskToolAdapter
    from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
    from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthTransactionFactory
    from app.modules.identity.adapters.current_actor import CurrentActorResolver
    from app.modules.identity.application.current_actor_service import CurrentActorService
    from app.modules.risk.adapters.repository import RiskTransactions
    from app.modules.risk.application.read_service import RiskReadService
    from app.modules.work.adapters.task_repository import SqlAlchemyTaskTransactionFactory
    from app.modules.work.application.task_service import TaskService
    from tests.test_orchestrator_tool_pipeline_integration import Policy
    from work_management_ai.model_gateway.tool_contracts import ToolCall
    from work_management_ai.runtime.contracts import AgentHandoffV2
    from work_management_ai.runtime.delegation import DelegationAdapterV2
    from work_management_ai.runtime.run_budget import (
        CandidateBudget,
        UsageReservation,
        candidate_budget,
    )
    from work_management_ai.runtime.tool_catalog import PreparedToolSet
    from work_management_ai.runtime.tool_loop_contracts import AdmittedTool
    from work_management_ai.runtime.tool_pipeline import ToolPipeline, ToolPolicyError
    from work_management_ai.runtime.tool_registry import ToolRegistry, load_tool_manifest_resource
    from work_management_ai.runtime.triggers import ChatTurnTrigger, ExecutionScope
    from work_management_ai.tools.delegation.contracts import (
        DelegateAgentInputV2,
        DelegateAgentOutputV2,
    )

    engine, sessions, actor, invocation_store, scope, lease, _, _ = journal
    project_id, task_id = uuid4(), uuid4()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO projects(id,organization_id,name,description,"
                "created_by_membership_id,updated_by_membership_id) "
                "VALUES(:id,:org,'Risk Project','',:member,:member)"
            ),
            {"id": project_id, "org": actor.organization_id, "member": actor.membership_id},
        )
        await connection.execute(
            text(
                "INSERT INTO tasks(id,organization_id,project_id,title,status,"
                "created_by_membership_id,updated_by_membership_id) "
                "VALUES(:id,:org,:project,'Release review','IN_PROGRESS',:member,:member)"
            ),
            {
                "id": task_id,
                "org": actor.organization_id,
                "member": actor.membership_id,
                "project": project_id,
            },
        )
    registry, _ = build_agent_registry()
    registry.register_resource("work_management_ai.agents.risk", "agent.v2.yaml")
    pins = registry.specialist_pins(goal_directed=True)
    tools = ToolRegistry(
        [load_tool_manifest_resource("work_management_ai.tools.delegation", "delegate.v2.yaml")]
    )
    recorder = PostgreSQLExecutionRecorder(
        transaction_factory=PostgreSQLAssistantTransactionFactory(sessions),
        registry=registry,
        scope=ExecutionScope(
            organization_id=scope.organization_id,
            actor_membership_id=scope.membership_id,
            orchestration_run_id=scope.orchestration_run_id,
            trigger=ChatTurnTrigger(turn_id=scope.turn_id),
        ),
        actor=actor,
    )
    parent_id = await recorder.ensure_orchestrator_run()
    actors = CurrentActorResolver(CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions)))
    reads = RiskReadService(RiskTransactions(sessions))
    risk_tools = RiskToolAdapter(
        actors=actors, tasks=TaskService(SqlAlchemyTaskTransactionFactory(sessions)), reads=reads
    )
    risk = RiskHarness(
        model_gateway=MockModelGateway(fixtures={}),
        tool_executor=risk_tools,
        actor_resolver=CurrentAgentActorResolver(actors),
    )
    observed: list[AgentHandoffV2] = []

    class CaptureHandoff:
        async def run(self, handoff: AgentHandoffV2) -> AgentResult:
            observed.append(handoff)
            return await risk.run(handoff)

    runner = PostgreSQLGoalSpecialistRunner(
        recorder=recorder, harness=CaptureHandoff(), pin=pins[0], risk_tools=risk_tools
    )
    adapter = DelegationAdapterV2(
        scope=scope,
        parent_agent_run_id=parent_id,
        actor_role="MANAGER",
        active_phase=5,
        locale="en",
        timezone="UTC",
        agent_registry=registry,
        specialist_pins=pins,
        runners={AgentId.RISK: runner},
        tool_registry=tools,
    )
    admitted = PreparedToolSet(
        (
            AdmittedTool(
                schema=tools.candidate_schemas(("delegate_agent@2",))[0],
                input_contract=DelegateAgentInputV2,
                output_contract=DelegateAgentOutputV2,
                version="2.0.0",
                fingerprint="test-risk-goal",
            ),
        ),
        "test-risk-replay",
        lambda _: True,
    )
    call = ToolCall(
        call_id="risk",
        name="delegate_agent",
        arguments=DelegateAgentInputV2(
            agent_id=AgentId.RISK,
            objective='Explain the risk of task "Release review".',
            expected_result="Verified risk",
        ).model_dump(mode="json"),
    )
    policy = Policy()
    policy.fail_post = replay == "pending"
    pipeline = ToolPipeline(
        store=invocation_store, policy=policy, backends={"delegate_agent@2": adapter}
    )
    saved = await reads.read(actor, task_id)
    budget, interval_id = PostgreSQLRunBudget(sessions, actor), uuid4()
    await budget.reserve(
        scope, UsageReservation(operation_id=interval_id, kind="INTERVAL"), lease=lease
    )
    with candidate_budget(
        CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=interval_id)
    ):
        first = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert first.kind == ("REJECTED" if replay == "pending" else "COMPLETED"), first
        if first.kind == "REJECTED":
            assert first.safe_code == "TOOL_POSTCONDITION_FAILED"
        else:
            assert await pipeline.execute(scope, call, admitted=admitted, lease=lease) == first
        (handoff,) = observed
        recovered = await runner.reconcile(handoff)
        assert (
            recovered is not None and recovered[1].typed_output["fingerprint"] == saved.fingerprint
        )
        child_id, _ = recovered
        wrong_member = handoff.model_copy(
            update={"actor": handoff.actor.model_copy(update={"membership_id": uuid4()})}
        )
        with pytest.raises(RuntimeError, match="AGENT_HANDOFF_SCOPE_MISMATCH"):
            await runner.reconcile(wrong_member)
        unconfigured = PostgreSQLGoalSpecialistRunner(
            recorder=recorder, harness=CaptureHandoff(), pin=pins[0]
        )
        with pytest.raises(ToolPolicyError, match="RISK_CONTEXT_UNAVAILABLE"):
            await unconfigured.reconcile(handoff)
        policy.fail_post = False
        async with engine.begin() as connection:
            if change == "permission":
                await connection.execute(
                    text("UPDATE memberships SET role='EMPLOYEE' WHERE id=:id"),
                    {
                        "id": actor.membership_id,
                    },
                )
            elif change == "fingerprint":
                # Change the source fingerprint while keeping the task version pinned.
                await connection.execute(
                    text("UPDATE tasks SET title='Changed' WHERE id=:id"),
                    {
                        "id": task_id,
                    },
                )
            else:
                await connection.execute(
                    text(
                        "UPDATE agent_runs SET typed_output=typed_output - 'evidence' WHERE id=:id"
                    ),
                    {"id": child_id},
                )
        if change == "fingerprint":
            current = await reads.read(actor, task_id)
            assert current.task_version == saved.task_version
            assert current.fingerprint != saved.fingerprint
        returned = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert returned.kind == "REJECTED" and returned.safe_code == "RISK_CONTEXT_UNAVAILABLE", (
            returned
        )
        with pytest.raises(ToolPolicyError, match="RISK_CONTEXT_UNAVAILABLE"):
            await runner.reconcile(handoff)
        with pytest.raises(ToolPolicyError, match="RISK_CONTEXT_UNAVAILABLE"):
            await runner.run(handoff)
        assert observed == [handoff]
