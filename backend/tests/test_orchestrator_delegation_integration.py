"""Task 9 PostgreSQL integration: delegation receipts, effect fences and workflow waits."""

import os
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.execution_fence import ExecutionFenceContext, execution_fence
from app.modules.assistant.adapters.agent_runtime import CurrentAgentActorResolver
from app.modules.assistant.adapters.daily_update_tools import DailyUpdateToolAdapter
from app.modules.assistant.adapters.delegation_tools import (
    BackendDelegationTool,
    PostgreSQLDailyUpdateReceiptReader,
    PostgreSQLGoalSourceTurnReader,
    PostgreSQLGoalSpecialistRunner,
    PostgreSQLWorkflowWaitVerifier,
)
from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.adapters.work_tools import RecordingToolExecutor, WorkToolExecutor
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthTransactionFactory
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.planning_runs.adapters.database_models import WorkflowRunModel
from app.modules.planning_runs.adapters.transaction import PostgreSQLPlanningRunTransactionFactory
from app.modules.planning_runs.domain.models import WorkflowJob, WorkflowJobStatus, WorkflowRun
from app.modules.progress.adapters.assessment_repository import SqlAlchemyAssessmentTransactions
from app.modules.progress.adapters.daily_update_repository import SqlAlchemyDailyUpdateTransactions
from app.modules.progress.application.assessment_service import AssessmentService
from app.modules.progress.application.daily_update_service import DailyUpdateService
from app.modules.work.adapters.project_repository import SqlAlchemyProjectTransactionFactory
from app.modules.work.adapters.task_repository import SqlAlchemyTaskTransactionFactory
from app.modules.work.application.project_service import ProjectService
from app.modules.work.application.task_service import TaskService
from app.modules.work.planning.adapters.manual_repository import (
    SqlAlchemyManualPlanningTransactionFactory,
)
from app.modules.work.planning.application.manual_service import ManualPlanningService
from tests.retrieval_fixtures import RetrievalHarness
from tests.retrieval_fixtures import retrieval_harness as retrieval_harness
from tests.test_orchestrator_session_log_integration import Journal
from tests.test_orchestrator_session_log_integration import journal as journal
from tests.test_orchestrator_tool_pipeline_integration import Policy
from work_management_ai.agents.daily_update.harness import DailyUpdateHarness
from work_management_ai.agents.work_intelligence.harness import WorkIntelligenceHarness
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.tool_contracts import ToolCall
from work_management_ai.runtime.agent_registry import AgentRegistry
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentBudget,
    AgentHandoff,
    AgentHandoffV2,
    AgentId,
    AgentResult,
    AgentRunStatus,
    DelegationBrief,
)
from work_management_ai.runtime.delegation import (
    DelegationAdapter,
    DelegationAdapterV2,
    DelegationError,
    VerifiedDelegationReferences,
    filtered_delegate_tool_schema,
    filtered_delegate_v2_tool_schema,
)
from work_management_ai.runtime.execution_engine import DurableSpecialistRunner, RecordedAgentRun
from work_management_ai.runtime.run_budget import (
    BudgetedSpecialistTools,
    BudgetedToolLoopGateway,
    BudgetReceipt,
    CandidateBudget,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.session_history import ToolResultPayload
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_catalog import PreparedToolSet
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    CompletedToolResult,
    DeferredToolResult,
    RunScope,
    ToolInvocationContext,
    ToolLoopLimits,
)
from work_management_ai.runtime.tool_pipeline import ToolPipeline
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.runtime.triggers import ChatTurnTrigger, ExecutionScope
from work_management_ai.tools.delegation.contracts import (
    DelegateAgentInput,
    DelegateAgentInputV2,
    DelegateAgentOutput,
    DelegateAgentOutputV2,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


async def _seed_workflow_run(
    sessions: async_sessionmaker[AsyncSession],
    actor: AuthenticatedActor,
    wf_id: UUID,
    *,
    explicit_commit: bool = True,
    status: str = "RUNNING",
) -> None:
    async with PostgreSQLPlanningRunTransactionFactory(sessions)(actor) as tx:
        tx.session.add(
            WorkflowRunModel(
                id=wf_id,
                organization_id=actor.organization_id,
                requested_by_membership_id=actor.membership_id,
                status=status,
                workflow_name="project_planning",
                workflow_version="1.0.0",
                verifier_version="1.0.0",
                input_goal_text="Launch course",
            )
        )
        if explicit_commit:
            await tx.commit()


@pytest.mark.parametrize("explicit_commit", [True, False])
async def test_fenced_effect_blocks_stale_lease_but_manual_flow_remains_usable(
    journal: Journal,
    explicit_commit: bool,
) -> None:
    _, sessions, actor, _, scope, lease, _, _ = journal
    budget = PostgreSQLRunBudget(sessions, actor)
    interval_id = uuid4()
    op_id = uuid4()
    await budget.reserve(
        scope, UsageReservation(operation_id=interval_id, kind="INTERVAL"), lease=lease
    )
    await budget.reserve(
        scope,
        UsageReservation(operation_id=op_id, kind="TOOL", interval_id=interval_id),
        lease=lease,
    )

    # 1) Manual flow (no execution_fence) succeeds cleanly on planning_runs transaction
    manual_wf_id = uuid4()
    await _seed_workflow_run(sessions, actor, manual_wf_id, explicit_commit=explicit_commit)

    # 2) Candidate run with STALE fencing_token must be blocked before commit
    stale_lease = lease.model_copy(update={"fencing_token": 999})
    stale_fence = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(stale_lease),
        operation_id=op_id,
        interval_id=interval_id,
    )
    blocked_wf_id = uuid4()
    with pytest.raises(SessionError), execution_fence(stale_fence):
        await _seed_workflow_run(sessions, actor, blocked_wf_id, explicit_commit=explicit_commit)

    # Verify blocked workflow row was never committed
    async with sessions() as session, session.begin():
        count = await session.scalar(
            text("SELECT count(*) FROM workflow_runs WHERE id = :id"), {"id": blocked_wf_id}
        )
    assert count == 0


@pytest.mark.parametrize("interrupt_before_settle", [False, True])
@pytest.mark.parametrize("goal_directed", [False, True])
async def test_pipeline_replays_work_harness_after_child_result_commit(
    journal: Journal,
    interrupt_before_settle: bool,
    goal_directed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, sessions, actor, journal_recorder, scope, lease, _, _ = journal
    budget = PostgreSQLRunBudget(sessions, actor)
    interval_id = uuid4()
    interval = await budget.reserve(
        scope, UsageReservation(operation_id=interval_id, kind="INTERVAL"), lease=lease
    )

    skills = SkillRegistry.from_resources(
        (f"work_management_ai.skills.{n}", "skill.yaml") for n in ("answer_work_question",)
    )
    tools = ToolRegistry.from_resources(
        (
            ("work_management_ai.tools.automation", "tool.yaml"),
            ("work_management_ai.tools.work.read_my_tasks", "tool.yaml"),
            ("work_management_ai.tools.work.read_resource", "tool.yaml"),
            ("work_management_ai.tools.risk", "tool.yaml"),
            ("work_management_ai.tools.delegation", "delegate.yaml"),
            ("work_management_ai.tools.delegation", "delegate.v2.yaml"),
        )
    )
    registry = AgentRegistry(
        skill_registry=skills,
        tool_registry=tools,
        evaluator_ids=frozenset(
            {
                "orchestrator_plan@1",
                "work_grounding@1",
            }
        ),
    )
    registry.register_resource("work_management_ai.agents.orchestrator", "agent.yaml")
    registry.register_resource("work_management_ai.agents.work_intelligence", "agent.yaml")
    if goal_directed:
        registry.register_resource("work_management_ai.agents.work_intelligence", "agent.v2.yaml")

    exec_scope = ExecutionScope(
        organization_id=scope.organization_id,
        actor_membership_id=scope.membership_id,
        orchestration_run_id=scope.orchestration_run_id,
        trigger=ChatTurnTrigger(turn_id=scope.turn_id),
    )
    recorder = PostgreSQLExecutionRecorder(
        transaction_factory=PostgreSQLAssistantTransactionFactory(sessions),
        registry=registry,
        scope=exec_scope,
        actor=actor,
    )
    parent_run_id = await recorder.ensure_orchestrator_run()

    project_id, task_id = uuid4(), uuid4()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO projects(id,organization_id,name,description,"
                "created_by_membership_id,updated_by_membership_id) "
                "VALUES(:id,:org,'Release','',:member,:member)"
            ),
            {"id": project_id, "org": scope.organization_id, "member": scope.membership_id},
        )
        await connection.execute(
            text(
                "INSERT INTO tasks(id,organization_id,project_id,title,status,"
                "assignee_membership_id,created_by_membership_id,updated_by_membership_id) "
                "VALUES(:id,:org,:project,'Ship release','IN_PROGRESS',:member,:member,:member)"
            ),
            {
                "id": task_id,
                "org": scope.organization_id,
                "member": scope.membership_id,
                "project": project_id,
            },
        )
    evidence_id = f"TASK:{task_id}:1"
    live_actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
    live_tasks = TaskService(SqlAlchemyTaskTransactionFactory(sessions))
    read_tool = RecordingToolExecutor(
        transaction_factory=PostgreSQLAssistantTransactionFactory(sessions),
        tool_registry=tools,
        backend=WorkToolExecutor(
            actor_resolver=live_actors,
            tool_registry=tools,
            task_service=live_tasks,
            project_service=ProjectService(SqlAlchemyProjectTransactionFactory(sessions)),
            planning_service=ManualPlanningService(
                SqlAlchemyManualPlanningTransactionFactory(sessions)
            ),
        ),
    )
    mock = MockModelGateway(
        fixtures={
            "work_intelligence.en.plan": {
                "question_kind": "MY_TASKS",
                "skill_reference": "answer_work_question@1",
                "tool_id": "work.read_my_tasks",
                "tool_input": {"status": None, "due_from": None, "due_to": None, "limit": 5},
                "requested_handoff": None,
            },
            "work_intelligence.en.synthesize": {
                "question_kind": "MY_TASKS",
                "needs_clarification": False,
                "clarification_question": None,
                "claims": [
                    {
                        "text": "Ship release is in progress.",
                        "evidence_ids": [evidence_id],
                        "assertions": [
                            {
                                "evidence_id": evidence_id,
                                "field": "status",
                                "value": "IN_PROGRESS",
                            }
                        ],
                    }
                ],
            },
        },
        usage_fixtures={
            key: ModelUsage(input_tokens=800, output_tokens=150)
            for key in ("work_intelligence.en.plan", "work_intelligence.en.synthesize")
        },
    )
    harness = WorkIntelligenceHarness(
        model_gateway=BudgetedToolLoopGateway(mock),
        tool_executor=BudgetedSpecialistTools(read_tool),
        actor_resolver=CurrentAgentActorResolver(live_actors),
    )
    runner = DurableSpecialistRunner(
        recorder=recorder, harnesses={AgentId.WORK_INTELLIGENCE: harness}
    )
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        tool_registry=tools,
        runner=runner,
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=parent_run_id,
    )
    backend = BackendDelegationTool(actor=actor, adapter=adapter)
    args = DelegateAgentInput.model_validate(
        {
            "agent_id": "work_intelligence",
            "capability": "work.answer_question",
            "objective": "Check active tasks",
            "expected_result": "Grounded task status",
            "input": {
                "capability": "work.answer_question",
                "question": "What am I working on?",
                "locale": "en",
            },
        }
    )
    schema = filtered_delegate_tool_schema(
        tools.candidate_schemas(("delegate_agent@1",))[0],
        allowed_capabilities={(AgentId.WORK_INTELLIGENCE, args.capability)},
    )
    admitted_tool = AdmittedTool(
        schema=schema,
        input_contract=DelegateAgentInput,
        output_contract=DelegateAgentOutput,
        version="1.0.0",
        fingerprint="test-delegation",
    )
    if goal_directed:
        from work_management_ai.runtime.contracts import ContextReference

        async def verify_refs(current: RunScope, refs: tuple[ContextReference, ...]) -> None:
            assert current == scope
            for ref in refs:
                current_task = await live_tasks.get_task(actor=actor, task_id=ref.resource_id)
                assert current_task.version == ref.version

        pins = registry.specialist_pins(goal_directed=True)
        goal_backend = DelegationAdapterV2(
            scope=scope,
            parent_agent_run_id=parent_run_id,
            actor_role="MANAGER",
            active_phase=5,
            locale="en",
            timezone="UTC",
            agent_registry=registry,
            specialist_pins=pins,
            tool_registry=tools,
            runners={
                AgentId.WORK_INTELLIGENCE: PostgreSQLGoalSpecialistRunner(
                    recorder=recorder, harness=harness, pin=pins[0]
                )
            },
            references=VerifiedDelegationReferences(
                scope=scope,
                exploration=(),
                active_context=(),
                completed_specialists=(),
                recheck=verify_refs,
            ),
        )
        goal_args = DelegateAgentInputV2(
            agent_id=AgentId.WORK_INTELLIGENCE,
            objective="What am I working on?",
            expected_result="Grounded task status",
        )
        goal_schema = filtered_delegate_v2_tool_schema(
            tools.candidate_schemas(("delegate_agent@2",))[0],
            allowed_specialists=goal_backend.available_specialists(),
        )
        assert goal_schema.input_contract is not None
        admitted_tool = AdmittedTool(
            schema=goal_schema,
            input_contract=goal_schema.input_contract,
            output_contract=DelegateAgentOutputV2,
            version="2.0.0",
            fingerprint="test-goal",
        )
    else:
        goal_backend = None
        goal_args = None
    admitted = PreparedToolSet((admitted_tool,), "test-delegation", lambda _: True)
    policy = Policy()
    backends = {"delegate_agent@2": goal_backend} if goal_backend else {"delegate_agent@1": backend}
    pipeline = ToolPipeline(store=journal_recorder, policy=policy, backends=backends)
    call = ToolCall(
        call_id="d" * 128,
        name="delegate_agent",
        arguments=(goal_args or args).model_dump(mode="json"),
    )
    with candidate_budget(
        CandidateBudget(scope=scope, lease=lease, store=budget, interval_id=interval_id)
    ):
        original_settle = budget.settle

        async def interrupt(
            run_scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
        ) -> None:
            reservation = receipt.reservation
            if reservation.kind == "INTERVAL" and reservation.allocation_id is not None:
                raise RuntimeError("simulate process exit before settling child allocation")
            await original_settle(run_scope, receipt, usage)

        if interrupt_before_settle:
            with monkeypatch.context() as patch:
                patch.setattr(budget, "settle", interrupt)
                first = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
            assert first.kind == "REJECTED" and first.safe_code == "TOOL_RECONCILIATION_REQUIRED"
        else:
            policy.fail_post = True
            first = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
            assert first.kind == "REJECTED" and first.safe_code == "TOOL_POSTCONDITION_FAILED", (
                first
            )
            policy.fail_post = False
        restarted = ToolPipeline(store=journal_recorder, policy=policy, backends=backends)
        replayed = await restarted.execute(scope, call, admitted=admitted, lease=lease)
        assert isinstance(replayed, CompletedToolResult), replayed
        assert await restarted.execute(scope, call, admitted=admitted, lease=lease) == replayed
    payload = ToolResultPayload.model_validate(
        await journal_recorder.get_payload(scope, replayed.payload_ref)
    )
    output = (DelegateAgentOutputV2 if goal_directed else DelegateAgentOutput).model_validate(
        payload.output
    )
    assert "Ship release is in progress." in str(output.typed_output["claims"])
    assert any(ref.resource_id == task_id and ref.version == 1 for ref in output.evidence)
    receipt_id = replayed.invocation_id
    child_key = f"{'goal-delegate' if goal_directed else 'delegate'}:{receipt_id}"
    expected_child_id = uuid5(NAMESPACE_URL, f"agent-run:{child_key}")
    found = await recorder.lookup_agent_run(child_key)
    assert found is not None
    assert found.id == expected_child_id
    assert found.status is AgentRunStatus.COMPLETED
    if isinstance(output, DelegateAgentOutputV2):
        assert output.resolved_action == "work.read_my_tasks"
    usage = (await budget.remaining(scope)).usage
    assert usage.handoffs == 1
    assert usage.total_tool_calls == 2
    assert usage.total_model_requests == 2
    assert (await budget.remaining(scope)).available.total_model_requests == 22
    await budget.settle(scope, interval, None)


@pytest.mark.parametrize("case", ["success", "protected", "publication_rejected"])
async def test_goal_retrieve_runs_and_replays_with_postgresql(
    retrieval_harness: RetrievalHarness, case: str
) -> None:
    from app.core.config import Settings
    from app.core.database import create_session_factory
    from app.modules.assistant.adapters.agent_runtime import build_agent_registry
    from app.modules.assistant.adapters.retrieval_runtime import (
        RetrievalRuntime,
        retrieval_job_scope,
    )
    from app.modules.identity.adapters.current_actor import CurrentActorResolver
    from app.modules.planning_runs.adapters.ai_runtime import build_model_gateway
    from app.modules.retrieval.adapters.mock import MockRetrievalAdapter
    from app.modules.retrieval.application.service import RetrievalService
    from app.modules.retrieval.domain.models import (
        PublicSearchRequest,
        RetrievalPolicyError,
        SearchBatch,
    )
    from work_management_ai.agents.retrieve.harness import RetrieveHarness
    from work_management_ai.runtime.activation import ActivationContext
    from work_management_ai.runtime.contracts import GoalDirectedAgentResult

    h = retrieval_harness
    activation = ActivationContext(5, True, frozenset({"I3"}), True)
    registry, tools = build_agent_registry(activation=activation)
    registry.register_resource("work_management_ai.agents.retrieve", "agent.v2.yaml")
    sessions = create_session_factory(h.engine)
    transactions = PostgreSQLAssistantTransactionFactory(sessions)
    actors = CurrentActorResolver(CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions)))
    searches: list[str] = []

    class Provider(MockRetrievalAdapter):
        async def search(
            self, request: PublicSearchRequest, *, timeout_seconds: float
        ) -> SearchBatch:
            searches.append(request.query)
            return await super().search(request, timeout_seconds=timeout_seconds)

    class DNS:
        async def resolve(self, hostname: str) -> tuple[str, ...]:
            return ("93.184.216.34",)

    budgets, provider = h.service(), Provider()
    runtime = RetrievalRuntime(
        budgets=budgets,
        service=RetrievalService(search=provider, pages=provider, access=budgets, dns=DNS()),
        actors=actors,
        tools=tools,
        transactions=transactions,
    )
    recorder = PostgreSQLExecutionRecorder(
        transaction_factory=transactions,
        registry=registry,
        job=h.job,
        actor=h.actor,
        activation=activation,
        retrieval_replay_validator=runtime,
    )
    scope = RunScope(
        organization_id=h.actor.organization_id,
        membership_id=h.actor.membership_id,
        conversation_id=h.job.conversation_id,
        turn_id=h.job.turn_id,
        orchestration_run_id=h.orchestration_id,
    )
    if case == "protected":
        await h.sql(
            "INSERT INTO projects(id,organization_id,name,created_by_membership_id,"
            "updated_by_membership_id) VALUES(:id,:org,'public planning',:member,:member)",
            {"id": uuid4(), "org": scope.organization_id, "member": scope.membership_id},
        )
    if case == "success":
        await h.sql(
            "UPDATE assistant_messages SET content_blocks="
            '\'[{"kind":"text","text":"Search the public web for public planning"}]\'::jsonb'
            " WHERE id=:id",
            {"id": h.message_id},
        )
    handoff = AgentHandoffV2(
        orchestration_run_id=scope.orchestration_run_id,
        parent_agent_run_id=await recorder.ensure_orchestrator_run(),
        target_agent_id=AgentId.RETRIEVE,
        target_agent_version="1.1.0",
        context_references=(),
        actor=ActorReference(
            organization_id=scope.organization_id, membership_id=scope.membership_id
        ),
        brief=DelegationBrief(
            objective="Find guidance about planning" if case == "success" else "public planning",
            expected_result="Public sources",
        ),
        source_turn_ref=scope.turn_id,
        locale="en",
        timezone="UTC",
        budget=AgentBudget(
            max_model_attempts=2,
            max_input_tokens=12000,
            max_output_tokens=3200,
            max_iterations=12,
            max_tool_calls=8,
            timeout_seconds=25,
        ),
        step_id="retrieve-goal",
        idempotency_key=f"{scope.turn_id}:retrieve-goal",
    )
    pin = next(
        pin for pin in registry.specialist_pins(goal_directed=True) if pin.reference == "retrieve"
    )
    rejected_sources: list[UUID] = []
    with retrieval_job_scope(h.job):
        specialist = RetrieveHarness(
            model_gateway=build_model_gateway(Settings(environment="test", ai_provider="mock")),
            tool_executor=runtime,
            usage_port=await runtime.usage(handoff),
            actor_resolver=CurrentAgentActorResolver(actors),
            activation=activation,
            source_turns=PostgreSQLGoalSourceTurnReader(
                session_factory=sessions, actor=h.actor, scope=scope
            ),
            public_query_validator=runtime,
        )

        class Harness:
            async def run(self, handoff: AgentHandoffV2) -> AgentResult:
                if case == "success":
                    run_id = uuid5(NAMESPACE_URL, f"agent-run:{handoff.idempotency_key}")
                    await budgets.start_run(h.actor, run_id, h.lease())
                    for query in ("Public planning", "the public web"):
                        with pytest.raises(
                            RetrievalPolicyError, match="PUBLIC_QUERY_PROVENANCE_REQUIRED"
                        ):
                            await budgets.reserve_attempt(
                                h.actor,
                                run_id,
                                "forged-search",
                                1,
                                "SEARCH",
                                lease=h.lease(),
                                request=PublicSearchRequest(query),
                            )
                result = await specialist.run(handoff)
                if case == "publication_rejected":
                    assert result.status is AgentRunStatus.COMPLETED and result.evidence
                    rejected_sources.extend(ref.resource_id for ref in result.evidence)
                    h.enabled = False
                return result

        runner = PostgreSQLGoalSpecialistRunner(recorder=recorder, harness=Harness(), pin=pin)
        run_id, result = await runner.run(handoff)
        expected = {
            "success": AgentRunStatus.COMPLETED,
            "protected": AgentRunStatus.AWAITING_INPUT,
            "publication_rejected": AgentRunStatus.FAILED,
        }[case]
        assert result.status is expected, result
        assert isinstance(result, GoalDirectedAgentResult) and result.agent_version == "1.1.0"
        assert bool(result.evidence) == (case == "success")
        if case == "success":
            assert result.typed_output["outcome"] == "ANSWER_READY"
        if case == "protected":
            assert result.resolved_action is None
            assert result.typed_output["outcome"] == "NEEDS_INPUT"
        if case == "publication_rejected":
            assert result.safe_error_code == "RETRIEVAL_CAPABILITY_INACTIVE"
        h.enabled = True
        assert await runner.reconcile(handoff) == (run_id, result)
        assert await runner.run(handoff) == (run_id, result)
        if case == "publication_rejected":
            with pytest.raises(RetrievalPolicyError, match="RETRIEVAL_RUN_STOPPED"):
                await budgets.projection_sources(h.actor, run_id, tuple(rejected_sources))
        if case == "success":
            foreign = await runtime.revalidate_agent_result(
                run_id,
                ActorReference(
                    organization_id=h.foreign.organization_id, membership_id=h.foreign.membership_id
                ),
                result,
            )
            assert isinstance(foreign, GoalDirectedAgentResult)
            assert foreign.status is AgentRunStatus.FAILED and not foreign.evidence
            h.enabled = False
            replay = await runner.reconcile(handoff)
            assert replay is not None
            assert replay[1].status is AgentRunStatus.FAILED and not replay[1].evidence
    assert searches == ([] if case == "protected" else ["public planning"])
    row = (
        await h.sql("SELECT status, typed_input FROM agent_runs WHERE id=:id", {"id": run_id})
    ).one()
    assert row.status == expected.value
    assert row.typed_input == {"source_turn_ref": str(scope.turn_id)}
    if case == "protected":
        assert (
            await h.sql(
                "SELECT 1 FROM retrieval_run_budgets WHERE agent_run_id=:id", {"id": run_id}
            )
        ).scalar_one_or_none() is None


async def test_goal_source_turn_reads_original_under_exact_actor_and_run(journal: Journal):
    _, sessions, actor, _, scope, _, _, _ = journal
    source = PostgreSQLGoalSourceTurnReader(session_factory=sessions, actor=actor, scope=scope)
    handoff = AgentHandoffV2(
        orchestration_run_id=scope.orchestration_run_id,
        parent_agent_run_id=uuid4(),
        target_agent_id=AgentId.RETRIEVE,
        target_agent_version="1.1.0",
        brief=DelegationBrief(objective="Model paraphrase", expected_result="Verified context"),
        context_references=(),
        actor=ActorReference(
            organization_id=scope.organization_id,
            membership_id=scope.membership_id,
        ),
        source_turn_ref=scope.turn_id,
        locale="en",
        timezone="UTC",
        budget=AgentBudget(max_iterations=1, max_tool_calls=0, timeout_seconds=1),
        step_id="read-original",
        idempotency_key="read-original",
    )
    assert await source.read(handoff) == "Plan a customer event"
    with pytest.raises(ValueError, match="DELEGATION_CONTEXT_DENIED"):
        await source.read(handoff.model_copy(update={"source_turn_ref": uuid4()}))
    with pytest.raises(ValueError, match="DELEGATION_CONTEXT_DENIED"):
        await source.read(
            handoff.model_copy(
                update={"actor": handoff.actor.model_copy(update={"organization_id": uuid4()})}
            )
        )


async def test_verifier_reads_canonical_workflow_state(journal: Journal) -> None:
    _, sessions, actor, _, scope, _, _, _ = journal
    wf_id = uuid4()
    await _seed_workflow_run(sessions, actor, wf_id, status="NEEDS_INPUT")
    verifier = PostgreSQLWorkflowWaitVerifier(session_factory=sessions, actor=actor)
    verified = await verifier.verify_workflow(scope, wf_id)
    assert verified is not None
    assert verified.status == "NEEDS_INPUT"


@pytest.mark.parametrize("legacy_step", [False, True])
async def test_planning_recovery_reads_committed_effect_before_child_result(
    journal: Journal,
    legacy_step: bool,
) -> None:
    engine, sessions, actor, _, scope, lease, _, _ = journal
    receipt_id = uuid4()
    child_id = uuid5(NAMESPACE_URL, f"agent-run:delegate:{receipt_id}")
    call_id = "planning-commit"
    step_binding = call_id if legacy_step else str(receipt_id)
    run = WorkflowRun.create(
        organization_id=scope.organization_id,
        project_id=None,
        requested_by_membership_id=scope.membership_id,
        workflow_name="project_planning",
        workflow_version="1.0.0",
        verifier_version="1.0.0",
        input_goal_text="Build course",
    )
    job = WorkflowJob(
        id=uuid4(),
        organization_id=scope.organization_id,
        workflow_run_id=run.id,
        job_type="planning.start",
        status=WorkflowJobStatus.QUEUED,
        payload={},
    )
    async with PostgreSQLPlanningRunTransactionFactory(sessions)(actor) as tx:
        await tx.repository.create_planning_run_mutation(
            actor=actor,
            run=run,
            job=job,
            request_id=f"assistant-tool:{child_id}:delegate:{step_binding}:manage:1",
            idempotency_key=f"assistant:{scope.turn_id}:planning:create",
            request_fingerprint="c" * 64,
        )

    class PendingChild:
        async def run_specialist(self, handoff: AgentHandoff) -> AgentResult:
            pytest.fail("reconciliation attempted to dispatch a Specialist")

        async def lookup_agent_run(self, idempotency_key: str) -> RecordedAgentRun | None:
            return RecordedAgentRun(
                id=child_id,
                status=AgentRunStatus.RUNNING,
                target_agent_id=AgentId.PLANNING,
                target_agent_version="1.0.0",
                capability="planning.create",
                manifest_fingerprint=registry.resolve(AgentId.PLANNING, "1.0.0", 5).fingerprint,
            )

    skills = SkillRegistry.from_resources(
        (f"work_management_ai.skills.{name}", "skill.yaml")
        for name in ("create_project_plan", "revise_project_plan")
    )
    tools = ToolRegistry.from_resources(
        (("work_management_ai.tools.planning.manage_run", "tool.yaml"),)
    )
    registry = AgentRegistry(
        skill_registry=skills,
        tool_registry=tools,
        evaluator_ids=frozenset(
            {"planning_schema@1", "planning_invariants@1", "planning_grounding@1"}
        ),
    )
    registry.register_resource("work_management_ai.agents.planning", "agent.yaml")
    verifier = PostgreSQLWorkflowWaitVerifier(session_factory=sessions, actor=actor)
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        runner=PendingChild(),
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=uuid4(),
        workflow_verifier=verifier,
    )
    args = DelegateAgentInput.model_validate(
        {
            "agent_id": "planning",
            "capability": "planning.create",
            "objective": "Build course",
            "expected_result": "Draft",
            "input": {"capability": "planning.create", "goal": "Build course"},
        }
    )
    context = ToolInvocationContext(
        scope=scope,
        lease=lease,
        invocation_id=receipt_id,
        call_id=call_id,
        tool_reference="delegate_agent@1",
        tool_version="1.0.0",
        arguments_fingerprint="a" * 64,
        policy_version="test-policy",
        canonical_arguments=args,
        receipt_id=receipt_id,
        budget_allocation_id=uuid4(),
        limits=ToolLoopLimits(),
    )
    if legacy_step:
        missing = context.model_copy(update={"call_id": "unrelated"})
        assert await adapter.reconcile(missing, args) is None
    else:
        missing = context.model_copy(update={"receipt_id": uuid4()})
        assert await adapter.reconcile(missing, args) is None
    recovered = await adapter.reconcile(context, args)
    assert recovered is not None and recovered.wait_ref is not None
    assert recovered.wait_ref.kind == "WORKFLOW_RESULT"
    assert recovered.wait_ref.resource_id == run.id

    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE workflow_runs SET status='FAILED' WHERE organization_id=:org AND id=:id"),
            {"org": scope.organization_id, "id": run.id},
        )
    with pytest.raises(DelegationError) as error:
        await adapter.reconcile(context, args)
    assert error.value.code == "DELEGATION_WORKFLOW_FAILED"


async def test_daily_draft_receipt_is_scoped_and_recovered_without_confirming(
    journal: Journal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, sessions, actor, store, scope, lease, _, _ = journal
    project_id, task_id = uuid4(), uuid4()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO projects(id, organization_id, name, created_by_membership_id, "
                "updated_by_membership_id) VALUES (:id,:org,'Daily',:member,:member)"
            ),
            {"id": project_id, "org": scope.organization_id, "member": scope.membership_id},
        )
        await connection.execute(
            text(
                "INSERT INTO tasks(id, organization_id, project_id, title, status,"
                " assignee_membership_id, created_by_membership_id, updated_by_membership_id) "
                "VALUES (:id,:org,:project,'Write report','IN_PROGRESS',:member,:member,:member)"
            ),
            {
                "id": task_id,
                "org": scope.organization_id,
                "member": scope.membership_id,
                "project": project_id,
            },
        )
    updates = DailyUpdateService(SqlAlchemyDailyUpdateTransactions(sessions))
    context = await updates.context(actor, task_id)
    registry_tools = ToolRegistry.from_resources(
        (
            ("work_management_ai.tools.automation", "tool.yaml"),
            ("work_management_ai.tools.daily_update", "tool.yaml"),
            ("work_management_ai.tools.delegation", "delegate.yaml"),
        )
    )
    skills = SkillRegistry.from_resources(
        (f"work_management_ai.skills.{name}", "skill.yaml")
        for name in ("extract_daily_update", "compare_daily_update_evidence")
    )
    registry = AgentRegistry(
        tool_registry=registry_tools,
        skill_registry=skills,
        evaluator_ids=frozenset({"orchestrator_plan@1", "daily_update_grounding@1"}),
    )
    registry.register_resource("work_management_ai.agents.orchestrator", "agent.yaml")
    registry.register_resource("work_management_ai.agents.daily_update", "agent.yaml")
    transactions = PostgreSQLAssistantTransactionFactory(sessions)
    recorder = PostgreSQLExecutionRecorder(
        transaction_factory=transactions,
        registry=registry,
        actor=actor,
        scope=ExecutionScope(
            organization_id=scope.organization_id,
            actor_membership_id=scope.membership_id,
            orchestration_run_id=scope.orchestration_run_id,
            trigger=ChatTurnTrigger(turn_id=scope.turn_id),
        ),
    )
    parent = await recorder.ensure_orchestrator_run()
    actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
    daily = DailyUpdateHarness(
        actor_resolver=CurrentAgentActorResolver(actors),
        model_gateway=BudgetedToolLoopGateway(
            MockModelGateway(
                fixtures={
                    "daily_update.en.extract": {
                        "blockers": [],
                        "reported_percent": 25,
                        "remaining_hours": None,
                        "spent_hours": None,
                        "done_text": "Wrote the first section.",
                        "next_steps": "",
                        "needs_clarification": False,
                    }
                },
                usage_fixtures={
                    "daily_update.en.extract": ModelUsage(input_tokens=800, output_tokens=130)
                },
            )
        ),
        tool_executor=BudgetedSpecialistTools(
            RecordingToolExecutor(
                transaction_factory=transactions,
                tool_registry=registry_tools,
                backend=DailyUpdateToolAdapter(
                    actors=actors,
                    updates=updates,
                    assessments=AssessmentService(SqlAlchemyAssessmentTransactions(sessions)),
                ),
            )
        ),
    )
    receipts = PostgreSQLDailyUpdateReceiptReader(session_factory=sessions, actor=actor)
    adapter = DelegationAdapter(
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(),
        runner=DurableSpecialistRunner(recorder=recorder, harnesses={AgentId.DAILY_UPDATE: daily}),
        actor_role="MANAGER",
        active_phase=5,
        parent_agent_run_id=parent,
        daily_update_receipts=receipts,
    )
    (schema,) = registry_tools.candidate_schemas(("delegate_agent@1",))
    schema = filtered_delegate_tool_schema(
        schema,
        allowed_capabilities={(AgentId.DAILY_UPDATE, "daily_update.prepare")},
        daily_update_ready=True,
    )
    admitted = PreparedToolSet(
        (
            AdmittedTool(
                schema=schema,
                input_contract=DelegateAgentInput,
                output_contract=DelegateAgentOutput,
                version="1.0.0",
                fingerprint="delegation-daily",
            ),
        ),
        "delegation-daily",
        lambda _: True,
    )
    pipeline = ToolPipeline(
        store=store,
        policy=Policy(),
        backends={"delegate_agent@1": BackendDelegationTool(actor=actor, adapter=adapter)},
    )
    call = ToolCall(
        call_id="daily-draft",
        name="delegate_agent",
        arguments={
            "agent_id": "daily_update",
            "capability": "daily_update.prepare",
            "objective": "Prepare a progress draft",
            "expected_result": "Owner review",
            "input": {
                "capability": "daily_update.prepare",
                "task_reference": str(task_id),
                "task_version": context.task_version,
                "locale": "en",
                "update_text": "I wrote the first section and am at 25 percent.",
            },
        },
    )
    budget = PostgreSQLRunBudget(sessions, actor)
    interval = await budget.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease)

    async def interrupt(run_id: UUID, result: AgentResult) -> None:
        assert result.status is AgentRunStatus.AWAITING_HUMAN, result
        raise RuntimeError("before child result publication")

    with candidate_budget(
        CandidateBudget(
            scope=scope, lease=lease, store=budget, interval_id=interval.reservation.operation_id
        )
    ):
        with monkeypatch.context() as patch:
            patch.setattr(recorder, "finish_agent_run", interrupt)
            first = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert first.kind == "REJECTED" and first.safe_code == "TOOL_RECONCILIATION_REQUIRED"
        replayed = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert isinstance(replayed, DeferredToolResult), replayed
        assert replayed.wait_ref.kind == "HUMAN_DECISION"
        assert replayed.wait_ref.resource_id is not None
        assert await pipeline.execute(scope, call, admitted=admitted, lease=lease) == replayed
    receipt_id = replayed.invocation_id
    assert (
        await receipts.recover_draft(
            scope, receipt_id=uuid4(), task_id=task_id, task_version=context.task_version
        )
        is None
    )
    foreign = scope.model_copy(update={"membership_id": uuid4()})
    assert (
        await receipts.recover_draft(
            foreign, receipt_id=receipt_id, task_id=task_id, task_version=context.task_version
        )
        is None
    )
    recovered = await receipts.recover_draft(
        scope, receipt_id=receipt_id, task_id=task_id, task_version=context.task_version
    )
    assert recovered is not None and recovered.assessment_id is not None
    assert (recovered.draft_id, recovered.draft_version) == (
        replayed.wait_ref.resource_id,
        replayed.wait_ref.resource_version,
    )
    draft = await updates.get_draft(actor, recovered.draft_id)
    assert draft.confirmed_update_id is None
    stale = ExecutionFenceContext(
        actor=actor,
        scope=session_scope(scope),
        lease=session_lease(lease.model_copy(update={"fencing_token": lease.fencing_token + 1})),
        operation_id=receipt_id,
        interval_id=interval.reservation.operation_id,
    )
    with execution_fence(stale), pytest.raises(SessionError, match="LEASE_LOST"):
        await updates.create_draft(actor, draft.items, str(uuid4()), "stale-worker")
    async with engine.connect() as connection:
        count = await connection.scalar(
            text("SELECT count(*) FROM daily_update_drafts WHERE organization_id=:org"),
            {"org": scope.organization_id},
        )
    assert count == 1
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE tasks SET version=version+1 WHERE organization_id=:org AND id=:id"),
            {"org": scope.organization_id, "id": task_id},
        )
    with candidate_budget(
        CandidateBudget(
            scope=scope, lease=lease, store=budget, interval_id=interval.reservation.operation_id
        )
    ):
        stale_replay = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
    assert stale_replay.kind == "REJECTED" and stale_replay.safe_code == "DELEGATION_WAIT_STALE"


async def test_planning_job_reclaim_rejects_an_old_attempt_from_the_same_worker(
    journal: Journal,
) -> None:
    engine, sessions, actor, _, scope, _, _, _ = journal
    factory = PostgreSQLPlanningRunTransactionFactory(sessions)
    wf_id = uuid4()
    await _seed_workflow_run(sessions, actor, wf_id)
    job_id = uuid4()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO workflow_jobs(id, organization_id, workflow_run_id, job_type, "
                "status, payload) VALUES (:id,:org,:run,'planning.start','QUEUED','{}'::jsonb)"
            ),
            {"id": job_id, "org": scope.organization_id, "run": wf_id},
        )
    now = datetime.now(UTC)
    async with factory(scope.organization_id) as tx:
        old = await tx.repository.claim_job(
            organization_id=scope.organization_id,
            worker_id="worker-a",
            now=now,
            lease_until=now + timedelta(seconds=60),
        )
        assert old is not None
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE workflow_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"),
            {"id": job_id},
        )
    async with factory(scope.organization_id) as tx:
        now = datetime.now(UTC)
        current = await tx.repository.claim_job(
            organization_id=scope.organization_id,
            worker_id="worker-a",
            now=now,
            lease_until=now + timedelta(seconds=60),
        )
        assert current is not None and current.attempt_count == old.attempt_count + 1
    async with factory(scope.organization_id) as tx:
        await tx.repository.fail_job(
            job_id=job_id,
            worker_id="worker-a",
            attempt_count=old.attempt_count,
            lease_until=old.lease_until,
            error_message="STALE",
            next_available_at=now,
        )
        await tx.repository.complete_job(
            job_id=job_id,
            worker_id="worker-a",
            attempt_count=old.attempt_count,
            lease_until=old.lease_until,
        )
    async with engine.connect() as connection:
        state = await connection.scalar(
            text("SELECT status FROM workflow_jobs WHERE id=:id"), {"id": job_id}
        )
    assert state == "RUNNING"
    async with factory(scope.organization_id) as tx:
        await tx.repository.complete_job(
            job_id=job_id,
            worker_id="worker-a",
            attempt_count=current.attempt_count,
            lease_until=current.lease_until,
        )
    async with engine.connect() as connection:
        state = await connection.scalar(
            text("SELECT status FROM workflow_jobs WHERE id=:id"), {"id": job_id}
        )
    assert state == "COMPLETED"
