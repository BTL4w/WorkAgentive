"""Exercise candidate enqueue receipts through the real application transaction."""

import os
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.modules.assistant.adapters.agent_runtime import (
    CurrentAgentActorResolver,
    build_agent_registry,
)
from app.modules.assistant.adapters.child_job_receipts import PostgreSQLChildJobReceipts
from app.modules.assistant.adapters.delegation_tools import (
    BackendDelegationTool,
    PostgreSQLGoalSpecialistRunner,
    PostgreSQLWorkflowWaitVerifier,
)
from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
from app.modules.assistant.adapters.planning_tools import AssistantPlanningToolAdapter
from app.modules.assistant.adapters.report_projection import ReportStatusContexts
from app.modules.assistant.adapters.reporting_tools import ChatReportingToolAdapter
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthTransactionFactory
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.planning_runs.adapters.ai_runtime import (
    PlanningAIRuntime,
    build_model_gateway,
    build_planning_job_handlers,
)
from app.modules.planning_runs.adapters.transaction import PostgreSQLPlanningRunTransactionFactory
from app.modules.planning_runs.application.job_service import JobService
from app.modules.planning_runs.application.proposal_service import ProposalService
from app.modules.planning_runs.application.run_service import PlanningRunService
from app.modules.reporting.adapters.generation_repository import GenerationTransactions
from app.modules.reporting.adapters.narrative_runtime import ReportNarrativeRuntime
from app.modules.reporting.adapters.transaction import ReportTransactions
from app.modules.reporting.application.chat_service import ReportChatService
from app.modules.reporting.application.job_service import ReportJobService
from app.modules.reporting.application.report_service import ReportService
from app.modules.work.adapters.project_repository import SqlAlchemyProjectTransactionFactory
from app.modules.work.application.project_service import ProjectService
from tests.test_orchestrator_session_log_integration import Journal
from tests.test_orchestrator_session_log_integration import journal as journal
from tests.test_orchestrator_tool_pipeline_integration import Policy
from work_management_ai.agents.planning.harness import PlanningAgentHarness
from work_management_ai.agents.reporting.harness import ReportingHarness
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.tool_contracts import ToolCall
from work_management_ai.runtime.contracts import (
    AgentHandoffV2,
    AgentId,
    ContextReference,
    JsonValue,
)
from work_management_ai.runtime.delegation import (
    DelegationAdapterV2,
    VerifiedDelegationReferences,
    filtered_delegate_v2_tool_schema,
)
from work_management_ai.runtime.run_budget import (
    BudgetedSpecialistTools,
    BudgetedToolLoopGateway,
    CandidateBudget,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.tool_catalog import PreparedToolSet
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    DeferredToolResult,
    RunScope,
)
from work_management_ai.runtime.tool_pipeline import ToolPipeline
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.runtime.triggers import ChatTurnTrigger, ExecutionScope

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", [AgentId.PLANNING, AgentId.REPORTING])
async def test_reconciles_canonical_job_when_child_result_was_interrupted(
    journal: Journal,
    agent_id: Literal[AgentId.PLANNING, AgentId.REPORTING],
) -> None:
    engine, sessions, actor, log, scope, lease, _, _ = journal
    registry, _ = build_agent_registry()
    registry.register_resource(f"work_management_ai.agents.{agent_id.value}", "agent.v2.yaml")
    pin = next(
        pin
        for pin in registry.specialist_pins(goal_directed=True)
        if pin.reference == agent_id.value
    )
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
    planning = PostgreSQLPlanningRunTransactionFactory(sessions)
    runtime = PlanningAIRuntime(progressive_enabled=True)
    actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
    mock = MockModelGateway(
        fixtures={"planning_agent.en.goal_intake": {"operation": "CREATE"}},
        usage_fixtures={
            "planning_agent.en.goal_intake": ModelUsage(input_tokens=30, output_tokens=10)
        },
    )
    planning_harness = PlanningAgentHarness(
        model_gateway=BudgetedToolLoopGateway(mock),
        tool_executor=BudgetedSpecialistTools(
            AssistantPlanningToolAdapter(
                actor_resolver=actors,
                assistant_transaction_factory=transactions,
                planning_run_service=PlanningRunService(
                    transaction_factory=planning, runtime=runtime
                ),
                proposal_service=ProposalService(transaction_factory=planning, runtime=runtime),
            )
        ),
        actor_resolver=CurrentAgentActorResolver(actors),
    )
    refs = None
    context_ids: list[JsonValue] = []
    if agent_id is AgentId.REPORTING:
        project_id = uuid4()
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO projects (id, organization_id, name, created_by_membership_id, "
                    "updated_by_membership_id) VALUES (:id, :org, 'Launch', :actor, :actor)"
                ),
                {"id": project_id, "org": scope.organization_id, "actor": scope.membership_id},
            )
        projects = ProjectService(SqlAlchemyProjectTransactionFactory(sessions))
        ref = ContextReference(
            reference_id=uuid4(),
            organization_id=scope.organization_id,
            resource_type="project",
            resource_id=project_id,
            version=1,
            observed_at=datetime.now(UTC),
        )

        async def check_refs(target: RunScope, selected: tuple[ContextReference, ...]) -> None:
            assert target == scope and selected == (ref,)
            project = await projects.get_project(actor=actor, project_id=project_id)
            assert project.version == ref.version

        class ReportAccess:
            async def verify(self, handoff: AgentHandoffV2) -> None:
                await check_refs(scope, handoff.context_references)

        refs = VerifiedDelegationReferences(
            scope=scope,
            exploration=(),
            active_context=(ref,),
            completed_specialists=(),
            recheck=check_refs,
        )
        context_ids = [str(ref.reference_id)]
        report_gateway = MockModelGateway(
            fixtures={
                "reporting.en.goal_intake": {"operation": "PREPARE_REPORT", "kind": "WEEKLY"}
            },
            usage_fixtures={
                "reporting.en.goal_intake": ModelUsage(input_tokens=30, output_tokens=10)
            },
        )
        report_harness = ReportingHarness(
            model_gateway=BudgetedToolLoopGateway(report_gateway),
            tool_executor=BudgetedSpecialistTools(
                ChatReportingToolAdapter(
                    actors=actors,
                    service=ReportChatService(
                        reports=ReportService(ReportTransactions(sessions, "UTC")),
                        projects=projects,
                    ),
                    contexts=ReportStatusContexts(sessions, "UTC"),
                )
            ),
            actor_resolver=CurrentAgentActorResolver(actors),
            goal_access=ReportAccess(),
        )
        real = report_harness
        key = "generation_id"
    else:
        real = planning_harness
        key = "workflow_run_id"

    finished: list[UUID] = []

    class Interrupted:
        async def run(self, handoff: AgentHandoffV2):
            result = await real.run(handoff)
            job = result.typed_output.get(key)
            if isinstance(job, str):
                finished.append(UUID(job))
            raise RuntimeError("interrupted before saving the child result")

    tools = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.v2.yaml"),)
    )
    adapter = DelegationAdapterV2(
        scope=scope,
        parent_agent_run_id=parent,
        actor_role="MANAGER",
        active_phase=5,
        locale="en",
        timezone="UTC",
        agent_registry=registry,
        specialist_pins=(pin,),
        runners={
            agent_id: PostgreSQLGoalSpecialistRunner(
                recorder=recorder, harness=Interrupted(), pin=pin
            )
        },
        tool_registry=tools,
        references=refs,
        workflow_verifier=PostgreSQLWorkflowWaitVerifier(session_factory=sessions, actor=actor),
        child_job_receipts=PostgreSQLChildJobReceipts(sessions=sessions, actor=actor),
    )
    (item,) = tools.candidate_definitions(("delegate_agent@2",))
    schema = filtered_delegate_v2_tool_schema(
        item.schema,
        allowed_specialists=adapter.available_specialists(),
    )
    assert schema.input_contract is not None
    admitted = PreparedToolSet(
        (
            AdmittedTool(
                schema=schema,
                input_contract=schema.input_contract,
                output_contract=item.output_contract,
                version=item.pin.version,
                fingerprint=item.pin.fingerprint,
            ),
        ),
        "child-jobs",
        lambda _: True,
    )
    pipeline = ToolPipeline(
        store=log,
        policy=Policy(),
        backends={"delegate_agent@2": BackendDelegationTool(actor=actor, adapter=adapter)},
    )
    budget = PostgreSQLRunBudget(sessions, actor)
    root = await budget.reserve(scope, UsageReservation(kind="INTERVAL"), lease=lease)
    call = ToolCall(
        call_id=str(uuid4()),
        name="delegate_agent",
        arguments={
            "agent_id": agent_id.value,
            "objective": "Prepare a draft for the customer launch",
            "expected_result": "Draft",
            "context_ref_ids": context_ids,
        },
    )
    with candidate_budget(CandidateBudget(scope, lease, budget, root.reservation.operation_id)):
        first = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert first.kind == "REJECTED", first
        assert len(finished) == 1
        result = await pipeline.execute(scope, call, admitted=admitted, lease=lease)
        assert isinstance(result, DeferredToolResult), result
        assert result.wait_ref.kind == "WORKFLOW_RESULT"
        assert result.wait_ref.resource_id == finished[0]
        assert await pipeline.execute(scope, call, admitted=admitted, lease=lease) == result
    async with engine.connect() as conn:
        if agent_id is AgentId.PLANNING:
            assert (
                await conn.scalar(
                    text(
                        "SELECT count(*) FROM workflow_jobs WHERE organization_id=:org "
                        "AND workflow_run_id=:resource"
                    ),
                    {"org": scope.organization_id, "resource": finished[0]},
                )
                == 1
            )
        else:
            assert (
                await conn.scalar(
                    text(
                        "SELECT count(*) FROM report_generation_jobs WHERE organization_id=:org "
                        "AND id=:resource"
                    ),
                    {"org": scope.organization_id, "resource": finished[0]},
                )
                == 1
            )
        assert (
            await conn.scalar(
                text(
                    "SELECT count(*) FROM orchestration_child_jobs WHERE organization_id=:org "
                    "AND orchestration_run_id=:run"
                ),
                {"org": scope.organization_id, "run": scope.orchestration_run_id},
            )
            == 1
        )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE orchestration_runtime_states SET lease_until=clock_timestamp() "
                "WHERE organization_id=:org AND orchestration_run_id=:run"
            ),
            {"org": scope.organization_id, "run": scope.orchestration_run_id},
        )
    settings = Settings(environment="test", ai_provider="mock")
    if agent_id is AgentId.PLANNING:
        planning_worker = JobService(
            transaction_factory=planning,
            handlers=build_planning_job_handlers(
                settings,
                planning,
                actors,
                candidate_sessions=sessions,
            ),
            organization_scopes={scope.organization_id},
        )
        assert await planning_worker.run_once("child-worker", scope.organization_id)
    else:
        reporting_worker = ReportJobService(
            GenerationTransactions(sessions, "UTC"),
            ReportNarrativeRuntime(
                sessions=sessions,
                actors=actors,
                gateway=build_model_gateway(settings),
                timezone="UTC",
            ),
        )
        assert await reporting_worker.run_once(
            worker_id="child-worker",
            organization_id=scope.organization_id,
        )
    async with engine.connect() as conn:
        if agent_id is AgentId.PLANNING:
            assert (
                await conn.scalar(
                    text(
                        "SELECT status FROM workflow_jobs WHERE organization_id=:org "
                        "AND workflow_run_id=:resource AND job_type='planning.start'"
                    ),
                    {"org": scope.organization_id, "resource": finished[0]},
                )
                == "COMPLETED"
            )
        else:
            state = (
                await conn.execute(
                    text(
                        "SELECT state, proposed_version_id, safe_error_code "
                        "FROM report_generation_jobs "
                        "WHERE organization_id=:org AND id=:resource"
                    ),
                    {"org": scope.organization_id, "resource": finished[0]},
                )
            ).one()
            assert state.state == "AWAITING_REVIEW", state
            assert state.proposed_version_id is not None
            assert (
                await conn.scalar(
                    text("SELECT count(*) FROM report_publications WHERE organization_id=:org"),
                    {"org": scope.organization_id},
                )
                == 0
            )
        entries = (
            await conn.execute(
                text(
                    "SELECT kind, closed FROM orchestration_usage "
                    "WHERE organization_id=:org AND orchestration_run_id=:run AND job_owned"
                ),
                {"org": scope.organization_id, "run": scope.orchestration_run_id},
            )
        ).all()
        assert any(kind == "MODEL" for kind, _ in entries), entries
        assert all(closed for _, closed in entries), entries
