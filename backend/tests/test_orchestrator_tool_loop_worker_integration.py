"""Candidate admission and durable worker execution against PostgreSQL."""

import os
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.core.database import create_database_engine, create_session_factory
from app.modules.assistant.adapters.agent_runtime import (
    CurrentAgentActorResolver,
    build_agent_registry,
)
from app.modules.assistant.adapters.candidate_access import CandidateAccess
from app.modules.assistant.adapters.child_job_receipts import PostgreSQLChildJobReceipts
from app.modules.assistant.adapters.delegation_tools import (
    BackendDelegationTool,
    PostgreSQLGoalSpecialistRunner,
    PostgreSQLWorkflowWaitVerifier,
)
from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
from app.modules.assistant.adapters.planning_tools import AssistantPlanningToolAdapter
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.execution_service import AssistantExecutionService
from app.modules.assistant.application.job_service import AssistantJobService
from app.modules.assistant.application.projection_service import AssistantProjectionService
from app.modules.assistant.application.service import AssistantService
from app.modules.assistant.domain.models import AssistantJob
from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthTransactionFactory
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.domain.roles import MembershipRole
from app.modules.planning_runs.adapters.ai_runtime import (
    PlanningAIRuntime,
    build_planning_job_handlers,
)
from app.modules.planning_runs.adapters.transaction import PostgreSQLPlanningRunTransactionFactory
from app.modules.planning_runs.application.job_service import JobService
from app.modules.planning_runs.application.proposal_service import ProposalService
from app.modules.planning_runs.application.run_service import PlanningRunService
from tests.test_orchestrator_session_log_integration import seed_members
from work_management_ai.agents.planning.harness import PlanningAgentHarness
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.runtime.context_exploration import ExplorationProjector
from work_management_ai.runtime.contracts import AgentId, JsonValue
from work_management_ai.runtime.delegation import DelegationAdapterV2
from work_management_ai.runtime.run_budget import BudgetedSpecialistTools, BudgetedToolLoopGateway
from work_management_ai.runtime.tool_loop_contracts import RunScope

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires PostgreSQL"),
]


class PlanningSnapshot:
    async def get_proposal_version(
        self, *, actor: AuthenticatedActor, proposal_id: UUID
    ) -> int | None:
        return None


@pytest.mark.parametrize("planning", [False, True])
async def test_candidate_worker_executes_the_saved_pin_after_rollback(planning: bool) -> None:
    await run_candidate_turn(planning=planning)


async def run_candidate_turn(
    *,
    planning: bool,
    interrupt: Callable[
        [AsyncEngine, AssistantJob, AssistantJobService, list[Exception]], Awaitable[None]
    ]
    | None = None,
    interrupt_resume: Callable[[AsyncEngine, UUID, UUID, AssistantJobService], Awaitable[None]]
    | None = None,
    lost_model_response: bool = False,
) -> None:
    from app.modules.assistant.adapters.tool_loop_runtime import (
        CandidateBackends,
        DriverAdmission,
        ToolLoopTurnExecutor,
        build_candidate_artifacts,
        register_candidate_specialists,
    )

    organization_id, membership_id = uuid4(), uuid4()
    settings = Settings(
        environment="test",
        ai_provider="mock",
        ai_model="native-test",
        progressive_planning_enabled=True,
        orchestrator_driver="tool_loop",
        orchestrator_tool_loop_organizations=(organization_id,),
    )
    engine = create_database_engine(settings)
    try:
        async with engine.begin() as connection:
            await seed_members(connection, ((organization_id, membership_id),))
            user_id = await connection.scalar(
                text("SELECT user_id FROM memberships WHERE id=:id"), {"id": membership_id}
            )
        assert isinstance(user_id, UUID)
        actor = AuthenticatedActor(
            user_id=user_id,
            email="actor@example.test",
            display_name="Minh",
            membership_id=membership_id,
            organization_id=organization_id,
            organization_name="Pilot",
            role=MembershipRole.MANAGER,
        )
        sessions = create_session_factory(engine)
        transactions = PostgreSQLAssistantTransactionFactory(sessions)
        actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
        artifacts = build_candidate_artifacts()
        service = AssistantService(
            transaction_factory=transactions,
            planning_snapshot=PlanningSnapshot(),
            orchestrator_version="1.0.0",
            orchestrator_fingerprint="legacy",
            driver_admission=DriverAdmission(settings, artifacts),
        )
        conversation = await service.create_conversation(
            actor=actor,
            locale="en",
            title="Test",
            request_id="create",
            idempotency_key="create",
        )
        submitted = await service.post_message(
            actor=actor,
            conversation_id=conversation.conversation.id,
            message="Plan a customer launch this week" if planning else "Hello",
            locale="en",
            card_action=None,
            if_match_version=None,
            request_id="turn",
            idempotency_key="turn",
        )
        assert submitted.run.orchestrator_version == "2.0.0"
        key = f"orchestrator.tool_loop:{submitted.run.id}"
        terminal: dict[str, JsonValue] = {
            "final": {
                "outcome": "ANSWER",
                "summary": "Planning draft is ready." if planning else "Hello!",
                "blocks": [
                    {
                        "kind": "text",
                        "text": "Planning draft is ready." if planning else "Hello!",
                        "citations": [],
                    }
                ],
                "artifact_state": "DRAFT_AVAILABLE" if planning else "NONE",
                "stop_reason": "DRAFT_AVAILABLE" if planning else "GREETING_ANSWERED",
            }
        }
        steps: tuple[object, ...] = (terminal,)
        if planning:
            steps = (
                {
                    "tool_call": {
                        "call_id": "planning-once",
                        "name": "delegate_agent",
                        "arguments": {
                            "agent_id": "planning",
                            "objective": "Plan a customer launch this week",
                            "context_ref_ids": [],
                            "constraints": [],
                            "expected_result": "Draft",
                        },
                    }
                },
                terminal,
            )
            if lost_model_response:
                steps = (steps[0], *steps)
        gateway = MockModelGateway(
            fixtures={"planning_agent.en.goal_intake": {"operation": "CREATE"}},
            usage_fixtures={
                key: ModelUsage(input_tokens=10, output_tokens=7),
                "planning_agent.en.goal_intake": ModelUsage(input_tokens=8, output_tokens=4),
            },
            tool_sequence_fixtures={key: steps},
        )
        registry, _ = build_agent_registry()
        register_candidate_specialists(registry, retrieval_enabled=False)
        planning_pin = next(
            pin
            for pin in registry.specialist_pins(goal_directed=True)
            if pin.reference == "planning"
        )
        planning_transactions = PostgreSQLPlanningRunTransactionFactory(sessions)
        planning_runtime = PlanningAIRuntime(progressive_enabled=True)
        planning_runs = PlanningRunService(
            transaction_factory=planning_transactions, runtime=planning_runtime
        )
        harness = PlanningAgentHarness(
            model_gateway=BudgetedToolLoopGateway(gateway),
            tool_executor=BudgetedSpecialistTools(
                AssistantPlanningToolAdapter(
                    actor_resolver=actors,
                    assistant_transaction_factory=transactions,
                    planning_run_service=planning_runs,
                    proposal_service=ProposalService(
                        transaction_factory=planning_transactions, runtime=planning_runtime
                    ),
                )
            ),
            actor_resolver=CurrentAgentActorResolver(actors),
        )

        async def backends(
            candidate_actor: AuthenticatedActor,
            scope: RunScope,
            job: AssistantJob,
            recorder: PostgreSQLExecutionRecorder,
            root_id: UUID,
            access: CandidateAccess,
            projector: ExplorationProjector,
            locale: Literal["vi", "en"],
        ) -> CandidateBackends:
            delegate = DelegationAdapterV2(
                scope=scope,
                parent_agent_run_id=root_id,
                actor_role=candidate_actor.role.value,
                active_phase=5,
                locale=locale,
                timezone="UTC",
                agent_registry=registry,
                specialist_pins=(planning_pin,),
                runners={
                    AgentId.PLANNING: PostgreSQLGoalSpecialistRunner(
                        recorder=recorder,
                        harness=harness,
                        pin=planning_pin,
                    )
                },
                tool_registry=artifacts.tools,
                workflow_verifier=PostgreSQLWorkflowWaitVerifier(
                    session_factory=sessions, actor=candidate_actor
                ),
                child_job_receipts=PostgreSQLChildJobReceipts(
                    sessions=sessions, actor=candidate_actor
                ),
            )
            return CandidateBackends(
                {
                    "delegate_agent@2": BackendDelegationTool(
                        actor=candidate_actor, adapter=delegate
                    )
                },
                specialists=lambda _: delegate.available_specialists(),
            )

        executor = ToolLoopTurnExecutor(
            sessions=sessions,
            transactions=transactions,
            actors=actors,
            admission=DriverAdmission(
                settings.model_copy(update={"orchestrator_driver": "legacy"}), artifacts
            ),
            artifacts=artifacts,
            gateway=gateway,
            backend_factory=backends,
            specialist_registry=registry,
        )
        execution = AssistantExecutionService(actor_resolver=actors, runtime=executor)
        unexpected: list[Exception] = []

        async def execute(*, job: AssistantJob, worker_id: str) -> None:
            try:
                await execution.execute(job=job, worker_id=worker_id)
            except Exception as error:
                unexpected.append(error)
                raise

        worker = AssistantJobService(
            transaction_factory=transactions,
            handler=execute,
            organization_scopes={organization_id},
        )
        if interrupt is not None:
            await interrupt(engine, submitted.job, worker, unexpected)
        assert await worker.run_once(worker_id="candidate-test", organization_id=organization_id)
        assert not await worker.run_once(
            worker_id="candidate-test", organization_id=organization_id
        )
        if unexpected:
            raise unexpected[0]
        if planning:
            async with engine.connect() as connection:
                waiting = (
                    await connection.execute(
                        text("SELECT status, stop_reason FROM orchestration_runs WHERE id=:id"),
                        {"id": submitted.run.id},
                    )
                ).one()
            assert waiting.status == "RUNNING", waiting
            assert len(gateway.step_requests) == 1 + lost_model_response
            delegate_schema = next(
                tool
                for tool in gateway.step_requests[0].tools
                if tool.reference == "delegate_agent@2"
            )
            properties = delegate_schema.parameters["properties"]
            assert isinstance(properties, dict)
            assert set(properties) == {
                "agent_id",
                "objective",
                "context_ref_ids",
                "constraints",
                "expected_result",
            }
            planning_worker = JobService(
                transaction_factory=planning_transactions,
                handlers=build_planning_job_handlers(
                    settings, planning_transactions, actors, candidate_sessions=sessions
                ),
                organization_scopes={organization_id},
            )
            assert await planning_worker.run_once("planning-test", organization_id)
            projection = AssistantProjectionService(transaction_factory=transactions)
            assert await projection.project_once(organization_id=organization_id)
            wake_worker = AssistantJobService(
                transaction_factory=transactions,
                handler=execute,
                organization_scopes={organization_id},
                job_type="assistant.turn.resume",
            )
            if interrupt_resume is not None:
                await interrupt_resume(engine, organization_id, submitted.run.id, wake_worker)
            assert await wake_worker.run_once(
                worker_id="resume-test", organization_id=organization_id
            )
            if unexpected:
                raise unexpected[0]
            assert len(gateway.step_requests) == 2 + lost_model_response
        gateway.assert_complete()
        async with engine.connect() as connection:
            result = await connection.scalar(
                text("SELECT status FROM orchestration_runs WHERE id=:id"),
                {"id": submitted.run.id},
            )
            agent_version = await connection.scalar(
                text(
                    "SELECT agent_version FROM agent_runs WHERE orchestration_run_id=:id "
                    "AND agent_id='orchestrator'"
                ),
                {"id": submitted.run.id},
            )
            if planning:
                closed = await connection.scalar(
                    text("SELECT closed FROM orchestration_waits WHERE owner_run_id=:id"),
                    {"id": submitted.run.id},
                )
                assert closed is True
                if interrupt is not None or interrupt_resume is not None:
                    counts = await connection.execute(
                        text(
                            "SELECT "
                            "(SELECT count(*) FROM agent_runs WHERE orchestration_run_id=:run), "
                            "(SELECT count(*) FROM workflow_runs WHERE organization_id=:org), "
                            "(SELECT count(*) FROM planning_drafts WHERE organization_id=:org), "
                            "(SELECT count(*) FROM proposals WHERE organization_id=:org), "
                            "(SELECT count(*) FROM assistant_messages WHERE turn_id=:turn "
                            "AND role='ASSISTANT'), "
                            "(SELECT count(*) FROM assistant_jobs WHERE orchestration_run_id=:run "
                            "AND job_type='assistant.turn.resume'), "
                            "(SELECT count(*) FROM workflow_jobs WHERE organization_id=:org), "
                            "(SELECT count(*) FROM audit_events WHERE organization_id=:org "
                            "AND action='planning_run.created'), "
                            "(SELECT count(*) FROM orchestration_usage WHERE "
                            "orchestration_run_id=:run AND kind='INTERVAL'), "
                            "(SELECT count(*) FROM orchestration_session_events WHERE "
                            "orchestration_run_id=:run AND kind='tool.started'), "
                            "(SELECT count(*) FROM assistant_events WHERE turn_id=:turn "
                            "AND event_type='assistant.turn.response.v1')"
                        ),
                        {
                            "run": submitted.run.id,
                            "org": organization_id,
                            "turn": submitted.turn.id,
                        },
                    )
                    assert tuple(counts.one()) == (2, 1, 1, 0, 1, 1, 1, 1, 4, 1, 1)
        if unexpected:
            raise unexpected[0]
        assert result == "COMPLETED"
        assert agent_version == "2.0.0"
        if not planning:
            legacy_service = AssistantService(
                transaction_factory=transactions,
                planning_snapshot=PlanningSnapshot(),
                orchestrator_version="1.0.0",
                orchestrator_fingerprint="legacy",
                driver_admission=DriverAdmission(
                    settings.model_copy(update={"orchestrator_driver": "legacy"}), artifacts
                ),
            )
            legacy_turn = await legacy_service.post_message(
                actor=actor,
                conversation_id=conversation.conversation.id,
                message="Hello again",
                locale="en",
                card_action=None,
                if_match_version=None,
                request_id="legacy-turn",
                idempotency_key="legacy-turn",
            )
            assert legacy_turn.run.orchestrator_version == "1.0.0"
            assert "driver_pin" not in legacy_turn.run.execution_plan
    finally:
        await engine.dispose()
