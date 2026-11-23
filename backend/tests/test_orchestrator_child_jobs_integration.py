"""Exercise candidate enqueue receipts through the real application transaction."""

import json
import os
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

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
    DelegationError,
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
    ToolInvocationContext,
)
from work_management_ai.runtime.tool_pipeline import ToolPipeline
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.runtime.triggers import ChatTurnTrigger, ExecutionScope

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="PostgreSQL opt-in"),
]


async def _seed_planning_target(
    engine: AsyncEngine,
    scope: RunScope,
    operation: Literal["RESUME_INPUT", "REVISE"],
) -> tuple[ContextReference, ...]:
    workflow, proposal, approval = uuid4(), uuid4(), uuid4()
    resume = operation == "RESUME_INPUT"
    ids = {
        "org": scope.organization_id,
        "actor": scope.membership_id,
        "run": workflow,
        "proposal": proposal,
        "approval": approval,
        "turn": scope.turn_id,
        "parent": scope.orchestration_run_id,
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO workflow_runs (id, organization_id, requested_by_membership_id, "
                "status, workflow_name, workflow_version, verifier_version, input_goal_text) "
                "VALUES (:run, :org, :actor, :status, 'project_planning', '3.0.0', '1.0.0', "
                "'Plan the launch')"
            ),
            {**ids, "status": "NEEDS_INPUT" if resume else "WAITING_FOR_DECISION"},
        )
        if resume:
            await conn.execute(
                text(
                    "INSERT INTO workflow_checkpoints (id, organization_id, workflow_run_id, "
                    "node, sequence, state) VALUES "
                    "(:id, :org, :run, 'await_manager_input', 1, '{}')"
                ),
                {**ids, "id": uuid4()},
            )
            await conn.execute(
                text(
                    "INSERT INTO agent_runs (id, organization_id, orchestration_run_id, "
                    "agent_id, agent_version, manifest_fingerprint, capability, typed_input, "
                    "budget, status, workflow_run_id) VALUES "
                    "(:id, :org, :parent, 'planning', '1.0.0', :fingerprint, "
                    "'planning.create', '{}', '{}', 'COMPLETED', :run)"
                ),
                {**ids, "id": uuid4(), "fingerprint": "a" * 64},
            )
        else:
            await conn.execute(
                text(
                    "INSERT INTO proposals (id, organization_id, workflow_run_id, status, "
                    "current_version_number) VALUES "
                    "(:proposal, :org, :run, 'READY_FOR_DECISION', 1)"
                ),
                ids,
            )
            await conn.execute(
                text(
                    "INSERT INTO proposal_versions (id, organization_id, proposal_id, "
                    "version_number, created_by_membership_id, content, assumptions) VALUES "
                    "(:id, :org, :proposal, 1, :actor, '{}', '[]')"
                ),
                {**ids, "id": uuid4()},
            )
            await conn.execute(
                text(
                    "INSERT INTO approvals (id, organization_id, proposal_id, "
                    "proposal_version_number, status) VALUES "
                    "(:approval, :org, :proposal, 1, 'PENDING')"
                ),
                ids,
            )
            await conn.execute(
                text("UPDATE proposals SET approval_id = :approval WHERE id = :proposal"),
                ids,
            )
            await conn.execute(
                text(
                    "UPDATE assistant_messages SET content_blocks = CAST(:blocks AS jsonb) "
                    "WHERE organization_id = :org AND id = "
                    "(SELECT user_message_id FROM assistant_turns WHERE id = :turn)"
                ),
                {
                    **ids,
                    "blocks": json.dumps(
                        [
                            {
                                "kind": "accepted_card_action",
                                "action": {
                                    "kind": "PLANNING_REVISE",
                                    "proposal_id": str(proposal),
                                    "expected_version": 1,
                                },
                            }
                        ]
                    ),
                },
            )
    resources = (("workflow_run", workflow),)
    if not resume:
        resources += (("proposal", proposal),)
    return tuple(
        ContextReference(
            reference_id=uuid4(),
            resource_type=kind,
            resource_id=resource,
            organization_id=scope.organization_id,
            version=1,
            observed_at=datetime.now(UTC),
        )
        for kind, resource in resources
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_id", "operation", "job_type", "interrupted"),
    [
        (AgentId.PLANNING, "CREATE", "planning.start", True),
        (AgentId.PLANNING, "RESUME_INPUT", "planning.resume", True),
        (AgentId.PLANNING, "REVISE", "proposal.ai_revise", True),
        (AgentId.PLANNING, "REVISE", "proposal.ai_revise", False),
        (AgentId.REPORTING, "CREATE", None, True),
    ],
)
async def test_uses_canonical_child_job_for_dispatch_and_replay(
    journal: Journal,
    agent_id: Literal[AgentId.PLANNING, AgentId.REPORTING],
    operation: Literal["CREATE", "RESUME_INPUT", "REVISE"],
    job_type: str | None,
    interrupted: bool,
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
    planning_runs = PlanningRunService(transaction_factory=planning, runtime=runtime)
    actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
    refs = None
    context_ids: list[JsonValue] = []
    planning_refs: tuple[ContextReference, ...] = ()
    if operation != "CREATE":
        planning_refs = await _seed_planning_target(engine, scope, operation)

        async def check_planning_refs(
            target: RunScope, selected: tuple[ContextReference, ...]
        ) -> None:
            assert target == scope and selected == planning_refs
            snapshot = await planning_runs.get_workflow_run_snapshot(
                actor=actor, run_id=planning_refs[0].resource_id
            )
            assert snapshot.run.id == planning_refs[0].resource_id
            if operation == "REVISE":
                assert snapshot.proposal is not None
                assert snapshot.proposal.id == planning_refs[1].resource_id
                assert any(
                    version is not None and version.version_number == planning_refs[1].version
                    for version in (snapshot.proposal_version, snapshot.previous_proposal_version)
                )

        refs = VerifiedDelegationReferences(
            scope=scope,
            exploration=(),
            active_context=planning_refs,
            completed_specialists=(),
            recheck=check_planning_refs,
        )
        context_ids = [str(ref.reference_id) for ref in planning_refs]

    class PlanningAccess:
        async def verify(self, handoff: AgentHandoffV2) -> None:
            assert refs is not None
            await refs.revalidate(scope, handoff.context_references)

    mock = MockModelGateway(
        fixtures={"planning_agent.en.goal_intake": {"operation": operation}},
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
                planning_run_service=planning_runs,
                proposal_service=ProposalService(transaction_factory=planning, runtime=runtime),
            )
        ),
        actor_resolver=CurrentAgentActorResolver(actors),
        goal_access=PlanningAccess(),
    )
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

    class ChildHarness:
        async def run(self, handoff: AgentHandoffV2):
            result = await real.run(handoff)
            job = result.typed_output.get(key)
            if isinstance(job, str):
                finished.append(UUID(job))
            if interrupted:
                raise RuntimeError("interrupted before saving the child result")
            return result

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
                recorder=recorder, harness=ChildHarness(), pin=pin
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
        if interrupted:
            assert first.kind == "REJECTED", first
        assert len(finished) == 1
        result = (
            await pipeline.execute(scope, call, admitted=admitted, lease=lease)
            if interrupted
            else first
        )
        assert isinstance(result, DeferredToolResult), result
        assert result.wait_ref.kind == "WORKFLOW_RESULT"
        assert result.wait_ref.resource_id == finished[0]
        assert await pipeline.execute(scope, call, admitted=admitted, lease=lease) == result
    arguments = schema.input_contract.model_validate(call.arguments)
    receipt = await log.get_invocation(scope, result.invocation_id)
    assert receipt is not None
    context = ToolInvocationContext(
        scope=scope,
        lease=lease,
        invocation_id=result.invocation_id,
        call_id=call.call_id,
        tool_reference=schema.reference,
        tool_version=item.pin.version,
        arguments_fingerprint=receipt.descriptor.arguments_fingerprint,
        canonical_arguments=arguments,
        policy_version="child-jobs",
        receipt_id=result.invocation_id,
        budget_allocation_id=root.reservation.operation_id,
        limits=pipeline.limits,
    )
    async with engine.connect() as conn:
        if agent_id is AgentId.PLANNING:
            assert (
                await conn.scalar(
                    text(
                        "SELECT count(*) FROM workflow_jobs WHERE organization_id=:org "
                        "AND workflow_run_id=:resource AND job_type=:job_type"
                    ),
                    {"org": scope.organization_id, "resource": finished[0], "job_type": job_type},
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
    if operation == "REVISE":
        revision_job_id = None
        for status in ("FAILED", "COMPLETED"):
            async with engine.begin() as conn:
                revision_job_id = await conn.scalar(
                    text(
                        "UPDATE workflow_jobs SET status=:status WHERE organization_id=:org "
                        "AND workflow_run_id=:resource AND job_type=:job_type RETURNING id"
                    ),
                    {
                        "org": scope.organization_id,
                        "resource": finished[0],
                        "job_type": job_type,
                        "status": status,
                    },
                )
            with pytest.raises(DelegationError, match="DELEGATION_WORKFLOW_FAILED"):
                await adapter.reconcile(context, arguments)
        assert isinstance(revision_job_id, UUID)
        proposal_id = next(
            ref.resource_id for ref in planning_refs if ref.resource_type == "proposal"
        )
        validation: dict[str, object] = {"can_approve": True, "errors": [], "warnings": []}
        async with planning(actor) as transaction:
            new_version = await transaction.repository.finalize_ai_revision_mutation(
                actor=actor,
                proposal_id=proposal_id,
                base_version=1,
                content={"tasks": []},
                change_summary="Update launch scope",
                model_reference="mock",
                validation_result=validation,
                request_id=f"workflow-job:{revision_job_id}",
                idempotency_key=f"proposal-ai-revise:{revision_job_id}",
            )
            assert new_version is not None and new_version.version_number == 2
            await transaction.commit()
        revalidating = await adapter.reconcile(context, arguments)
        assert revalidating is not None and revalidating.wait_ref is not None
        assert revalidating.wait_ref.kind == "WORKFLOW_RESULT"
        async with planning(actor) as transaction:
            await transaction.repository.complete_proposal_revalidation(
                actor=actor,
                proposal_id=proposal_id,
                version_number=new_version.version_number,
                validation_result=validation,
                request_id="test-revalidate",
            )
            await transaction.commit()
        awaiting = await adapter.reconcile(context, arguments)
        assert awaiting is not None and awaiting.wait_ref is not None
        assert awaiting.wait_ref.kind == "HUMAN_DECISION"
        assert awaiting.wait_ref.resource_id == proposal_id
        assert awaiting.wait_ref.resource_version == new_version.version_number
    if operation != "CREATE":
        return
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
    review_proposal: tuple[UUID, int] | None = None
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
                        "SELECT job.state, job.proposed_version_id, job.safe_error_code, "
                        "report.version AS report_version FROM report_generation_jobs job "
                        "JOIN reports report ON (report.organization_id, report.id) = "
                        "(job.organization_id, job.report_id) "
                        "WHERE job.organization_id=:org AND job.id=:resource"
                    ),
                    {"org": scope.organization_id, "resource": finished[0]},
                )
            ).one()
            assert state.state == "AWAITING_REVIEW", state
            assert state.proposed_version_id is not None
            review_proposal = (state.proposed_version_id, state.report_version)
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
    if agent_id is AgentId.REPORTING:
        assert review_proposal is not None
        recovered = await adapter.reconcile(context, arguments)
        assert recovered is not None and recovered.wait_ref is not None
        assert recovered.wait_ref.kind == "HUMAN_DECISION"
        assert (recovered.wait_ref.resource_id, recovered.wait_ref.resource_version) == (
            review_proposal
        )
