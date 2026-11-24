"""Compose the existing Specialist harnesses within a pinned Assistant run."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.agent_runtime import (
    CurrentActorResolverPort,
    CurrentAgentActorResolver,
)
from app.modules.assistant.adapters.assignment_receipts import PostgreSQLAssignmentReceiptReader
from app.modules.assistant.adapters.assignment_tools import (
    AssistantAssignmentContextResolver,
    AssistantAssignmentExplicitCommandAdapter,
)
from app.modules.assistant.adapters.candidate_access import CandidateAccess
from app.modules.assistant.adapters.child_job_receipts import PostgreSQLChildJobReceipts
from app.modules.assistant.adapters.delegation_tools import (
    BackendDelegationTool,
    GoalSpecialistHarness,
    PostgreSQLDailyUpdateReceiptReader,
    PostgreSQLGoalSourceTurnReader,
    PostgreSQLGoalSpecialistRunner,
    PostgreSQLWorkflowWaitVerifier,
)
from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
from app.modules.assistant.adapters.orchestration_tools import ApplicationToolBackend
from app.modules.assistant.adapters.retrieval_runtime import RetrievalRuntime
from app.modules.assistant.adapters.risk_tools import RiskToolAdapter
from app.modules.assistant.adapters.tool_loop_runtime import CandidateBackends
from app.modules.assistant.domain.models import AssistantJob
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.agents.assignment.harness import AssignmentAgentHarness
from work_management_ai.agents.daily_update.harness import DailyUpdateHarness
from work_management_ai.agents.planning.harness import PlanningAgentHarness
from work_management_ai.agents.reporting.harness import ReportingHarness
from work_management_ai.agents.retrieve.harness import RetrieveHarness
from work_management_ai.agents.risk.harness import RiskHarness
from work_management_ai.agents.work_intelligence.harness import WorkIntelligenceHarness
from work_management_ai.model_gateway.contracts import ModelGateway
from work_management_ai.runtime.activation import ActivationContext
from work_management_ai.runtime.agent_registry import AgentRegistry
from work_management_ai.runtime.context_exploration import ExplorationProjector
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentHandoffV2,
    AgentId,
    AgentResult,
    ContextReference,
    DailySummaryResponseBlock,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutorPort,
)
from work_management_ai.runtime.delegation import DelegationAdapterV2, DelegationError
from work_management_ai.runtime.run_budget import BudgetedSpecialistTools, BudgetedToolLoopGateway
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    RunScope,
    ToolInvocationContext,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry


class CandidateReferences:
    def __init__(self, policy: CandidateAccess, projector: ExplorationProjector) -> None:
        self.policy, self.projector = policy, projector
        self.evidence: dict[UUID, ContextReference] = {}

    async def resolve(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]:
        refs = self.recorded(scope, reference_ids)
        await self.revalidate(scope, refs)
        return refs

    def recorded(
        self, scope: RunScope, reference_ids: tuple[UUID, ...]
    ) -> tuple[ContextReference, ...]:
        if scope != self.projector.scope:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        values = dict(self.evidence)
        report = self.projector.current
        if report is not None:
            for ref in report.context_refs:
                values[ref.ref_id] = ContextReference(
                    reference_id=ref.ref_id,
                    organization_id=scope.organization_id,
                    resource_type=ref.resource_type,
                    resource_id=ref.resource_id,
                    version=ref.version,
                    fingerprint=ref.fingerprint,
                    observed_at=ref.observed_at,
                )
        try:
            return tuple(values[ref_id] for ref_id in reference_ids)
        except KeyError:
            raise DelegationError("DELEGATION_CONTEXT_DENIED") from None

    async def revalidate(self, scope: RunScope, refs: tuple[ContextReference, ...]) -> None:
        if scope != self.projector.scope:
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        report = self.projector.current
        if report is not None:
            blocked = self.projector.unresolved_resources(report)
            if any((ref.resource_type.casefold(), ref.resource_id) in blocked for ref in refs):
                raise DelegationError("DELEGATION_CLARIFICATION_REQUIRED")
        await self.policy.assert_refs(scope, refs)

    def observe(self, refs: tuple[ContextReference, ...]) -> None:
        for ref in refs:
            if ref.organization_id != self.projector.scope.organization_id:
                raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
            known = self.evidence.setdefault(ref.reference_id, ref)
            if known != ref:
                raise DelegationError("DELEGATION_CONTEXT_CONFLICT", stop_segment=True)

    async def verify(self, handoff: AgentHandoffV2) -> None:
        scope = self.projector.scope
        if (
            handoff.actor.organization_id != scope.organization_id
            or handoff.actor.membership_id != scope.membership_id
            or handoff.orchestration_run_id != scope.orchestration_run_id
            or handoff.source_turn_ref != scope.turn_id
        ):
            raise DelegationError("DELEGATION_CONTEXT_DENIED", stop_segment=True)
        await self.revalidate(scope, handoff.context_references)


@dataclass(frozen=True, slots=True)
class _ReadTools:
    work: ToolExecutorPort
    risk: ToolExecutorPort

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        return await (self.risk if request.tool_id == "risk.read" else self.work).execute(request)


@dataclass(frozen=True, slots=True)
class _Retrieve:
    gateway: ModelGateway
    runtime: RetrievalRuntime
    actors: CurrentAgentActorResolver
    source: PostgreSQLGoalSourceTurnReader
    activation: ActivationContext | None

    async def run(self, handoff: AgentHandoffV2) -> AgentResult:
        return await RetrieveHarness(
            model_gateway=self.gateway,
            tool_executor=self.runtime,
            usage_port=await self.runtime.usage(handoff),
            actor_resolver=self.actors,
            activation=self.activation,
            source_turns=self.source,
            public_query_validator=self.runtime,
        ).run(handoff)


class _AutomationPreview:
    def __init__(self, actor: AuthenticatedActor, root_id: UUID, tools: ToolExecutorPort) -> None:
        self.actor, self.root_id, self.tools = actor, root_id, tools

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        response = await self.tools.execute(
            ToolExecutionRequest(
                agent_run_id=self.root_id,
                tool_id="automation.preview",
                tool_version="1.0.0",
                call_id=str(context.invocation_id),
                actor=ActorReference(
                    organization_id=self.actor.organization_id,
                    membership_id=self.actor.membership_id,
                ),
                typed_input=arguments.model_dump(mode="json"),
                idempotency_key=f"candidate-preview:{context.invocation_id}",
            )
        )
        if response.status != "SUCCEEDED":
            raise ToolPolicyError("SCHEDULE_PREVIEW_UNAVAILABLE")
        return BackendToolResult(
            output=DailySummaryResponseBlock.model_validate(response.typed_output)
        )

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        return await self.execute(context, arguments)


@dataclass(frozen=True, slots=True)
class CandidateSpecialists:
    sessions: async_sessionmaker[AsyncSession]
    registry: AgentRegistry
    tools: ToolRegistry
    actors: CurrentActorResolverPort
    gateway: ModelGateway
    work: ToolExecutorPort
    planning: ToolExecutorPort
    assignment: ToolExecutorPort
    assignment_contexts: AssistantAssignmentContextResolver
    daily_update: ToolExecutorPort
    risk: RiskToolAdapter
    recorded_risk: ToolExecutorPort
    reporting: ToolExecutorPort
    automation: ToolExecutorPort
    retrieval: RetrievalRuntime | None
    activation: ActivationContext | None
    timezone: str

    async def build(
        self,
        actor: AuthenticatedActor,
        scope: RunScope,
        job: AssistantJob,
        recorder: PostgreSQLExecutionRecorder,
        root_id: UUID,
        access: CandidateAccess,
        projector: ExplorationProjector,
        locale: Literal["vi", "en"],
    ) -> CandidateBackends:
        refs = CandidateReferences(access, projector)
        actors = CurrentAgentActorResolver(self.actors)
        gateway = BudgetedToolLoopGateway(self.gateway)
        source = PostgreSQLGoalSourceTurnReader(
            session_factory=self.sessions, actor=actor, scope=scope
        )
        harnesses: dict[AgentId, GoalSpecialistHarness] = {
            AgentId.WORK_INTELLIGENCE: WorkIntelligenceHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(_ReadTools(self.work, self.recorded_risk)),
                actor_resolver=actors,
                goal_access=refs,
            ),
            AgentId.PLANNING: PlanningAgentHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(self.planning),
                actor_resolver=actors,
                goal_access=refs,
            ),
            AgentId.ASSIGNMENT: AssignmentAgentHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(self.assignment),
                actor_resolver=actors,
                goal_access=refs,
                explicit_command=AssistantAssignmentExplicitCommandAdapter(
                    source_turn=source,
                    actor_resolver=self.actors,
                    resolver=self.assignment_contexts,
                ),
            ),
            AgentId.DAILY_UPDATE: DailyUpdateHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(self.daily_update),
                actor_resolver=actors,
                goal_access=refs,
                source_turns=source,
            ),
            AgentId.RISK: RiskHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(self.recorded_risk),
                actor_resolver=actors,
                goal_access=refs,
            ),
            AgentId.REPORTING: ReportingHarness(
                model_gateway=gateway,
                tool_executor=BudgetedSpecialistTools(self.reporting),
                actor_resolver=actors,
                goal_access=refs,
            ),
        }
        if self.retrieval is not None:
            harnesses[AgentId.RETRIEVE] = _Retrieve(
                gateway, self.retrieval, actors, source, self.activation
            )
        pins = self.registry.specialist_pins(goal_directed=True)
        runners = {
            agent_id: PostgreSQLGoalSpecialistRunner(
                recorder=recorder,
                harness=harnesses[agent_id],
                pin=pin,
                risk_tools=self.risk if agent_id is AgentId.RISK else None,
                assignment_receipts=PostgreSQLAssignmentReceiptReader(
                    session_factory=self.sessions,
                    actor=actor,
                )
                if agent_id is AgentId.ASSIGNMENT
                else None,
            )
            for pin in pins
            if (agent_id := AgentId(pin.reference)) in harnesses
        }
        delegate = DelegationAdapterV2(
            scope=scope,
            parent_agent_run_id=root_id,
            actor_role=actor.role.value,
            locale=locale,
            timezone=self.timezone,
            active_phase=5,
            agent_registry=self.registry,
            specialist_pins=pins,
            runners=runners,
            tool_registry=self.tools,
            activation=self.activation,
            references=refs,
            workflow_verifier=PostgreSQLWorkflowWaitVerifier(
                session_factory=self.sessions, actor=actor
            ),
            child_job_receipts=PostgreSQLChildJobReceipts(sessions=self.sessions, actor=actor),
            daily_update_goal_receipts=PostgreSQLDailyUpdateReceiptReader(
                session_factory=self.sessions,
                actor=actor,
            ),
        )
        preview = _AutomationPreview(actor, root_id, self.automation)
        return CandidateBackends(
            {
                "delegate_agent@2": BackendDelegationTool(actor=actor, adapter=delegate),
                "automation.preview@1": ApplicationToolBackend(
                    actor=actor,
                    execute=preview.execute,
                    reconcile=preview.reconcile,
                ),
            },
            specialists=lambda request: (
                delegate.available_specialists() if request == scope else ()
            ),
            observe=refs.observe,
        )
