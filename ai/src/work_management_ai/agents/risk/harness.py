"""One model attempt between authorized read and exact-context revalidation."""

import asyncio
import json
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from work_management_ai.agents.orchestrator.contracts import ActorContextResolverPort
from work_management_ai.agents.risk.contracts import (
    ObservationExplanation,
    RiskExplanation,
    RiskExplanationInput,
    RiskQuestion,
    RiskReplanRequest,
)
from work_management_ai.agents.risk.evaluators.grounding import verify_explanation
from work_management_ai.agents.risk.prompts.system_v1 import SYSTEM_V1
from work_management_ai.agents.risk.workflows.graph import RiskGraph, RiskState
from work_management_ai.model_gateway.contracts import (
    ModelGateway,
    ModelMessage,
    StructuredModelRequest,
)
from work_management_ai.runtime.contracts import (
    AgentHandoff,
    AgentHandoffV2,
    AgentId,
    AgentResult,
    AgentRunStatus,
    ContextReference,
    GoalActionAccessPort,
    GoalDirectedAgentResult,
    InteractionOutcome,
    JsonValue,
    RequestedHandoff,
    RiskLevel,
    ToolExecutionRequest,
    ToolExecutorPort,
    VerifierResult,
)
from work_management_ai.runtime.manifests import (
    AgentManifest,
    GoalDirectedAgentManifest,
    load_yaml_resource,
)
from work_management_ai.runtime.policy_guard import PolicyGuard
from work_management_ai.runtime.skill_bundle import (
    load_packaged_skill_manifest,
    read_skill_instructions,
)


class _RiskTaskRequired(ValueError):
    pass


class RiskHarness:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        tool_executor: ToolExecutorPort,
        actor_resolver: ActorContextResolverPort,
        goal_access: GoalActionAccessPort | None = None,
    ):
        self.gateway = model_gateway
        self.tools = tool_executor
        self.actors = actor_resolver
        self.goal_access = goal_access
        self.manifest = load_yaml_resource(
            "work_management_ai.agents.risk", "agent.yaml", AgentManifest
        )
        self.goal_manifest = load_yaml_resource(
            "work_management_ai.agents.risk", "agent.v2.yaml", GoalDirectedAgentManifest
        )

    async def run(self, handoff: AgentHandoff | AgentHandoffV2) -> AgentResult:
        result = await self._run(handoff)
        if isinstance(handoff, AgentHandoffV2):
            return GoalDirectedAgentResult.model_validate(
                {
                    **result.model_dump(),
                    "agent_version": self.goal_manifest.agent.version,
                    "resolved_action": "risk.explain",
                }
            )
        return result

    async def _run(self, handoff: AgentHandoff | AgentHandoffV2) -> AgentResult:
        runner = _RiskRun(self, handoff)
        state = RiskState()
        try:
            async with asyncio.timeout(min(90, handoff.budget.timeout_seconds)):
                await RiskGraph().run(runner, state, handoff.budget.max_iterations)
            context, explanation = state.context, state.explanation
            if context is None or explanation is None:
                raise ValueError("RISK_CONTEXT_REQUIRED")
            replan = RiskReplanRequest(
                fingerprint=context.fingerprint,
                affected_week_ids=context.affected_week_ids,
                observation_ids=tuple(o.id for o in context.observations),
                risk_assessment_id=context.risk_assessment_id,
                task_id=context.task_id,
            )
            requested = (
                RequestedHandoff(
                    target_capability="planning.revise",
                    objective=runner.value.question,
                    typed_input=cast(dict[str, JsonValue], replan.model_dump(mode="json")),
                )
                if explanation.replan_requested and context.scope == "MANAGER"
                else None
            )
            if context.scope != "MANAGER" and explanation.replan_requested:
                explanation = explanation.model_copy(update={"replan_requested": False})
            output = {
                **context.model_dump(mode="json"),
                "explanation": explanation.model_dump(mode="json"),
                "fallback": state.fallback,
            }
            return AgentResult(
                agent_id=AgentId.RISK,
                agent_version="1.0.0",
                status=AgentRunStatus.COMPLETED,
                typed_output=cast(dict[str, JsonValue], output),
                requested_handoff=requested,
                iterations_used=state.iterations,
                tool_calls_used=state.tools,
                model_attempts_used=state.attempts,
                stop_reason="STORED_RATIONALE_FALLBACK" if state.fallback else "COMPLETED",
                verifier_results=(
                    VerifierResult(
                        verifier_id="risk_grounding", verifier_version="1.0.0", passed=True
                    ),
                ),
            )
        except _RiskTaskRequired:
            return AgentResult(
                agent_id=AgentId.RISK,
                agent_version=self.goal_manifest.agent.version,
                status=AgentRunStatus.AWAITING_INPUT,
                interaction_outcome=InteractionOutcome.NEEDS_INPUT,
                typed_output={
                    "question": (
                        "Bạn muốn xem rủi ro của công việc nào?"
                        if isinstance(handoff, AgentHandoffV2) and handoff.locale == "vi"
                        else "Which Task should I assess?"
                    )
                },
                stop_reason="RISK_TASK_REQUIRED",
            )
        except Exception:
            return AgentResult(
                agent_id=AgentId.RISK,
                agent_version="1.0.0",
                status=AgentRunStatus.FAILED,
                typed_output={"fallback": "manual_risk"},
                iterations_used=state.iterations,
                tool_calls_used=state.tools,
                model_attempts_used=state.attempts,
                stop_reason="RISK_CONTEXT_UNAVAILABLE",
                safe_error_code="RISK_CONTEXT_UNAVAILABLE",
            )


class _RiskRun:
    def __init__(self, harness: RiskHarness, handoff: AgentHandoff | AgentHandoffV2):
        self.harness, self.handoff = harness, handoff
        self.value = RiskQuestion(task_reference="unresolved", locale="en", question="unresolved")
        self.instructions = ""
        self.task: ContextReference | None = None

    async def authorize_goal(self, handoff: AgentHandoffV2) -> None:
        actor = await self.harness.actors.resolve(handoff.actor)
        PolicyGuard().authorize_goal_action(
            current_actor=actor,
            handoff=handoff,
            manifest=self.harness.goal_manifest,
            resolved_action="risk.explain",
            action_risk=RiskLevel.READ_ONLY,
            requested_tool_ids=("risk.read@1",),
        )
        if any(ref.organization_id != actor.organization_id for ref in handoff.context_references):
            raise ValueError("RISK_CONTEXT_TENANT")
        if handoff.context_references:
            if self.harness.goal_access is None:
                raise ValueError("RISK_CONTEXT_UNAVAILABLE")
            await self.harness.goal_access.verify(handoff)

    async def read(self, state: RiskState, expected: str | None = None) -> RiskExplanationInput:
        if state.tools >= self.handoff.budget.max_tool_calls:
            raise ValueError("RISK_TOOL_BUDGET")
        if isinstance(self.handoff, AgentHandoffV2):
            await self.authorize_goal(self.handoff)
        state.tools += 1
        result = await self.harness.tools.execute(
            ToolExecutionRequest(
                agent_run_id=uuid5(NAMESPACE_URL, f"agent-run:{self.handoff.idempotency_key}"),
                tool_id="risk.read",
                tool_version="1.0.0",
                call_id=f"{self.handoff.step_id}:read:{state.tools}",
                actor=self.handoff.actor,
                typed_input={
                    "task_reference": self.value.task_reference,
                    "expected_fingerprint": expected,
                },
                idempotency_key=f"{self.handoff.idempotency_key}:read:{state.tools}",
            )
        )
        if result.status != "SUCCEEDED":
            raise ValueError("RISK_READ_DENIED")
        facts = RiskExplanationInput.model_validate(result.typed_output)
        if self.task is not None and (
            facts.task_id != self.task.resource_id or facts.task_version != self.task.version
        ):
            raise ValueError("RISK_CONTEXT_CHANGED")
        return facts

    async def execute_node(self, node: str, state: RiskState) -> None:
        if node == "authorize":
            if isinstance(self.handoff, AgentHandoffV2):
                await self.authorize_goal(self.handoff)
                tasks = [
                    ref
                    for ref in self.handoff.context_references
                    if ref.resource_type.casefold() == "task"
                ]
                if len(tasks) != 1 or tasks[0].version is None:
                    raise _RiskTaskRequired
                self.task = tasks[0]
                self.value = RiskQuestion(
                    task_reference=str(self.task.resource_id),
                    locale=self.handoff.locale,
                    question=self.handoff.brief.objective,
                )
            else:
                actor = await self.harness.actors.resolve(self.handoff.actor)
                PolicyGuard().authorize_handoff(
                    current_actor=actor,
                    parent_agent_id=AgentId.ORCHESTRATOR,
                    handoff=self.handoff,
                    manifest=self.harness.manifest,
                    requested_tool_ids=("risk.read@1",),
                    requested_skill_ids=self.harness.manifest.allowed_skills,
                )
                if any(
                    r.organization_id != actor.organization_id
                    for r in self.handoff.context_references
                ):
                    raise ValueError("RISK_CONTEXT_TENANT")
                self.value = RiskQuestion.model_validate(self.handoff.typed_input)
            instructions: list[str] = []
            for ref in self.harness.manifest.allowed_skills:
                package = f"work_management_ai.skills.{ref.split('@')[0]}"
                skill = load_packaged_skill_manifest(package)
                instructions.append(read_skill_instructions(package, skill))
            self.instructions = "\n".join(instructions)
        elif node == "read":
            state.context = await self.read(state)
        elif node == "explain":
            context = state.context
            if context is None:
                raise ValueError("RISK_CONTEXT_REQUIRED")
            try:
                if self.handoff.budget.max_model_attempts < 1:
                    raise ValueError("RISK_MODEL_BUDGET")
                content = json.dumps(
                    {
                        "question": self.value.question,
                        "locale": self.value.locale,
                        "context": context.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                system = SYSTEM_V1 + "\n\n## Skill instructions\n" + self.instructions
                # UTF-8 bytes bound conservatively bounds token count for text.
                input_bound = (
                    len(content.encode())
                    + len(system.encode())
                    + len(json.dumps(RiskExplanation.model_json_schema()).encode())
                    + 1024
                )
                if input_bound > self.handoff.budget.max_input_tokens:
                    raise ValueError("RISK_INPUT_BUDGET")
                state.attempts += 1
                response = await self.harness.gateway.generate_structured(
                    StructuredModelRequest(
                        invocation_key=f"risk.{self.value.locale}.explain",
                        messages=(
                            ModelMessage(role="system", content=system),
                            ModelMessage(role="user", content=content),
                        ),
                        output_schema=RiskExplanation,
                        timeout_seconds=min(60, self.handoff.budget.timeout_seconds),
                        max_output_tokens=self.handoff.budget.max_output_tokens,
                    )
                )
                verify_explanation(context, response.parsed)
                state.explanation = response.parsed
            except Exception:
                state.fallback = True
                state.explanation = RiskExplanation(
                    observation_explanations=tuple(
                        ObservationExplanation(
                            text=o.text,
                            observation_ids=(o.id,),
                            source_ids=o.source_ids,
                            assertions=(),
                        )
                        for o in context.observations[:10]
                        if not o.id.startswith("fact:")
                    ),
                    limitations=context.limitations[:10],
                    recommendations=context.recommendations[:5],
                    replan_requested=False,
                )
        elif node == "revalidate":
            if state.context is None:
                raise ValueError("RISK_CONTEXT_REQUIRED")
            current = await self.read(state, state.context.fingerprint)
            if current != state.context:
                raise ValueError("RISK_CONTEXT_CHANGED")
