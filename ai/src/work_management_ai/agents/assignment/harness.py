"""Guarded Assignment Specialist Harness around deterministic application tools."""

import json
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from work_management_ai.agents.assignment.contracts import (
    AssignmentAgentInput,
    AssignmentAgentOutput,
    AssignmentExplanation,
    AssignmentExplicitCommandPort,
    AssignmentGoalChoice,
    AssignmentOperation,
    ExplicitAssignmentSnapshot,
    ProjectWorkloadSnapshot,
    TeamRecommendationSnapshot,
    TeamRequirementsPendingSnapshot,
    VerifiedAssignmentCommand,
    WorkloadExplanation,
    explicit_goal_assignment_key,
)
from work_management_ai.agents.assignment.evaluators.explanation import (
    AssignmentExplanationError,
    verify_assignment_explanation,
    verify_explanation_context,
    verify_workload_explanation,
)
from work_management_ai.agents.assignment.prompts import build_explanation_messages
from work_management_ai.agents.assignment.workflows.graph import (
    AssignmentAgentGraph,
    AssignmentAgentState,
)
from work_management_ai.agents.orchestrator.contracts import ActorContextResolverPort
from work_management_ai.model_gateway.contracts import (
    ModelGateway,
    ModelMessage,
    ModelRequestLimits,
    StructuredModelRequest,
)
from work_management_ai.model_gateway.errors import ModelGatewayError
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
    ProposedAction,
    RiskLevel,
    ToolExecutionRequest,
    ToolExecutorPort,
    VerifierResult,
)
from work_management_ai.runtime.manifests import (
    AgentManifest,
    GoalDirectedAgentManifest,
    ToolManifest,
    load_yaml_resource,
)
from work_management_ai.runtime.policy_guard import PolicyGuard
from work_management_ai.runtime.skill_bundle import (
    load_packaged_skill_manifest,
    read_skill_instructions,
)
from work_management_ai.tools.assignment.assign_task.contracts import AssignTaskInput
from work_management_ai.tools.assignment.manage_team.contracts import ManageTeamInput
from work_management_ai.tools.assignment.read_workload.contracts import ReadWorkloadInput

_AGENT_PACKAGE = "work_management_ai.agents.assignment"
_SKILL_PACKAGES = {
    "recommend_project_team@1": "work_management_ai.skills.recommend_project_team",
    "analyze_workload@1": "work_management_ai.skills.analyze_workload",
}
_TOOL_PACKAGES = {
    "assignment.manage_team": "work_management_ai.tools.assignment.manage_team",
    "assignment.read_workload": "work_management_ai.tools.assignment.read_workload",
    "assignment.assign_task": "work_management_ai.tools.assignment.assign_task",
}
_CAPABILITY_BY_OPERATION = {
    AssignmentOperation.RECOMMEND_TEAM: "assignment.recommend_team",
    AssignmentOperation.REVISE_TEAM: "assignment.revise_team",
    AssignmentOperation.ANALYZE_WORKLOAD: "assignment.analyze_workload",
    AssignmentOperation.ASSIGN_TASK_EXPLICITLY: "assignment.assign_task_explicitly",
}
_EXPLANATION_UNAVAILABLE = "ASSIGNMENT_EXPLANATION_UNAVAILABLE"
_MANUAL_FALLBACK = "ASSIGNMENT_MANUAL_FALLBACK"


class AssignmentAgentHarness:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        tool_executor: ToolExecutorPort,
        actor_resolver: ActorContextResolverPort,
        goal_access: GoalActionAccessPort | None = None,
        explicit_command: AssignmentExplicitCommandPort | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._tool_executor = tool_executor
        self._actor_resolver = actor_resolver
        self._manifest = load_yaml_resource(_AGENT_PACKAGE, "agent.yaml", AgentManifest)
        self._goal_manifest = load_yaml_resource(
            _AGENT_PACKAGE, "agent.v2.yaml", GoalDirectedAgentManifest
        )
        self._goal_access = goal_access
        self._explicit_command = explicit_command
        self._graph = AssignmentAgentGraph(self)

    async def run(self, handoff: AgentHandoff | AgentHandoffV2) -> AgentResult:
        if isinstance(handoff, AgentHandoffV2):
            return await self._run_goal(handoff)
        return await self._run_graph(handoff)

    async def _run_graph(
        self,
        handoff: AgentHandoff,
        *,
        goal_intake_attempts: int | None = None,
        goal_assignment_key: str | None = None,
    ) -> AgentResult:
        return await self._graph.run(
            AssignmentAgentState(
                handoff=handoff,
                actor=None,
                value=None,
                selected_skill=None,
                skill_instructions="",
                tool_result=None,
                snapshot=None,
                explanation=None,
                output=None,
                result=None,
                route="execute",
                stop_reason="NOT_STARTED",
                safe_error_code=None,
                iterations_used=0,
                tool_calls_used=0,
                model_refs=(),
                goal_model_attempts=goal_intake_attempts,
                goal_assignment_key=goal_assignment_key,
            )
        )

    async def _run_goal(self, handoff: AgentHandoffV2) -> GoalDirectedAgentResult:
        attempts = 0
        try:
            actor = await self._actor_resolver.resolve(handoff.actor)
            guard = PolicyGuard()
            guard.authorize_goal_action(
                current_actor=actor,
                handoff=handoff,
                manifest=self._goal_manifest,
                resolved_action="assignment.analyze_workload",
                action_risk=RiskLevel.READ_ONLY,
            )
            if any(
                ref.organization_id != actor.organization_id for ref in handoff.context_references
            ):
                raise ValueError("ASSIGNMENT_CONTEXT_TENANT_MISMATCH")
            command = None
            ref_types = {ref.resource_type.upper() for ref in handoff.context_references}
            if self._explicit_command is not None and {"TASK", "MEMBERSHIP"} <= ref_types:
                command = await self._explicit_command.read(handoff)
            if command is not None:
                choice = AssignmentGoalChoice(
                    operation=AssignmentOperation.ASSIGN_TASK_EXPLICITLY,
                    revision_instruction=None,
                )
            elif (
                handoff.budget.max_model_attempts
                and handoff.budget.max_input_tokens
                and handoff.budget.max_output_tokens
            ):
                attempts = 1
                choice = await self._choose_goal_action(handoff)
            else:
                return self._goal_boundary_result(handoff, needs_input=True, attempts=attempts)
            value = self._intake_v2(handoff, choice, command)
            if value is None:
                return self._goal_boundary_result(handoff, needs_input=True, attempts=attempts)
            if self._goal_access is None:
                raise ValueError("ASSIGNMENT_CONTEXT_UNAVAILABLE")
            action = _CAPABILITY_BY_OPERATION[value.operation]
            tool_id, _ = self._trusted_tool_input(value)
            skill_ids: tuple[str, ...] = ()
            risk = RiskLevel.EXPLICIT_WRITE
            if value.operation is AssignmentOperation.ANALYZE_WORKLOAD:
                skill_ids = ("analyze_workload@1",)
                risk = RiskLevel.READ_ONLY
            elif value.operation is not AssignmentOperation.ASSIGN_TASK_EXPLICITLY:
                skill_ids = ("recommend_project_team@1",)
                risk = RiskLevel.PROPOSAL_ONLY
            guard.authorize_goal_action(
                current_actor=actor,
                handoff=handoff,
                manifest=self._goal_manifest,
                resolved_action=action,
                action_risk=risk,
                requested_tool_ids=(f"{tool_id}@1",),
                requested_skill_ids=skill_ids,
            )
            await self._goal_access.verify(handoff)
        except (ModelGatewayError, ValidationError):
            return self._goal_boundary_result(handoff, needs_input=True, attempts=attempts)
        except ValueError:
            return self._goal_boundary_result(handoff, needs_input=False, attempts=attempts)

        legacy = AgentHandoff(
            orchestration_run_id=handoff.orchestration_run_id,
            parent_agent_run_id=handoff.parent_agent_run_id,
            target_agent_id=handoff.target_agent_id,
            target_agent_version=self._manifest.agent.version,
            capability=action,
            objective=handoff.brief.objective,
            typed_input=value.model_dump(mode="json"),
            context_references=handoff.context_references,
            actor=handoff.actor,
            budget=handoff.budget,
            step_id=handoff.step_id,
            idempotency_key=handoff.idempotency_key,
        )
        assignment_key = (
            explicit_goal_assignment_key(handoff.source_turn_ref) if command is not None else None
        )
        result = await self._run_graph(
            legacy, goal_intake_attempts=attempts, goal_assignment_key=assignment_key
        )
        status = result.status
        outcome = InteractionOutcome.FAILED
        if status is AgentRunStatus.COMPLETED:
            outcome = InteractionOutcome.RESULT_READY
        elif status is AgentRunStatus.AWAITING_HUMAN:
            if result.proposed_actions:
                outcome = InteractionOutcome.ACTION_REVIEW_REQUIRED
            else:
                status = AgentRunStatus.COMPLETED
                outcome = InteractionOutcome.DRAFT_READY
        return GoalDirectedAgentResult.model_validate(
            {
                **result.model_dump(),
                "agent_version": self._goal_manifest.agent.version,
                "status": status,
                "interaction_outcome": outcome,
                "resolved_action": action,
                "iterations_used": attempts + result.iterations_used,
            }
        )

    async def _choose_goal_action(self, handoff: AgentHandoffV2) -> AssignmentGoalChoice:
        result = await self._model_gateway.generate_structured(
            StructuredModelRequest(
                invocation_key=f"assignment_agent.{handoff.locale}.goal_intake",
                output_schema=AssignmentGoalChoice,
                timeout_seconds=min(handoff.budget.timeout_seconds, 30),
                max_output_tokens=min(handoff.budget.max_output_tokens, 400),
                limits=ModelRequestLimits(
                    max_text_bytes=handoff.budget.max_input_tokens,
                    max_output_tokens=handoff.budget.max_output_tokens,
                ),
                messages=(
                    ModelMessage(
                        role="system",
                        content=(
                            "Choose the Assignment operation that fits the objective. "
                            "Use ANALYZE_WORKLOAD to inspect workload, RECOMMEND_TEAM for a new "
                            "team proposal, REVISE_TEAM to change an existing proposal, and "
                            "ASSIGN_TASK_EXPLICITLY for a request to assign an exact task to a "
                            "person. Return null if unclear. For a team revision, reproduce the "
                            "requested replacement and reason in revision_instruction."
                        ),
                    ),
                    ModelMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "brief": handoff.brief.model_dump(mode="json"),
                                "resource_types": [
                                    ref.resource_type for ref in handoff.context_references
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ),
            )
        )
        return result.parsed

    @staticmethod
    def _intake_v2(
        handoff: AgentHandoffV2,
        choice: AssignmentGoalChoice,
        command: VerifiedAssignmentCommand | None,
    ) -> AssignmentAgentInput | None:
        def only(resource_type: str) -> ContextReference | None:
            refs = tuple(
                ref
                for ref in handoff.context_references
                if ref.resource_type.upper() == resource_type
            )
            return refs[0] if len(refs) == 1 else None

        operation = choice.operation
        if operation is None:
            return None
        project = only("PROJECT")
        if operation is AssignmentOperation.ASSIGN_TASK_EXPLICITLY:
            task, member = only("TASK"), only("MEMBERSHIP")
            if project is None and any(
                ref.resource_type.upper() == "PROJECT" for ref in handoff.context_references
            ):
                return None
            if (
                command is None
                or task is None
                or member is None
                or task.resource_id != command.task_id
                or task.version != command.expected_task_version
                or member.resource_id != command.assignee_membership_id
                or (project is not None and project.resource_id != command.project_id)
            ):
                return None
            return AssignmentAgentInput(
                operation=operation,
                locale=handoff.locale,
                project_id=command.project_id,
                task_id=command.task_id,
                task_version=command.expected_task_version,
                membership_id=command.assignee_membership_id,
            )
        if project is None:
            return None
        if operation is AssignmentOperation.REVISE_TEAM:
            recommendation = only("TEAM_RECOMMENDATION")
            if (
                recommendation is None
                or recommendation.version is None
                or not choice.revision_instruction
            ):
                return None
            return AssignmentAgentInput(
                operation=operation,
                locale=handoff.locale,
                project_id=project.resource_id,
                recommendation_id=recommendation.resource_id,
                recommendation_version=recommendation.version,
                revision_instruction=choice.revision_instruction,
            )
        return AssignmentAgentInput(
            operation=operation, locale=handoff.locale, project_id=project.resource_id
        )

    def _goal_boundary_result(
        self, handoff: AgentHandoffV2, *, needs_input: bool, attempts: int
    ) -> GoalDirectedAgentResult:
        question = (
            "Bạn muốn phân tích tải, đề xuất đội hình hay giao công việc? "
            "Hãy xác nhận dự án hoặc tên công việc và người được giao."
            if handoff.locale == "vi"
            else "Would you like workload analysis, a team proposal, or a task assignment? "
            "Please confirm the project or the exact task and assignee."
        )
        return GoalDirectedAgentResult(
            agent_id=AgentId.ASSIGNMENT,
            agent_version=self._goal_manifest.agent.version,
            status=AgentRunStatus.AWAITING_INPUT if needs_input else AgentRunStatus.FAILED,
            interaction_outcome=InteractionOutcome.NEEDS_INPUT
            if needs_input
            else InteractionOutcome.FAILED,
            typed_output={"question": question}
            if needs_input
            else {"fallback": "manual_assignment"},
            stop_reason="ASSIGNMENT_GOAL_NEEDS_INPUT"
            if needs_input
            else "ASSIGNMENT_GOAL_UNAVAILABLE",
            model_attempts_used=attempts,
            iterations_used=attempts,
        )

    async def receive_handoff(self, state: AssignmentAgentState) -> dict[str, object]:
        return {"stop_reason": "RUNNING"}

    async def validate_contract_and_policy(self, state: AssignmentAgentState) -> dict[str, object]:
        handoff = state["handoff"]
        runtime = self._manifest.runtime
        if (
            handoff.target_agent_id is not AgentId.ASSIGNMENT
            or handoff.target_agent_version != self._manifest.agent.version
            or handoff.capability not in self._manifest.capabilities
        ):
            return self._failure("ASSIGNMENT_HANDOFF_INVALID")
        if (
            handoff.budget.max_iterations > runtime.max_iterations
            or handoff.budget.max_tool_calls > runtime.max_tool_calls
            or handoff.budget.max_handoffs > runtime.max_handoffs
            or handoff.budget.max_replans > runtime.max_replans
            or handoff.budget.timeout_seconds > runtime.timeout_seconds
        ):
            return self._failure("ASSIGNMENT_BUDGET_INVALID")
        if any(
            reference.organization_id != handoff.actor.organization_id
            for reference in handoff.context_references
        ):
            return self._failure("ASSIGNMENT_CONTEXT_TENANT_MISMATCH")
        try:
            actor = await self._actor_resolver.resolve(handoff.actor)
        except Exception:
            return self._failure("ASSIGNMENT_ACTOR_NOT_FOUND")
        if (
            not actor.is_active
            or actor.membership_id != handoff.actor.membership_id
            or actor.organization_id != handoff.actor.organization_id
        ):
            return self._failure("ASSIGNMENT_ACTOR_NOT_FOUND")
        if actor.role not in self._manifest.permissions.roles:
            return self._failure("ASSIGNMENT_ROLE_FORBIDDEN", preserve_code=True)
        try:
            value = AssignmentAgentInput.model_validate(handoff.typed_input)
        except ValidationError:
            return self._failure("ASSIGNMENT_INPUT_INVALID", preserve_code=True)
        if handoff.capability != _CAPABILITY_BY_OPERATION[value.operation]:
            return self._failure("ASSIGNMENT_CAPABILITY_MISMATCH")
        return {"actor": actor, "value": value, "route": "execute"}

    async def load_skill(self, state: AssignmentAgentState) -> dict[str, object]:
        value = state["value"]
        if value is None:
            return self._failure("ASSIGNMENT_INPUT_INVALID")
        if value.operation is AssignmentOperation.ASSIGN_TASK_EXPLICITLY:
            return {"route": "execute"}
        reference = (
            "analyze_workload@1"
            if value.operation is AssignmentOperation.ANALYZE_WORKLOAD
            else "recommend_project_team@1"
        )
        package = _SKILL_PACKAGES[reference]
        try:
            skill = load_packaged_skill_manifest(package)
            instructions = read_skill_instructions(package, skill)
        except (OSError, ValueError, ValidationError):
            return self._failure("ASSIGNMENT_SKILL_INVALID")
        if reference not in self._manifest.allowed_skills:
            return self._failure("ASSIGNMENT_SKILL_NOT_ALLOWED")
        return {
            "selected_skill": reference,
            "skill_instructions": instructions,
            "route": "execute",
        }

    async def execute_deterministic_tool(self, state: AssignmentAgentState) -> dict[str, object]:
        value = state["value"]
        if value is None:
            return self._failure("ASSIGNMENT_INPUT_INVALID")
        if state["tool_calls_used"] >= state["handoff"].budget.max_tool_calls:
            return self._failure("ASSIGNMENT_TOOL_BUDGET_EXHAUSTED", preserve_code=True)
        tool_id, typed_input = self._trusted_tool_input(value)
        reference = f"{tool_id}@1"
        if reference not in self._manifest.allowed_tools:
            return self._failure("ASSIGNMENT_TOOL_NOT_ALLOWED")
        try:
            manifest = load_yaml_resource(_TOOL_PACKAGES[tool_id], "tool.yaml", ToolManifest)
        except (ValueError, ValidationError):
            return self._failure("ASSIGNMENT_TOOL_NOT_ALLOWED")
        if manifest.name != tool_id:
            return self._failure("ASSIGNMENT_TOOL_NOT_ALLOWED")
        request = ToolExecutionRequest(
            agent_run_id=uuid5(NAMESPACE_URL, f"agent-run:{state['handoff'].idempotency_key}"),
            tool_id=tool_id,
            tool_version=manifest.version,
            call_id=f"{state['handoff'].step_id}:{tool_id}:1",
            actor=state["handoff"].actor,
            typed_input=typed_input,
            idempotency_key=state["goal_assignment_key"]
            or f"{state['handoff'].idempotency_key}:{tool_id}",
        )
        consumed = state["tool_calls_used"] + 1
        goal_run = state["goal_model_attempts"] is not None
        counted = {"tool_calls_used": consumed} if goal_run else {}
        try:
            result = await self._tool_executor.execute(request)
        except Exception:
            return {**self._failure("ASSIGNMENT_TOOL_FAILED"), **counted}
        if result.status != "SUCCEEDED":
            return {
                **self._failure(
                    result.safe_error_code or "ASSIGNMENT_TOOL_FAILED", preserve_code=True
                ),
                **counted,
            }
        try:
            snapshot = self._parse_snapshot(value.operation, result.typed_output)
        except ValidationError:
            return {**self._failure("ASSIGNMENT_TOOL_OUTPUT_INVALID"), **counted}
        if goal_run and (
            snapshot.organization_id != state["handoff"].actor.organization_id
            or snapshot.project_id != value.project_id
            or not self._snapshot_matches_action(value, snapshot)
            or any(ref.organization_id != snapshot.organization_id for ref in result.evidence)
        ):
            return {**self._failure("ASSIGNMENT_TOOL_OUTPUT_INVALID"), **counted}
        return {
            "tool_result": result,
            "snapshot": snapshot,
            "tool_calls_used": consumed,
            "route": (
                "verify"
                if value.operation is AssignmentOperation.ASSIGN_TASK_EXPLICITLY
                else "execute"
            ),
        }

    async def generate_structured_explanation(
        self, state: AssignmentAgentState
    ) -> dict[str, object]:
        value, snapshot = state["value"], state["snapshot"]
        if value is None or snapshot is None:
            return self._failure("ASSIGNMENT_STATE_INVALID")
        if state["iterations_used"] >= state["handoff"].budget.max_iterations:
            return self._deterministic_failure("ASSIGNMENT_ITERATION_BUDGET_EXHAUSTED")
        output_schema: type[AssignmentExplanation] | type[WorkloadExplanation]
        if isinstance(snapshot, TeamRecommendationSnapshot):
            output_schema = AssignmentExplanation
            try:
                verify_explanation_context(snapshot)
            except AssignmentExplanationError:
                return self._deterministic_failure(_EXPLANATION_UNAVAILABLE)
        elif isinstance(snapshot, ProjectWorkloadSnapshot):
            output_schema = WorkloadExplanation
        else:
            return {"route": "verify"}
        goal_attempts = state["goal_model_attempts"]
        budget = state["handoff"].budget
        if goal_attempts is not None and (
            goal_attempts >= budget.max_model_attempts
            or goal_attempts + state["iterations_used"] >= budget.max_iterations
            or not budget.max_input_tokens
            or budget.max_output_tokens <= 400
        ):
            return self._deterministic_failure("ASSIGNMENT_MODEL_BUDGET_EXHAUSTED")
        request = StructuredModelRequest(
            invocation_key=(
                f"assignment_agent.{value.locale}.{value.operation.value.lower()}_explanation"
            ),
            messages=build_explanation_messages(
                snapshot,
                locale=value.locale,
                skill_instructions=state["skill_instructions"],
            ),
            output_schema=output_schema,
            timeout_seconds=budget.timeout_seconds,
            limits=(
                None
                if goal_attempts is None
                else ModelRequestLimits(
                    max_text_bytes=budget.max_input_tokens,
                    max_output_tokens=budget.max_output_tokens - 400,
                )
            ),
        )
        try:
            response = await self._model_gateway.generate_structured(request)
        except ModelGatewayError:
            return {
                **self._deterministic_failure(_EXPLANATION_UNAVAILABLE),
                "goal_model_attempts": None if goal_attempts is None else goal_attempts + 1,
                "iterations_used": state["iterations_used"] + (goal_attempts is not None),
            }
        return {
            "explanation": response.parsed,
            "goal_model_attempts": None if goal_attempts is None else goal_attempts + 1,
            "iterations_used": state["iterations_used"] + 1,
            "model_refs": (*state["model_refs"], response.model_ref),
            "route": "verify",
        }

    async def verify_result(self, state: AssignmentAgentState) -> dict[str, object]:
        value, snapshot, explanation = state["value"], state["snapshot"], state["explanation"]
        if value is None or snapshot is None:
            return self._failure("ASSIGNMENT_RESULT_INVALID")
        try:
            if isinstance(snapshot, TeamRecommendationSnapshot):
                if not isinstance(explanation, AssignmentExplanation):
                    raise AssignmentExplanationError("EXPLANATION_TYPE_INVALID")
                verify_assignment_explanation(snapshot, explanation)
            elif isinstance(snapshot, ProjectWorkloadSnapshot):
                if not isinstance(explanation, WorkloadExplanation):
                    raise AssignmentExplanationError("EXPLANATION_TYPE_INVALID")
                verify_workload_explanation(snapshot, explanation)
            elif explanation is not None:
                raise AssignmentExplanationError("EXPLANATION_NOT_ALLOWED")
        except AssignmentExplanationError:
            return self._deterministic_failure(_EXPLANATION_UNAVAILABLE)
        tool_result = state["tool_result"]
        if tool_result is None:
            return self._failure("ASSIGNMENT_RESULT_INVALID")
        output = AssignmentAgentOutput(
            operation=value.operation,
            deterministic_result=tool_result.typed_output,
            explanation_status="AVAILABLE" if explanation is not None else "NOT_REQUESTED",
            explanation=explanation,
        )
        return {"output": output, "route": "execute", "stop_reason": "COMPLETED"}

    async def deterministic_fallback(self, state: AssignmentAgentState) -> dict[str, object]:
        value, tool_result = state["value"], state["tool_result"]
        if value is None or tool_result is None:
            return self._failure("ASSIGNMENT_RESULT_INVALID")
        return {
            "output": AssignmentAgentOutput(
                operation=value.operation,
                deterministic_result=tool_result.typed_output,
                explanation_status="UNAVAILABLE",
                explanation=None,
            ),
            "route": "deterministic_fallback",
            "safe_error_code": _EXPLANATION_UNAVAILABLE,
        }

    async def manual_fallback(self, state: AssignmentAgentState) -> dict[str, object]:
        return {
            "route": "manual_fallback",
            "safe_error_code": state["safe_error_code"] or _MANUAL_FALLBACK,
        }

    async def return_agent_result(self, state: AssignmentAgentState) -> dict[str, object]:
        value, output = state["value"], state["output"]
        if state["route"] == "manual_fallback" or output is None or value is None:
            status = AgentRunStatus.FAILED
        elif value.operation in {
            AssignmentOperation.RECOMMEND_TEAM,
            AssignmentOperation.REVISE_TEAM,
        }:
            status = AgentRunStatus.AWAITING_HUMAN
        else:
            status = AgentRunStatus.COMPLETED
        proposed_actions: tuple[ProposedAction, ...] = ()
        if status is AgentRunStatus.AWAITING_HUMAN and isinstance(
            state["snapshot"], TeamRecommendationSnapshot
        ):
            proposed_actions = (
                ProposedAction(
                    action_type="assignment.team_proposal_review",
                    risk=RiskLevel.PROPOSAL_ONLY,
                    requires_human_gate=True,
                    reference_id=state["snapshot"].recommendation_id,
                ),
            )
        typed_output = (
            output.model_dump(mode="json")
            if output is not None
            else cast(dict[str, JsonValue], {"fallback": "manual_assignment"})
        )
        passed = status in {AgentRunStatus.COMPLETED, AgentRunStatus.AWAITING_HUMAN}
        explanation_unavailable = output is not None and output.explanation_status == "UNAVAILABLE"
        verifier_results = tuple(
            VerifierResult(
                verifier_id=reference.split("@", maxsplit=1)[0],
                verifier_version="1.0.0",
                passed=passed
                and not (
                    reference.startswith("assignment_explanation@") and explanation_unavailable
                ),
                safe_codes=(
                    (state["safe_error_code"],)
                    if state["safe_error_code"] is not None
                    and (not passed or reference.startswith("assignment_explanation@"))
                    else ()
                ),
            )
            for reference in self._manifest.evaluators
        )
        return {
            "result": AgentResult(
                agent_id=AgentId.ASSIGNMENT,
                agent_version=self._manifest.agent.version,
                status=status,
                typed_output=typed_output,
                evidence=state["tool_result"].evidence if state["tool_result"] else (),
                proposed_actions=proposed_actions,
                verifier_results=verifier_results,
                requested_handoff=None,
                model_attempts_used=state["goal_model_attempts"] or 0,
                iterations_used=state["iterations_used"],
                tool_calls_used=state["tool_calls_used"],
                stop_reason=state["stop_reason"],
                safe_error_code=state["safe_error_code"],
            )
        }

    @staticmethod
    def _trusted_tool_input(value: AssignmentAgentInput) -> tuple[str, dict[str, JsonValue]]:
        if value.operation is AssignmentOperation.RECOMMEND_TEAM:
            typed = ManageTeamInput(
                action="CREATE",
                project_id=value.project_id,
                planning_proposal_id=value.planning_proposal_id,
                planning_proposal_version=value.planning_proposal_version,
            )
            return "assignment.manage_team", cast(
                dict[str, JsonValue], typed.model_dump(mode="json", exclude_none=True)
            )
        if value.operation is AssignmentOperation.REVISE_TEAM:
            typed = ManageTeamInput(
                action="REVISE",
                project_id=value.project_id,
                recommendation_id=value.recommendation_id,
                recommendation_version=value.recommendation_version,
                revision_instruction=value.revision_instruction,
            )
            return "assignment.manage_team", cast(
                dict[str, JsonValue], typed.model_dump(mode="json", exclude_none=True)
            )
        if value.operation is AssignmentOperation.ANALYZE_WORKLOAD:
            typed = ReadWorkloadInput(project_id=cast(UUID, value.project_id))
            return "assignment.read_workload", cast(
                dict[str, JsonValue], typed.model_dump(mode="json")
            )
        typed = AssignTaskInput(
            task_id=cast(UUID, value.task_id),
            task_version=cast(int, value.task_version),
            membership_id=cast(UUID, value.membership_id),
        )
        return "assignment.assign_task", cast(dict[str, JsonValue], typed.model_dump(mode="json"))

    @staticmethod
    def _parse_snapshot(
        operation: AssignmentOperation, value: dict[str, JsonValue]
    ) -> (
        TeamRecommendationSnapshot
        | TeamRequirementsPendingSnapshot
        | ProjectWorkloadSnapshot
        | ExplicitAssignmentSnapshot
    ):
        if operation in {AssignmentOperation.RECOMMEND_TEAM, AssignmentOperation.REVISE_TEAM}:
            if value.get("kind") == "team_requirements_pending":
                return TeamRequirementsPendingSnapshot.model_validate(value)
            return TeamRecommendationSnapshot.model_validate(value)
        if operation is AssignmentOperation.ANALYZE_WORKLOAD:
            return ProjectWorkloadSnapshot.model_validate(value)
        return ExplicitAssignmentSnapshot.model_validate(value)

    @staticmethod
    def _snapshot_matches_action(
        value: AssignmentAgentInput,
        snapshot: TeamRecommendationSnapshot
        | TeamRequirementsPendingSnapshot
        | ProjectWorkloadSnapshot
        | ExplicitAssignmentSnapshot,
    ) -> bool:
        if isinstance(snapshot, ExplicitAssignmentSnapshot):
            return (
                snapshot.task_id == value.task_id
                and snapshot.membership_id == value.membership_id
                and value.task_version is not None
                and snapshot.task_version == value.task_version + 1
            )
        if isinstance(snapshot, TeamRecommendationSnapshot):
            if snapshot.status != "PROPOSED":
                return False
            if value.operation is AssignmentOperation.REVISE_TEAM:
                return (
                    snapshot.recommendation_id == value.recommendation_id
                    and value.recommendation_version is not None
                    and snapshot.version == value.recommendation_version + 1
                )
            return snapshot.version == 1
        return not (
            isinstance(snapshot, TeamRequirementsPendingSnapshot)
            and value.operation is AssignmentOperation.REVISE_TEAM
        )

    @staticmethod
    def _failure(code: str, *, preserve_code: bool = False) -> dict[str, object]:
        return {
            "route": "manual_fallback",
            "stop_reason": code,
            "safe_error_code": code if preserve_code else _MANUAL_FALLBACK,
        }

    @staticmethod
    def _deterministic_failure(code: str) -> dict[str, object]:
        return {
            "route": "deterministic_fallback",
            "stop_reason": code,
            "safe_error_code": _EXPLANATION_UNAVAILABLE,
        }
