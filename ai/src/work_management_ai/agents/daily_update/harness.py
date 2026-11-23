"""Guarded specialist: typed report interpretation ends at owner review."""

import asyncio
import json
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from work_management_ai.agents.daily_update.contracts import (
    DailyUpdateHandoff,
    DailyUpdateResult,
    EvidenceRef,
    ExtractedReport,
    ReportingSnapshot,
)
from work_management_ai.agents.daily_update.evaluators.grounding import verify_context
from work_management_ai.agents.daily_update.prompts.system_v1 import SYSTEM_V1
from work_management_ai.agents.daily_update.workflows.graph import (
    DailyUpdateGraph,
    DailyUpdateState,
)
from work_management_ai.agents.orchestrator.contracts import ActorContextResolverPort
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
    GoalActionAccessPort,
    GoalDirectedAgentResult,
    InteractionOutcome,
    JsonValue,
    RiskLevel,
    SourceTurnTextPort,
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
from work_management_ai.tools.daily_update.contracts import DailyUpdateToolInput


class DailyUpdateHarness:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        tool_executor: ToolExecutorPort,
        actor_resolver: ActorContextResolverPort,
        source_turns: SourceTurnTextPort | None = None,
        goal_access: GoalActionAccessPort | None = None,
    ) -> None:
        self.gateway = model_gateway
        self.tools = tool_executor
        self.actors = actor_resolver
        self.source_turns = source_turns
        self.goal_access = goal_access
        self.manifest = load_yaml_resource(
            "work_management_ai.agents.daily_update", "agent.yaml", AgentManifest
        )
        self.goal_manifest = load_yaml_resource(
            "work_management_ai.agents.daily_update", "agent.v2.yaml", GoalDirectedAgentManifest
        )
        self.graph = DailyUpdateGraph(self)

    async def run(self, handoff: AgentHandoff | AgentHandoffV2) -> AgentResult:
        state = DailyUpdateState(
            handoff=handoff,
            value=None,
            tool_input=None,
            report=None,
            card=None,
            instructions="",
            iterations=0,
            model_attempts=0,
            tool_calls=0,
            status=AgentRunStatus.RUNNING,
            stopped=False,
            result=None,
        )
        try:
            async with asyncio.timeout(handoff.budget.timeout_seconds):
                state = await self.graph.run(state)
        except Exception:
            state["status"] = AgentRunStatus.FAILED
        status = state["status"]
        output: dict[str, JsonValue]
        outcome: InteractionOutcome
        if status is AgentRunStatus.AWAITING_HUMAN and state["card"] is not None:
            output = cast(dict[str, JsonValue], state["card"].model_dump(mode="json"))
            code = "AWAIT_OWNER"
            outcome = InteractionOutcome.ACTION_REVIEW_REQUIRED
        elif status is AgentRunStatus.AWAITING_INPUT:
            output = {"question_key": "dailyUpdate.chat.clarify"}
            code = "NEEDS_INPUT"
            outcome = InteractionOutcome.NEEDS_INPUT
        else:
            status = AgentRunStatus.FAILED
            output = {"fallback": "manual_daily_update"}
            code = "DAILY_UPDATE_MANUAL_FALLBACK"
            outcome = InteractionOutcome.FAILED
        result = AgentResult(
            agent_id=AgentId.DAILY_UPDATE,
            agent_version="1.0.0",
            status=status,
            typed_output=output,
            requested_handoff=None,
            model_attempts_used=state["model_attempts"],
            iterations_used=state["iterations"],
            tool_calls_used=state["tool_calls"],
            stop_reason=code,
            safe_error_code=code if status is AgentRunStatus.FAILED else None,
            verifier_results=(
                VerifierResult(
                    verifier_id="daily_update_grounding",
                    verifier_version="1.0.0",
                    passed=status is not AgentRunStatus.FAILED,
                ),
            ),
        )
        if isinstance(handoff, AgentHandoffV2):
            return GoalDirectedAgentResult.model_validate(
                {
                    **result.model_dump(),
                    "agent_version": self.goal_manifest.agent.version,
                    "interaction_outcome": outcome,
                    "resolved_action": "daily_update.prepare" if state["value"] else None,
                }
            )
        return result

    async def _intake_v2(self, handoff: AgentHandoffV2) -> DailyUpdateHandoff | None:
        tasks = tuple(r for r in handoff.context_references if r.resource_type.upper() == "TASK")
        drafts = tuple(
            r for r in handoff.context_references if r.resource_type.upper() == "DAILY_UPDATE_DRAFT"
        )
        evidence = tuple(
            r for r in handoff.context_references if r.resource_type.upper() == "EVIDENCE"
        )
        if len(tasks) != 1 or tasks[0].version is None or len(drafts) > 1:
            return None
        if any(r.version is None or not r.freshness_required for r in (*tasks, *drafts, *evidence)):
            return None
        if self.source_turns is None or self.goal_access is None:
            raise ValueError("DAILY_UPDATE_CONTEXT_UNAVAILABLE")
        await self.goal_access.verify(handoff)
        report = await self.source_turns.read(handoff)
        if not report.strip():
            return None
        return DailyUpdateHandoff(
            text=report,
            locale=handoff.locale,
            task_id=tasks[0].resource_id,
            task_version=tasks[0].version,
            draft_id=drafts[0].resource_id if drafts else None,
            draft_version=drafts[0].version if drafts else None,
            evidence_refs=tuple(
                EvidenceRef(evidence_id=r.resource_id, version=r.version)
                for r in evidence
                if r.version is not None
            ),
        )

    async def _tool(
        self, state: DailyUpdateState, value: DailyUpdateToolInput
    ) -> dict[str, JsonValue]:
        handoff = state["handoff"]
        if state["tool_calls"] >= handoff.budget.max_tool_calls:
            raise ValueError("DAILY_UPDATE_TOOL_BUDGET")
        state["tool_calls"] += 1
        result = await self.tools.execute(
            ToolExecutionRequest(
                agent_run_id=uuid5(NAMESPACE_URL, f"agent-run:{handoff.idempotency_key}"),
                tool_id="daily_update.prepare",
                tool_version="1.0.0",
                call_id=f"{handoff.step_id}:{value.action}",
                actor=handoff.actor,
                typed_input=cast(dict[str, JsonValue], value.model_dump(mode="json")),
                idempotency_key=str(
                    uuid5(NAMESPACE_URL, f"daily:{handoff.idempotency_key}:{value.action}")
                ),
            )
        )
        if result.status != "SUCCEEDED":
            raise ValueError(result.safe_error_code or "DAILY_UPDATE_TOOL_FAILED")
        return result.typed_output

    async def execute_node(self, node: str, state: DailyUpdateState) -> None:
        handoff = state["handoff"]
        if node == "authorize":
            actor = await self.actors.resolve(handoff.actor)
            if any(r.organization_id != actor.organization_id for r in handoff.context_references):
                raise ValueError("DAILY_UPDATE_CONTEXT_TENANT_MISMATCH")
            if handoff.budget.max_model_attempts < 1:
                raise ValueError("DAILY_UPDATE_MODEL_BUDGET")
            if isinstance(handoff, AgentHandoffV2):
                skills = self.goal_manifest.allowed_skills
                PolicyGuard().authorize_goal_action(
                    current_actor=actor,
                    handoff=handoff,
                    manifest=self.goal_manifest,
                    resolved_action="daily_update.prepare",
                    action_risk=RiskLevel.PROPOSAL_ONLY,
                    requested_tool_ids=("daily_update.prepare@1",),
                    requested_skill_ids=skills,
                )
                value = await self._intake_v2(handoff)
                if value is None:
                    state["status"] = AgentRunStatus.AWAITING_INPUT
                    state["stopped"] = True
                    return
            else:
                skills = self.manifest.allowed_skills
                PolicyGuard().authorize_handoff(
                    current_actor=actor,
                    parent_agent_id=AgentId.ORCHESTRATOR,
                    handoff=handoff,
                    manifest=self.manifest,
                    requested_tool_ids=("daily_update.prepare@1",),
                    requested_skill_ids=skills,
                )
                value = DailyUpdateHandoff.model_validate(handoff.typed_input)
            state["value"] = value
            instructions: list[str] = []
            for reference in skills:
                name = reference.split("@", maxsplit=1)[0]
                package = f"work_management_ai.skills.{name}"
                skill = load_packaged_skill_manifest(package)
                instructions.append(read_skill_instructions(package, skill))
            state["instructions"] = "\n".join(instructions)
            state["tool_input"] = DailyUpdateToolInput(
                action="CONTEXT",
                task_id=value.task_id,
                task_version=value.task_version,
                evidence_refs=value.evidence_refs,
                draft_id=value.draft_id,
                draft_version=value.draft_version,
                inherit_draft_evidence=(
                    isinstance(handoff, AgentHandoffV2)
                    and value.draft_id is not None
                    and not value.evidence_refs
                ),
            )
            return
        value, base = state["value"], state["tool_input"]
        if value is None or base is None:
            raise ValueError("DAILY_UPDATE_STATE_INVALID")
        if node == "context":
            snapshot = ReportingSnapshot.model_validate(await self._tool(state, base))
            verify_context(value, snapshot)
            if isinstance(handoff, AgentHandoffV2):
                if base.inherit_draft_evidence:
                    state["tool_input"] = base.model_copy(
                        update={
                            "evidence_refs": snapshot.evidence_refs,
                            "inherit_draft_evidence": False,
                        }
                    )
                elif snapshot.evidence_refs != base.evidence_refs:
                    raise ValueError("DAILY_UPDATE_STALE_EVIDENCE")
        elif node == "extract_report":
            state["model_attempts"] += 1
            system = SYSTEM_V1 + "\n\n## Skill instructions\n" + state["instructions"]
            response = await self.gateway.generate_structured(
                StructuredModelRequest(
                    invocation_key=f"daily_update.{value.locale}.extract",
                    messages=(
                        ModelMessage(role="system", content=system),
                        ModelMessage(
                            role="user",
                            content=json.dumps(
                                {"report": value.text, "locale": value.locale}, ensure_ascii=False
                            ),
                        ),
                    ),
                    output_schema=ExtractedReport,
                    timeout_seconds=min(60, handoff.budget.timeout_seconds),
                    max_output_tokens=min(1500, handoff.budget.max_output_tokens),
                )
            )
            report = response.parsed
            state["report"] = report
            if (
                report.needs_clarification
                or report.reported_percent is None
                or not report.done_text.strip()
            ):
                state["status"] = AgentRunStatus.AWAITING_INPUT
                state["stopped"] = True
        elif node == "save_draft":
            report = state["report"]
            if report is None:
                raise ValueError("DAILY_UPDATE_REPORT_REQUIRED")
            draft = DailyUpdateResult.model_validate(
                await self._tool(
                    state, base.model_copy(update={"action": "DRAFT", "report": report})
                )
            )
            if isinstance(handoff, AgentHandoffV2) and (
                (draft.task_id, draft.task_version) != (base.task_id, base.task_version)
                or draft.draft_version != (base.draft_version or 0) + 1
                or (base.draft_id is not None and draft.draft_id != base.draft_id)
            ):
                raise ValueError("DAILY_UPDATE_DRAFT_MISMATCH")
            state["card"] = draft
        elif node == "assess_originals":
            card = state["card"]
            if card is None:
                raise ValueError("DAILY_UPDATE_DRAFT_REQUIRED")
            assessed = DailyUpdateResult.model_validate(
                await self._tool(
                    state,
                    base.model_copy(
                        update={
                            "action": "ASSESS",
                            "draft_id": card.draft_id,
                            "draft_version": card.draft_version,
                        }
                    ),
                )
            )
            if isinstance(handoff, AgentHandoffV2) and (
                assessed.assessment_id is None
                or assessed.model_copy(update={"assessment_id": card.assessment_id}) != card
            ):
                raise ValueError("DAILY_UPDATE_ASSESSMENT_MISMATCH")
            state["card"] = assessed
        elif node == "await_owner":
            state["status"] = AgentRunStatus.AWAITING_HUMAN
            state["stopped"] = True
        else:
            raise ValueError("DAILY_UPDATE_NODE_NOT_ALLOWED")
