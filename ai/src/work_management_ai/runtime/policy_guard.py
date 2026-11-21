"""Deterministic delegation policy and operation budgets."""

from dataclasses import dataclass
from typing import cast

from work_management_ai.runtime.activation import ActivationContext, capability_enabled
from work_management_ai.runtime.contracts import (
    AgentBudget,
    AgentHandoff,
    AgentHandoffV2,
    AgentId,
    JsonValue,
    ResolvedActorContext,
    RiskLevel,
)
from work_management_ai.runtime.manifests import AnyAgentManifest, GoalDirectedAgentManifest

_RESERVED_AUTHORITY_KEYS = frozenset(
    {
        "organization_id",
        "role",
        "allowed_tools",
        "approved",
        "approval_id",
        "activation",
        "core_exit_verified",
        "active_tracks",
        "active_phase",
        "provider_ready",
    }
)
_RISK_ORDER = {RiskLevel.READ_ONLY: 0, RiskLevel.PROPOSAL_ONLY: 1, RiskLevel.EXPLICIT_WRITE: 2}


class AgentPolicyError(ValueError):
    pass


class AgentBudgetExceededError(ValueError):
    pass


@dataclass(slots=True)
class BudgetTracker:
    budget: AgentBudget
    iterations: int = 0
    tool_calls: int = 0
    handoffs: int = 0
    replans: int = 0

    def start_iteration(self) -> None:
        if self.iterations >= self.budget.max_iterations:
            raise AgentBudgetExceededError("ITERATION_BUDGET_EXHAUSTED")
        self.iterations += 1

    def start_tool_call(self) -> None:
        if self.tool_calls >= self.budget.max_tool_calls:
            raise AgentBudgetExceededError("TOOL_BUDGET_EXHAUSTED")
        self.tool_calls += 1

    def start_handoff(self) -> None:
        if self.handoffs >= self.budget.max_handoffs:
            raise AgentBudgetExceededError("HANDOFF_BUDGET_EXHAUSTED")
        self.handoffs += 1

    def start_replan(self) -> None:
        if self.replans >= self.budget.max_replans:
            raise AgentBudgetExceededError("REPLAN_BUDGET_EXHAUSTED")
        self.replans += 1


class PolicyGuard:
    def authorize_handoff(
        self,
        *,
        current_actor: ResolvedActorContext,
        parent_agent_id: AgentId,
        handoff: AgentHandoff,
        manifest: AnyAgentManifest,
        requested_skill_ids: tuple[str, ...] = (),
        requested_tool_ids: tuple[str, ...] = (),
        activation: ActivationContext | None = None,
    ) -> None:
        if parent_agent_id is not AgentId.ORCHESTRATOR:
            raise AgentPolicyError("ORCHESTRATOR_REQUIRED")
        if isinstance(manifest, GoalDirectedAgentManifest):
            raise AgentPolicyError("AGENT_HANDOFF_SCHEMA_MISMATCH")
        self._authorize_action(
            current_actor=current_actor,
            handoff=handoff,
            manifest=manifest,
            capability=handoff.capability,
            requested_skill_ids=requested_skill_ids,
            requested_tool_ids=requested_tool_ids,
            activation=activation,
        )
        self._reject_reserved_keys(handoff.typed_input)

    def authorize_goal_action(
        self,
        *,
        current_actor: ResolvedActorContext,
        handoff: AgentHandoffV2,
        manifest: AnyAgentManifest,
        resolved_action: str,
        action_risk: RiskLevel,
        requested_skill_ids: tuple[str, ...] = (),
        requested_tool_ids: tuple[str, ...] = (),
        activation: ActivationContext | None = None,
    ) -> None:
        """Authorize the operation selected by a Specialist before business tool use."""
        if not isinstance(manifest, GoalDirectedAgentManifest):
            raise AgentPolicyError("AGENT_HANDOFF_SCHEMA_MISMATCH")
        self._authorize_action(
            current_actor=current_actor,
            handoff=handoff,
            manifest=manifest,
            capability=resolved_action,
            requested_skill_ids=requested_skill_ids,
            requested_tool_ids=requested_tool_ids,
            activation=activation,
        )
        if _RISK_ORDER[action_risk] > _RISK_ORDER[manifest.permissions.risk_ceiling]:
            raise AgentPolicyError("ACTION_RISK_EXCEEDED")

    def _authorize_action(
        self,
        *,
        current_actor: ResolvedActorContext,
        handoff: AgentHandoff | AgentHandoffV2,
        manifest: AnyAgentManifest,
        capability: str,
        requested_skill_ids: tuple[str, ...],
        requested_tool_ids: tuple[str, ...],
        activation: ActivationContext | None,
    ) -> None:
        if not current_actor.is_active:
            raise AgentPolicyError("ACTOR_INACTIVE")
        if (
            handoff.actor.membership_id != current_actor.membership_id
            or handoff.actor.organization_id != current_actor.organization_id
        ):
            raise AgentPolicyError("ACTOR_CONTEXT_MISMATCH")
        if current_actor.role not in manifest.permissions.roles:
            raise AgentPolicyError("AGENT_ROLE_FORBIDDEN")
        if handoff.target_agent_id is not manifest.agent.id:
            raise AgentPolicyError("AGENT_ID_MISMATCH")
        if handoff.target_agent_version != manifest.agent.version:
            raise AgentPolicyError("AGENT_VERSION_MISMATCH")
        if capability not in manifest.capabilities:
            raise AgentPolicyError("CAPABILITY_NOT_ALLOWED")
        if manifest.agent.id is AgentId.RETRIEVE and not capability_enabled(capability, activation):
            raise AgentPolicyError("AGENT_CAPABILITY_INACTIVE")
        self._validate_budget(handoff.budget, manifest)
        if not set(requested_skill_ids).issubset(manifest.allowed_skills):
            raise AgentPolicyError("SKILL_NOT_ALLOWED")
        if not set(requested_tool_ids).issubset(manifest.allowed_tools):
            raise AgentPolicyError("TOOL_NOT_ALLOWED")

    @staticmethod
    def _validate_budget(budget: AgentBudget, manifest: AnyAgentManifest) -> None:
        maximum = manifest.runtime
        if (
            budget.max_model_attempts > maximum.max_model_attempts
            or budget.max_input_tokens > maximum.max_input_tokens
            or budget.max_output_tokens > maximum.max_output_tokens
            or budget.max_iterations > maximum.max_iterations
            or budget.max_tool_calls > maximum.max_tool_calls
            or budget.max_handoffs > maximum.max_handoffs
            or budget.max_replans > maximum.max_replans
            or budget.timeout_seconds > maximum.timeout_seconds
        ):
            raise AgentPolicyError("AGENT_BUDGET_EXCEEDS_MANIFEST")

    @classmethod
    def _reject_reserved_keys(cls, value: JsonValue) -> None:
        if isinstance(value, dict):
            if _RESERVED_AUTHORITY_KEYS.intersection(value):
                raise AgentPolicyError("RESERVED_AUTHORITY_KEY")
            for child in value.values():
                cls._reject_reserved_keys(cast(JsonValue, child))
        elif isinstance(value, list):
            for child in value:
                cls._reject_reserved_keys(cast(JsonValue, child))
