"""Goal-directed declarations and exact version compatibility."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from work_management_ai.runtime.activation import ActivationContext
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentBudget,
    AgentHandoffV2,
    AgentId,
    AgentRunStatus,
    ContextReference,
    DelegationBrief,
    GoalDirectedAgentResult,
    InteractionOutcome,
    ResolvedActorContext,
    RiskLevel,
)
from work_management_ai.runtime.delegation import (
    DelegationAdapterV2,
    VerifiedDelegationReferences,
    filtered_delegate_v2_tool_schema,
)
from work_management_ai.runtime.policy_guard import AgentPolicyError, PolicyGuard
from work_management_ai.runtime.run_budget import CandidateBudget, candidate_budget
from work_management_ai.runtime.tool_loop_contracts import RunScope
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.delegation.contracts import DelegateAgentInputV2

from .test_delegate_agent import (
    _agent_registry,  # pyright: ignore[reportPrivateUsage]
    _context,  # pyright: ignore[reportPrivateUsage]
    _FakeBudget,  # pyright: ignore[reportPrivateUsage]
    _scope,  # pyright: ignore[reportPrivateUsage]
)


def test_delegate_v2_exports_flat_scoped_schema():
    registry = ToolRegistry.from_resources(
        (("work_management_ai.tools.delegation", "delegate.v2.yaml"),)
    )
    (base,) = registry.candidate_schemas(("delegate_agent@2",))
    schema = filtered_delegate_v2_tool_schema(
        base,
        allowed_specialists=(
            (AgentId.RISK, "Explain persisted risk with evidence."),
            (AgentId.WORK_INTELLIGENCE, "Answer authorized work questions."),
        ),
    )
    properties = schema.parameters["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {
        "agent_id",
        "objective",
        "context_ref_ids",
        "constraints",
        "expected_result",
    }
    assert "anyOf" not in json.dumps(schema.parameters)
    assert "Explain persisted risk" in schema.description
    assert "planning" not in schema.description
    assert schema.input_contract is not None
    with pytest.raises(ValidationError):
        schema.input_contract.model_validate(
            {"agent_id": "planning", "objective": "Plan", "expected_result": "Draft"}
        )


def test_delegate_v2_rejects_self_and_forged_source_turn():
    with pytest.raises(ValidationError, match="DELEGATION_SELF_FORBIDDEN"):
        DelegateAgentInputV2(
            agent_id=AgentId.ORCHESTRATOR, objective="Loop", expected_result="Loop"
        )
    with pytest.raises(ValidationError, match="source_turn_ref"):
        DelegateAgentInputV2.model_validate(
            {
                "agent_id": "risk",
                "objective": "Review",
                "expected_result": "Evidence",
                "source_turn_ref": "forged",
            }
        )


@pytest.mark.parametrize("agent", [agent for agent in AgentId if agent is not AgentId.ORCHESTRATOR])
def test_exact_legacy_specialist_pin_resolves_after_registering_v2(agent: AgentId):
    registry = _agent_registry(enable_retrieve=True)
    activation = ActivationContext(
        active_phase=5,
        core_exit_verified=True,
        active_tracks=frozenset({"I3"}),
        provider_ready=True,
    )
    previous = registry.resolve(agent, "1.0.0", 5, activation=activation)
    current = registry.register_resource(
        f"work_management_ai.agents.{agent.value}", "agent.v2.yaml"
    )
    assert current.manifest.agent.version == "1.1.0"
    assert current.manifest.permissions == previous.manifest.permissions
    assert current.manifest.capabilities == previous.manifest.capabilities
    assert current.fingerprint != previous.fingerprint
    assert registry.resolve(agent, "1.0.0", 5, activation=activation) == previous
    assert all(
        entry["agent_version"] == "1.0.0"
        for entry in registry.planning_catalog(
            active_phase=5, role="MANAGER", activation=activation
        )
    )


def test_goal_handoff_policy_checks_selected_operation_and_risk_before_tool():
    registry = _agent_registry()
    manifest = registry.register_resource(
        "work_management_ai.agents.work_intelligence", "agent.v2.yaml"
    ).manifest
    actor = ResolvedActorContext(
        organization_id=uuid4(), membership_id=uuid4(), role="EMPLOYEE", is_active=True
    )
    handoff = AgentHandoffV2(
        orchestration_run_id=uuid4(),
        parent_agent_run_id=uuid4(),
        target_agent_id=AgentId.WORK_INTELLIGENCE,
        target_agent_version="1.1.0",
        brief=DelegationBrief(objective="Read tasks", expected_result="Task status"),
        context_references=(),
        actor=ActorReference(
            organization_id=actor.organization_id, membership_id=actor.membership_id
        ),
        source_turn_ref=uuid4(),
        locale="en",
        timezone="UTC",
        budget=AgentBudget(max_iterations=1, max_tool_calls=1, timeout_seconds=1),
        step_id="child",
        idempotency_key="child",
    )

    def authorize(action: str, risk: RiskLevel) -> None:
        PolicyGuard().authorize_goal_action(
            current_actor=actor,
            handoff=handoff,
            manifest=manifest,
            resolved_action=action,
            action_risk=risk,
        )

    authorize("work.answer_question", RiskLevel.READ_ONLY)
    with pytest.raises(AgentPolicyError, match="CAPABILITY_NOT_ALLOWED"):
        authorize("assignment.assign_task_explicitly", RiskLevel.READ_ONLY)
    with pytest.raises(AgentPolicyError, match="ACTION_RISK_EXCEEDED"):
        authorize("work.answer_question", RiskLevel.PROPOSAL_ONLY)


@pytest.mark.asyncio
async def test_v2_handoff_is_server_bound_and_only_ready_specialists_dispatch():
    registry = _agent_registry()
    registry.register_resource("work_management_ai.agents.work_intelligence", "agent.v2.yaml")
    registry.register_resource("work_management_ai.agents.risk", "agent.v2.yaml")
    seen: list[AgentHandoffV2] = []
    child_run_id = uuid4()

    class Runner:
        async def run(self, handoff: AgentHandoffV2):
            seen.append(handoff)
            return child_run_id, GoalDirectedAgentResult(
                agent_id=handoff.target_agent_id,
                agent_version=handoff.target_agent_version,
                status=AgentRunStatus.COMPLETED,
                interaction_outcome=InteractionOutcome.RESULT_READY,
                typed_output={"summary": "Task status"},
                resolved_action="work.answer_question",
                stop_reason="VERIFIED",
            )

        async def reconcile(self, handoff: AgentHandoffV2):
            return None

    scope = _scope()
    parent_id = uuid4()
    adapter = DelegationAdapterV2(
        scope=scope,
        parent_agent_run_id=parent_id,
        actor_role="EMPLOYEE",
        active_phase=5,
        locale="en",
        timezone="UTC",
        agent_registry=registry,
        specialist_pins=registry.specialist_pins(goal_directed=True),
        runners={AgentId.WORK_INTELLIGENCE: Runner()},
    )
    args = DelegateAgentInputV2(
        agent_id=AgentId.WORK_INTELLIGENCE, objective="Read tasks", expected_result="Status"
    )
    context = _context(scope, args).model_copy(
        update={"tool_reference": "delegate_agent@2", "tool_version": "2.0.0"}
    )
    budget = _FakeBudget()
    binding = CandidateBudget(scope=scope, lease=context.lease, store=budget, interval_id=uuid4())
    with candidate_budget(binding):
        result = await adapter.execute(context, args)
        assert result.output is not None
        assert result.output.model_dump()["child_agent_run_id"] == child_run_id
        unavailable = args.model_copy(update={"agent_id": AgentId.RISK})
        with pytest.raises(ToolPolicyError, match="DELEGATION_TARGET_UNAVAILABLE"):
            await adapter.execute(
                context.model_copy(update={"canonical_arguments": unavailable}), unavailable
            )
    assert len(seen) == 1
    assert seen[0].source_turn_ref == scope.turn_id
    assert seen[0].parent_agent_run_id == parent_id
    assert seen[0].target_agent_version == "1.1.0"


@pytest.mark.asyncio
async def test_verified_reference_lookup_rechecks_every_source_and_scope():
    scope = _scope()
    refs = tuple(
        ContextReference(
            reference_id=uuid4(),
            organization_id=scope.organization_id,
            resource_type=resource,
            resource_id=uuid4(),
            version=1,
            observed_at=datetime.now(UTC),
        )
        for resource in ("project", "proposal", "task")
    )
    checked: list[tuple[ContextReference, ...]] = []

    async def recheck(current: RunScope, candidates: tuple[ContextReference, ...]) -> None:
        assert current == scope
        checked.append(candidates)
        if len(checked) > 1:
            raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)

    lookup = VerifiedDelegationReferences(
        scope=scope,
        exploration=refs[:1],
        active_context=refs[1:2],
        completed_specialists=refs[2:],
        recheck=recheck,
    )
    supplied = tuple(ref.reference_id for ref in refs)
    assert await lookup.resolve(scope, supplied) == refs
    with pytest.raises(ToolPolicyError, match="CONTEXT_STALE"):
        await lookup.resolve(scope, supplied)
    with pytest.raises(ToolPolicyError, match="DELEGATION_CONTEXT_DENIED"):
        await lookup.resolve(_scope(), supplied)
    assert checked == [refs, refs]
