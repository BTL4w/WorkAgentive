"""Resolve clarification from an accepted user message and canonical Explore receipt."""

from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.assistant.adapters.context_tools import revalidate_refs
from app.modules.assistant.adapters.continuation_models import OrchestrationWaitModel
from app.modules.assistant.adapters.session_repository import PostgreSQLSessionRepository
from app.modules.assistant.adapters.tool_loop_recorder import session_scope
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.context_service import ContextReadError, ContextReadService
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.context_exploration import (
    ContextAccessDecision,
    ContextUseAccessPort,
)
from work_management_ai.runtime.continuation import ResolvedContextBinding
from work_management_ai.runtime.session_history import ToolResultPayload
from work_management_ai.runtime.tool_loop_contracts import (
    RunScope,
    ToolLoopCheckpoint,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.tools.context.contracts import ContextReadObservation, ContextRef


async def resolve_input(
    journal: PostgreSQLSessionRepository,
    reference: WaitReference,
    *,
    user_turn_id: UUID,
    content_blocks: list[dict[str, object]],
    access: ContextUseAccessPort | None,
) -> ResolvedContextBinding | None:
    scope = session_scope(reference.scope)
    events = await journal.events(scope, invocation_id=reference.source_invocation_id)
    source = next(
        (
            event.event
            for event in events
            if event.event.kind == "tool.completed"
            and event.event.tool_reference == "explore_projects@1"
            and event.event.settlement_status == "COMPLETED"
        ),
        None,
    )
    if source is None:
        # Existing validated Planning card actions carry the exact workflow ID.
        for block in content_blocks:
            candidate = block.get("action")
            if block.get("kind") != "accepted_card_action" or not isinstance(candidate, dict):
                continue
            action = cast(dict[str, object], candidate)
            if (
                reference.resource_id is not None
                and action.get("kind") == "PLANNING_INPUT"
                and action.get("workflow_run_id") == str(reference.resource_id)
                and action.get("expected_version") == reference.resource_version
            ):
                return None
        raise SessionError("CONTINUATION_TARGET_UNRELATED")
    if access is None or source.payload_id is None:
        raise SessionError("CONTINUATION_CLARIFICATION_REQUIRED")
    payload = await journal.payload(scope, source.payload_id)
    if payload is None or not await journal.payload_live(scope, payload.id):
        raise SessionError("CONTEXT_EXPIRED")
    result = ToolResultPayload.model_validate(payload.body)
    observation = ContextReadObservation.model_validate(result.output)
    if (
        observation.scope != reference.scope
        or observation.invocation_id != reference.source_invocation_id
    ):
        raise SessionError("CONTINUATION_WAIT_MISMATCH")
    if len(observation.entities) != 1 or observation.entities[0].resolution != "AMBIGUOUS":
        raise SessionError("CONTINUATION_TARGET_UNRELATED")
    texts = [
        value.strip().casefold()
        for block in content_blocks
        if block.get("kind") == "text" and isinstance(value := block.get("text"), str)
    ]
    if len(texts) != 1 or not texts[0]:
        raise SessionError("CONTINUATION_TARGET_UNRELATED")
    entity = observation.entities[0]
    names = {
        fact.ref_id: fact.value.strip().casefold()
        for fact in observation.facts
        if fact.field == "project.name" and isinstance(fact.value, str)
    }
    candidates: tuple[ContextRef, ...] = tuple(
        ref
        for ref in observation.context_refs
        if ref.ref_id in entity.candidate_ref_ids
        and ref.resource_type == "project"
        and texts[0] in {names.get(ref.ref_id), str(ref.resource_id)}
    )
    if len(candidates) != 1 or candidates[0].version is None:
        raise SessionError("CONTINUATION_TARGET_UNRELATED")
    selected = candidates[0]
    decision = await access.check(
        reference.scope,
        target_tool="delegate_agent@2",
        input_fingerprint=entity.query_fingerprint,
        refs=(selected,),
    )
    if decision.outcome != "ACCEPTED":
        raise SessionError("CONTINUATION_CONTEXT_REFRESH_REQUIRED")
    state = await journal.state(scope)
    if state is None or state.checkpoint_payload_id is None:
        raise SessionError("CONTINUATION_CHECKPOINT_REQUIRED")
    stored = await journal.payload(scope, state.checkpoint_payload_id)
    if stored is None or not await journal.payload_live(scope, stored.id):
        raise SessionError("CONTEXT_EXPIRED")
    checkpoint = ToolLoopCheckpoint.model_validate(stored.body)
    pin = next(
        (
            pin
            for pin in checkpoint.tool_versions
            if pin.reference == observation.tool_reference
            and pin.version == observation.tool_version
        ),
        None,
    )
    if pin is None:
        raise SessionError("PIN_CONFLICT")
    assert selected.version is not None
    return ResolvedContextBinding(
        source_scope=reference.scope,
        user_turn_id=user_turn_id,
        source_invocation_id=observation.invocation_id,
        query_fingerprint=entity.query_fingerprint,
        source_tool_pin=pin,
        selected_ref_id=selected.ref_id,
        resource_id=selected.resource_id,
        resource_version=selected.version,
        resource_fingerprint=selected.fingerprint,
        actor_revision=decision.actor_revision,
        policy_revision=decision.policy_revision,
    )


class PostgreSQLUserClarifications:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], actor: AuthenticatedActor
    ) -> None:
        self._transactions = PostgreSQLAssistantTransactionFactory(sessions)
        self._actor = actor

    async def accepted(
        self, *, source_scope: RunScope, user_turn_id: UUID, source_invocation_id: UUID
    ) -> ResolvedContextBinding | None:
        if (source_scope.organization_id, source_scope.membership_id) != (
            self._actor.organization_id,
            self._actor.membership_id,
        ):
            return None
        async with self._transactions(self._actor) as tx:
            journal = PostgreSQLSessionRepository(tx.session)
            scope = session_scope(source_scope)
            if not await journal.authorize(scope, self._actor):
                return None
            await journal.retention_deadline(scope)
            row = OrchestrationWaitModel
            stored = await tx.session.scalar(
                select(row.context_binding).where(
                    row.organization_id == scope.organization_id,
                    row.source_run_id == scope.orchestration_run_id,
                    row.source_turn_id == scope.turn_id,
                    row.conversation_id == scope.conversation_id,
                    row.membership_id == scope.membership_id,
                    row.owner_turn_id == user_turn_id,
                    row.source_invocation_id == source_invocation_id,
                    row.context_binding.is_not(None),
                    row.closed.is_(False),
                )
            )
            return ResolvedContextBinding.model_validate(stored) if stored is not None else None


class ApplicationClarificationAccess:
    def __init__(self, contexts: ContextReadService, actors: CurrentActorService) -> None:
        self.contexts, self.actors = contexts, actors

    async def check(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision:
        current = await self.actors.resolve(
            organization_id=scope.organization_id,
            membership_id=scope.membership_id,
        )
        if current is None:
            return ContextAccessDecision(
                outcome="DENIED",
                actor_revision="unavailable",
                policy_revision="unavailable",
            )
        revision = f"candidate-v2:{current.role.value}"
        try:
            await revalidate_refs(service=self.contexts, actor=current, scope=scope, refs=refs)
        except (ContextReadError, ToolPolicyError):
            return ContextAccessDecision(
                outcome="REFRESH_REQUIRED",
                actor_revision=revision,
                policy_revision=revision,
            )
        return ContextAccessDecision(
            outcome="ACCEPTED",
            actor_revision=revision,
            policy_revision=revision,
        )
