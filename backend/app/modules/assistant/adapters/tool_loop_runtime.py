"""Versioned candidate artifacts and trusted Assistant driver admission."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from secrets import token_bytes
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.modules.assistant.adapters.candidate_access import CandidateAccess
from app.modules.assistant.adapters.context_tools import ContextToolBackend
from app.modules.assistant.adapters.database_models import AssistantJobModel
from app.modules.assistant.adapters.execution_recorder import PostgreSQLExecutionRecorder
from app.modules.assistant.adapters.report_chat_usage import report_chat_job_scope
from app.modules.assistant.adapters.retrieval_runtime import retrieval_job_scope
from app.modules.assistant.adapters.session_continuations import PostgreSQLContinuations
from app.modules.assistant.adapters.session_repository import (
    PostgreSQLSessionRepository,
    PostgreSQLSessionTransactionFactory,
)
from app.modules.assistant.adapters.session_usage import PostgreSQLRunBudget
from app.modules.assistant.adapters.tool_loop_recorder import (
    PostgreSQLToolLoopRecorder,
    session_lease,
    session_scope,
)
from app.modules.assistant.adapters.transaction import PostgreSQLAssistantTransactionFactory
from app.modules.assistant.application.context_service import ContextReadService
from app.modules.assistant.application.driver_admission import PinnedDriver
from app.modules.assistant.application.execution_service import AssistantExecutionPort
from app.modules.assistant.application.session_service import SessionService
from app.modules.assistant.domain.continuation import WaitDraft, WakeReceipt
from app.modules.assistant.domain.models import AssistantJob, AssistantTurn
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.work.adapters.project_repository import SqlAlchemyProjectTransactionFactory
from app.modules.work.application.project_service import ProjectService
from work_management_ai.agents.orchestrator.contracts import (
    ActiveConversationContext,
    OrchestratorInput,
    OrchestratorStatus,
)
from work_management_ai.agents.orchestrator.evaluators.tool_loop import (
    ORCHESTRATOR_TOOL_LOOP_EVALUATORS,
)
from work_management_ai.agents.orchestrator.prompts.tool_loop_v1 import PROMPT_VERSION
from work_management_ai.agents.orchestrator.tool_loop import ToolLoopOrchestrator
from work_management_ai.agents.orchestrator.tool_loop_contracts import (
    OrchestratorOutputV2,
    PublishedResponse,
)
from work_management_ai.agents.orchestrator.workflows.tool_loop_graph import WORKFLOW_ID
from work_management_ai.evaluation.context_exploration import CONTEXT_EXPLORATION_EVALUATORS
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.openai_tools import OpenAINativeToolGateway
from work_management_ai.model_gateway.tool_contracts import (
    NativeToolGateway,
    ToolModelRequest,
    ToolModelResponse,
)
from work_management_ai.runtime.activation import ActivationContext
from work_management_ai.runtime.agent_registry import AgentRegistry, RegisteredToolLoopAgent
from work_management_ai.runtime.context_exploration import ExplorationProjector
from work_management_ai.runtime.continuation import ResolvedContextBinding
from work_management_ai.runtime.contracts import (
    ActorReference,
    AgentId,
    ContextReference,
    TextResponseBlock,
)
from work_management_ai.runtime.manifests import (
    canonical_manifest_fingerprint,
    load_agent_manifest_resource,
)
from work_management_ai.runtime.run_budget import BudgetedToolLoopGateway
from work_management_ai.runtime.session_history import (
    ContextPayload,
    MessagePayload,
    SessionHistory,
)
from work_management_ai.runtime.session_log import EventDraft
from work_management_ai.runtime.skill_loading import (
    ApprovedPackageResources,
    SkillLoader,
    SkillLoaderBackend,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_catalog import CatalogAccess, ToolCatalog
from work_management_ai.runtime.tool_loop_contracts import (
    RunLease,
    RunScope,
    ToolLoopCheckpoint,
    VersionPin,
)
from work_management_ai.runtime.tool_pipeline import RecoverableToolBackend, ToolPipeline
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context.contracts import ContextReadObservation

_TOOL_RESOURCES = (
    ("work_management_ai.tools.skills", "load.yaml"),
    ("work_management_ai.tools.context", "projects.v103.yaml"),
    ("work_management_ai.tools.context", "project.yaml"),
    ("work_management_ai.tools.context", "actor.yaml"),
    ("work_management_ai.tools.delegation", "delegate.v2.yaml"),
    ("work_management_ai.tools.automation", "tool.v102.yaml"),
)
_SKILL_PACKAGE = "work_management_ai.skills.explore_work_context"
_DRIVER_VERSION = "tool_loop.v2"
_SPECIALIST_IDS = (
    AgentId.ASSIGNMENT,
    AgentId.DAILY_UPDATE,
    AgentId.PLANNING,
    AgentId.REPORTING,
    AgentId.RETRIEVE,
    AgentId.RISK,
    AgentId.WORK_INTELLIGENCE,
)


@dataclass(frozen=True, slots=True)
class CandidateArtifacts:
    agent: RegisteredToolLoopAgent
    tools: ToolRegistry
    skills: SkillRegistry
    registry: AgentRegistry
    skill_bundles: tuple[VersionPin, ...]
    specialist_pins: tuple[VersionPin, ...]


def build_candidate_artifacts() -> CandidateArtifacts:
    tools = ToolRegistry.from_resources(_TOOL_RESOURCES)
    skills = SkillRegistry.from_resources(((_SKILL_PACKAGE, "skill.yaml"),))
    evaluators = {**CONTEXT_EXPLORATION_EVALUATORS, **ORCHESTRATOR_TOOL_LOOP_EVALUATORS}
    registry = AgentRegistry(
        skill_registry=skills,
        tool_registry=tools,
        evaluator_ids=frozenset(evaluators),
        evaluator_bindings=evaluators,
    )
    agent = registry.register_tool_loop_resource(
        "work_management_ai.agents.orchestrator", "agent.v2.yaml"
    )
    if agent.manifest.runtime.workflow != WORKFLOW_ID:
        raise SessionError("DRIVER_WORKFLOW_PIN_MISMATCH")
    bundles: list[VersionPin] = []
    for pin in agent.skill_pins:
        entry = skills.resolve(pin.reference)
        if entry.bundle is None:
            raise SessionError("DRIVER_SKILL_DESCRIPTOR_MISSING")
        bundles.append(
            VersionPin(
                reference=entry.bundle.name,
                version=pin.version,
                fingerprint=canonical_manifest_fingerprint(entry.bundle),
            )
        )
    specialist_pins: list[VersionPin] = []
    for identifier in _SPECIALIST_IDS:
        manifest = load_agent_manifest_resource(
            f"work_management_ai.agents.{identifier.value}", "agent.v2.yaml"
        )
        specialist_pins.append(
            VersionPin(
                reference=identifier.value,
                version=manifest.agent.version,
                fingerprint=canonical_manifest_fingerprint(manifest),
            )
        )
    return CandidateArtifacts(
        agent, tools, skills, registry, tuple(bundles), tuple(specialist_pins)
    )


def register_candidate_specialists(registry: AgentRegistry, *, retrieval_enabled: bool) -> None:
    for identifier in _SPECIALIST_IDS:
        if identifier == AgentId.RETRIEVE and not retrieval_enabled:
            continue
        registry.register_resource(f"work_management_ai.agents.{identifier.value}", "agent.v2.yaml")


def build_native_gateway(settings: Settings) -> NativeToolGateway:
    if settings.ai_provider == "openai" and settings.openai_api_key is not None:
        return OpenAINativeToolGateway(
            model_name=settings.ai_model, api_key=settings.openai_api_key
        )
    return MockModelGateway(fixtures={}, model_ref=settings.ai_model or "mock")


class DriverAdmission:
    def __init__(self, settings: Settings, artifacts: CandidateArtifacts) -> None:
        self._settings = settings
        self._artifacts = artifacts

    def _pin(self) -> PinnedDriver:
        provider = self._settings.ai_provider
        if provider == "disabled":
            raise SessionError("DRIVER_MODEL_UNAVAILABLE")
        if provider == "openai" and (
            self._settings.openai_api_key is None or not self._settings.ai_model
        ):
            raise SessionError("DRIVER_MODEL_UNAVAILABLE")
        agent = self._artifacts.agent
        return PinnedDriver(
            driver_version=_DRIVER_VERSION,
            agent_version=agent.manifest.agent.version,
            manifest_fingerprint=agent.fingerprint,
            prompt_version=PROMPT_VERSION,
            workflow_id=agent.manifest.runtime.workflow,
            model_provider=provider,
            model_ref=self._settings.ai_model or "mock",
            tool_versions=agent.tool_pins,
            skill_versions=agent.skill_pins,
            skill_bundle_fingerprints=self._artifacts.skill_bundles,
            specialist_versions=self._artifacts.specialist_pins,
        )

    def resolve_saved(self, pin: PinnedDriver) -> PinnedDriver:
        if pin != self._pin():
            raise SessionError("DRIVER_PIN_MISMATCH")
        return pin

    def select(
        self,
        actor: AuthenticatedActor,
        *,
        verified_action: bool,
        continuation_source: PinnedDriver | Literal["legacy"] | None,
    ) -> PinnedDriver | None:
        if verified_action or continuation_source == "legacy":
            return None
        if isinstance(continuation_source, PinnedDriver):
            return self.resolve_saved(continuation_source)
        settings = self._settings
        if (
            settings.orchestrator_driver != "tool_loop"
            or actor.organization_id not in settings.orchestrator_tool_loop_organizations
            or actor.role.value not in self._artifacts.agent.manifest.permissions.roles
            or settings.ai_provider == "disabled"
        ):
            return None
        return self._pin()


class _CanonicalExploration:
    def __init__(self, scope: RunScope):
        self.scope = scope
        self.observations: dict[UUID, ContextReadObservation] = {}

    def completed(self, *, scope: RunScope, invocation_id: UUID) -> ContextReadObservation | None:
        if scope != self.scope:
            return None
        return self.observations.get(invocation_id)


class _AuthorizedGateway:
    def __init__(self, gateway: NativeToolGateway, access: CandidateAccess, scope: RunScope):
        self.gateway, self.access, self.scope = gateway, access, scope

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        await self.access.ensure_authorized(self.scope)
        return await self.gateway.generate_step(request)


@dataclass(frozen=True, slots=True)
class CandidateBackends:
    backends: Mapping[str, RecoverableToolBackend]
    specialists: Callable[[RunScope], tuple[tuple[AgentId, str], ...]] | None = None
    observe: Callable[[tuple[ContextReference, ...]], None] | None = None


type BackendFactory = Callable[
    [
        AuthenticatedActor,
        RunScope,
        AssistantJob,
        PostgreSQLExecutionRecorder,
        UUID,
        CandidateAccess,
        ExplorationProjector,
        Literal["vi", "en"],
    ],
    Awaitable[CandidateBackends],
]


class ResponseBlockProjector(Protocol):
    async def prepare_for_storage(
        self, actor: AuthenticatedActor, blocks: tuple[dict[str, object], ...], /
    ) -> tuple[dict[str, object], ...]: ...


type ActiveContextSeed = Callable[
    [AuthenticatedActor, AssistantTurn, RunScope], Awaitable[tuple[ContextReference, ...]]
]


class ToolLoopTurnExecutor:
    """Consume an already admitted run without selecting a new driver or model."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        transactions: PostgreSQLAssistantTransactionFactory,
        actors: CurrentActorService,
        admission: DriverAdmission,
        artifacts: CandidateArtifacts,
        gateway: NativeToolGateway,
        legacy: AssistantExecutionPort | None = None,
        specialist_registry: AgentRegistry | None = None,
        backend_factory: BackendFactory | None = None,
        activation: ActivationContext | None = None,
        lease_seconds: int = 60,
        reference_check: Callable[[AuthenticatedActor, ContextReference], Awaitable[bool]]
        | None = None,
        block_projector: ResponseBlockProjector | None = None,
        active_context_seed: ActiveContextSeed | None = None,
    ) -> None:
        self._sessions, self._transactions, self._actors = sessions, transactions, actors
        self._admission, self._artifacts, self._gateway = admission, artifacts, gateway
        self._legacy, self._specialists, self._backend_factory = (
            legacy,
            specialist_registry,
            backend_factory,
        )
        self._activation = activation
        self._lease_seconds = lease_seconds
        self._reference_check, self._block_projector = reference_check, block_projector
        self._active_context_seed = active_context_seed
        self._contexts = ContextReadService(
            actors=actors,
            projects=ProjectService(SqlAlchemyProjectTransactionFactory(sessions)),
            cursor_key=token_bytes(32),
        )

    async def execute_job(self, *, job: AssistantJob, actor: AuthenticatedActor) -> None:
        if (actor.organization_id, actor.membership_id) != (
            job.organization_id,
            job.requester_membership_id,
        ):
            raise SessionError("SESSION_NOT_FOUND")
        async with self._transactions(actor) as transaction:
            await transaction.repository.assert_job_claim(job=job)
            execution = await transaction.repository.get_job_run_and_turn(actor=actor, job=job)
        if execution is None:
            raise SessionError("SESSION_NOT_FOUND")
        run, turn = execution
        raw = run.execution_plan.get("driver_pin")
        if raw is None:
            if (
                self._legacy is None
                or job.job_type != "assistant.turn.execute"
                or run.orchestrator_version.startswith("2.")
            ):
                raise SessionError("DRIVER_PIN_REQUIRED")
            await self._legacy.execute_job(job=job, actor=actor)
            return
        if run.status.is_terminal:
            return
        pin = self._admission.resolve_saved(PinnedDriver.model_validate(raw))
        if (run.orchestrator_version, run.orchestrator_fingerprint) != (
            pin.agent_version,
            pin.manifest_fingerprint,
        ):
            raise SessionError("DRIVER_PIN_MISMATCH")
        await self._execute_candidate(job, actor, turn, pin)

    async def _execute_candidate(
        self,
        job: AssistantJob,
        actor: AuthenticatedActor,
        turn: AssistantTurn,
        pin: PinnedDriver,
    ) -> None:
        if job.locked_by is None:
            raise SessionError("LEASE_LOST")
        scope = RunScope(
            organization_id=job.organization_id,
            membership_id=actor.membership_id,
            conversation_id=job.conversation_id,
            turn_id=job.turn_id,
            orchestration_run_id=job.orchestration_run_id,
        )
        async with self._transactions(actor) as transaction:
            await transaction.repository.assert_job_claim(job=job)
            inbound = await PostgreSQLContinuations(transaction.session).load_user_choice(
                actor, job
            )
            wake = (
                await PostgreSQLContinuations(transaction.session).load_wake(actor, job)
                if job.job_type == "assistant.turn.resume"
                else None
            )
            await transaction.repository.begin_orchestration(job=job)
            await transaction.commit()
        recorder = PostgreSQLExecutionRecorder(
            transaction_factory=self._transactions,
            registry=self._specialists or self._artifacts.registry,
            job=job,
            actor=actor,
            activation=self._activation,
            candidate=self._artifacts.agent,
            block_projector=self._block_projector,
        )
        root_id = await recorder.ensure_orchestrator_run()
        observed = _CanonicalExploration(scope)
        objective = uuid5(NAMESPACE_URL, f"candidate-objective:{scope.turn_id}")
        projector = ExplorationProjector(
            scope=scope,
            objective_ref=objective,
            sources=observed,
            tool_pins=pin.tool_versions,
        )
        log = PostgreSQLToolLoopRecorder(
            service=SessionService(PostgreSQLSessionTransactionFactory(self._sessions)), actor=actor
        )
        capabilities = frozenset(p.reference for p in pin.tool_versions)
        access = CandidateAccess(
            facts=CatalogAccess(
                scope=scope,
                role=actor.role.value,
                active_phase=5,
                enabled=True,
                permission_revision=f"candidate-v2:{actor.role.value}",
                activation_revision=pin.manifest_fingerprint,
                allowed_tools=capabilities,
                active_tools=capabilities,
                allowed_skills=frozenset(p.reference for p in pin.skill_versions),
            ),
            actor=actor,
            job=job,
            transactions=self._transactions,
            contexts=self._contexts,
            model_ref=pin.model_ref,
            reference_check=self._reference_check,
        )
        catalog = ToolCatalog(
            agent=self._artifacts.agent,
            tools=self._artifacts.tools,
            access=access,
            readiness=access,
        )
        loader = SkillLoader(
            skills=self._artifacts.skills,
            resources=ApprovedPackageResources(
                {"explore_work_context@1": _SKILL_PACKAGE},
            ),
            catalog=catalog,
            model_ref=pin.model_ref,
        )
        contexts = ContextToolBackend(
            actor=actor,
            scope=scope,
            objective_ref=objective,
            service=self._contexts,
            tool_registry=self._artifacts.tools,
        )
        specialized = (
            await self._backend_factory(
                actor, scope, job, recorder, root_id, access, projector, turn.locale
            )
            if self._backend_factory
            else CandidateBackends({})
        )
        backends: dict[str, RecoverableToolBackend] = {
            "explore_projects@1": contexts,
            "read_project_context@1": contexts,
            "read_actor_context@1": contexts,
            "load_skill@1": SkillLoaderBackend(loader),
            **specialized.backends,
        }
        catalog.readiness = _AvailableBackends(access, frozenset(backends))
        lease, checkpoint = await self._prepare_history(
            scope, log, catalog, job, pin, turn.objective
        )
        if inbound is not None:
            imported = await self._import_choice(scope, log, lease, inbound, access, pin)
            if specialized.observe is not None:
                specialized.observe((imported,))
        elif self._active_context_seed is not None:
            seeded = await self._active_context_seed(actor, turn, scope)
            if seeded:
                imported_refs = await self._persist_active_refs(scope, log, lease, access, seeded)
                if specialized.observe is not None:
                    specialized.observe(imported_refs)
        if wake is not None and wake.wait.reference.scope != scope:
            await self._import_wake(scope, log, lease, wake)
        orchestrator = ToolLoopOrchestrator(
            gateway=BudgetedToolLoopGateway(
                MockModelGateway(fixtures={}),
                native=_AuthorizedGateway(self._gateway, access, scope),
            ),
            catalog=catalog,
            pipeline=ToolPipeline(
                store=log,
                policy=access,
                backends=backends,
                limits=self._artifacts.agent.manifest.runtime.limits,
            ),
            store=log,
            history=SessionHistory(log),
            access=access,
            budget=PostgreSQLRunBudget(self._sessions, actor),
            model_ref=pin.model_ref,
            skill_loader=loader,
            projector=projector,
            observation_sink=observed.observations,
            driver_version=pin.driver_version,
            delegate_specialists=specialized.specialists,
            evidence_sink=specialized.observe,
        )
        if wake is not None and wake.outcome == "REJECTED":
            text = "Đề xuất đã bị từ chối." if turn.locale == "vi" else "The proposal was rejected."
            result = PublishedResponse(
                output=OrchestratorOutputV2(
                    status=OrchestratorStatus.COMPLETED,
                    blocks=(TextResponseBlock(text=text),),
                    stop_reason="PROPOSAL_REJECTED",
                )
            )
        else:
            with retrieval_job_scope(job), report_chat_job_scope(job):
                async with asyncio.TaskGroup() as tasks:
                    heartbeat = tasks.create_task(self._renew(job, actor, lease))
                    try:
                        result = await orchestrator.run_segment(
                            OrchestratorInput(
                                orchestration_run_id=scope.orchestration_run_id,
                                conversation_id=scope.conversation_id,
                                turn_id=scope.turn_id,
                                message=turn.objective,
                                locale=turn.locale,
                                actor=ActorReference(
                                    organization_id=scope.organization_id,
                                    membership_id=scope.membership_id,
                                ),
                                active_context=ActiveConversationContext(recent_messages=()),
                            ),
                            scope=scope,
                            checkpoint=checkpoint,
                            lease=lease,
                            resume_wait=(
                                wake.wait.reference
                                if wake is not None
                                and wake.wait.reference.scope == scope
                                and wake.wait.reference.kind != "USER_INPUT"
                                else None
                            ),
                        )
                    finally:
                        heartbeat.cancel()
        if isinstance(result, PublishedResponse):
            blocks = await self._public_blocks(actor, result)
            await recorder.finish_orchestrator_run(root_id, result.output)
        else:
            blocks = ()
        async with self._transactions(actor) as transaction:
            await transaction.repository.assert_job_claim(job=job)
            if isinstance(result, PublishedResponse):
                await transaction.repository.append_assistant_blocks(
                    job=job,
                    blocks=blocks,
                    dedupe_key=f"candidate:{job.id}:result",
                )
            if result.wait_ref is not None:
                await PostgreSQLContinuations(transaction.session).open_wait(
                    actor,
                    WaitDraft(reference=result.wait_ref, lease=lease),
                )
            else:
                await transaction.repository.close_job_wait(job=job)
                await transaction.repository.finish_orchestration(
                    job=job,
                    status=result.output.status.value,
                    stop_reason=result.output.stop_reason,
                    safe_error_code=(
                        "ORCHESTRATOR_MANUAL_FALLBACK"
                        if result.output.status.value == "FAILED"
                        else None
                    ),
                )
            await transaction.commit()

    async def _renew(self, job: AssistantJob, actor: AuthenticatedActor, lease: RunLease) -> None:
        scope = session_scope(lease.scope)
        while True:
            await asyncio.sleep(max(1, self._lease_seconds / 3))
            async with self._transactions(actor) as transaction:
                journal = PostgreSQLSessionRepository(transaction.session)
                state = await SessionService.locked(journal, scope, session_lease(lease))
                await transaction.repository.assert_job_claim(job=job)
                expires = await journal.now() + timedelta(seconds=self._lease_seconds)
                await transaction.session.execute(
                    update(AssistantJobModel)
                    .where(
                        AssistantJobModel.organization_id == job.organization_id,
                        AssistantJobModel.id == job.id,
                    )
                    .values(lease_until=expires)
                )
                await journal.save_state(
                    replace(state, lease=replace(session_lease(lease), expires_at=expires))
                )
                await transaction.commit()

    async def _import_choice(
        self,
        scope: RunScope,
        log: PostgreSQLToolLoopRecorder,
        lease: RunLease,
        value: ResolvedContextBinding,
        access: CandidateAccess,
        pin: PinnedDriver,
    ) -> ContextReference:
        if (
            value.source_tool_pin not in pin.tool_versions
            or value.actor_revision != access.current(scope).permission_revision
            or value.policy_revision != access.current(scope).permission_revision
        ):
            raise SessionError("CONTINUATION_CONTEXT_REFRESH_REQUIRED")
        # Compare only stable authority fields. Observed time is tied to the
        # accepted original invocation, not to the model's new turn.
        records = await log.read(scope, after_sequence=0)
        ref = ContextReference(
            reference_id=value.selected_ref_id,
            organization_id=scope.organization_id,
            resource_type="project",
            resource_id=value.resource_id,
            version=value.resource_version,
            fingerprint=value.resource_fingerprint,
            observed_at=records[0].occurred_at,
        )
        await access.assert_refs(scope, (ref,))
        event_id = uuid5(value.source_invocation_id, f"context-choice:{scope.turn_id}")
        if not any(event.event_id == event_id for event in records):
            payload = await log.put_payload(
                scope,
                ContextPayload(
                    content={
                        "verified_project_reference": str(ref.reference_id),
                        "resource_version": ref.version,
                    },
                    references=(ref,),
                ).model_dump(mode="json"),
                classification="RAW_CONTEXT",
                kind="CONTEXT",
                lease=lease,
            )
            await log.append(
                scope,
                EventDraft(
                    event_id=event_id,
                    kind="context.selected",
                    payload_ref=payload,
                ),
                expected_sequence=records[-1].sequence,
                lease=lease,
            )
        return ref

    async def _persist_active_refs(
        self,
        scope: RunScope,
        log: PostgreSQLToolLoopRecorder,
        lease: RunLease,
        access: CandidateAccess,
        refs: tuple[ContextReference, ...],
    ) -> tuple[ContextReference, ...]:
        records = await log.read(scope, after_sequence=0)
        event_id = uuid5(scope.orchestration_run_id, "verified-active-context")
        previous = next((event for event in records if event.event_id == event_id), None)
        if previous is not None:
            if previous.payload_ref is None:
                raise SessionError("HISTORY_INVALID")
            payload = ContextPayload.model_validate(
                await log.get_payload(scope, previous.payload_ref)
            )
            values = payload.references
        else:
            values = tuple(
                ref.model_copy(update={"observed_at": records[0].occurred_at}) for ref in refs
            )
        await access.assert_refs(scope, values)
        if previous is None:
            payload_ref = await log.put_payload(
                scope,
                ContextPayload(
                    content={
                        "verified_context_reference_ids": [str(ref.reference_id) for ref in values]
                    },
                    references=values,
                ).model_dump(mode="json"),
                classification="RAW_CONTEXT",
                kind="CONTEXT",
                lease=lease,
            )
            await log.append(
                scope,
                EventDraft(event_id=event_id, kind="context.selected", payload_ref=payload_ref),
                expected_sequence=records[-1].sequence,
                lease=lease,
            )
        return values

    async def _import_wake(
        self,
        scope: RunScope,
        log: PostgreSQLToolLoopRecorder,
        lease: RunLease,
        wake: WakeReceipt,
    ) -> None:
        event_id = uuid5(wake.wait.reference.wait_id, f"import:{scope.orchestration_run_id}")
        records = await log.read(scope, after_sequence=0)
        if any(record.event_id == event_id for record in records):
            return
        payload = await log.put_payload(
            scope,
            MessagePayload(
                role="user",
                text=f"Verified backend continuation outcome: {wake.outcome}.",
            ).model_dump(mode="json"),
            classification="RAW_CONTEXT",
            kind="CONTEXT",
            lease=lease,
        )
        await log.append(
            scope,
            EventDraft(
                event_id=event_id,
                kind="continuation.received",
                payload_ref=payload,
                correlation_id=wake.wait.reference.wait_id,
            ),
            expected_sequence=records[-1].sequence,
            lease=lease,
        )

    async def _public_blocks(
        self,
        actor: AuthenticatedActor,
        result: PublishedResponse,
    ) -> tuple[dict[str, object], ...]:
        blocks: tuple[dict[str, object], ...] = tuple(
            block.model_dump(mode="json") for block in result.output.blocks
        )
        if self._block_projector is not None:
            return tuple(await self._block_projector.prepare_for_storage(actor, blocks))
        return blocks

    async def _prepare_history(
        self,
        scope: RunScope,
        log: PostgreSQLToolLoopRecorder,
        catalog: ToolCatalog,
        job: AssistantJob,
        pin: PinnedDriver,
        message: str,
    ) -> tuple[RunLease, ToolLoopCheckpoint]:
        state = await log.load_checkpoint(scope)
        if state is None:
            admitted = catalog.prepare(scope, model_ref=pin.model_ref)
            state = ToolLoopCheckpoint(
                scope=scope,
                driver_version=pin.driver_version,
                prompt_version=pin.prompt_version,
                catalog_fingerprint=admitted.fingerprint,
                agent_version=pin.agent_version,
                manifest_fingerprint=pin.manifest_fingerprint,
                model_ref=pin.model_ref,
                tool_versions=pin.tool_versions,
                limits=self._artifacts.agent.manifest.runtime.limits,
            )
            admitted.close()
            await log.initialize(scope, state)
        if job.locked_by is None:
            raise SessionError("LEASE_LOST")
        lease = await log.claim(
            scope, job_id=job.id, worker_id=job.locked_by, lease_seconds=self._lease_seconds
        )
        if state.objective_ref is None:
            records = await log.read(scope, after_sequence=0)
            if records:
                if (
                    len(records) != 1
                    or records[0].kind != "turn.started"
                    or records[0].payload_ref is None
                ):
                    raise SessionError("HISTORY_INVALID")
                event = records[0]
                payload = event.payload_ref
            else:
                payload = await log.put_payload(
                    scope,
                    MessagePayload(role="user", text=message).model_dump(mode="json"),
                    classification="RAW_CONTEXT",
                    kind="OBJECTIVE",
                    lease=lease,
                )
                event = await log.append(
                    scope,
                    EventDraft(
                        event_id=uuid5(scope.orchestration_run_id, f"turn.started:{scope.turn_id}"),
                        kind="turn.started",
                        payload_ref=payload,
                    ),
                    expected_sequence=0,
                    lease=lease,
                )
            state = state.model_copy(
                update={
                    "sequence": event.sequence,
                    "objective_ref": payload,
                    "payload_refs": (payload,),
                }
            )
            await log.checkpoint(scope, state, lease=lease)
        return lease, state


@dataclass(frozen=True, slots=True)
class _AvailableBackends:
    access: CandidateAccess
    references: frozenset[str]

    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        return reference in self.references and self.access.available(scope, reference, model_ref)
