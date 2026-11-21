"""Candidate read tools and live reference validation; no production activation."""

from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel

from app.modules.assistant.application.context_service import (
    ContextReadError,
    ContextReadService,
    ProjectReadSnapshot,
    fingerprint,
)
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.context_exploration import ContextAccessDecision
from work_management_ai.runtime.contracts import ContextReference
from work_management_ai.runtime.tool_loop_contracts import (
    BackendToolResult,
    RunScope,
    ToolInvocationContext,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry, ToolRegistryError
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextGap,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    ExploreProjectsInput,
    ExploreProjectsInputV103,
    ReadActorInput,
    ReadCoverage,
    ReadProjectInput,
    SourceContent,
    UseOutcome,
)

_CONTEXT_TOOL_RESOURCES = tuple(
    ("work_management_ai.tools.context", resource)
    for resource in ("projects.yaml", "project.yaml", "actor.yaml")
)


def _bounded_text(text: str, limit: int = 2048) -> tuple[str, bool]:
    data = text.encode()
    return data[:limit].decode("utf-8", errors="ignore"), len(data) > limit


class ContextRequirementsPort(Protocol):
    """Current deterministic capability prerequisites; supplied by application policy."""

    async def validate(
        self,
        *,
        actor: AuthenticatedActor,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision: ...


class ContextToolBackend:
    def __init__(
        self,
        *,
        actor: AuthenticatedActor,
        scope: RunScope,
        objective_ref: UUID,
        service: ContextReadService,
        tool_registry: ToolRegistry | None = None,
    ):
        if (actor.organization_id, actor.membership_id) != (
            scope.organization_id,
            scope.membership_id,
        ):
            raise ValueError("CONTEXT_SCOPE_MISMATCH")
        self.actor, self.scope, self.objective_ref, self.service = (
            actor,
            scope,
            objective_ref,
            service,
        )
        self._tools = (
            ToolRegistry.from_resources(_CONTEXT_TOOL_RESOURCES)
            if tool_registry is None
            else tool_registry
        )

    def _check(self, context: ToolInvocationContext) -> None:
        if context.scope != self.scope or context.lease.scope != self.scope:
            raise ToolPolicyError("CONTEXT_SCOPE_MISMATCH", stop_segment=True)
        try:
            version = self._tools.resolve(context.tool_reference).manifest.version
        except ToolRegistryError:
            raise ToolPolicyError("CONTEXT_TOOL_MISMATCH", stop_segment=True) from None
        if context.tool_version != version:
            raise ToolPolicyError("CONTEXT_TOOL_VERSION_CONFLICT", stop_segment=True)

    def _ref(
        self,
        context: ToolInvocationContext,
        *,
        resource_id: UUID,
        resource_type: str,
        version: int | None,
        digest: str,
        observed_at: object,
    ) -> ContextRef:
        return ContextRef.model_validate(
            {
                "ref_id": uuid5(context.invocation_id, f"{resource_type}:{resource_id}:{digest}"),
                "scope": context.scope,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "version": version,
                "fingerprint": digest,
                "observed_at": observed_at,
                "invocation_id": context.invocation_id,
                "source": "project_service"
                if resource_type == "project"
                else "current_actor_service",
            }
        )

    def _project(
        self,
        context: ToolInvocationContext,
        snap: ProjectReadSnapshot,
        *,
        include_overview_if_unique: bool = False,
    ) -> ContextReadObservation:
        refs = tuple(
            self._ref(
                context,
                resource_id=p.id,
                resource_type="project",
                version=p.version,
                digest=p.fingerprint,
                observed_at=snap.observed_at,
            )
            for p in snap.projects
        )
        facts = tuple(
            f
            for p, r in zip(snap.projects, refs, strict=True)
            for f in (
                ContextFact(ref_id=r.ref_id, field="project.name", value=p.name),
                ContextFact(ref_id=r.ref_id, field="project.version", value=p.version),
            )
        )
        discovery_overview = (
            include_overview_if_unique
            and snap.resolution == "UNIQUE"
            and snap.outcome == "OK"
            and snap.cursor is None
            and not snap.gaps
            and len(refs) == 1
        )
        content: tuple[SourceContent, ...] = ()
        overview_truncated = False
        if (
            discovery_overview or context.tool_reference == "read_project_context@1"
        ) and snap.projects:
            description = snap.projects[0].description
            if description:
                text, overview_truncated = _bounded_text(description)
                # Discovery includes only complete overviews. Direct reads can return an excerpt.
                if not (discovery_overview and overview_truncated):
                    content = (
                        SourceContent(
                            ref_id=refs[0].ref_id, text=text, truncated=overview_truncated
                        ),
                    )
        truncated = bool(snap.cursor) or overview_truncated
        gaps = tuple(ContextGap(code="NOT_SUPPORTED", section=g) for g in snap.gaps)
        if snap.cursor:
            gaps = (*gaps, ContextGap(code="PAGINATION"))
        if overview_truncated:
            gaps = (*gaps, ContextGap(code="TRUNCATED", section="overview"))
        entities = (
            EntityResolution.model_validate(
                {
                    "query_fingerprint": snap.query_fingerprint,
                    "resolution": snap.resolution,
                    "candidate_ref_ids": tuple(r.ref_id for r in refs),
                    "selected_ref_id": next(
                        (r.ref_id for r in refs if r.resource_id == snap.selected_id), None
                    ),
                }
            ),
        )
        return ContextReadObservation.model_validate(
            {
                "scope": context.scope,
                "objective_ref": self.objective_ref,
                "invocation_id": context.invocation_id,
                "tool_reference": context.tool_reference,
                "tool_version": context.tool_version,
                "observed_at": snap.observed_at,
                "outcome": "PARTIAL" if truncated else snap.outcome,
                "context_refs": refs,
                "entities": entities,
                "facts": facts,
                "source_content_refs": content,
                "gaps": gaps,
                "coverage": (
                    ReadCoverage(
                        requested=(*snap.requested, "overview")
                        if discovery_overview
                        else snap.requested,
                        returned=(*snap.returned, "overview")
                        if discovery_overview and not overview_truncated
                        else snap.returned,
                        status="PARTIAL"
                        if truncated or snap.gaps or not snap.returned
                        else "COMPLETE",
                        query_fingerprint=snap.query_fingerprint,
                        cursor=snap.cursor,
                        truncated=truncated,
                    ),
                ),
            }
        )

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        self._check(context)
        try:
            if context.tool_reference == "explore_projects@1" and isinstance(
                arguments, ExploreProjectsInput
            ):
                snap = await self.service.search_projects(
                    actor=self.actor,
                    query=arguments.query,
                    cursor=arguments.cursor,
                    page_size=arguments.page_size,
                )
                observation = self._project(
                    context,
                    snap,
                    include_overview_if_unique=isinstance(arguments, ExploreProjectsInputV103)
                    and arguments.include_overview_if_unique,
                )
            elif context.tool_reference == "read_project_context@1" and isinstance(
                arguments, ReadProjectInput
            ):
                snap = await self.service.read_project(
                    actor=self.actor, reference=arguments.reference, sections=arguments.sections
                )
                observation = self._project(context, snap)
            elif context.tool_reference == "read_actor_context@1" and isinstance(
                arguments, ReadActorInput
            ):
                actor = await self.service.read_actor(actor=self.actor, sections=arguments.sections)
                refs = (
                    (
                        self._ref(
                            context,
                            resource_id=actor.membership_id,
                            resource_type="actor",
                            version=None,
                            digest=actor.fingerprint,
                            observed_at=actor.observed_at,
                        ),
                    )
                    if actor.returned
                    else ()
                )
                facts = (
                    ()
                    if not refs
                    else (
                        ContextFact(
                            ref_id=refs[0].ref_id,
                            field="actor.display_name",
                            value=actor.display_name or "",
                        ),
                        ContextFact(
                            ref_id=refs[0].ref_id,
                            field="actor.organization_name",
                            value=actor.organization_name or "",
                        ),
                    )
                )
                observation = ContextReadObservation(
                    scope=context.scope,
                    objective_ref=self.objective_ref,
                    invocation_id=context.invocation_id,
                    tool_reference="read_actor_context@1",
                    tool_version=context.tool_version,
                    observed_at=actor.observed_at,
                    outcome="PARTIAL" if actor.gaps else "OK",
                    context_refs=refs,
                    facts=facts,
                    coverage=(
                        ReadCoverage(
                            requested=actor.requested,
                            returned=actor.returned,
                            status="PARTIAL" if actor.gaps else "COMPLETE",
                            query_fingerprint=fingerprint(actor.requested),
                        ),
                    ),
                    gaps=tuple(ContextGap(code="NOT_SUPPORTED", section=g) for g in actor.gaps),
                )
            else:
                raise ToolPolicyError("CONTEXT_TOOL_MISMATCH", stop_segment=True)
        except ContextReadError as error:
            raise ToolPolicyError(str(error), stop_segment=True) from None
        # Bound the wire result, atomically dropping a ref with its facts; disclose the gap.
        while len(observation.model_dump_json().encode()) > 15000 and observation.context_refs:
            drop = observation.context_refs[-1].ref_id
            observation = observation.model_copy(
                update={
                    "context_refs": observation.context_refs[:-1],
                    "facts": tuple(f for f in observation.facts if f.ref_id != drop),
                    "source_content_refs": tuple(
                        s for s in observation.source_content_refs if s.ref_id != drop
                    ),
                    "entities": tuple(
                        e.model_copy(
                            update={
                                "candidate_ref_ids": tuple(
                                    i for i in e.candidate_ref_ids if i != drop
                                ),
                                "selected_ref_id": None
                                if e.selected_ref_id == drop
                                else e.selected_ref_id,
                            }
                        )
                        for e in observation.entities
                    ),
                    "outcome": "PARTIAL",
                    "coverage": tuple(
                        c.model_copy(update={"status": "PARTIAL", "truncated": True})
                        for c in observation.coverage
                    ),
                    "gaps": tuple(dict.fromkeys((*observation.gaps, ContextGap(code="TRUNCATED")))),
                }
            )
        return BackendToolResult(
            output=observation,
            evidence=tuple(
                ContextReference(
                    reference_id=r.ref_id,
                    organization_id=r.scope.organization_id,
                    resource_type=r.resource_type,
                    resource_id=r.resource_id,
                    version=r.version,
                    fingerprint=r.fingerprint,
                    observed_at=r.observed_at,
                )
                for r in observation.context_refs
            ),
        )

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        # These tools have no side effect; reconstruct a live read after uncertain completion.
        return await self.execute(context, arguments)

    async def revalidate_result(
        self, context: ToolInvocationContext, result: BackendToolResult
    ) -> None:
        self._check(context)
        o = result.output
        if not isinstance(o, ContextReadObservation) or (
            o.scope,
            o.objective_ref,
            o.invocation_id,
            o.tool_reference,
            o.tool_version,
        ) != (
            self.scope,
            self.objective_ref,
            context.invocation_id,
            context.tool_reference,
            context.tool_version,
        ):
            raise ToolPolicyError("CONTEXT_SCOPE_MISMATCH", stop_segment=True)
        fresh = await self.execute(context, context.canonical_arguments)
        if not isinstance(fresh.output, ContextReadObservation):
            raise ToolPolicyError("CONTEXT_UNAVAILABLE", stop_segment=True)

        def semantic(value: ContextReadObservation) -> dict[str, object]:
            result = value.model_dump(exclude={"observed_at"})
            result["context_refs"] = tuple(
                r.model_dump(exclude={"observed_at"}) for r in value.context_refs
            )
            return result

        if semantic(fresh.output) != semantic(o):
            raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)


async def revalidate_refs(
    *,
    service: ContextReadService,
    actor: AuthenticatedActor,
    scope: RunScope,
    refs: tuple[ContextRef, ...],
) -> None:
    current = await service.current_actor(actor=actor)
    for ref in refs:
        if ref.scope != scope:
            raise ToolPolicyError("CONTEXT_SCOPE_MISMATCH", stop_segment=True)
        if ref.resource_type == "project":
            snap = await service.read_project(
                actor=current, reference=ref.resource_id, sections=("overview",)
            )
            if not snap.projects:
                raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True)
            if (snap.projects[0].version, snap.projects[0].fingerprint) != (
                ref.version,
                ref.fingerprint,
            ):
                raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)
        else:
            if ref.resource_id != current.membership_id:
                raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True)
            snap_actor = await service.read_actor(actor=current, sections=("identity",))
            if snap_actor.fingerprint != ref.fingerprint:
                raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)


class ContextUseAccess:
    def __init__(self, *, backend: ContextToolBackend, requirements: ContextRequirementsPort):
        self.backend, self.requirements = backend, requirements

    async def check(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision:
        outcome: UseOutcome = "DENIED"
        if scope == self.backend.scope:
            try:
                await revalidate_refs(
                    service=self.backend.service, actor=self.backend.actor, scope=scope, refs=refs
                )
                actor = await self.backend.service.current_actor(actor=self.backend.actor)
                return await self.requirements.validate(
                    actor=actor,
                    target_tool=target_tool,
                    input_fingerprint=input_fingerprint,
                    refs=refs,
                )
            except ToolPolicyError as error:
                if str(error) == "CONTEXT_STALE":
                    outcome = "REFRESH_REQUIRED"
            except ContextReadError:
                pass
        return ContextAccessDecision(
            outcome=outcome, actor_revision="unavailable", policy_revision="unavailable"
        )
