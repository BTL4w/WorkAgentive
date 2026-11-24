"""Live authorization and provenance for the candidate tool loop."""

from collections.abc import Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel

from app.modules.assistant.application.context_service import ContextReadError, ContextReadService
from app.modules.assistant.application.ports import AssistantTransactionFactory
from app.modules.assistant.domain.models import AssistantJob
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.contracts import ContextReference
from work_management_ai.runtime.session_history import HistoryReference, RevalidatedContext
from work_management_ai.runtime.tool_catalog import CatalogAccess
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    BackendToolResult,
    PayloadReference,
    RunScope,
    ToolInvocationContext,
)
from work_management_ai.runtime.tool_pipeline import PreparedTool, ToolPolicyError


class CandidateAccess:
    def __init__(
        self,
        *,
        facts: CatalogAccess,
        actor: AuthenticatedActor,
        job: AssistantJob,
        transactions: AssistantTransactionFactory,
        contexts: ContextReadService,
        model_ref: str,
        reference_check: Callable[[AuthenticatedActor, ContextReference], Awaitable[bool]]
        | None = None,
    ) -> None:
        self._facts, self.actor, self.job = facts, actor, job
        self._transactions, self.contexts, self._model_ref = transactions, contexts, model_ref
        self._reference_check = reference_check

    def current(self, scope: RunScope) -> CatalogAccess:
        if scope != self._facts.scope:
            raise ToolPolicyError("CONTEXT_SCOPE_MISMATCH", stop_segment=True)
        return self._facts

    def available(self, scope: RunScope, reference: str, model_ref: str) -> bool:
        return (
            scope == self._facts.scope
            and model_ref == self._model_ref
            and reference in self._facts.active_tools
        )

    async def ensure_authorized(self, scope: RunScope) -> None:
        self.current(scope)
        try:
            current = await self.contexts.current_actor(actor=self.actor)
        except ContextReadError:
            raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True) from None
        if current.role != self.actor.role:
            raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True)
        async with self._transactions(current) as transaction:
            await transaction.repository.assert_job_claim(job=self.job)
            await transaction.commit()

    async def assert_refs(self, scope: RunScope, refs: tuple[ContextReference, ...]) -> None:
        await self.ensure_authorized(scope)
        for ref in refs:
            if ref.organization_id != scope.organization_id:
                raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True)
            if ref.resource_type == "project":
                project = await self.contexts.read_project(
                    actor=self.actor, reference=ref.resource_id, sections=("overview",)
                )
                if not project.projects or project.projects[0].version != ref.version:
                    raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)
                if ref.fingerprint and project.projects[0].fingerprint != ref.fingerprint:
                    raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)
            elif ref.resource_type.casefold() == "actor":
                actor = await self.contexts.read_actor(actor=self.actor, sections=("identity",))
                if (
                    ref.resource_id != self.actor.membership_id
                    or actor.fingerprint != ref.fingerprint
                ):
                    raise ToolPolicyError("CONTEXT_STALE", stop_segment=True)
            elif self._reference_check is None or not await self._reference_check(self.actor, ref):
                raise ToolPolicyError("CONTEXT_DENIED", stop_segment=True)

    async def revalidate(
        self, scope: RunScope, refs: tuple[HistoryReference, ...]
    ) -> RevalidatedContext:
        if any(
            isinstance(ref, PayloadReference) and ref.organization_id != scope.organization_id
            for ref in refs
        ):
            return RevalidatedContext(allowed=False, references=())
        try:
            await self.assert_refs(
                scope, tuple(ref for ref in refs if isinstance(ref, ContextReference))
            )
        except ToolPolicyError as error:
            return RevalidatedContext(
                allowed=error.code == "CONTEXT_STALE",
                references=refs,
                fresh=False,
            )
        except ContextReadError:
            return RevalidatedContext(allowed=False, references=())
        return RevalidatedContext(allowed=True, references=refs)

    async def pre_execute(
        self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
    ) -> PreparedTool:
        await self.ensure_authorized(scope)
        if tool.reference not in self._facts.allowed_tools:
            raise ToolPolicyError("TOOL_NOT_ALLOWED", stop_segment=True)
        return PreparedTool(arguments=arguments, policy_version=self._facts.permission_revision)

    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None:
        if (
            context.policy_version != self._facts.permission_revision
            or tool.reference not in self._facts.allowed_tools
        ):
            raise ToolPolicyError("TOOL_NOT_ALLOWED", stop_segment=True)
        await self.assert_refs(context.scope, evidence)

    async def post_execute(self, context: ToolInvocationContext, result: BackendToolResult) -> None:
        if result.wait_ref is not None and result.wait_ref.scope != context.scope:
            raise ToolPolicyError("CONTEXT_SCOPE_MISMATCH", stop_segment=True)

    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None:
        if scope != self._facts.scope:
            return
        async with self._transactions(self.actor) as transaction:
            await transaction.repository.append_rejected_audit(
                actor=self.actor,
                action="assistant.candidate_tool",
                resource_type="tool_invocation",
                resource_id=invocation_id,
                request_id=str(invocation_id),
                reason_code=safe_code,
            )
            await transaction.commit()
