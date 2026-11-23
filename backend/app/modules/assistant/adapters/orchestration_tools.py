"""Candidate tool bridge to application handlers; no production registration.

Handlers own transactional effect receipts and must use the effect fence in
the transaction that creates them. Reconciliation only reads those receipts.
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from app.core.execution_fence import ExecutionFenceContext, execution_fence
from app.modules.assistant.adapters.tool_loop_recorder import session_lease, session_scope
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.run_budget import current_candidate_budget
from work_management_ai.runtime.tool_loop_contracts import BackendToolResult, ToolInvocationContext

type ExecuteHandler = Callable[[ToolInvocationContext, BaseModel], Awaitable[BackendToolResult]]
type ReconcileHandler = Callable[
    [ToolInvocationContext, BaseModel], Awaitable[BackendToolResult | None]
]


class ApplicationToolBackend:
    def __init__(
        self, *, actor: AuthenticatedActor, execute: ExecuteHandler, reconcile: ReconcileHandler
    ):
        self.actor, self._execute, self._reconcile = actor, execute, reconcile

    def _context(self, context: ToolInvocationContext) -> ExecutionFenceContext:
        binding = current_candidate_budget()
        if binding is None or binding.scope != context.scope or binding.lease != context.lease:
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        if (context.scope.organization_id, context.scope.membership_id) != (
            self.actor.organization_id,
            self.actor.membership_id,
        ):
            raise SessionError("SESSION_NOT_FOUND")
        return ExecutionFenceContext(
            actor=self.actor,
            scope=session_scope(context.scope),
            lease=session_lease(context.lease),
            operation_id=context.receipt_id,
            interval_id=binding.interval_id,
            allow_job_enqueue=context.tool_reference == "delegate_agent@2",
        )

    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult:
        with execution_fence(self._context(context)):
            return await self._execute(context, arguments)

    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        self._context(context)
        return await self._reconcile(context, arguments)
