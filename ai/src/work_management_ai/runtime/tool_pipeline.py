"""One guarded invocation, durable dispatch claim, and receipt-based recovery.

Policies/handlers are trusted application registrations, never model selections.
A started call is never dispatched again: missing effect receipts require inspection.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, ValidationError

from work_management_ai.model_gateway.tool_contracts import ToolCall
from work_management_ai.runtime.contracts import ContextReference, JsonValue
from work_management_ai.runtime.run_budget import (
    BudgetError,
    BudgetReceipt,
    UsageReservation,
    current_candidate_budget,
)
from work_management_ai.runtime.session_history import ToolArgumentsPayload, ToolResultPayload
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationDescriptor,
    InvocationReceipt,
    InvocationStartedError,
    SessionLogPort,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    AdmittedToolSet,
    BackendToolResult,
    CompletedToolResult,
    DeferredToolResult,
    RejectedToolResult,
    RunLease,
    RunScope,
    ToolInvocationContext,
    ToolLoopLimits,
    ToolOutcome,
    WaitReference,
)
from work_management_ai.tools.skills.contracts import LoadedSkill


class ToolPolicyError(ValueError):
    """Only registered safe codes may cross this boundary; never exception text."""

    def __init__(self, code: str, *, stop_segment: bool = False):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", code):
            raise ValueError("INVALID_TOOL_ERROR_CODE")
        self.code, self.stop_segment = code, stop_segment
        super().__init__(code)


class PreparedTool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    arguments: SerializeAsAny[BaseModel]
    policy_version: str = Field(min_length=1, max_length=128)
    resource_versions: tuple[ContextReference, ...] = ()
    decision: Literal["ALLOW", "NEEDS_INPUT"] = "ALLOW"


class ToolPolicyPort(Protocol):
    """Mandatory application checks, including current role/risk/approval/version.

    pre_execute normalizes only. guard rechecks current authorization and evidence,
    including cached results. post_execute verifies without rewriting committed facts.
    rejected persists sensitive denial audit using safe identifiers only.
    """

    async def pre_execute(
        self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
    ) -> PreparedTool: ...
    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None: ...
    async def post_execute(
        self, context: ToolInvocationContext, result: BackendToolResult
    ) -> None: ...
    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None: ...


class RecoverableToolBackend(Protocol):
    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult: ...
    async def reconcile(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult | None:
        """Read a canonical service receipt by identity; None means unknown, not retry."""
        ...


@runtime_checkable
class ToolResultAccessPort(Protocol):
    async def revalidate_result(
        self, context: ToolInvocationContext, result: BackendToolResult
    ) -> None:
        """Check resource-specific access/pins for fresh and cached observations."""
        ...


@runtime_checkable
class ToolWaitAccessPort(Protocol):
    async def revalidate_wait(self, context: ToolInvocationContext, wait: WaitReference) -> None:
        """Verify exact resource access for replay of an already settled wait."""
        ...


# Authority is supplied by the runtime and application, never as model arguments.
_RESERVED = frozenset(
    {
        "organization_id",
        "tenant_id",
        "membership_id",
        "actor",
        "actor_id",
        "actor_membership_id",
        "role",
        "permissions",
        "approved",
        "approval_state",
        "approval_granted",
        "policy_version",
        "lease",
        "fencing_token",
        "receipt_id",
        "budget_allocation_id",
        "source_turn_ref",
        "system_prompt",
        "chain_of_thought",
        "hidden_reasoning",
        "raw_provider_response",
    }
)
_SECURITY = frozenset(
    {
        "PERMISSION_DENIED",
        "TENANT_MISMATCH",
        "APPROVAL_REQUIRED",
        "TOOL_ARGUMENTS_CHANGED",
        "INVOCATION_CONFLICT",
    }
)


def _encoded(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _visible(value: JsonValue, *, arguments: bool) -> None:
    if isinstance(value, dict):
        forbidden = (
            _RESERVED
            if arguments
            else _RESERVED
            & {"system_prompt", "chain_of_thought", "hidden_reasoning", "raw_provider_response"}
        )
        if forbidden.intersection(value):
            raise ToolPolicyError(
                "TOOL_AUTHORITY_FIELDS" if arguments else "TOOL_OUTPUT_INVALID", stop_segment=True
            )
        for item in value.values():
            _visible(item, arguments=arguments)
    elif isinstance(value, list):
        for item in value:
            _visible(item, arguments=arguments)


class ToolPipeline:
    def __init__(
        self,
        *,
        store: SessionLogPort,
        policy: ToolPolicyPort,
        backends: Mapping[str, RecoverableToolBackend],
        limits: ToolLoopLimits | None = None,
    ):
        self.store, self.policy, self.backends = store, policy, dict(backends)
        self.limits = limits or ToolLoopLimits()

    @staticmethod
    def invocation_id(scope: RunScope, call_id: str) -> UUID:
        return uuid5(
            NAMESPACE_URL, f"tool:{scope.organization_id}:{scope.orchestration_run_id}:{call_id}"
        )

    async def _event(
        self,
        scope: RunScope,
        lease: RunLease,
        receipt: InvocationReceipt,
        kind: str,
        safe_code: str | None = None,
    ) -> None:
        records = await self.store.read(scope, after_sequence=0)
        event = EventDraft.model_validate(
            dict(
                event_id=uuid5(receipt.descriptor.invocation_id, f"{kind}:{safe_code}"),
                safe_code=safe_code,
                kind=kind,
                invocation=receipt.descriptor,
                causation_id=receipt.requested_event_id,
            )
        )
        await self.store.append(
            scope, event, expected_sequence=records[-1].sequence if records else 0, lease=lease
        )

    async def _deny(
        self,
        scope: RunScope,
        call: ToolCall,
        invocation_id: UUID,
        reference: str | None,
        error: ToolPolicyError,
    ) -> RejectedToolResult:
        code, stop = error.code, error.stop_segment or error.code in _SECURITY
        try:
            await self.policy.rejected(scope, invocation_id, reference, code)
        except Exception:
            code, stop = "TOOL_AUDIT_UNAVAILABLE", True
        return RejectedToolResult(
            invocation_id=invocation_id, call_id=call.call_id, safe_code=code, stop_segment=stop
        )

    @staticmethod
    def _unchanged(context: ToolInvocationContext, arguments: BaseModel, original: str) -> None:
        if (
            _encoded(context.canonical_arguments.model_dump(mode="json")) != original
            or _encoded(arguments.model_dump(mode="json")) != original
        ):
            raise ToolPolicyError("TOOL_ARGUMENTS_CHANGED", stop_segment=True)

    async def execute(
        self,
        scope: RunScope,
        call: ToolCall,
        *,
        admitted: AdmittedToolSet,
        lease: RunLease,
        resume_wait: WaitReference | None = None,
    ) -> ToolOutcome:
        invocation_id = self.invocation_id(scope, call.call_id)
        tool: AdmittedTool | None = None
        receipt: InvocationReceipt | None = None
        started = False
        budget: BudgetReceipt | None = None
        binding = current_candidate_budget()
        try:
            if (
                binding is None
                or binding.scope != scope
                or binding.lease != lease
                or lease.scope != scope
            ):
                raise ToolPolicyError("TOOL_RUNTIME_SCOPE_INVALID", stop_segment=True)
            call = ToolCall.model_validate(call)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", call.call_id):
                raise ToolPolicyError("TOOL_CALL_ID_INVALID")
            try:
                tool = admitted.resolve(call.name)
            except ToolPolicyError:
                raise
            except (KeyError, ValueError):
                raise ToolPolicyError("UNKNOWN_TOOL") from None
            if tool.schema.wire_name != call.name:
                raise ToolPolicyError("UNKNOWN_TOOL")
            backend = self.backends.get(tool.reference)
            if backend is None:
                raise ToolPolicyError("TOOL_UNAVAILABLE")
            _visible(call.arguments, arguments=True)
            # JSON mode accepts UUID/date strings while retaining strict numeric validation.
            arguments = tool.input_contract.model_validate_json(
                _encoded(call.arguments), strict=True
            )
            prepared = PreparedTool.model_validate(
                await self.policy.pre_execute(scope, tool, arguments)
            )
            if any(
                ref.organization_id != scope.organization_id for ref in prepared.resource_versions
            ):
                raise ToolPolicyError("TENANT_MISMATCH", stop_segment=True)
            arguments = tool.input_contract.model_validate_json(
                prepared.arguments.model_dump_json(), strict=True
            )
            raw = arguments.model_dump(mode="json")
            _visible(raw, arguments=True)
            original = _encoded(raw)
            body = ToolArgumentsPayload(
                arguments=raw,
                references=prepared.resource_versions,
                policy_version=prepared.policy_version,
                tool_fingerprint=tool.fingerprint,
            )
            descriptor = InvocationDescriptor(
                invocation_id=invocation_id,
                call_id=call.call_id,
                tool_reference=tool.reference,
                tool_version=tool.version,
                arguments_fingerprint=_fingerprint(body.model_dump(mode="json")),
            )
            receipt = await self.store.get_invocation(scope, invocation_id)
            if receipt is not None and receipt.descriptor != descriptor:
                raise ToolPolicyError("INVOCATION_CONFLICT", stop_segment=True)
            resume = resume_wait is not None
            if resume and (
                receipt is None
                or receipt.settlement is None
                or receipt.settlement.wait_ref != resume_wait
                or resume_wait.scope != scope
                or resume_wait.source_invocation_id != invocation_id
            ):
                raise ToolPolicyError("CONTINUATION_WAIT_MISMATCH", stop_segment=True)
            reservation = UsageReservation(
                operation_id=invocation_id,
                kind="HANDOFF"
                if tool.reference in {"delegate_agent@1", "delegate_agent@2"}
                else "TOOL",
                interval_id=binding.interval_id,
                allocation_id=binding.allocation_id,
            )
            budget = (
                await binding.store.reserve(scope, reservation, lease=lease)
                if receipt is None
                else await binding.store.recover_tool(
                    scope,
                    invocation_id,
                    lease=lease,
                    interval_id=binding.interval_id,
                    allocation_id=binding.allocation_id,
                )
            )
            if receipt is None:
                ref = await self.store.put_payload(
                    scope,
                    body.model_dump(mode="json"),
                    classification="RAW_CONTEXT",
                    kind="TOOL_ARGUMENTS",
                    lease=lease,
                )
                records = await self.store.read(scope, after_sequence=0)
                await self.store.append(
                    scope,
                    EventDraft(
                        event_id=uuid5(invocation_id, "requested"),
                        kind="tool.requested",
                        invocation=descriptor,
                        payload_ref=ref,
                    ),
                    expected_sequence=records[-1].sequence if records else 0,
                    lease=lease,
                )
                receipt = await self.store.get_invocation(scope, invocation_id)
                assert receipt is not None
            else:
                # Expired original arguments cannot be recreated from a loaded copy.
                await self.store.get_payload(scope, receipt.arguments_ref)
            context = ToolInvocationContext(
                scope=scope,
                lease=lease,
                invocation_id=invocation_id,
                call_id=call.call_id,
                tool_reference=tool.reference,
                tool_version=tool.version,
                arguments_fingerprint=descriptor.arguments_fingerprint,
                canonical_arguments=arguments.model_copy(deep=True),
                policy_version=prepared.policy_version,
                resource_versions=prepared.resource_versions,
                receipt_id=invocation_id,
                budget_allocation_id=binding.allocation_id or binding.interval_id,
                limits=self.limits,
            )
            await self.policy.guard(context, tool, ())
            self._unchanged(context, arguments, original)
            self._admission(admitted, call.name, tool)
            if receipt.settlement is not None and not resume:
                outcome = await self._replay(context, tool, receipt, backend)
                self._unchanged(context, arguments, original)
                self._admission(admitted, call.name, tool)
                await binding.store.settle(scope, budget, None)
                return outcome
            if not resume:
                await self._event(scope, lease, receipt, "tool.authorized")
            if resume and prepared.decision == "NEEDS_INPUT":
                raise ToolPolicyError("CONTINUATION_INPUT_REQUIRED", stop_segment=True)
            if prepared.decision == "NEEDS_INPUT" and not resume:
                wait = WaitReference(
                    wait_id=uuid5(invocation_id, "needs-input"),
                    scope=scope,
                    kind="USER_INPUT",
                    source_invocation_id=invocation_id,
                )
                await self.store.settle_invocation(
                    scope,
                    receipt,
                    SettledInvocation(status="DEFERRED", wait_ref=wait),
                    lease=lease,
                )
                await binding.store.settle(scope, budget, None)
                return DeferredToolResult(
                    invocation_id=invocation_id, call_id=call.call_id, wait_ref=wait
                )
            fresh = not resume and await self.store.start_invocation(scope, receipt, lease=lease)
            started = True
            seconds = min(tool.timeout_seconds, budget.seconds_remaining, 180)
            if seconds <= 0:
                raise ToolPolicyError("TOOL_TIMEOUT", stop_segment=True)
            async with asyncio.timeout(seconds):
                if fresh:
                    result = await backend.execute(context, arguments.model_copy(deep=True))
                else:
                    result = await backend.reconcile(context, arguments.model_copy(deep=True))
                    if result is None:
                        raise ToolPolicyError("TOOL_RECONCILIATION_REQUIRED", stop_segment=True)
                    if resume and result.wait_ref == resume_wait:
                        raise ToolPolicyError("CONTINUATION_RESULT_UNAVAILABLE", stop_segment=True)
            self._unchanged(context, arguments, original)
            result = self._validate(context, tool, result)
            try:
                await self.policy.post_execute(context, result.model_copy(deep=True))
            except ToolPolicyError:
                raise
            except Exception:
                raise ToolPolicyError("TOOL_POSTCONDITION_FAILED", stop_segment=True) from None
            self._unchanged(context, arguments, original)
            await self.policy.guard(context, tool, result.evidence)
            await self._result_access(context, tool, backend, result)
            self._unchanged(context, arguments, original)
            self._admission(admitted, call.name, tool)
            if result.wait_ref is not None:
                settlement = SettledInvocation(status="DEFERRED", wait_ref=result.wait_ref)
                outcome: ToolOutcome = DeferredToolResult(
                    invocation_id=invocation_id, call_id=call.call_id, wait_ref=result.wait_ref
                )
            else:
                assert result.output is not None
                value = ToolResultPayload(
                    output=result.output.model_dump(mode="json"),
                    references=result.evidence,
                    loaded_skill=result.loaded_skill,
                )
                ref = await self.store.put_payload(
                    scope,
                    value.model_dump(mode="json"),
                    classification="RAW_CONTEXT",
                    kind="TOOL_RESULT",
                    lease=lease,
                )
                settlement = SettledInvocation(status="COMPLETED", payload_ref=ref)
                outcome = CompletedToolResult(
                    invocation_id=invocation_id,
                    call_id=call.call_id,
                    payload_ref=ref,
                    evidence=result.evidence,
                )
            await self.store.settle_invocation(scope, receipt, settlement, lease=lease)
            await binding.store.settle(scope, budget, None)
            return outcome
        except asyncio.CancelledError:
            # Caller cancellation propagates. A started receipt remains pending for reconciliation.
            raise
        except ValidationError:
            error = ToolPolicyError(
                "TOOL_INPUT_INVALID" if not started else "TOOL_OUTPUT_INVALID", stop_segment=started
            )
        except ToolPolicyError as exc:
            error = exc
        except (BudgetError, TimeoutError):
            error = ToolPolicyError(
                "TOOL_TIMEOUT" if started else "TOOL_BUDGET_DENIED", stop_segment=True
            )
        except Exception:
            error = ToolPolicyError(
                "TOOL_RECONCILIATION_REQUIRED" if started else "TOOL_RUNTIME_UNAVAILABLE",
                stop_segment=True,
            )
        # Only pre-dispatch failures are terminal. Started failures preserve service receipt truth.
        rejected_terminal = False
        if receipt is not None and receipt.settlement is None and not started:
            try:
                await self.store.settle_invocation(
                    scope,
                    receipt,
                    SettledInvocation(status="REJECTED", safe_code=error.code),
                    lease=lease,
                )
                rejected_terminal = True
            except InvocationStartedError:
                started = True
            except Exception:
                error = ToolPolicyError("TOOL_RUNTIME_UNAVAILABLE", stop_segment=True)
        if receipt is not None and receipt.settlement is None and started:
            error = ToolPolicyError(error.code, stop_segment=True)
            try:
                await self._event(scope, lease, receipt, "tool.denied", error.code)
            except Exception:
                error = ToolPolicyError("TOOL_RUNTIME_UNAVAILABLE", stop_segment=True)
        if budget is not None and binding is not None and rejected_terminal:
            try:
                await binding.store.settle(scope, budget, None)
            except Exception:
                error = ToolPolicyError("TOOL_RUNTIME_UNAVAILABLE", stop_segment=True)
        return await self._deny(scope, call, invocation_id, tool.reference if tool else None, error)

    @staticmethod
    def _admission(admitted: AdmittedToolSet, name: str, tool: AdmittedTool) -> None:
        try:
            if admitted.resolve(name) != tool:
                raise KeyError(name)
        except (KeyError, ValueError):
            raise ToolPolicyError("TOOL_ADMISSION_REVOKED", stop_segment=True) from None

    @staticmethod
    def _validate(
        context: ToolInvocationContext, tool: AdmittedTool, value: BackendToolResult
    ) -> BackendToolResult:
        value = BackendToolResult.model_validate(value)
        if any(ref.organization_id != context.scope.organization_id for ref in value.evidence):
            raise ToolPolicyError("TENANT_MISMATCH", stop_segment=True)
        if value.wait_ref is not None:
            if (
                value.wait_ref.scope != context.scope
                or value.wait_ref.source_invocation_id != context.invocation_id
            ):
                raise ToolPolicyError("TENANT_MISMATCH", stop_segment=True)
        else:
            assert value.output is not None
            output = tool.output_contract.model_validate_json(
                value.output.model_dump_json(), strict=True
            )
            _visible(output.model_dump(mode="json"), arguments=False)
            output_limit = 16384
            if tool.reference == "load_skill@1" and isinstance(output, LoadedSkill):
                if value.loaded_skill != output.pin:
                    raise ToolPolicyError("TOOL_OUTPUT_INVALID", stop_segment=True)
                # Complete 16-KiB content also needs room for its index and JSON escaping.
                # LoadedSkill independently validates the content byte limit.
                output_limit = 49152
            elif value.loaded_skill is not None:
                raise ToolPolicyError("TOOL_OUTPUT_INVALID", stop_segment=True)
            if len(output.model_dump_json().encode()) > output_limit:
                raise ToolPolicyError("TOOL_OUTPUT_LIMIT", stop_segment=True)
            value = value.model_copy(update={"output": output})
        return value

    async def _replay(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        receipt: InvocationReceipt,
        backend: RecoverableToolBackend,
    ) -> ToolOutcome:
        value = receipt.settlement
        assert value is not None
        if value.status in {"FAILED", "REJECTED"}:
            raise ToolPolicyError(value.safe_code or "TOOL_FAILED", stop_segment=True)
        if value.wait_ref is not None:
            result = self._validate(context, tool, BackendToolResult(wait_ref=value.wait_ref))
            await self.policy.guard(context, tool, result.evidence)
            if isinstance(backend, ToolWaitAccessPort):
                await backend.revalidate_wait(context, value.wait_ref)
            return DeferredToolResult(
                invocation_id=context.invocation_id,
                call_id=context.call_id,
                wait_ref=value.wait_ref,
            )
        assert value.payload_ref is not None
        payload = ToolResultPayload.model_validate(
            await self.store.get_payload(context.scope, value.payload_ref)
        )
        result = self._validate(
            context,
            tool,
            BackendToolResult(
                output=tool.output_contract.model_validate_json(
                    _encoded(payload.output), strict=True
                ),
                evidence=payload.references,
                loaded_skill=payload.loaded_skill,
            ),
        )
        await self.policy.guard(context, tool, result.evidence)
        await self._result_access(context, tool, backend, result)
        return CompletedToolResult(
            invocation_id=context.invocation_id,
            call_id=context.call_id,
            payload_ref=value.payload_ref,
            evidence=result.evidence,
        )

    @staticmethod
    async def _result_access(
        context: ToolInvocationContext,
        tool: AdmittedTool,
        backend: RecoverableToolBackend,
        result: BackendToolResult,
    ) -> None:
        if isinstance(backend, ToolResultAccessPort):
            await backend.revalidate_result(context, result.model_copy(deep=True))
        elif tool.reference == "load_skill@1":
            raise ToolPolicyError("TOOL_RESULT_VALIDATOR_UNAVAILABLE", stop_segment=True)
