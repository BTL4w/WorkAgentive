"""Candidate-only conserved budget port and provider boundary.

A reservation is durable before a physical attempt. Replayed receipts authorize
reconciliation, never another provider request. Legacy calls outside candidate
scope retain their existing behavior and v1 budgets.
"""

import asyncio
import json
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal, Protocol, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from work_management_ai.model_gateway.contracts import (
    ModelGateway,
    ModelMessage,
    ModelUsage,
    StructuredModelRequest,
    StructuredModelResponse,
    validate_request,
)
from work_management_ai.model_gateway.errors import ModelInvalidOutputError, ModelTimeoutError
from work_management_ai.model_gateway.tool_contracts import (
    NativeToolGateway,
    ToolModelRequest,
    ToolModelResponse,
)
from work_management_ai.runtime.contracts import (
    DelegationBrief,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutorPort,
)
from work_management_ai.runtime.tool_loop_contracts import RunLease, RunScope, ToolLoopUsage


class BudgetError(ValueError):
    """Safe deterministic denial; no provider payloads."""


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class UsageReservation(_Contract):
    operation_id: UUID = Field(default_factory=uuid4)
    kind: Literal["MODEL", "TOOL", "HANDOFF", "INTERVAL"]
    allocation_id: UUID | None = None
    interval_id: UUID | None = None
    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    retry_of: UUID | None = None
    repair_of: UUID | None = None

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.kind == "MODEL":
            valid = self.input_tokens > 0 and self.output_tokens > 0
        else:
            valid = self.input_tokens == self.output_tokens == 0
        valid = valid and ((self.kind == "INTERVAL") == (self.interval_id is None))
        valid = valid and not (self.retry_of and self.repair_of)
        valid = valid and (self.kind == "MODEL" or not (self.retry_of or self.repair_of))
        if not valid:
            raise BudgetError("BUDGET_RESERVATION_INVALID")
        return self


class BudgetReceipt(_Contract):
    scope: RunScope
    lease: RunLease
    reservation: UsageReservation
    replayed: bool = False
    seconds_remaining: float = Field(default=30, gt=0, le=180)


class ChildJobClaim(_Contract):
    kind: Literal["PLANNING", "REPORTING"]
    job_id: UUID
    worker_id: str = Field(min_length=1, max_length=128)
    attempt: int = Field(gt=0, strict=True)
    lease_until: AwareDatetime


class JobBudgetReceipt(_Contract):
    scope: RunScope
    claim: ChildJobClaim
    reservation: UsageReservation
    replayed: bool = False
    seconds_remaining: float = Field(gt=0, le=180)


class ChildAllocation(_Contract):
    allocation_id: UUID
    allowance: ToolLoopUsage


class RemainingBudget(_Contract):
    usage: ToolLoopUsage
    available: ToolLoopUsage


class RunBudgetPort(Protocol):
    async def recover_tool(
        self,
        scope: RunScope,
        operation_id: UUID,
        *,
        lease: RunLease,
        interval_id: UUID,
        allocation_id: UUID | None,
    ) -> BudgetReceipt:
        """Recover a tool's original reservation without charging a resumed interval."""
        ...

    async def reserve(
        self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt: ...
    async def allocate_child(
        self, scope: RunScope, allocation: ChildAllocation, *, lease: RunLease
    ) -> ChildAllocation: ...
    async def close_child(
        self, scope: RunScope, allocation_id: UUID, *, lease: RunLease
    ) -> None: ...
    async def settle(
        self, scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
    ) -> None: ...
    async def remaining(self, scope: RunScope) -> RemainingBudget: ...


class ChildJobBudgetPort(Protocol):
    async def reserve_for_job(
        self,
        scope: RunScope,
        allocation_id: UUID,
        *,
        job_claim_token: ChildJobClaim,
        reservation: UsageReservation,
    ) -> JobBudgetReceipt: ...
    async def settle_for_job(
        self,
        scope: RunScope,
        receipt: JobBudgetReceipt,
        usage: ModelUsage | None,
    ) -> None: ...


@dataclass(frozen=True)
class ChildJobBudget:
    scope: RunScope
    allocation_id: UUID
    claim: ChildJobClaim
    store: ChildJobBudgetPort
    interval_id: UUID
    model_output_cap: int | None = None
    operation_id: UUID | None = None
    retry_of: UUID | None = None
    repair_of: UUID | None = None
    delegation_brief: DelegationBrief | None = None


_CHILD_JOB: ContextVar[ChildJobBudget | None] = ContextVar("child_job_budget", default=None)


@contextmanager
def child_job_budget(binding: ChildJobBudget) -> Generator[None]:
    token = _CHILD_JOB.set(binding)
    try:
        yield
    finally:
        _CHILD_JOB.reset(token)


@dataclass(frozen=True)
class CandidateBudget:
    scope: RunScope
    lease: RunLease
    store: RunBudgetPort
    interval_id: UUID
    allocation_id: UUID | None = None
    operation_id: UUID | None = None
    retry_of: UUID | None = None
    repair_of: UUID | None = None
    model_output_cap: int | None = None
    delegation_brief: DelegationBrief | None = None


_CANDIDATE: ContextVar[CandidateBudget | None] = ContextVar(
    "tool_loop_candidate_budget", default=None
)


@contextmanager
def candidate_budget(binding: CandidateBudget) -> Generator[None]:
    if binding.lease.scope != binding.scope:
        raise BudgetError("BUDGET_SCOPE_MISMATCH")
    token = _CANDIDATE.set(binding)
    try:
        yield
    finally:
        _CANDIDATE.reset(token)


def current_candidate_budget() -> CandidateBudget | None:
    """Runtime binding only; never populated from model or client content."""
    return _CANDIDATE.get()


class BudgetedSpecialistTools:
    """Charge each legacy Specialist tool call against the candidate child allocation."""

    def __init__(self, executor: ToolExecutorPort) -> None:
        self._executor = executor

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        binding = current_candidate_budget()
        if binding is None:
            return await self._executor.execute(request)
        if (
            binding.allocation_id is None
            or request.actor.organization_id != binding.scope.organization_id
            or request.actor.membership_id != binding.scope.membership_id
        ):
            raise BudgetError("BUDGET_SCOPE_MISMATCH")
        receipt = await binding.store.reserve(
            binding.scope,
            UsageReservation(
                kind="TOOL",
                allocation_id=binding.allocation_id,
                interval_id=binding.interval_id,
            ),
            lease=binding.lease,
        )
        try:
            async with asyncio.timeout(receipt.seconds_remaining):
                return await self._executor.execute(request)
        finally:
            await binding.store.settle(binding.scope, receipt, None)


def _bytes(value: object) -> int:
    # ASCII JSON also bounds escaped text; include schemas, aliases and framing.
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode())


class BudgetedToolLoopGateway:
    def __init__(
        self,
        gateway: ModelGateway,
        *,
        native: NativeToolGateway | None = None,
        image_token_bound: int | None = None,
    ):
        self.gateway, self.native = gateway, native
        self.image_token_bound = image_token_bound

    async def _reserve(
        self,
        context: CandidateBudget | ChildJobBudget,
        inputs: int,
        outputs: int,
    ) -> BudgetReceipt | JobBudgetReceipt:
        reservation = UsageReservation(
            operation_id=context.operation_id or uuid4(),
            kind="MODEL",
            allocation_id=context.allocation_id,
            interval_id=context.interval_id,
            input_tokens=inputs,
            output_tokens=outputs,
            retry_of=context.retry_of,
            repair_of=context.repair_of,
        )
        if isinstance(context, ChildJobBudget):
            receipt = await context.store.reserve_for_job(
                context.scope,
                context.allocation_id,
                job_claim_token=context.claim,
                reservation=reservation,
            )
        else:
            receipt = await context.store.reserve(context.scope, reservation, lease=context.lease)
        if receipt.replayed:
            raise BudgetError("MODEL_ATTEMPT_ALREADY_RESERVED")
        return receipt

    @staticmethod
    async def _settle(
        context: CandidateBudget | ChildJobBudget,
        receipt: BudgetReceipt | JobBudgetReceipt,
        usage: ModelUsage | None,
    ) -> None:
        exceeds = usage is not None and (
            usage.input_tokens > receipt.reservation.input_tokens
            or usage.output_tokens > receipt.reservation.output_tokens
        )
        measured = None if exceeds else usage
        if isinstance(context, ChildJobBudget) and isinstance(receipt, JobBudgetReceipt):
            await context.store.settle_for_job(context.scope, receipt, measured)
        elif isinstance(context, CandidateBudget) and isinstance(receipt, BudgetReceipt):
            await context.store.settle(context.scope, receipt, measured)
        else:
            raise BudgetError("BUDGET_SCOPE_MISMATCH")
        if exceeds:
            raise BudgetError("PROVIDER_USAGE_EXCEEDS_RESERVATION")

    async def generate_structured[OutputT: BaseModel](
        self, request: StructuredModelRequest[OutputT]
    ) -> StructuredModelResponse[OutputT]:
        context = _CANDIDATE.get() or _CHILD_JOB.get()
        if context is None:
            return await self.gateway.generate_structured(request)
        if context.delegation_brief is not None:
            message = ModelMessage(
                role="user",
                content=json.dumps(
                    {"delegated_objective": context.delegation_brief.model_dump(mode="json")},
                    ensure_ascii=False,
                ),
            )
            request = replace(request, messages=(*request.messages, message))
        cap = validate_request(request)
        images = sum(len(m.images) for m in request.messages)
        if images and (self.image_token_bound is None or self.image_token_bound <= 0):
            raise BudgetError("MEDIA_TOKEN_BOUND_UNAVAILABLE")
        inputs = _bytes([{"role": m.role, "content": m.content} for m in request.messages])
        inputs += (
            _bytes(request.output_schema.model_json_schema()) + 1024 + 64 * len(request.messages)
        )
        inputs += images * (self.image_token_bound or 0)
        outputs = cap or 4000
        if context.model_output_cap is not None:
            if context.model_output_cap < 1:
                raise BudgetError("BUDGET_EXHAUSTED")
            outputs = min(outputs, context.model_output_cap)
        receipt = await self._reserve(context, inputs, outputs)
        seconds = min(request.timeout_seconds, receipt.seconds_remaining, 60)
        bounded = replace(request, timeout_seconds=seconds, max_output_tokens=outputs)
        try:
            async with asyncio.timeout(seconds):
                result = await self.gateway.generate_structured(bounded)
        except ModelInvalidOutputError as exc:
            await self._settle(context, receipt, exc.usage)
            raise
        except TimeoutError:
            await self._settle(context, receipt, None)
            raise ModelTimeoutError("MODEL_TIMEOUT") from None
        except BaseException:
            await self._settle(context, receipt, None)
            raise
        await self._settle(context, receipt, result.usage)
        return result

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        if self.native is None:
            raise BudgetError("NATIVE_GATEWAY_UNAVAILABLE")
        context = _CANDIDATE.get()
        if context is None:
            return await self.native.generate_step(request)
        request = type(request).model_validate(request)
        inputs = _bytes([item.model_dump(mode="json") for item in request.items])
        inputs += _bytes([tool.model_dump(mode="json") for tool in request.tools]) + 64 * len(
            request.tools
        )
        inputs += (
            _bytes(request.response_schema.model_json_schema()) + 1024 + 64 * len(request.items)
        )
        receipt = await self._reserve(context, inputs, request.max_output_tokens)
        seconds = min(request.timeout_seconds, receipt.seconds_remaining, 60)
        bounded = request.model_copy(update={"timeout_seconds": seconds})
        try:
            async with asyncio.timeout(seconds):
                result = await self.native.generate_step(bounded)
        except ModelInvalidOutputError as exc:
            await self._settle(context, receipt, exc.usage)
            raise
        except TimeoutError:
            await self._settle(context, receipt, None)
            raise ModelTimeoutError("MODEL_TIMEOUT") from None
        except BaseException:
            await self._settle(context, receipt, None)
            raise
        await self._settle(context, receipt, result.usage)
        return result
