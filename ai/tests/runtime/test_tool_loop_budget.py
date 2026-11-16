"""Candidate wrapper tests: reservations precede every physical provider attempt."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from work_management_ai.model_gateway.contracts import (
    ModelMessage,
    ModelUsage,
    StructuredModelRequest,
    StructuredModelResponse,
)
from work_management_ai.model_gateway.errors import ModelInvalidInputError
from work_management_ai.model_gateway.tool_contracts import (
    ToolItem,
    ToolModelRequest,
    ToolModelResponse,
)
from work_management_ai.runtime.run_budget import (
    BudgetedToolLoopGateway,
    BudgetReceipt,
    CandidateBudget,
    ChildAllocation,
    RemainingBudget,
    UsageReservation,
    candidate_budget,
)
from work_management_ai.runtime.tool_loop_contracts import (
    RunLease,
    RunScope,
    ToolLoopLimits,
    ToolLoopUsage,
)


class Answer(BaseModel):
    text: str


@dataclass(frozen=True)
class CapturedCall:
    timeout_seconds: float


class Provider:
    def __init__(self):
        self.calls: list[CapturedCall] = []
        self.usage: ModelUsage | None = ModelUsage(input_tokens=20, output_tokens=4)

    async def generate_structured[T: BaseModel](
        self, request: StructuredModelRequest[T]
    ) -> StructuredModelResponse[T]:
        self.calls.append(CapturedCall(request.timeout_seconds))
        return StructuredModelResponse(
            parsed=request.output_schema.model_validate({"text": "ok"}),
            model_ref="fixture",
            usage=self.usage,
        )

    async def generate_step[T: BaseModel](
        self, request: ToolModelRequest[T]
    ) -> ToolModelResponse[T]:
        self.calls.append(CapturedCall(request.timeout_seconds))
        return ToolModelResponse(
            final=request.response_schema.model_validate({"text": "ok"}),
            model_ref="fixture",
            usage=self.usage,
        )


class Budget:
    def __init__(self, scope: RunScope, lease: RunLease):
        self.scope, self.lease = scope, lease
        self.reservations: list[UsageReservation] = []
        self.settlements: list[ModelUsage | None] = []
        self.reject = False

    async def remaining(self, scope: RunScope) -> RemainingBudget:
        return RemainingBudget(
            usage=ToolLoopUsage(),
            available=ToolLoopUsage(input_tokens=48000, output_tokens=8000),
        )

    async def reserve(
        self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
    ) -> BudgetReceipt:
        if self.reject:
            raise ValueError("BUDGET_EXHAUSTED")
        self.reservations.append(reservation)
        return BudgetReceipt(scope=scope, lease=lease, reservation=reservation)

    async def settle(
        self, scope: RunScope, receipt: BudgetReceipt, usage: ModelUsage | None
    ) -> None:
        self.settlements.append(usage)

    async def allocate_child(
        self, scope: RunScope, allocation: ChildAllocation, *, lease: RunLease
    ) -> ChildAllocation:
        return allocation

    async def close_child(self, scope: RunScope, allocation_id: UUID, *, lease: RunLease) -> None:
        pass


@pytest.fixture
def binding() -> tuple[CandidateBudget, Budget]:
    scope = RunScope(**{key: uuid4() for key in RunScope.model_fields})
    lease = RunLease(
        scope=scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="w",
        fencing_token=1,
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    store = Budget(scope, lease)
    return CandidateBudget(scope=scope, lease=lease, store=store, interval_id=uuid4()), store


def request() -> StructuredModelRequest[Answer]:
    return StructuredModelRequest(
        invocation_key="same-key",
        messages=(ModelMessage(role="user", content="hello"),),
        output_schema=Answer,
        timeout_seconds=60,
        max_output_tokens=100,
    )


@pytest.mark.asyncio
async def test_legacy_is_noop_outside_candidate():
    provider = Provider()
    gateway = BudgetedToolLoopGateway(provider, native=provider)
    await gateway.generate_structured(request())
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_native_and_legacy_reserve_schema_framing_then_settle(
    binding: tuple[CandidateBudget, Budget], native: bool
):
    context, store = binding
    provider = Provider()
    gateway = BudgetedToolLoopGateway(provider, native=provider)
    with candidate_budget(context):
        if native:
            await gateway.generate_step(
                ToolModelRequest(
                    invocation_key="n",
                    items=(ToolItem(kind="message", role="user", content="hello"),),
                    tools=(),
                    response_schema=Answer,
                    max_output_tokens=100,
                )
            )
        else:
            await gateway.generate_structured(request())
    assert len(store.reservations) == len(provider.calls) == 1
    assert store.reservations[0].input_tokens > 1024
    assert store.reservations[0].output_tokens == 100
    assert provider.calls[0].timeout_seconds <= 30
    assert store.settlements == [ModelUsage(input_tokens=20, output_tokens=4)]


@pytest.mark.asyncio
async def test_denial_precedes_provider(binding: tuple[CandidateBudget, Budget]):
    context, store = binding
    store.reject = True
    provider = Provider()
    with candidate_budget(context), pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        await BudgetedToolLoopGateway(provider).generate_structured(request())
    assert provider.calls == []


@pytest.mark.asyncio
async def test_unknown_usage_and_distinct_physical_attempts(
    binding: tuple[CandidateBudget, Budget],
):
    context, store = binding
    provider = Provider()
    provider.usage = None
    with candidate_budget(context):
        gateway = BudgetedToolLoopGateway(provider)
        await gateway.generate_structured(request())
        await gateway.generate_structured(request())
    assert store.settlements == [None, None]
    assert store.reservations[0].operation_id != store.reservations[1].operation_id


def test_limits_still_match_approved_caps():
    assert ToolLoopLimits().model_dump() == dict(
        root_model_requests=8,
        total_model_requests=24,
        total_tool_calls=12,
        handoffs=6,
        input_tokens=48000,
        output_tokens=8000,
        segments=4,
        segment_seconds=180,
        active_seconds=720,
        retry_per_invocation=1,
        repair_per_response=1,
    )
    with pytest.raises(ValueError):
        UsageReservation(kind="MODEL", input_tokens=0, output_tokens=100, interval_id=uuid4())


@pytest.mark.asyncio
async def test_overreported_usage_does_not_refund(binding: tuple[CandidateBudget, Budget]):
    context, store = binding
    provider = Provider()
    provider.usage = ModelUsage(input_tokens=99999, output_tokens=4)
    with (
        candidate_budget(context),
        pytest.raises(ValueError, match="PROVIDER_USAGE_EXCEEDS_RESERVATION"),
    ):
        await BudgetedToolLoopGateway(provider).generate_structured(request())
    assert store.settlements == [None]


@pytest.mark.asyncio
async def test_media_without_bound_never_enters_provider(binding: tuple[CandidateBudget, Budget]):
    from dataclasses import replace

    from work_management_ai.model_gateway.contracts import ModelImagePart

    context, store = binding
    provider = Provider()
    media = replace(
        request(),
        messages=(
            ModelMessage(
                role="user",
                content="image",
                images=(
                    ModelImagePart(
                        mime_type="image/png",
                        data=b"\x89PNG\r\n\x1a\nfixture",
                        source_ref="authorized-fixture",
                    ),
                ),
            ),
        ),
    )
    with (
        candidate_budget(context),
        pytest.raises(ValueError, match="MEDIA_TOKEN_BOUND_UNAVAILABLE"),
    ):
        await BudgetedToolLoopGateway(provider).generate_structured(media)
    assert provider.calls == store.reservations == []


@pytest.mark.asyncio
async def test_replayed_receipt_does_not_call_provider(binding: tuple[CandidateBudget, Budget]):
    class ReplayingBudget(Budget):
        async def reserve(
            self, scope: RunScope, reservation: UsageReservation, *, lease: RunLease
        ) -> BudgetReceipt:
            receipt = await super().reserve(scope, reservation, lease=lease)
            return receipt.model_copy(update={"replayed": True})

    from dataclasses import replace

    context, _ = binding
    replay = ReplayingBudget(context.scope, context.lease)
    provider = Provider()
    with (
        candidate_budget(replace(context, store=replay)),
        pytest.raises(ValueError, match="MODEL_ATTEMPT_ALREADY_RESERVED"),
    ):
        await BudgetedToolLoopGateway(provider).generate_structured(request())
    assert provider.calls == []


@pytest.mark.asyncio
async def test_invalid_timeout_denied_before_reservation(binding: tuple[CandidateBudget, Budget]):
    from dataclasses import replace

    context, store = binding
    provider = Provider()
    with candidate_budget(context), pytest.raises(ModelInvalidInputError):
        await BudgetedToolLoopGateway(provider).generate_structured(
            replace(request(), timeout_seconds=float("nan"))
        )
    assert store.reservations == provider.calls == []
