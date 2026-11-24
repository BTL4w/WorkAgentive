"""Operational v2 contracts. IDs, snapshots and pins are never authorization."""

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from work_management_ai.model_gateway.tool_contracts import ToolSchema
from work_management_ai.runtime.contracts import ContextReference, RiskLevel


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class RunScope(_Contract):
    organization_id: UUID
    membership_id: UUID
    conversation_id: UUID
    turn_id: UUID
    orchestration_run_id: UUID


class RunLease(_Contract):
    scope: RunScope
    lease_id: UUID
    job_id: UUID
    worker_id: str = Field(min_length=1, max_length=128)
    fencing_token: int = Field(ge=1, strict=True)
    expires_at: AwareDatetime
    job_attempt: int | None = Field(default=None, ge=1, strict=True)


class ToolLoopLimits(_Contract):
    root_model_requests: int = Field(default=8, ge=1, le=8, strict=True)
    total_model_requests: int = Field(default=24, ge=1, le=24, strict=True)
    total_tool_calls: int = Field(default=12, ge=0, le=12, strict=True)
    handoffs: int = Field(default=6, ge=0, le=6, strict=True)
    input_tokens: int = Field(default=48000, ge=1, le=48000, strict=True)
    output_tokens: int = Field(default=8000, ge=1, le=8000, strict=True)
    segments: int = Field(default=4, ge=1, le=4, strict=True)
    segment_seconds: int = Field(default=180, ge=1, le=180, strict=True)
    active_seconds: int = Field(default=720, ge=1, le=720, strict=True)
    retry_per_invocation: int = Field(default=1, ge=0, le=1, strict=True)
    repair_per_response: int = Field(default=1, ge=0, le=1, strict=True)

    @model_validator(mode="after")
    def total_covers_root(self) -> Self:
        if self.root_model_requests > self.total_model_requests:
            raise ValueError("ROOT_BUDGET_EXCEEDS_TOTAL")
        return self


class ToolLoopUsage(_Contract):
    root_model_requests: int = Field(default=0, ge=0, strict=True)
    total_model_requests: int = Field(default=0, ge=0, strict=True)
    total_tool_calls: int = Field(default=0, ge=0, strict=True)
    handoffs: int = Field(default=0, ge=0, strict=True)
    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    segments: int = Field(default=0, ge=0, strict=True)
    active_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def total_includes_root(self) -> Self:
        if self.root_model_requests > self.total_model_requests:
            raise ValueError("ROOT_USAGE_EXCEEDS_TOTAL")
        return self


class PayloadReference(_Contract):
    """Expiring payload identity only; raw content belongs in scoped storage."""

    payload_id: UUID
    organization_id: UUID
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    expires_at: AwareDatetime


class VersionPin(_Contract):
    reference: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    fingerprint: str = Field(min_length=1, max_length=128)


class WaitReference(_Contract):
    wait_id: UUID
    scope: RunScope
    kind: Literal["USER_INPUT", "WORKFLOW_RESULT", "HUMAN_DECISION"]
    source_invocation_id: UUID
    resource_id: UUID | None = None
    resource_version: int | None = Field(default=None, ge=1, strict=True)
    policy_fingerprint: str | None = Field(default=None, min_length=1, max_length=128)
    source_fingerprint: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def complete_resource_binding(self) -> Self:
        if (self.resource_id is None) != (self.resource_version is None):
            raise ValueError("WAIT_VERSION_BINDING_INCOMPLETE")
        return self


class ToolLoopState(_Contract):
    schema_version: Literal["2.0"] = "2.0"
    scope: RunScope
    driver_version: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    catalog_fingerprint: str = Field(min_length=1, max_length=128)
    agent_version: str = Field(default="2.0.0", min_length=1, max_length=64)
    manifest_fingerprint: str | None = Field(default=None, min_length=1, max_length=128)
    model_ref: str | None = Field(default=None, min_length=1, max_length=200)
    tool_versions: tuple[VersionPin, ...] = ()
    skill_versions: tuple[VersionPin, ...] = ()
    sequence: int = Field(default=0, ge=0, strict=True)
    segment_id: UUID | None = None
    pending_invocation_id: UUID | None = None
    pending_wait: WaitReference | None = None
    objective_ref: PayloadReference | None = None
    working_plan_ref: PayloadReference | None = None
    payload_refs: tuple[PayloadReference, ...] = ()
    limits: ToolLoopLimits = Field(default_factory=ToolLoopLimits)
    usage: ToolLoopUsage = Field(default_factory=ToolLoopUsage)

    @model_validator(mode="after")
    def scoped_references_and_bounded_usage(self) -> Self:
        refs = (*self.payload_refs, self.objective_ref, self.working_plan_ref)
        if any(
            ref is not None and ref.organization_id != self.scope.organization_id for ref in refs
        ):
            raise ValueError("CHECKPOINT_PAYLOAD_SCOPE_MISMATCH")
        if self.pending_wait is not None and self.pending_wait.scope != self.scope:
            raise ValueError("CHECKPOINT_WAIT_SCOPE_MISMATCH")
        for name in ToolLoopUsage.model_fields:
            if getattr(self.usage, name) > getattr(self.limits, name):
                raise ValueError("CHECKPOINT_USAGE_EXCEEDS_LIMIT")
        return self


class ToolLoopCheckpoint(ToolLoopState):
    """Serializable recovery state with no raw messages, results or authority."""


@dataclass(frozen=True, slots=True)
class AdmittedTool:
    """Registry metadata only. Handlers and provider SDKs stay outside this type."""

    schema: ToolSchema
    input_contract: type[BaseModel]
    output_contract: type[BaseModel]
    version: str
    fingerprint: str
    risk: RiskLevel = RiskLevel.READ_ONLY
    permissions: tuple[str, ...] = ()
    timeout_seconds: int = 30
    retry_limit: int = 0
    idempotency: Literal["NONE", "REQUIRED"] = "NONE"
    audit: Literal["NONE", "REQUIRED"] = "NONE"

    @property
    def reference(self) -> str:
        return self.schema.reference


class AdmittedToolSet(Protocol):
    @property
    def schemas(self) -> tuple[ToolSchema, ...]: ...
    @property
    def fingerprint(self) -> str: ...
    def resolve(self, wire_name: str) -> AdmittedTool: ...
    def close(self) -> None: ...


class ToolInvocationContext(_Contract):
    scope: RunScope
    lease: RunLease
    invocation_id: UUID
    call_id: str = Field(min_length=1, max_length=128)
    tool_reference: str = Field(min_length=1, max_length=100)
    tool_version: str = Field(min_length=1, max_length=64)
    arguments_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_arguments: SerializeAsAny[BaseModel]
    policy_version: str = Field(min_length=1, max_length=128)
    resource_versions: tuple[ContextReference, ...] = ()
    receipt_id: UUID
    budget_allocation_id: UUID
    limits: ToolLoopLimits

    @model_validator(mode="after")
    def bindings_match(self) -> Self:
        if self.lease.scope != self.scope or any(
            ref.organization_id != self.scope.organization_id for ref in self.resource_versions
        ):
            raise ValueError("INVOCATION_SCOPE_MISMATCH")
        return self


class BackendToolResult(_Contract):
    """Backend result before canonical payload settlement by the pipeline."""

    output: SerializeAsAny[BaseModel] | None = None
    wait_ref: WaitReference | None = None
    evidence: tuple[ContextReference, ...] = ()
    loaded_skill: VersionPin | None = None

    @model_validator(mode="after")
    def output_or_wait(self) -> Self:
        if (self.output is None) == (self.wait_ref is None):
            raise ValueError("BACKEND_RESULT_REQUIRES_OUTPUT_OR_WAIT")
        if self.wait_ref is not None and self.loaded_skill is not None:
            raise ValueError("LOADED_SKILL_REQUIRES_COMPLETED_READ")
        return self


class CompletedToolResult(_Contract):
    kind: Literal["COMPLETED"] = "COMPLETED"
    invocation_id: UUID
    call_id: str = Field(min_length=1, max_length=128)
    payload_ref: PayloadReference
    evidence: tuple[ContextReference, ...] = ()


class DeferredToolResult(_Contract):
    kind: Literal["DEFERRED"] = "DEFERRED"
    invocation_id: UUID
    call_id: str = Field(min_length=1, max_length=128)
    wait_ref: WaitReference


class RejectedToolResult(_Contract):
    kind: Literal["REJECTED"] = "REJECTED"
    invocation_id: UUID
    call_id: str = Field(min_length=1, max_length=128)
    safe_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,99}$")
    retryable: bool = False
    stop_segment: bool = False


type ToolOutcome = Annotated[
    CompletedToolResult | DeferredToolResult | RejectedToolResult, Field(discriminator="kind")
]


class ToolBackendPort(Protocol):
    async def execute(
        self, context: ToolInvocationContext, arguments: BaseModel
    ) -> BackendToolResult: ...
