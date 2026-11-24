"""AI-port adapter to application-owned journaling. No SQL or model calls here."""

from dataclasses import asdict
from typing import Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import TypeAdapter

from app.modules.assistant.application.session_service import SessionService
from app.modules.assistant.domain.session import (
    SessionError,
    SessionEvent,
    SessionLease,
    SessionPayload,
    SessionPins,
    SessionScope,
    StoredSessionEvent,
)
from app.modules.identity.domain.auth import AuthenticatedActor
from work_management_ai.runtime.contracts import JsonValue
from work_management_ai.runtime.run_budget import UsageReservation, current_candidate_budget
from work_management_ai.runtime.session_log import (
    EventDraft,
    InvocationDescriptor,
    InvocationReceipt,
    InvocationStartedError,
    PayloadKind,
    SessionRecord,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import (
    PayloadReference,
    RunLease,
    RunScope,
    ToolLoopCheckpoint,
    WaitReference,
)


def session_scope(scope: RunScope) -> SessionScope:
    return SessionScope(**scope.model_dump())


def session_lease(lease: RunLease) -> SessionLease:
    return SessionLease(
        session_scope(lease.scope),
        lease.lease_id,
        lease.job_id,
        lease.worker_id,
        lease.fencing_token,
        lease.expires_at,
        lease.job_attempt,
    )


def _pins(state: ToolLoopCheckpoint) -> SessionPins:
    return SessionPins(
        state.driver_version,
        state.prompt_version,
        state.catalog_fingerprint,
        state.agent_version,
        state.manifest_fingerprint,
        state.model_ref,
    )


def _reference(payload: SessionPayload) -> PayloadReference:
    return PayloadReference(
        payload_id=payload.id,
        organization_id=payload.organization_id,
        fingerprint=payload.fingerprint,
        expires_at=payload.expires_at,
    )


def _event(event: EventDraft) -> SessionEvent:
    invocation = event.invocation
    settlement = event.settlement
    wait = settlement.wait_ref if settlement else None
    return SessionEvent(
        event_id=event.event_id,
        kind=event.kind,
        payload_id=event.payload_ref.payload_id if event.payload_ref else None,
        invocation_id=invocation.invocation_id if invocation else None,
        call_id=invocation.call_id if invocation else None,
        tool_reference=invocation.tool_reference if invocation else None,
        tool_version=invocation.tool_version if invocation else None,
        arguments_fingerprint=invocation.arguments_fingerprint if invocation else None,
        settlement_status=settlement.status
        if settlement and settlement.status != "DEFERRED"
        else None,
        wait_id=wait.wait_id if wait else None,
        wait_kind=wait.kind if wait else None,
        resource_id=wait.resource_id if wait else None,
        resource_version=wait.resource_version if wait else None,
        policy_fingerprint=wait.policy_fingerprint if wait else None,
        source_fingerprint=wait.source_fingerprint if wait else None,
        safe_code=settlement.safe_code if settlement else event.safe_code,
        agent_run_id=event.agent_run_id,
        causation_id=event.causation_id,
        correlation_id=event.correlation_id,
    )


class PostgreSQLToolLoopRecorder:
    def __init__(self, *, service: SessionService, actor: AuthenticatedActor):
        self.service = service
        self._actor = actor

    def _assert_scope(self, scope: RunScope) -> None:
        if (scope.organization_id, scope.membership_id) != (
            self._actor.organization_id,
            self._actor.membership_id,
        ):
            raise SessionError("SESSION_NOT_FOUND")

    async def initialize(self, scope: RunScope, state: ToolLoopCheckpoint) -> None:
        self._assert_scope(scope)
        state = ToolLoopCheckpoint.model_validate(state)
        if state.scope != scope:
            raise SessionError("CHECKPOINT_SCOPE_MISMATCH")
        await self.service.initialize(
            self._actor, session_scope(scope), _pins(state), state.model_dump(mode="json")
        )

    async def claim(
        self,
        scope: RunScope,
        *,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
        job_attempt: int | None = None,
    ) -> RunLease:
        value = await self.service.claim(
            self._actor,
            session_scope(scope),
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            job_attempt=job_attempt,
        )
        return RunLease.model_validate({**asdict(value), "scope": scope})

    async def renew(self, scope: RunScope, lease: RunLease, *, lease_seconds: int) -> RunLease:
        value = await self.service.renew(
            self._actor, session_scope(scope), session_lease(lease), lease_seconds=lease_seconds
        )
        return RunLease.model_validate({**asdict(value), "scope": scope})

    async def put_payload(
        self,
        scope: RunScope,
        body: JsonValue,
        *,
        classification: Literal["RAW_CONTEXT"],
        kind: PayloadKind,
        lease: RunLease,
        ttl_seconds: int | None = None,
    ) -> PayloadReference:
        if classification != "RAW_CONTEXT" or kind == "CHECKPOINT":
            raise SessionError("PAYLOAD_KIND_INVALID")
        body = TypeAdapter[JsonValue](JsonValue).validate_python(body)
        value = await self.service.put_payload(
            self._actor,
            session_scope(scope),
            body,
            kind=kind,
            lease=session_lease(lease),
            ttl_seconds=ttl_seconds,
        )
        return _reference(value)

    async def _check_reference(self, scope: RunScope, reference: PayloadReference) -> None:
        self._assert_scope(scope)
        if reference.organization_id != scope.organization_id:
            raise SessionError("SESSION_NOT_FOUND")
        metadata = await self.service.payload(
            self._actor, session_scope(scope), reference.payload_id, live=False
        )
        if _reference(metadata) != reference:
            raise SessionError("PAYLOAD_REFERENCE_MISMATCH")

    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> JsonValue:
        """Live scoped bytes; history must additionally revalidate resource access."""
        await self._check_reference(scope, reference)
        payload = await self.service.payload(
            self._actor, session_scope(scope), reference.payload_id
        )
        return cast(JsonValue, payload.body)

    async def checkpoint(
        self, scope: RunScope, state: ToolLoopCheckpoint, *, lease: RunLease
    ) -> None:
        state = ToolLoopCheckpoint.model_validate(state)
        if state.scope != scope:
            raise SessionError("CHECKPOINT_SCOPE_MISMATCH")
        refs = (*state.payload_refs, state.objective_ref, state.working_plan_ref)
        for ref in refs:
            if ref is not None:
                await self._check_reference(scope, ref)
        await self.service.checkpoint(
            self._actor,
            session_scope(scope),
            _pins(state),
            state.model_dump(mode="json"),
            state.sequence,
            lease=session_lease(lease),
            references=tuple(ref.payload_id for ref in refs if ref is not None),
        )

    async def load_checkpoint(self, scope: RunScope) -> ToolLoopCheckpoint | None:
        value = await self.service.load_checkpoint(self._actor, session_scope(scope))
        return None if value is None else ToolLoopCheckpoint.model_validate(value.body)

    async def _record(self, scope: RunScope, value: StoredSessionEvent) -> SessionRecord:
        event = value.event
        ref = None
        if event.payload_id is not None:
            ref = _reference(
                await self.service.payload(
                    self._actor, session_scope(scope), event.payload_id, live=False
                )
            )
        invocation = None
        if event.invocation_id is not None:
            invocation = InvocationDescriptor.model_validate(
                {
                    "invocation_id": event.invocation_id,
                    "call_id": event.call_id,
                    "tool_reference": event.tool_reference,
                    "tool_version": event.tool_version,
                    "arguments_fingerprint": event.arguments_fingerprint,
                }
            )
        settlement = None
        if event.settlement_status is not None or event.wait_id is not None:
            wait = None
            if event.wait_id is not None:
                wait = WaitReference.model_validate(
                    {
                        "wait_id": event.wait_id,
                        "scope": scope,
                        "kind": event.wait_kind,
                        "source_invocation_id": event.invocation_id,
                        "resource_id": event.resource_id,
                        "resource_version": event.resource_version,
                        "policy_fingerprint": event.policy_fingerprint,
                        "source_fingerprint": event.source_fingerprint,
                    }
                )
            settlement = SettledInvocation.model_validate(
                {
                    "status": event.settlement_status or "DEFERRED",
                    "payload_ref": ref,
                    "wait_ref": wait,
                    "safe_code": event.safe_code,
                }
            )
        return SessionRecord.model_validate(
            {
                "scope": scope,
                "sequence": value.sequence,
                "occurred_at": value.occurred_at,
                "event_id": event.event_id,
                "kind": event.kind,
                "safe_code": event.safe_code if settlement is None else None,
                "payload_ref": ref,
                "invocation": invocation,
                "settlement": settlement,
                "agent_run_id": event.agent_run_id,
                "causation_id": event.causation_id,
                "correlation_id": event.correlation_id,
            }
        )

    async def append(
        self, scope: RunScope, event: EventDraft, *, expected_sequence: int, lease: RunLease
    ) -> SessionRecord:
        event = EventDraft.model_validate(event)
        if event.payload_ref is not None:
            await self._check_reference(scope, event.payload_ref)
        binding = current_candidate_budget()
        if binding is not None and event.kind == "tool.requested":
            if binding.scope != scope or binding.lease != lease:
                raise SessionError("BUDGET_SCOPE_MISMATCH")
            assert event.invocation is not None
            await binding.store.reserve(
                scope,
                UsageReservation(
                    operation_id=event.invocation.invocation_id,
                    kind="HANDOFF"
                    if event.invocation.tool_reference.rsplit("@", 1)[0] == "delegate_agent"
                    else "TOOL",
                    allocation_id=binding.allocation_id,
                    interval_id=binding.interval_id,
                ),
                lease=lease,
            )
        value = await self.service.append(
            self._actor,
            session_scope(scope),
            _event(event),
            expected_sequence=expected_sequence,
            lease=session_lease(lease),
        )
        return await self._record(scope, value)

    async def read(self, scope: RunScope, *, after_sequence: int) -> tuple[SessionRecord, ...]:
        """Safe headers survive expiry; never feed their payloads to a model directly."""
        values = await self.service.read(
            self._actor, session_scope(scope), after_sequence=after_sequence
        )
        return tuple([await self._record(scope, value) for value in values])

    async def get_invocation(
        self, scope: RunScope, invocation_id: UUID
    ) -> InvocationReceipt | None:
        values = await self.service.read(
            self._actor, session_scope(scope), after_sequence=0, invocation_id=invocation_id
        )
        requested = next((v for v in values if v.event.kind == "tool.requested"), None)
        if requested is None:
            return None
        record = await self._record(scope, requested)
        assert record.invocation is not None and record.payload_ref is not None
        settled = next(
            (
                v
                for v in reversed(values)
                if v.event.settlement_status is not None or v.event.wait_id is not None
            ),
            None,
        )
        result = await self._record(scope, settled) if settled else None
        return InvocationReceipt(
            scope=scope,
            descriptor=record.invocation,
            requested_event_id=record.event_id,
            requested_sequence=record.sequence,
            arguments_ref=record.payload_ref,
            settlement=result.settlement if result else None,
            settled_event_id=result.event_id if result else None,
        )

    async def start_invocation(
        self, scope: RunScope, receipt: InvocationReceipt, *, lease: RunLease
    ) -> bool:
        if receipt.scope != scope:
            raise SessionError("INVOCATION_CONFLICT")
        records = await self.read(scope, after_sequence=0)
        original = next((r for r in records if r.event_id == receipt.requested_event_id), None)
        if (
            original is None
            or original.invocation != receipt.descriptor
            or original.payload_ref != receipt.arguments_ref
        ):
            raise SessionError("INVOCATION_CONFLICT")
        event = EventDraft(
            event_id=uuid5(receipt.descriptor.invocation_id, "started"),
            kind="tool.started",
            correlation_id=lease.lease_id,
            invocation=receipt.descriptor,
            causation_id=receipt.requested_event_id,
        )
        return await self.service.start_invocation(
            self._actor,
            session_scope(scope),
            _event(original),
            _event(event),
            lease=session_lease(lease),
        )

    async def settle_invocation(
        self,
        scope: RunScope,
        receipt: InvocationReceipt,
        value: SettledInvocation,
        *,
        lease: RunLease,
    ) -> SessionRecord:
        value = SettledInvocation.model_validate(value)
        if receipt.scope != scope or (
            value.wait_ref is not None
            and (
                value.wait_ref.scope != scope
                or value.wait_ref.source_invocation_id != receipt.descriptor.invocation_id
            )
        ):
            raise SessionError("INVOCATION_CONFLICT")
        if value.payload_ref is not None:
            await self._check_reference(scope, value.payload_ref)
        records = await self.service.read(
            self._actor,
            session_scope(scope),
            after_sequence=0,
            invocation_id=receipt.descriptor.invocation_id,
        )
        original = next((v for v in records if v.event.kind == "tool.requested"), None)
        if original is None:
            raise SessionError("INVOCATION_CONFLICT")
        actual = await self._record(scope, original)
        if (actual.event_id, actual.sequence, actual.invocation, actual.payload_ref) != (
            receipt.requested_event_id,
            receipt.requested_sequence,
            receipt.descriptor,
            receipt.arguments_ref,
        ):
            raise SessionError("INVOCATION_CONFLICT")
        draft = EventDraft(
            event_id=uuid5(
                NAMESPACE_URL,
                f"session-settle:{scope.organization_id}:{scope.orchestration_run_id}:"
                f"{receipt.descriptor.invocation_id}:"
                f"{value.wait_ref.wait_id if value.wait_ref else 'terminal'}",
            ),
            kind="run.waiting"
            if value.status == "DEFERRED"
            else "tool.failed"
            if value.status == "FAILED"
            else "tool.completed",
            invocation=receipt.descriptor,
            payload_ref=value.payload_ref,
            settlement=value,
            causation_id=receipt.requested_event_id,
        )
        try:
            result = await self.service.settle(
                self._actor,
                session_scope(scope),
                original.event,
                _event(draft),
                lease=session_lease(lease),
            )
        except SessionError as exc:
            if str(exc) == "INVOCATION_ALREADY_STARTED":
                raise InvocationStartedError("INVOCATION_ALREADY_STARTED") from None
            raise
        return await self._record(scope, result)
