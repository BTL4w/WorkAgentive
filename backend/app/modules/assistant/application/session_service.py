"""Short application-owned transactions for private journaling, leases and payloads."""

import hashlib
import json
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from typing import cast
from uuid import UUID, uuid4

from app.core.config import get_settings
from app.modules.assistant.application.session_ports import (
    SessionRepository,
    SessionTransactionFactory,
)
from app.modules.assistant.domain.session import (
    RuntimeState,
    SessionError,
    SessionEvent,
    SessionLease,
    SessionPayload,
    SessionPins,
    SessionScope,
    StoredSessionEvent,
)
from app.modules.assistant.domain.session_usage import limits_from_checkpoint
from app.modules.identity.domain.auth import AuthenticatedActor


def canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class SessionService:
    def __init__(self, transactions: SessionTransactionFactory):
        self._transactions = transactions

    @asynccontextmanager
    async def _transaction(
        self, actor: AuthenticatedActor, scope: SessionScope
    ) -> AsyncGenerator[SessionRepository]:
        if (scope.organization_id, scope.membership_id) != (
            actor.organization_id,
            actor.membership_id,
        ):
            raise SessionError("SESSION_NOT_FOUND")
        async with self._transactions(actor) as tx:
            if not await tx.repository.authorize(scope, actor):
                raise SessionError("SESSION_NOT_FOUND")
            yield tx.repository
            await tx.commit()

    @staticmethod
    async def locked(
        repo: SessionRepository, scope: SessionScope, lease: SessionLease
    ) -> RuntimeState:
        state = await repo.state(scope, lock=True)
        if state is None:
            raise SessionError("SESSION_NOT_INITIALIZED")
        current = state.lease
        if (
            lease.scope != scope
            or current is None
            or (current.lease_id, current.job_id, current.worker_id, current.fencing_token)
            != (lease.lease_id, lease.job_id, lease.worker_id, lease.fencing_token)
            or current.expires_at <= await repo.now()
        ):
            raise SessionError("LEASE_LOST")
        await repo.retention_deadline(scope)
        return state

    @staticmethod
    async def _payload(
        repo: SessionRepository, scope: SessionScope, payload_id: UUID, *, live: bool = True
    ) -> SessionPayload:
        value = await repo.payload(scope, payload_id)
        if value is None:
            raise SessionError("SESSION_NOT_FOUND")
        # JSON null is valid content. Expiry/purge provenance, not body=None,
        # determines whether a payload may be replayed.
        if live and not await repo.payload_live(scope, payload_id):
            raise SessionError("CONTEXT_EXPIRED")
        return value

    @staticmethod
    async def _put(
        repo: SessionRepository,
        scope: SessionScope,
        body: object,
        kind: str,
        ttl_seconds: int | None,
    ) -> SessionPayload:
        if kind not in {
            "OBJECTIVE",
            "CONTEXT",
            "TOOL_ARGUMENTS",
            "TOOL_RESULT",
            "MODEL_OUTPUT",
            "SKILL_BODY",
            "WORKING_PLAN",
            "CHECKPOINT",
        }:
            raise SessionError("PAYLOAD_KIND_INVALID")
        # No default=str for context: reject non-JSON objects and non-finite values.
        try:
            encoded = json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
        except (TypeError, ValueError, RecursionError):
            raise SessionError("PAYLOAD_INVALID") from None
        if len(encoded.encode()) > 65536:
            raise SessionError("PAYLOAD_TOO_LARGE")
        now = await repo.now()
        seconds = get_settings().ai_raw_context_retention_days * 86400
        if ttl_seconds is not None:
            if type(ttl_seconds) is not int or not 0 < ttl_seconds <= seconds:
                raise SessionError("PAYLOAD_TTL_INVALID")
            seconds = ttl_seconds
        deadline = min(now + timedelta(seconds=seconds), await repo.retention_deadline(scope))
        if deadline <= now:
            raise SessionError("CONTEXT_EXPIRED")
        value = SessionPayload(
            uuid4(),
            scope.organization_id,
            scope.orchestration_run_id,
            kind,
            "RAW_CONTEXT",
            json.loads(encoded),
            hashlib.sha256(encoded.encode()).hexdigest(),
            now,
            deadline,
        )
        await repo.put_payload(value)
        return value

    @staticmethod
    def _checkpoint_body(
        body: object, scope: SessionScope, pins: SessionPins, sequence: int
    ) -> None:
        if not isinstance(body, dict):
            raise SessionError("CHECKPOINT_INVALID")
        mapping = cast(dict[str, object], body)
        if set(mapping) - {
            "schema_version",
            "scope",
            "driver_version",
            "prompt_version",
            "catalog_fingerprint",
            "agent_version",
            "manifest_fingerprint",
            "model_ref",
            "tool_versions",
            "skill_versions",
            "sequence",
            "segment_id",
            "pending_invocation_id",
            "pending_wait",
            "objective_ref",
            "working_plan_ref",
            "payload_refs",
            "limits",
            "usage",
        }:
            raise SessionError("CHECKPOINT_INVALID")
        if (
            mapping.get("scope") != {key: str(value) for key, value in asdict(scope).items()}
            or mapping.get("sequence") != sequence
        ):
            raise SessionError("CHECKPOINT_SCOPE_MISMATCH")
        for name in (
            "schema_version",
            "driver_version",
            "prompt_version",
            "catalog_fingerprint",
            "agent_version",
            "manifest_fingerprint",
            "model_ref",
        ):
            if mapping.get(name) != getattr(pins, name):
                raise SessionError("PIN_CONFLICT")

    async def initialize(
        self, actor: AuthenticatedActor, scope: SessionScope, pins: SessionPins, body: object
    ) -> None:
        self._checkpoint_body(body, scope, pins, 0)
        limits = limits_from_checkpoint(body)
        async with self._transaction(actor, scope) as repo:
            await repo.authorize(scope, actor, lock=True)
            existing = await repo.state(scope, lock=True)
            if existing is not None:
                if existing.pins != pins:
                    raise SessionError("PIN_CONFLICT")
                return
            state = RuntimeState(scope, pins)
            await repo.create_state(state)
            await repo.initialize_budget(scope, limits)
            payload = await self._put(repo, scope, body, "CHECKPOINT", None)
            await repo.save_state(replace(state, checkpoint_payload_id=payload.id))

    async def claim(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        *,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> SessionLease:
        self._lease_options(worker_id, lease_seconds)
        async with self._transaction(actor, scope) as repo:
            state = await repo.state(scope, lock=True)
            if state is None:
                raise SessionError("SESSION_NOT_INITIALIZED")
            if not await repo.job_matches(scope, job_id):
                raise SessionError("SESSION_NOT_FOUND")
            now = await repo.now()
            if state.lease is not None and state.lease.expires_at > now:
                if (state.lease.job_id, state.lease.worker_id) != (job_id, worker_id):
                    raise SessionError("LEASE_BUSY")
                return state.lease
            lease = SessionLease(
                scope,
                uuid4(),
                job_id,
                worker_id,
                state.fence + 1,
                now + timedelta(seconds=lease_seconds),
            )
            await repo.save_state(replace(state, lease=lease, fence=lease.fencing_token))
            return lease

    @staticmethod
    def _lease_options(worker_id: str, seconds: int) -> None:
        if (
            not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", worker_id)
            or type(seconds) is not int
            or not 1 <= seconds <= 180
        ):
            raise SessionError("LEASE_OPTIONS_INVALID")

    async def renew(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        lease: SessionLease,
        *,
        lease_seconds: int,
    ) -> SessionLease:
        self._lease_options(lease.worker_id, lease_seconds)
        async with self._transaction(actor, scope) as repo:
            state = await self.locked(repo, scope, lease)
            renewed = replace(lease, expires_at=await repo.now() + timedelta(seconds=lease_seconds))
            await repo.save_state(replace(state, lease=renewed))
            return renewed

    async def put_payload(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        body: object,
        *,
        kind: str,
        lease: SessionLease,
        ttl_seconds: int | None = None,
    ) -> SessionPayload:
        async with self._transaction(actor, scope) as repo:
            await self.locked(repo, scope, lease)
            return await self._put(repo, scope, body, kind, ttl_seconds)

    async def payload(
        self, actor: AuthenticatedActor, scope: SessionScope, payload_id: UUID, *, live: bool = True
    ) -> SessionPayload:
        async with self._transaction(actor, scope) as repo:
            value = await self._payload(repo, scope, payload_id, live=live)
            return value if live else replace(value, body=None)

    async def load_checkpoint(
        self, actor: AuthenticatedActor, scope: SessionScope
    ) -> SessionPayload | None:
        async with self._transaction(actor, scope) as repo:
            state = await repo.state(scope)
            if state is None or state.checkpoint_payload_id is None:
                return None
            return await self._payload(repo, scope, state.checkpoint_payload_id)

    async def checkpoint(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        pins: SessionPins,
        body: object,
        sequence: int,
        *,
        lease: SessionLease,
        references: tuple[UUID, ...],
    ) -> None:
        self._checkpoint_body(body, scope, pins, sequence)
        async with self._transaction(actor, scope) as repo:
            state = await self.locked(repo, scope, lease)
            if pins != state.pins:
                raise SessionError("PIN_CONFLICT")
            if sequence != state.sequence or sequence < state.checkpoint_sequence:
                raise SessionError("SEQUENCE_CONFLICT")
            for reference in references:
                await self._payload(repo, scope, reference)
            if state.checkpoint_payload_id is not None and state.checkpoint_sequence == sequence:
                current = await self._payload(repo, scope, state.checkpoint_payload_id)
                if current.fingerprint == fingerprint(body):
                    return
            payload = await self._put(repo, scope, body, "CHECKPOINT", None)
            await repo.save_state(
                replace(state, checkpoint_payload_id=payload.id, checkpoint_sequence=sequence)
            )

    @staticmethod
    async def append_locked(
        repo: SessionRepository,
        state: RuntimeState,
        event: SessionEvent,
        expected_sequence: int,
    ) -> StoredSessionEvent:
        digest = fingerprint(asdict(event))
        existing = await repo.event(state.scope, event.event_id)
        if existing is not None:
            if digest != existing.fingerprint:
                raise SessionError("EVENT_ID_CONFLICT")
            return existing
        if type(expected_sequence) is not int or expected_sequence != state.sequence:
            raise SessionError("SEQUENCE_CONFLICT")
        if event.payload_id is not None:
            payload = await SessionService._payload(repo, state.scope, event.payload_id)
            if event.kind == "tool.requested" and (
                payload.kind != "TOOL_ARGUMENTS"
                or payload.fingerprint != event.arguments_fingerprint
            ):
                raise SessionError("INVOCATION_CONFLICT")
        if event.kind == "tool.requested":
            previous = await repo.events(state.scope, invocation_id=event.invocation_id)
            if previous:
                raise SessionError("INVOCATION_CONFLICT")
        value = StoredSessionEvent(state.scope, state.sequence + 1, await repo.now(), event, digest)
        await repo.insert_event(value)
        await repo.save_state(replace(state, sequence=value.sequence))
        return value

    async def append(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        event: SessionEvent,
        *,
        expected_sequence: int,
        lease: SessionLease,
    ) -> StoredSessionEvent:
        if event.settlement_status is not None or event.wait_id is not None:
            raise SessionError("USE_SETTLE_INVOCATION")
        async with self._transaction(actor, scope) as repo:
            state = await self.locked(repo, scope, lease)
            return await self.append_locked(repo, state, event, expected_sequence)

    async def start_invocation(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        requested: SessionEvent,
        event: SessionEvent,
        *,
        lease: SessionLease,
    ) -> bool:
        """One dispatch owner even for two executors sharing the current lease."""
        async with self._transaction(actor, scope) as repo:
            state = await self.locked(repo, scope, lease)
            history = await repo.events(scope, invocation_id=requested.invocation_id)
            original = next((r for r in history if r.event.kind == "tool.requested"), None)
            if (
                original is None
                or original.event != requested
                or event.kind != "tool.started"
                or event.invocation_id != requested.invocation_id
            ):
                raise SessionError("INVOCATION_CONFLICT")
            if any(
                r.event.kind == "tool.started"
                or r.event.settlement_status is not None
                or r.event.wait_id is not None
                for r in history
            ):
                return False
            await self.append_locked(repo, state, event, state.sequence)
            return True

    async def read(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        *,
        after_sequence: int,
        invocation_id: UUID | None = None,
    ) -> tuple[StoredSessionEvent, ...]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise SessionError("SEQUENCE_INVALID")
        async with self._transaction(actor, scope) as repo:
            return await repo.events(scope, after=after_sequence, invocation_id=invocation_id)

    async def settle(
        self,
        actor: AuthenticatedActor,
        scope: SessionScope,
        requested: SessionEvent,
        event: SessionEvent,
        *,
        lease: SessionLease,
    ) -> StoredSessionEvent:
        async with self._transaction(actor, scope) as repo:
            state = await self.locked(repo, scope, lease)
            if (
                requested.invocation_id is None
                or event.invocation_id != requested.invocation_id
                or (event.settlement_status is None and event.wait_id is None)
            ):
                raise SessionError("INVOCATION_CONFLICT")
            history = await repo.events(scope, invocation_id=requested.invocation_id)
            original = next((r for r in history if r.event.kind == "tool.requested"), None)
            if original is None or original.event != requested:
                raise SessionError("INVOCATION_CONFLICT")
            if event.settlement_status == "REJECTED" and any(
                r.event.kind == "tool.started" for r in history
            ):
                raise SessionError("INVOCATION_ALREADY_STARTED")
            existing = next(
                (
                    r
                    for r in reversed(history)
                    if r.event.settlement_status is not None or r.event.wait_id is not None
                ),
                None,
            )
            if existing is not None:
                if existing.event == event:
                    return existing
                if existing.event.settlement_status is not None or event.wait_id is not None:
                    raise SessionError("INVOCATION_CONFLICT")
            return await self.append_locked(repo, state, event, state.sequence)
