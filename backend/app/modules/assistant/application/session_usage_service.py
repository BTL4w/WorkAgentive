"""Serial admission/settlement under the durable run row lock.

The effect guard holds this same lock until its caller's transaction commits.
A reservation alone is neither authorization nor an exactly-once effect claim.
"""

import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import UUID

from app.modules.assistant.application.session_service import SessionService, fingerprint
from app.modules.assistant.application.session_usage_ports import (
    UsageRepository,
    UsageTransactionFactory,
)
from app.modules.assistant.domain.session import SessionError, SessionLease, SessionScope
from app.modules.assistant.domain.session_usage import (
    ChildJobKind,
    JobClaim,
    UsageEntry,
    add,
    assistant_job_interval,
    balances,
    subtract,
    validate_vector,
    vector,
)
from app.modules.identity.domain.auth import AuthenticatedActor


class RunUsageService:
    def __init__(self, transactions: UsageTransactionFactory, actor: AuthenticatedActor):
        self.transactions, self.actor = transactions, actor

    @asynccontextmanager
    async def transaction(
        self,
        scope: SessionScope,
        lease: SessionLease | None = None,
        *,
        job: tuple[UUID, JobClaim] | None = None,
    ) -> AsyncGenerator[UsageRepository]:
        if (scope.organization_id, scope.membership_id) != (
            self.actor.organization_id,
            self.actor.membership_id,
        ):
            raise SessionError("SESSION_NOT_FOUND")
        async with self.transactions(self.actor) as tx:
            repo = tx.repository
            if not await repo.authorize(scope, self.actor):
                raise SessionError("SESSION_NOT_FOUND")
            if job is not None:
                await repo.lock_job_actor(scope, self.actor)
            if lease is not None:
                await SessionService.locked(repo, scope, lease)
            elif await repo.state(scope, lock=True) is None:
                raise SessionError("SESSION_NOT_INITIALIZED")
            if job is not None:
                await repo.guard_job(scope, *job)
                await repo.retention_deadline(scope)
            yield repo
            if lease is not None:
                await SessionService.locked(repo, scope, lease)
            if job is not None:
                await repo.guard_job(scope, *job)
            await tx.commit()

    @asynccontextmanager
    async def effect_transaction(
        self, scope: SessionScope, lease: SessionLease, operation_id: UUID
    ) -> AsyncGenerator[UsageRepository]:
        """Fence before execution and before commit, retaining the lock throughout.

        Business services must additionally enforce their own authorization, policy,
        approval, version, idempotency, outbox and audit requirements in this transaction.
        """
        async with self.transaction(scope, lease) as repo:
            await self.guard_effect(repo, scope, lease, operation_id)
            yield repo
            await self.guard_effect(repo, scope, lease, operation_id)

    async def remaining(self, scope: SessionScope) -> tuple[dict[str, int], dict[str, int]]:
        async with self.transaction(scope) as repo:
            return balances(await repo.usage_entries(scope))

    @staticmethod
    def entry(entries: tuple[UsageEntry, ...], operation_id: UUID) -> UsageEntry:
        entry = next((e for e in entries if e.operation_id == operation_id), None)
        if entry is None:
            raise SessionError("BUDGET_OPERATION_NOT_FOUND")
        return entry

    @staticmethod
    async def active(
        repo: UsageRepository,
        entries: tuple[UsageEntry, ...],
        interval_id: UUID | None,
        allocation_id: UUID | None,
        lease: SessionLease | JobClaim,
    ) -> float:
        if interval_id is None:
            raise SessionError("ACTIVE_INTERVAL_REQUIRED")
        entry = RunUsageService.entry(entries, interval_id)
        if (
            entry.kind != "INTERVAL"
            or entry.closed
            or entry.allocation_id != allocation_id
            or entry.fence != lease.fencing_token
            or entry.job_owned != isinstance(lease, JobClaim)
        ):
            raise SessionError("ACTIVE_INTERVAL_INVALID")
        seconds = (
            entry.amount["active_milliseconds"] / 1000
            - ((await repo.now()) - entry.started_at).total_seconds()
        )
        if seconds <= 0:
            raise SessionError("ACTIVE_INTERVAL_EXPIRED")
        return min(seconds, 180)

    async def reserve(
        self,
        scope: SessionScope,
        lease: SessionLease | JobClaim,
        *,
        operation_id: UUID,
        kind: str,
        allocation_id: UUID | None,
        interval_id: UUID | None,
        inputs: int,
        outputs: int,
        retry_of: UUID | None,
        repair_of: UUID | None,
    ) -> tuple[bool, float]:
        command = dict(
            kind=kind,
            allocation_id=allocation_id,
            interval_id=interval_id,
            inputs=inputs,
            outputs=outputs,
            retry_of=retry_of,
            repair_of=repair_of,
        )
        identity = fingerprint(command)
        job = isinstance(lease, JobClaim)
        if job and (allocation_id is None or kind == "HANDOFF"):
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        claim = (allocation_id, lease) if isinstance(lease, JobClaim) and allocation_id else None
        parent = lease if isinstance(lease, SessionLease) else None
        async with self.transaction(scope, parent, job=claim) as repo:
            entries = await repo.usage_entries(scope)
            old = next((e for e in entries if e.operation_id == operation_id), None)
            if old is not None:
                if old.fingerprint != identity or old.job_owned != job:
                    raise SessionError("RESERVATION_CONFLICT")
                if job and old.fence != lease.fencing_token:
                    raise SessionError("WORKFLOW_LEASE_LOST")
                if (
                    parent is not None
                    and kind == "INTERVAL"
                    and allocation_id is None
                    and operation_id == assistant_job_interval(parent.job_id)
                ):
                    state = await SessionService.locked(repo, scope, parent)
                    assert state.lease is not None
                    now = await repo.now()
                    remaining = (
                        old.amount["active_milliseconds"] / 1000
                        - (now - old.started_at).total_seconds()
                    )
                    seconds = min(remaining, (state.lease.expires_at - now).total_seconds())
                    if old.closed or seconds <= 0:
                        raise SessionError("BUDGET_EXHAUSTED")
                    if old.fence != parent.fencing_token:
                        await repo.update_interval_fence(scope, operation_id, parent.fencing_token)
                    return True, float(seconds)
                return True, 1
            if not job and allocation_id is not None and await repo.has_job(scope, allocation_id):
                raise SessionError("ALLOCATION_OWNED_BY_JOB")
            pool = next((e for e in entries if e.kind == "POOL"), None)
            if pool is None:
                raise SessionError("BUDGET_NOT_INITIALIZED")
            amount = vector()
            if (
                kind == "MODEL"
                and type(inputs) is int
                and type(outputs) is int
                and inputs > 0
                and outputs > 0
            ):
                amount.update(
                    root_model_requests=int(allocation_id is None),
                    total_model_requests=1,
                    input_tokens=inputs,
                    output_tokens=outputs,
                )
            elif kind in {"TOOL", "HANDOFF"} and inputs == outputs == 0:
                amount.update(total_tool_calls=1, handoffs=int(kind == "HANDOFF"))
            elif kind == "INTERVAL" and inputs == outputs == 0 and interval_id is None:
                amount.update(segments=1, active_milliseconds=pool.amount["segment_seconds"] * 1000)
            else:
                raise SessionError("BUDGET_RESERVATION_INVALID")
            if (kind != "MODEL" and (retry_of or repair_of)) or (retry_of and repair_of):
                raise SessionError("BUDGET_ATTEMPT_INVALID")
            attempt_parent = retry_of or repair_of
            attempt_kind = "RETRY" if retry_of else "REPAIR" if repair_of else None
            if attempt_parent:
                parent = self.entry(entries, attempt_parent)
                cap = pool.amount["retry_per_invocation" if retry_of else "repair_per_response"]
                if (
                    not cap
                    or parent.kind != "MODEL"
                    or not parent.closed
                    or parent.allocation_id != allocation_id
                    or parent.attempt_parent is not None
                    or any(
                        e.attempt_parent == attempt_parent and e.attempt_kind == attempt_kind
                        for e in entries
                    )
                ):
                    raise SessionError("BUDGET_ATTEMPT_EXHAUSTED")
            seconds = (
                pool.amount["segment_seconds"]
                if kind == "INTERVAL"
                else await self.active(repo, entries, interval_id, allocation_id, lease)
            )
            if allocation_id is None:
                _, available = balances(entries)
            else:
                allocation = self.entry(entries, allocation_id)
                if allocation.kind != "ALLOCATION" or allocation.closed:
                    raise SessionError("ALLOCATION_CLOSED")
                spent = vector()
                for entry in entries:
                    if entry.allocation_id == allocation_id:
                        spent = add(spent, entry.charge)
                available = subtract(allocation.amount, spent)
            subtract(available, amount)
            await repo.insert_usage(
                scope,
                UsageEntry(
                    operation_id,
                    kind,
                    amount,
                    identity,
                    lease.fencing_token,
                    await repo.now(),
                    allocation_id,
                    interval_id,
                    attempt_parent,
                    attempt_kind,
                    job_owned=job,
                ),
            )
            if isinstance(lease, JobClaim):
                assert allocation_id is not None
                deadline = await repo.guard_job(scope, allocation_id, lease)
            else:
                current = await repo.state(scope)
                assert current is not None and current.lease is not None
                deadline = current.lease.expires_at
            seconds = min(seconds, (deadline - await repo.now()).total_seconds())
            if seconds <= 0:
                raise SessionError("LEASE_LOST")
            return False, float(seconds)

    async def allocate(
        self,
        scope: SessionScope,
        lease: SessionLease,
        allocation_id: UUID,
        allowance: dict[str, int],
    ) -> None:
        validate_vector(allowance)
        if allowance["root_model_requests"] or not any(allowance.values()):
            raise SessionError("CHILD_ALLOWANCE_INVALID")
        identity = fingerprint(allowance)
        async with self.transaction(scope, lease) as repo:
            entries = await repo.usage_entries(scope)
            old = next((e for e in entries if e.operation_id == allocation_id), None)
            if old:
                if old.kind != "ALLOCATION" or old.fingerprint != identity:
                    raise SessionError("RESERVATION_CONFLICT")
                return
            _, available = balances(entries)
            subtract(available, allowance)
            await repo.insert_usage(
                scope,
                UsageEntry(
                    allocation_id,
                    "ALLOCATION",
                    allowance,
                    identity,
                    lease.fencing_token,
                    await repo.now(),
                ),
            )

    async def close_child(
        self, scope: SessionScope, lease: SessionLease, allocation_id: UUID
    ) -> None:
        async with self.transaction(scope, lease) as repo:
            if await repo.has_job(scope, allocation_id):
                return
            entries = await repo.usage_entries(scope)
            allocation = self.entry(entries, allocation_id)
            if allocation.kind != "ALLOCATION":
                raise SessionError("ALLOCATION_INVALID")
            if allocation.closed:
                return
            if any(e.allocation_id == allocation_id and not e.closed for e in entries):
                raise SessionError("CHILD_OPERATIONS_PENDING")
            await repo.close_allocation(scope, allocation_id)

    async def bind_job(
        self,
        scope: SessionScope,
        lease: SessionLease,
        allocation_id: UUID,
        kind: ChildJobKind,
        job_id: UUID,
    ) -> None:
        async with self.transaction(scope, lease) as repo:
            await self.bind_job_under_lock(repo, scope, allocation_id, kind, job_id)

    async def close_job(self, scope: SessionScope, allocation_id: UUID, claim: JobClaim) -> None:
        async with self.transaction(scope, job=(allocation_id, claim)) as repo:
            entries = await repo.usage_entries(scope)
            allocation = self.entry(entries, allocation_id)
            if allocation.kind != "ALLOCATION":
                raise SessionError("ALLOCATION_INVALID")
            if allocation.closed:
                return
            # Open operations outside this claim remain charged at their reserved amounts.
            # Closing releases only unspent allowance and cannot authorize stale attempts.
            if any(
                e.allocation_id == allocation_id
                and e.job_owned
                and e.fence == claim.attempt
                and not e.closed
                for e in entries
            ):
                raise SessionError("CHILD_OPERATIONS_PENDING")
            await repo.close_allocation(scope, allocation_id)

    @staticmethod
    async def bind_job_under_lock(
        repo: UsageRepository,
        scope: SessionScope,
        allocation_id: UUID,
        kind: ChildJobKind,
        job_id: UUID,
    ) -> None:
        entry = RunUsageService.entry(await repo.usage_entries(scope), allocation_id)
        if entry.kind != "ALLOCATION" or entry.closed:
            raise SessionError("ALLOCATION_INVALID")
        await repo.bind_job(scope, allocation_id, kind, job_id)

    async def settle(
        self,
        scope: SessionScope,
        lease: SessionLease | JobClaim,
        operation_id: UUID,
        request_fingerprint: str,
        usage: tuple[int, int] | None,
        *,
        allocation_id: UUID | None = None,
    ) -> None:
        if isinstance(lease, JobClaim) and allocation_id is None:
            raise SessionError("BUDGET_SCOPE_MISMATCH")
        claim = (allocation_id, lease) if isinstance(lease, JobClaim) and allocation_id else None
        parent = lease if isinstance(lease, SessionLease) else None
        async with self.transaction(scope, parent, job=claim) as repo:
            entries = await repo.usage_entries(scope)
            entry = self.entry(entries, operation_id)
            if entry.fingerprint != request_fingerprint or entry.kind in {"POOL", "ALLOCATION"}:
                raise SessionError("RESERVATION_CONFLICT")
            if entry.job_owned != isinstance(lease, JobClaim) or (
                isinstance(lease, JobClaim)
                and (entry.allocation_id != allocation_id or entry.fence != lease.attempt)
            ):
                raise SessionError("WORKFLOW_LEASE_LOST")
            if entry.kind != "MODEL" and usage is not None:
                raise SessionError("BUDGET_SETTLEMENT_INVALID")
            identity = fingerprint(usage)
            if entry.closed:
                if entry.settlement_fingerprint != identity:
                    raise SessionError("SETTLEMENT_CONFLICT")
                return
            if entry.kind == "INTERVAL" and any(
                e.interval_id == operation_id and not e.closed for e in entries
            ):
                raise SessionError("INTERVAL_OPERATIONS_PENDING")
            charged = dict(entry.amount)
            if entry.kind == "MODEL" and usage is not None:
                inputs, outputs = usage
                if (
                    type(inputs) is not int
                    or type(outputs) is not int
                    or not (
                        0 <= inputs <= charged["input_tokens"]
                        and 0 <= outputs <= charged["output_tokens"]
                    )
                ):
                    raise SessionError("PROVIDER_USAGE_EXCEEDS_RESERVATION")
                charged.update(input_tokens=inputs, output_tokens=outputs)
            elif entry.kind == "INTERVAL" and entry.fence == lease.fencing_token:
                elapsed = math.ceil(((await repo.now()) - entry.started_at).total_seconds() * 1000)
                charged["active_milliseconds"] = min(
                    charged["active_milliseconds"], max(0, elapsed)
                )
            elif usage is not None:
                raise SessionError("BUDGET_SETTLEMENT_INVALID")
            await repo.settle_usage(
                scope, replace(entry, settled=charged, settlement_fingerprint=identity, closed=True)
            )

    @staticmethod
    async def guard_effect(
        repo: UsageRepository,
        scope: SessionScope,
        lease: SessionLease,
        operation_id: UUID,
        *,
        interval_id: UUID | None = None,
    ) -> None:
        """Call inside the effect's application transaction, before its own policy/write."""
        await SessionService.locked(repo, scope, lease)
        entries = await repo.usage_entries(scope)
        entry = RunUsageService.entry(entries, operation_id)
        if entry.kind not in {"TOOL", "HANDOFF"} or entry.closed:
            raise SessionError("EFFECT_RESERVATION_INVALID")
        history = await repo.events(scope, invocation_id=operation_id)
        requested = next((r for r in history if r.event.kind == "tool.requested"), None)
        if requested is not None:
            started = next((r for r in history if r.event.kind == "tool.started"), None)
            if (
                started is None
                or started.event.correlation_id != lease.lease_id
                or any(
                    r.event.settlement_status is not None or r.event.wait_id is not None
                    for r in history
                )
            ):
                raise SessionError("EFFECT_INVOCATION_INVALID")
            if requested.event.payload_id is None or not await repo.payload_live(
                scope, requested.event.payload_id
            ):
                raise SessionError("CONTEXT_EXPIRED")
        elif interval_id is not None and interval_id != entry.interval_id:
            raise SessionError("EFFECT_INVOCATION_INVALID")
        await RunUsageService.active(
            repo, entries, interval_id or entry.interval_id, entry.allocation_id, lease
        )
