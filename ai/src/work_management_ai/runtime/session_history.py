"""Ephemeral model projection from canonical events; never a prompt/memory store.

Only explicit visible payload schemas are read. System prompts are rebuilt by
an approved prompt loader at request time. Incomplete calls are recovery state,
not fabricated observations or provider-ready orphan calls.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Protocol, TypedDict
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from work_management_ai.model_gateway.tool_contracts import ToolCall, ToolItem, validate_history
from work_management_ai.runtime.contracts import ContextReference, JsonValue
from work_management_ai.runtime.session_log import InvocationDescriptor, SessionRecord
from work_management_ai.runtime.tool_loop_contracts import (
    PayloadReference,
    RunScope,
    ToolLoopCheckpoint,
    VersionPin,
    WaitReference,
)


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", allow_inf_nan=False
    )


type HistoryReference = PayloadReference | ContextReference
type HistoryStatus = Literal[
    "READY",
    "WAITING",
    "RECONCILE",
    "CONTEXT_EXPIRED",
    "CONTEXT_REVOKED",
    "CONTEXT_STALE",
    "CONTEXT_UNAVAILABLE",
    "HISTORY_LIMIT",
    "HISTORY_INVALID",
]


class RevalidatedContext(_Contract):
    allowed: bool = Field(strict=True)
    references: tuple[HistoryReference, ...]
    fresh: bool = Field(default=True, strict=True)


class ContextAccessPort(Protocol):
    async def revalidate(
        self, scope: RunScope, refs: tuple[HistoryReference, ...]
    ) -> RevalidatedContext: ...


class RevalidatedSkills(_Contract):
    allowed: bool = Field(strict=True)
    pins: tuple[VersionPin, ...]
    fresh: bool = Field(default=True, strict=True)


class SkillPinAccessPort(Protocol):
    async def revalidate_skills(
        self, scope: RunScope, pins: tuple[VersionPin, ...]
    ) -> RevalidatedSkills: ...


class HistoryPayloadPort(Protocol):
    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> object: ...


class HistoryLimits(_Contract):
    max_input_bytes: int = Field(default=24000, ge=128, le=48000, strict=True)
    max_items: int = Field(default=48, ge=1, le=64, strict=True)
    max_records: int = Field(default=256, ge=1, le=512, strict=True)
    max_references: int = Field(default=64, ge=1, le=128, strict=True)
    max_prior_turns: int = Field(default=2, ge=0, le=4, strict=True)


class MessagePayload(_Contract):
    kind: Literal["MESSAGE"] = "MESSAGE"
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=32000)
    references: tuple[ContextReference, ...] = ()


class ToolArgumentsPayload(_Contract):
    kind: Literal["TOOL_ARGUMENTS"] = "TOOL_ARGUMENTS"
    arguments: dict[str, JsonValue]
    references: tuple[ContextReference, ...] = ()
    policy_version: str | None = Field(default=None, min_length=1, max_length=128)
    tool_fingerprint: str | None = Field(default=None, min_length=1, max_length=128)


class ToolResultPayload(_Contract):
    kind: Literal["TOOL_RESULT"] = "TOOL_RESULT"
    output: JsonValue
    references: tuple[ContextReference, ...] = ()
    loaded_skill: VersionPin | None = None


class ContextPayload(_Contract):
    kind: Literal["CONTEXT"] = "CONTEXT"
    category: Literal["CONTEXT", "EXPLORATION_REPORT", "WORKING_PLAN"] = "CONTEXT"
    content: JsonValue
    references: tuple[ContextReference, ...] = ()


class PriorTurn(_Contract):
    """Application-selected relevant completed turn; no copied transcript."""

    scope: RunScope
    request_ref: PayloadReference
    response_ref: PayloadReference


class HistoryGap(_Contract):
    reason: HistoryStatus
    payload_ref: PayloadReference | None = None


class ModelHistory(_Contract):
    status: HistoryStatus
    items: tuple[ToolItem, ...] = ()
    catalog_fingerprint: str
    prompt_version: str
    tool_versions: tuple[VersionPin, ...] = ()
    loaded_skills: tuple[VersionPin, ...] = ()
    references: tuple[PayloadReference, ...] = ()
    evidence: tuple[ContextReference, ...] = ()
    gaps: tuple[HistoryGap, ...] = ()
    pending_call: InvocationDescriptor | None = None
    pending_wait: WaitReference | None = None


class _HistoryPins(TypedDict):
    catalog_fingerprint: str
    prompt_version: str
    tool_versions: tuple[VersionPin, ...]


class _Stop(Exception):
    def __init__(self, code: HistoryStatus, ref: PayloadReference | None = None):
        self.code: HistoryStatus = code
        self.ref = ref


def _visible_value(value: JsonValue) -> None:
    if isinstance(value, dict):
        if set(value) & {
            "thought",
            "hidden_reasoning",
            "chain_of_thought",
            "system_prompt",
            "raw_provider_response",
        }:
            raise _Stop("HISTORY_INVALID")
        for item in value.values():
            _visible_value(item)
    elif isinstance(value, list):
        for item in value:
            _visible_value(item)


class SessionHistory:
    def __init__(self, payloads: HistoryPayloadPort, *, now: Callable[[], datetime] | None = None):
        self.payloads = payloads
        self.now = now or (lambda: datetime.now(UTC))

    @staticmethod
    async def _skill_access(
        scope: RunScope, pins: tuple[VersionPin, ...], access: SkillPinAccessPort | None
    ) -> None:
        if not pins:
            return
        if access is None:
            raise _Stop("CONTEXT_UNAVAILABLE")
        try:
            verdict = RevalidatedSkills.model_validate(await access.revalidate_skills(scope, pins))
        except Exception:
            raise _Stop("CONTEXT_UNAVAILABLE") from None
        if verdict.pins != pins:
            raise _Stop("HISTORY_INVALID")
        if not verdict.allowed:
            raise _Stop("CONTEXT_REVOKED")
        if not verdict.fresh:
            raise _Stop("CONTEXT_STALE")

    @staticmethod
    async def _access(
        scope: RunScope, refs: tuple[HistoryReference, ...], access: ContextAccessPort
    ) -> None:
        if any(ref.organization_id != scope.organization_id for ref in refs):
            raise _Stop("HISTORY_INVALID")
        try:
            verdict = RevalidatedContext.model_validate(await access.revalidate(scope, refs))
        except Exception:
            raise _Stop("CONTEXT_UNAVAILABLE") from None
        if verdict.references != refs:
            raise _Stop("HISTORY_INVALID")
        if not verdict.allowed:
            raise _Stop("CONTEXT_REVOKED")
        if not verdict.fresh:
            raise _Stop("CONTEXT_STALE")

    async def _read[T: BaseModel](
        self, scope: RunScope, ref: PayloadReference, schema: type[T], access: ContextAccessPort
    ) -> T:
        await self._access(scope, (ref,), access)
        if ref.expires_at <= self.now():
            raise _Stop("CONTEXT_EXPIRED", ref)
        try:
            body = await self.payloads.get_payload(scope, ref)
            payload = schema.model_validate(body)
            if isinstance(payload, ToolArgumentsPayload):
                _visible_value(payload.arguments)
            elif isinstance(payload, ToolResultPayload):
                _visible_value(payload.output)
            elif isinstance(payload, ContextPayload):
                _visible_value(payload.content)
        except _Stop:
            raise
        except ValueError as exc:
            raise _Stop(
                "CONTEXT_EXPIRED" if str(exc) == "CONTEXT_EXPIRED" else "HISTORY_INVALID", ref
            ) from None
        except Exception:
            raise _Stop("CONTEXT_UNAVAILABLE", ref) from None
        return payload

    async def project(
        self,
        scope: RunScope,
        records: tuple[SessionRecord, ...],
        *,
        access: ContextAccessPort,
        limits: HistoryLimits,
        checkpoint: ToolLoopCheckpoint,
        prior_turns: tuple[PriorTurn, ...] = (),
        skill_access: SkillPinAccessPort | None = None,
    ) -> ModelHistory:
        """Return READY only when every retained pair/payload passed current gates."""
        checkpoint = ToolLoopCheckpoint.model_validate(checkpoint)
        limits = HistoryLimits.model_validate(limits)
        refs: list[PayloadReference] = []
        evidence: list[ContextReference] = []
        skills: list[VersionPin] = []
        gaps: list[HistoryGap] = []
        pending_call: InvocationDescriptor | None = None
        pending_wait = checkpoint.pending_wait
        common = _HistoryPins(
            catalog_fingerprint=checkpoint.catalog_fingerprint,
            prompt_version=checkpoint.prompt_version,
            tool_versions=checkpoint.tool_versions,
        )
        units: list[tuple[int, tuple[ToolItem, ...]]] = []
        try:
            if checkpoint.scope != scope or len(records) > limits.max_records:
                raise _Stop("HISTORY_INVALID" if checkpoint.scope != scope else "HISTORY_LIMIT")
            ordered: list[SessionRecord] = []
            seen_events: dict[UUID, SessionRecord] = {}
            for record in records:
                record = SessionRecord.model_validate(record)
                if record.scope != scope:
                    raise _Stop("HISTORY_INVALID")
                previous = seen_events.get(record.event_id)
                if previous is not None:
                    if previous != record:
                        raise _Stop("HISTORY_INVALID")
                    continue
                if record.sequence != len(ordered) + 1:
                    raise _Stop("HISTORY_INVALID")
                ordered.append(record)
                seen_events[record.event_id] = record
            if checkpoint.sequence > len(ordered):
                raise _Stop("HISTORY_INVALID")
            # Child headers remain in the ordered recovery stream, but child
            # transcript bodies must never become the root model conversation.
            ordered = [record for record in ordered if record.agent_run_id is None]
            requests: dict[UUID, SessionRecord] = {}
            terminals: dict[UUID, SessionRecord] = {}
            waits: dict[UUID, SessionRecord] = {}
            call_ids: set[str] = set()
            resumed_waits: set[UUID] = set()
            for record in ordered:
                if (
                    record.kind == "continuation.received"
                    and record.correlation_id is not None
                    and record.causation_id is not None
                    and record.event_id == uuid5(record.correlation_id, "canonical-completion")
                ):
                    resumed_waits.add(record.correlation_id)
                desc = record.invocation
                if record.kind == "tool.requested":
                    assert desc is not None
                    if desc.invocation_id in requests or desc.call_id in call_ids:
                        raise _Stop("HISTORY_INVALID")
                    if not any(
                        p.reference == desc.tool_reference and p.version == desc.tool_version
                        for p in checkpoint.tool_versions
                    ):
                        raise _Stop("HISTORY_INVALID")
                    requests[desc.invocation_id] = record
                    call_ids.add(desc.call_id)
                elif record.settlement is not None:
                    assert desc is not None
                    original = requests.get(desc.invocation_id)
                    if (
                        original is None
                        or original.invocation != desc
                        or desc.invocation_id in terminals
                    ):
                        raise _Stop("HISTORY_INVALID")
                    if record.settlement.wait_ref is not None:
                        wait = record.settlement.wait_ref
                        if wait.scope != scope or wait.source_invocation_id != desc.invocation_id:
                            raise _Stop("HISTORY_INVALID")
                        earlier = waits.get(desc.invocation_id)
                        if earlier is not None:
                            assert earlier.settlement is not None
                            old_wait = earlier.settlement.wait_ref
                            if (
                                old_wait is None
                                or old_wait.wait_id not in resumed_waits
                                or old_wait.wait_id == wait.wait_id
                            ):
                                raise _Stop("HISTORY_INVALID")
                            if pending_wait == old_wait:
                                pending_wait = wait
                        waits[desc.invocation_id] = record
                    else:
                        terminals[desc.invocation_id] = record
            # Recover pending identity before touching expiring bodies. Even a
            # stopped projection must retain its wait and active arguments ref.
            if pending_wait is not None and pending_wait.source_invocation_id in terminals:
                pending_wait = None
            for invocation_id, requested in requests.items():
                if invocation_id in terminals:
                    continue
                if pending_call is not None:
                    raise _Stop("HISTORY_INVALID")
                assert requested.invocation is not None and requested.payload_ref is not None
                if requested.payload_ref.organization_id != scope.organization_id:
                    raise _Stop("HISTORY_INVALID")
                pending_call = requested.invocation
                refs.append(requested.payload_ref)
                waiting = waits.get(invocation_id)
                if waiting is not None:
                    assert waiting.settlement is not None
                    logged_wait = waiting.settlement.wait_ref
                    if pending_wait is not None and pending_wait != logged_wait:
                        raise _Stop("HISTORY_INVALID")
                    pending_wait = logged_wait
            await self._skill_access(scope, checkpoint.skill_versions, skill_access)
            # context.selected is a complete selected-context snapshot, not a
            # delta. Superseded snapshots are never fetched (even after expiry).
            latest_context = max(
                (r.sequence for r in ordered if r.kind == "context.selected"), default=-1
            )
            message_refs: set[UUID] = set()
            for record in ordered:
                ref = record.payload_ref
                if ref is None:
                    continue
                if record.kind in {
                    "turn.started",
                    "continuation.received",
                    "model.completed",
                    "turn.completed",
                }:
                    if ref.payload_id in message_refs:
                        continue
                    message = await self._read(scope, ref, MessagePayload, access)
                    expected = (
                        "assistant"
                        if record.kind in {"model.completed", "turn.completed"}
                        else "user"
                    )
                    if message.role != expected:
                        raise _Stop("HISTORY_INVALID", ref)
                    refs.append(ref)
                    evidence.extend(message.references)
                    units.append(
                        (
                            record.sequence,
                            (ToolItem(kind="message", role=message.role, content=message.text),),
                        )
                    )
                    message_refs.add(ref.payload_id)
                elif record.kind == "context.selected" and record.sequence == latest_context:
                    payload = await self._read(scope, ref, ContextPayload, access)
                    refs.append(ref)
                    evidence.extend(payload.references)
                    text = json.dumps(
                        dict(kind=payload.category, content=payload.content), ensure_ascii=False
                    )
                    units.append(
                        (record.sequence, (ToolItem(kind="message", role="user", content=text),))
                    )
            if (
                checkpoint.objective_ref is not None
                and checkpoint.objective_ref.payload_id not in message_refs
            ):
                objective = await self._read(
                    scope, checkpoint.objective_ref, MessagePayload, access
                )
                if objective.role != "user":
                    raise _Stop("HISTORY_INVALID", checkpoint.objective_ref)
                refs.append(checkpoint.objective_ref)
                evidence.extend(objective.references)
                units.append((0, (ToolItem(kind="message", role="user", content=objective.text),)))
            if checkpoint.working_plan_ref is not None:
                plan = await self._read(scope, checkpoint.working_plan_ref, ContextPayload, access)
                if plan.category != "WORKING_PLAN":
                    raise _Stop("HISTORY_INVALID", checkpoint.working_plan_ref)
                refs.append(checkpoint.working_plan_ref)
                evidence.extend(plan.references)
                units.append(
                    (
                        len(records) + 1,
                        (
                            ToolItem(
                                kind="message",
                                role="user",
                                content=json.dumps(
                                    {"kind": "WORKING_PLAN", "content": plan.content}
                                ),
                            ),
                        ),
                    )
                )
            for invocation_id, requested in requests.items():
                desc = requested.invocation
                args_ref = requested.payload_ref
                assert desc is not None and args_ref is not None
                args = await self._read(scope, args_ref, ToolArgumentsPayload, access)
                refs.append(args_ref)
                evidence.extend(args.references)
                terminal = terminals.get(invocation_id)
                if terminal is None:
                    continue
                settlement = terminal.settlement
                assert settlement is not None
                output: JsonValue
                if settlement.payload_ref is not None:
                    result = await self._read(
                        scope, settlement.payload_ref, ToolResultPayload, access
                    )
                    refs.append(settlement.payload_ref)
                    evidence.extend(result.references)
                    output = result.output
                    if result.loaded_skill is not None:
                        if result.loaded_skill not in checkpoint.skill_versions:
                            raise _Stop("HISTORY_INVALID")
                        skills.append(result.loaded_skill)
                else:
                    output = {"status": settlement.status, "safe_code": settlement.safe_code}
                units.append(
                    (
                        requested.sequence,
                        (
                            ToolItem(
                                kind="tool_call",
                                call=ToolCall(
                                    call_id=desc.call_id,
                                    name=desc.tool_reference.rsplit("@", 1)[0].replace(".", "__"),
                                    arguments=args.arguments,
                                ),
                            ),
                            ToolItem(kind="tool_result", call_id=desc.call_id, content=output),
                        ),
                    )
                )
            if (
                checkpoint.pending_invocation_id is not None
                and checkpoint.pending_invocation_id not in requests
            ):
                raise _Stop("HISTORY_INVALID")
            refs = list(dict.fromkeys(refs))
            unique_evidence: dict[UUID, ContextReference] = {}
            for ref in evidence:
                if ref.reference_id in unique_evidence and unique_evidence[ref.reference_id] != ref:
                    raise _Stop("HISTORY_INVALID")
                unique_evidence[ref.reference_id] = ref
            evidence = list(unique_evidence.values())
            if len(refs) + len(evidence) > limits.max_references:
                raise _Stop("HISTORY_LIMIT")
            await self._access(scope, (*refs, *evidence), access)
            items = tuple(item for _, group in sorted(units, key=lambda u: u[0]) for item in group)
            if (
                len(items) > limits.max_items
                or len(
                    json.dumps(
                        [i.model_dump(mode="json") for i in items], ensure_ascii=True
                    ).encode()
                )
                > limits.max_input_bytes
            ):
                raise _Stop("HISTORY_LIMIT")
            # Prior turns are optional, application-selected completed snapshots.
            # Keep whole user/final pairs, newest first for admission, oldest
            # first in the resulting request. Never fetch another tenant/owner.
            current_refs, current_evidence = tuple(refs), tuple(evidence)
            optional: list[tuple[ToolItem, ...]] = []
            retained_prior: list[tuple[RunScope, tuple[HistoryReference, ...]]] = []
            selected = prior_turns[-limits.max_prior_turns :] if limits.max_prior_turns else ()
            for prior in reversed(selected):
                if (
                    (
                        prior.scope.organization_id,
                        prior.scope.membership_id,
                        prior.scope.conversation_id,
                    )
                    != (scope.organization_id, scope.membership_id, scope.conversation_id)
                    or prior.scope.turn_id == scope.turn_id
                    or prior.scope.orchestration_run_id == scope.orchestration_run_id
                ):
                    raise _Stop("HISTORY_INVALID")
                try:
                    user = await self._read(prior.scope, prior.request_ref, MessagePayload, access)
                    final = await self._read(
                        prior.scope, prior.response_ref, MessagePayload, access
                    )
                    if user.role != "user" or final.role != "assistant":
                        raise _Stop("HISTORY_INVALID")
                    await self._access(prior.scope, (*user.references, *final.references), access)
                except _Stop as stop:
                    gaps.append(HistoryGap(reason=stop.code, payload_ref=stop.ref))
                    continue
                pair = (
                    ToolItem(kind="message", role="user", content=user.text),
                    ToolItem(kind="message", role="assistant", content=final.text),
                )
                candidate = (
                    *pair,
                    *(item for group in reversed(optional) for item in group),
                    *items,
                )
                candidate_refs = (*refs, prior.request_ref, prior.response_ref)
                candidate_evidence = (*evidence, *user.references, *final.references)
                if (
                    len(candidate) > limits.max_items
                    or len(
                        json.dumps(
                            [i.model_dump(mode="json") for i in candidate], ensure_ascii=True
                        ).encode()
                    )
                    > limits.max_input_bytes
                    or len(candidate_refs) + len(candidate_evidence) > limits.max_references
                ):
                    gaps.append(HistoryGap(reason="HISTORY_LIMIT"))
                    continue
                optional.append(pair)
                retained_prior.append(
                    (
                        prior.scope,
                        (
                            prior.request_ref,
                            prior.response_ref,
                            *user.references,
                            *final.references,
                        ),
                    )
                )
                refs.extend((prior.request_ref, prior.response_ref))
                evidence.extend((*user.references, *final.references))
            items = (*(item for group in reversed(optional) for item in group), *items)
            await self._access(scope, (*current_refs, *current_evidence), access)
            for prior_scope, prior_refs in retained_prior:
                await self._access(prior_scope, prior_refs, access)
            for ref in refs:
                if ref.expires_at <= self.now():
                    raise _Stop("CONTEXT_EXPIRED", ref)
            await self._skill_access(scope, checkpoint.skill_versions, skill_access)
            validate_history(items)
            if pending_wait is not None and pending_wait.source_invocation_id in terminals:
                pending_wait = None
            return ModelHistory(
                status="WAITING"
                if pending_wait is not None
                else "RECONCILE"
                if pending_call
                else "READY",
                items=items,
                loaded_skills=tuple(dict.fromkeys(skills)),
                references=tuple(refs),
                evidence=tuple(evidence),
                gaps=tuple(gaps),
                pending_call=pending_call,
                pending_wait=pending_wait,
                **common,
            )
        except _Stop as stop:
            gaps.append(HistoryGap(reason=stop.code, payload_ref=stop.ref))
            return ModelHistory(
                status=stop.code,
                references=tuple(refs[: limits.max_references]),
                gaps=tuple(gaps),
                pending_call=pending_call,
                pending_wait=pending_wait,
                **common,
            )
        except (ValidationError, ValueError, AssertionError):
            return ModelHistory(
                status="HISTORY_INVALID",
                gaps=(HistoryGap(reason="HISTORY_INVALID"),),
                pending_wait=checkpoint.pending_wait,
                **common,
            )
