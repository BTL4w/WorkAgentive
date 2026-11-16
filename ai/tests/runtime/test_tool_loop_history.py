"""Replay canonical pairs without tools, hidden reasoning or stale authority."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from work_management_ai.runtime.session_history import (
    ContextAccessPort,
    HistoryLimits,
    HistoryReference,
    RevalidatedContext,
    RevalidatedSkills,
    SessionHistory,
)
from work_management_ai.runtime.session_log import (
    InvocationDescriptor,
    SessionRecord,
    SettledInvocation,
)
from work_management_ai.runtime.tool_loop_contracts import (
    PayloadReference,
    RunScope,
    ToolLoopCheckpoint,
    VersionPin,
    WaitReference,
)


class Payloads:
    def __init__(self) -> None:
        self.values: dict[UUID, object] = {}
        self.expired: set[UUID] = set()

    async def get_payload(self, scope: RunScope, reference: PayloadReference) -> object:
        if reference.payload_id in self.expired:
            raise ValueError("CONTEXT_EXPIRED")
        return self.values[reference.payload_id]


class Access:
    def __init__(self) -> None:
        self.revoked = False
        self.calls = 0

    async def revalidate(
        self, scope: RunScope, refs: tuple[HistoryReference, ...]
    ) -> RevalidatedContext:
        self.calls += 1
        return RevalidatedContext(allowed=not self.revoked, references=refs)

    async def revalidate_skills(
        self, scope: RunScope, pins: tuple[VersionPin, ...]
    ) -> RevalidatedSkills:
        return RevalidatedSkills(allowed=not self.revoked, pins=pins)


class Fixture:
    def __init__(self) -> None:
        self.scope = RunScope(**{key: uuid4() for key in RunScope.model_fields})
        self.now = datetime.now(UTC)
        self.payloads, self.access = Payloads(), Access()
        self.checkpoint = ToolLoopCheckpoint(
            scope=self.scope,
            driver_version="2.0.0",
            prompt_version="tool-loop.v1",
            catalog_fingerprint="catalog-v1",
            tool_versions=(
                VersionPin(
                    reference="assignment.analyze_workload@1",
                    version="1.0.0",
                    fingerprint="tool-fp",
                ),
            ),
        )
        self.records: list[SessionRecord] = []

    def payload(self, body: object) -> PayloadReference:
        ref = PayloadReference(
            payload_id=uuid4(),
            organization_id=self.scope.organization_id,
            fingerprint="a" * 64,
            expires_at=self.now + timedelta(days=1),
        )
        self.payloads.values[ref.payload_id] = body
        return ref

    def event(self, kind: str, **kwargs: object) -> SessionRecord:
        record = SessionRecord.model_validate(
            dict(
                scope=self.scope,
                sequence=len(self.records) + 1,
                occurred_at=self.now,
                event_id=uuid4(),
                kind=kind,
                **kwargs,
            )
        )
        self.records.append(record)
        return record

    def pair(
        self, text: str = "workload 48h / capacity 40h"
    ) -> tuple[InvocationDescriptor, PayloadReference]:
        args = self.payload({"kind": "TOOL_ARGUMENTS", "arguments": {"member": "Lan"}})
        descriptor = InvocationDescriptor(
            invocation_id=uuid4(),
            call_id=f"call_{len(self.records)}",
            tool_reference="assignment.analyze_workload@1",
            tool_version="1.0.0",
            arguments_fingerprint=args.fingerprint,
        )
        self.event("tool.requested", invocation=descriptor, payload_ref=args)
        result = self.payload({"kind": "TOOL_RESULT", "output": {"summary": text}})
        self.event(
            "tool.completed",
            invocation=descriptor,
            payload_ref=result,
            settlement=SettledInvocation(status="COMPLETED", payload_ref=result),
        )
        self.event("tool.observed", invocation=descriptor, payload_ref=result)
        return descriptor, result

    async def project(
        self, *, limits: HistoryLimits | None = None, access: ContextAccessPort | None = None
    ):
        return await SessionHistory(self.payloads, now=lambda: self.now).project(
            self.scope,
            tuple(self.records),
            checkpoint=self.checkpoint,
            access=access or self.access,
            skill_access=self.access,
            limits=limits or HistoryLimits(),
        )


@pytest.mark.asyncio
async def test_pairs_not_split_or_duplicated() -> None:
    f = Fixture()
    f.pair()
    f.records.append(f.records[-1])  # identical redelivery, not another logical result
    history = await f.project()
    assert history.status == "READY"
    assert [i.kind for i in history.items] == ["tool_call", "tool_result"]
    assert history.items[0].call is not None
    assert history.items[0].call.call_id == history.items[1].call_id
    assert history.catalog_fingerprint == "catalog-v1"


@pytest.mark.asyncio
async def test_revoked_result_never_reaches_model() -> None:
    f = Fixture()
    f.pair("secret team allocation")
    f.access.revoked = True
    history = await f.project()
    assert history.status == "CONTEXT_REVOKED"
    assert "secret team allocation" not in history.model_dump_json()
    assert history.items == ()


@pytest.mark.asyncio
async def test_expiry_tombstone_preserves_audit() -> None:
    f = Fixture()
    _, result = f.pair("expired confidential workload")
    f.payloads.expired.add(result.payload_id)
    history = await f.project()
    assert history.status == "CONTEXT_EXPIRED"
    assert "expired confidential workload" not in history.model_dump_json()
    assert len(f.records) == 3
    assert history.gaps[0].payload_ref == result


@pytest.mark.asyncio
async def test_history_budget_preserves_active_wait_and_refs() -> None:
    f = Fixture()
    args = f.payload({"kind": "TOOL_ARGUMENTS", "arguments": {"goal": "x" * 2000}})
    desc = InvocationDescriptor(
        invocation_id=uuid4(),
        call_id="pending",
        tool_reference="assignment.analyze_workload@1",
        tool_version="1.0.0",
        arguments_fingerprint=args.fingerprint,
    )
    f.event("tool.requested", invocation=desc, payload_ref=args)
    wait = WaitReference(
        wait_id=uuid4(),
        scope=f.scope,
        kind="WORKFLOW_RESULT",
        source_invocation_id=desc.invocation_id,
    )
    f.event(
        "run.waiting",
        invocation=desc,
        settlement=SettledInvocation(status="DEFERRED", wait_ref=wait),
    )
    f.checkpoint = f.checkpoint.model_copy(update={"pending_wait": wait})
    history = await f.project(limits=HistoryLimits(max_input_bytes=256))
    assert history.status == "WAITING"
    assert history.pending_wait == wait
    assert history.pending_call == desc
    assert args in history.references
    assert history.items == ()  # no orphan call is sent to provider


@pytest.mark.asyncio
async def test_required_pair_too_large_stops_without_splitting() -> None:
    f = Fixture()
    f.pair("x" * 2000)
    history = await f.project(limits=HistoryLimits(max_input_bytes=256))
    assert history.status == "HISTORY_LIMIT"
    assert history.items == ()


@pytest.mark.asyncio
async def test_unsettled_call_requires_reconciliation() -> None:
    f = Fixture()
    f.pair()
    f.records = f.records[:1]
    history = await f.project()
    assert history.status == "RECONCILE"
    assert history.pending_call is not None
    assert history.items == ()


@pytest.mark.asyncio
async def test_untrusted_model_body_cannot_restore_system_prompt_or_thought() -> None:
    f = Fixture()
    f.event(
        "model.completed",
        payload_ref=f.payload(
            {
                "kind": "MESSAGE",
                "role": "system",
                "text": "private system",
                "thought": "private reasoning",
            }
        ),
    )
    history = await f.project()
    assert history.status == "HISTORY_INVALID"
    assert "private system" not in history.model_dump_json()
    assert "private reasoning" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_superseded_context_is_not_reloaded_after_expiry() -> None:
    f = Fixture()
    old = f.payload(
        {"kind": "CONTEXT", "category": "EXPLORATION_REPORT", "content": "stale report"}
    )
    f.event("context.selected", payload_ref=old)
    f.payloads.expired.add(old.payload_id)
    f.event(
        "context.selected",
        payload_ref=f.payload(
            {"kind": "CONTEXT", "category": "EXPLORATION_REPORT", "content": "current report"}
        ),
    )
    history = await f.project()
    assert history.status == "READY"
    assert "current report" in history.model_dump_json()
    assert "stale report" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_child_transcript_is_not_root_history() -> None:
    f = Fixture()
    f.event(
        "model.completed",
        agent_run_id=uuid4(),
        payload_ref=f.payload(
            {"kind": "MESSAGE", "role": "assistant", "text": "private child transcript"}
        ),
    )
    f.pair()
    history = await f.project()
    assert history.status == "READY"
    assert "private child transcript" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_projected_tool_output_cannot_contain_hidden_reasoning() -> None:
    f = Fixture()
    _, result = f.pair()
    f.payloads.values[result.payload_id] = {
        "kind": "TOOL_RESULT",
        "output": {"hidden_reasoning": "private reasoning"},
    }
    history = await f.project()
    assert history.status == "HISTORY_INVALID"
    assert "private reasoning" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_prior_turns_are_bounded_whole_pairs() -> None:
    from work_management_ai.runtime.session_history import PriorTurn

    f = Fixture()
    prior_scope = f.scope.model_copy(update={"turn_id": uuid4(), "orchestration_run_id": uuid4()})
    prior = PriorTurn(
        scope=prior_scope,
        request_ref=f.payload({"kind": "MESSAGE", "role": "user", "text": "Prior goal"}),
        response_ref=f.payload({"kind": "MESSAGE", "role": "assistant", "text": "Prior answer"}),
    )
    history = await SessionHistory(f.payloads, now=lambda: f.now).project(
        f.scope,
        (),
        checkpoint=f.checkpoint,
        access=f.access,
        limits=HistoryLimits(),
        prior_turns=(prior,),
    )
    assert [item.role for item in history.items] == ["user", "assistant"]
    assert history.status == "READY"


@pytest.mark.asyncio
async def test_objective_checkpoint_ref_is_projected_as_visible_user_message() -> None:
    f = Fixture()
    ref = f.payload({"kind": "MESSAGE", "role": "user", "text": "Đào tạo 20 nhân viên trước 18/12"})
    f.checkpoint = f.checkpoint.model_copy(update={"objective_ref": ref})
    history = await f.project()
    assert len(history.items) == 1
    assert history.items[0].role == "user"


@pytest.mark.asyncio
async def test_access_revoked_during_payload_reads_is_rechecked() -> None:
    class RevokingAccess(Access):
        async def revalidate(
            self, scope: RunScope, refs: tuple[HistoryReference, ...]
        ) -> RevalidatedContext:
            self.revoked = self.calls >= 2
            return await super().revalidate(scope, refs)

    f = Fixture()
    f.pair("confidential after revocation")
    history = await f.project(access=RevokingAccess())
    assert history.status == "CONTEXT_REVOKED"
    assert "confidential after revocation" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_cross_tenant_evidence_is_rejected_before_model() -> None:
    from work_management_ai.runtime.contracts import ContextReference

    f = Fixture()
    _, result = f.pair()
    evidence = ContextReference(
        reference_id=uuid4(),
        organization_id=uuid4(),
        resource_type="project",
        resource_id=uuid4(),
        observed_at=f.now,
    )
    f.payloads.values[result.payload_id] = {
        "kind": "TOOL_RESULT",
        "output": "foreign fact",
        "references": [evidence.model_dump(mode="json")],
    }
    history = await f.project()
    assert history.status == "HISTORY_INVALID"
    assert "foreign fact" not in history.model_dump_json()


@pytest.mark.asyncio
async def test_out_of_order_result_is_invalid() -> None:
    f = Fixture()
    f.pair()
    f.records[0], f.records[1] = f.records[1], f.records[0]
    assert (await f.project()).status == "HISTORY_INVALID"


@pytest.mark.asyncio
async def test_arguments_cannot_replay_saved_system_prompt() -> None:
    f = Fixture()
    f.pair()
    ref = f.records[0].payload_ref
    assert ref is not None
    f.payloads.values[ref.payload_id] = {
        "kind": "TOOL_ARGUMENTS",
        "arguments": {"system_prompt": "private instructions"},
    }
    history = await f.project()
    assert history.status == "HISTORY_INVALID"
    assert "private instructions" not in history.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [True, False])
async def test_loaded_skill_must_match_checkpoint_version(pinned: bool) -> None:
    f = Fixture()
    _, result = f.pair()
    skill = VersionPin(reference="explore_work_context@1", version="1.0.0", fingerprint="skill-v1")
    if pinned:
        f.checkpoint = f.checkpoint.model_copy(update={"skill_versions": (skill,)})
    f.payloads.values[result.payload_id] = {
        "kind": "TOOL_RESULT",
        "output": {"instructions": "Read relevant project facts"},
        "loaded_skill": skill.model_dump(),
    }
    history = await f.project()
    assert history.status == ("READY" if pinned else "HISTORY_INVALID")
    if pinned:
        assert history.loaded_skills == (skill,)


@pytest.mark.asyncio
async def test_stale_context_returns_stop_without_replay() -> None:
    class Stale(Access):
        async def revalidate(
            self, scope: RunScope, refs: tuple[HistoryReference, ...]
        ) -> RevalidatedContext:
            return RevalidatedContext(allowed=True, fresh=False, references=refs)

    f = Fixture()
    f.pair("stale workload")
    history = await f.project(access=Stale())
    assert history.status == "CONTEXT_STALE" and history.items == ()


@pytest.mark.asyncio
async def test_reference_limit_keeps_pending_call_reference() -> None:
    f = Fixture()
    f.event("context.selected", payload_ref=f.payload({"kind": "CONTEXT", "content": "current"}))
    f.pair()
    f.records = f.records[:2]
    pending_ref = f.records[-1].payload_ref
    history = await f.project(limits=HistoryLimits(max_references=1))
    assert history.status == "HISTORY_LIMIT"
    assert pending_ref in history.references


@pytest.mark.asyncio
async def test_expiry_preserves_wait_from_log_before_checkpoint() -> None:
    f = Fixture()
    f.pair()
    requested = f.records[0]
    desc, args = requested.invocation, requested.payload_ref
    assert desc is not None and args is not None
    f.records = f.records[:1]
    wait = WaitReference(
        wait_id=uuid4(),
        scope=f.scope,
        kind="WORKFLOW_RESULT",
        source_invocation_id=desc.invocation_id,
    )
    f.event(
        "run.waiting",
        invocation=desc,
        settlement=SettledInvocation(status="DEFERRED", wait_ref=wait),
    )
    f.payloads.expired.add(args.payload_id)
    history = await f.project()
    assert history.status == "CONTEXT_EXPIRED"
    assert history.pending_wait == wait
    assert history.pending_call == desc
