"""Canonical context projection and action-specific validation."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from work_management_ai.runtime.context_exploration import (
    ContextAccessDecision,
    ContextUseValidator,
    ExplorationProjector,
)
from work_management_ai.runtime.tool_loop_contracts import RunScope
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    ReadCoverage,
    Resolution,
    UseOutcome,
)

NOW = datetime(2026, 11, 17, tzinfo=UTC)


class Sources:
    def __init__(self) -> None:
        self.values: dict[object, ContextReadObservation] = {}

    def completed(self, *, scope: RunScope, invocation_id: object) -> ContextReadObservation | None:
        value = self.values.get(invocation_id)
        return value if value and value.scope == scope else None


class Access:
    outcome: UseOutcome = "ACCEPTED"

    async def check(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision:
        return ContextAccessDecision(
            outcome=self.outcome, policy_revision="policy1", actor_revision="actor1"
        )


def fixture():
    scope = RunScope(
        organization_id=uuid4(),
        membership_id=uuid4(),
        conversation_id=uuid4(),
        turn_id=uuid4(),
        orchestration_run_id=uuid4(),
    )
    objective = uuid4()
    sources = Sources()
    registry = ToolRegistry.from_resources(
        ("work_management_ai.tools.context", resource)
        for resource in ("projects.yaml", "project.yaml", "actor.yaml")
    )
    projector = ExplorationProjector(
        scope=scope,
        objective_ref=objective,
        sources=sources,
        tool_pins=tuple(
            entry.pin
            for entry in registry.candidate_definitions(
                ("explore_projects@1", "read_project_context@1", "read_actor_context@1")
            )
        ),
        clock=lambda: NOW,
    )
    return scope, objective, sources, projector


def observation(
    scope: RunScope,
    objective: UUID,
    *,
    name: str = "Project",
    resolution: Resolution = "UNIQUE",
    count: int = 1,
) -> ContextReadObservation:
    refs = tuple(
        ContextRef(
            ref_id=uuid4(),
            resource_type="project",
            resource_id=uuid4(),
            version=1,
            fingerprint="a" * 64,
            observed_at=NOW,
            scope=scope,
        )
        for _ in range(count)
    )
    return ContextReadObservation(
        scope=scope,
        objective_ref=objective,
        invocation_id=uuid4(),
        tool_reference="explore_projects@1",
        tool_version="1.0.2",
        observed_at=NOW,
        outcome="OK",
        context_refs=refs,
        entities=(
            EntityResolution(
                query_fingerprint="b" * 64,
                resolution=resolution,
                candidate_ref_ids=tuple(r.ref_id for r in refs),
                selected_ref_id=refs[0].ref_id if resolution == "UNIQUE" else None,
            ),
        ),
        facts=tuple(ContextFact(ref_id=r.ref_id, field="project.name", value=name) for r in refs),
        coverage=(
            ReadCoverage(
                requested=("overview",),
                returned=("overview",),
                status="COMPLETE",
                query_fingerprint="b" * 64,
            ),
        ),
    )


def test_projector_rejects_observation_from_unpinned_tool_version() -> None:
    scope, objective, sources, projector = fixture()
    stale = observation(scope, objective).model_copy(update={"tool_version": "1.0.1"})
    sources.values[stale.invocation_id] = stale
    with pytest.raises(ToolPolicyError, match="CONTEXT_TOOL_VERSION_CONFLICT") as error:
        projector.apply(projector.empty(), stale)
    assert error.value.stop_segment


def test_only_exact_canonical_observations_and_objective_enter_report():
    scope, obj, src, p = fixture()
    o = observation(scope, obj)
    with pytest.raises(ValueError, match="CANONICAL"):
        p.apply(p.empty(), o)
    src.values[o.invocation_id] = o
    report = p.apply(p.empty(), o)
    assert report.facts[0].value == "Project"
    assert p.apply(report, o) == report
    with pytest.raises(ValueError):
        p.apply(p.empty(), o.model_copy(update={"objective_ref": uuid4()}))
    with pytest.raises(ValueError):
        p.apply(p.empty(), o.model_copy(update={"facts": ()}))
    forged = report.model_copy(update={"facts": ()})
    with pytest.raises(ValueError):
        p.select(forged, (o.context_refs[0].ref_id,))


def test_selection_is_subset_and_conversation_owned():
    scope, obj, src, p = fixture()
    o = observation(scope, obj)
    src.values[o.invocation_id] = o
    report = p.apply(p.empty(), o)
    assert p.select(report, (o.context_refs[0].ref_id,)).selected_context_refs
    with pytest.raises(ValueError):
        p.select(report, (uuid4(),))
    other = scope.model_copy(update={"conversation_id": uuid4()})
    with pytest.raises(ValueError):
        p.select(report.model_copy(update={"scope": other}), ())


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["ACCEPTED", "DENIED", "REFRESH_REQUIRED"])
async def test_use_validation_is_bound_to_tool_input_policy_and_live_access(outcome: UseOutcome):
    scope, obj, src, p = fixture()
    o = observation(scope, obj)
    src.values[o.invocation_id] = o
    access = Access()
    access.outcome = outcome
    v = ContextUseValidator(projector=p, access=access)
    report = p.apply(p.empty(), o)
    result = await v.validate(
        scope, target_tool="delegate_agent@1", input_fingerprint="c" * 64, refs=report.context_refs
    )
    assert result.outcome == outcome
    assert result.target_tool == "delegate_agent@1" and result.input_fingerprint == "c" * 64
    assert result.actor_revision == "actor1" and result.context_versions == report.context_refs
    forged = report.context_refs[0].model_copy(update={"version": 2})
    result = await v.validate(
        scope, target_tool="delegate_agent@1", input_fingerprint="c" * 64, refs=(forged,)
    )
    assert result.outcome == "DENIED"


@pytest.mark.asyncio
async def test_valid_ambiguous_discovery_does_not_select_or_authorize_binding():
    scope, obj, src, p = fixture()
    o = observation(scope, obj, resolution="AMBIGUOUS", count=2)
    src.values[o.invocation_id] = o
    report = p.apply(p.empty(), o)
    assert report.entities[0].resolution == "AMBIGUOUS"
    v = ContextUseValidator(projector=p, access=Access())
    result = await v.validate(
        scope,
        target_tool="delegate_agent@1",
        input_fingerprint="c" * 64,
        refs=(report.context_refs[0],),
    )
    assert result.outcome == "NEEDS_INPUT"


@pytest.mark.parametrize("limit", ["refs", "events", "gaps", "bytes"])
def test_projection_bounds_remain_valid_json_with_provenance_and_partial(limit: str):
    scope, obj, src, p = fixture()
    report = p.empty()
    for i in range(18):
        o = observation(
            scope,
            obj,
            count=4,
            resolution="AMBIGUOUS",
            name="界" * 160 if limit == "bytes" else "Project",
        )
        if limit == "gaps":
            from work_management_ai.tools.context.contracts import ContextGap

            o = o.model_copy(
                update={
                    "gaps": tuple(
                        ContextGap(code="NOT_SUPPORTED", section=f"s{i}_{j}") for j in range(4)
                    )
                }
            )
        src.values[o.invocation_id] = o
        report = p.apply(report, o)
    assert (
        len(report.context_refs) <= 32
        and len(report.source_invocation_ids) <= 16
        and len(report.gaps) <= 16
    )
    assert len(report.model_dump_json().encode()) <= 16384
    assert report.truncated and report.status == "PARTIAL"
    ids = {r.ref_id for r in report.context_refs}
    assert all(f.ref_id in ids for f in report.facts)
    assert all(r.invocation_id in report.source_invocation_ids for r in report.context_refs)


def test_selection_reserves_json_bytes_and_cannot_clear_partial_truncation():
    scope, obj, src, p = fixture()
    report = p.empty()
    for _ in range(12):
        o = observation(scope, obj, name="界" * 160, resolution="AMBIGUOUS", count=3)
        src.values[o.invocation_id] = o
        report = p.apply(report, o)
    selected = p.select(report, tuple(r.ref_id for r in report.context_refs))
    assert len(selected.model_dump_json().encode()) <= 16384
    if report.truncated:
        with pytest.raises(ValueError):
            p.select(report.model_copy(update={"truncated": False}), ())


@pytest.mark.asyncio
async def test_ambiguous_resource_stays_unresolved_after_direct_read_creates_new_ref():
    scope, obj, src, p = fixture()
    discovered = observation(scope, obj, resolution="AMBIGUOUS", count=2)
    src.values[discovered.invocation_id] = discovered
    report = p.apply(p.empty(), discovered)
    read = observation(scope, obj)
    resource = discovered.context_refs[0].resource_id
    read = read.model_copy(
        update={
            "context_refs": (read.context_refs[0].model_copy(update={"resource_id": resource}),),
            "tool_reference": "read_project_context@1",
        }
    )
    src.values[read.invocation_id] = read
    report = p.apply(report, read)
    result = await ContextUseValidator(projector=p, access=Access()).validate(
        scope,
        target_tool="delegate_agent@1",
        input_fingerprint="d" * 64,
        refs=(report.context_refs[-1],),
    )
    assert result.outcome == "NEEDS_INPUT"


def test_event_overflow_and_runtime_owned_metadata_cannot_be_forged():
    scope, obj, src, p = fixture()
    report = p.empty()
    for _ in range(17):
        o = observation(scope, obj).model_copy(
            update={"context_refs": (), "facts": (), "entities": (), "coverage": ()}
        )
        src.values[o.invocation_id] = o
        report = p.apply(report, o)
    assert len(report.source_invocation_ids) == 16 and report.truncated
    with pytest.raises(ValueError):
        p.select(report.model_copy(update={"truncated": False, "status": "COMPLETE"}), ())
    with pytest.raises(ValueError):
        p.select(report.model_copy(update={"revision": 1000}), ())
    _, _, _, fresh = fixture()
    with pytest.raises(ValueError):
        fresh.select(fresh.empty().model_copy(update={"status": "COMPLETE"}), ())
