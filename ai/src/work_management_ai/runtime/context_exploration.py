"""Bounded deterministic projection from committed reads; no model or effect."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid5

from work_management_ai.runtime.continuation import UserClarificationPort
from work_management_ai.runtime.tool_loop_contracts import RunScope, VersionPin
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.tools.context.contracts import (
    ContextReadObservation,
    ContextRef,
    ContextUseValidation,
    Contract,
    ExplorationReport,
    UseOutcome,
)


class CanonicalContextPort(Protocol):
    def completed(
        self, *, scope: RunScope, invocation_id: UUID
    ) -> ContextReadObservation | None: ...


class ContextAccessDecision(Contract):
    outcome: UseOutcome
    policy_revision: str
    actor_revision: str


class ContextUseAccessPort(Protocol):
    async def check(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextAccessDecision: ...


class ExplorationProjector:
    def __init__(
        self,
        *,
        scope: RunScope,
        objective_ref: UUID,
        sources: CanonicalContextPort,
        tool_pins: tuple[VersionPin, ...],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.scope, self.objective_ref, self.sources, self.clock = (
            scope,
            objective_ref,
            sources,
            clock,
        )
        self._initial = self._new_empty()
        self._current: ExplorationReport = self._initial
        self._tool_pins = {(pin.reference, pin.version): pin for pin in tool_pins}

    @property
    def current(self) -> ExplorationReport | None:
        return self._current

    def empty(self) -> ExplorationReport:
        return self._initial

    def _new_empty(self) -> ExplorationReport:
        return ExplorationReport(
            scope=self.scope,
            objective_ref=self.objective_ref,
            report_id=uuid5(self.scope.orchestration_run_id, str(self.objective_ref)),
            built_at=self.clock(),
        )

    def _source(self, invocation_id: UUID) -> ContextReadObservation:
        o = self.sources.completed(scope=self.scope, invocation_id=invocation_id)
        if (
            o is None
            or o.scope != self.scope
            or o.objective_ref != self.objective_ref
            or o.invocation_id != invocation_id
        ):
            raise ValueError("CANONICAL_CONTEXT_REQUIRED")
        observation = ContextReadObservation.model_validate(o)
        if (observation.tool_reference, observation.tool_version) not in self._tool_pins:
            raise ToolPolicyError("CONTEXT_TOOL_VERSION_CONFLICT", stop_segment=True)
        return observation

    def _fold(self, ids: tuple[UUID, ...]) -> ExplorationReport:
        report = self.empty()
        for invocation in ids:
            o = self._source(invocation)
            refs = (*report.context_refs, *o.context_refs)
            lost = len(refs) > 32
            refs = refs[:32]
            kept = {r.ref_id for r in refs}
            facts = tuple(f for f in (*report.facts, *o.facts) if f.ref_id in kept)
            content = tuple(
                s for s in (*report.source_content_refs, *o.source_content_refs) if s.ref_id in kept
            )
            entities = (*report.entities, *o.entities)
            # Keep resolution honest even if some candidates cannot fit.
            entities = tuple(
                e.model_copy(
                    update={
                        "candidate_ref_ids": tuple(i for i in e.candidate_ref_ids if i in kept),
                        "selected_ref_id": e.selected_ref_id if e.selected_ref_id in kept else None,
                    }
                )
                for e in entities
            )
            gaps = tuple(dict.fromkeys((*report.gaps, *o.gaps)))
            lost = lost or len(gaps) > 16
            candidate = report.model_copy(
                update={
                    "source_invocation_ids": (*report.source_invocation_ids, invocation),
                    "context_refs": refs,
                    "facts": facts,
                    "source_content_refs": content,
                    "entities": entities,
                    "coverage": (*report.coverage, *o.coverage),
                    "gaps": gaps[:16],
                    "status": "PARTIAL"
                    if lost or report.status == "PARTIAL" or o.outcome != "OK"
                    else "COMPLETE",
                    "truncated": report.truncated or lost or any(c.truncated for c in o.coverage),
                }
            )
            while len(candidate.model_dump_json().encode()) > 14500:
                lost = True
                if candidate.context_refs:
                    drop = candidate.context_refs[-1].ref_id
                    candidate = candidate.model_copy(
                        update={
                            "context_refs": candidate.context_refs[:-1],
                            "facts": tuple(f for f in candidate.facts if f.ref_id != drop),
                            "source_content_refs": tuple(
                                s for s in candidate.source_content_refs if s.ref_id != drop
                            ),
                            "entities": tuple(
                                e.model_copy(
                                    update={
                                        "candidate_ref_ids": tuple(
                                            i for i in e.candidate_ref_ids if i != drop
                                        ),
                                        "selected_ref_id": None
                                        if e.selected_ref_id == drop
                                        else e.selected_ref_id,
                                    }
                                )
                                for e in candidate.entities
                            ),
                        }
                    )
                elif candidate.coverage:
                    candidate = candidate.model_copy(
                        update={
                            "coverage": candidate.coverage[1:],
                            "entities": candidate.entities[1:],
                        }
                    )
                else:
                    raise ValueError("CONTEXT_REPORT_TOO_LARGE")
            if lost:
                candidate = candidate.model_copy(update={"status": "PARTIAL", "truncated": True})
            report = candidate
        return report

    def verify(self, report: ExplorationReport) -> None:
        # Only the runtime-issued head may advance. A model/checkpoint object is not a head.
        # Recovery must replay canonical observations (including overflow) before selection.
        if report != self._current:
            raise ValueError("CANONICAL_REPORT_MISMATCH")
        if report.scope != self.scope or report.objective_ref != self.objective_ref:
            raise ValueError("CONTEXT_SCOPE_MISMATCH")
        expected = self._fold(report.source_invocation_ids)
        fields = (
            "report_id",
            "context_refs",
            "entities",
            "facts",
            "source_content_refs",
            "coverage",
            "gaps",
        )
        if any(getattr(report, f) != getattr(expected, f) for f in fields) or not set(
            report.selected_context_refs
        ) <= {r.ref_id for r in report.context_refs}:
            raise ValueError("CANONICAL_REPORT_MISMATCH")
        if expected.truncated and not report.truncated:
            raise ValueError("COVERAGE_MISMATCH")
        if len(set(report.source_invocation_ids)) != len(report.source_invocation_ids):
            raise ValueError("DUPLICATE_CONTEXT_SOURCE")
        if expected.status == "PARTIAL" and report.status != "PARTIAL":
            raise ValueError("COVERAGE_MISMATCH")

    def apply(
        self, report: ExplorationReport, observation: ContextReadObservation
    ) -> ExplorationReport:
        self.verify(report)
        if self._source(observation.invocation_id) != observation:
            raise ValueError("CANONICAL_CONTEXT_MISMATCH")
        if observation.invocation_id in report.source_invocation_ids:
            return report
        if len(report.source_invocation_ids) == 16:
            result = report.model_copy(
                update={
                    "status": "PARTIAL",
                    "truncated": True,
                    "revision": report.revision + 1,
                    "snapshot_sequence": report.snapshot_sequence + 1,
                    "built_at": self.clock(),
                }
            )
        else:
            folded = self._fold((*report.source_invocation_ids, observation.invocation_id))
            kept = {r.ref_id for r in folded.context_refs}
            result = folded.model_copy(
                update={
                    "revision": report.revision + 1,
                    "snapshot_sequence": report.snapshot_sequence + 1,
                    "built_at": self.clock(),
                    "selected_context_refs": tuple(
                        i for i in report.selected_context_refs if i in kept
                    ),
                }
            )
        self._current = result
        return result

    def unresolved_resources(self, report: ExplorationReport) -> frozenset[tuple[str, UUID]]:
        self.verify(report)
        unresolved: set[tuple[str, UUID]] = set()
        for invocation_id in report.source_invocation_ids:
            observation = self._source(invocation_id)
            candidates = {
                i
                for e in observation.entities
                if e.resolution != "UNIQUE"
                for i in e.candidate_ref_ids
            }
            unresolved.update(
                (r.resource_type, r.resource_id)
                for r in observation.context_refs
                if r.ref_id in candidates
            )
        return frozenset(unresolved)

    async def check_clarification(
        self,
        report: ExplorationReport,
        *,
        refs: tuple[ContextRef, ...],
        decision: ContextAccessDecision,
        choices: UserClarificationPort,
        user_turn_id: UUID,
    ) -> UseOutcome:
        self.verify(report)
        if user_turn_id == self.scope.turn_id:
            return "NEEDS_INPUT"
        for invocation_id in report.source_invocation_ids:
            observation = self._source(invocation_id)
            for entity in observation.entities:
                if entity.resolution == "UNIQUE":
                    continue
                candidates = {
                    ref.ref_id: ref
                    for ref in observation.context_refs
                    if ref.ref_id in entity.candidate_ref_ids
                }
                resource_ids = {ref.resource_id for ref in candidates.values()}
                requested = tuple(
                    ref
                    for ref in refs
                    if ref.resource_type == "project" and ref.resource_id in resource_ids
                )
                if not requested:
                    continue
                if entity.resolution != "AMBIGUOUS":
                    return "NEEDS_INPUT"
                binding = await choices.accepted(
                    source_scope=self.scope,
                    user_turn_id=user_turn_id,
                    source_invocation_id=invocation_id,
                )
                if binding is None or (
                    binding.source_scope != self.scope
                    or binding.user_turn_id != user_turn_id
                    or binding.source_invocation_id != invocation_id
                    or binding.query_fingerprint != entity.query_fingerprint
                    or binding.selected_ref_id not in candidates
                ):
                    return "NEEDS_INPUT"
                selected = candidates[binding.selected_ref_id]
                if any(ref.resource_id != selected.resource_id for ref in requested):
                    return "NEEDS_INPUT"
                if (
                    binding.source_tool_pin
                    != self._tool_pins[observation.tool_reference, observation.tool_version]
                    or binding.resource_id != selected.resource_id
                    or binding.actor_revision != decision.actor_revision
                    or binding.policy_revision != decision.policy_revision
                    or any(
                        (ref.version, ref.fingerprint)
                        != (binding.resource_version, binding.resource_fingerprint)
                        for ref in (selected, *requested)
                    )
                ):
                    return "REFRESH_REQUIRED"
        return "ACCEPTED"

    def select(
        self, report: ExplorationReport, context_ref_ids: tuple[UUID, ...]
    ) -> ExplorationReport:
        self.verify(report)
        if len(set(context_ref_ids)) != len(context_ref_ids) or not set(context_ref_ids) <= {
            r.ref_id for r in report.context_refs
        }:
            raise ValueError("UNKNOWN_CONTEXT_REFERENCE")
        result = report.model_copy(
            update={
                "selected_context_refs": context_ref_ids,
                "revision": report.revision + 1,
                "built_at": self.clock(),
            }
        )
        self._current = result
        return result


class ContextUseValidator:
    def __init__(
        self,
        *,
        projector: ExplorationProjector,
        access: ContextUseAccessPort,
        choices: UserClarificationPort | None = None,
        user_turn_id: UUID | None = None,
    ):
        self.projector, self.access = projector, access
        self._choices, self._user_turn_id = choices, user_turn_id

    async def validate(
        self,
        scope: RunScope,
        *,
        target_tool: str,
        input_fingerprint: str,
        refs: tuple[ContextRef, ...],
    ) -> ContextUseValidation:
        report = self.projector.current
        decision = ContextAccessDecision(
            outcome="DENIED", actor_revision="unavailable", policy_revision="unavailable"
        )
        if scope == self.projector.scope and report is not None:
            self.projector.verify(report)
            known = {r.ref_id: r for r in report.context_refs}
            if len({r.ref_id for r in refs}) == len(refs) and all(
                known.get(r.ref_id) == r for r in refs
            ):
                decision = await self.access.check(
                    scope, target_tool=target_tool, input_fingerprint=input_fingerprint, refs=refs
                )
                ambiguous = self.projector.unresolved_resources(report)
                if decision.outcome == "ACCEPTED" and any(
                    (r.resource_type, r.resource_id) in ambiguous for r in refs
                ):
                    outcome = "NEEDS_INPUT"
                    if self._choices is not None and self._user_turn_id is not None:
                        outcome = await self.projector.check_clarification(
                            report,
                            refs=refs,
                            decision=decision,
                            choices=self._choices,
                            user_turn_id=self._user_turn_id,
                        )
                    decision = decision.model_copy(update={"outcome": outcome})
        return ContextUseValidation(
            scope=scope,
            target_tool=target_tool,
            input_fingerprint=input_fingerprint,
            context_versions=refs,
            **decision.model_dump(),
        )
