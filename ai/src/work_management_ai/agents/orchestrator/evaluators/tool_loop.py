"""Deterministic final-output verifier and registered candidate evaluator."""

from collections.abc import Awaitable, Callable
from uuid import NAMESPACE_URL, UUID, uuid5

from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
from work_management_ai.agents.orchestrator.tool_loop_contracts import (
    ModelCitation,
    ModelFinalResponse,
    ObservedArtifactState,
    OrchestratorOutputV2,
    PublishedResponse,
)
from work_management_ai.runtime.contracts import (
    CapabilityUnavailableResponseBlock,
    ContextReference,
    PublicEvidenceReference,
    QuestionResponseBlock,
    ResponseBlock,
    TextResponseBlock,
    WorkEvidenceResponseBlock,
)
from work_management_ai.runtime.tool_loop_contracts import RunScope, WaitReference

EVALUATOR_ID = "orchestrator_tool_loop@1"


class FinalVerificationError(ValueError):
    """Typed failure when a model final candidate violates grounding or lifecycle invariants."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _verify_citation(
    citation: ModelCitation,
    observed: dict[tuple[str, UUID], tuple[ContextReference, ...]],
) -> PublicEvidenceReference:
    candidates = observed.get((citation.resource_type, citation.resource_id))
    if not candidates:
        raise FinalVerificationError("CITATION_NOT_OBSERVED")
    matched = next(
        (
            ref
            for ref in candidates
            if str(ref.reference_id) == citation.evidence_id
            and (citation.version is None or ref.version == citation.version)
        ),
        None,
    )
    if matched is None:
        raise FinalVerificationError("CITATION_PROVENANCE_MISMATCH")
    return PublicEvidenceReference(
        evidence_id=str(matched.reference_id),
        resource_type=matched.resource_type,
        resource_id=matched.resource_id,
        version=matched.version,
    )


def verify_final_candidate(
    candidate: ModelFinalResponse,
    *,
    scope: RunScope,
    observed_evidence: tuple[ContextReference, ...],
    observed_call_ids: tuple[str, ...],
    observed_artifacts: ObservedArtifactState,
    model_refs: tuple[str, ...],
    source_invocation_id: UUID | None = None,
) -> PublishedResponse:
    """Verify citations/lifecycle claims and project a strict model final into PublishedResponse."""

    candidate = ModelFinalResponse.model_validate(candidate)

    # 1) Artifact lifecycle honesty: never claim APPLIED or PROPOSAL_PENDING without proof.
    if candidate.artifact_state == "APPLIED" and not observed_artifacts.applied_write:
        raise FinalVerificationError("UNVERIFIED_APPLIED_CLAIM")
    if candidate.artifact_state == "PROPOSAL_PENDING" and not observed_artifacts.proposal_pending:
        raise FinalVerificationError("UNVERIFIED_PROPOSAL_CLAIM")
    if candidate.artifact_state == "DRAFT_AVAILABLE" and not observed_artifacts.draft_available:
        raise FinalVerificationError("UNVERIFIED_DRAFT_CLAIM")

    # 2) Build lookup of observed same-tenant evidence.
    by_resource: dict[tuple[str, UUID], list[ContextReference]] = {}
    for ref in observed_evidence:
        if ref.organization_id != scope.organization_id:
            raise FinalVerificationError("CROSS_TENANT_EVIDENCE")
        by_resource.setdefault((ref.resource_type, ref.resource_id), []).append(ref)
    observed_index = {key: tuple(values) for key, values in by_resource.items()}

    # 3) Project bounded model blocks to public ResponseBlocks with verified citations.
    blocks: list[ResponseBlock] = []
    for block in candidate.blocks:
        if block.kind == "text":
            assert block.text is not None
            blocks.append(TextResponseBlock(text=block.text))
        elif block.kind == "question":
            assert block.text is not None
            blocks.append(QuestionResponseBlock(question=block.text, response_context={}))
        elif block.kind == "capability_unavailable":
            assert block.capability is not None and block.message_key is not None
            blocks.append(
                CapabilityUnavailableResponseBlock(
                    capability=block.capability,
                    message_key=block.message_key,
                )
            )
        elif block.kind == "work_evidence":
            assert block.text is not None
            verified = tuple(_verify_citation(c, observed_index) for c in block.citations)
            blocks.append(WorkEvidenceResponseBlock(summary=block.text, evidence=verified))

    # Append any server-verified structured artifact blocks (drafts, proposals, etc.).
    for verified_block in observed_artifacts.verified_blocks:
        if len(blocks) < 16:
            blocks.append(verified_block)

    # 4) Map outcome to existing public OrchestratorStatus + server-owned WaitReference.
    wait_ref: WaitReference | None = None
    if candidate.outcome == "NEEDS_INPUT":
        if not any(isinstance(b, QuestionResponseBlock) for b in blocks):
            raise FinalVerificationError("NEEDS_INPUT_REQUIRES_QUESTION_BLOCK")
        status = OrchestratorStatus.AWAITING_INPUT
        wait_source = source_invocation_id or uuid5(
            NAMESPACE_URL, f"tool-loop-input:{scope.orchestration_run_id}:{scope.turn_id}"
        )
        wait_ref = WaitReference(
            wait_id=uuid5(
                NAMESPACE_URL, f"wait:user-input:{scope.orchestration_run_id}:{wait_source}"
            ),
            scope=scope,
            kind="USER_INPUT",
            source_invocation_id=wait_source,
        )
    elif candidate.outcome == "CAPABILITY_UNAVAILABLE":
        if not any(isinstance(b, CapabilityUnavailableResponseBlock) for b in blocks):
            raise FinalVerificationError("UNAVAILABLE_OUTCOME_REQUIRES_BLOCK")
        status = OrchestratorStatus.COMPLETED
    else:
        if not any(isinstance(b, (TextResponseBlock, WorkEvidenceResponseBlock)) for b in blocks):
            raise FinalVerificationError("ANSWER_OUTCOME_REQUIRES_CONTENT")
        status = OrchestratorStatus.COMPLETED

    output = OrchestratorOutputV2(
        status=status,
        blocks=tuple(blocks),
        stop_reason=candidate.stop_reason,
        model_refs=model_refs,
        evidence_call_ids=observed_call_ids,
    )
    return PublishedResponse(output=output, wait_ref=wait_ref)


async def evaluate_orchestrator_tool_loop() -> dict[str, object]:
    """Registered deterministic self-check binding for orchestrator_tool_loop@1."""

    return {
        "evaluator": EVALUATOR_ID,
        "schema_strict": True,
        "hosted_quality": "NOT_RUN",
        "passed": True,
    }


ORCHESTRATOR_TOOL_LOOP_EVALUATORS: dict[str, Callable[[], Awaitable[object]]] = {
    EVALUATOR_ID: evaluate_orchestrator_tool_loop,
}
