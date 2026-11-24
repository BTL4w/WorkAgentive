"""Execute the bilingual Orchestrator v2 acceptance gate with synthetic adapters."""

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from work_management_ai.agents.orchestrator.contracts import OrchestratorStatus
from work_management_ai.agents.orchestrator.evaluators.tool_loop import (
    EVALUATOR_ID,
    ORCHESTRATOR_TOOL_LOOP_EVALUATORS,
)
from work_management_ai.agents.orchestrator.tool_loop_contracts import (
    PublishedResponse,
    SuspendedWorkflow,
)
from work_management_ai.evaluation.context_exploration import (
    CONTEXT_EXPLORATION_EVALUATORS,
    ContextExplorationEvaluationReport,
)
from work_management_ai.evaluation.tool_loop_harness import (
    FIXTURE_TIME,
    AllowAllPolicy,
    FixtureToolBackend,
    answer_fixture,
    make_harness,
)
from work_management_ai.model_gateway.contracts import ModelUsage
from work_management_ai.model_gateway.mock import MockModelGateway
from work_management_ai.model_gateway.tool_contracts import (
    NativeToolGateway,
    ToolModelRequest,
    ToolModelResponse,
)
from work_management_ai.runtime.contracts import AgentId, AgentRunStatus, ContextReference
from work_management_ai.runtime.manifests import canonical_manifest_fingerprint
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_loop_contracts import (
    AdmittedTool,
    BackendToolResult,
    RunScope,
    ToolInvocationContext,
    VersionPin,
    WaitReference,
)
from work_management_ai.runtime.tool_pipeline import ToolPolicyError
from work_management_ai.tools.context.contracts import (
    ContextFact,
    ContextGap,
    ContextReadObservation,
    ContextRef,
    EntityResolution,
    ExploreProjectsInputV103,
    ReadCoverage,
)
from work_management_ai.tools.delegation.contracts import (
    DelegateAgentInputV2,
    DelegateAgentOutputV2,
)

Locale = Literal["vi", "en"]
Scenario = Literal["greeting", "fresh-context", "ambiguity", "workload", "waits", "revoked"]
Variant = Literal["default", "ambiguous", "partial", "32h", "48h", "workflow", "human", "stale"]
_CORPUS = Path(__file__).resolve().parents[3] / "evaluations" / "orchestrator_tool_loop.jsonl"
_SKILL_PACKAGE = "work_management_ai.skills.explore_work_context"


@dataclass(frozen=True, slots=True)
class ToolLoopCase:
    locale: Locale
    scenario: Scenario
    prompt: str

    @property
    def case_id(self) -> str:
        return f"{self.locale}-{self.scenario}"


_PROMPTS: tuple[tuple[Scenario, str, str], ...] = (
    ("greeting", "Xin chào", "Hello"),
    ("fresh-context", "Tóm tắt dự án đã chọn", "Summarize the selected project"),
    ("ambiguity", "Cho tôi biết tổng quan dự án Sao Mai", "Give me an overview of Project Sao Mai"),
    (
        "workload",
        "Kiểm tra tải tuần này và đề xuất điều chỉnh nếu quá tải",
        "Check this week's workload and suggest an adjustment if there is an overload",
    ),
    ("waits", "Chuẩn bị kế hoạch để quản lý xem xét", "Prepare a plan for manager review"),
    ("revoked", "Đề xuất sửa kế hoạch dự án", "Propose a change to the project plan"),
)
REQUIRED_CASES = tuple(
    case
    for scenario, vi, en in _PROMPTS
    for case in (ToolLoopCase("vi", scenario, vi), ToolLoopCase("en", scenario, en))
)
_VARIANTS: dict[Scenario, tuple[Variant, ...]] = {
    "greeting": ("default",),
    "fresh-context": ("default",),
    "ambiguity": ("ambiguous", "partial"),
    "workload": ("32h", "48h"),
    "waits": ("workflow", "human"),
    "revoked": ("default", "stale"),
}


class TrialResult(BaseModel):
    variant: Variant
    outcome: str
    requested_tools: tuple[str, ...]
    specialists: tuple[str, ...]
    wait_kind: str | None
    wait_version: int | None
    evidence_count: int
    model_requests: int
    skill_body_reads: int
    fallback: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    violations: tuple[str, ...]


class CaseResult(BaseModel):
    case_id: str
    locale: Locale
    scenario: Scenario
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    trials: tuple[TrialResult, ...] = ()


class WorkloadFact(BaseModel):
    allocated_hours: int
    capacity_hours: int


class ToolLoopEvaluationReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluator: str = EVALUATOR_ID
    engine: Literal["MOCK_RUNTIME"] = "MOCK_RUNTIME"
    hosted_quality: Literal["NOT_RUN"] = "NOT_RUN"
    baseline_runtime: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    broad_business_cases: Literal["DEFERRED"] = "DEFERRED"
    latency_basis: Literal["LOCAL_MOCK_ELAPSED"] = "LOCAL_MOCK_ELAPSED"
    token_basis: Literal["GATEWAY_REPORTED_OR_UNMEASURED"] = "GATEWAY_REPORTED_OR_UNMEASURED"
    revision: str
    workspace_dirty: bool | None
    manifest_version: str
    manifest_sha256: str
    agent_schema_version: str
    output_schema_version: Literal["2.0"] = "2.0"
    tool_pins: tuple[VersionPin, ...]
    skill_pin: VersionPin
    skill_descriptor_sha256: str
    skill_body_sha256: str
    case_set_version: Literal["1.0"] = "1.0"
    case_set_sha256: str
    legacy_authored_cases: int
    legacy_corpus_sha256: str
    context_prerequisite_passed: bool
    context_prerequisite_counts: dict[str, int]
    context_evaluator_version: str
    context_corpus_sha256: str | None
    manifest_evaluator_passed: bool
    cases: tuple[CaseResult, ...]
    violations: tuple[str, ...]
    passed: bool


Decision = Callable[[ToolModelRequest[BaseModel]], dict[str, object]]
GatewayFactory = Callable[[ToolLoopCase, Variant, Decision], NativeToolGateway]


def _base_request[OutputT: BaseModel](
    request: ToolModelRequest[OutputT],
) -> ToolModelRequest[BaseModel]:
    return ToolModelRequest[BaseModel](
        invocation_key=request.invocation_key,
        items=request.items,
        tools=request.tools,
        response_schema=request.response_schema,
        max_output_tokens=request.max_output_tokens,
        timeout_seconds=request.timeout_seconds,
    )


class FixtureGateway:
    def __init__(self, decide: Decision) -> None:
        self.decide = decide

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        response = self.decide(_base_request(request))
        # Run the same strict mock gateway validation used by the tool-loop harness.
        gateway = MockModelGateway(fixtures={request.invocation_key: response})
        return await gateway.generate_step(request)


class RecordingGateway:
    def __init__(self, inner: NativeToolGateway) -> None:
        self.inner = inner
        self.responses: list[ModelUsage | None] = []
        self.requests: list[ToolModelRequest[BaseModel]] = []

    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        self.requests.append(_base_request(request))
        response = await self.inner.generate_step(request)
        self.responses.append(response.usage)
        return response


class TrialPolicy(AllowAllPolicy):
    def __init__(self) -> None:
        self.denial: str | None = None
        self.rejections: list[str] = []

    async def guard(
        self,
        context: ToolInvocationContext,
        tool: AdmittedTool,
        evidence: tuple[ContextReference, ...],
    ) -> None:
        if self.denial is not None:
            raise ToolPolicyError(self.denial, stop_segment=True)

    async def rejected(
        self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
    ) -> None:
        self.rejections.append(safe_code)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"tool_call": {"call_id": call_id, "name": name, "arguments": arguments}}


def _delegate(agent: AgentId, objective: str, refs: list[str]) -> dict[str, object]:
    return _call(
        agent.value,
        "delegate_agent",
        {
            "agent_id": agent.value,
            "objective": objective,
            "context_ref_ids": refs,
            "constraints": [],
            "expected_result": objective,
        },
    )


def _flat_handoff(request: ToolModelRequest[BaseModel]) -> bool:
    schema = next((item for item in request.tools if item.reference == "delegate_agent@2"), None)
    if schema is None:
        return False
    properties = schema.parameters.get("properties")
    return (
        isinstance(properties, dict)
        and properties.keys() == DelegateAgentInputV2.model_fields.keys()
        and "anyOf" not in schema.parameters
        and schema.parameters.get("additionalProperties") is False
    )


def _citation(ref: ContextReference) -> dict[str, object]:
    return {
        "evidence_id": str(ref.reference_id),
        "resource_type": ref.resource_type,
        "resource_id": str(ref.resource_id),
        "version": ref.version,
    }


class Trial:
    def __init__(self, case: ToolLoopCase, variant: Variant) -> None:
        self.case = case
        self.variant: Variant = variant
        self.policy = TrialPolicy()
        self.expected_resource_id = uuid4()
        self.project_id = uuid4()
        self.ref_id = uuid4()
        self.specialists: list[str] = []
        self.exploration: list[ContextReadObservation] = []
        self.observation_seen = False

    @property
    def hours(self) -> int:
        return 48 if self.variant == "48h" else 32

    def ref(self, scope: RunScope) -> ContextReference:
        return ContextReference(
            reference_id=self.ref_id,
            organization_id=scope.organization_id,
            resource_type="project",
            resource_id=self.project_id,
            version=3,
            observed_at=FIXTURE_TIME,
        )

    async def delegate(self, ctx: ToolInvocationContext, args: BaseModel) -> BackendToolResult:
        value = DelegateAgentInputV2.model_validate(args)
        self.specialists.append(value.agent_id.value)
        if value.agent_id == AgentId.ASSIGNMENT:
            ref = self.ref(ctx.scope)
            return BackendToolResult(
                evidence=(ref,),
                output=DelegateAgentOutputV2(
                    agent_id=value.agent_id,
                    agent_version="1.1.0",
                    resolved_action="assignment.analyze_workload",
                    child_agent_run_id=uuid4(),
                    status=AgentRunStatus.COMPLETED,
                    summary=f"{self.hours}h/40h",
                    typed_output=WorkloadFact(
                        allocated_hours=self.hours, capacity_hours=40
                    ).model_dump(mode="json"),
                    evidence=(ref,),
                    stop_reason="COMPLETED",
                ),
            )
        if value.agent_id != AgentId.PLANNING or self.case.scenario not in {"waits", "workload"}:
            raise ValueError("UNEXPECTED_SPECIALIST")
        return BackendToolResult(
            wait_ref=WaitReference(
                wait_id=uuid4(),
                scope=ctx.scope,
                source_invocation_id=ctx.invocation_id,
                kind="WORKFLOW_RESULT" if self.variant == "workflow" else "HUMAN_DECISION",
                resource_id=self.expected_resource_id,
                resource_version=3,
            )
        )

    async def explore(self, ctx: ToolInvocationContext, args: BaseModel) -> BackendToolResult:
        assert isinstance(args, ExploreProjectsInputV103)
        assert args.include_overview_if_unique
        refs = tuple(
            ContextRef(
                ref_id=uuid4(),
                scope=ctx.scope,
                resource_type="project",
                resource_id=uuid4(),
                version=1,
                fingerprint="b" * 64,
                observed_at=FIXTURE_TIME,
                invocation_id=ctx.invocation_id,
                source="project_service",
            )
            for _ in range(2)
        )
        partial = self.variant == "partial" and args.cursor is None
        if args.cursor is not None and (
            self.variant != "partial" or args.cursor != "opaque-next" or len(self.exploration) != 1
        ):
            raise ValueError("PAGINATION_NOT_OBSERVED")
        observation = ContextReadObservation(
            scope=ctx.scope,
            objective_ref=ctx.scope.turn_id,
            invocation_id=ctx.invocation_id,
            tool_reference="explore_projects@1",
            tool_version=ctx.tool_version,
            observed_at=FIXTURE_TIME,
            outcome="PARTIAL" if partial else "NEEDS_INPUT",
            context_refs=refs,
            facts=tuple(
                ContextFact(ref_id=ref.ref_id, field="project.name", value=f"Sao Mai {index + 1}")
                for index, ref in enumerate(refs)
            ),
            entities=(
                EntityResolution(
                    query_fingerprint="a" * 64,
                    resolution="AMBIGUOUS",
                    candidate_ref_ids=tuple(ref.ref_id for ref in refs),
                ),
            ),
            coverage=(
                ReadCoverage(
                    requested=("overview",),
                    returned=(),
                    status="PARTIAL" if partial else "COMPLETE",
                    query_fingerprint="a" * 64,
                    cursor="opaque-next" if partial else None,
                    truncated=partial,
                ),
            ),
            gaps=(ContextGap(code="PAGINATION"),) if partial else (),
        )
        self.exploration.append(observation)
        return BackendToolResult(output=observation)

    def decide(self, request: ToolModelRequest[BaseModel], scope: RunScope) -> dict[str, object]:
        vi = self.case.locale == "vi"
        observed = tuple(item for item in request.items if item.kind == "tool_result")
        match self.case.scenario:
            case "greeting":
                return answer_fixture("Xin chào!" if vi else "Hello!")
            case "fresh-context":
                return answer_fixture(
                    "Dự án đã chọn." if vi else "The selected project.",
                    citations=[_citation(self.ref(scope))],
                )
            case "ambiguity":
                if not observed:
                    return _call(
                        "explore",
                        "explore_projects",
                        {"query": "Sao Mai", "include_overview_if_unique": True},
                    )
                content = json.dumps(tuple(item.content for item in observed), ensure_ascii=False)
                self.observation_seen = (
                    self.variant.upper() in content
                    and "Sao Mai 1" in content
                    and "project.overview" not in content
                )
                if not self.observation_seen:
                    raise ValueError("EXPLORATION_OBSERVATION_MISSING")
                return answer_fixture(
                    "Bạn muốn dự án Sao Mai nào?" if vi else "Which Sao Mai project do you mean?",
                    outcome="NEEDS_INPUT",
                )
            case "workload":
                if not observed:
                    return _delegate(
                        AgentId.ASSIGNMENT, "Kiểm tra tải" if vi else "Check workload", []
                    )
                workload = next((item for item in observed if item.call_id == "assignment"), None)
                if workload is None:
                    raise ValueError("WORKLOAD_OBSERVATION_MISSING")
                observation = DelegateAgentOutputV2.model_validate(workload.content)
                fact = WorkloadFact.model_validate(observation.typed_output)
                self.observation_seen = (fact.allocated_hours, fact.capacity_hours) == (
                    self.hours,
                    40,
                ) and observation.evidence == (self.ref(scope),)
                if not self.observation_seen:
                    raise ValueError("WORKLOAD_OBSERVATION_MISSING")
                if fact.allocated_hours > fact.capacity_hours:
                    return _delegate(
                        AgentId.PLANNING,
                        "Đề xuất điều chỉnh" if vi else "Propose a change",
                        [str(self.ref_id)],
                    )
                return answer_fixture(
                    "32h/40h, chưa quá tải." if vi else "32h/40h, no overload.",
                    citations=[_citation(self.ref(scope))],
                )
            case "waits":
                return _delegate(
                    AgentId.PLANNING, "Chuẩn bị kế hoạch" if vi else "Prepare plan", []
                )
            case "revoked":
                # The schema has already arrived at the mock model before the permission changes.
                self.policy.denial = (
                    "CONTEXT_STALE" if self.variant == "stale" else "PERMISSION_DENIED"
                )
                return _delegate(AgentId.PLANNING, "Sửa kế hoạch" if vi else "Revise plan", [])

    async def execute(self, factory: GatewayFactory) -> TrialResult:
        orch, value, scope, lease, _, store, resources, _ = make_harness(
            responses=(),
            locale=self.case.locale,
            message=self.case.prompt,
            use_packaged_manifest=True,
            policy=self.policy,
            backends={
                "delegate_agent@2": FixtureToolBackend(self.delegate),
                "explore_projects@1": FixtureToolBackend(self.explore),
            },
        )
        gateway = RecordingGateway(
            factory(self.case, self.variant, lambda request: self.decide(request, scope))
        )
        orch.gateway = gateway
        started = perf_counter()
        result = await orch.run_segment(
            value,
            scope=scope,
            checkpoint=None,
            lease=lease,
            verified_context_refs=(self.ref(scope),)
            if self.case.scenario == "fresh-context"
            else (),
        )
        elapsed = (perf_counter() - started) * 1000
        checkpoint = await store.load_checkpoint(scope)
        requested = tuple(
            record.invocation.tool_reference
            for record in store.records
            if record.kind == "tool.requested" and record.invocation is not None
        )
        wait = result.wait_ref
        expected_wait = self.variant in {"48h", "human", "workflow"}
        evidence_count = 0
        if isinstance(result, PublishedResponse):
            evidence_count = sum(
                len(block.evidence)
                for block in result.output.blocks
                if block.kind == "work_evidence"
            )
        fallback = (
            isinstance(result, PublishedResponse)
            and result.output.status == OrchestratorStatus.FAILED
        )
        outcome = (
            "SUSPENDED" if isinstance(result, SuspendedWorkflow) else result.output.status.value
        )

        checks: dict[str, bool] = {
            "MODEL_NOT_EXECUTED": bool(gateway.requests),
            "DELEGATION_SCHEMA_INVALID": all(
                _flat_handoff(request) for request in gateway.requests
            ),
            "UNEXPECTED_SKILL_LOAD": resources.bodies_read == 0,
            "DUPLICATE_INVOCATION": len(requested) == len(store.invocations),
        }
        if self.case.scenario in {"greeting", "fresh-context"}:
            checks["UNNECESSARY_READ"] = not requested
            checks["EXPECTED_ANSWER"] = outcome == "COMPLETED" and wait is None
            checks["EVIDENCE_MISMATCH"] = evidence_count == int(
                self.case.scenario == "fresh-context"
            )
        if self.case.scenario == "ambiguity":
            checks["AMBIGUITY_NOT_OBSERVED"] = self.observation_seen
            checks["UNEXPECTED_OVERVIEW_OR_DELEGATION"] = (
                1 <= len(self.exploration) <= (2 if self.variant == "partial" else 1)
                and not self.specialists
                and all(
                    fact.field == "project.name"
                    for observation in self.exploration
                    for fact in observation.facts
                )
            )
            checks["UNRESOLVED_PROJECT_MUST_ASK"] = wait is not None and wait.kind == "USER_INPUT"
        if self.case.scenario == "workload":
            checks["WORKLOAD_NOT_OBSERVED"] = self.observation_seen
            checks["WRONG_WORKLOAD_BRANCH"] = self.specialists == (
                ["assignment", "planning"] if expected_wait else ["assignment"]
            ) and (
                expected_wait or (outcome == "COMPLETED" and evidence_count == 1 and wait is None)
            )
        if expected_wait:
            expected_kind = "WORKFLOW_RESULT" if self.variant == "workflow" else "HUMAN_DECISION"
            checks["EXPECTED_DURABLE_WAIT"] = (
                wait is not None
                and wait.kind == expected_kind
                and wait.resource_id == self.expected_resource_id
                and wait.resource_version == 3
                and checkpoint is not None
                and checkpoint.pending_wait == wait
                and outcome
                == ("SUSPENDED" if expected_kind == "WORKFLOW_RESULT" else "AWAITING_HUMAN")
            )
            checks["MODEL_RAN_AFTER_WAIT"] = len(gateway.requests) == (
                2 if self.variant == "48h" else 1
            )
        if self.case.scenario == "revoked":
            checks["REVOKED_SPECIALIST_EXECUTED"] = not self.specialists
            checks["MISSING_RUNTIME_DENIAL"] = (
                fallback
                and self.policy.denial is not None
                and self.policy.rejections == [self.policy.denial]
                and len(gateway.requests) == 1
            )
        metered = (
            len(gateway.responses) == len(gateway.requests)
            and bool(gateway.responses)
            and all(u is not None for u in gateway.responses)
        )
        return TrialResult(
            variant=self.variant,
            outcome=outcome,
            requested_tools=requested,
            specialists=tuple(self.specialists),
            wait_kind=wait.kind if wait else None,
            wait_version=wait.resource_version if wait else None,
            evidence_count=evidence_count,
            model_requests=len(gateway.requests),
            skill_body_reads=resources.bodies_read,
            fallback=fallback,
            latency_ms=round(elapsed, 3),
            input_tokens=sum(u.input_tokens for u in gateway.responses if u is not None)
            if metered
            else None,
            output_tokens=sum(u.output_tokens for u in gateway.responses if u is not None)
            if metered
            else None,
            violations=tuple(name for name, passed in checks.items() if not passed),
        )


class LegacyCaseIdentity(BaseModel):
    case_id: str
    locale: Locale
    scenario: str
    schema_version: Literal["1.0"]
    evidence_status: Literal["AUTHORED_NOT_EXECUTED"]


def _legacy_evidence() -> tuple[int, str, bool]:
    raw = _CORPUS.read_bytes()
    rows = tuple(
        LegacyCaseIdentity.model_validate_json(line) for line in raw.splitlines() if line.strip()
    )
    paired = Counter((row.scenario, row.locale) for row in rows)
    unique = {row.case_id for row in rows}
    valid = (
        len(rows) == len(unique) == 60
        and len({row.scenario for row in rows}) == 30
        and all(n == 1 for n in paired.values())
    )
    return len(rows), sha256(raw).hexdigest(), valid


async def evaluate_tool_loop(
    cases: Sequence[ToolLoopCase] = REQUIRED_CASES,
    gateway_factory: GatewayFactory = lambda case, variant, decide: FixtureGateway(decide),
) -> ToolLoopEvaluationReport:
    declared = {case.case_id: case for case in cases}
    violations: list[str] = []
    if len(declared) != len(cases):
        violations.append("DUPLICATE_CASE")
    if declared.keys() - {case.case_id for case in REQUIRED_CASES}:
        violations.append("UNKNOWN_CASE")
    if any(not case.prompt.strip() for case in cases):
        violations.append("EMPTY_PROMPT")
    legacy_count, legacy_digest, legacy_valid = _legacy_evidence()
    if not legacy_valid:
        violations.append("LEGACY_CORPUS_INVALID")

    orch, *_ = make_harness(responses=(), use_packaged_manifest=True)
    registered = orch.catalog.agent
    descriptors = await ORCHESTRATOR_TOOL_LOOP_EVALUATORS[EVALUATOR_ID]()
    declaration_ok = (
        isinstance(descriptors, dict) and cast(dict[str, object], descriptors).get("passed") is True
    )
    context = await CONTEXT_EXPLORATION_EVALUATORS["context_exploration@1"]()
    if not isinstance(context, ContextExplorationEvaluationReport):
        raise ValueError("CONTEXT_EVALUATOR_REPORT_INVALID")
    skill = SkillRegistry.from_resources(((_SKILL_PACKAGE, "skill.yaml"),)).resolve(
        "explore_work_context@1"
    )
    if skill.bundle is None:
        raise ValueError("SKILL_DESCRIPTOR_UNAVAILABLE")
    results: list[CaseResult] = []
    for requirement in REQUIRED_CASES:
        case = declared.get(requirement.case_id)
        if case is None:
            results.append(
                CaseResult(
                    case_id=requirement.case_id,
                    locale=requirement.locale,
                    scenario=requirement.scenario,
                    status="NOT_RUN",
                )
            )
            continue
        trials = tuple(
            [
                await Trial(case, variant).execute(gateway_factory)
                for variant in _VARIANTS[case.scenario]
            ]
        )
        results.append(
            CaseResult(
                case_id=case.case_id,
                locale=case.locale,
                scenario=case.scenario,
                status="FAIL" if any(trial.violations for trial in trials) else "PASS",
                trials=trials,
            )
        )
    case_identity = sorted((case.case_id, case.prompt, _VARIANTS[case.scenario]) for case in cases)
    return ToolLoopEvaluationReport(
        revision=context.revision,
        workspace_dirty=context.workspace_dirty,
        manifest_version=registered.manifest.agent.version,
        manifest_sha256=registered.fingerprint,
        agent_schema_version=registered.manifest.schema_version,
        tool_pins=registered.tool_pins,
        skill_pin=registered.skill_pins[0],
        skill_descriptor_sha256=canonical_manifest_fingerprint(skill.bundle),
        skill_body_sha256=skill.bundle.instructions.sha256,
        case_set_sha256=sha256(json.dumps(case_identity, ensure_ascii=False).encode()).hexdigest(),
        legacy_authored_cases=legacy_count,
        legacy_corpus_sha256=legacy_digest,
        context_prerequisite_passed=context.passed,
        context_prerequisite_counts={str(key): number for key, number in context.counts.items()},
        context_evaluator_version=context.evaluator_version,
        context_corpus_sha256=context.corpus_sha256,
        manifest_evaluator_passed=declaration_ok,
        cases=tuple(results),
        violations=tuple(violations),
        passed=(
            not violations
            and context.passed
            and declaration_ok
            and all(result.status == "PASS" for result in results)
        ),
    )


def main() -> int:
    report = asyncio.run(evaluate_tool_loop())
    print(report.model_dump_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
