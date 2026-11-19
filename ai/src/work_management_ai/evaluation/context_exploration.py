"""Execute the Explore fixture corpus and write a versioned, safe run report."""

import argparse
import asyncio
import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from work_management_ai.evaluation.context_exploration_cases import (
    SCENARIO_RUNNERS,
    AssertionName,
    Fixture,
)
from work_management_ai.evaluation.context_exploration_skill_cases import SKILL, SkillToolFixture
from work_management_ai.runtime.manifests import SkillManifestV2
from work_management_ai.runtime.skill_bundle import read_skill_instructions
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.runtime.tool_loop_contracts import VersionPin
from work_management_ai.runtime.tool_registry import ToolRegistry
from work_management_ai.tools.context.contracts import Contract

_AI_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY = _AI_ROOT.parent
DEFAULT_CORPUS = _AI_ROOT / "evaluations/context_exploration.jsonl"
DEFAULT_OUTPUT = _REPOSITORY / "docs/superpowers/verification/orchestrator-runs/task-8.4"
EVALUATOR = "context_exploration@1"
REVISION = re.compile(r"[a-f0-9]{40}")
TOOL_RESOURCES = (
    ("work_management_ai.tools.skills", "load.yaml"),
    *(
        ("work_management_ai.tools.context", f"{resource}.yaml")
        for resource in ("projects", "project", "actor")
    ),
)
# Required coverage is independent of the fixture file and of runner availability.
BASELINE_SCENARIOS = (
    "hello",
    "mixed",
    "unique",
    "ambiguous",
    "missing",
    "partial",
    "stale",
    "injection",
    "unsupported",
    "overflow",
    "cross_scope",
    "actor",
)
REQUIRED_SCENARIOS = (
    *BASELINE_SCENARIOS,
    "cursor_resume",
    "project_exact",
    "project_unavailable",
    "actor_mixed_sections",
    "actor_roles_only",
    "skill_load",
    "skill_missing_resource",
    "skill_revoked",
)
type Locale = Literal["vi", "en"]
_LOCALES: tuple[Locale, ...] = ("vi", "en")
EXPECTED_CASES: dict[str, tuple[str, Locale]] = {
    f"context_{scenario}_{locale}@1": (scenario, locale)
    for scenario in REQUIRED_SCENARIOS
    for locale in _LOCALES
}

type Status = Literal["PASS", "FAIL", "NOT_RUN"]
type ExecutionMode = Literal["STATIC", "DETERMINISTIC"]
type Violation = Literal[
    "CASE_MISSING", "CASE_UNEXECUTED", "CASE_EXECUTION_ERROR", "ASSERTION_FAILED"
]
type CorpusViolation = Literal[
    "CORPUS_UNAVAILABLE",
    "CORPUS_INVALID",
    "CASE_UNKNOWN",
    "CASE_DUPLICATE",
    "CASE_IDENTITY_MISMATCH",
    "MANIFEST_CASE_MISMATCH",
    "CASE_LOCALE_DUPLICATE",
]
type Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Applicability(Contract):
    expected: Literal["SKIP", "USE"]
    guidance: str = Field(min_length=1, max_length=300)


class Case(Contract):
    case_id: str
    locale: Locale
    scenario: str
    request: str = Field(min_length=1, max_length=500)
    applicability: Applicability | None = None


class Assertion(Contract):
    name: AssertionName
    passed: bool


class CaseResult(Contract):
    case_id: str
    locale: Locale
    execution_mode: ExecutionMode
    status: Status
    assertions: tuple[Assertion, ...] = ()
    violations: tuple[Violation, ...] = ()

    @model_validator(mode="after")
    def verify_result(self) -> Self:
        expected = EXPECTED_CASES.get(self.case_id)
        if expected is None or expected[1] != self.locale:
            raise ValueError("CASE_IDENTITY_MISMATCH")
        if self.execution_mode != _mode(expected[0]):
            raise ValueError("CASE_MODE_MISMATCH")
        if (self.status == "PASS") != (
            bool(self.assertions) and all(a.passed for a in self.assertions) and not self.violations
        ):
            raise ValueError("CASE_STATUS_MISMATCH")
        if self.status == "NOT_RUN" and self.assertions:
            raise ValueError("CASE_STATUS_MISMATCH")
        return self


class ContextExplorationEvaluationReport(Contract):
    schema_version: Literal["1.0"] = "1.0"
    evaluator: Literal["context_exploration@1"] = EVALUATOR
    evaluator_version: Literal["1.0.0", "1.1.0"] = "1.1.0"
    run_id: UUID
    started_at: AwareDatetime
    revision: str = Field(pattern=r"^(?:[a-f0-9]{40}|unknown)$")
    workspace_dirty: bool | None
    corpus_sha256: Sha256 | None
    skill_pin: VersionPin
    tool_pins: tuple[VersionPin, ...]
    mode: Literal["SYNTHETIC_OFFLINE"] = "SYNTHETIC_OFFLINE"
    model_selection: Literal["NOT_RUN"] = "NOT_RUN"
    hosted_quality: Literal["NOT_RUN"] = "NOT_RUN"
    cases: tuple[CaseResult, ...]
    counts: dict[Status, int]
    corpus_violations: tuple[CorpusViolation, ...]
    passed: bool

    @model_validator(mode="after")
    def verify_report(self) -> Self:
        ids = {case.case_id for case in self.cases}
        scenarios = BASELINE_SCENARIOS if self.evaluator_version == "1.0.0" else REQUIRED_SCENARIOS
        expected_ids = {
            f"context_{scenario}_{locale}@1" for scenario in scenarios for locale in _LOCALES
        }
        if ids != expected_ids or len(ids) != len(self.cases):
            raise ValueError("CASE_INVENTORY_MISMATCH")
        if self.counts != _counts(self.cases) or self.passed != _passed(
            self.cases, self.corpus_violations
        ):
            raise ValueError("EVALUATION_COUNTS_MISMATCH")
        return self


def _mode(scenario: str) -> ExecutionMode:
    return "STATIC" if scenario == "hello" else "DETERMINISTIC"


def _counts(results: Sequence[CaseResult]) -> dict[Status, int]:
    counts = Counter(row.status for row in results)
    return {"PASS": counts["PASS"], "FAIL": counts["FAIL"], "NOT_RUN": counts["NOT_RUN"]}


def _passed(results: Sequence[CaseResult], violations: Sequence[CorpusViolation]) -> bool:
    return not violations and bool(results) and all(case.status == "PASS" for case in results)


def _revision() -> tuple[str, bool | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    return (head, bool(status)) if REVISION.fullmatch(head) else ("unknown", None)


def _load_corpus(path: Path) -> tuple[dict[str, Case], Sha256 | None, set[CorpusViolation]]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}, None, {"CORPUS_UNAVAILABLE"}
    digest = hashlib.sha256(raw).hexdigest()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return {}, digest, {"CORPUS_INVALID"}
    cases: dict[str, Case] = {}
    violations: set[CorpusViolation] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            case = Case.model_validate_json(line)
        except ValidationError:
            violations.add("CORPUS_INVALID")
            continue
        expected = EXPECTED_CASES.get(case.case_id)
        if expected is None:
            violations.add("CASE_UNKNOWN")
        elif expected != (case.scenario, case.locale):
            violations.add("CASE_IDENTITY_MISMATCH")
        elif case.case_id in cases:
            violations.add("CASE_DUPLICATE")
        else:
            cases[case.case_id] = case
    for scenario in REQUIRED_SCENARIOS:
        vi = cases.get(f"context_{scenario}_vi@1")
        en = cases.get(f"context_{scenario}_en@1")
        if vi and en and vi.request.strip().casefold() == en.request.strip().casefold():
            violations.add("CASE_LOCALE_DUPLICATE")
    return cases, digest, violations


def _check_applicability(case: Case, instructions: str) -> Assertion | None:
    expected = {"hello": "SKIP", "mixed": "USE"}.get(case.scenario)
    if expected is None:
        return None
    # Check the authored playbook against independently labeled fixture guidance.
    # User text is deliberately not classified here.
    normalized = "\n".join(line.strip() for line in instructions.splitlines())
    vi, separator, en = normalized.partition("\nen:")
    localized = vi if case.locale == "vi" else "en:" + en
    return Assertion(
        name="playbook_applicability",
        passed=bool(
            separator
            and case.applicability
            and case.applicability.expected == expected
            and case.applicability.guidance in localized.splitlines()
        ),
    )


async def _execute_case(
    case_id: str,
    scenario: str,
    locale: Locale,
    case: Case | None,
    skill_tools: SkillToolFixture,
    instructions: str,
) -> CaseResult:
    mode = _mode(scenario)
    runner = SCENARIO_RUNNERS.get(scenario)
    if case is None or runner is None:
        return CaseResult(
            case_id=case_id,
            locale=locale,
            execution_mode=mode,
            status="NOT_RUN",
            violations=("CASE_MISSING" if case is None else "CASE_UNEXECUTED",),
        )
    try:
        checks = await runner(
            Fixture(case_id, case.request, skill_tools.agent.tool_pins, skill_tools)
        )
        assertions = tuple(Assertion(name=name, passed=value) for name, value in checks.items())
    except Exception:
        return CaseResult(
            case_id=case_id,
            locale=locale,
            execution_mode=mode,
            status="FAIL",
            violations=("CASE_EXECUTION_ERROR",),
        )
    if not assertions:
        return CaseResult(
            case_id=case_id,
            locale=locale,
            execution_mode=mode,
            status="NOT_RUN",
            violations=("CASE_UNEXECUTED",),
        )
    applicability = _check_applicability(case, instructions)
    if applicability is not None:
        assertions = (*assertions, applicability)
    passed = all(assertion.passed for assertion in assertions)
    return CaseResult(
        case_id=case_id,
        locale=locale,
        execution_mode=mode,
        status="PASS" if passed else "FAIL",
        assertions=assertions,
        violations=() if passed else ("ASSERTION_FAILED",),
    )


async def evaluate_context_exploration(
    corpus: Path = DEFAULT_CORPUS,
) -> ContextExplorationEvaluationReport:
    started_at = datetime.now(UTC)
    cases, digest, violations = _load_corpus(corpus)
    registry = SkillRegistry.from_resources(
        (("work_management_ai.skills.explore_work_context", "skill.yaml"),)
    )
    skill = registry.resolve(SKILL)
    if not isinstance(skill.manifest, SkillManifestV2) or skill.bundle is None:
        raise ValueError("CONTEXT_EVALUATION_SKILL_INVALID")
    instructions = read_skill_instructions(skill.bundle.package, skill.manifest)
    if (
        tuple(skill.manifest.evaluators) != (EVALUATOR,)
        or set(skill.manifest.evaluation_cases) != EXPECTED_CASES.keys()
    ):
        violations.add("MANIFEST_CASE_MISMATCH")
    tools = ToolRegistry.from_resources(TOOL_RESOURCES)
    skill_tools = SkillToolFixture(
        registry, tools, CONTEXT_EXPLORATION_EVALUATORS, skill.manifest.allowed_tools
    )
    results = tuple(
        await asyncio.gather(
            *(
                _execute_case(
                    case_id, scenario, locale, cases.get(case_id), skill_tools, instructions
                )
                for case_id, (scenario, locale) in EXPECTED_CASES.items()
            )
        )
    )
    revision, workspace_dirty = _revision()
    return ContextExplorationEvaluationReport(
        run_id=uuid4(),
        started_at=started_at,
        revision=revision,
        workspace_dirty=workspace_dirty,
        corpus_sha256=digest,
        skill_pin=VersionPin(
            reference=SKILL, version=skill.manifest.version, fingerprint=skill.fingerprint
        ),
        tool_pins=skill_tools.agent.tool_pins,
        cases=results,
        counts=_counts(results),
        corpus_violations=tuple(sorted(violations)),
        passed=_passed(results, tuple(violations)),
    )


CONTEXT_EXPLORATION_EVALUATORS: Mapping[str, Callable[[], Awaitable[object]]] = {
    EVALUATOR: evaluate_context_exploration,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(evaluate_context_exploration(args.corpus))
    except (OSError, ValueError):
        print("CONTEXT_EVALUATION_INITIALIZATION_FAILED")
        return 1
    stamp = report.started_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = args.output_dir / f"{stamp}-{report.revision[:7]}-{report.run_id.hex}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2) + "\n")
    except OSError:
        print("CONTEXT_EVALUATION_REPORT_WRITE_FAILED")
        return 1
    print(f"Explore evaluation report: {path.resolve()}")
    print(
        f"PASS={report.counts['PASS']} FAIL={report.counts['FAIL']} "
        f"NOT_RUN={report.counts['NOT_RUN']} model_selection={report.model_selection}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
