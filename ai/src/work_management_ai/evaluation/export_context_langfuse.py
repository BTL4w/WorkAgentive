"""Export a saved, validated Explore report to Langfuse without rerunning it."""

import argparse
import importlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import ValidationError

from work_management_ai.evaluation.context_exploration import (
    ContextExplorationEvaluationReport,
    ExecutionMode,
    Locale,
    Status,
    Violation,
)
from work_management_ai.evaluation.context_exploration_cases import AssertionName


class SafeAssertion(TypedDict):
    name: AssertionName
    passed: bool


class SafeCase(TypedDict):
    case_id: str
    locale: Locale
    execution_mode: ExecutionMode
    status: Status
    assertions: list[SafeAssertion]
    violations: tuple[Violation, ...]


class ExportItem(TypedDict):
    input: SafeCase


class ExportTask(Protocol):
    def __call__(self, *, item: ExportItem, **kwargs: object) -> SafeCase: ...


class EvaluationFactory(Protocol):
    def __call__(
        self, *, name: str, value: str | float, data_type: Literal["CATEGORICAL", "NUMERIC"]
    ) -> object: ...


class ExperimentEvaluator(Protocol):
    def __call__(self, *, output: SafeCase, **kwargs: object) -> list[object]: ...


class ExperimentItemResult(Protocol):
    @property
    def evaluations(self) -> Sequence[object]: ...


class ExperimentResult(Protocol):
    @property
    def experiment_url(self) -> str | None: ...

    @property
    def item_results(self) -> Sequence[ExperimentItemResult]: ...


class ExperimentClient(Protocol):
    def auth_check(self) -> bool: ...

    def run_experiment(
        self,
        *,
        name: str,
        run_name: str,
        experiment_id: str,
        data: Sequence[ExportItem],
        task: ExportTask,
        evaluators: list[ExperimentEvaluator],
        metadata: dict[str, str],
    ) -> ExperimentResult: ...

    def flush(self) -> None: ...


class ExportError(ValueError):
    pass


def _saved_item(*, item: ExportItem, **_: object) -> SafeCase:
    return item["input"]


def _safe_link(url: str | None, experiment_id: str) -> str | None:
    if url:
        parts = urlsplit(url)
        if (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and not any((parts.username, parts.password, parts.fragment))
            and (
                not parts.query
                or (
                    parts.path.endswith("/experiments/results")
                    and parse_qsl(parts.query, keep_blank_values=True)
                    == [("baseline", experiment_id)]
                )
            )
        ):
            return url
    return None


def export_report(
    report: ContextExplorationEvaluationReport,
    client: ExperimentClient,
    *,
    evaluation_factory: EvaluationFactory | None = None,
) -> str | None:
    report = ContextExplorationEvaluationReport.model_validate(report)
    cases: list[ExportItem] = [
        {
            "input": {
                "case_id": case.case_id,
                "locale": case.locale,
                "execution_mode": case.execution_mode,
                "status": case.status,
                "assertions": [
                    {"name": assertion.name, "passed": assertion.passed}
                    for assertion in case.assertions
                ],
                "violations": case.violations,
            }
        }
        for case in report.cases
    ]
    if not client.auth_check():
        raise ExportError("LANGFUSE_AUTH_FAILED")
    evaluators: list[ExperimentEvaluator] = []
    if evaluation_factory is not None:

        def scores(*, output: SafeCase, **_: object) -> list[object]:
            status = output["status"]
            values = [evaluation_factory(name="case_status", value=status, data_type="CATEGORICAL")]
            if status != "NOT_RUN":
                values.append(
                    evaluation_factory(
                        name="executed_case_pass",
                        value=float(status == "PASS"),
                        data_type="NUMERIC",
                    )
                )
            return values

        evaluators.append(scores)
    result = client.run_experiment(
        name="WorkCrew Explore evaluator",
        run_name=f"task-8.4-{report.run_id.hex}",
        experiment_id=report.run_id.hex,
        data=cases,
        task=_saved_item,
        evaluators=evaluators,
        metadata={
            "revision": report.revision,
            "corpus_sha256": report.corpus_sha256 or "unavailable",
            "evaluator_version": report.evaluator_version,
            "execution_mode": report.mode,
            "model_selection": report.model_selection,
            "hosted_quality": report.hosted_quality,
            "workspace_dirty": str(report.workspace_dirty),
            "gate_passed": str(report.passed),
            "corpus_violations": ",".join(report.corpus_violations),
            "pass": str(report.counts["PASS"]),
            "fail": str(report.counts["FAIL"]),
            "not_run": str(report.counts["NOT_RUN"]),
        },
    )
    client.flush()
    if len(result.item_results) != len(cases) or (
        evaluators
        and sum(len(item.evaluations) for item in result.item_results)
        != len(cases) + report.counts["PASS"] + report.counts["FAIL"]
    ):
        raise ExportError("LANGFUSE_EXPORT_INCOMPLETE")
    return _safe_link(result.experiment_url, report.run_id.hex)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], ExperimentClient] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    if not all(os.environ.get(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        print("LANGFUSE_CREDENTIALS_MISSING")
        return 1
    try:
        report = ContextExplorationEvaluationReport.model_validate_json(args.report.read_bytes())
    except (OSError, ValidationError):
        print("CONTEXT_EVALUATION_REPORT_INVALID")
        return 1
    try:
        evaluation_factory = None
        if client_factory is None:
            sdk = importlib.import_module("langfuse")
            client_factory = cast(Callable[[], ExperimentClient], sdk.get_client)
            evaluation_factory = cast(EvaluationFactory, sdk.Evaluation)
        url = export_report(report, client_factory(), evaluation_factory=evaluation_factory)
    except ModuleNotFoundError:
        print("LANGFUSE_SDK_UNAVAILABLE")
        return 1
    except Exception:
        print("LANGFUSE_EXPORT_FAILED")
        return 1
    print(f"Langfuse experiment exported: {report.run_id.hex}")
    if url:
        print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
