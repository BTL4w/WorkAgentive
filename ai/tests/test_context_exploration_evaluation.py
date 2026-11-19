import json
from pathlib import Path

import pytest

from work_management_ai.evaluation import context_exploration as evaluation


@pytest.mark.asyncio
async def test_default_cases_run_bilingually_and_preserve_unmeasured_model_quality() -> None:
    report = await evaluation.evaluate_context_exploration()
    assert report.passed
    assert {case.locale for case in report.cases} == {"vi", "en"}
    assert {case.execution_mode for case in report.cases} == {"STATIC", "DETERMINISTIC"}
    assert all(case.status == "PASS" and case.assertions for case in report.cases)
    assert report.model_selection == report.hosted_quality == "NOT_RUN"
    assert report.corpus_sha256 and report.skill_pin.version == "1.0.3"
    assert report.counts == {"PASS": 40, "FAIL": 0, "NOT_RUN": 0}
    assert {pin.reference for pin in report.tool_pins} == {
        "explore_projects@1",
        "read_project_context@1",
        "read_actor_context@1",
        "load_skill@1",
    }
    for scenario in ("skill_load", "skill_missing_resource", "skill_revoked"):
        assert sum(case.case_id.startswith(f"context_{scenario}_") for case in report.cases) == 2


@pytest.mark.asyncio
async def test_reports_from_before_the_corpus_expansion_still_validate() -> None:
    report = await evaluation.evaluate_context_exploration()
    payload = report.model_dump(mode="json")
    payload["evaluator_version"] = "1.0.0"
    baseline_ids = {
        f"context_{scenario}_{locale}@1"
        for scenario in evaluation.BASELINE_SCENARIOS
        for locale in ("vi", "en")
    }
    payload["cases"] = [case for case in payload["cases"] if case["case_id"] in baseline_ids]
    payload["tool_pins"] = [
        pin for pin in payload["tool_pins"] if pin["reference"] != "load_skill@1"
    ]
    payload["counts"] = {"PASS": 24, "FAIL": 0, "NOT_RUN": 0}
    baseline = evaluation.ContextExplorationEvaluationReport.model_validate(payload)
    assert baseline.passed and len(baseline.cases) == 24


def test_cli_saves_distinct_reports_for_same_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for _ in range(2):
        assert evaluation.main(["--output-dir", str(tmp_path)]) == 0
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) == 2
    reports = [json.loads(p.read_text()) for p in paths]
    assert reports[0]["run_id"] != reports[1]["run_id"]
    assert reports[0]["revision"] == reports[1]["revision"]
    output = capsys.readouterr().out
    assert all(str(p) in output for p in paths)


def test_missing_required_case_fails_and_still_writes_report(tmp_path: Path) -> None:
    lines = evaluation.DEFAULT_CORPUS.read_text().splitlines()
    corpus = tmp_path / "missing.jsonl"
    corpus.write_text("\n".join(lines[1:]) + "\n")
    output = tmp_path / "output"
    assert evaluation.main(["--corpus", str(corpus), "--output-dir", str(output)]) == 1
    [path] = list(output.glob("*.json"))
    report = json.loads(path.read_text())
    assert not report["passed"]
    assert report["counts"]["NOT_RUN"] == 1
    assert any(case["violations"] == ["CASE_MISSING"] for case in report["cases"])


@pytest.mark.asyncio
async def test_unknown_case_is_rejected_without_including_raw_case_data(tmp_path: Path) -> None:
    corpus = tmp_path / "unknown.jsonl"
    hidden = "private-fixture-sentinel"
    extra = {"case_id": hidden, "scenario": "hello", "locale": "vi", "request": hidden}
    corpus.write_text(evaluation.DEFAULT_CORPUS.read_text() + json.dumps(extra) + "\n")
    report = await evaluation.evaluate_context_exploration(corpus)
    assert not report.passed
    assert report.corpus_violations == ("CASE_UNKNOWN",)
    assert hidden not in report.model_dump_json()


@pytest.mark.asyncio
async def test_required_case_without_runner_is_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(evaluation.SCENARIO_RUNNERS, "partial")
    report = await evaluation.evaluate_context_exploration()
    assert not report.passed and report.counts["NOT_RUN"] == 2
    assert sum(case.violations == ("CASE_UNEXECUTED",) for case in report.cases) == 2


@pytest.mark.asyncio
async def test_applicability_detects_missing_localized_playbook_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = evaluation.read_skill_instructions

    def without_skip_rule(package: str, manifest: evaluation.SkillManifestV2) -> str:
        return "\n".join(
            line
            for line in original(package, manifest).splitlines()
            if not line.startswith("Skip standalone")
        )

    monkeypatch.setattr(evaluation, "read_skill_instructions", without_skip_rule)
    report = await evaluation.evaluate_context_exploration()
    assert not report.passed
    english = next(case for case in report.cases if case.case_id == "context_hello_en@1")
    assert english.execution_mode == "STATIC"
    assert english.status == "FAIL"
    assert any(a.name == "playbook_applicability" and not a.passed for a in english.assertions)


def test_cli_returns_nonzero_when_report_cannot_be_written(tmp_path: Path) -> None:
    occupied = tmp_path / "file"
    occupied.touch()
    assert evaluation.main(["--output-dir", str(occupied)]) == 1


def test_cli_persists_failed_assertion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from work_management_ai.evaluation.context_exploration_cases import Checks, Fixture

    async def failed_assertion(fixture: Fixture) -> Checks:
        return {"no_context_requested": False}

    monkeypatch.setitem(evaluation.SCENARIO_RUNNERS, "hello", failed_assertion)
    assert evaluation.main(["--output-dir", str(tmp_path)]) == 1
    [path] = list(tmp_path.glob("*.json"))
    data = evaluation.ContextExplorationEvaluationReport.model_validate_json(path.read_bytes())
    assert data.counts["FAIL"] == 2
    assert data.counts["NOT_RUN"] == 0
    assert not data.passed
