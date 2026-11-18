from collections.abc import Sequence
from pathlib import Path

import pytest

from work_management_ai.evaluation import context_exploration as evaluation
from work_management_ai.evaluation import export_context_langfuse as exporter


class ItemResult:
    def __init__(self, scores: list[object]) -> None:
        self.evaluations = scores


class Result:
    def __init__(self, items: Sequence[ItemResult], experiment_id: str) -> None:
        self.item_results = items
        self.experiment_url = (
            f"https://example.invalid/project/test/experiments/results?baseline={experiment_id}"
        )


class FakeClient:
    def __init__(self) -> None:
        self.run_name = ""
        self.data: Sequence[exporter.ExportItem] = ()
        self.scores: list[object] = []
        self.metadata: dict[str, str] = {}
        self.flushed = False

    def auth_check(self) -> bool:
        return True

    def run_experiment(
        self,
        *,
        name: str,
        run_name: str,
        experiment_id: str,
        data: Sequence[exporter.ExportItem],
        task: exporter.ExportTask,
        evaluators: list[exporter.ExperimentEvaluator],
        metadata: dict[str, str],
    ) -> Result:
        self.run_name, self.data = run_name, data
        self.metadata = metadata
        assert name == "WorkCrew Explore evaluator"
        assert experiment_id in run_name and metadata["corpus_sha256"]
        assert task(item=data[0])["status"] == data[0]["input"]["status"]
        items = [
            ItemResult(
                [score for evaluate in evaluators for score in evaluate(output=task(item=item))]
            )
            for item in data
        ]
        self.scores = items[0].evaluations
        return Result(items, experiment_id)

    def flush(self) -> None:
        self.flushed = True


def test_export_uses_saved_safe_items_and_preserves_not_run(tmp_path: Path) -> None:
    corpus = tmp_path / "empty.jsonl"
    corpus.touch()
    target = tmp_path / "reports"
    assert evaluation.main(["--corpus", str(corpus), "--output-dir", str(target)]) == 1
    [path] = list(target.glob("*.json"))
    report = evaluation.ContextExplorationEvaluationReport.model_validate_json(path.read_text())
    client = FakeClient()
    url = exporter.export_report(report, client, evaluation_factory=dict)
    assert url is not None and url.endswith(f"?baseline={report.run_id.hex}")
    assert report.run_id.hex in client.run_name
    assert len(client.data) == len(report.cases)
    assert all(item["input"]["status"] == "NOT_RUN" for item in client.data)
    assert client.scores == [
        {"name": "case_status", "value": "NOT_RUN", "data_type": "CATEGORICAL"}
    ]
    assert client.metadata["gate_passed"] == "False"
    assert client.flushed
    assert all(
        set(item["input"])
        == {"case_id", "locale", "execution_mode", "status", "assertions", "violations"}
        for item in client.data
    )


def test_missing_credentials_fail_without_importing_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert exporter.main([str(tmp_path / "unused.json")]) == 1


def test_export_failure_keeps_the_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    class FailingClient(FakeClient):
        def flush(self) -> None:
            raise RuntimeError("private-service-error")

    assert evaluation.main(["--output-dir", str(tmp_path)]) == 0
    [path] = list(tmp_path.glob("*.json"))
    before = path.read_bytes()
    assert exporter.main([str(path)], client_factory=FailingClient) == 1
    assert path.read_bytes() == before
