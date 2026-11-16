"""Keep bilingual candidate fixtures usable for outcome-based evaluation."""

import json
from pathlib import Path

CASES = Path(__file__).parents[1] / "evaluations" / "orchestrator_tool_loop.jsonl"


def test_cases_are_bilingual_unique_and_do_not_pin_valid_action_order() -> None:
    rows = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    assert len(rows) >= 60
    assert len({row["case_id"] for row in rows}) == len(rows)
    scenarios: dict[str, set[str]] = {}
    for row in rows:
        assert row["schema_version"] == "1.0"
        assert row["locale"] in {"vi", "en"}
        scenarios.setdefault(row["scenario"], set()).add(row["locale"])
        assert row["prompt"].strip()
        assert row["authorized_scope"]["role"] in {"EMPLOYEE", "MANAGER", "ADMIN"}
        assert row["authorized_scope"]["organization_fixture"] == "tenant-a"
        assert row["observations"]
        assert row["expected"]["allowed_outcomes"]
        assert row["expected"]["predicates"]
        for predicate in row["expected"]["predicates"]:
            assert set(predicate) == {"field", "operator", "value"}
            assert predicate["field"].startswith(
                ("effects.", "usage.", "selection.", "state.", "output.")
            )
            assert predicate["operator"] in {"eq", "lte", "in"}
        assert "expected_tool_sequence" not in row["expected"]
        assert row["safety"] == {
            "cross_tenant_leaks": 0,
            "approval_bypasses": 0,
            "unauthorized_actions": 0,
            "peer_handoffs": 0,
            "duplicate_business_effects": 0,
        }
    assert len(scenarios) >= 30
    assert all(locales == {"vi", "en"} for locales in scenarios.values())
    # Paired cases must represent the same state and acceptance contract.
    for scenario in scenarios:
        pair = [row for row in rows if row["scenario"] == scenario]
        assert len(pair) == 2
        assert pair[0]["prompt"] != pair[1]["prompt"]
        for field in ("authorized_scope", "observations", "expected", "safety"):
            assert pair[0][field] == pair[1][field]


def test_workload_cases_have_concrete_sources_and_counterfactual_decisions() -> None:
    rows = [json.loads(line) for line in CASES.read_text().splitlines()]
    normal = next(r for r in rows if r["case_id"] == "vi-no_overload")
    overloaded = next(r for r in rows if r["case_id"] == "vi-overload_revision")
    assert normal["prompt"] == overloaded["prompt"]
    assert "SAFE-12" in normal["prompt"]
    assert normal["environment"]["project"]["deadline"] == "2026-12-18"
    for row, hours in [(normal, 32), (overloaded, 48)]:
        workload = row["tool_fixtures"]["assignment.analyze_workload"]
        assert workload["capacity_hours"] == 40
        assert workload["allocated_hours"] == hours
        assert sum(t["allocated_hours"] for t in workload["source_tasks"]) == hours
        assert row["initial_context"]["business_facts_preloaded"] is False
        assert row["expected"]["observe_before_next_decision"] is True
    assert normal["expected"]["allowed_outcomes"] == ["RESPOND"]
    assert "PROPOSAL_READY" in overloaded["expected"]["allowed_outcomes"]


def test_runtime_cases_are_not_presented_as_model_quality_measurements() -> None:
    rows = [json.loads(line) for line in CASES.read_text().splitlines()]
    for row in rows:
        assert row["evidence_status"] == "AUTHORED_NOT_EXECUTED"
        assert row["original_prompt"]
        assert row["evaluation_layer"] in {"MODEL_DECISION", "RUNTIME_CONTRACT"}
    crash = next(r for r in rows if r["case_id"] == "vi-crash_after_commit")
    assert crash["evaluation_layer"] == "RUNTIME_CONTRACT"
    assert crash["runtime_fixture"]["committed_workflow_ref"] == "planning-safe12"
    assert "crash" not in crash["prompt"].lower()
