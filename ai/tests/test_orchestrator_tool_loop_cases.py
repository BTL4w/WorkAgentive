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
