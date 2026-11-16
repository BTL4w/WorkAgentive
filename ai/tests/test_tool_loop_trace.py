"""Trace verification catches missing observations, premature stops and effects."""

import json
from copy import deepcopy
from pathlib import Path

from work_management_ai.evaluation.tool_loop_trace import validate_tool_loop_trace

PATH = Path(__file__).parents[1] / "evaluations" / "orchestrator_tool_loop_traces.jsonl"


def test_authored_trace_scenarios_are_valid_but_not_runtime_evidence() -> None:
    rows = [json.loads(line) for line in PATH.read_text().splitlines()]
    assert len(rows) == 4
    for trace in rows:
        assert trace["evidence_origin"] == "AUTHORED_NOT_RUNTIME_CAPTURE"
        assert validate_tool_loop_trace(trace) == ()


def test_missing_tool_result_in_next_model_input_is_rejected() -> None:
    trace = json.loads(PATH.read_text().splitlines()[0])
    trace["events"][4]["input_results"] = []
    assert "TOOL_RESULT_NOT_OBSERVED" in validate_tool_loop_trace(trace)


def test_unconsumed_script_and_premature_business_effect_are_rejected() -> None:
    trace = json.loads(PATH.read_text().splitlines()[0])
    broken = deepcopy(trace)
    broken["expected_model_steps"] += 1
    assert "MODEL_STEPS_INCOMPLETE" in validate_tool_loop_trace(broken)
    broken = deepcopy(trace)
    broken["events"].insert(-1, {"kind": "business.commit", "approval_valid": False})
    assert "UNAPPROVED_BUSINESS_EFFECT" in validate_tool_loop_trace(broken)


def test_planning_without_overload_and_omitted_followup_are_rejected() -> None:
    rows = [json.loads(line) for line in PATH.read_text().splitlines()]
    trace = deepcopy(next(t for t in rows if t["overloaded"]))
    trace["overloaded"] = False
    assert "UNNECESSARY_PLANNING" in validate_tool_loop_trace(trace)
    trace = deepcopy(next(t for t in rows if not t["overloaded"]))
    trace["overloaded"] = True
    assert "MISSING_PLANNING_FOLLOWUP" in validate_tool_loop_trace(trace)


def test_workload_snapshot_uses_existing_contract() -> None:
    from work_management_ai.agents.assignment.contracts import ProjectWorkloadSnapshot

    rows = [json.loads(line) for line in PATH.read_text().splitlines()]
    for trace in rows:
        snapshot = ProjectWorkloadSnapshot.model_validate(trace["events"][3]["output"])
        assert snapshot.workloads[0].effective_capacity_hours == 40
        assert (snapshot.workloads[0].allocated_effort_hours > 40) == trace["overloaded"]


def test_approval_flag_cannot_authorize_effect_in_read_and_propose_scenario() -> None:
    trace = json.loads(PATH.read_text().splitlines()[0])
    trace["events"].insert(-1, {"kind": "business.commit", "approval_valid": True})
    assert "UNEXPECTED_BUSINESS_EFFECT" in validate_tool_loop_trace(trace)


def test_wait_is_bound_to_the_returned_proposal_version() -> None:
    rows = [json.loads(line) for line in PATH.read_text().splitlines()]
    trace = next(t for t in rows if t["overloaded"])
    trace["events"][-1]["version"] = 99
    assert "PROPOSAL_BINDING_MISMATCH" in validate_tool_loop_trace(trace)
