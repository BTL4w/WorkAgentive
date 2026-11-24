"""Acceptance gate executes all required bilingual tool-loop scenarios."""

from dataclasses import replace

import pytest
from pydantic import BaseModel

from work_management_ai.evaluation.orchestrator_tool_loop import (
    REQUIRED_CASES,
    Decision,
    FixtureGateway,
    ToolLoopCase,
    Variant,
    evaluate_tool_loop,
)
from work_management_ai.evaluation.tool_loop_harness import answer_fixture
from work_management_ai.model_gateway.tool_contracts import NativeToolGateway, ToolModelRequest


@pytest.mark.asyncio
async def test_required_bilingual_runtime_cases_pass() -> None:
    report = await evaluate_tool_loop(REQUIRED_CASES)
    assert report.passed
    assert len(report.cases) == 12
    assert report.context_prerequisite_passed
    assert report.legacy_authored_cases == 60
    assert report.case_set_version == "1.1"
    assert report.hosted_quality == "NOT_RUN"
    for case in report.cases:
        assert case.status == "PASS"
        assert case.trials
        assert all(trial.latency_ms >= 0 for trial in case.trials)
    workload = next(case for case in report.cases if case.case_id == "en-workload")
    assert [trial.specialists for trial in workload.trials] == [
        ("assignment",),
        ("assignment", "planning"),
    ]


@pytest.mark.asyncio
async def test_missing_or_duplicate_required_case_fails_gate() -> None:
    report = await evaluate_tool_loop(REQUIRED_CASES[1:])
    assert not report.passed
    assert report.cases[0].status == "NOT_RUN"

    duplicate = replace(REQUIRED_CASES[0], prompt="alternate fixture")
    report = await evaluate_tool_loop((*REQUIRED_CASES, duplicate))
    assert not report.passed
    assert "DUPLICATE_CASE" in report.violations


@pytest.mark.asyncio
async def test_gate_rejects_workload_answer_without_observing_tool_result() -> None:
    def early_answer(case: ToolLoopCase, variant: Variant, decide: Decision) -> NativeToolGateway:
        if case.case_id == "en-workload" and variant == "32h":
            return FixtureGateway(lambda request: answer_fixture("No overload."))
        return FixtureGateway(decide)

    report = await evaluate_tool_loop(REQUIRED_CASES, early_answer)
    assert not report.passed
    failure = next(case for case in report.cases if case.case_id == "en-workload")
    assert "WORKLOAD_NOT_OBSERVED" in failure.trials[0].violations


@pytest.mark.asyncio
async def test_gate_rejects_another_workload_read_instead_of_proposal() -> None:
    def repeat_assignment(
        case: ToolLoopCase, variant: Variant, decide: Decision
    ) -> NativeToolGateway:
        if case.scenario != "workload" or variant != "48h":
            return FixtureGateway(decide)

        def repeat_after_observation(request: ToolModelRequest[BaseModel]) -> dict[str, object]:
            decision = decide(request)
            call = next((item.call for item in request.items if item.call is not None), None)
            results = [item for item in request.items if item.kind == "tool_result"]
            if call is not None and len(results) == 1:
                retry = call.model_dump(mode="json")
                retry["call_id"] = "assignment-again"
                return {"tool_call": retry}
            return decision

        return FixtureGateway(repeat_after_observation)

    report = await evaluate_tool_loop(REQUIRED_CASES, repeat_assignment)
    assert not report.passed
    for case in (case for case in report.cases if case.scenario == "workload"):
        assert case.trials[0].violations == ()
        overload = case.trials[1]
        assert overload.specialists[:2] == ("assignment", "assignment")
        assert "WRONG_WORKLOAD_BRANCH" in overload.violations


@pytest.mark.asyncio
async def test_partial_results_allow_following_an_observed_cursor_before_clarifying() -> None:
    def paginate(case: ToolLoopCase, variant: Variant, decide: Decision) -> NativeToolGateway:
        if case.case_id != "en-ambiguity" or variant != "partial":
            return FixtureGateway(decide)

        def with_next_page(request: ToolModelRequest[BaseModel]) -> dict[str, object]:
            observations = [item for item in request.items if item.kind == "tool_result"]
            if len(observations) == 1 and "opaque-next" in str(observations[0].content):
                return {
                    "tool_call": {
                        "call_id": "explore-next",
                        "name": "explore_projects",
                        "arguments": {
                            "query": "Sao Mai",
                            "include_overview_if_unique": True,
                            "cursor": "opaque-next",
                        },
                    }
                }
            return decide(request)

        return FixtureGateway(with_next_page)

    report = await evaluate_tool_loop(REQUIRED_CASES, paginate)
    assert report.passed
    ambiguity = next(case for case in report.cases if case.case_id == "en-ambiguity")
    assert ambiguity.trials[1].requested_tools == ("explore_projects@1", "explore_projects@1")
