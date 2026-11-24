"""Playwright worker entry point using the application runtime and fixture models."""

from decimal import Decimal
from typing import NoReturn
from unittest.mock import patch

from pydantic import BaseModel

from app import worker
from app.core.config import Settings, get_settings
from app.modules.assistant.adapters import tool_loop_runtime
from tests.orchestrator_e2e_data import is_browser_database
from work_management_ai.agents.assignment.contracts import (
    AssignmentAgentOutput,
    ProjectWorkloadSnapshot,
)
from work_management_ai.model_gateway.contracts import (
    ModelGateway,
    ModelUsage,
    StructuredModelRequest,
    StructuredModelResponse,
)
from work_management_ai.model_gateway.errors import ModelUnavailableError
from work_management_ai.model_gateway.tool_contracts import (
    NativeToolGateway,
    ToolCall,
    ToolModelRequest,
    ToolModelResponse,
)
from work_management_ai.runtime.contracts import JsonValue
from work_management_ai.tools.context.contracts import ContextReadObservation
from work_management_ai.tools.delegation.contracts import DelegateAgentOutputV2

_MODEL = "mock:orchestrator-e2e-v1"
_USAGE = ModelUsage(input_tokens=20, output_tokens=10)
_WORKLOAD_PREFIX = "Check workload and rebalance if needed for "
_HOURS = Decimal(40)


def _delegate(
    call_id: str,
    agent: str,
    objective: str,
    reference: str,
    expected_result: str,
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="delegate_agent",
        arguments={
            "agent_id": agent,
            "objective": objective,
            "context_ref_ids": [reference],
            "constraints": [],
            "expected_result": expected_result,
        },
    )


def _answer[OutputT: BaseModel](
    request: ToolModelRequest[OutputT], text: str, reason: str
) -> ToolModelResponse[OutputT]:
    return ToolModelResponse(
        model_ref=_MODEL,
        usage=_USAGE,
        final=request.response_schema.model_validate(
            {
                "outcome": "ANSWER",
                "summary": text,
                "blocks": [{"kind": "text", "text": text, "citations": []}],
                "artifact_state": "NONE",
                "stop_reason": reason,
            }
        ),
    )


class BrowserGateway:
    async def generate_step[OutputT: BaseModel](
        self, request: ToolModelRequest[OutputT]
    ) -> ToolModelResponse[OutputT]:
        message = next(
            (
                item.content
                for item in request.items
                if item.kind == "message" and item.role == "user"
            ),
            None,
        )
        if message == "Hello, Orchestrator":
            return _answer(request, "Hello! How can I help with your work?", "GREETING")
        if not isinstance(message, str) or not message.startswith(_WORKLOAD_PREFIX):
            raise ModelUnavailableError("MOCK_BROWSER_MODEL_UNAVAILABLE")
        project = message.removeprefix(_WORKLOAD_PREFIX)
        observed = {
            item.call_id: item.content for item in request.items if item.kind == "tool_result"
        }
        discovered = observed.get("discover")
        if discovered is None:
            call = ToolCall(
                call_id="discover",
                name="explore_projects",
                arguments={
                    "query": project,
                    "cursor": None,
                    "page_size": 10,
                    "include_overview_if_unique": True,
                },
            )
        else:
            discovery = ContextReadObservation.model_validate(discovered)
            reference = next((item.selected_ref_id for item in discovery.entities), None)
            if reference is None:
                raise ModelUnavailableError("MOCK_PROJECT_NOT_RESOLVED")
            workload = observed.get("workload")
            if workload is None:
                call = _delegate(
                    "workload",
                    "assignment",
                    f"Check the workload for {project}",
                    str(reference),
                    "Current verified hours and capacity",
                )
            elif "rebalance" in observed:
                return _answer(
                    request,
                    "I prepared a proposal for Manager approval.",
                    "PROPOSAL_PENDING",
                )
            else:
                metrics = ProjectWorkloadSnapshot.model_validate(
                    AssignmentAgentOutput.model_validate(
                        DelegateAgentOutputV2.model_validate(workload).typed_output
                    ).deterministic_result
                )
                workload_sample = next(
                    (
                        row
                        for row in metrics.workloads
                        if row.effective_capacity_hours == _HOURS and row.allocated_effort_hours > 0
                    ),
                    None,
                )
                if workload_sample is None:
                    raise ModelUnavailableError("MOCK_WORKLOAD_EVIDENCE_MISSING")
                effort = workload_sample.allocated_effort_hours
                capacity = workload_sample.effective_capacity_hours
                assert capacity is not None
                if effort <= capacity:
                    return _answer(
                        request,
                        f"Workload is {effort}/{capacity} hours. No rebalance is needed.",
                        "WITHIN_CAPACITY",
                    )
                call = _delegate(
                    "rebalance",
                    "planning",
                    f"Create a project plan to rebalance {project} because current effort is "
                    f"{effort}/{capacity} hours. Schedule the launch work over two weeks "
                    "so that the overloaded member can focus on the highest priority tasks.",
                    str(reference),
                    "A project plan proposal for Manager approval",
                )
        return ToolModelResponse(model_ref=_MODEL, tool_call=call, usage=_USAGE)


class SpecialistGateway:
    def __init__(self, fallback: ModelGateway) -> None:
        self._fallback = fallback

    async def generate_structured[OutputT: BaseModel](
        self, request: StructuredModelRequest[OutputT]
    ) -> StructuredModelResponse[OutputT]:
        key = request.invocation_key
        if key.startswith("planning_agent.") and key.endswith(".goal_intake"):
            proposal: dict[str, JsonValue] = {"operation": "CREATE"}
        elif key.startswith("assignment_agent.") and key.endswith(".goal_intake"):
            proposal = {"operation": "ANALYZE_WORKLOAD", "revision_instruction": None}
        elif key.startswith("assignment_agent.") and key.endswith("_explanation"):
            raise ModelUnavailableError("MOCK_EXPLANATION_UNAVAILABLE")
        else:
            return await self._fallback.generate_structured(request)
        return StructuredModelResponse(
            parsed=request.output_schema.model_validate(proposal),
            model_ref=_MODEL,
            usage=_USAGE,
        )


def main() -> NoReturn:
    settings = get_settings()
    if (
        settings.environment != "local"
        or settings.ai_provider != "mock"
        or settings.orchestrator_driver != "tool_loop"
        or not is_browser_database(settings.database_url)
        or len(settings.worker_organization_ids) != 1
        or tuple(settings.worker_organization_ids) != settings.orchestrator_tool_loop_organizations
        or settings.langsmith_tracing
    ):
        raise RuntimeError("MODEL_FIXTURES_REQUIRE_DEDICATED_LOCAL_ORCHESTRATOR_E2E")

    structured = worker.build_model_gateway

    def build_native(_settings: Settings) -> NativeToolGateway:
        return BrowserGateway()

    def build_structured(value: Settings) -> ModelGateway:
        return SpecialistGateway(structured(value))

    with (
        patch.object(tool_loop_runtime, "build_native_gateway", build_native),
        patch.object(worker, "build_model_gateway", build_structured),
    ):
        worker.main()


if __name__ == "__main__":
    main()
