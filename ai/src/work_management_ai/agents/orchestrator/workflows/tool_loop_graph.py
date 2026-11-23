"""Generic LangGraph control shell for the native Orchestrator tool loop."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from work_management_ai.agents.orchestrator.tool_loop_contracts import (
    ModelFinalResponse,
    OrchestratorExecutionResult,
)
from work_management_ai.model_gateway.tool_contracts import ToolCall
from work_management_ai.runtime.tool_loop_contracts import ToolOutcome

WORKFLOW_ID = "orchestrator.tool_loop.v1"

type LoopRoute = Literal[
    "prepare",
    "model",
    "execute_tool",
    "observe_result",
    "verify_final",
    "fallback",
    "done",
]


class ToolLoopGraphState(TypedDict):
    step: int
    repair_attempts: int
    transient_retries: int
    repair_hint: str | None
    pending_call: ToolCall | None
    last_outcome: ToolOutcome | None
    candidate_final: ModelFinalResponse | None
    fallback_code: str | None
    fallback_stop_reason: str | None
    result: OrchestratorExecutionResult | None
    route: LoopRoute


class ToolLoopNodeHandlers(Protocol):
    async def prepare_step(self, state: ToolLoopGraphState) -> dict[str, object]: ...
    async def call_model(self, state: ToolLoopGraphState) -> dict[str, object]: ...
    async def execute_tool(self, state: ToolLoopGraphState) -> dict[str, object]: ...
    async def observe_result(self, state: ToolLoopGraphState) -> dict[str, object]: ...
    async def verify_final(self, state: ToolLoopGraphState) -> dict[str, object]: ...
    async def fallback(self, state: ToolLoopGraphState) -> dict[str, object]: ...


def _route(state: ToolLoopGraphState) -> LoopRoute:
    return state["route"]


class ToolLoopGraph:
    """Generic prepare -> model -> pipeline -> observe loop with control boundaries only."""

    def __init__(self, handlers: ToolLoopNodeHandlers) -> None:
        self._compiled = self._compile(handlers)

    async def run(
        self, initial: ToolLoopGraphState, *, recursion_limit: int = 64
    ) -> OrchestratorExecutionResult:
        final_state = await self._compiled.ainvoke(
            initial, config={"recursion_limit": recursion_limit}
        )
        result = final_state["result"]
        if result is None:
            raise RuntimeError("TOOL_LOOP_COMPLETED_WITHOUT_RESULT")
        return result

    @staticmethod
    def _compile(
        handlers: ToolLoopNodeHandlers,
    ) -> CompiledStateGraph[ToolLoopGraphState, None, ToolLoopGraphState, ToolLoopGraphState]:
        builder = StateGraph(ToolLoopGraphState)
        builder.add_node("prepare_step", handlers.prepare_step)
        builder.add_node("call_model", handlers.call_model)
        builder.add_node("execute_tool", handlers.execute_tool)
        builder.add_node("observe_result", handlers.observe_result)
        builder.add_node("verify_final", handlers.verify_final)
        builder.add_node("fallback", handlers.fallback)

        builder.add_edge(START, "prepare_step")
        builder.add_conditional_edges(
            "prepare_step",
            _route,
            {
                "model": "call_model",
                "execute_tool": "execute_tool",
                "fallback": "fallback",
                "done": END,
            },
        )
        builder.add_conditional_edges(
            "call_model",
            _route,
            {
                "execute_tool": "execute_tool",
                "verify_final": "verify_final",
                "prepare": "prepare_step",
                "fallback": "fallback",
            },
        )
        builder.add_conditional_edges(
            "execute_tool",
            _route,
            {
                "observe_result": "observe_result",
                "fallback": "fallback",
            },
        )
        builder.add_conditional_edges(
            "observe_result",
            _route,
            {
                "prepare": "prepare_step",
                "fallback": "fallback",
                "done": END,
            },
        )
        builder.add_conditional_edges(
            "verify_final",
            _route,
            {
                "prepare": "prepare_step",
                "fallback": "fallback",
                "done": END,
            },
        )
        builder.add_edge("fallback", END)
        return builder.compile()
