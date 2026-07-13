from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class PlannerState(TypedDict):
    status: str


def _start(state: PlannerState) -> dict:
    return {"status": "done"}


def build_graph():
    builder = StateGraph(PlannerState)
    builder.add_node("start", _start)
    builder.add_edge(START, "start")
    builder.add_edge("start", END)
    return builder.compile(checkpointer=InMemorySaver())
