from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import interrupt

from agentic_secretary.state import PlannerState

# Deterministic, not LLM-classified: the human is confirming they're done,
# not asking the agent to do anything, so this doesn't need judgment.
_EXIT_PHRASES = {"done", "no", "nothing else", "that's all", "bye"}


def review(state: PlannerState) -> dict:
    summary = state["messages"][-1].content
    reply = interrupt(summary)
    return {"messages": [HumanMessage(content=reply)]}


def route_after_review(state: PlannerState) -> str:
    reply = state["messages"][-1].content
    if reply.strip().lower() in _EXIT_PHRASES:
        return END
    return "agent"
