from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic_secretary.review import review, route_after_review
from agentic_secretary.state import PlannerState


def _base_state(**overrides) -> PlannerState:
    state: PlannerState = {
        "messages": [AIMessage(content="All done.")],
        "emails": [],
        "calendar_events": [],
        "action_items": [],
        "status": "done",
    }
    state.update(overrides)
    return state


def _build_test_graph():
    builder = StateGraph(PlannerState)
    builder.add_node("review", review)
    builder.add_node(
        "agent_stub",
        lambda state: {"messages": [AIMessage(content="agent_stub reached")]},
    )
    builder.add_edge(START, "review")
    builder.add_conditional_edges(
        "review", route_after_review, {"agent": "agent_stub", END: END}
    )
    builder.add_edge("agent_stub", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_review_interrupts_with_the_agents_final_summary():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke(_base_state(), config=config)

    assert result["__interrupt__"][0].value == "All done."


def test_exit_phrase_ends_without_looping_back_to_agent():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="done"), config=config)

    contents = [m.content for m in final["messages"]]
    assert "agent_stub reached" not in contents
    assert contents[-1] == "done"


def test_non_exit_reply_loops_back_to_agent():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="move it to 2pm instead"), config=config)

    contents = [m.content for m in final["messages"]]
    assert "move it to 2pm instead" in contents
    assert contents[-1] == "agent_stub reached"


def test_exit_phrase_matching_is_case_and_whitespace_insensitive():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="  Done  "), config=config)

    contents = [m.content for m in final["messages"]]
    assert "agent_stub reached" not in contents
