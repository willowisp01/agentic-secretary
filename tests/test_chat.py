from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic_secretary.chat import _Intent, classify_intent, greet
from agentic_secretary.state import PlannerState


def _base_state(**overrides) -> PlannerState:
    state: PlannerState = {
        "messages": [],
        "emails": [],
        "calendar_events": [],
        "action_items": [],
        "status": "pending",
    }
    state.update(overrides)
    return state


def _build_test_graph():
    builder = StateGraph(PlannerState)
    builder.add_node("greet", greet)
    builder.add_node("in_scope", lambda state: {"messages": []})
    builder.add_edge(START, "greet")
    builder.add_conditional_edges(
        "greet", classify_intent, {"fetch_emails": "in_scope", "greet": "greet"}
    )
    builder.add_edge("in_scope", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_greet_opens_with_a_greeting_on_the_first_turn():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke(_base_state(), config=config)

    assert "AI secretary" in result["__interrupt__"][0].value


def test_greet_reprompts_on_a_later_turn_instead_of_the_opening_message():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke(_base_state(), config=config)

    with patch("agentic_secretary.chat.ChatAnthropic") as mock_chat_anthropic:
        mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = _Intent(
            wants_conflict_check=False
        )
        result = graph.invoke(Command(resume="what's the weather?"), config=config)

    assert "AI secretary" not in result["__interrupt__"][0].value


@patch("agentic_secretary.chat.ChatAnthropic")
def test_classify_intent_routes_in_scope_message_to_fetch_emails(mock_chat_anthropic):
    mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = _Intent(
        wants_conflict_check=True
    )
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="check for conflicts"), config=config)

    assert "__interrupt__" not in final


@patch("agentic_secretary.chat.ChatAnthropic")
def test_classify_intent_routes_out_of_scope_message_back_to_greet(mock_chat_anthropic):
    mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = _Intent(
        wants_conflict_check=False
    )
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="what's the weather?"), config=config)

    assert (
        final["__interrupt__"][0].value == 'Anything else? (e.g. "check for conflicts")'
    )
