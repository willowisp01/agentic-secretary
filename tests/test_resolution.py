from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from agentic_secretary.resolution import make_agent_node, make_tools_node
from agentic_secretary.review import _latest_proposals
from agentic_secretary.state import (
    CalendarOverlapConflict,
    PlannerState,
    RescheduleRequest,
)
from agentic_secretary.tools import CalendarEvent, EmailSummary, EventProposal

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
EVENT_A = CalendarEvent(
    id="e1", title="Standup", start=NOW, end=NOW + timedelta(minutes=30)
)
EVENT_B = CalendarEvent(
    id="e2",
    title="Client Call",
    start=NOW + timedelta(minutes=15),
    end=NOW + timedelta(minutes=45),
)
OVERLAP = CalendarOverlapConflict(
    description="'Standup' overlaps with 'Client Call'", events=(EVENT_A, EVENT_B)
)


def _base_state(**overrides) -> PlannerState:
    state: PlannerState = {
        "messages": [],
        "emails": [],
        "calendar_events": [],
        "action_items": [OVERLAP],
        "status": "done",
    }
    state.update(overrides)
    return state


def _llm_returning(*invoke_results):
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.side_effect = invoke_results
    llm.bind_tools.return_value = bound
    return llm


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_seeds_system_and_context_message_on_first_call(mock_chat_anthropic):
    final = AIMessage(content="All done.")
    mock_chat_anthropic.return_value = _llm_returning(final)

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(_base_state())

    messages = result["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    # Ids, not just prose, so the LLM can copy them verbatim into tool calls.
    assert "e1" in messages[1].content
    assert "e2" in messages[1].content
    assert messages[-1] is final


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_context_includes_the_email_body_not_just_subject(mock_chat_anthropic):
    # Live-discovered bug: the agent asked the human to "share the actual
    # email content" for a reschedule/meeting-request item -- it wasn't
    # being cautious, it genuinely never received the body, only
    # id/thread_id/from/subject. The proposed time lives in the body text
    # ("can we push our client sync to Thursday instead?"), so without it
    # the agent has no way to know what was actually requested.
    email = EmailSummary(
        id="m1",
        thread_id="t1",
        from_="priya.patel@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push our client sync from tomorrow to Thursday instead?",
        received_at=NOW,
    )
    reschedule = RescheduleRequest(
        description="asks to reschedule 'Client Sync'", email=email, event=EVENT_A
    )
    final = AIMessage(content="What time works for Thursday?")
    mock_chat_anthropic.return_value = _llm_returning(final)

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(_base_state(action_items=[reschedule]))

    context = result["messages"][1].content
    assert "Can we push our client sync from tomorrow to Thursday instead?" in context


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_does_not_reseed_on_loop_back(mock_chat_anthropic):
    final = AIMessage(content="Ok, updated.")
    mock_chat_anthropic.return_value = _llm_returning(final)

    agent = make_agent_node(MagicMock(name="gmail_service"))
    existing = [
        SystemMessage(content="..."),
        HumanMessage(content="..."),
        AIMessage(content="prior summary"),
        HumanMessage(content="move it to 2pm instead"),
    ]

    result = agent(_base_state(messages=existing))

    assert result["messages"] == [final]


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_tools_loop_terminates_and_calls_the_right_tool(mock_chat_anthropic):
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "propose_event",
                "args": {
                    "title": "Client Call",
                    "start": "2026-07-16T14:00:00+00:00",
                    "duration_minutes": 30,
                    "existing_event_id": "e2",
                },
                "id": "call_1",
            }
        ],
    )
    final_message = AIMessage(content="Proposed moving Client Call to 2pm.")
    mock_chat_anthropic.return_value = _llm_returning(tool_call_message, final_message)

    gmail_service = MagicMock(name="gmail_service")
    agent = make_agent_node(gmail_service)
    tools_node = make_tools_node(gmail_service)

    builder = StateGraph(PlannerState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    result = graph.invoke(_base_state())

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_messages[-1].content == "Proposed moving Client Call to 2pm."

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Client Call" in str(tool_messages[0].content)


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_failure_with_no_prior_proposals_reports_nothing_was_done(
    mock_chat_anthropic,
):
    mock_chat_anthropic.return_value = _llm_returning(
        Exception("Anthropic API is down")
    )

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(_base_state())

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls
    assert "error" in final.content.lower()
    assert "try again" in final.content.lower()


@patch("agentic_secretary.resolution.ChatAnthropic")
def test_agent_failure_with_prior_proposals_reports_what_was_actually_done(
    mock_chat_anthropic,
):
    mock_chat_anthropic.return_value = _llm_returning(
        Exception("Anthropic API is down")
    )
    proposal_a = EventProposal(
        title="Client Call",
        start=NOW + timedelta(hours=2),
        duration_minutes=30,
        existing_event_id="e2",
    )
    proposal_b = EventProposal(
        title="New Sync", start=NOW + timedelta(hours=3), duration_minutes=45
    )
    existing_messages = [
        SystemMessage(content="..."),
        HumanMessage(content="..."),
        ToolMessage(
            content=str(proposal_a), artifact=proposal_a, tool_call_id="call_1"
        ),
        ToolMessage(
            content=str(proposal_b), artifact=proposal_b, tool_call_id="call_2"
        ),
    ]

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(_base_state(messages=existing_messages))

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls
    assert "error" in final.content.lower()
    assert "try again" in final.content.lower()
    # Verified against _latest_proposals' own output, not a hand-duplicated
    # list -- guards against the message-building logic drifting from what
    # the helper it reuses actually reports.
    expected_proposals = _latest_proposals(existing_messages)
    assert len(expected_proposals) == 2
    for proposal in expected_proposals:
        assert proposal.title in final.content
