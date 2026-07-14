from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.types import Command

from agentic_secretary.graph import PlannerState, _ChatIntent, _EmailIntent, build_graph, greet
from agentic_secretary.tools import CalendarEvent, EmailSummary

FAKE_EMAILS = [
    EmailSummary(
        id="m1",
        thread_id="t1",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
]
FAKE_EVENTS = [
    CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc),
    )
]
# detect_actions now runs on every invoke; stub its LLM call so these
# fetch/check tests stay about fetch/check, not action detection.
NO_INTENT = _EmailIntent(proposes_new_meeting=False, requests_reschedule=False)


def _build_test_graph():
    gmail_service = MagicMock(name="gmail_service")
    calendar_service = MagicMock(name="calendar_service")
    graph = build_graph(gmail_service, calendar_service)
    return graph, gmail_service, calendar_service


def test_planner_state_has_expected_fields():
    assert set(PlannerState.__annotations__) == {
        "messages",
        "emails",
        "calendar_events",
        "action_items",
        "intent",
        "status",
    }


def test_greet_emits_an_ai_message():
    result = greet(
        {"messages": [], "emails": [], "calendar_events": [], "action_items": [], "status": "pending"}
    )

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content


@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_classify_intent_interrupts_with_the_latest_ai_message(
    mock_list_emails, mock_list_events, mock_analyze_email
):
    # interrupt()'s value is what the CLI actually displays as the prompt --
    # this proves greet's message reaches the human via classify_intent,
    # not just that it sits unused in state.
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    result = graph.invoke(
        {"emails": [], "calendar_events": [], "status": "pending"},
        config=config,
    )

    assert "__interrupt__" in result
    assert "scheduling assistant" in result["__interrupt__"][0].value


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_classify_intent_routes_to_fetch_emails_on_check_actions(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, gmail_service, calendar_service = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    graph.invoke({"emails": [], "calendar_events": [], "status": "pending"}, config=config)
    result = graph.invoke(Command(resume="check for conflicts"), config=config)

    assert "__interrupt__" not in result
    assert result["emails"] == FAKE_EMAILS
    assert result["calendar_events"] == FAKE_EVENTS
    assert result["status"] == "done"

    mock_list_emails.assert_called_once_with(gmail_service)
    mock_list_events.assert_called_once_with(calendar_service)


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_classify_intent_loops_on_unrecognized_reply(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.side_effect = [
        _ChatIntent(intent="others"),
        _ChatIntent(intent="check_actions"),
    ]
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    graph.invoke({"emails": [], "calendar_events": [], "status": "pending"}, config=config)
    second_result = graph.invoke(Command(resume="asdf"), config=config)

    # Looped back to classify_intent instead of proceeding -- still paused,
    # now showing the clarifying re-prompt rather than the original greeting.
    assert "__interrupt__" in second_result
    assert "didn't quite catch" in second_result["__interrupt__"][0].value

    third_result = graph.invoke(Command(resume="check for conflicts"), config=config)
    assert "__interrupt__" not in third_result
    assert third_result["emails"] == FAKE_EMAILS
