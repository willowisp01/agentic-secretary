from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agentic_secretary.graph import PlannerState, _EmailIntent, build_graph
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
        "emails",
        "calendar_events",
        "action_items",
        "status",
    }


@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_fetch_emails_runs_before_check_calendar(
    mock_list_emails, mock_list_events, mock_analyze_email
):
    graph, gmail_service, calendar_service = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    states = list(
        graph.stream(
            {"emails": [], "calendar_events": [], "status": "pending"},
            config=config,
            stream_mode="values",
        )
    )

    # states[0] is the input snapshot; states[1] after fetch_emails; states[2] after check_calendar.
    assert states[1]["emails"] == FAKE_EMAILS
    assert states[1]["calendar_events"] == []
    assert states[2]["emails"] == FAKE_EMAILS
    assert states[2]["calendar_events"] == FAKE_EVENTS

    mock_list_emails.assert_called_once_with(gmail_service)
    mock_list_events.assert_called_once_with(calendar_service)


@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_graph_invoke_returns_final_state_with_status_done(
    mock_list_emails, mock_list_events, mock_analyze_email
):
    graph, _, _ = _build_test_graph()

    result = graph.invoke(
        {"emails": [], "calendar_events": [], "status": "pending"},
        config={"configurable": {"thread_id": "test"}},
    )

    assert result["emails"] == FAKE_EMAILS
    assert result["calendar_events"] == FAKE_EVENTS
    assert result["status"] == "done"
