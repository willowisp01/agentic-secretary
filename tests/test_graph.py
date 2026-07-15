from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.types import Command

from agentic_secretary.chat import _Intent
from agentic_secretary.detection import _EmailIntent
from agentic_secretary.graph import build_graph
from agentic_secretary.state import PlannerState
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
# fetch/check tests stay about fetch/check, not conflict detection. This
# fixture data has no overlap and no meeting request, so detect_actions
# finds nothing -- the graph loops back to another greet interrupt rather
# than reaching the agent, which these tests don't otherwise exercise.
NO_INTENT = _EmailIntent(proposes_new_meeting=False, requests_reschedule=False)

_INITIAL_STATE: PlannerState = {
    "messages": [],
    "emails": [],
    "calendar_events": [],
    "action_items": [],
    "status": "pending",
}


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
        "status",
    }


@patch("agentic_secretary.chat.ChatAnthropic")
@patch("agentic_secretary.detection._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_fetch_emails_runs_before_check_calendar(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_chat_anthropic
):
    mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = _Intent(
        wants_conflict_check=True
    )
    graph, gmail_service, calendar_service = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}
    graph.invoke(_INITIAL_STATE, config=config)  # halts at greet's opening interrupt

    states = list(
        graph.stream(
            Command(resume="check for conflicts"),
            config=config,
            stream_mode="values",
        )
    )

    emails_states = [s for s in states if s.get("emails")]
    events_states = [s for s in states if s.get("calendar_events")]
    assert emails_states, "expected a state where emails got populated"
    assert events_states, "expected a state where calendar_events got populated"
    assert states.index(emails_states[0]) <= states.index(events_states[0])

    mock_list_emails.assert_called_once_with(gmail_service)
    mock_list_events.assert_called_once_with(calendar_service)


@patch("agentic_secretary.chat.ChatAnthropic")
@patch("agentic_secretary.detection._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=FAKE_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=FAKE_EMAILS)
def test_graph_invoke_returns_final_state_with_status_done(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_chat_anthropic
):
    mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = _Intent(
        wants_conflict_check=True
    )
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}
    graph.invoke(_INITIAL_STATE, config=config)

    result = graph.invoke(Command(resume="check for conflicts"), config=config)

    assert result["emails"] == FAKE_EMAILS
    assert result["calendar_events"] == FAKE_EVENTS
    assert result["status"] == "done"
    # No action items in this fixture -- detect_actions routes back to
    # greet for another turn rather than into the agent, so this second
    # invoke ends at another interrupt, not a graph-complete state.
    assert "__interrupt__" in result


OVERLAPPING_EVENTS = [
    CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc),
    ),
    CalendarEvent(
        id="e2",
        title="Client Sync",
        start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc),
    ),
]


def _llm_returning(*invoke_results):
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.side_effect = invoke_results
    llm.bind_tools.return_value = bound
    return llm


@patch("agentic_secretary.resolution.ChatAnthropic")
@patch("agentic_secretary.chat.ChatAnthropic")
@patch("agentic_secretary.detection._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_checkpointing_a_full_resolution_round_trip_does_not_warn_about_unregistered_types(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_chat_anthropic,
    mock_resolution_chat_anthropic,
    caplog,
):
    # Live-discovered gap: LangGraph's default checkpoint serializer warns
    # (and, in a future version, will refuse) to deserialize any custom
    # type it doesn't recognize, and every interrupt/resume round-trip
    # goes through the checkpointer. CalendarOverlapConflict (in
    # action_items) and EventProposal (as a ToolMessage artifact) both
    # flow through PlannerState across this round trip.
    mock_chat_anthropic.return_value.with_structured_output.return_value.invoke.return_value = (
        _Intent(wants_conflict_check=True)
    )
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "propose_event",
                "args": {
                    "title": "Client Sync",
                    "start": "2026-07-10T14:00:00+00:00",
                    "duration_minutes": 45,
                    "existing_event_id": "e2",
                },
                "id": "call_1",
            }
        ],
    )
    final_message = AIMessage(content="Proposed moving Client Sync to 2pm.")
    mock_resolution_chat_anthropic.return_value = _llm_returning(
        tool_call_message, final_message
    )

    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    with caplog.at_level("WARNING"):
        graph.invoke(_INITIAL_STATE, config=config)  # halts at greet's opening interrupt
        graph.invoke(Command(resume="check for conflicts"), config=config)
        # The warning fires on *deserializing* a checkpoint, not writing
        # one -- calendar_events/action_items/the EventProposal artifact
        # were written by the resume above but never read back until this
        # next resume, which is what actually exercises the gap.
        result = graph.invoke(Command(resume="done"), config=config)

    assert "__interrupt__" not in result
    assert "unregistered type" not in caplog.text
