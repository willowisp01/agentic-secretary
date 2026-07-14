from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.types import Command

from agentic_secretary.graph import (
    ActionResolution,
    CalendarOverlapConflict,
    EmailConflict,
    PlannerState,
    _ChatIntent,
    _EmailIntent,
    build_graph,
    greet,
)
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


def _advance_past_classify_intent(graph, config):
    # Every present_menu test needs to get past greet + classify_intent
    # first; _classify_intent is mocked by the caller's @patch.
    graph.invoke({"emails": [], "calendar_events": [], "status": "pending"}, config=config)
    return graph.invoke(Command(resume="check for conflicts"), config=config)


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


def test_planner_state_has_expected_fields():
    assert set(PlannerState.__annotations__) == {
        "messages",
        "emails",
        "calendar_events",
        "action_items",
        "intent",
        "resolutions",
        "pending_action_index",
        "status",
    }


def test_action_resolution_holds_skip_remedy_with_no_proposal():
    standup = CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc),
    )
    client_sync = CalendarEvent(
        id="e2",
        title="Client Sync",
        start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc),
    )
    item = CalendarOverlapConflict(
        description="'Team Standup' overlaps with 'Client Sync'",
        events=[standup, client_sync],
    )

    resolution = ActionResolution(action_item=item, remedy="skip")

    assert resolution.remedy == "skip"
    assert resolution.proposal is None
    assert resolution.shift_event_id is None


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


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=[])
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_detect_actions_with_no_action_items_skips_present_menu(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    result = _advance_past_classify_intent(graph, config)

    assert "__interrupt__" not in result
    assert result["action_items"] == []
    assert any("No action items found" in m.content for m in result["messages"])


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_present_menu_shows_shift_and_skip_for_calendar_overlap(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    # calendar_overlap has no email at all -- draft_reply must not appear.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    result = _advance_past_classify_intent(graph, config)

    assert "__interrupt__" in result
    menu = result["__interrupt__"][0].value
    assert "Shift the slot" in menu
    assert "Skip" in menu
    assert "Draft a reply" not in menu


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_present_menu_asks_which_event_when_shifting_a_two_event_kind(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    which_event_result = graph.invoke(Command(resume="1"), config=config)  # "1. Shift the slot"

    assert "__interrupt__" in which_event_result
    assert "Which event should move?" in which_event_result["__interrupt__"][0].value

    final_result = graph.invoke(Command(resume="2"), config=config)  # 2nd event listed

    assert "__interrupt__" not in final_result
    assert len(final_result["resolutions"]) == 1
    resolution = final_result["resolutions"][0]
    assert resolution.remedy == "shift_slot"
    assert resolution.shift_event_id == "e2"
    assert final_result["pending_action_index"] == 1


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_present_menu_records_skip_and_reaches_end(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    result = graph.invoke(Command(resume="2"), config=config)  # "2. Skip"

    assert "__interrupt__" not in result
    assert len(result["resolutions"]) == 1
    assert result["resolutions"][0].remedy == "skip"
    assert result["resolutions"][0].proposal is None
    assert result["pending_action_index"] == 1


@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_present_menu_reprompts_the_same_item_on_invalid_choice(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    first_menu = _advance_past_classify_intent(graph, config)
    retry_result = graph.invoke(Command(resume="not a number"), config=config)

    # Same menu shown again -- pending_action_index never advanced, so the
    # routing after present_menu re-enters it for the same item.
    assert "__interrupt__" in retry_result
    assert retry_result["__interrupt__"][0].value == first_menu["__interrupt__"][0].value
    assert retry_result["resolutions"] == []

    final_result = graph.invoke(Command(resume="2"), config=config)  # "2. Skip"
    assert "__interrupt__" not in final_result
    assert len(final_result["resolutions"]) == 1
