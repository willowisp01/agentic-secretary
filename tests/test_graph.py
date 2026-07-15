from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agentic_secretary.graph import (
    ActionResolution,
    BackToBackConflict,
    CalendarOverlapConflict,
    EmailConflict,
    PlannerState,
    RescheduleRequest,
    _applicable_remedies,
    _ChatIntent,
    _EmailIntent,
    _explicit_time_overlap_warning,
    _present_item_text,
    _ProposedPlan,
    _related_resolution_note,
    _related_resolutions,
    _route_after_confirm_plan,
    _route_after_content_generation,
    _route_after_propose_plan,
    _run_content_generation,
    build_graph,
    greet,
    propose_plan,
)
from agentic_secretary.tools import CalendarEvent, DraftResult, EmailSummary, EventProposal

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
    # Every present_item test needs to get past greet + classify_intent
    # first; _classify_intent is mocked by the caller's @patch.
    graph.invoke({"emails": [], "calendar_events": [], "status": "pending"}, config=config)
    return graph.invoke(Command(resume="check for conflicts"), config=config)


def _advance_past_present_item(graph, config, reply):
    # Submits a free-text reply to present_item's interrupt; propose_plan
    # (mocked by the caller's @patch) runs next, landing on confirm_plan's
    # interrupt with the resulting plan summary.
    return graph.invoke(Command(resume=reply), config=config)


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
        "pending_clarification",
        "pending_clarification_rounds",
        "pending_remedies",
        "pending_shift_event_ids",
        "pending_explicit_time",
        "pending_plan_summary",
        "status",
    }


def test_applicable_remedies_for_calendar_overlap_excludes_draft_and_accept():
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
    item = CalendarOverlapConflict(description="overlap", events=[standup, client_sync])

    assert _applicable_remedies(item) == ["shift_slot", "skip"]


def test_applicable_remedies_for_back_to_back_excludes_draft_and_accept():
    lunch = CalendarEvent(
        id="e1",
        title="Lunch",
        start=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc),
    )
    review = CalendarEvent(
        id="e2",
        title="Design Review",
        start=datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 13, 45, tzinfo=timezone.utc),
    )
    item = BackToBackConflict(description="no buffer", events=[lunch, review])

    assert _applicable_remedies(item) == ["shift_slot", "skip"]


def test_applicable_remedies_for_reschedule_excludes_accept():
    event = CalendarEvent(
        id="e1",
        title="Client Sync",
        start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc),
    )
    email = EmailSummary(
        id="m1",
        thread_id="t1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = RescheduleRequest(description="reschedule", event=event, email=email)

    assert _applicable_remedies(item) == ["shift_slot", "draft_reply", "skip"]


def test_applicable_remedies_for_email_conflict_includes_accept_meeting():
    event = CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc),
    )
    email = EmailSummary(
        id="m1",
        thread_id="t1",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = EmailConflict(
        description="overlap",
        events=[event],
        email=email,
        proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )

    assert _applicable_remedies(item) == ["shift_slot", "draft_reply", "accept_meeting", "skip"]


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


def test_present_item_text_shows_description_and_suggested_remedies():
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

    text = _present_item_text(item)

    assert "'Team Standup' overlaps with 'Client Sync'" in text
    assert "Shift the slot" in text
    assert "Skip" in text
    # calendar_overlap has no email at all -- draft/accept must not appear.
    assert "Draft a reply" not in text
    assert "Accept the meeting" not in text


def test_present_item_text_offers_accept_meeting_for_email_conflict():
    event = CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc),
    )
    email = EmailSummary(
        id="m1",
        thread_id="t1",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = EmailConflict(
        description="'Quick sync tomorrow?' requests a time overlapping 'Team Standup'",
        events=[event],
        email=email,
        proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )

    text = _present_item_text(item)

    assert "Shift the slot" in text
    assert "Draft a reply" in text
    assert "Accept the meeting" in text
    assert "Skip" in text
    assert "you decide" in text


EMAIL_CONFLICT_TWO_EVENTS = EmailConflict(
    description="'Quick sync tomorrow?' requests a time overlapping 'Team Standup', 'Client Sync'",
    events=OVERLAPPING_EVENTS,
    email=EmailSummary(
        id="m1",
        thread_id="t1",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    ),
    proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
    proposed_duration_minutes=30,
)


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_filters_out_inapplicable_remedies(mock_propose_plan):
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot", "draft_reply", "accept_meeting"],
        summary="plan",
    )
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift it")],
    }

    result = propose_plan(state)

    # calendar_overlap only ever applies shift_slot/skip -- draft_reply and
    # accept_meeting must be stripped even though the LLM named them.
    assert result["pending_remedies"] == ["shift_slot"]


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_supports_multiple_remedies(mock_propose_plan):
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot", "accept_meeting"],
        summary="plan",
    )
    state = {
        "action_items": [EMAIL_CONFLICT_TWO_EVENTS],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift it and accept the meeting")],
    }

    result = propose_plan(state)

    assert result["pending_remedies"] == ["shift_slot", "accept_meeting"]


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_defaults_to_skip_when_nothing_applicable_remains(mock_propose_plan):
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    mock_propose_plan.return_value = _ProposedPlan(remedies=["draft_reply"], summary="plan")
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="draft something")],
    }

    result = propose_plan(state)

    assert result["pending_remedies"] == ["skip"]


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_defaults_shift_event_ids_to_all_events_for_email_conflict(mock_propose_plan):
    # EmailConflict can have more than one candidate event; "shift to make
    # room" with no disambiguation means moving all of them, not guessing one.
    mock_propose_plan.return_value = _ProposedPlan(remedies=["shift_slot"], summary="plan")
    state = {
        "action_items": [EMAIL_CONFLICT_TWO_EVENTS],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift it")],
    }

    result = propose_plan(state)

    assert set(result["pending_shift_event_ids"]) == {"e1", "e2"}


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_defaults_shift_event_ids_to_first_event_for_pairwise_kind(mock_propose_plan):
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    mock_propose_plan.return_value = _ProposedPlan(remedies=["shift_slot"], summary="plan")
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift it")],
    }

    result = propose_plan(state)

    assert result["pending_shift_event_ids"] == ["e1"]


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_uses_llm_disambiguated_shift_event_id(mock_propose_plan):
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot"], shift_event_ids=["e2"], summary="plan"
    )
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift Client Sync")],
    }

    result = propose_plan(state)

    assert result["pending_shift_event_ids"] == ["e2"]


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_carries_explicit_time_and_summary(mock_propose_plan):
    target = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot"],
        shift_event_ids=["e2"],
        explicit_time=target,
        summary="I'll shift Client Sync to 3pm.",
    )
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift it to 3pm")],
    }

    result = propose_plan(state)

    assert result["pending_explicit_time"] == target
    assert result["pending_plan_summary"] == "I'll shift Client Sync to 3pm."


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_routes_to_clarification_when_llm_flags_it(mock_propose_plan):
    # Live-discovered gap: a terse reply on a two-event item ("shift", with
    # no target named) used to silently default to the first event while
    # the LLM's own summary asked a clarifying question -- the human had no
    # way to actually answer it, since confirm_plan only accepts yes/no.
    mock_propose_plan.return_value = _ProposedPlan(
        needs_clarification=True,
        clarifying_question="Which event should move -- Team Standup or Client Sync?",
        remedies=["shift_slot"],
        summary="I'll shift one of the overlapping events.",
    )
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="shift")],
    }

    result = propose_plan(state)

    assert result["pending_clarification"] == (
        "Which event should move -- Team Standup or Client Sync?"
    )
    assert result["pending_remedies"] == []
    assert result["pending_shift_event_ids"] == []
    assert result["pending_explicit_time"] is None
    assert result["pending_plan_summary"] == ""
    assert result["pending_clarification_rounds"] == 1


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_circuit_breaker_forces_shift_slot_after_rounds_exhausted(mock_propose_plan):
    # Live-discovered gap: needs_clarification could fire indefinitely for
    # some replies (worst case: "you decide", which per _propose_plan's
    # prompt should never need clarification at all) -- confirmed live as
    # a genuine infinite loop with no way out. Once the retry budget is
    # spent, propose_plan must stop asking and force a decision regardless
    # of what the LLM says on the next call.
    mock_propose_plan.return_value = _ProposedPlan(
        needs_clarification=True,
        clarifying_question="Which event should move -- Team Standup or Client Sync?",
        summary="I'll shift one of the overlapping events.",
    )
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": "Which event should move -- Team Standup or Client Sync?",
        "pending_clarification_rounds": 1,  # already at _MAX_CLARIFICATION_ROUNDS
        "resolutions": [],
        "messages": [HumanMessage(content="you decide")],
    }

    result = propose_plan(state)

    # Forced a real plan instead of asking a third time -- never silently
    # falls back to skip, which is a different, more conservative action
    # than what was actually being asked about.
    assert result["pending_clarification"] is None
    assert result["pending_clarification_rounds"] == 0
    assert result["pending_remedies"] == ["shift_slot"]
    assert result["pending_shift_event_ids"] == ["e1"]
    assert result["pending_plan_summary"] != ""


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_passes_pending_clarification_as_prior_question(mock_propose_plan):
    # Live-discovered gap: propose_plan used to always call _propose_plan
    # with only the latest reply, no memory of a clarifying question it had
    # just asked -- a terse answer like "team standup" (answering "which
    # event should move?") read as an under-specified fresh statement and
    # could trigger needs_clarification again instead of resolving it.
    mock_propose_plan.return_value = _ProposedPlan(remedies=["shift_slot"], summary="plan")
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": "Which event should move -- Team Standup or Client Sync?",
        "pending_clarification_rounds": 0,
        "resolutions": [],
        "messages": [HumanMessage(content="team standup")],
    }

    propose_plan(state)

    mock_propose_plan.assert_called_once_with(
        item, "team standup", "Which event should move -- Team Standup or Client Sync?", None
    )


def test_route_after_propose_plan_goes_to_present_item_when_clarification_needed():
    assert _route_after_propose_plan({"pending_clarification": "which event?"}) == "present_item"


def test_route_after_propose_plan_goes_to_confirm_plan_when_plan_is_ready():
    assert _route_after_propose_plan({"pending_clarification": None}) == "confirm_plan"


def test_explicit_time_overlap_warning_returns_none_when_no_overlap():
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    # Well clear of both OVERLAPPING_EVENTS (9:00-9:30, 9:15-10:00).
    target = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)

    warning = _explicit_time_overlap_warning(item, ["e1"], target, OVERLAPPING_EVENTS, [])

    assert warning is None


def test_explicit_time_overlap_warning_detects_overlap_with_calendar_event():
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    # e1 (Team Standup, 30min) is being shifted here to 9:20, which would
    # land it inside e2's (Client Sync) 9:15-10:00 window.
    target = datetime(2026, 7, 10, 9, 20, tzinfo=timezone.utc)

    warning = _explicit_time_overlap_warning(item, ["e1"], target, OVERLAPPING_EVENTS, [])

    assert warning is not None
    assert "Client Sync" in warning


def test_explicit_time_overlap_warning_excludes_the_event_being_shifted_itself():
    # e1 is the event being moved -- comparing its proposed new time against
    # its own current calendar entry would always "overlap" and warn on
    # every ordinary shift, which isn't useful. Move e1 to exactly its
    # current start with e2 removed from the picture to isolate this.
    standup_only = [OVERLAPPING_EVENTS[0]]
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    target = OVERLAPPING_EVENTS[0].start

    warning = _explicit_time_overlap_warning(item, ["e1"], target, standup_only, [])

    assert warning is None


def test_explicit_time_overlap_warning_detects_overlap_with_already_proposed_shift():
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    already_proposed = ActionResolution(
        action_item=item,
        remedy="shift_slot",
        shift_event_id="e2",
        proposal=EventProposal(
            title="Client Sync",
            start=datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc),
            duration_minutes=45,
            existing_event_id="e2",
        ),
    )
    # e1 shifted to 2:15pm would collide with the already-proposed Client
    # Sync shift to 2:00-2:45pm this session, even though neither is on the
    # original calendar_events list anymore.
    target = datetime(2026, 7, 10, 14, 15, tzinfo=timezone.utc)

    warning = _explicit_time_overlap_warning(item, ["e1"], target, [], [already_proposed])

    assert warning is not None
    assert "Client Sync" in warning


_CLIENT_SYNC = OVERLAPPING_EVENTS[1]  # "e2", shared across the fixtures below


def test_related_resolutions_finds_shared_event_across_different_kinds():
    # Live-confirmed as a real bug, not hypothetical: detect_actions runs
    # independent detection passes with no cross-referencing, so the same
    # event (Client Sync) ended up the subject of a RescheduleRequest item
    # AND an EmailConflict item in the same session, resolved independently
    # to two different, contradictory times -- with nothing surfacing that
    # they were about the same event at all.
    reschedule_item = RescheduleRequest(
        description="reschedule",
        event=_CLIENT_SYNC,
        email=EmailSummary(
            id="m1",
            thread_id="t1",
            from_="priya@example.com",
            to="you@example.com",
            subject="Re: Client Sync -- need to move",
            body="Can we push this?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
    )
    already_resolved = ActionResolution(
        action_item=reschedule_item,
        remedy="shift_slot",
        shift_event_id="e2",
        proposal=EventProposal(
            title="Client Sync",
            start=datetime(2026, 7, 17, 9, 15, tzinfo=timezone.utc),
            duration_minutes=45,
            existing_event_id="e2",
        ),
    )
    email_conflict_item = EmailConflict(
        description="'Quick sync tomorrow?' requests a time overlapping 'Client Sync'",
        events=[_CLIENT_SYNC],
        email=EmailSummary(
            id="m2",
            thread_id="t2",
            from_="alex@example.com",
            to="you@example.com",
            subject="Quick sync tomorrow?",
            body="Are you free tomorrow?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
        proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )

    related = _related_resolutions(email_conflict_item, [already_resolved])

    assert related == [already_resolved]


def test_related_resolutions_empty_when_no_shared_event():
    item_a = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    unrelated_item = BackToBackConflict(
        description="no buffer",
        events=[
            CalendarEvent(
                id="e3",
                title="Lunch",
                start=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
                end=datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc),
            ),
            CalendarEvent(
                id="e4",
                title="Design Review",
                start=datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc),
                end=datetime(2026, 7, 10, 13, 45, tzinfo=timezone.utc),
            ),
        ],
    )
    resolution = ActionResolution(action_item=unrelated_item, remedy="skip")

    assert _related_resolutions(item_a, [resolution]) == []


def test_related_resolution_note_mentions_the_proposed_time():
    item = EmailConflict(
        description="overlap",
        events=[_CLIENT_SYNC],
        email=EmailSummary(
            id="m2",
            thread_id="t2",
            from_="alex@example.com",
            to="you@example.com",
            subject="Quick sync tomorrow?",
            body="Are you free tomorrow?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
        proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )
    prior_item = RescheduleRequest(
        description="reschedule",
        event=_CLIENT_SYNC,
        email=EmailSummary(
            id="m1",
            thread_id="t1",
            from_="priya@example.com",
            to="you@example.com",
            subject="Re: Client Sync -- need to move",
            body="Can we push this?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
    )
    already_resolved = ActionResolution(
        action_item=prior_item,
        remedy="shift_slot",
        shift_event_id="e2",
        proposal=EventProposal(
            title="Client Sync",
            start=datetime(2026, 7, 17, 9, 15, tzinfo=timezone.utc),
            duration_minutes=45,
            existing_event_id="e2",
        ),
    )

    note = _related_resolution_note(item, [already_resolved])

    assert note is not None
    assert "2026-07-17T09:15:00" in note
    assert "shift_slot" in note


def test_related_resolution_note_none_when_nothing_related():
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    assert _related_resolution_note(item, []) is None


@patch("agentic_secretary.graph._propose_plan")
def test_propose_plan_passes_related_resolution_note_to_llm(mock_propose_plan):
    mock_propose_plan.return_value = _ProposedPlan(remedies=["shift_slot"], summary="plan")
    item = EmailConflict(
        description="overlap",
        events=[_CLIENT_SYNC],
        email=EmailSummary(
            id="m2",
            thread_id="t2",
            from_="alex@example.com",
            to="you@example.com",
            subject="Quick sync tomorrow?",
            body="Are you free tomorrow?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
        proposed_start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )
    prior_item = RescheduleRequest(
        description="reschedule",
        event=_CLIENT_SYNC,
        email=EmailSummary(
            id="m1",
            thread_id="t1",
            from_="priya@example.com",
            to="you@example.com",
            subject="Re: Client Sync -- need to move",
            body="Can we push this?",
            received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        ),
    )
    already_resolved = ActionResolution(
        action_item=prior_item,
        remedy="shift_slot",
        shift_event_id="e2",
        proposal=EventProposal(
            title="Client Sync",
            start=datetime(2026, 7, 17, 9, 15, tzinfo=timezone.utc),
            duration_minutes=45,
            existing_event_id="e2",
        ),
    )
    state = {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_clarification": None,
        "pending_clarification_rounds": 0,
        "resolutions": [already_resolved],
        "messages": [HumanMessage(content="you decide")],
    }

    propose_plan(state)

    related_context = mock_propose_plan.call_args.args[3]
    assert related_context is not None
    assert "2026-07-17T09:15:00" in related_context


def test_route_after_confirm_plan_goes_to_content_generation_when_remedies_queued():
    assert _route_after_confirm_plan({"pending_remedies": ["shift_slot"]}) == "content_generation"


def test_route_after_confirm_plan_goes_to_present_item_when_remedies_empty():
    assert _route_after_confirm_plan({"pending_remedies": []}) == "present_item"


def _generation_state(
    item,
    pending_remedies,
    pending_shift_event_ids=None,
    resolutions=None,
    pending_explicit_time=None,
    pending_plan_summary="",
):
    return {
        "action_items": [item],
        "pending_action_index": 0,
        "pending_remedies": pending_remedies,
        "pending_shift_event_ids": pending_shift_event_ids or [],
        "pending_explicit_time": pending_explicit_time,
        "pending_plan_summary": pending_plan_summary,
        "calendar_events": OVERLAPPING_EVENTS,
        "resolutions": resolutions or [],
    }


def test_run_content_generation_shift_slot_uses_explicit_time_deterministically():
    # Live-discovered gap: pending_explicit_time (the human's confirmed
    # target time) used to never reach generation at all -- content_
    # generation would call the LLM-based _generate_shift_proposal, which
    # had no visibility into the confirmed time and could pick something
    # else entirely, silently overriding what the human just confirmed.
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    explicit_time = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)
    state = _generation_state(item, ["shift_slot"], ["e2"], pending_explicit_time=explicit_time)

    with patch("agentic_secretary.graph._generate_shift_proposal") as mock_shift_proposal:
        result = _run_content_generation(MagicMock(name="gmail_service"), state)

    mock_shift_proposal.assert_not_called()
    resolution = result["resolutions"][0]
    assert resolution.proposal.start == explicit_time
    # e2 (Client Sync) is 45 minutes in OVERLAPPING_EVENTS -- the shifted
    # proposal must keep that duration, not just the start time.
    assert resolution.proposal.duration_minutes == 45
    assert resolution.proposal.existing_event_id == "e2"


@patch("agentic_secretary.graph._generate_shift_proposal")
def test_run_content_generation_shift_slot_fills_in_a_proposal(mock_shift_proposal):
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    fake_proposal = EventProposal(
        title="Client Sync",
        start=datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc),
        duration_minutes=45,
        existing_event_id="e2",
    )
    mock_shift_proposal.return_value = fake_proposal
    state = _generation_state(item, ["shift_slot"], ["e2"])

    result = _run_content_generation(MagicMock(name="gmail_service"), state)

    assert len(result["resolutions"]) == 1
    resolution = result["resolutions"][0]
    assert resolution.remedy == "shift_slot"
    assert resolution.shift_event_id == "e2"
    assert resolution.proposal == fake_proposal
    assert result["pending_action_index"] == 1
    assert result["pending_remedies"] == []


def test_run_content_generation_skip_makes_no_llm_or_tool_call():
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = _generation_state(item, ["skip"])

    with (
        patch("agentic_secretary.graph._generate_shift_proposal") as mock_shift,
        patch("agentic_secretary.graph._generate_reply_body") as mock_reply,
        patch("agentic_secretary.graph.tools.propose_event") as mock_propose_event,
        patch("agentic_secretary.graph.tools.draft_reply") as mock_draft_reply,
    ):
        result = _run_content_generation(MagicMock(name="gmail_service"), state)

    assert result["resolutions"][0].remedy == "skip"
    assert result["resolutions"][0].proposal is None
    mock_shift.assert_not_called()
    mock_reply.assert_not_called()
    mock_propose_event.assert_not_called()
    mock_draft_reply.assert_not_called()


@patch("agentic_secretary.graph._generate_reply_body")
def test_run_content_generation_draft_reply_passes_the_plan_summary(mock_reply_body):
    # Live-discovered gap: _generate_reply_body used to only see the
    # original email, never what the human actually asked for this turn
    # (e.g. "ask if they can push it to Friday") -- drafted replies agreed
    # to the sender's original ask instead of reflecting the confirmed plan.
    mock_reply_body.return_value = "Sure, Thursday works!"
    email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this to Thursday, same time?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = RescheduleRequest(description="reschedule", event=OVERLAPPING_EVENTS[1], email=email)
    state = _generation_state(
        item, ["draft_reply"], pending_plan_summary="I'll ask if they can push it to Friday."
    )
    gmail_service = MagicMock(name="gmail_service")
    gmail_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "d1",
        "message": {"threadId": "thread-1"},
    }

    _run_content_generation(gmail_service, state)

    mock_reply_body.assert_called_once_with(item, "I'll ask if they can push it to Friday.")


@patch("agentic_secretary.graph._generate_shift_proposal")
def test_run_content_generation_shift_slot_passes_the_plan_summary(mock_shift_proposal):
    # Live-discovered gap: _generate_shift_proposal never saw what the
    # human actually asked for this turn -- a directional/vague request
    # with no resolvable clock time (e.g. "make it earlier") left
    # pending_explicit_time unset, so this LLM call was the only thing that
    # could have honored it, and it had no parameter to receive it at all.
    # Live testing showed "make it earlier" come back an hour *later*.
    mock_shift_proposal.return_value = EventProposal(
        title="Team Standup",
        start=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
        duration_minutes=30,
        existing_event_id="e1",
    )
    item = CalendarOverlapConflict(description="overlap", events=OVERLAPPING_EVENTS)
    state = _generation_state(
        item,
        ["shift_slot"],
        ["e1"],
        pending_plan_summary="I'll shift Team Standup to an earlier time.",
    )

    _run_content_generation(MagicMock(name="gmail_service"), state)

    mock_shift_proposal.assert_called_once_with(
        OVERLAPPING_EVENTS[0],
        item,
        state["calendar_events"],
        state["resolutions"],
        "I'll shift Team Standup to an earlier time.",
    )


@patch("agentic_secretary.graph._generate_reply_body")
def test_run_content_generation_draft_reply_creates_gmail_draft(mock_reply_body):
    mock_reply_body.return_value = "Sure, Thursday works!"
    email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this to Thursday, same time?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = RescheduleRequest(description="reschedule", event=OVERLAPPING_EVENTS[1], email=email)
    state = _generation_state(item, ["draft_reply"])
    gmail_service = MagicMock(name="gmail_service")
    gmail_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "d1",
        "message": {"threadId": "thread-1"},
    }

    result = _run_content_generation(gmail_service, state)

    resolution = result["resolutions"][0]
    assert resolution.remedy == "draft_reply"
    assert resolution.proposal == DraftResult(draft_id="d1", thread_id="thread-1")
    create_kwargs = gmail_service.users.return_value.drafts.return_value.create.call_args.kwargs
    assert create_kwargs["body"]["message"]["threadId"] == "thread-1"


def test_run_content_generation_accept_meeting_proposes_the_requested_meeting_without_llm_call():
    item = EMAIL_CONFLICT_TWO_EVENTS
    state = _generation_state(item, ["accept_meeting"])

    with (
        patch("agentic_secretary.graph._generate_shift_proposal") as mock_shift,
        patch("agentic_secretary.graph._generate_reply_body") as mock_reply,
    ):
        result = _run_content_generation(MagicMock(name="gmail_service"), state)

    resolution = result["resolutions"][0]
    assert resolution.remedy == "accept_meeting"
    assert resolution.proposal == EventProposal(
        title=item.email.subject,
        start=item.proposed_start,
        duration_minutes=item.proposed_duration_minutes,
        existing_event_id=None,
    )
    mock_shift.assert_not_called()
    mock_reply.assert_not_called()


@patch("agentic_secretary.graph._generate_reply_body")
@patch("agentic_secretary.graph._generate_shift_proposal")
def test_run_content_generation_multi_remedy_produces_multiple_resolutions_across_calls(
    mock_shift_proposal, mock_reply_body
):
    email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    item = RescheduleRequest(description="reschedule", event=OVERLAPPING_EVENTS[1], email=email)
    mock_shift_proposal.return_value = EventProposal(
        title="Client Sync",
        start=datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc),
        duration_minutes=45,
        existing_event_id="e2",
    )
    mock_reply_body.return_value = "Sure!"
    gmail_service = MagicMock(name="gmail_service")
    gmail_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "d1",
        "message": {"threadId": "thread-1"},
    }
    state = _generation_state(item, ["shift_slot", "draft_reply"], ["e2"])

    first = _run_content_generation(gmail_service, state)

    assert len(first["resolutions"]) == 1
    assert first["resolutions"][0].remedy == "shift_slot"
    assert first["pending_remedies"] == ["draft_reply"]
    assert "pending_action_index" not in first  # not advanced yet -- draft_reply still queued

    state.update(
        pending_remedies=first["pending_remedies"],
        pending_shift_event_ids=first["pending_shift_event_ids"],
        resolutions=first["resolutions"],
    )
    second = _run_content_generation(gmail_service, state)

    assert len(second["resolutions"]) == 1
    assert second["resolutions"][0].remedy == "draft_reply"
    assert second["pending_action_index"] == 1
    assert second["pending_remedies"] == []


@patch("agentic_secretary.graph._generate_shift_proposal")
def test_run_content_generation_shift_slot_multiple_events_produces_one_resolution_per_event(
    mock_shift_proposal
):
    mock_shift_proposal.return_value = EventProposal(
        title="placeholder",
        start=datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc),
        duration_minutes=30,
    )
    state = _generation_state(EMAIL_CONFLICT_TWO_EVENTS, ["shift_slot"], ["e1", "e2"])

    first = _run_content_generation(MagicMock(name="gmail_service"), state)

    assert first["resolutions"][0].shift_event_id == "e1"
    assert first["pending_shift_event_ids"] == ["e2"]
    assert "pending_remedies" not in first  # unchanged -- still mid shift_slot

    state.update(pending_shift_event_ids=first["pending_shift_event_ids"], resolutions=first["resolutions"])
    second = _run_content_generation(MagicMock(name="gmail_service"), state)

    assert second["resolutions"][0].shift_event_id == "e2"
    assert second["pending_action_index"] == 1
    assert second["pending_remedies"] == []


def test_route_after_content_generation_stays_on_self_when_remedies_remain():
    assert (
        _route_after_content_generation({"pending_remedies": ["draft_reply"]}) == "content_generation"
    )


def test_route_after_content_generation_goes_to_present_item_when_more_items_remain():
    state = {"pending_remedies": [], "pending_action_index": 1, "action_items": [1, 2]}
    assert _route_after_content_generation(state) == "present_item"


def test_route_after_content_generation_goes_to_end_when_exhausted():
    state = {"pending_remedies": [], "pending_action_index": 1, "action_items": [1]}
    assert _route_after_content_generation(state) == "end"


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
def test_detect_actions_with_no_action_items_skips_present_item(
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
def test_present_item_interrupt_shows_shift_and_skip_for_calendar_overlap(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent
):
    # calendar_overlap has no email at all -- draft_reply/accept_meeting
    # must not appear.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    result = _advance_past_classify_intent(graph, config)

    assert "__interrupt__" in result
    text = result["__interrupt__"][0].value
    assert "Shift the slot" in text
    assert "Skip" in text
    assert "Draft a reply" not in text
    assert "Accept the meeting" not in text


@patch("agentic_secretary.graph._generate_shift_proposal")
@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_shift_slot_produces_a_resolution_and_reaches_end(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
    mock_shift_proposal,
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot"],
        shift_event_ids=["e2"],
        summary="I'll shift Client Sync.",
    )
    fake_proposal = EventProposal(
        title="Client Sync",
        start=datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc),
        duration_minutes=45,
        existing_event_id="e2",
    )
    mock_shift_proposal.return_value = fake_proposal
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    confirm_result = _advance_past_present_item(graph, config, "shift Client Sync")

    assert "__interrupt__" in confirm_result
    assert "I'll shift Client Sync." in confirm_result["__interrupt__"][0].value

    result = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in result
    assert result["status"] == "done"
    assert len(result["resolutions"]) == 1
    resolution = result["resolutions"][0]
    assert resolution.remedy == "shift_slot"
    assert resolution.shift_event_id == "e2"
    assert resolution.proposal == fake_proposal

    # Shifted the chosen event (e2, "Client Sync"), not just the first one.
    shifted_event = mock_shift_proposal.call_args.args[0]
    assert shifted_event.id == "e2"


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email")
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails")
def test_full_flow_confirm_plan_shows_related_resolution_from_a_different_item(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
):
    # Live-confirmed as a real bug: detect_actions runs independent
    # detection passes with no cross-referencing, so Client Sync ended up
    # the subject of a calendar_overlap item AND a reschedule item in the
    # same session, resolved independently to two different, contradictory
    # times, with nothing surfacing that they were the same event.
    reschedule_email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    mock_list_emails.return_value = [reschedule_email]
    mock_analyze_email.return_value = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=True,
        references_event_id="e2",
    )
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.side_effect = [
        _ProposedPlan(
            remedies=["shift_slot"],
            shift_event_ids=["e2"],
            explicit_time=datetime(2026, 7, 17, 9, 15, tzinfo=timezone.utc),
            summary="I'll shift Client Sync to Friday.",
        ),
        _ProposedPlan(remedies=["skip"], summary="I'll skip this."),
    ]
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    _advance_past_present_item(graph, config, "shift Client Sync to Friday")
    graph.invoke(Command(resume="yes"), config=config)

    # Second item (reschedule) also involves Client Sync -- its confirm_plan
    # turn should surface the first item's already-resolved decision,
    # regardless of what the second propose_plan call's own summary says.
    second_confirm = _advance_past_present_item(graph, config, "skip it")

    assert "__interrupt__" in second_confirm
    displayed = second_confirm["__interrupt__"][0].value
    assert "already resolved earlier this session" in displayed
    assert "2026-07-17T09:15:00" in displayed


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_reject_at_confirm_plan_reprompts_the_same_item(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent, mock_propose_plan
):
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(remedies=["skip"], summary="I'll skip this.")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    first_item = _advance_past_classify_intent(graph, config)
    _advance_past_present_item(graph, config, "not sure yet")
    rejected = graph.invoke(Command(resume="no"), config=config)

    # Same item re-shown -- pending_action_index never advanced, and
    # confirm_plan's rejection branch cleared pending_remedies, so routing
    # sends the human back to present_item for the same item.
    assert "__interrupt__" in rejected
    assert rejected["__interrupt__"][0].value == first_item["__interrupt__"][0].value
    assert rejected["resolutions"] == []

    _advance_past_present_item(graph, config, "actually, skip it")
    final_result = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in final_result
    assert len(final_result["resolutions"]) == 1
    assert final_result["resolutions"][0].remedy == "skip"


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_needs_clarification_reprompts_with_the_question(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent, mock_propose_plan
):
    # Live-discovered gap: propose_plan used to always forward to
    # confirm_plan's yes/no gate even when its own summary was a hedging
    # clarifying question ("Please clarify which event...") -- the human
    # had no way to actually answer it there, since "yes"/"no" can't
    # answer an open question. A needs_clarification plan should skip
    # confirm_plan and re-show present_item with the question instead.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.side_effect = [
        _ProposedPlan(
            needs_clarification=True,
            clarifying_question="Which event should move -- Team Standup or Client Sync?",
            remedies=["shift_slot"],
            summary="I'll shift one of the overlapping events.",
        ),
        _ProposedPlan(remedies=["skip"], summary="I'll skip this."),
    ]
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    reprompt = _advance_past_present_item(graph, config, "shift")

    # Routed straight back to present_item with the clarifying question --
    # not to confirm_plan's yes/no gate.
    assert "__interrupt__" in reprompt
    assert "Which event should move" in reprompt["__interrupt__"][0].value
    assert "Shift the slot" in reprompt["__interrupt__"][0].value  # item still shown too

    confirm_result = _advance_past_present_item(graph, config, "never mind, skip it")
    assert "__interrupt__" in confirm_result
    assert "Confirm this plan?" in confirm_result["__interrupt__"][0].value

    final_result = graph.invoke(Command(resume="yes"), config=config)
    assert "__interrupt__" not in final_result
    assert final_result["resolutions"][0].remedy == "skip"


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_clarification_answer_carries_the_prior_question(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent, mock_propose_plan
):
    # Live-discovered gap: after re-prompting with a clarifying question,
    # present_item used to clear pending_clarification before propose_plan
    # ever saw it -- the follow-up reply ("team standup") reached
    # _propose_plan with no memory of what question it was answering, so
    # a terse answer could trigger needs_clarification all over again
    # instead of resolving the original ambiguity.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.side_effect = [
        _ProposedPlan(
            needs_clarification=True,
            clarifying_question="Which event should move -- Team Standup or Client Sync?",
            summary="I'll shift one of the overlapping events.",
        ),
        _ProposedPlan(remedies=["shift_slot"], shift_event_ids=["e1"], summary="I'll shift Team Standup."),
    ]
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    _advance_past_present_item(graph, config, "shift")
    _advance_past_present_item(graph, config, "team standup")

    second_call = mock_propose_plan.call_args_list[1]
    assert second_call.args[1] == "team standup"
    assert second_call.args[2] == "Which event should move -- Team Standup or Client Sync?"


@patch("agentic_secretary.graph._generate_shift_proposal")
@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_repeated_needs_clarification_terminates_instead_of_looping(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
    mock_shift_proposal,
):
    # Live-confirmed as a genuine infinite loop: replying "you decide" to a
    # calendar_overlap/back_to_back item kept re-asking "which event should
    # move?" no matter what was said next, including plain "yes" -- the
    # session never reached a confirm_plan gate at all. propose_plan's own
    # circuit breaker must force a decision after one retry, regardless of
    # how many times the LLM itself claims it still needs clarification.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(
        needs_clarification=True,
        clarifying_question="Which event should move -- Team Standup or Client Sync?",
        summary="I'll shift one of the overlapping events.",
    )
    mock_shift_proposal.return_value = EventProposal(
        title="Team Standup",
        start=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
        duration_minutes=30,
        existing_event_id="e1",
    )
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    first_reprompt = _advance_past_present_item(graph, config, "you decide")
    assert "__interrupt__" in first_reprompt
    assert "Which event should move" in first_reprompt["__interrupt__"][0].value

    # Second round: the LLM (mocked) still insists on needs_clarification,
    # exactly like the live "you decide" case -- but the circuit breaker
    # must have exhausted its one retry by now and force a plan anyway,
    # landing on confirm_plan's gate instead of asking a third time.
    second_result = _advance_past_present_item(graph, config, "you decide")
    assert "__interrupt__" in second_result
    assert "Confirm this plan?" in second_result["__interrupt__"][0].value

    final_result = graph.invoke(Command(resume="yes"), config=config)
    assert "__interrupt__" not in final_result
    assert final_result["resolutions"][0].remedy == "shift_slot"


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_full_flow_skip_makes_no_llm_or_tool_call(
    mock_list_emails, mock_list_events, mock_analyze_email, mock_classify_intent, mock_propose_plan
):
    # Reproduces what would otherwise be a wasted LLM call: skip needs
    # neither a generated proposal nor a tool call, just a recorded
    # resolution -- content_generation must short-circuit before the LLM.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(remedies=["skip"], summary="I'll skip this.")
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    with (
        patch("agentic_secretary.graph._generate_shift_proposal") as mock_shift,
        patch("agentic_secretary.graph._generate_reply_body") as mock_reply,
    ):
        _advance_past_classify_intent(graph, config)
        _advance_past_present_item(graph, config, "skip it")
        result = graph.invoke(Command(resume="yes"), config=config)

    assert result["resolutions"][0].remedy == "skip"
    assert result["resolutions"][0].proposal is None
    mock_shift.assert_not_called()
    mock_reply.assert_not_called()


@patch("agentic_secretary.graph._generate_reply_body")
@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email")
@patch("agentic_secretary.graph.tools.list_upcoming_events")
@patch("agentic_secretary.graph.tools.list_recent_emails")
def test_full_flow_creates_a_gmail_draft_for_draft_reply(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
    mock_reply_body,
):
    reschedule_email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="priya@example.com",
        to="you@example.com",
        subject="Re: Client Sync -- need to move",
        body="Can we push this to Thursday, same time?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    client_sync = CalendarEvent(
        id="e2",
        title="Client Sync",
        start=datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc),
    )
    mock_list_emails.return_value = [reschedule_email]
    mock_list_events.return_value = [client_sync]
    mock_analyze_email.return_value = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=True,
        references_event_id="e2",
    )
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["draft_reply"], summary="I'll draft a reply to Priya."
    )
    mock_reply_body.return_value = "Sure, Thursday works!"

    gmail_service = MagicMock(name="gmail_service")
    gmail_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "d1",
        "message": {"threadId": "thread-1"},
    }
    calendar_service = MagicMock(name="calendar_service")
    graph = build_graph(gmail_service, calendar_service)
    config = {"configurable": {"thread_id": "test"}}

    _advance_past_classify_intent(graph, config)
    _advance_past_present_item(graph, config, "draft a reply")
    result = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in result
    resolution = result["resolutions"][0]
    assert resolution.remedy == "draft_reply"
    assert resolution.proposal == DraftResult(draft_id="d1", thread_id="thread-1")

    create_kwargs = gmail_service.users.return_value.drafts.return_value.create.call_args.kwargs
    assert create_kwargs["body"]["message"]["threadId"] == "thread-1"


@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email")
@patch("agentic_secretary.graph.tools.list_upcoming_events")
@patch("agentic_secretary.graph.tools.list_recent_emails")
def test_full_flow_accept_meeting_proposes_new_event_without_llm_call(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
):
    meeting_request_email = EmailSummary(
        id="m1",
        thread_id="thread-1",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow at 9:15am?",
        received_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    standup = CalendarEvent(
        id="e1",
        title="Team Standup",
        start=datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 11, 9, 30, tzinfo=timezone.utc),
    )
    mock_list_emails.return_value = [meeting_request_email]
    mock_list_events.return_value = [standup]
    mock_analyze_email.return_value = _EmailIntent(
        proposes_new_meeting=True,
        requests_reschedule=False,
        proposed_start=datetime(2026, 7, 11, 9, 15, tzinfo=timezone.utc),
        proposed_duration_minutes=30,
    )
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["accept_meeting"], summary="I'll accept Alex's meeting."
    )
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    with patch("agentic_secretary.graph.tools.propose_event") as mock_propose_event:
        mock_propose_event.return_value = EventProposal(
            title="Quick sync tomorrow?",
            start=datetime(2026, 7, 11, 9, 15, tzinfo=timezone.utc),
            duration_minutes=30,
            existing_event_id=None,
        )
        _advance_past_classify_intent(graph, config)
        _advance_past_present_item(graph, config, "accept the meeting")
        result = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in result
    resolution = result["resolutions"][0]
    assert resolution.remedy == "accept_meeting"
    assert resolution.proposal.existing_event_id is None
    mock_propose_event.assert_called_once_with(
        title="Quick sync tomorrow?",
        start=datetime(2026, 7, 11, 9, 15, tzinfo=timezone.utc),
        duration_minutes=30,
        existing_event_id=None,
    )


@patch("agentic_secretary.graph._generate_shift_proposal")
@patch("agentic_secretary.graph._propose_plan")
@patch("agentic_secretary.graph._classify_intent")
@patch("agentic_secretary.graph._analyze_email", return_value=NO_INTENT)
@patch("agentic_secretary.graph.tools.list_upcoming_events", return_value=OVERLAPPING_EVENTS)
@patch("agentic_secretary.graph.tools.list_recent_emails", return_value=[])
def test_checkpointing_does_not_warn_about_unregistered_types(
    mock_list_emails,
    mock_list_events,
    mock_analyze_email,
    mock_classify_intent,
    mock_propose_plan,
    mock_shift_proposal,
    caplog,
):
    # Reproduces a live-discovered gap: LangGraph's default checkpoint
    # serializer warns (and, in a future version, will refuse) to
    # deserialize any custom type it doesn't recognize, and every
    # interrupt/resume round-trip goes through the checkpointer --
    # CalendarEvent, CalendarOverlapConflict, and ActionResolution
    # (EventProposal nested inside it) all flow through PlannerState.
    mock_classify_intent.return_value = _ChatIntent(intent="check_actions")
    mock_propose_plan.return_value = _ProposedPlan(
        remedies=["shift_slot"], shift_event_ids=["e2"], summary="I'll shift Client Sync."
    )
    mock_shift_proposal.return_value = EventProposal(
        title="Client Sync",
        start=datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc),
        duration_minutes=45,
        existing_event_id="e2",
    )
    graph, _, _ = _build_test_graph()
    config = {"configurable": {"thread_id": "test"}}

    with caplog.at_level("WARNING"):
        _advance_past_classify_intent(graph, config)
        _advance_past_present_item(graph, config, "shift Client Sync")
        graph.invoke(Command(resume="yes"), config=config)

    assert "unregistered type" not in caplog.text
