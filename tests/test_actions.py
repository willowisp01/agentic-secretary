from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentic_secretary import seed_data
from agentic_secretary.config import DEMO_TIMEZONE
from agentic_secretary.graph import (
    CalendarOverlapConflict,
    EmailConflict,
    RescheduleRequest,
    _EmailIntent,
    detect_actions,
)
from agentic_secretary.tools import CalendarEvent, EmailSummary
from seed_demo_data import resolve_relative_time

SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

_FIXTURE_EMAILS = {e.id: e for e in seed_data.load_emails(SEED_DATA_DIR / "emails.yaml")}
_FIXTURE_EVENTS = {
    e.id: e for e in seed_data.load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
}


def _email(id_: str) -> EmailSummary:
    fixture = _FIXTURE_EMAILS[id_]
    return EmailSummary(
        id=fixture.id,
        thread_id=f"thread_{fixture.id}",
        from_=fixture.from_,
        to=fixture.to,
        subject=fixture.subject,
        body=fixture.body,
        received_at=resolve_relative_time(fixture.sent_relative, NOW),
    )


def _event(id_: str) -> CalendarEvent:
    fixture = _FIXTURE_EVENTS[id_]
    start = resolve_relative_time(fixture.start_relative, NOW)
    return CalendarEvent(
        id=fixture.id,
        title=fixture.title,
        start=start,
        end=start + timedelta(minutes=fixture.duration_minutes),
    )


STANDUP = _event("evt_standup")
CLIENT_CALL = _event("evt_client_call")
LUNCH = _event("evt_lunch")
REVIEW = _event("evt_review")

MEETING_REQUEST_EMAIL = _email("email_meeting_request")
RESCHEDULE_EMAIL = _email("email_reschedule")
MENTION_EMAIL = _email("email_casual_mention")
DIGEST_EMAIL = _email("email_internal_digest")


def test_detect_actions_finds_calendar_overlap():
    state = {"emails": [], "calendar_events": [STANDUP, CLIENT_CALL], "action_items": [], "status": "done"}

    result = detect_actions(state)

    kinds = {item.kind for item in result["action_items"]}
    assert "calendar_overlap" in kinds


def test_detect_actions_finds_back_to_back():
    state = {"emails": [], "calendar_events": [LUNCH, REVIEW], "action_items": [], "status": "done"}

    result = detect_actions(state)

    kinds = {item.kind for item in result["action_items"]}
    assert "back_to_back" in kinds


def test_detect_actions_no_false_positive_for_well_spaced_events():
    state = {"emails": [], "calendar_events": [STANDUP, LUNCH], "action_items": [], "status": "done"}

    result = detect_actions(state)

    assert result["action_items"] == []


def test_calendar_overlap_conflict_requires_exactly_two_events():
    # The arity invariant used to be enforced only by convention (every
    # caller happened to pass exactly 2 events) rather than by the type
    # itself. This proves it's now a real, enforced constraint.
    with pytest.raises(ValidationError):
        CalendarOverlapConflict(description="bad", events=[STANDUP])


def test_email_conflict_requires_at_least_one_event():
    with pytest.raises(ValidationError):
        EmailConflict(
            description="bad",
            events=[],
            email=MEETING_REQUEST_EMAIL,
            proposed_start=STANDUP.start,
            proposed_duration_minutes=30,
        )


def test_email_intent_clears_reference_id_when_not_rescheduling():
    # Reproduces a live finding: the LLM can attach a real, valid event id to
    # an email that never mentions any event at all (observed for a digest
    # email), because a "required, always-present" field pressures it to
    # fill in something. Normalize that at the schema boundary rather than
    # trusting every future caller to re-derive the same gating logic.
    intent = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=False,
        references_event_id="evt_standup",
    )

    assert intent.references_event_id is None


def test_email_intent_clears_proposed_time_when_no_new_meeting():
    intent = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=False,
        proposed_start="2026-07-14T09:00:00+00:00",
        proposed_duration_minutes=30,
    )

    assert intent.proposed_start is None
    assert intent.proposed_duration_minutes is None


def test_email_intent_normalizes_naive_proposed_start_to_demo_timezone():
    # Reproduces a live-discovered gap: nothing in the schema/prompt forces
    # the LLM to include an offset, and a naive datetime compared against a
    # timezone-aware CalendarEvent time raises TypeError. The prompt tells
    # the LLM to assume DEMO_TIMEZONE for bare times, so a naive response is
    # normalized the same way -- defaulting to UTC instead would silently
    # shift a real "9am" mention by 8 hours.
    intent = _EmailIntent(
        proposes_new_meeting=True,
        requests_reschedule=False,
        proposed_start="2026-07-14T09:00:00",
        proposed_duration_minutes=30,
    )

    assert intent.proposed_start == datetime(2026, 7, 14, 9, 0, tzinfo=DEMO_TIMEZONE)


def test_email_intent_clears_proposed_reschedule_start_when_not_rescheduling():
    intent = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=False,
        proposed_reschedule_start="2026-07-16T09:15:00+00:00",
    )

    assert intent.proposed_reschedule_start is None


def test_email_intent_normalizes_naive_proposed_reschedule_start_to_demo_timezone():
    # A reschedule request can name a target time ("Thursday, same time")
    # independently of proposes_new_meeting, which is always false for a
    # reschedule-classified email -- so this field's naive-datetime handling
    # needs the same DEMO_TIMEZONE fallback as proposed_start, gated on
    # requests_reschedule instead.
    intent = _EmailIntent(
        proposes_new_meeting=False,
        requests_reschedule=True,
        references_event_id="evt_client_call",
        proposed_reschedule_start="2026-07-16T09:15:00",
    )

    assert intent.proposed_reschedule_start == datetime(2026, 7, 16, 9, 15, tzinfo=DEMO_TIMEZONE)


# _analyze_email is mocked below rather than calling a real LLM: these tests
# assert on detect_actions' handling of the LLM's output (overlap math,
# reschedule lookup), not on the LLM's own judgment, and no live API calls
# belong in the automated suite (see docs/spec/ai-secretary.md Testing Strategy).
def _no_intent() -> _EmailIntent:
    return _EmailIntent(proposes_new_meeting=False, requests_reschedule=False)


def _intent_for(email_id: str) -> _EmailIntent:
    if email_id == "email_meeting_request":
        return _EmailIntent(
            proposes_new_meeting=True,
            requests_reschedule=False,
            proposed_start=resolve_relative_time(
                _FIXTURE_EVENTS["evt_standup"].start_relative, NOW
            ),
            proposed_duration_minutes=30,
        )
    if email_id == "email_reschedule":
        return _EmailIntent(
            proposes_new_meeting=False,
            requests_reschedule=True,
            references_event_id="evt_client_call",
            # Matches the fixture email's body: "push our client sync from
            # tomorrow to Thursday instead? Same time works" -- two days
            # after evt_client_call's "+1d 09:15", same time of day.
            proposed_reschedule_start=resolve_relative_time("+3d 09:15", NOW),
        )
    return _no_intent()


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_finds_email_meeting_request_conflict(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {item.kind for item in result["action_items"]}
    assert "email_conflict" in kinds


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_email_conflict_carries_the_proposed_time(mock_analyze):
    # EmailConflict used to discard the LLM-extracted proposed_start/duration
    # after using them once to compute the overlap -- a future step that
    # needs to generate a shift proposal would have nothing to work with.
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    email_conflicts = [item for item in result["action_items"] if isinstance(item, EmailConflict)]
    assert len(email_conflicts) == 1
    expected_intent = _intent_for("email_meeting_request")
    assert email_conflicts[0].proposed_start == expected_intent.proposed_start
    assert email_conflicts[0].proposed_duration_minutes == expected_intent.proposed_duration_minutes


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_finds_email_conflict_with_zero_duration_proposal(mock_analyze):
    # Reproduces a bug: a truthiness check (`and intent.proposed_duration_minutes`)
    # treated a legitimate 0-minute proposal the same as "unset", silently
    # skipping conflict detection for it.
    mock_analyze.return_value = _EmailIntent(
        proposes_new_meeting=True,
        requests_reschedule=False,
        proposed_start=STANDUP.start + timedelta(minutes=15),
        proposed_duration_minutes=0,
    )
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {item.kind for item in result["action_items"]}
    assert "email_conflict" in kinds


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_groups_email_conflict_across_multiple_overlapping_events(mock_analyze):
    # Live finding: a proposed meeting overlapping two calendar events used
    # to produce two independent EmailConflict items for the same email,
    # which meant each got resolved separately (e.g. two contradictory
    # Gmail drafts on the same thread when both were answered via
    # draft_reply). One email conflicting with several events should be one
    # action item, not one per overlap.
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP, CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    email_conflicts = [item for item in result["action_items"] if isinstance(item, EmailConflict)]
    assert len(email_conflicts) == 1
    assert {e.id for e in email_conflicts[0].events} == {STANDUP.id, CLIENT_CALL.id}


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_finds_reschedule_request(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [RESCHEDULE_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {item.kind for item in result["action_items"]}
    assert "reschedule" in kinds


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_reschedule_request_carries_the_proposed_time(mock_analyze):
    # RescheduleRequest's target time used to be discarded before it ever
    # reached _find_email_actions (nulled by _EmailIntent._normalize
    # whenever proposes_new_meeting was false, which it always is for a
    # reschedule-classified email) -- a future shift-proposal step would
    # have no way to know the sender asked specifically for "Thursday".
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [RESCHEDULE_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    reschedules = [item for item in result["action_items"] if isinstance(item, RescheduleRequest)]
    assert len(reschedules) == 1
    expected_intent = _intent_for("email_reschedule")
    assert reschedules[0].proposed_reschedule_start == expected_intent.proposed_reschedule_start


@patch("agentic_secretary.graph._analyze_email")
def test_detect_actions_no_false_positive_for_mention_and_digest_emails(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MENTION_EMAIL, DIGEST_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    assert result["action_items"] == []
