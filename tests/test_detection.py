from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agentic_secretary import seed_data
from agentic_secretary.detection import _EmailIntent, detect_actions
from agentic_secretary.tools import CalendarEvent, EmailSummary
from seed_demo_data import resolve_relative_time

SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

_FIXTURE_EMAILS = {
    e.id: e for e in seed_data.load_emails(SEED_DATA_DIR / "emails.yaml")
}
_FIXTURE_EVENTS = {
    e.id: e
    for e in seed_data.load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
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
    state = {
        "emails": [],
        "calendar_events": [STANDUP, CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {a.kind for a in result["action_items"]}
    assert "calendar_overlap" in kinds


def test_detect_actions_finds_back_to_back():
    state = {
        "emails": [],
        "calendar_events": [LUNCH, REVIEW],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {a.kind for a in result["action_items"]}
    assert "back_to_back" in kinds


def test_detect_actions_no_false_positive_for_well_spaced_events():
    state = {
        "emails": [],
        "calendar_events": [STANDUP, LUNCH],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    assert result["action_items"] == []


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
        )
    return _no_intent()


@patch("agentic_secretary.detection._analyze_email")
def test_detect_actions_finds_email_meeting_request_conflict(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {a.kind for a in result["action_items"]}
    assert "email_conflict" in kinds


@patch("agentic_secretary.detection._analyze_email")
def test_detect_actions_finds_reschedule_request(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [RESCHEDULE_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    kinds = {a.kind for a in result["action_items"]}
    assert "reschedule" in kinds


@patch("agentic_secretary.detection._analyze_email")
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


@patch("agentic_secretary.detection._analyze_email")
def test_detect_actions_groups_multiple_overlapping_events_into_one_email_conflict(
    mock_analyze,
):
    # The multi-event fix: an email whose proposed time collides with more
    # than one existing event should produce a single EmailConflict
    # referencing all of them, not one EmailConflict per collided event.
    early = CalendarEvent(
        id="e1", title="Early Meeting", start=NOW, end=NOW + timedelta(minutes=60)
    )
    late = CalendarEvent(
        id="e2",
        title="Late Meeting",
        start=NOW + timedelta(minutes=30),
        end=NOW + timedelta(minutes=90),
    )
    mock_analyze.return_value = _EmailIntent(
        proposes_new_meeting=True,
        requests_reschedule=False,
        proposed_start=NOW,
        proposed_duration_minutes=90,
    )
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [early, late],
        "action_items": [],
        "status": "done",
    }

    result = detect_actions(state)

    email_conflicts = [a for a in result["action_items"] if a.kind == "email_conflict"]
    assert len(email_conflicts) == 1
    assert {e.id for e in email_conflicts[0].events} == {"e1", "e2"}
