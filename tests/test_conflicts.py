from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agentic_secretary import seed_data
from agentic_secretary.graph import detect_conflicts
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


def test_detect_conflicts_finds_calendar_overlap():
    state = {"emails": [], "calendar_events": [STANDUP, CLIENT_CALL], "conflicts": [], "status": "done"}

    result = detect_conflicts(state)

    kinds = {c["kind"] for c in result["conflicts"]}
    assert "calendar_overlap" in kinds


def test_detect_conflicts_finds_back_to_back():
    state = {"emails": [], "calendar_events": [LUNCH, REVIEW], "conflicts": [], "status": "done"}

    result = detect_conflicts(state)

    kinds = {c["kind"] for c in result["conflicts"]}
    assert "back_to_back" in kinds


def test_detect_conflicts_no_false_positive_for_well_spaced_events():
    state = {"emails": [], "calendar_events": [STANDUP, LUNCH], "conflicts": [], "status": "done"}

    result = detect_conflicts(state)

    assert result["conflicts"] == []


# _analyze_email is mocked below rather than calling a real LLM: these tests
# assert on detect_conflicts' handling of the LLM's output (overlap math,
# reschedule lookup), not on the LLM's own judgment, and no live API calls
# belong in the automated suite (see docs/spec/ai-secretary.md Testing Strategy).
NO_INTENT = {
    "proposes_new_meeting": False,
    "proposed_start": None,
    "proposed_duration_minutes": None,
    "references_event_id": None,
    "requests_reschedule": False,
}


def _intent_for(email_id: str) -> dict:
    if email_id == "email_meeting_request":
        return {
            **NO_INTENT,
            "proposes_new_meeting": True,
            "proposed_start": resolve_relative_time(
                _FIXTURE_EVENTS["evt_standup"].start_relative, NOW
            ).isoformat(),
            "proposed_duration_minutes": 30,
        }
    if email_id == "email_reschedule":
        return {**NO_INTENT, "references_event_id": "evt_client_call", "requests_reschedule": True}
    return NO_INTENT


@patch("agentic_secretary.graph._analyze_email")
def test_detect_conflicts_finds_email_meeting_request_conflict(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MEETING_REQUEST_EMAIL],
        "calendar_events": [STANDUP],
        "conflicts": [],
        "status": "done",
    }

    result = detect_conflicts(state)

    kinds = {c["kind"] for c in result["conflicts"]}
    assert "email_conflict" in kinds


@patch("agentic_secretary.graph._analyze_email")
def test_detect_conflicts_finds_reschedule_request(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [RESCHEDULE_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "conflicts": [],
        "status": "done",
    }

    result = detect_conflicts(state)

    kinds = {c["kind"] for c in result["conflicts"]}
    assert "reschedule" in kinds


@patch("agentic_secretary.graph._analyze_email")
def test_detect_conflicts_no_false_positive_for_mention_and_digest_emails(mock_analyze):
    mock_analyze.side_effect = lambda email, events: _intent_for(email.id)
    state = {
        "emails": [MENTION_EMAIL, DIGEST_EMAIL],
        "calendar_events": [CLIENT_CALL],
        "conflicts": [],
        "status": "done",
    }

    result = detect_conflicts(state)

    assert result["conflicts"] == []
