from datetime import datetime, timedelta, timezone

from agentic_secretary.graph import detect_conflicts
from agentic_secretary.tools import CalendarEvent


def _event(id_: str, title: str, start: datetime, duration_minutes: int) -> CalendarEvent:
    return CalendarEvent(
        id=id_, title=title, start=start, end=start + timedelta(minutes=duration_minutes)
    )


STANDUP = _event("evt_standup", "Team Standup", datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc), 30)
CLIENT_CALL = _event(
    "evt_client_call", "Client Sync", datetime(2026, 7, 14, 9, 15, tzinfo=timezone.utc), 45
)
LUNCH = _event("evt_lunch", "Lunch", datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc), 60)
REVIEW = _event(
    "evt_review", "Design Review", datetime(2026, 7, 14, 13, 0, tzinfo=timezone.utc), 45
)


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
