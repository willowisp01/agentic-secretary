from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_secretary.state import CalendarOverlapConflict, EmailConflict
from agentic_secretary.tools import CalendarEvent, EmailSummary

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

EVENT_A = CalendarEvent(id="a", title="A", start=NOW, end=NOW)
EVENT_B = CalendarEvent(id="b", title="B", start=NOW, end=NOW)
EVENT_C = CalendarEvent(id="c", title="C", start=NOW, end=NOW)

EMAIL = EmailSummary(
    id="m1",
    thread_id="t1",
    from_="alex@example.com",
    to="you@example.com",
    subject="Subject",
    body="Body",
    received_at=NOW,
)


def test_calendar_overlap_conflict_requires_exactly_two_events():
    with pytest.raises(ValidationError):
        CalendarOverlapConflict(description="x", events=(EVENT_A,))

    with pytest.raises(ValidationError):
        CalendarOverlapConflict(description="x", events=(EVENT_A, EVENT_B, EVENT_C))

    # Exactly 2 is valid.
    CalendarOverlapConflict(description="x", events=(EVENT_A, EVENT_B))


def test_email_conflict_requires_at_least_one_event():
    with pytest.raises(ValidationError):
        EmailConflict(description="x", email=EMAIL, events=[])

    # At least 1 is valid, and more than 1 is allowed (multi-event support).
    EmailConflict(description="x", email=EMAIL, events=[EVENT_A])
    EmailConflict(description="x", email=EMAIL, events=[EVENT_A, EVENT_B])
