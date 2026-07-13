import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agentic_secretary.seed_data import CalendarEvent, Email
from seed_demo_data import (
    _build_email_insert_body,
    _build_event_insert_body,
    resolve_relative_time,
    seed,
)

NOW = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)


def test_resolve_relative_time_offset_hours():
    assert resolve_relative_time("-2h", NOW) == NOW - timedelta(hours=2)


def test_resolve_relative_time_offset_minutes():
    assert resolve_relative_time("+30m", NOW) == NOW + timedelta(minutes=30)


def test_resolve_relative_time_offset_days():
    assert resolve_relative_time("-1d", NOW) == NOW - timedelta(days=1)


def test_resolve_relative_time_day_and_clock_time():
    resolved = resolve_relative_time("+1d 09:00", NOW)
    assert resolved == datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc)


def test_resolve_relative_time_rejects_unrecognized_format():
    with pytest.raises(ValueError):
        resolve_relative_time("tomorrow", NOW)


def test_build_email_insert_body_encodes_message_with_labels():
    email = Email(
        id="email_x",
        from_="alex@example.com",
        to="you@example.com",
        subject="Quick sync tomorrow?",
        body="Are you free tomorrow?",
        sent_relative="-2h",
    )

    body = _build_email_insert_body(email, NOW)

    assert body["labelIds"] == ["INBOX", "UNREAD"]
    raw = body["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    assert "Are you free tomorrow?" in decoded
    assert "Quick sync tomorrow?" in decoded


def test_build_event_insert_body_computes_end_from_duration():
    event = CalendarEvent(
        id="evt_standup",
        title="Team Standup",
        start_relative="+1d 09:00",
        duration_minutes=30,
    )

    body = _build_event_insert_body(event, NOW)

    assert body["summary"] == "Team Standup"
    assert body["start"]["dateTime"] == datetime(
        2026, 7, 11, 9, 0, tzinfo=timezone.utc
    ).isoformat()
    assert body["end"]["dateTime"] == datetime(
        2026, 7, 11, 9, 30, tzinfo=timezone.utc
    ).isoformat()


def test_seed_inserts_all_fixture_emails_and_events_via_direct_api_calls():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.insert.return_value.execute.return_value = {
        "id": "m1"
    }
    calendar_service = MagicMock()
    calendar_service.events.return_value.insert.return_value.execute.return_value = {
        "id": "e1"
    }

    seed(gmail_service, calendar_service, now=NOW)

    messages = gmail_service.users.return_value.messages.return_value
    assert messages.insert.call_count == 4
    messages.send.assert_not_called()

    events = calendar_service.events.return_value
    assert events.insert.call_count == 4
