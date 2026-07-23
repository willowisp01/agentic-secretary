import base64
from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_secretary.seed_data import load_calendar_events, load_emails
from seed_demo_data import (
    _build_email_insert_body,
    _build_event_insert_body,
    resolve_relative_time,
    seed,
)

NOW = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)
SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"
_FIXTURE_EMAILS = {e.id: e for e in load_emails(SEED_DATA_DIR / "emails.yaml")}
_FIXTURE_EVENTS = {
    e.id: e for e in load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
}


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
    # email_meeting_request's body has a non-ASCII em-dash, which makes
    # MIMEText pick base64 for its own inner Content-Transfer-Encoding --
    # parse the MIME message rather than substring-searching the outer
    # decoded bytes, which would still contain that inner encoding as-is.
    email = _FIXTURE_EMAILS["email_meeting_request"]

    body = _build_email_insert_body(email, NOW)

    assert body["labelIds"] == ["INBOX", "UNREAD"]
    raw = body["raw"]
    decoded_bytes = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    message = message_from_bytes(decoded_bytes)
    assert message["Subject"] == email.subject
    assert message.get_payload(decode=True).decode("utf-8") == email.body


def test_build_event_insert_body_computes_end_from_duration():
    event = _FIXTURE_EVENTS["evt_standup"]

    body = _build_event_insert_body(event, NOW)

    expected_start = resolve_relative_time(event.start_relative, NOW)
    expected_end = expected_start + timedelta(minutes=event.duration_minutes)
    assert body["summary"] == event.title
    assert body["start"]["dateTime"] == expected_start.isoformat()
    assert body["end"]["dateTime"] == expected_end.isoformat()


def test_seed_uses_demo_timezone_for_default_now():
    # Reproduces a live-discovered bug: seed() defaulted `now` to UTC, so a
    # day-time fixture like "+1d 09:00" (meant as 9am in the demo persona's
    # local time) was inserted as 9am UTC -- landing at 5pm in the burner
    # account's actual +08:00 calendar. No `now=` override here: this
    # exercises the real default, unlike the other tests in this file.
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.insert.return_value.execute.return_value = {
        "id": "m1"
    }
    calendar_service = MagicMock()
    calendar_service.events.return_value.insert.return_value.execute.return_value = {
        "id": "e1"
    }

    seed(gmail_service, calendar_service)

    events = calendar_service.events.return_value
    first_call_body = events.insert.call_args_list[0].kwargs["body"]
    assert first_call_body["start"]["dateTime"].endswith("+08:00")


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
    assert messages.insert.call_count == len(_FIXTURE_EMAILS)
    messages.send.assert_not_called()

    events = calendar_service.events.return_value
    assert events.insert.call_count == len(_FIXTURE_EVENTS)
