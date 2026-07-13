import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock

from agentic_secretary.tools import (
    CalendarEvent,
    DraftResult,
    EmailSummary,
    EventProposal,
    draft_reply,
    list_recent_emails,
    list_upcoming_events,
    propose_event,
)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8").rstrip("=")


def test_list_recent_emails_parses_simple_message():
    service = MagicMock()
    messages = service.users.return_value.messages.return_value
    messages.list.return_value.execute.return_value = {
        "messages": [{"id": "m1", "threadId": "t1"}]
    }
    messages.get.return_value.execute.return_value = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "alex@example.com"},
                {"name": "To", "value": "you@example.com"},
                {"name": "Subject", "value": "Quick sync tomorrow?"},
            ],
            "body": {"data": _b64("Are you free tomorrow?")},
        },
    }

    emails = list_recent_emails(service, max_results=5)

    messages.list.assert_called_once_with(userId="me", maxResults=5, labelIds=["INBOX"])
    assert emails == [
        EmailSummary(
            id="m1",
            thread_id="t1",
            from_="alex@example.com",
            to="you@example.com",
            subject="Quick sync tomorrow?",
            body="Are you free tomorrow?",
            received_at=datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )
    ]


def test_list_recent_emails_extracts_body_from_multipart_message():
    service = MagicMock()
    messages = service.users.return_value.messages.return_value
    messages.list.return_value.execute.return_value = {
        "messages": [{"id": "m1", "threadId": "t1"}]
    }
    messages.get.return_value.execute.return_value = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "alex@example.com"},
                {"name": "To", "value": "you@example.com"},
                {"name": "Subject", "value": "Quick sync tomorrow?"},
            ],
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64("<p>Are you free?</p>")},
                },
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64("Are you free?")},
                },
            ],
        },
    }

    emails = list_recent_emails(service)

    assert emails[0].body == "Are you free?"


def test_list_upcoming_events_parses_timed_and_all_day_events():
    service = MagicMock()
    service.events.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "e1",
                "summary": "Team Standup",
                "start": {"dateTime": "2026-07-10T09:00:00-07:00"},
                "end": {"dateTime": "2026-07-10T09:30:00-07:00"},
            },
            {
                "id": "e2",
                "summary": "Company Offsite",
                "start": {"date": "2026-07-11"},
                "end": {"date": "2026-07-12"},
            },
        ]
    }

    events = list_upcoming_events(service, max_results=10)

    list_kwargs = service.events.return_value.list.call_args.kwargs
    assert list_kwargs["calendarId"] == "primary"
    assert list_kwargs["maxResults"] == 10
    assert list_kwargs["singleEvents"] is True
    assert list_kwargs["orderBy"] == "startTime"
    assert isinstance(list_kwargs["timeMin"], str)
    assert events == [
        CalendarEvent(
            id="e1",
            title="Team Standup",
            start=datetime.fromisoformat("2026-07-10T09:00:00-07:00"),
            end=datetime.fromisoformat("2026-07-10T09:30:00-07:00"),
        ),
        CalendarEvent(
            id="e2",
            title="Company Offsite",
            # All-day events have no inherent timezone from the API; normalized
            # to UTC here so this is always comparable against timed events'
            # timezone-aware datetimes elsewhere (e.g. detect_conflicts'
            # overlap checks), which would otherwise raise TypeError.
            start=datetime(2026, 7, 11, tzinfo=timezone.utc),
            end=datetime(2026, 7, 12, tzinfo=timezone.utc),
        ),
    ]


def test_draft_reply_creates_draft_and_never_sends():
    service = MagicMock()
    service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
        "id": "draft1",
        "message": {"id": "m2", "threadId": "t1"},
    }

    result = draft_reply(
        service,
        to="alex@example.com",
        subject="Re: Quick sync tomorrow?",
        body="Sure, 9:15 works.",
        thread_id="t1",
    )

    assert result == DraftResult(draft_id="draft1", thread_id="t1")

    create_kwargs = (
        service.users.return_value.drafts.return_value.create.call_args.kwargs
    )
    assert create_kwargs["body"]["message"]["threadId"] == "t1"
    raw = create_kwargs["body"]["message"]["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    assert "Sure, 9:15 works." in decoded
    assert "Re: Quick sync tomorrow?" in decoded

    service.users.return_value.messages.return_value.send.assert_not_called()


def test_propose_event_returns_proposal_without_touching_calendar_api():
    service = MagicMock()
    start = datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc)

    proposal = propose_event(
        title="Client Sync",
        start=start,
        duration_minutes=45,
        attendees=["priya.patel@example.com"],
        existing_event_id="evt_client_call",
    )

    assert proposal == EventProposal(
        title="Client Sync",
        start=start,
        duration_minutes=45,
        attendees=["priya.patel@example.com"],
        existing_event_id="evt_client_call",
    )
    service.events.return_value.insert.assert_not_called()
    service.events.return_value.patch.assert_not_called()
