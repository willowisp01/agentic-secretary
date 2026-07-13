"""Seed the burner Gmail/Calendar account with fixture data for local demos.

Reads seed_data/*.yaml, resolves each fixture's relative-time string against
"now", and inserts the result directly via the Gmail/Calendar APIs —
`agentic_secretary.tools` is deliberately not used here, since its
`draft_reply`/`propose_event` only ever prepare drafts/proposals and never
call an endpoint that lands data in the account (see tools.py docstrings).
Seeding needs exactly that: `messages().insert()` for mail that should
appear already-received, and `events().insert()` for real calendar events.
"""

import base64
import re
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

from googleapiclient.discovery import Resource, build

from _google_account_safety import confirm_target_account
from agentic_secretary.auth import load_credentials
from agentic_secretary.config import settings
from agentic_secretary.seed_data import (
    CalendarEvent,
    Email,
    load_calendar_events,
    load_emails,
    load_relations,
    validate_relations,
)

SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"

# Broader than the agent's runtime SCOPES (auth.py) on purpose, and kept on a
# separate token cache (settings.google_seed_token_path) so this elevated
# write grant never leaks into the agent's runtime credentials.
SEED_SCOPES = [
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.metadata",
    "https://www.googleapis.com/auth/calendar.events",
]

_OFFSET_RE = re.compile(r"^([+-])(\d+)([mhd])$")
_DAY_TIME_RE = re.compile(r"^([+-])(\d+)d (\d{2}):(\d{2})$")
_OFFSET_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def resolve_relative_time(relative: str, now: datetime) -> datetime:
    """Resolve a fixture's relative-time string against `now`.

    Two conventions appear in the fixtures: a plain offset like "-2h"/"+30m"
    (emails' `sent_relative`), and a day-offset plus wall-clock time like
    "+1d 09:00" (events' `start_relative`), which anchors to a specific time
    of day rather than "now plus N hours".
    """
    day_time_match = _DAY_TIME_RE.match(relative)
    if day_time_match:
        sign, days, hour, minute = day_time_match.groups()
        day_delta = timedelta(days=int(days) * (1 if sign == "+" else -1))
        target_date = (now + day_delta).date()
        return datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            int(hour),
            int(minute),
            tzinfo=now.tzinfo,
        )

    offset_match = _OFFSET_RE.match(relative)
    if offset_match:
        sign, amount, unit = offset_match.groups()
        signed_amount = int(amount) * (1 if sign == "+" else -1)
        return now + timedelta(**{_OFFSET_UNITS[unit]: signed_amount})

    raise ValueError(f"Unrecognized relative-time format: {relative!r}")


def _build_email_insert_body(email: Email, now: datetime) -> dict:
    sent_at = resolve_relative_time(email.sent_relative, now)
    message = MIMEText(email.body)
    message["From"] = email.from_
    message["To"] = email.to
    message["Subject"] = email.subject
    message["Date"] = sent_at.strftime("%a, %d %b %Y %H:%M:%S %z")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return {"raw": raw, "labelIds": ["INBOX", "UNREAD"]}


def _build_event_insert_body(event: CalendarEvent, now: datetime) -> dict:
    start_at = resolve_relative_time(event.start_relative, now)
    end_at = start_at + timedelta(minutes=event.duration_minutes)
    return {
        "summary": event.title,
        "start": {"dateTime": start_at.isoformat()},
        "end": {"dateTime": end_at.isoformat()},
    }


def _seed_email(service: Resource, email: Email, now: datetime) -> str:
    body = _build_email_insert_body(email, now)
    result = (
        service.users()
        .messages()
        .insert(userId="me", internalDateSource="dateHeader", body=body)
        .execute()
    )
    return result["id"]


def _seed_event(service: Resource, event: CalendarEvent, now: datetime) -> str:
    body = _build_event_insert_body(event, now)
    result = service.events().insert(calendarId="primary", body=body).execute()
    return result["id"]


def seed(
    gmail_service: Resource,
    calendar_service: Resource,
    now: datetime | None = None,
) -> None:
    """Load, validate, and insert all seed_data fixtures into the given services."""
    now = now or datetime.now(timezone.utc)

    emails = load_emails(SEED_DATA_DIR / "emails.yaml")
    events = load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
    relations = load_relations(SEED_DATA_DIR / "relations.yaml")
    validate_relations(
        relations,
        email_ids={e.id for e in emails},
        event_ids={e.id for e in events},
    )

    print(f"Seeding {len(emails)} emails and {len(events)} events...")
    for email in emails:
        message_id = _seed_email(gmail_service, email, now)
        print(f"  email {email.id!r} -> Gmail message {message_id}")
    for event in events:
        event_id = _seed_event(calendar_service, event, now)
        print(f"  event {event.id!r} -> Calendar event {event_id}")
    print("Done.")


def main() -> None:
    creds = load_credentials(
        SEED_SCOPES, settings.google_seed_token_path, settings.google_client_secret_path
    )
    gmail_service = build("gmail", "v1", credentials=creds)
    calendar_service = build("calendar", "v3", credentials=creds)
    confirm_target_account(gmail_service, action="seed demo data")
    seed(gmail_service, calendar_service)


if __name__ == "__main__":
    main()
