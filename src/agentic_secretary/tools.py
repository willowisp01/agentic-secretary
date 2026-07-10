import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText

from googleapiclient.discovery import Resource


@dataclass(frozen=True)
class EmailSummary:
    id: str
    thread_id: str
    from_: str
    to: str
    subject: str
    body: str
    received_at: datetime


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    title: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class DraftResult:
    draft_id: str
    thread_id: str


@dataclass(frozen=True)
class EventProposal:
    title: str
    start: datetime
    duration_minutes: int
    attendees: list[str] | None = None
    existing_event_id: str | None = None


def _get_header(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""


def _decode_body(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8")


def _extract_plain_text_body(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return _decode_body(data)
    for part in payload.get("parts") or []:
        body = _extract_plain_text_body(part)
        if body:
            return body
    return ""


def _parse_event_time(time_field: dict[str, str]) -> datetime:
    if "dateTime" in time_field:
        return datetime.fromisoformat(time_field["dateTime"])
    return datetime.fromisoformat(time_field["date"])


def list_recent_emails(service: Resource, max_results: int = 10) -> list[EmailSummary]:
    """Fetch the most recent inbox messages, parsed into typed summaries."""
    response = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
        .execute()
    )

    summaries = []
    for ref in response.get("messages", []):
        message = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="full")
            .execute()
        )
        headers = message["payload"]["headers"]
        summaries.append(
            EmailSummary(
                id=message["id"],
                thread_id=message["threadId"],
                from_=_get_header(headers, "From"),
                to=_get_header(headers, "To"),
                subject=_get_header(headers, "Subject"),
                body=_extract_plain_text_body(message["payload"]),
                received_at=datetime.fromtimestamp(
                    int(message["internalDate"]) / 1000, tz=timezone.utc
                ),
            )
        )
    return summaries


def list_upcoming_events(
    service: Resource, max_results: int = 10
) -> list[CalendarEvent]:
    """Fetch upcoming calendar events, parsed into typed objects."""
    now = datetime.now(timezone.utc).isoformat()
    response = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    return [
        CalendarEvent(
            id=item["id"],
            title=item.get("summary", ""),
            start=_parse_event_time(item["start"]),
            end=_parse_event_time(item["end"]),
        )
        for item in response.get("items", [])
    ]


def draft_reply(
    service: Resource, to: str, subject: str, body: str, thread_id: str
) -> DraftResult:
    """Create a Gmail draft reply. Never sends — only prepares a draft."""
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    result = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw, "threadId": thread_id}})
        .execute()
    )
    return DraftResult(draft_id=result["id"], thread_id=result["message"]["threadId"])


def propose_event(
    title: str,
    start: datetime,
    duration_minutes: int,
    attendees: list[str] | None = None,
    existing_event_id: str | None = None,
) -> EventProposal:
    """Build a structured event proposal. Never calls Calendar's insert/patch."""
    return EventProposal(
        title=title,
        start=start,
        duration_minutes=duration_minutes,
        attendees=attendees,
        existing_event_id=existing_event_id,
    )
