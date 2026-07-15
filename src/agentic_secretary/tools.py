import base64
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText

from googleapiclient.discovery import Resource
from langchain_core.tools import BaseTool, tool


@dataclass(frozen=True)
class EmailSummary:
    """An email as actually read back from Gmail: `received_at` is an
    absolute datetime decoded from the API's internalDate, and `thread_id`
    is Gmail-assigned. Compare `seed_data.Email`, the pre-seeding fixture
    shape with a still-relative send time.
    """

    id: str
    thread_id: str
    from_: str
    to: str
    subject: str
    body: str
    received_at: datetime


@dataclass(frozen=True)
class CalendarEvent:
    """An event as actually read back from Calendar: `start`/`end` are
    absolute datetimes. Same class name as `seed_data.CalendarEvent` (the
    pre-seeding fixture shape with a relative `start_relative` string) —
    import with an alias if both are needed in one file.
    """

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
    # All-day events have no inherent timezone from the API; normalize to
    # UTC so this is always comparable against timed events' timezone-aware
    # datetimes elsewhere (e.g. detect_actions' overlap checks), which
    # would otherwise raise TypeError comparing naive vs. aware datetimes.
    return datetime.fromisoformat(time_field["date"]).replace(tzinfo=timezone.utc)


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


def make_draft_reply_tool(service: Resource) -> BaseTool:
    """Bind `draft_reply` to a specific Gmail service instance as a
    callable LangChain tool. `service` is a runtime dependency the LLM
    can't supply (it isn't JSON-schema-able), so it's captured via closure
    rather than exposed as a tool argument.
    """
    # LangGraph's ToolNode executes multiple tool calls from one AIMessage
    # concurrently via a thread pool -- e.g. the agent drafting replies to
    # two different emails in the same turn. googleapiclient's Resource
    # objects aren't thread-safe (live-discovered: concurrent draft_reply
    # calls sharing one service instance corrupted the TLS connection --
    # "SSL: WRONG_VERSION_NUMBER"). Serialize actual calls through this
    # service instance rather than relying on ToolNode to run sequentially.
    lock = threading.Lock()

    @tool
    def draft_reply_tool(
        to: str, subject: str, body: str, thread_id: str
    ) -> DraftResult:
        """Draft a reply email to the sender of an email-related action
        item — e.g. proposing a different time, or acknowledging a
        reschedule. Never sends; only prepares a Gmail draft for human
        review.
        """
        with lock:
            return draft_reply(service, to, subject, body, thread_id)

    return draft_reply_tool


def propose_event(
    title: str,
    start: datetime,
    duration_minutes: int,
    attendees: list[str] | None = None,
    existing_event_id: str | None = None,
) -> EventProposal:
    """Propose a calendar event time. Never calls Calendar's insert/patch —
    only builds a structured proposal for human review.

    Omit `existing_event_id` to propose a brand-new event — e.g. accepting
    a meeting request from an email at the time it suggests. Set
    `existing_event_id` to the id of an event already on the calendar to
    propose moving that event to a new start/duration instead.
    """
    return EventProposal(
        title=title,
        start=start,
        duration_minutes=duration_minutes,
        attendees=attendees,
        existing_event_id=existing_event_id,
    )


@tool("propose_event", response_format="content_and_artifact")
def propose_event_tool(
    title: str,
    start: datetime,
    duration_minutes: int,
    attendees: list[str] | None = None,
    existing_event_id: str | None = None,
) -> tuple[str, EventProposal]:
    """Propose a calendar event time. Never calls Calendar's insert/patch —
    only builds a structured proposal for human review.

    Omit `existing_event_id` to propose a brand-new event — e.g. accepting
    a meeting request from an email at the time it suggests. Set
    `existing_event_id` to the id of an event already on the calendar to
    propose moving that event to a new start/duration instead.
    """
    # response_format="content_and_artifact" keeps the real EventProposal
    # attached to the resulting ToolMessage as .artifact, not just its str()
    # in .content -- callers that need the structured value (e.g. the
    # deterministic collision check in review.py) don't have to re-parse a
    # repr string to get it back.
    proposal = propose_event(
        title, start, duration_minutes, attendees, existing_event_id
    )
    return str(proposal), proposal


@tool("withdraw_proposal")
def withdraw_proposal_tool(event_id_or_title: str) -> str:
    """Withdraw a previously made propose_event call for the same target,
    when you decide not to go through with it after all -- e.g. declining
    a time and asking for alternatives instead of committing to a new one.
    Pass the same value used as existing_event_id (for an existing event)
    or title (for a brand-new event) in the propose_event call being
    withdrawn. Without this, the collision check keeps treating the
    withdrawn proposal as still live even after you've moved on from it.
    """
    return event_id_or_title
