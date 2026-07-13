from datetime import datetime, timedelta, timezone
from typing import TypedDict

from googleapiclient.discovery import Resource
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from agentic_secretary import tools
from agentic_secretary.config import settings

# Gap threshold under which two adjacent events count as "back-to-back, no
# buffer" (soft conflict) rather than just a normally-spaced schedule.
NO_BUFFER_THRESHOLD = timedelta(minutes=15)


class Conflict(TypedDict):
    kind: str
    description: str
    events: list[tools.CalendarEvent]
    email: tools.EmailSummary | None


class PlannerState(TypedDict):
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    conflicts: list[Conflict]
    status: str


def _find_calendar_overlaps(events: list[tools.CalendarEvent]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for i, a in enumerate(sorted_events):
        for b in sorted_events[i + 1 :]:
            if a.start < b.end and b.start < a.end:
                conflicts.append(
                    Conflict(
                        kind="calendar_overlap",
                        description=f"{a.title!r} overlaps with {b.title!r}",
                        events=[a, b],
                        email=None,
                    )
                )
    return conflicts


def _find_back_to_back(events: list[tools.CalendarEvent]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for a, b in zip(sorted_events, sorted_events[1:]):
        gap = b.start - a.end
        if timedelta(0) <= gap <= NO_BUFFER_THRESHOLD:
            conflicts.append(
                Conflict(
                    kind="back_to_back",
                    description=f"{a.title!r} ends right as {b.title!r} starts, no buffer",
                    events=[a, b],
                    email=None,
                )
            )
    return conflicts


class _EmailIntent(BaseModel):
    proposes_new_meeting: bool = Field(
        description="True only if the email proposes a specific new meeting time "
        "that is not about an existing calendar event listed below."
    )
    proposed_start: datetime | None = Field(
        default=None,
        description="Datetime of the proposed meeting's start, resolved against "
        "the email's received time. Null unless proposes_new_meeting is true.",
    )
    proposed_duration_minutes: int | None = Field(
        default=None,
        description="Duration of the proposed meeting in minutes. Null unless "
        "proposes_new_meeting is true.",
    )
    references_event_id: str | None = Field(
        default=None,
        description="The id of an existing calendar event this email discusses, if any.",
    )
    requests_reschedule: bool = Field(
        description="True only if the email explicitly asks to move or cancel "
        "references_event_id, as opposed to merely mentioning it in passing."
    )

    @model_validator(mode="after")
    def _normalize(self) -> "_EmailIntent":
        # The LLM boundary isn't trustworthy about leaving unrelated fields
        # null (observed live: a digest email that mentions no event at all
        # still got a real, valid event id attached to references_event_id).
        # Normalize here so every caller doesn't have to re-derive this gating.
        if not self.proposes_new_meeting:
            self.proposed_start = None
            self.proposed_duration_minutes = None
        if not self.requests_reschedule:
            self.references_event_id = None

        # Nothing forces the LLM to include a UTC offset, and a naive
        # datetime compared against a timezone-aware CalendarEvent time
        # raises TypeError. The prompt gives the LLM UTC-anchored times, so
        # treat a naive response as UTC rather than leaving it to crash the
        # comparison in _find_email_conflicts.
        if self.proposed_start is not None and self.proposed_start.tzinfo is None:
            self.proposed_start = self.proposed_start.replace(tzinfo=timezone.utc)

        return self


def _analyze_email(
    email: tools.EmailSummary, calendar_events: list[tools.CalendarEvent]
) -> _EmailIntent:
    """LLM-assisted extraction of an email's scheduling intent — the
    interpretive part deterministic time comparison can't do: does this email
    propose a new meeting time, or ask to reschedule/cancel an existing event
    versus merely mention it?
    """
    llm = ChatAnthropic(model_name=settings.model_name, api_key=settings.anthropic_api_key)
    structured_llm = llm.with_structured_output(_EmailIntent, method="json_schema")
    events_context = (
        "\n".join(
            f"- id={e.id!r} title={e.title!r} start={e.start.isoformat()} end={e.end.isoformat()}"
            for e in calendar_events
        )
        or "(none)"
    )
    prompt = (
        "Analyze this email against the recipient's calendar to detect scheduling "
        "conflicts.\n\n"
        f"Email received at: {email.received_at.isoformat()}\n"
        f"Subject: {email.subject}\n"
        f"Body:\n{email.body}\n\n"
        f"Existing calendar events:\n{events_context}"
    )
    return structured_llm.invoke(prompt)


def _find_email_conflicts(
    emails: list[tools.EmailSummary], calendar_events: list[tools.CalendarEvent]
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    events_by_id = {event.id: event for event in calendar_events}

    for email in emails:
        intent = _analyze_email(email, calendar_events)

        if (
            intent.proposes_new_meeting
            and intent.proposed_start is not None
            and intent.proposed_duration_minutes is not None
        ):
            proposed_start = intent.proposed_start
            proposed_end = proposed_start + timedelta(minutes=intent.proposed_duration_minutes)
            for event in calendar_events:
                if proposed_start < event.end and event.start < proposed_end:
                    conflicts.append(
                        Conflict(
                            kind="email_conflict",
                            description=(
                                f"{email.subject!r} requests a time overlapping {event.title!r}"
                            ),
                            events=[event],
                            email=email,
                        )
                    )

        if intent.requests_reschedule and intent.references_event_id in events_by_id:
            event = events_by_id[intent.references_event_id]
            conflicts.append(
                Conflict(
                    kind="reschedule",
                    description=f"{email.subject!r} asks to reschedule {event.title!r}",
                    events=[event],
                    email=email,
                )
            )

    return conflicts


def detect_conflicts(state: PlannerState) -> dict:
    calendar_events = state["calendar_events"]
    emails = state["emails"]
    conflicts = (
        _find_calendar_overlaps(calendar_events)
        + _find_back_to_back(calendar_events)
        + _find_email_conflicts(emails, calendar_events)
    )
    return {"conflicts": conflicts}


def build_graph(gmail_service: Resource, calendar_service: Resource):
    def fetch_emails(state: PlannerState) -> dict:
        return {"emails": tools.list_recent_emails(gmail_service)}

    def check_calendar(state: PlannerState) -> dict:
        return {
            "calendar_events": tools.list_upcoming_events(calendar_service),
            "status": "done",
        }

    builder = StateGraph(PlannerState)
    builder.add_node("fetch_emails", fetch_emails)
    builder.add_node("check_calendar", check_calendar)
    builder.add_node("detect_conflicts", detect_conflicts)
    builder.add_edge(START, "fetch_emails")
    builder.add_edge("fetch_emails", "check_calendar")
    builder.add_edge("check_calendar", "detect_conflicts")
    builder.add_edge("detect_conflicts", END)
    return builder.compile(checkpointer=InMemorySaver())
