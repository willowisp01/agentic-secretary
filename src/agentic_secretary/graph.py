from datetime import datetime, timedelta
from typing import Annotated, Literal, TypedDict

from googleapiclient.discovery import Resource
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator

from agentic_secretary import tools
from agentic_secretary.config import DEMO_TIMEZONE, settings

# Gap threshold under which two adjacent events count as "back-to-back, no
# buffer" (soft conflict) rather than just a normally-spaced schedule.
NO_BUFFER_THRESHOLD = timedelta(minutes=15)


class CalendarOverlapConflict(BaseModel):
    kind: Literal["calendar_overlap"] = "calendar_overlap"
    description: str
    events: list[tools.CalendarEvent]

    @model_validator(mode="after")
    def _check_arity(self) -> "CalendarOverlapConflict":
        if len(self.events) != 2:
            raise ValueError("calendar_overlap requires exactly 2 events")
        return self


class BackToBackConflict(BaseModel):
    kind: Literal["back_to_back"] = "back_to_back"
    description: str
    events: list[tools.CalendarEvent]

    @model_validator(mode="after")
    def _check_arity(self) -> "BackToBackConflict":
        if len(self.events) != 2:
            raise ValueError("back_to_back requires exactly 2 events")
        return self


class EmailConflict(BaseModel):
    kind: Literal["email_conflict"] = "email_conflict"
    description: str
    event: tools.CalendarEvent
    email: tools.EmailSummary
    # The email's proposed time, kept (not just used transiently to compute
    # the overlap) so a future remedy step has something to build a shift
    # proposal from. Required, not optional: this type is only ever
    # constructed once both are already confirmed non-null (see
    # _find_email_actions).
    proposed_start: datetime
    proposed_duration_minutes: int


class RescheduleRequest(BaseModel):
    # Not a time collision like the other three kinds: it flags an email that
    # asks to move/cancel a referenced event (no start/end comparison is
    # involved). Kept in the same ActionNeeded union because it goes through
    # the same detect_actions -> remedy-menu pipeline as the true
    # time-collision kinds.
    kind: Literal["reschedule"] = "reschedule"
    description: str
    event: tools.CalendarEvent
    email: tools.EmailSummary
    # Optional, unlike EmailConflict's proposed_start/proposed_duration_minutes:
    # not every reschedule email names a specific target time ("something
    # came up, can we reschedule?" has none). No matching duration field --
    # a reschedule moves an existing event that already has one.
    proposed_reschedule_start: datetime | None = None


ActionNeeded = Annotated[
    CalendarOverlapConflict | BackToBackConflict | EmailConflict | RescheduleRequest,
    Field(discriminator="kind"),
]


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    action_items: list[ActionNeeded]
    status: str


def _find_calendar_overlaps(events: list[tools.CalendarEvent]) -> list[CalendarOverlapConflict]:
    conflicts: list[CalendarOverlapConflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for i, a in enumerate(sorted_events):
        for b in sorted_events[i + 1 :]:
            if a.start < b.end and b.start < a.end:
                conflicts.append(
                    CalendarOverlapConflict(
                        description=f"{a.title!r} overlaps with {b.title!r}",
                        events=[a, b],
                    )
                )
    return conflicts


def _find_back_to_back(events: list[tools.CalendarEvent]) -> list[BackToBackConflict]:
    conflicts: list[BackToBackConflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for a, b in zip(sorted_events, sorted_events[1:]):
        gap = b.start - a.end
        if timedelta(0) <= gap <= NO_BUFFER_THRESHOLD:
            conflicts.append(
                BackToBackConflict(
                    description=f"{a.title!r} ends right as {b.title!r} starts, no buffer",
                    events=[a, b],
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
    proposed_reschedule_start: datetime | None = Field(
        default=None,
        description="If requests_reschedule is true and the email specifies a "
        "target date/time for the move (e.g. 'Thursday, same time'), the "
        "resolved datetime of that target. Null if no specific target time is "
        "mentioned, or if requests_reschedule is false.",
    )

    @model_validator(mode="after")
    def _normalize(self) -> "_EmailIntent":
        # Unlike a typical mode="after" validator that rejects an invalid
        # instance (e.g. raising when width != height), this one repairs it:
        # LLM output is expected to be inconsistent sometimes, and erroring
        # out on every such response would be worse than just fixing it up.
        #
        # The LLM boundary isn't trustworthy about leaving unrelated fields
        # null (observed live: a digest email that mentions no event at all
        # still got a real, valid event id attached to references_event_id).
        # Normalize here so every caller doesn't have to re-derive this gating.
        if not self.proposes_new_meeting:
            self.proposed_start = None
            self.proposed_duration_minutes = None
        if not self.requests_reschedule:
            self.references_event_id = None
            self.proposed_reschedule_start = None

        # Nothing forces the LLM to include an offset, and a naive datetime
        # compared against a timezone-aware CalendarEvent time raises
        # TypeError. The prompt tells the LLM to assume DEMO_TIMEZONE for
        # bare times, so treat a naive response the same way rather than
        # defaulting to UTC (which would silently shift it by 8 hours) or
        # leaving it to crash downstream comparisons. Both time fields need
        # this independently: proposed_start is gated on proposes_new_meeting,
        # proposed_reschedule_start on requests_reschedule.
        for field_name in ("proposed_start", "proposed_reschedule_start"):
            value = getattr(self, field_name)
            if value is not None and value.tzinfo is None:
                setattr(self, field_name, value.replace(tzinfo=DEMO_TIMEZONE))

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
        f"Existing calendar events:\n{events_context}\n\n"
        f"If the email mentions a time with no explicit UTC offset (e.g. "
        f"\"9:15am\"), assume it's in {DEMO_TIMEZONE}, matching the calendar "
        f"events above."
    )
    return structured_llm.invoke(prompt)


def _find_email_actions(
    emails: list[tools.EmailSummary], calendar_events: list[tools.CalendarEvent]
) -> list[EmailConflict | RescheduleRequest]:
    # One loop, one _analyze_email call per email: the two checks below both
    # need the same LLM-extracted intent, so splitting this into two
    # functions would double the LLM calls per email for no benefit.
    actions: list[EmailConflict | RescheduleRequest] = []
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
                    actions.append(
                        EmailConflict(
                            description=(
                                f"{email.subject!r} requests a time overlapping {event.title!r}"
                            ),
                            event=event,
                            email=email,
                            proposed_start=proposed_start,
                            proposed_duration_minutes=intent.proposed_duration_minutes,
                        )
                    )

        if intent.requests_reschedule and intent.references_event_id in events_by_id:
            event = events_by_id[intent.references_event_id]
            actions.append(
                RescheduleRequest(
                    description=f"{email.subject!r} asks to reschedule {event.title!r}",
                    event=event,
                    email=email,
                    proposed_reschedule_start=intent.proposed_reschedule_start,
                )
            )

    return actions


def detect_actions(state: PlannerState) -> dict:
    calendar_events = state["calendar_events"]
    emails = state["emails"]
    action_items: list[ActionNeeded] = (
        _find_calendar_overlaps(calendar_events)
        + _find_back_to_back(calendar_events)
        + _find_email_actions(emails, calendar_events)
    )
    return {"action_items": action_items}


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
    builder.add_node("detect_actions", detect_actions)
    builder.add_edge(START, "fetch_emails")
    builder.add_edge("fetch_emails", "check_calendar")
    builder.add_edge("check_calendar", "detect_actions")
    builder.add_edge("detect_actions", END)
    return builder.compile(checkpointer=InMemorySaver())
