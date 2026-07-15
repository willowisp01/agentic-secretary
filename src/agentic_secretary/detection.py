from datetime import datetime, timedelta

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field, model_validator

from agentic_secretary import tools
from agentic_secretary.config import settings
from agentic_secretary.state import (
    ActionNeeded,
    BackToBackConflict,
    CalendarOverlapConflict,
    EmailConflict,
    PlannerState,
    RescheduleRequest,
)

# Gap threshold under which two adjacent events count as "back-to-back, no
# buffer" (soft conflict) rather than just a normally-spaced schedule.
NO_BUFFER_THRESHOLD = timedelta(minutes=15)


def _find_calendar_overlaps(
    events: list[tools.CalendarEvent],
) -> list[CalendarOverlapConflict]:
    conflicts: list[CalendarOverlapConflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for i, a in enumerate(sorted_events):
        for b in sorted_events[i + 1 :]:
            if a.start < b.end and b.start < a.end:
                conflicts.append(
                    CalendarOverlapConflict(
                        description=f"{a.title!r} overlaps with {b.title!r}",
                        events=(a, b),
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
                    events=(a, b),
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
    def _clear_irrelevant_fields(self) -> "_EmailIntent":
        # The LLM boundary isn't trustworthy about leaving unrelated fields
        # null (observed live: a digest email that mentions no event at all
        # still got a real, valid event id attached to references_event_id).
        # Normalize here so every caller doesn't have to re-derive this gating.
        if not self.proposes_new_meeting:
            self.proposed_start = None
            self.proposed_duration_minutes = None
        if not self.requests_reschedule:
            self.references_event_id = None
        return self


def _analyze_email(
    email: tools.EmailSummary, calendar_events: list[tools.CalendarEvent]
) -> _EmailIntent:
    """LLM-assisted extraction of an email's scheduling intent — the
    interpretive part deterministic time comparison can't do: does this email
    propose a new meeting time, or ask to reschedule/cancel an existing event
    versus merely mention it?
    """
    llm = ChatAnthropic(
        model_name=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
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
        "If proposed_start is set, express it using the same UTC offset shown "
        "in the existing calendar events above (not the email's received-at "
        "offset, which may differ)."
    )
    intent = structured_llm.invoke(prompt)
    return _correct_proposed_start_offset(intent, calendar_events)


def _correct_proposed_start_offset(
    intent: "_EmailIntent", calendar_events: list[tools.CalendarEvent]
) -> "_EmailIntent":
    """The LLM reliably reads a casual time like "9:15am" into the right
    wall-clock hour/minute, but has no trustworthy signal for which UTC
    offset to attach -- observed live defaulting to the email's received_at
    offset (itself just a parsing artifact: Gmail's internalDate always
    parses to UTC, regardless of the account's real timezone), while
    calendar events carry Google Calendar's actual local offset. Compared
    as absolute instants, the same intended wall-clock time can end up
    hours apart. Don't trust the LLM's offset choice -- re-stamp its
    wall-clock reading with the calendar's own offset instead.
    """
    if intent.proposed_start is None or not calendar_events:
        return intent
    local_tzinfo = calendar_events[0].start.tzinfo
    if intent.proposed_start.utcoffset() == calendar_events[0].start.utcoffset():
        return intent
    corrected_start = intent.proposed_start.replace(tzinfo=local_tzinfo)
    return intent.model_copy(update={"proposed_start": corrected_start})


def _find_email_conflicts(
    emails: list[tools.EmailSummary], calendar_events: list[tools.CalendarEvent]
) -> list[EmailConflict | RescheduleRequest]:
    conflicts: list[EmailConflict | RescheduleRequest] = []
    events_by_id = {event.id: event for event in calendar_events}

    print(
        f"=== DEBUG: {len(emails)} emails passed to _find_email_conflicts ==="
    )  # TEMP
    for e in emails:  # TEMP
        print(f"  id={e.id!r} subject={e.subject!r}")  # TEMP

    for email in emails:
        intent = _analyze_email(email, calendar_events)
        print(f"=== DEBUG: intent for {email.subject!r} -> {intent!r} ===")  # TEMP

        if (
            intent.proposes_new_meeting
            and intent.proposed_start
            and intent.proposed_duration_minutes
        ):
            proposed_start = intent.proposed_start
            proposed_end = proposed_start + timedelta(
                minutes=intent.proposed_duration_minutes
            )
            overlapping = [
                event
                for event in calendar_events
                if proposed_start < event.end and event.start < proposed_end
            ]
            print(  # TEMP
                f"=== DEBUG: proposed_start={proposed_start!r} proposed_end={proposed_end!r} "  # TEMP
                f"overlapping={[e.id for e in overlapping]!r} ==="  # TEMP
            )  # TEMP
            if overlapping:
                titles = ", ".join(repr(e.title) for e in overlapping)
                conflicts.append(
                    EmailConflict(
                        description=f"{email.subject!r} requests a time overlapping {titles}",
                        email=email,
                        events=overlapping,
                    )
                )

        if intent.requests_reschedule and intent.references_event_id in events_by_id:
            event = events_by_id[intent.references_event_id]
            conflicts.append(
                RescheduleRequest(
                    description=f"{email.subject!r} asks to reschedule {event.title!r}",
                    email=email,
                    event=event,
                )
            )

    return conflicts


def detect_actions(state: PlannerState) -> dict:
    calendar_events = state["calendar_events"]
    emails = state["emails"]
    action_items: list[ActionNeeded] = (
        _find_calendar_overlaps(calendar_events)
        + _find_back_to_back(calendar_events)
        + _find_email_conflicts(emails, calendar_events)
    )
    return {"action_items": action_items}
