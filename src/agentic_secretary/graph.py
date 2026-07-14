import operator
from datetime import datetime, timedelta
from typing import Annotated, Literal, TypedDict

from googleapiclient.discovery import Resource
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
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


ChatIntentValue = Literal["check_actions", "others"]


class ActionResolution(BaseModel):
    action_item: ActionNeeded
    remedy: Literal["shift_slot", "draft_reply", "skip"]
    # Which event to shift, set whenever remedy is "shift_slot" regardless
    # of kind. CalendarOverlapConflict/BackToBackConflict reference two
    # events, so this disambiguates which one -- rather than mutating
    # action_item down to a single event, which would silently bypass its
    # own exactly-2 arity validator.
    shift_event_id: str | None = None
    # None for "skip" (no tool call) and, transiently, before the
    # content-generation step has produced a proposal for a remedy that
    # needs one.
    proposal: tools.EventProposal | tools.DraftResult | None = None


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    action_items: list[ActionNeeded]
    # Routing signal for classify_intent's conditional edge, not
    # conversation content -- overwritten each time classify_intent runs.
    intent: ChatIntentValue
    # Appended to (never replaced) as present_menu works through
    # action_items one at a time.
    resolutions: Annotated[list[ActionResolution], operator.add]
    # Which action_items entry is currently awaiting a menu choice.
    pending_action_index: int
    # The remedy choice (proposal not yet filled in) awaiting
    # content_generation, or None on an invalid menu choice that needs
    # re-prompting. Set on every present_menu return, so always defined by
    # the time its own routing function reads it.
    pending_resolution: ActionResolution | None
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


def greet(state: PlannerState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Hi! I'm your scheduling assistant. Ask me to check "
                "for conflicts and I'll take a look at your calendar and inbox."
            )
        ]
    }


class _ChatIntent(BaseModel):
    intent: ChatIntentValue = Field(
        description="'check_actions' if the user is asking to check for "
        "scheduling conflicts or action items (e.g. 'check for conflicts', "
        "'what's going on today'). 'others' for anything else, including "
        "unclear or unrelated replies."
    )


def _classify_intent(reply: str) -> _ChatIntent:
    llm = ChatAnthropic(model_name=settings.model_name, api_key=settings.anthropic_api_key)
    structured_llm = llm.with_structured_output(_ChatIntent, method="json_schema")
    return structured_llm.invoke(
        "Classify the user's reply in a chat session with a scheduling "
        f"assistant.\n\nUser reply: {reply}"
    )


def classify_intent(state: PlannerState) -> dict:
    # interrupt()'s value is what the CLI actually displays as the prompt,
    # so show whatever the assistant last said -- the greeting on first
    # entry, or the clarifying re-prompt appended below on a looped-back
    # entry (self-loop edge re-invokes this same node).
    last_ai_message = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    reply = interrupt(last_ai_message.content)

    chat_intent = _classify_intent(reply)
    new_messages: list[AnyMessage] = [HumanMessage(content=reply)]
    if chat_intent.intent == "others":
        new_messages.append(
            AIMessage(
                content="I didn't quite catch that -- try something like "
                "'check for conflicts'."
            )
        )

    return {"messages": new_messages, "intent": chat_intent.intent}


def _route_after_classify_intent(state: PlannerState) -> str:
    return state["intent"]


def detect_actions(state: PlannerState) -> dict:
    calendar_events = state["calendar_events"]
    emails = state["emails"]
    action_items: list[ActionNeeded] = (
        _find_calendar_overlaps(calendar_events)
        + _find_back_to_back(calendar_events)
        + _find_email_actions(emails, calendar_events)
    )
    result: dict = {"action_items": action_items, "pending_action_index": 0}
    if not action_items:
        result["messages"] = [AIMessage(content="No action items found -- everything looks clear!")]
    return result


def _route_after_detect_actions(state: PlannerState) -> str:
    return "present_menu" if state["action_items"] else "end"


_REMEDY_LABELS: dict[Literal["shift_slot", "draft_reply", "skip"], str] = {
    "shift_slot": "Shift the slot",
    "draft_reply": "Draft a reply email",
    "skip": "Skip",
}


def _applicable_remedies(item: ActionNeeded) -> list[Literal["shift_slot", "draft_reply", "skip"]]:
    # Deterministic, not LLM-judged: which remedies make sense is a
    # structural fact about the item's type, not a judgment call. Every
    # kind has at least one event to shift; only EmailConflict/
    # RescheduleRequest have an email to reply to at all.
    remedies: list[Literal["shift_slot", "draft_reply", "skip"]] = ["shift_slot"]
    if isinstance(item, EmailConflict | RescheduleRequest):
        remedies.append("draft_reply")
    remedies.append("skip")
    return remedies


def present_menu(state: PlannerState) -> dict:
    item = state["action_items"][state["pending_action_index"]]
    remedies = _applicable_remedies(item)

    menu_lines = [f"[{item.kind}] {item.description}"]
    menu_lines += [f"{i}. {_REMEDY_LABELS[r]}" for i, r in enumerate(remedies, start=1)]
    choice_text = interrupt("\n".join(menu_lines))

    try:
        remedy = remedies[int(choice_text.strip()) - 1]
    except (ValueError, IndexError):
        # Invalid choice: record nothing -- the routing after this node
        # checks pending_resolution, which stays None, so it re-shows this
        # same menu rather than proceeding to content_generation.
        return {"pending_resolution": None}

    shift_event_id: str | None = None
    if remedy == "shift_slot":
        if isinstance(item, CalendarOverlapConflict | BackToBackConflict):
            # Two events on this kind, unlike EmailConflict/RescheduleRequest
            # (exactly one) -- ask which one, rather than guessing.
            event_lines = [f"{i}. {e.title!r}" for i, e in enumerate(item.events, start=1)]
            which_text = interrupt("Which event should move?\n" + "\n".join(event_lines))
            try:
                shift_event_id = item.events[int(which_text.strip()) - 1].id
            except (ValueError, IndexError):
                shift_event_id = item.events[0].id
        else:
            shift_event_id = item.event.id

    # Not yet appended to resolutions or advancing pending_action_index --
    # content_generation (next) fills in proposal and finalizes both, since
    # skip needs no LLM/tool call but shift_slot/draft_reply do.
    resolution = ActionResolution(action_item=item, remedy=remedy, shift_event_id=shift_event_id)
    return {"pending_resolution": resolution}


def _route_after_present_menu(state: PlannerState) -> str:
    return "content_generation" if state["pending_resolution"] is not None else "present_menu"


def _event_by_id(item: ActionNeeded, event_id: str) -> tools.CalendarEvent:
    if isinstance(item, CalendarOverlapConflict | BackToBackConflict):
        return next(e for e in item.events if e.id == event_id)
    return item.event


class _ShiftProposal(BaseModel):
    new_start: datetime = Field(
        description="The proposed new start time for the event being moved. "
        "Must not overlap any of the busy times listed. If the request names "
        "a specific target time, use it; otherwise pick a reasonable nearby "
        "free slot."
    )

    @model_validator(mode="after")
    def _normalize(self) -> "_ShiftProposal":
        if self.new_start.tzinfo is None:
            self.new_start = self.new_start.replace(tzinfo=DEMO_TIMEZONE)
        return self


def _busy_times_context(
    calendar_events: list[tools.CalendarEvent], resolutions: list[ActionResolution]
) -> str:
    lines = [f"- {e.title!r}: {e.start.isoformat()} to {e.end.isoformat()}" for e in calendar_events]
    for r in resolutions:
        if r.remedy == "shift_slot" and isinstance(r.proposal, tools.EventProposal):
            proposed_end = r.proposal.start + timedelta(minutes=r.proposal.duration_minutes)
            lines.append(
                f"- {r.proposal.title!r} (already proposed this session): "
                f"{r.proposal.start.isoformat()} to {proposed_end.isoformat()}"
            )
    return "\n".join(lines) or "(none)"


def _generate_shift_proposal(
    event_to_shift: tools.CalendarEvent,
    item: ActionNeeded,
    calendar_events: list[tools.CalendarEvent],
    resolutions: list[ActionResolution],
) -> tools.EventProposal:
    hint = ""
    if isinstance(item, RescheduleRequest) and item.proposed_reschedule_start is not None:
        hint = f"\n\nThe email requested this specific new time: {item.proposed_reschedule_start.isoformat()}."
    elif isinstance(item, EmailConflict):
        avoid_end = item.proposed_start + timedelta(minutes=item.proposed_duration_minutes)
        hint = (
            f"\n\nThis event is being moved to make room for a different "
            f"incoming request at {item.proposed_start.isoformat()} to "
            f"{avoid_end.isoformat()} -- do not propose a time that still "
            f"overlaps that."
        )

    prompt = (
        f"Propose a new time to move {event_to_shift.title!r} "
        f"(currently {event_to_shift.start.isoformat()} to "
        f"{event_to_shift.end.isoformat()}), because: {item.description}"
        f"{hint}\n\n"
        f"Busy times to avoid:\n{_busy_times_context(calendar_events, resolutions)}"
    )
    llm = ChatAnthropic(model_name=settings.model_name, api_key=settings.anthropic_api_key)
    structured_llm = llm.with_structured_output(_ShiftProposal, method="json_schema")
    result = structured_llm.invoke(prompt)

    duration_minutes = int((event_to_shift.end - event_to_shift.start).total_seconds() // 60)
    return tools.propose_event(
        title=event_to_shift.title,
        start=result.new_start,
        duration_minutes=duration_minutes,
        existing_event_id=event_to_shift.id,
    )


class _ReplyDraft(BaseModel):
    body: str = Field(description="Body text for the reply email.")


def _generate_reply_body(item: EmailConflict | RescheduleRequest) -> str:
    prompt = (
        "Draft a brief, professional reply to this email.\n\n"
        f"Original subject: {item.email.subject}\n"
        f"Original body:\n{item.email.body}\n\n"
        f"Context: {item.description}"
    )
    llm = ChatAnthropic(model_name=settings.model_name, api_key=settings.anthropic_api_key)
    structured_llm = llm.with_structured_output(_ReplyDraft, method="json_schema")
    return structured_llm.invoke(prompt).body


def _route_after_content_generation(state: PlannerState) -> str:
    if state["pending_action_index"] >= len(state["action_items"]):
        return "end"
    return "present_menu"


# Every custom type that ends up inside PlannerState -- LangGraph's default
# checkpoint serializer warns (and, in a future version, will refuse) to
# deserialize any type outside its own built-in allowlist, since every
# interrupt/resume round-trip goes through the checkpointer.
_CHECKPOINT_ALLOWED_TYPES = (
    tools.EmailSummary,
    tools.CalendarEvent,
    tools.EventProposal,
    tools.DraftResult,
    CalendarOverlapConflict,
    BackToBackConflict,
    EmailConflict,
    RescheduleRequest,
    ActionResolution,
)


def build_graph(gmail_service: Resource, calendar_service: Resource):
    def fetch_emails(state: PlannerState) -> dict:
        return {"emails": tools.list_recent_emails(gmail_service)}

    def check_calendar(state: PlannerState) -> dict:
        return {
            "calendar_events": tools.list_upcoming_events(calendar_service),
            "status": "done",
        }

    def content_generation(state: PlannerState) -> dict:
        pending = state["pending_resolution"]
        item = pending.action_item

        if pending.remedy == "skip":
            proposal = None
        elif pending.remedy == "shift_slot":
            event_to_shift = _event_by_id(item, pending.shift_event_id)
            proposal = _generate_shift_proposal(
                event_to_shift, item, state["calendar_events"], state["resolutions"]
            )
        else:  # draft_reply -- only EmailConflict/RescheduleRequest offer it
            body = _generate_reply_body(item)
            original_subject = item.email.subject
            subject = (
                original_subject
                if original_subject.lower().startswith("re:")
                else f"Re: {original_subject}"
            )
            # draft_reply actually calls Gmail's drafts().create() (unlike
            # propose_event, which is a pure constructor) -- never sends,
            # but is a real API write, so it needs gmail_service in scope.
            proposal = tools.draft_reply(
                gmail_service,
                to=item.email.from_,
                subject=subject,
                body=body,
                thread_id=item.email.thread_id,
            )

        finalized = pending.model_copy(update={"proposal": proposal})
        return {
            "resolutions": [finalized],
            "pending_action_index": state["pending_action_index"] + 1,
            "pending_resolution": None,
        }

    builder = StateGraph(PlannerState)
    builder.add_node("greet", greet)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("fetch_emails", fetch_emails)
    builder.add_node("check_calendar", check_calendar)
    builder.add_node("detect_actions", detect_actions)
    builder.add_node("present_menu", present_menu)
    builder.add_node("content_generation", content_generation)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classify_intent,
        {"check_actions": "fetch_emails", "others": "classify_intent"},
    )
    builder.add_edge("fetch_emails", "check_calendar")
    builder.add_edge("check_calendar", "detect_actions")
    builder.add_conditional_edges(
        "detect_actions",
        _route_after_detect_actions,
        {"present_menu": "present_menu", "end": END},
    )
    builder.add_conditional_edges(
        "present_menu",
        _route_after_present_menu,
        {"content_generation": "content_generation", "present_menu": "present_menu"},
    )
    builder.add_conditional_edges(
        "content_generation",
        _route_after_content_generation,
        {"present_menu": "present_menu", "end": END},
    )
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES)
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
