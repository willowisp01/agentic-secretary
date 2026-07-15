from datetime import datetime, timezone
from typing import Callable

from googleapiclient.discovery import Resource
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

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

SYSTEM_PROMPT = """You are an AI secretary helping a busy professional handle scheduling \
action items detected in their calendar and inbox.

You have three tools available:
- propose_event: propose a new calendar event time, or propose moving an existing one \
(set existing_event_id to move an existing event; omit it to propose a brand-new event, \
e.g. accepting a meeting request at the time it suggests).
- draft_reply_tool: draft a reply email to the sender of an email-related action item.
- withdraw_proposal: withdraw a previously made propose_event call for the same target, \
when you decide not to go through with it after all.

Boundaries (enforced in code, but stated here too): neither propose_event nor \
draft_reply_tool ever sends an email or books/patches a calendar event. Every tool call \
only ever produces a proposal or a draft for the human to review afterward -- never act \
as though something is final.

Whenever a draft reply commits to a specific time (not just asking an open question like \
"what time works?"), also call propose_event for that same time -- even if it's only \
tentative pending the recipient's reply. The human's collision check only sees times that \
went through propose_event; a specific time that exists only inside a drafted email's text \
is invisible to it, so a real conflict with an existing event could go unnoticed.

If you decide not to go through with a previously-proposed time after all -- e.g. declining \
it and asking for alternatives instead of committing to a new one -- call withdraw_proposal \
for it (pass the same existing_event_id or title you used in the original propose_event \
call). Otherwise the collision check keeps treating that abandoned proposal as still live \
even though you've moved on from it.

Work through every action item listed below using your own judgment about which tool (if \
either) applies. It's fine to skip an item and say why in your summary, rather than force \
a resolution that doesn't make sense. If you're genuinely unsure how to handle an item, \
don't guess -- ask a direct question in your response instead of calling a tool.

When you've addressed everything you can, reply with a plain-text summary of what you did \
(and anything you skipped or want to ask about) instead of calling another tool -- this \
ends your turn and shows the summary to the human for review.

If the human replies with a correction or follow-up after your summary (e.g. "move it to \
2pm instead"), treat it as amending the specific thing it refers to, using your own prior \
tool calls as context -- not as an unrelated new request. If they simply acknowledge or \
approve your summary (e.g. "ok", "approved", "looks good"), do not claim anything was \
booked, sent, or updated as a result -- nothing beyond what your own tool calls already \
did happens automatically. Acknowledge warmly without implying further action took place.

When computing a target date or time (from an email's wording or the human's own words), \
use the current date given below as your anchor rather than inferring or computing \
today's weekday yourself.

In your summaries, always state dates numerically (e.g. "2026-07-17"), never with a \
weekday name (not "Friday, 2026-07-17", not "Friday the 17th"). Live-discovered: weekday \
names in your own prose have repeatedly been wrong even when the underlying date was \
correct -- there's no need to compute or restate the weekday at all, so don't."""


def _format_event(event: tools.CalendarEvent) -> str:
    return (
        f"id={event.id!r} title={event.title!r} "
        f"start={event.start.isoformat()!r} end={event.end.isoformat()!r}"
    )


def _format_email(email: tools.EmailSummary) -> str:
    # Live-discovered: this used to omit `body` entirely -- the agent could
    # see that an email existed and its subject, but never the actual text
    # containing what the sender proposed ("are you free tomorrow at
    # 9:15am?"). It had no way to know a specific time without the human
    # repeating it, and correctly asked rather than guessing -- but the
    # right fix is giving it the information it needs, not relying on it to
    # keep asking.
    return (
        f"id={email.id!r} thread_id={email.thread_id!r} "
        f"from={email.from_!r} subject={email.subject!r} body={email.body!r}"
    )


def _format_action_item(item: ActionNeeded) -> str:
    match item:
        case CalendarOverlapConflict() | BackToBackConflict():
            events = ", ".join(_format_event(e) for e in item.events)
            return f"- kind={item.kind!r} description={item.description!r} events=[{events}]"
        case EmailConflict():
            events = ", ".join(_format_event(e) for e in item.events)
            return (
                f"- kind={item.kind!r} description={item.description!r} "
                f"email=({_format_email(item.email)}) events=[{events}]"
            )
        case RescheduleRequest():
            return (
                f"- kind={item.kind!r} description={item.description!r} "
                f"email=({_format_email(item.email)}) event=({_format_event(item.event)})"
            )


def _format_action_items(action_items: list[ActionNeeded]) -> str:
    return "\n".join(_format_action_item(item) for item in action_items) or "(none)"


def _build_context(state: PlannerState) -> str:
    now = datetime.now(timezone.utc)
    return (
        f"Current date: {now.strftime('%A')}, {now.date().isoformat()} (UTC).\n\n"
        f"Action items to resolve:\n{_format_action_items(state['action_items'])}"
    )


def _make_bound_tools(gmail_service: Resource) -> list[BaseTool]:
    return [
        tools.propose_event_tool,
        tools.make_draft_reply_tool(gmail_service),
        tools.withdraw_proposal_tool,
    ]


def make_agent_node(gmail_service: Resource) -> Callable[[PlannerState], dict]:
    # Built lazily inside agent() (matching classify_intent/_analyze_email's
    # existing pattern elsewhere in this codebase) rather than eagerly here.
    # ChatAnthropic's constructor validates api_key as a required string --
    # constructing it as soon as build_graph() runs meant simply building
    # the graph required a valid-looking key, even for tests/paths that
    # never reach this node. Live-discovered in CI, which correctly has no
    # ANTHROPIC_API_KEY configured (the test suite needs no live API calls).
    bound_tools = _make_bound_tools(gmail_service)

    def agent(state: PlannerState) -> dict:
        llm = ChatAnthropic(
            model_name=settings.agent_model_name, api_key=settings.anthropic_api_key
        )
        llm_with_tools = llm.bind_tools(bound_tools)

        messages = list(state["messages"])
        seed: list[AnyMessage] = []
        if not any(isinstance(m, SystemMessage) for m in messages):
            seed = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_build_context(state)),
            ]
        response = llm_with_tools.invoke(messages + seed)
        return {"messages": seed + [response]}

    return agent


def make_tools_node(gmail_service: Resource) -> ToolNode:
    return ToolNode(_make_bound_tools(gmail_service))
