from datetime import timedelta

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import interrupt

from agentic_secretary.detection import _find_calendar_overlaps
from agentic_secretary.state import PlannerState
from agentic_secretary.tools import CalendarEvent, EventProposal

# Deterministic, not LLM-classified: the human is confirming they're done,
# not asking the agent to do anything, so this doesn't need judgment.
_EXIT_PHRASES = {"done", "no", "nothing else", "that's all", "bye"}


def _latest_proposals(messages: list[AnyMessage]) -> list[EventProposal]:
    # Keyed by existing_event_id (or title, for brand-new events) so a
    # correction's later proposal for the same target supersedes its
    # earlier one instead of both being compared as if they were distinct
    # events -- messages accumulate across turns, they aren't replaced.
    latest: dict[str, EventProposal] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(
            message.artifact, EventProposal
        ):
            proposal = message.artifact
            key = proposal.existing_event_id or proposal.title
            latest[key] = proposal
    return list(latest.values())


def _as_calendar_event(proposal: EventProposal) -> CalendarEvent:
    return CalendarEvent(
        id=proposal.existing_event_id or f"proposed:{proposal.title}",
        title=proposal.title,
        start=proposal.start,
        end=proposal.start + timedelta(minutes=proposal.duration_minutes),
    )


def _collision_note(state: PlannerState) -> str | None:
    """Advisory-only, computed after the tool calls already ran -- not a
    gate. Reuses the existing overlap math (_find_calendar_overlaps)
    against the proposed times plus whichever original calendar events
    weren't themselves just moved, rather than reimplementing "overlap".
    """
    proposals = _latest_proposals(state["messages"])
    if not proposals:
        return None

    proposed_events = [_as_calendar_event(p) for p in proposals]
    moved_ids = {e.id for e in proposed_events}
    untouched_events = [e for e in state["calendar_events"] if e.id not in moved_ids]

    overlaps = _find_calendar_overlaps(proposed_events + untouched_events)
    if not overlaps:
        return None
    return "Note: " + "; ".join(o.description for o in overlaps)


def review(state: PlannerState) -> dict:
    summary = state["messages"][-1].content
    note = _collision_note(state)
    display = summary if note is None else f"{summary}\n\n{note}"
    reply = interrupt(display)
    return {"messages": [HumanMessage(content=reply)]}


def route_after_review(state: PlannerState) -> str:
    reply = state["messages"][-1].content
    if reply.strip().lower() in _EXIT_PHRASES:
        return END
    return "agent"
