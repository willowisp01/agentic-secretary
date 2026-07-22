from datetime import timedelta

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import interrupt

from agentic_secretary.detection import (
    _failed_email_note,
    _find_back_to_back,
    _find_calendar_overlaps,
)
from agentic_secretary.state import PlannerState
from agentic_secretary.tools import CalendarEvent, EventProposal

# Deterministic, not LLM-classified: the human is confirming they're done,
# not asking the agent to do anything, so this doesn't need judgment.
_EXIT_PHRASES = {"done", "no", "nothing else", "that's all", "bye"}

# Live-discovered: the agent's own closing text after a human "approved"/
# "ok" reply used finality language ("your calendar is now updated",
# "drafts are ready to send") that misrepresents what actually happened --
# propose_event/draft_reply never call insert/patch/send, so nothing was
# actually booked or sent. Don't trust the LLM to remember this
# consistently on every turn (the system prompt already says it once and
# that wasn't enough); attach a fixed, code-guaranteed disclaimer instead,
# the same "don't leave a safety-relevant fact to LLM judgment" pattern as
# the collision note above.
_DISCLAIMER = (
    "(Reminder: these are proposals only -- nothing has been booked or "
    "sent. Apply them yourself in Calendar/Gmail if you'd like to proceed.)"
)


def _as_event_proposal(artifact: object) -> EventProposal | None:
    # Live-discovered: ToolMessage.artifact survives as a real EventProposal
    # only within the same graph.invoke() call that created it. The moment
    # the graph resumes from an interrupt, it's been through a checkpoint
    # round-trip -- and .artifact is a nested Any-typed field inside a
    # LangChain "core" message object, serialized through LangChain's own
    # message protocol rather than the top-level PlannerState msgpack path
    # our allowlist covers. It comes back as a plain dict with the same
    # keys (datetime values survive intact; the dataclass type doesn't).
    # Every proposal from a prior turn would otherwise silently vanish from
    # collision-checking.
    if isinstance(artifact, EventProposal):
        return artifact
    if isinstance(artifact, dict):
        try:
            return EventProposal(**artifact)
        except TypeError:
            return None
    return None


def _latest_proposals(messages: list[AnyMessage]) -> list[EventProposal]:
    # Keyed by existing_event_id (or title, for brand-new events) so a
    # correction's later proposal for the same target supersedes its
    # earlier one instead of both being compared as if they were distinct
    # events -- messages accumulate across turns, they aren't replaced.
    #
    # Live-discovered: the agent can also decide *against* a proposal it
    # already made (declining a time in favor of asking for alternatives)
    # without ever calling propose_event again -- with no signal for that,
    # the abandoned proposal stayed "the latest" forever, and the collision
    # note kept flagging conflicts the conversation had already moved past.
    # A withdraw_proposal call clears its key instead of replacing it;
    # processing messages in order means a later re-proposal for the same
    # key still correctly wins back over an earlier withdrawal.
    latest: dict[str, EventProposal] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name == "withdraw_proposal":
            latest.pop(message.content, None)
            continue
        proposal = _as_event_proposal(message.artifact)
        if proposal is None:
            continue
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
    gate. Reuses the existing overlap/back-to-back math
    (_find_calendar_overlaps, _find_back_to_back) against the proposed
    times plus whichever original calendar events weren't themselves just
    moved, rather than reimplementing either check. Live-discovered gap:
    checking overlaps alone missed a real zero-buffer back-to-back the
    agent introduced itself (one proposal's start landing exactly on
    another event's end) -- adjacent times don't overlap by the strict
    definition, so they need their own check.
    """
    proposals = _latest_proposals(state["messages"])
    if not proposals:
        return None

    proposed_events = [_as_calendar_event(p) for p in proposals]
    moved_ids = {e.id for e in proposed_events}
    untouched_events = [e for e in state["calendar_events"] if e.id not in moved_ids]

    combined = proposed_events + untouched_events
    collisions = _find_calendar_overlaps(combined) + _find_back_to_back(combined)
    if not collisions:
        return None
    return "Note: " + "; ".join(c.description for c in collisions)


def review(state: PlannerState) -> dict:
    # The agent's own closing text -- whatever it wrote once it stopped
    # calling tools for this turn. This is LLM-generated prose, not
    # anything this node computes.
    summary = state["messages"][-1].content

    # The deterministic check's verdict: None if nothing collides, or a
    # "Note: ..." string. Computed fresh every call, independent of
    # whatever the agent's own summary claims.
    note = _collision_note(state)
    failed_note = _failed_email_note(state.get("failed_emails", []))

    # What the human actually sees: the agent's prose, then any notes
    # (collision, failed-email) underneath it, then the fixed disclaimer
    # always last.
    display = summary
    for extra in (note, failed_note):
        if extra is not None:
            display = f"{display}\n\n{extra}"
    display = f"{display}\n\n{_DISCLAIMER}"

    # Pause the graph here and show `display`. Execution resumes when the
    # CLI calls Command(resume=<human's reply>); `reply` is that string.
    reply = interrupt(display)
    return {"messages": [HumanMessage(content=reply)]}


def route_after_review(state: PlannerState) -> str:
    # Runs after review()'s interrupt resumes, deciding where to go next:
    # a recognized exit phrase ends the graph, anything else loops back to
    # agent (which sees the reply as just another turn in the conversation).
    reply = state["messages"][-1].content
    if reply.strip().lower() in _EXIT_PHRASES:
        return END
    return "agent"
