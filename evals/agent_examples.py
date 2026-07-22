"""Hand-authored single-step eval examples for `resolution.agent()`.

Each example scripts a `PlannerState` -- either a fresh first-turn
`action_items` list (letting `agent()` build its own system/context messages,
exactly as production does) or a fully scripted `messages` history standing
in for a prior turn the agent already took. Scripting the prior turn by hand
sidesteps having to run the live, non-deterministic agent just to get into a
given conversational state.

`expected` is structured (tool name/args, forbidden substrings), not an
exact-text match, since LLM phrasing varies run to run. No evaluator runs
these yet -- that's a follow-up task. `tests/test_agent_examples.py` only
checks that every example is well-formed and that `agent()` runs on it
without raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_secretary import resolution
from agentic_secretary.state import (
    BackToBackConflict,
    CalendarOverlapConflict,
    EmailConflict,
    PlannerState,
    RescheduleRequest,
)
from agentic_secretary.tools import CalendarEvent, EmailSummary, EventProposal

# `resolution._build_context` stamps in the real `datetime.now(timezone.utc)`
# as "Current date" for every live eval run (unlike the mocked unit tests in
# test_resolution.py, which never actually reason about dates) -- a fixed
# historical NOW eventually looks like "the past" to the real LLM and it
# correctly declines to propose fixing an already-happened conflict. Anchor
# relative to the real now instead, comfortably in the future, so fixtures
# never rot.
NOW = (datetime.now(timezone.utc) + timedelta(days=7)).replace(
    hour=12, minute=0, second=0, microsecond=0
)


def _base_state(**overrides) -> PlannerState:
    state: PlannerState = {
        "messages": [],
        "emails": [],
        "calendar_events": [],
        "action_items": [],
        "status": "done",
    }
    state.update(overrides)
    return state


@dataclass(frozen=True)
class ToolCallExpectation:
    name: str
    # Empty dict means "any args" -- only the tool name is being asserted on.
    args_subset: dict


@dataclass(frozen=True)
class Expected:
    tool_calls_include: tuple[ToolCallExpectation, ...] = ()
    content_must_not_contain: tuple[str, ...] = ()
    rubric: str = ""


@dataclass(frozen=True)
class Example:
    name: str
    state: PlannerState
    expected: Expected


EXAMPLES: list[Example] = []


def _register(example: Example) -> Example:
    EXAMPLES.append(example)
    return example


_STANDUP = CalendarEvent(
    id="e1", title="Standup", start=NOW, end=NOW + timedelta(minutes=30)
)
_CLIENT_CALL = CalendarEvent(
    id="e2",
    title="Client Call",
    start=NOW + timedelta(minutes=15),
    end=NOW + timedelta(minutes=45),
)

# "2pm"/"3pm" relative to NOW, reused below -- the prior proposal moves
# Client Call to _PROPOSED_START; the revision example asks to move it again
# to _REVISED_START.
_PROPOSED_START = NOW + timedelta(hours=2)
_REVISED_START = NOW + timedelta(hours=3)

# A prior-turn proposal, reused by every scripted-continuation example below.
# Anthropic's API requires every tool_use block to be immediately followed by
# its tool_result in the very next message -- mirroring what the real
# agent<->tools loop (resolution.py + graph.py's tools_condition) actually
# produces: a tool-call turn, the tool's result, then a narrated
# turn-ending summary with no further tool calls.
_PROPOSED_EVENT = EventProposal(
    title="Client Call",
    start=_PROPOSED_START,
    duration_minutes=30,
    existing_event_id="e2",
)
_PRIOR_TOOL_CALL = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "propose_event",
            "args": {
                "title": "Client Call",
                "start": _PROPOSED_START.isoformat(),
                "duration_minutes": 30,
                "existing_event_id": "e2",
            },
            "id": "call_1",
        }
    ],
)
_PRIOR_TOOL_RESULT = ToolMessage(content=str(_PROPOSED_EVENT), tool_call_id="call_1")
_PRIOR_SUMMARY = AIMessage(
    content="I've proposed moving Client Call to 2pm to resolve the overlap."
)


def _scripted_continuation(*, human_reply: str) -> list:
    return [
        SystemMessage(content=resolution.SYSTEM_PROMPT),
        HumanMessage(
            content="(scenario context: overlap between Standup and Client Call)"
        ),
        _PRIOR_TOOL_CALL,
        _PRIOR_TOOL_RESULT,
        _PRIOR_SUMMARY,
        HumanMessage(content=human_reply),
    ]


# ---------------------------------------------------------------------------
# 1. First-turn, single calendar overlap -> proposes a resolution.
# ---------------------------------------------------------------------------
_register(
    Example(
        name="first_turn_single_overlap",
        state=_base_state(
            action_items=[
                CalendarOverlapConflict(
                    description="'Standup' overlaps with 'Client Call'",
                    events=(_STANDUP, _CLIENT_CALL),
                ),
            ]
        ),
        expected=Expected(
            tool_calls_include=(
                # Moving *either* overlapping event resolves the conflict --
                # the agent has legitimate discretion over which one, so
                # accept both rather than assuming it always picks e2.
                ToolCallExpectation(
                    name="propose_event",
                    args_subset={"existing_event_id": {"e1", "e2"}},
                ),
            ),
            rubric="Should propose moving one of the two overlapping events to a "
            "non-conflicting time.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 2. First-turn, email reschedule request that commits to a specific time ->
#    must draft a reply *and* call propose_event for that time.
# ---------------------------------------------------------------------------
_CLIENT_SYNC = CalendarEvent(
    id="e3",
    title="Client Sync",
    start=NOW + timedelta(days=1),
    end=NOW + timedelta(days=1, minutes=30),
)
_RESCHEDULE_EMAIL = EmailSummary(
    id="m1",
    thread_id="t1",
    from_="priya.patel@example.com",
    to="you@example.com",
    subject="Re: Client Sync -- need to move",
    body="Can we push our client sync from tomorrow to Thursday at 2pm instead?",
    received_at=NOW,
)

_register(
    Example(
        name="first_turn_email_reschedule_commits_time",
        state=_base_state(
            action_items=[
                RescheduleRequest(
                    description="asks to reschedule 'Client Sync'",
                    email=_RESCHEDULE_EMAIL,
                    event=_CLIENT_SYNC,
                ),
            ]
        ),
        expected=Expected(
            tool_calls_include=(
                ToolCallExpectation(
                    name="draft_reply_tool", args_subset={"thread_id": "t1"}
                ),
                ToolCallExpectation(
                    name="propose_event", args_subset={"existing_event_id": "e3"}
                ),
            ),
            rubric="A reply committing to Thursday 2pm must also call propose_event for "
            "that time (resolution.py:37-41) -- otherwise the time only exists inside "
            "the drafted email's text, invisible to the collision check.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 3. Revision after a prior proposal: "actually make it 3pm instead".
# ---------------------------------------------------------------------------
_register(
    Example(
        name="revise_plan_to_3pm",
        state=_base_state(
            messages=_scripted_continuation(human_reply="actually make it 3pm instead")
        ),
        expected=Expected(
            tool_calls_include=(
                ToolCallExpectation(
                    name="propose_event",
                    args_subset={
                        "start": _REVISED_START.isoformat(),
                        "existing_event_id": "e2",
                    },
                ),
            ),
            rubric="Must amend the existing Client Call proposal to 3pm, treating it as "
            "the specific thing being corrected -- not an unrelated new request.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 4. Human just acknowledges -> must not claim anything was booked/sent.
# ---------------------------------------------------------------------------
_register(
    Example(
        name="acknowledge_only_no_overclaim",
        state=_base_state(
            messages=_scripted_continuation(human_reply="looks good, thanks")
        ),
        expected=Expected(
            content_must_not_contain=("booked", "sent", "confirmed", "scheduled it"),
            rubric="Must acknowledge warmly without implying anything beyond the prior "
            "tool call already happened (resolution.py:58-63).",
        ),
    )
)

# ---------------------------------------------------------------------------
# 5. Agent abandons a previously-proposed time -> should call
#    withdraw_proposal rather than leave the stale proposal live.
# ---------------------------------------------------------------------------
_register(
    Example(
        name="withdraw_abandoned_proposal",
        state=_base_state(
            messages=_scripted_continuation(
                human_reply="actually never mind, don't move it -- just tell me what "
                "other times would work instead"
            )
        ),
        expected=Expected(
            tool_calls_include=(
                ToolCallExpectation(name="withdraw_proposal", args_subset={}),
            ),
            rubric="Declining a previously-proposed time in favor of asking for "
            "alternatives should withdraw the stale propose_event call "
            "(resolution.py:43-47), not leave it live for the collision check.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 6. Multiple action items in one first turn -> must work through all of them,
#    not just the first.
# ---------------------------------------------------------------------------
_MANAGER_1_1 = CalendarEvent(
    id="e4",
    title="1:1 with Manager",
    start=NOW + timedelta(hours=3),
    end=NOW + timedelta(hours=3, minutes=30),
)
_DESIGN_REVIEW = CalendarEvent(
    id="e5",
    title="Design Review",
    start=NOW + timedelta(hours=3, minutes=30),
    end=NOW + timedelta(hours=4),
)

_register(
    Example(
        name="first_turn_multiple_action_items",
        state=_base_state(
            action_items=[
                CalendarOverlapConflict(
                    description="'Standup' overlaps with 'Client Call'",
                    events=(_STANDUP, _CLIENT_CALL),
                ),
                BackToBackConflict(
                    description="'1:1 with Manager' ends right as 'Design Review' "
                    "starts, no buffer",
                    events=(_MANAGER_1_1, _DESIGN_REVIEW),
                ),
            ]
        ),
        expected=Expected(
            # No tool_calls_include here -- resolution.py's SYSTEM_PROMPT
            # (49-52) explicitly permits the agent to skip an item or ask a
            # question instead of forcing a resolution, and live runs
            # confirmed real variance in whether it acts here at all
            # (temperature isn't pinned to 0 for this node, unlike
            # classify_intent/_analyze_email). Whether the response
            # reasonably addresses both items is exactly the kind of
            # judgment call the LLM-judge rubric is for, not a fixed
            # assertion.
            rubric="Must address both action items in the same turn (resolution.py:49-50, "
            "'work through every action item') -- for each one, either a tool call "
            "resolving it or an explicit note explaining why it was skipped should "
            "appear in the final summary. Silently ignoring an item is the failure "
            "mode to catch, not any particular choice of tool call vs. explanation.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 7. Ambiguous email conflict, no concrete time proposed -> either asking or
#    proposing a reasonable candidate time is acceptable; only an
#    unreasonable suggestion or silently ignoring the item is a failure.
#    (Originally asserted the agent must always ask rather than propose --
#    live-judged runs showed it proposing sensible candidate times instead,
#    and on reflection that's acceptable, helpful behavior, not a bug.)
# ---------------------------------------------------------------------------
_VAGUE_EMAIL = EmailSummary(
    id="m2",
    thread_id="t2",
    from_="alex@example.com",
    to="you@example.com",
    subject="Quick chat?",
    body="Hey, could we grab some time this week? No particular time in mind, just "
    "whenever works for you.",
    received_at=NOW,
)

_register(
    Example(
        name="ambiguous_email_conflict_reasonable_response",
        state=_base_state(
            action_items=[
                EmailConflict(
                    description="email from Alex conflicts with 'Client Call'",
                    email=_VAGUE_EMAIL,
                    events=[_CLIENT_CALL],
                ),
            ]
        ),
        expected=Expected(
            rubric="The email gives no concrete proposed time. Either asking a "
            "clarifying question or proposing a reasonable candidate time (e.g. one "
            "that doesn't collide with the existing 'Client Call' event) is "
            "acceptable, helpful behavior. The failure mode to catch is an "
            "unreasonable suggestion (a time that conflicts with an existing event, "
            "or one unrelated to the request) or silently ignoring the item -- not "
            "simply choosing to propose rather than ask.",
        ),
    )
)

# ---------------------------------------------------------------------------
# 8. Weekday names must never appear anywhere in the response, even though
#    the conflict genuinely falls on a Friday. Computed relative to NOW
#    (rather than a hardcoded date) so this always lands on a real, future
#    Friday regardless of when the suite runs.
# ---------------------------------------------------------------------------
_days_until_friday = (4 - NOW.weekday()) % 7 or 7
_FRIDAY = NOW + timedelta(days=_days_until_friday)

_QUARTERLY_REVIEW = CalendarEvent(
    id="e6",
    title="Quarterly Review",
    start=_FRIDAY.replace(hour=10, minute=0),
    end=_FRIDAY.replace(hour=11, minute=0),
)
_VENDOR_CALL = CalendarEvent(
    id="e7",
    title="Vendor Call",
    start=_FRIDAY.replace(hour=10, minute=30),
    end=_FRIDAY.replace(hour=11, minute=30),
)

_register(
    Example(
        name="no_weekday_names_in_response",
        state=_base_state(
            action_items=[
                CalendarOverlapConflict(
                    description="'Quarterly Review' overlaps with 'Vendor Call'",
                    events=(_QUARTERLY_REVIEW, _VENDOR_CALL),
                ),
            ]
        ),
        expected=Expected(
            content_must_not_contain=(
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ),
            rubric="_FRIDAY is a genuine Friday -- per resolution.py:69-73 (a "
            "live-discovered, repeatedly-recurring bug), the response must state the "
            "date numerically and never name the weekday, anywhere in its text.",
        ),
    )
)
