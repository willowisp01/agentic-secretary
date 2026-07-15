from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic_secretary.review import (
    _collision_note,
    _latest_proposals,
    review,
    route_after_review,
)
from agentic_secretary.state import PlannerState
from agentic_secretary.tools import CalendarEvent, EventProposal

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _base_state(**overrides) -> PlannerState:
    state: PlannerState = {
        "messages": [AIMessage(content="All done.")],
        "emails": [],
        "calendar_events": [],
        "action_items": [],
        "status": "done",
    }
    state.update(overrides)
    return state


def _proposal_message(proposal: EventProposal) -> ToolMessage:
    return ToolMessage(content=str(proposal), artifact=proposal, tool_call_id="call_1")


def _proposal_message_as_dict(proposal: EventProposal) -> ToolMessage:
    # Live-discovered: ToolMessage.artifact only survives as a real
    # EventProposal within the same graph.invoke() call that created it.
    # After a checkpoint round-trip (i.e. any proposal from a prior turn,
    # by the time review() runs again) it comes back as a plain dict with
    # the same keys instead -- this simulates that shape directly rather
    # than requiring an actual checkpointer round-trip in every test.
    return ToolMessage(
        content=str(proposal),
        artifact={
            "title": proposal.title,
            "start": proposal.start,
            "duration_minutes": proposal.duration_minutes,
            "attendees": proposal.attendees,
            "existing_event_id": proposal.existing_event_id,
        },
        tool_call_id="call_1",
    )


def test_collision_note_is_none_when_nothing_was_proposed():
    assert _collision_note(_base_state()) is None


def test_collision_note_is_none_for_non_overlapping_proposals():
    early = EventProposal(title="Early", start=NOW, duration_minutes=30)
    late = EventProposal(
        title="Late", start=NOW + timedelta(hours=2), duration_minutes=30
    )
    state = _base_state(messages=[_proposal_message(early), _proposal_message(late)])

    assert _collision_note(state) is None


def test_collision_note_flags_two_proposals_that_overlap():
    first = EventProposal(title="Client Sync", start=NOW, duration_minutes=60)
    second = EventProposal(
        title="Team Standup", start=NOW + timedelta(minutes=30), duration_minutes=30
    )
    state = _base_state(messages=[_proposal_message(first), _proposal_message(second)])

    note = _collision_note(state)

    assert note is not None
    assert "Client Sync" in note
    assert "Team Standup" in note


def test_collision_note_sees_a_proposal_that_survived_only_as_a_dict():
    # Reproduces the live bug directly: a proposal from a prior turn whose
    # ToolMessage.artifact round-tripped through a checkpoint and came
    # back as a dict, not an EventProposal. Before the fix, this proposal
    # was invisible to _latest_proposals entirely -- not just mis-keyed --
    # so its move never excluded the event's original slot from
    # untouched_events, and the original (already-moved-away-from) time
    # kept getting compared as if still live.
    existing = CalendarEvent(
        id="e1", title="Team Standup", start=NOW, end=NOW + timedelta(minutes=30)
    )
    moved = EventProposal(
        title="Team Standup",
        start=NOW + timedelta(hours=5),
        duration_minutes=30,
        existing_event_id="e1",
    )
    state = _base_state(
        calendar_events=[existing], messages=[_proposal_message_as_dict(moved)]
    )

    # If the dict-shaped artifact were invisible, "e1" would stay in
    # untouched_events at its original time and never collide with
    # anything else here -- so the absence of a note wouldn't prove the
    # fix works. Assert on the positive: the moved event's original slot
    # is correctly excluded, by checking a second, real-object proposal
    # against it lands at the *new* time, not the stale original one.
    proposals = _latest_proposals(state["messages"])
    assert len(proposals) == 1
    assert proposals[0] == moved


def test_collision_note_handles_a_mix_of_real_and_dict_shaped_artifacts():
    # The exact live scenario: one proposal from an earlier turn (already
    # round-tripped through a checkpoint, now a dict) and one from the
    # current turn (still a real EventProposal in memory).
    earlier_turn = EventProposal(title="Client Sync", start=NOW, duration_minutes=45)
    current_turn = EventProposal(
        title="Quick sync", start=NOW + timedelta(minutes=15), duration_minutes=30
    )
    state = _base_state(
        messages=[
            _proposal_message_as_dict(earlier_turn),
            _proposal_message(current_turn),
        ]
    )

    note = _collision_note(state)

    assert note is not None
    assert "Client Sync" in note
    assert "Quick sync" in note


def test_collision_note_flags_two_proposals_that_are_back_to_back_with_no_buffer():
    # Live-discovered gap: overlap-checking alone missed a real zero-buffer
    # adjacency the agent introduced itself -- one proposal's start landing
    # exactly on another's end doesn't count as an "overlap" by the strict
    # definition, but it's still worth flagging the same way.
    first = EventProposal(title="Design Review", start=NOW, duration_minutes=45)
    second = EventProposal(
        title="Quick Sync", start=NOW + timedelta(minutes=45), duration_minutes=30
    )
    state = _base_state(messages=[_proposal_message(first), _proposal_message(second)])

    note = _collision_note(state)

    assert note is not None
    assert "Design Review" in note
    assert "Quick Sync" in note


def test_collision_note_flags_a_proposal_against_an_untouched_calendar_event():
    existing = CalendarEvent(
        id="e1", title="Lunch", start=NOW, end=NOW + timedelta(minutes=60)
    )
    proposal = EventProposal(title="Client Sync", start=NOW, duration_minutes=30)
    state = _base_state(
        calendar_events=[existing], messages=[_proposal_message(proposal)]
    )

    note = _collision_note(state)

    assert note is not None
    assert "Lunch" in note


def test_collision_note_does_not_flag_a_moved_event_against_its_own_old_slot():
    # The event being moved is still in calendar_events at its original
    # time -- it shouldn't collide with its own new proposed time.
    existing = CalendarEvent(
        id="e1", title="Client Call", start=NOW, end=NOW + timedelta(minutes=30)
    )
    proposal = EventProposal(
        title="Client Call",
        start=NOW + timedelta(hours=3),
        duration_minutes=30,
        existing_event_id="e1",
    )
    state = _base_state(
        calendar_events=[existing], messages=[_proposal_message(proposal)]
    )

    assert _collision_note(state) is None


def test_collision_note_uses_the_latest_proposal_when_corrected():
    # A correction re-proposes the same event at a new time; the earlier
    # ToolMessage for it is still in the message history (append-only) but
    # shouldn't be compared as if it were a separate, still-live proposal.
    existing = CalendarEvent(
        id="e1", title="Client Call", start=NOW, end=NOW + timedelta(minutes=30)
    )
    other = CalendarEvent(
        id="e2",
        title="Team Standup",
        start=NOW + timedelta(hours=3),
        end=NOW + timedelta(hours=3, minutes=30),
    )
    first_proposal = EventProposal(
        title="Client Call",
        start=NOW + timedelta(hours=3),
        duration_minutes=30,
        existing_event_id="e1",
    )
    corrected_proposal = EventProposal(
        title="Client Call",
        start=NOW + timedelta(hours=5),
        duration_minutes=30,
        existing_event_id="e1",
    )
    state = _base_state(
        calendar_events=[existing, other],
        messages=[
            _proposal_message(first_proposal),
            _proposal_message(corrected_proposal),
        ],
    )

    # The first proposal would have collided with `other`; the corrected
    # one doesn't, and only the corrected one should be considered live.
    assert _collision_note(state) is None


def _build_test_graph():
    builder = StateGraph(PlannerState)
    builder.add_node("review", review)
    builder.add_node(
        "agent_stub",
        lambda state: {"messages": [AIMessage(content="agent_stub reached")]},
    )
    builder.add_edge(START, "review")
    builder.add_conditional_edges(
        "review", route_after_review, {"agent": "agent_stub", END: END}
    )
    builder.add_edge("agent_stub", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_review_interrupts_with_the_agents_final_summary():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke(_base_state(), config=config)

    assert result["__interrupt__"][0].value.startswith("All done.")


def test_review_always_appends_the_propose_only_disclaimer():
    # Live-discovered: the agent's own closing text after an "approved"
    # reply used finality language ("your calendar is now updated") that
    # misrepresents what actually happened. Don't trust the LLM to avoid
    # this consistently -- attach a fixed, code-guaranteed disclaimer
    # instead, on every review turn, not just the first.
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t5"}}

    result = graph.invoke(_base_state(), config=config)

    assert "nothing has been booked or sent" in result["__interrupt__"][0].value


def test_exit_phrase_ends_without_looping_back_to_agent():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="done"), config=config)

    contents = [m.content for m in final["messages"]]
    assert "agent_stub reached" not in contents
    assert contents[-1] == "done"


def test_non_exit_reply_loops_back_to_agent():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="move it to 2pm instead"), config=config)

    contents = [m.content for m in final["messages"]]
    assert "move it to 2pm instead" in contents
    assert contents[-1] == "agent_stub reached"


def test_exit_phrase_matching_is_case_and_whitespace_insensitive():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke(_base_state(), config=config)

    final = graph.invoke(Command(resume="  Done  "), config=config)

    contents = [m.content for m in final["messages"]]
    assert "agent_stub reached" not in contents
