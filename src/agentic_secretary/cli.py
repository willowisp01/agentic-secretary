from datetime import timedelta

from googleapiclient.discovery import build
from langgraph.types import Command

from agentic_secretary.auth import get_credentials
from agentic_secretary.config import settings
from agentic_secretary.graph import build_graph
from agentic_secretary.tools import DraftResult, EventProposal


def main() -> None:
    if not settings.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set (check your .env) -- required for "
            "action detection."
        )

    credentials = get_credentials()
    gmail_service = build("gmail", "v1", credentials=credentials)
    calendar_service = build("calendar", "v3", credentials=credentials)
    graph = build_graph(gmail_service, calendar_service)
    config = {"configurable": {"thread_id": "cli"}}

    result = graph.invoke(
        {"emails": [], "calendar_events": [], "action_items": [], "status": "pending"},
        config=config,
    )

    # Nodes that pause for chat input (classify_intent, present_item,
    # confirm_plan, ...) call interrupt(), which halts mid-graph and
    # returns here immediately rather than running to completion -- resume
    # with the human's reply until the graph actually reaches END with no
    # interrupt pending. A blank line before each prompt and a dedicated
    # line for the "> " cursor keep multi-line prompts (item + remedies,
    # plan summaries) visually separated from the previous turn's answer,
    # rather than running together on one line.
    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value
        decision = input(f"\n{prompt}\n> ")
        result = graph.invoke(Command(resume=decision), config=config)

    divider = "-" * 60
    print(f"\n{divider}\nSummary\n{divider}")

    print(f"\nFetched {len(result['emails'])} emails:")
    for email in result["emails"]:
        print(f"  - [{email.received_at}] {email.subject} (from {email.from_})")

    print(f"\nFetched {len(result['calendar_events'])} calendar events:")
    for event in result["calendar_events"]:
        print(f"  - [{event.start} - {event.end}] {event.title}")

    print(f"\nDetected {len(result['action_items'])} action items:")
    for item in result["action_items"]:
        print(f"  - [{item.kind}] {item.description}")

    # A shift-slot proposal has no persisted artifact anywhere (unlike a
    # Gmail draft, which the human can revisit later) -- this transcript is
    # the only place it's ever visible, so show the actual proposed time.
    # Each resolution gets a leading blank line since there can be more
    # resolutions than action items (multi-remedy plans), which otherwise
    # run together with no visual break between them.
    print(f"\nResolved {len(result['resolutions'])} action items:")
    for resolution in result["resolutions"]:
        print(f"\n  - [{resolution.action_item.kind}] {resolution.action_item.description}")
        print(f"    Remedy: {resolution.remedy}")
        if isinstance(resolution.proposal, EventProposal):
            proposed_end = resolution.proposal.start + timedelta(
                minutes=resolution.proposal.duration_minutes
            )
            # accept_meeting proposes a brand-new calendar entry
            # (existing_event_id is None); shift_slot moves one that's
            # already there -- worth distinguishing in the transcript.
            verb = "move" if resolution.proposal.existing_event_id else "propose new event"
            print(
                f"    Proposed: {verb} {resolution.proposal.title!r} at "
                f"{resolution.proposal.start} - {proposed_end}"
            )
        elif isinstance(resolution.proposal, DraftResult):
            print(f"    Draft created (id={resolution.proposal.draft_id})")


if __name__ == "__main__":
    main()
