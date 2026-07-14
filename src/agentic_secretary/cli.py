from googleapiclient.discovery import build
from langgraph.types import Command

from agentic_secretary.auth import get_credentials
from agentic_secretary.config import settings
from agentic_secretary.graph import build_graph


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

    # Nodes that pause for chat input (classify_intent, present_menu, ...)
    # call interrupt(), which halts mid-graph and returns here immediately
    # rather than running to completion -- resume with the human's reply
    # until the graph actually reaches END with no interrupt pending.
    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value
        decision = input(f"{prompt}> ")
        result = graph.invoke(Command(resume=decision), config=config)

    print(f"Fetched {len(result['emails'])} emails:")
    for email in result["emails"]:
        print(f"  - [{email.received_at}] {email.subject} (from {email.from_})")

    print(f"\nFetched {len(result['calendar_events'])} calendar events:")
    for event in result["calendar_events"]:
        print(f"  - [{event.start} - {event.end}] {event.title}")

    print(f"\nDetected {len(result['action_items'])} action items:")
    for item in result["action_items"]:
        print(f"  - [{item.kind}] {item.description}")


if __name__ == "__main__":
    main()
