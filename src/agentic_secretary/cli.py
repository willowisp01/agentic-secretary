from googleapiclient.discovery import build

from agentic_secretary.auth import get_credentials
from agentic_secretary.config import settings
from agentic_secretary.graph import build_graph


def main() -> None:
    if not settings.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set (check your .env) -- required for "
            "conflict detection."
        )

    credentials = get_credentials()
    gmail_service = build("gmail", "v1", credentials=credentials)
    calendar_service = build("calendar", "v3", credentials=credentials)
    graph = build_graph(gmail_service, calendar_service)

    result = graph.invoke(
        {"emails": [], "calendar_events": [], "status": "pending"},
        config={"configurable": {"thread_id": "cli"}},
    )

    print(f"Fetched {len(result['emails'])} emails:")
    for email in result["emails"]:
        print(f"  - [{email.received_at}] {email.subject} (from {email.from_})")

    print(f"\nFetched {len(result['calendar_events'])} calendar events:")
    for event in result["calendar_events"]:
        print(f"  - [{event.start} - {event.end}] {event.title}")

    print(f"\nDetected {len(result['conflicts'])} conflicts:")
    for conflict in result["conflicts"]:
        print(f"  - [{conflict['kind']}] {conflict['description']}")


if __name__ == "__main__":
    main()
