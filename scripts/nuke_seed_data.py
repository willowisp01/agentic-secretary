"""Reverse scripts/seed_demo_data.py: clear the burner Gmail/Calendar account.

Trashes every message in the mailbox and deletes every event on the primary
calendar. Messages are moved to Trash (recoverable for ~30 days), never
permanently deleted — permanent delete needs the far broader
`https://mail.google.com/` scope, which this script deliberately avoids
requesting.

Uses its own scope list and token cache (settings.google_cleanup_token_path),
separate from both the runtime grant (auth.py) and the seed grant
(seed_demo_data.py): gmail.modify can touch *any* message in the mailbox, not
just ones this project inserted, so it's the most dangerous grant of the
three and shouldn't be reachable via either other cached token.
"""

from googleapiclient.discovery import Resource, build

from _google_account_safety import confirm_target_account
from agentic_secretary.auth import load_credentials
from agentic_secretary.config import settings

CLEANUP_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
]


def list_all_message_ids(service: Resource) -> list[str]:
    ids: list[str] = []
    request = service.users().messages().list(userId="me")
    response = request.execute()
    ids.extend(m["id"] for m in response.get("messages", []))
    next_request = service.users().messages().list_next(request, response)
    while next_request is not None:
        response = next_request.execute()
        ids.extend(m["id"] for m in response.get("messages", []))
        next_request = service.users().messages().list_next(next_request, response)
    return ids


def list_all_event_ids(service: Resource) -> list[str]:
    ids: list[str] = []
    request = service.events().list(calendarId="primary")
    response = request.execute()
    ids.extend(e["id"] for e in response.get("items", []))
    next_request = service.events().list_next(request, response)
    while next_request is not None:
        response = next_request.execute()
        ids.extend(e["id"] for e in response.get("items", []))
        next_request = service.events().list_next(next_request, response)
    return ids


def _confirm_counts(message_count: int, event_count: int) -> None:
    """Require explicit confirmation of the blast radius before deleting anything."""
    response = input(
        f"About to trash {message_count} message(s) and delete {event_count} "
        f"event(s). Continue? [y/N] "
    )
    if response.strip().lower() not in ("y", "yes"):
        raise SystemExit("Aborted: nuke not confirmed.")


def nuke(gmail_service: Resource, calendar_service: Resource) -> None:
    """Trash every message and delete every event in the given services."""
    message_ids = list_all_message_ids(gmail_service)
    print(f"Trashing {len(message_ids)} messages...")
    for message_id in message_ids:
        gmail_service.users().messages().trash(userId="me", id=message_id).execute()
        print(f"  trashed {message_id}")

    event_ids = list_all_event_ids(calendar_service)
    print(f"Deleting {len(event_ids)} events...")
    for event_id in event_ids:
        calendar_service.events().delete(
            calendarId="primary", eventId=event_id
        ).execute()
        print(f"  deleted {event_id}")
    print("Done.")


def main() -> None:
    creds = load_credentials(
        CLEANUP_SCOPES,
        settings.google_cleanup_token_path,
        settings.google_client_secret_path,
    )
    gmail_service = build("gmail", "v1", credentials=creds)
    calendar_service = build("calendar", "v3", credentials=creds)
    confirm_target_account(gmail_service, action="trash and delete all demo data")

    message_ids = list_all_message_ids(gmail_service)
    event_ids = list_all_event_ids(calendar_service)
    _confirm_counts(len(message_ids), len(event_ids))

    nuke(gmail_service, calendar_service)


if __name__ == "__main__":
    main()
