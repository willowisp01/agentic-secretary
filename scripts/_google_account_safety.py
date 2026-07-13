"""Shared safety check for scripts/*.py that write or delete real data in the
burner Google account (seed_demo_data.py, nuke_seed_data.py). Neither script
owns this — each resolves credentials from its own separate token cache (see
their respective SCOPES/token-path constants), so this is the one check
standing between a stale/wrong cached token and acting on the wrong inbox.
"""

from googleapiclient.discovery import Resource


def confirm_target_account(gmail_service: Resource, action: str) -> str:
    """Print the authenticated account and require explicit confirmation
    before `action` proceeds against it. `action` has no default so each
    caller must describe what it's actually about to do (e.g. "seed demo
    data", "trash and delete all demo data") rather than risk showing the
    wrong script's wording.
    """
    profile = gmail_service.users().getProfile(userId="me").execute()
    email_address = profile["emailAddress"]
    response = input(f"About to {action} into {email_address}. Continue? [y/N] ")
    if response.strip().lower() not in ("y", "yes"):
        raise SystemExit("Aborted: target account not confirmed.")
    return email_address
