from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from agentic_secretary.config import settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.readonly",
]


def load_credentials(
    scopes: list[str], token_path: str, client_secret_path: str
) -> Credentials:
    """Return valid Google API credentials for `scopes`, prompting for consent
    on first use.

    Reads/writes the cached token at `token_path`. Runs the installed-app
    (Authorization Code + PKCE) consent flow via a local loopback server only
    when no valid cached token exists. Generic over scopes/token path so
    callers that need a different grant (e.g. seeding) don't have to share
    the runtime token cache — see `get_credentials` below and
    `scripts/seed_demo_data.py`.
    """
    path = Path(token_path)
    creds: Credentials | None = None

    if path.exists():
        # Case 3: cached token is still valid — reuse it, no server contact at all.
        creds = Credentials.from_authorized_user_file(str(path), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Case 2: access token expired — use the refresh token grant, no browser needed.
            creds.refresh(Request())
        else:
            # Case 1: no usable cached token — run the full Authorization Code + PKCE consent flow.
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
            creds = flow.run_local_server(port=0)
        path.write_text(creds.to_json())

    return creds


def get_credentials() -> Credentials:
    """Return valid Google API credentials for the agent's runtime scopes.

    Reads/writes the cached token at `settings.google_token_path`.
    """
    return load_credentials(
        SCOPES, settings.google_token_path, settings.google_client_secret_path
    )
