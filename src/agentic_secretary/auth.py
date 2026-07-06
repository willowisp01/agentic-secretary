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


def get_credentials() -> Credentials:
    """Return valid Google API credentials, prompting for consent on first use.

    Reads/writes the cached token at `settings.google_token_path`. Runs the
    installed-app (Authorization Code + PKCE) consent flow via a local
    loopback server only when no valid cached token exists.
    """
    token_path = Path(settings.google_token_path)
    creds: Credentials | None = None

    if token_path.exists():
        # Case 3: cached token is still valid — reuse it, no server contact at all.
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Case 2: access token expired — use the refresh token grant, no browser needed.
            creds.refresh(Request())
        else:
            # Case 1: no usable cached token — run the full Authorization Code + PKCE consent flow.
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.google_client_secret_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return creds
