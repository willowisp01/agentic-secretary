# agentic-secretary

## Google OAuth setup

1. In Google Cloud Console, create a project, enable the Gmail API and
   Google Calendar API, configure the OAuth consent screen (External,
   Testing status), add your burner account as a test user, and create an
   OAuth client of type **Desktop app**.
2. Download the client secret JSON and save it as `credentials.json` in the
   project root (path configurable via `GOOGLE_CLIENT_SECRET_PATH` in `.env`).
3. Run `uv run python -c "from agentic_secretary.auth import get_credentials; get_credentials()"`.
   On first run this opens a browser consent screen against the burner
   account; approve it. A `token.json` is cached afterward (path
   configurable via `GOOGLE_TOKEN_PATH`).
4. Run the same command again — it should complete with no browser prompt,
   reusing the cached token.

`credentials.json` and `token.json` are gitignored and must never be
committed.
