# agentic-secretary

An AI agent that reads a Gmail inbox and Google Calendar, detects scheduling
problems (overlaps, tight back-to-backs, meeting-request emails, reschedule
emails), and resolves what it can using its own judgment — always as a
proposal or draft for a human to review, never as a final sent/booked
action.

## Setup

1. Install dependencies:
   ```
   uv sync
   ```
2. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (required).
   `MODEL_NAME` is optional and defaults to `claude-haiku-4-5`.

### Google OAuth setup

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

### LangSmith tracing (optional, recommended)

1. Create a project at [smith.langchain.com](https://smith.langchain.com/)
   and generate an API key.
2. In `.env`, set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and
   `LANGSMITH_PROJECT` (see `.env.example`). `LANGSMITH_API_KEY` alone does
   nothing without `LANGSMITH_TRACING=true`.

Once enabled, each turn of the CLI loop (the initial `graph.invoke` and
every resumed turn after an `interrupt`) appears as its own trace in the
project dashboard, all sharing `thread_id="cli"` so they group into one
thread. Confirmed live: a trace's node path shows the full graph --
`greet` -> `classify_intent` -> `fetch_emails` -> `check_calendar` ->
`detect_actions` (-> `agent` <-> `tools` -> `review` once an action item
exists) -- with each node's own `ChatAnthropic` calls nested underneath it.

## Seed demo data

```
uv run python scripts/seed_demo_data.py
```

Loads the fixtures in `seed_data/*.yaml` and inserts them directly into the
burner account's Gmail and Calendar (a handful of emails and events,
including at least one deliberate scheduling conflict). It prints the
authenticated account's email address and asks for a `[y/N]` confirmation
before writing anything, as a guard against seeding into the wrong account.

**All seed content is synthetic.** The emails, calendar events, and the
"busy working professional" persona they belong to are fictional test data
written for this project — not drawn from the author's own inbox or
calendar. Real Gmail/Calendar access runs through a dedicated burner
account for this reason, so the demo never touches the author's real
schedule or a stranger's real correspondence.

`scripts/nuke_seed_data.py` removes previously-seeded messages/events from
the burner account, if you want to reset before re-seeding.

## Run the demo

```
uv run python -m agentic_secretary.cli
```

Opens a chat session in the terminal:

1. The agent greets you and asks what you'd like to do.
2. Reply with something like `check for conflicts`. The agent fetches the
   burner account's recent emails and upcoming calendar events, and detects
   any action items (conflicts, meeting requests, reschedules) among them.
3. If it finds any, it resolves what it can in one autonomous pass — using
   its own judgment to call `propose_event` (a new event, or shifting an
   existing one) and/or `draft_reply` for each item — then presents one
   summary of what it did for your review. It never actually sends an email
   or books/patches a calendar event; every resolution is a proposal or a
   Gmail draft, not a completed action.
4. You can reply conversationally with a correction (e.g. `move it to 2pm
   instead`), and the agent resolves it against what it already did rather
   than treating it as an unrelated new request. Reply `done` (or similar)
   to end the session.

## Testing

```
uv run pytest -m "not llm_eval"
uv run ruff check .
```

The `llm_eval` marker is for `tests/test_agent_examples_eval.py`, which
hits the real Anthropic API and costs money per run -- excluded from the
default suite and from CI (`.github/workflows/ci.yml`) for that reason. Run
it explicitly with `uv run pytest tests/test_agent_examples_eval.py` if you
want to exercise it.
