# Spec: AI Secretary (Portfolio Project) — Milestone 1: Planner

See [`docs/intent/ai-secretary.md`](../intent/ai-secretary.md) for the confirmed
project intent this spec implements.

## Objective

Build an AI planner agent that demonstrates applied agent-engineering skill
for a resume/portfolio audience. The agent reads a burner Gmail inbox and
Google Calendar, reasons about scheduling (conflicts, meeting requests,
reschedules), and drafts replies/events for human review rather than acting
autonomously. LangGraph orchestrates the reasoning; LangSmith provides
observability. RAG/vector-database work and multi-persona support are
explicitly out of scope for this phase (see intent doc).

**User:** a single fictional "busy working professional" persona, seeded with
synthetic emails and calendar events (see Seed Data below).

**Success looks like:** running the CLI against the seeded burner account
opens a chat session — the agent greets the user, the user replies with
free text (e.g. "check for conflicts"), and the agent (a) fetches the day's
schedule and inbox, (b) identifies at least one real time conflict between
a calendar event and an incoming meeting-request email, and (c) presents
each conflict with a menu of remedies (shift the slot / draft a reply
email / skip) for the human to choose — rather than unilaterally
authoring one draft to approve or reject. Chosen remedies stay
propose-only (a Gmail draft or a structured event proposal); actually
sending the email or booking the slot is deferred to a later milestone. The
full reasoning trace is visible in LangSmith.

## Tech Stack

- Python 3.13, managed with `uv`
- `langchain` + `langchain-anthropic` — LLM calls (Claude Haiku 4.5 default,
  Sonnet for harder-reasoning nodes if needed)
- `langgraph` — agent orchestration/graph
- `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
  — Gmail + Calendar API access
- `langsmith` — tracing/observability (env-var based, no code-level SDK calls
  beyond setting `LANGCHAIN_TRACING_V2=true`)
- `python-dotenv` — load `.env`
- `pytest` — testing
- `pyyaml` — seed data fixtures

## Commands

```
Install deps:  uv sync
Run agent:     uv run python -m agentic_secretary.cli
Seed demo data: uv run python scripts/seed_demo_data.py
Test:          uv run pytest
Lint/format:   uv run ruff check . && uv run ruff format .
```


## Project Structure

```
agentic-secretary/
├── .env                       # secrets (gitignored): ANTHROPIC_API_KEY,
│                               #   LANGSMITH_API_KEY, GOOGLE_CLIENT_SECRET_PATH
├── seed_data/
│   ├── emails.yaml            # synthetic scenario content
│   ├── calendar_events.yaml
│   └── relations.yaml         # cross-references (conflict/reschedule/mentions)
│                               #   between seeded emails and events
├── scripts/
│   ├── seed_demo_data.py      # pushes seed_data/ into the burner account
│   ├── nuke_seed_data.py      # clears seeded messages/events from the burner account
│   └── _google_account_safety.py  # confirms target account before a write script runs
├── src/agentic_secretary/
│   ├── __init__.py
│   ├── cli.py                 # entry point
│   ├── config.py              # env loading, model selection, constants
│   ├── auth.py                # Google OAuth flow (Gmail + Calendar)
│   ├── seed_data.py           # typed loader/validator for seed_data/*.yaml
│   ├── tools.py                # thin tool wrappers: list_recent_emails,
│   │                           #   list_upcoming_events, draft_reply, propose_event
│   └── graph.py                # LangGraph graph: PlannerState + nodes + edges
├── tests/                     # one test module per src/scripts module above
└── docs/
    ├── intent/ai-secretary.md
    └── spec/ai-secretary.md
```

Node/tool files stay single-file (not split into subpackages) while the node
count is small (~5). Revisit this once RAG (milestone 2) adds enough new
tools/nodes to justify splitting.

## Code Style

- Type-hint all function signatures; no bare `Any` for tool inputs/outputs.
- LangGraph node functions take and return the shared graph state
  (`TypedDict` or Pydantic model), e.g.:

```python
class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # chat turns: greeting,
                                                           # human input, menu replies
    emails: list[EmailSummary]
    calendar_events: list[CalendarEvent]
    conflicts: list[Conflict]
    pending_conflict_index: int          # which conflict is awaiting a menu choice
    resolutions: Annotated[list[ConflictResolution], operator.add]  # chosen
                                                           # remedy per conflict

def check_calendar(state: PlannerState) -> dict:
    events = tools.list_upcoming_events(calendar_service)
    return {"calendar_events": events}
```

Nodes return a partial dict of only the keys they update — not a full-state
spread — so each field's reducer (default: replace; `add_messages`/
`operator.add`: append) decides how it merges. `PlannerState` grows
incrementally with the graph: the Task 6 skeleton (`fetch_emails →
check_calendar`) only needs `emails`/`calendar_events`/`status`; `messages`,
`conflicts`, `pending_conflict_index`, and `resolutions` are added once the
conflict-detection and chat-menu nodes that use them exist (Task 7/8).

- Tools are thin wrappers around the Google API clients — no business logic
  in the tool layer; reasoning belongs in graph nodes / prompts.
- No inline secrets or hardcoded burner-account identifiers — everything
  sensitive comes from `.env` via `config.py`.

## Testing Strategy

- `pytest` for unit tests.
- `tests/test_tools.py`: mock the Google API clients (no live network calls
  in unit tests) — verify tool functions parse/shape data correctly.
- `tests/test_graph.py`: run the compiled LangGraph app against fixture
  state (not live APIs) and assert on conflict-detection logic and
  draft-output shape.
- No coverage percentage target for a portfolio project of this size;
  prioritize covering the conflict-detection logic and tool-parsing edges
  over exhaustive coverage.
- Live-API smoke testing (actually hitting the burner Gmail/Calendar) is
  manual, via `uv run python -m agentic_secretary.cli` against seeded data —
  not part of the automated test suite.

## Boundaries

- **Always do:** keep the agent's default behavior to draft-only (no
  auto-send email, no auto-create calendar event without explicit
  confirmation); keep all credentials/secrets out of git; run tests before
  committing node/tool logic changes.
- **Ask first:** adding new dependencies beyond the tech stack above;
  changing the default model tier (e.g., away from Haiku) in a way that
  changes cost profile; expanding scope into RAG/vector DB or multi-persona
  support (explicitly milestone 2+, per intent doc).
- **Never do:** commit real credentials, `.env`, `token.json`, or
  `credentials.json`; auto-send emails or auto-book events without a human
  approval step; deploy this publicly/hosted (local-only per intent doc).

## Conflict Response Behavior

Milestone 1 does not prescribe per-pattern draft content (i.e. no fixed
template like "for a calendar-calendar overlap, always draft X"). Instead,
once `detect_conflicts` finds a conflict, the agent presents it in a chat
turn and offers the human a menu:

1. **Shift the slot** — calls `propose_event(...)`, producing a structured
   `EventProposal`. Never calls Calendar's `insert`/`patch`.
2. **Draft a reply email** — calls `draft_reply(...)`, producing a Gmail
   draft via `drafts.create`. Never calls `send`.
3. **Skip** — no tool call; the conflict is left unresolved for this run.

The human's choice determines the action; the agent does not unilaterally
author and present a single draft for approve/reject. Both menu actions
(1 and 2) are propose-only, matching the tool-layer boundary already
enforced in `tools.py` — no write-capable tool (`patch`/`update`/`send`) is
in scope for milestone 1. Actually applying a slot shift or sending a
drafted email is deferred to a later milestone, gated behind its own
human-in-the-loop confirmation.

## Success Criteria

- [ ] `uv run python scripts/seed_demo_data.py` populates the burner Gmail +
      Calendar with the seeded synthetic scenarios (including at least one
      deliberate time conflict, per the conflict-seeding patterns below).
- [ ] `uv run python -m agentic_secretary.cli` opens a chat session against
      the seeded account: the agent greets the user, the user asks it to
      check for conflicts, the agent fetches emails/calendar, detects the
      seeded conflict, and presents a remedy menu (shift slot / draft email
      / skip) per the Conflict Response Behavior above.
- [ ] No action (send/create) happens without an explicit human menu choice
      in the chat flow, and even a chosen remedy only ever produces a
      proposal (draft or structured event), never a send/insert/patch call.
- [ ] A LangSmith trace exists for the run and shows the node-by-node
      reasoning path.
- [ ] `uv run pytest` passes, covering tool-parsing and conflict-detection
      logic against fixture data (no live API calls in the test suite).

## Seed Data — Conflict Patterns (Milestone 1 scope: time conflicts only)

Per the intent doc, seed data is synthetic and versioned in `seed_data/`.
Milestone 1 includes a small, curated set (4-6 scenarios) covering:

1. Direct calendar-to-calendar overlap (two seeded events collide).
2. Email meeting request that collides with an existing seeded event (the
   core "agentic reasoning" scenario — requires cross-referencing email
   content against calendar state).
3. Back-to-back events with no buffer (soft conflict).
4. A reschedule/cancellation email against an existing seeded event.

Resource conflicts (e.g., shared rooms) and priority conflicts (e.g.,
client vs. internal meeting importance) are out of scope for milestone 1.

## Open Questions

None outstanding. Resolved: lint/format tool is Ruff; LangGraph checkpointer
is in-memory for milestone 1 (resets each CLI run — revisit if a later demo
wants to show resuming a paused/interrupted session); milestone 1 entry
point is a chat loop rather than a fixed no-input pipeline, and conflict
response is a human-chosen remedy via menu rather than a bot-authored draft
per conflict pattern (see Conflict Response Behavior above).
