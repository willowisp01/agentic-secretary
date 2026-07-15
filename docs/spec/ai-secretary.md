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
each action item in an open-text turn where the human (or the agent, if
asked to decide) composes a remedy plan — shift the slot, draft a reply,
accept a proposed meeting, any combination of those, or skip — that's
confirmed by the human before anything is generated, rather than
unilaterally authoring one draft to approve or reject. Chosen remedies stay
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
- `pydantic` — schemas for LLM structured-output extraction (validated, not
  just type-hinted); used with `with_structured_output(..., method=
  "json_schema")` for Claude's constrained-decoding structured outputs

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
count is small (~9). Revisit this once RAG (milestone 2) adds enough new
tools/nodes to justify splitting.

## Code Style

- Type-hint all function signatures; no bare `Any` for tool inputs/outputs.
- LangGraph node functions take and return the shared graph state
  (`TypedDict` or Pydantic model), e.g.:

```python
class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # chat turns: greeting,
                                                           # human input, free-text replies
    emails: list[EmailSummary]
    calendar_events: list[CalendarEvent]
    action_items: list[ActionNeeded]
    pending_action_index: int          # which action item is awaiting a remedy turn
    pending_remedies: list[RemedyLiteral]       # remedies queued for the current item
    pending_shift_event_ids: list[str]          # event id(s) to shift, when queued
    pending_explicit_time: datetime | None      # a specific time named in the reply, if any
    pending_plan_summary: str                   # plan text shown by confirm_plan
    resolutions: Annotated[list[ActionResolution], operator.add]  # chosen
                                                           # remedy per action item

def check_calendar(state: PlannerState) -> dict:
    events = tools.list_upcoming_events(calendar_service)
    return {"calendar_events": events}
```

Nodes return a partial dict of only the keys they update — not a full-state
spread — so each field's reducer (default: replace; `add_messages`/
`operator.add`: append) decides how it merges. `PlannerState` grows
incrementally with the graph: the Task 6 skeleton (`fetch_emails →
check_calendar`) only needs `emails`/`calendar_events`/`status`; `messages`,
`action_items`, and `pending_action_index` are added once action detection
exists (Task 7); the `pending_remedies`/`pending_shift_event_ids`/
`pending_explicit_time`/`pending_plan_summary` queue fields and
`resolutions` are added once the open-turn remedy nodes that use them exist
(Task 8).

- Tools are thin wrappers around the Google API clients — no business logic
  in the tool layer; reasoning belongs in graph nodes / prompts.
- No inline secrets or hardcoded burner-account identifiers — everything
  sensitive comes from `.env` via `config.py`.

## Testing Strategy

- `pytest` for unit tests.
- `tests/test_tools.py`: mock the Google API clients (no live network calls
  in unit tests) — verify tool functions parse/shape data correctly.
- `tests/test_graph.py`: run the compiled LangGraph app against fixture
  state (not live APIs) and assert on graph wiring/state shape.
- `tests/test_actions.py`: exercise `detect_actions` (deterministic
  overlap/back-to-back math, plus the LLM-assisted email patterns with
  `_analyze_email` mocked — no live Anthropic calls in the automated suite,
  same "no live API calls" rule as the Google clients) against fixture data
  loaded from the real `seed_data/*.yaml`, not hand-duplicated lookalikes.
- No coverage percentage target for a portfolio project of this size;
  prioritize covering the action-detection logic and tool-parsing edges
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

## Action Response Behavior

The calendar is the source of truth: every `ActionNeeded` item is anchored
to at least one `CalendarEvent`, and an email is only ever checked against
the calendar, never against another email. Action detection has no
email-vs-email case for this reason (see the `ActionNeeded` shape in
Architecture above). `EmailConflict` is multi-event: a single proposed
meeting can overlap more than one existing calendar event, in which case
one `EmailConflict` lists all of them, rather than one item per overlap
(the latter was tried first and found, via live testing, to produce
duplicate/contradictory Gmail drafts when each overlap was resolved
independently).

Milestone 1 does not prescribe per-pattern draft content (i.e. no fixed
template like "for a calendar-calendar overlap, always draft X"). Instead,
once `detect_actions` finds an action item, the agent presents it in an
open-text chat turn — suggested remedies shown as plain text, not a forced
numbered menu — and the human replies freely (naming one remedy, several at
once, or "you decide" to delegate the choice). An LLM turns that reply into
a structured plan — which remedy or remedies, possibly more than one for
the same item, any specific time named, and a plain-text summary — which is
then shown back to the human for explicit confirmation *before* anything is
generated. Only after confirmation does the agent actually call a tool:

1. **Shift the slot** — calls `propose_event(...)`, producing a structured
   `EventProposal`. Never calls Calendar's `insert`/`patch`.
2. **Draft a reply email** — calls `draft_reply(...)`, producing a Gmail
   draft via `drafts.create`. Never calls `send`.
3. **Accept the meeting** — `EmailConflict` items only (the ones with a
   proposed *new* meeting to accept). Calls `propose_event(...)` using the
   time/duration already extracted from the email during detection, no new
   LLM reasoning needed. Never calls `insert`.
4. **Skip** — no tool call; the action item is left unresolved for this run.

Multiple remedies can be chosen together for one action item (e.g. shift
the slot *and* draft a reply) — each produces its own resolution. The
confirmation step includes a deterministic (non-LLM) check: if the plan
pins an explicit time, it's compared against known calendar events and any
shifts already proposed this session, and a warning is shown if it
overlaps — advisory only, the human can still confirm anyway. Rejecting the
plan re-shows the same item for another reply rather than moving on.

The human's confirmed plan determines the action; the agent does not
unilaterally author and present a single draft for approve/reject. All four
remedies are propose-only, matching the tool-layer boundary already
enforced in `tools.py` — no write-capable tool (`patch`/`update`/`send`) is
in scope for milestone 1. Actually applying a slot shift, booking a new
meeting, or sending a drafted email is deferred to a later milestone, gated
behind its own human-in-the-loop confirmation.

## Success Criteria

- [ ] `uv run python scripts/seed_demo_data.py` populates the burner Gmail +
      Calendar with the seeded synthetic scenarios (including at least one
      deliberate time conflict, per the conflict-seeding patterns below).
- [ ] `uv run python -m agentic_secretary.cli` opens a chat session against
      the seeded account: the agent greets the user, the user asks it to
      check for conflicts, the agent fetches emails/calendar, detects the
      seeded conflicts, and for each presents an open-text remedy turn
      (shift slot / draft email / accept meeting / skip, any combination),
      confirmed by the human before generation, per the Action Response
      Behavior above.
- [ ] No action (send/create) happens without an explicit human
      confirmation in the chat flow, and even a confirmed remedy only ever
      produces a proposal (draft or structured event), never a
      send/insert/patch call.
- [ ] A LangSmith trace exists for the run and shows the node-by-node
      reasoning path.
- [ ] `uv run pytest` passes, covering tool-parsing and action-detection
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

Resolved: lint/format tool is Ruff; LangGraph checkpointer is in-memory for
milestone 1 (resets each CLI run — revisit if a later demo wants to show
resuming a paused/interrupted session); milestone 1 entry point is a chat
loop rather than a fixed no-input pipeline. Resolved 2026-07-15: action
response moved from a fixed numbered remedy menu to an open-text turn with
LLM-interpreted, human-confirmed multi-remedy plans (see Action Response
Behavior above) — live testing found the fixed-menu design couldn't
express "shift AND draft" for one item, and silently produced duplicate,
contradictory Gmail drafts when one email conflicted with two calendar
events.

Deferred, not blocking milestone 1:
- **Shared-event-linking across different `ActionNeeded` kinds** — e.g. a
  back-to-back pair where one event is also independently named in a
  reschedule email. Cheap to add later (thread sibling-item context into
  the remedy-planning prompt, no schema change needed) — not scheduled.
- **No way to originate a fresh email for a pure calendar-calendar
  conflict** — those items have no attached email/thread, and `draft_reply`
  requires an existing `thread_id` (reply-only, not fresh-compose). Good to
  have, not needed for milestone 1.
- **The deterministic overlap check in the confirmation step is bounded by
  the calendar fetch window** — `calendar_events` comes from one
  `list_upcoming_events(max_results=10)` call at session start and is never
  refreshed; a target date beyond that window would pass the check with
  false confidence on a busier calendar. Good to have, not needed for this
  demo's scale.
