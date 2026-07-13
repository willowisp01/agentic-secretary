# Implementation Plan: AI Secretary — Milestone 1 (Planner)

Implements [`docs/spec/ai-secretary.md`](../docs/spec/ai-secretary.md).

## Overview

Build a LangGraph-orchestrated planner agent that reads a burner Gmail inbox
and Google Calendar, detects scheduling time-conflicts against seeded
synthetic data, and — via a chat loop rather than a fixed no-input
pipeline — presents each conflict with a human-chosen remedy menu (shift
slot / draft email / skip) instead of unilaterally authoring a draft. Every
remedy stays propose-only (never auto-sends/auto-books). Claude Haiku 4.5 is
the default model; LangSmith traces the reasoning path.

## Dependency Graph

```
Config/deps (Task 1)
    │
    ├── Google OAuth (Task 2) ──────────────┐
    │                                        │
    ├── Seed data fixtures (Task 3)          │
    │       │                                │
    │       └── Seeding script (Task 5) ◄────┘
    │
    └── Tool wrappers (Task 4) ◄── Task 2
            │
            └── Graph skeleton (Task 6)
                    │
                    └── Conflict-detection node (Task 7)
                            │
                            └── Draft + human-review interrupt (Task 8)
                                    │
                                    ├── LangSmith verification (Task 9)
                                    ├── Full test/lint pass (Task 10)
                                    └── README/demo notes (Task 11)
```

## Architecture Decisions

- **`src/agentic_secretary/` single-file modules** (`tools.py`, `graph.py`)
  rather than subpackages — node/tool count is small enough for milestone 1;
  revisit if milestone 2 (RAG) adds enough surface area to justify splitting.
- **In-memory LangGraph checkpointer** — state doesn't need to survive a
  process restart yet; simplest option for a CLI-driven demo.
- **Tools are thin wrappers, no reasoning** — `draft_reply`/`propose_event`
  only ever *prepare* actions (Gmail draft, structured event proposal);
  nothing in the tool layer can send an email or create a calendar event.
  This enforces the "draft-only, no auto-send/auto-book" boundary at the
  code level, not just by convention.
- **Deterministic + LLM-assisted conflict detection** — direct time-overlap
  comparisons are checked deterministically where possible, with the LLM
  used for the interpretive parts (e.g., reading intent out of an email).
  Reduces the risk of the LLM missing an unambiguous overlap.
- **Chat loop, not a fixed pipeline** (decided before Task 6+, see
  `docs/spec/ai-secretary.md`'s Conflict Response Behavior section) — the
  agent greets the user and waits for free text (e.g. "check for
  conflicts") rather than running `fetch_emails → check_calendar →
  detect_conflicts` unconditionally. After `detect_conflicts`, each
  conflict gets a human-chosen remedy via menu (shift slot / draft email /
  skip) instead of a bot-authored draft to approve/reject. Both menu
  actions stay propose-only — no write-capable tool is added for milestone
  1.

## Task List

### Phase 1: Foundation

- [x] Task 1: Project scaffolding and config
- [x] Task 2: Google OAuth flow
- [x] Task 3: Seed data fixtures (conflict patterns)

### Checkpoint: Foundation
- [ ] `uv sync` succeeds, `uv run ruff check .` clean
- [ ] Manual OAuth smoke test obtains a working token against the burner account
- [ ] Seed fixture YAML parses and covers all 4 conflict patterns from the spec
- [ ] Review with human before proceeding

### Phase 2: Data Layer

- [x] Task 4: Gmail/Calendar tool wrappers
- [x] Task 5: Seeding script

### Checkpoint: Data Layer
- [ ] Burner account seeded successfully with all scenarios (manual visual check)
- [ ] Tool tests pass against mocks; no live API calls in the automated test suite
- [ ] Review with human before proceeding to the reasoning layer

### Phase 3: Agent Reasoning

- [x] Task 6: PlannerState + graph skeleton
- [ ] Task 7: Conflict-detection node
- [ ] Task 8: Draft-response node + human-review interrupt

### Checkpoint: Core Agent Flow
- [ ] End-to-end CLI run against the seeded account: greet → free-text
      "check for conflicts" → fetch → detect seeded conflicts → pause at
      remedy menu per conflict
- [ ] All conflict-pattern tests pass against fixtures
- [ ] Review with human before observability/polish phase

### Phase 4: Observability and Polish

- [ ] Task 9: LangSmith tracing verification
- [ ] Task 10: Full test suite + lint pass
- [ ] Task 11: README + demo walkthrough notes

### Checkpoint: Complete
- [ ] All success criteria in `docs/spec/ai-secretary.md` are met
- [ ] `uv run pytest` and `uv run ruff check .` both pass
- [ ] LangSmith trace confirmed for a full run
- [ ] README walkthrough verified end-to-end
- [ ] Ready for human review / demo recording

## Task Details

### Task 1: Project scaffolding and config

**Description:** Add all phase-1 dependencies to `pyproject.toml` and
implement `config.py` to load `.env` (API keys, Google client secret path)
and expose the default model constant.

**Acceptance criteria:**
- [x] `pyproject.toml` lists `langchain`, `langchain-anthropic`, `langgraph`,
      `google-api-python-client`, `google-auth-oauthlib`,
      `google-auth-httplib2`, `langsmith`, `python-dotenv`, `pyyaml` plus
      `pytest`/`ruff` as dev dependencies
- [x] `config.py` loads `.env` and exposes `ANTHROPIC_API_KEY`,
      `LANGSMITH_API_KEY`, `GOOGLE_CLIENT_SECRET_PATH`, `GOOGLE_TOKEN_PATH`,
      and `MODEL_NAME` (default `claude-haiku-4-5`)

**Verification:**
- [x] `uv sync` succeeds
- [x] `uv run python -c "from agentic_secretary.config import settings; print(settings.model_name)"` prints `claude-haiku-4-5`
- [x] `uv run ruff check src/` passes

**Dependencies:** None

**Files likely touched:** `pyproject.toml`, `src/agentic_secretary/__init__.py`,
`src/agentic_secretary/config.py`, `.env.example`

**Estimated scope:** Small

---

### Task 2: Google OAuth flow

**Description:** Implement the installed-app OAuth flow for Gmail +
Calendar scopes, caching and refreshing the token locally.

**Acceptance criteria:**
- [x] `auth.py` exposes a function returning valid `Credentials`, running
      the browser consent flow on first use and refreshing silently after
- [x] Token cache path and client secret path come from `config.py`, never
      hardcoded

**Verification:**
- [x] `tests/test_auth.py` mocks the OAuth flow to verify caching/refresh
      logic without live network calls
- [x] Manual smoke test (documented in README): first run opens a consent
      screen against the burner account; second run reuses the cached token
      with no browser prompt

**Dependencies:** Task 1

**Files likely touched:** `src/agentic_secretary/auth.py`, `tests/test_auth.py`,
`README.md`

**Estimated scope:** Medium

---

### Task 3: Seed data fixtures (conflict patterns)

**Description:** Author `seed_data/emails.yaml`, `seed_data/calendar_events.yaml`,
and `seed_data/relations.yaml` covering the 4 conflict patterns from the
spec (calendar-calendar overlap, email-request-vs-calendar conflict,
back-to-back no-buffer, reschedule/cancellation email) plus a `mentions`
relation — an email that references an existing event without proposing a
conflict or reschedule, to test the "no false positives" side of Task 7.
`relations.yaml` keeps cross-references out of `emails.yaml`/
`calendar_events.yaml` so those stay pure records; each relation's `kind`
determines its required id shape (see below). 4-6 email/event scenarios
total, plus their relations.

Relation kinds and required shape:
| kind | fields | arity |
|---|---|---|
| `calendar_overlap` | `events: [...]` | ≥ 2 |
| `back_to_back` | `events: [...]` | exactly 2 |
| `email_conflict` | `email`, `events: [...]` | ≥ 1 event |
| `reschedule` | `email`, `event` | exactly 1 event |
| `mentions` | `email`, `event` | exactly 1 event |

An email/event with no entry in `relations.yaml` is the "relates to
nothing" distractor case (e.g. an internal digest email).

**Acceptance criteria:**
- [x] Each of the 4 conflict patterns, plus `mentions`, is represented at
      least once in `relations.yaml`
- [x] Time fields use the relative-time convention (`sent_relative`,
      `start_relative`) so re-seeding always looks current
- [x] A loader/validator (`src/agentic_secretary/seed_data.py`) parses all
      three files into typed objects and raises if a relation's `kind`
      doesn't match its required arity, or if it references an unknown
      email/event id

**Verification:**
- [x] `tests/test_seed_data.py` asserts all three fixtures parse, all 5
      relation kinds are represented, `validate_relations` accepts the real
      fixtures without error, and rejects a malformed arity/unknown
      reference in a synthetic bad-fixture case

**Dependencies:** None (can run in parallel with Tasks 1-2)

**Files likely touched:** `seed_data/emails.yaml`, `seed_data/calendar_events.yaml`,
`seed_data/relations.yaml`, `src/agentic_secretary/seed_data.py`,
`tests/test_seed_data.py`

**Estimated scope:** Small

---

### Task 4: Gmail/Calendar tool wrappers

**Description:** Implement `tools.py`: `list_recent_emails()`,
`list_upcoming_events()`, `draft_reply(...)`, `propose_event(...)`. Thin
wrappers only — no reasoning logic. `draft_reply`/`propose_event` must only
*prepare* actions (Gmail draft, structured proposal), never send or create.

**Acceptance criteria:**
- [ ] Every function has a typed signature and docstring
- [ ] `draft_reply` calls Gmail's draft-create endpoint, never `send`
- [ ] `propose_event` returns a structured proposal object, never calls
      Calendar's `events.insert`

**Verification:**
- [ ] `tests/test_tools.py` mocks the Google API client; asserts `list_*`
      functions correctly parse mock API responses into typed objects, and
      asserts (`assert_not_called`) that no send/insert-committing method is
      ever invoked by `draft_reply`/`propose_event`

**Dependencies:** Task 2

**Files likely touched:** `src/agentic_secretary/tools.py`, `tests/test_tools.py`

**Estimated scope:** Medium

---

### Task 5: Seeding script

**Description:** `scripts/seed_demo_data.py` reads the Task 3 fixtures,
resolves relative timestamps to absolute ones, and inserts the
messages/events into the burner account via the Task 4 tool layer (or
direct API calls if seeding needs endpoints the agent's tools don't expose,
e.g. `messages.insert` for received mail vs. `drafts.create`).

**Acceptance criteria:**
- [ ] Running the script populates the burner Gmail + Calendar with all
      seeded scenarios (pending: needs a live run against the burner
      account/credentials, not available in this environment)
- [x] Relative times resolve correctly relative to "now" at seed time
      (covered by `tests/test_seed_demo_data.py`: offset formats `-2h`/`+30m`/
      `-1d` and the day+clock-time format `+1d 09:00`, plus invalid-format
      rejection)

**Verification:**
- [ ] Manual run + visual check in Gmail/Calendar web UI (live-API action,
      not part of the automated suite) — still needs to be run by hand

**Dependencies:** Task 2, Task 3

**Files likely touched:** `scripts/seed_demo_data.py`

**Estimated scope:** Medium

---

### Task 6: PlannerState + graph skeleton

**Description:** Define `PlannerState` and a minimal LangGraph graph
(`fetch_emails → check_calendar → END`) with an in-memory checkpointer, to
prove the graph compiles and runs against the live tool layer before adding
reasoning nodes.

**Acceptance criteria:**
- [ ] `uv run python -m agentic_secretary.cli` (bare-bones CLI) runs the
      graph and prints fetched emails + calendar events from the seeded
      burner account (pending: needs a live run against the burner
      account/credentials, not available in this environment)

**Verification:**
- [x] `tests/test_graph.py` runs the compiled graph against fixture state
      (not live APIs), asserting state shape after each node

**Dependencies:** Task 4

**Files likely touched:** `src/agentic_secretary/graph.py`,
`src/agentic_secretary/cli.py`, `tests/test_graph.py`

**Estimated scope:** Medium

---

### Task 7: Conflict-detection node

**Description:** Add a `detect_conflicts` node implementing the 4 conflict
patterns from the spec, using deterministic time-comparison logic where
possible and an LLM call (Haiku) for the interpretive parts (e.g.,
extracting a requested meeting time from free-text email content). The node
itself is unchanged by the chat-loop decision — it still takes
`emails`/`calendar_events` and returns `conflicts`; only Task 8 changes how
it's triggered (from a chat turn) and what happens with its output (a menu,
not an auto-draft).

**Acceptance criteria:**
- [ ] Given the seeded fixture data, `detect_conflicts` identifies at least
      one real conflict of each of the 4 pattern types
- [ ] A conflict-free fixture produces no false-positive conflicts

**Verification:**
- [ ] `tests/test_conflicts.py` exercises `detect_conflicts` against fixture
      state for each pattern and asserts conflicts are found, plus a
      negative case for the conflict-free fixture

**Dependencies:** Task 6

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_conflicts.py`

**Estimated scope:** Medium

---

### Task 8: Chat loop + conflict remedy menu

**Description:** Add the chat entry point: a `greet` node that opens with a
greeting, then a turn that accepts free-text human input (e.g. "check for
conflicts") and routes to `detect_conflicts` (Task 7). For each conflict
found, an interrupt presents a menu — shift slot / draft email / skip — per
`docs/spec/ai-secretary.md`'s Conflict Response Behavior. The chosen remedy
calls `propose_event` (shift) or `draft_reply` (draft email), or makes no
tool call (skip), and the outcome is appended to `state["resolutions"]`.
Milestone 1 stops at the proposal — no send/create ever happens, regardless
of which menu option is chosen.

**Acceptance criteria:**
- [ ] Running the CLI opens a chat prompt; a check-for-conflicts message
      runs detection and pauses at a remedy-menu interrupt for each
      detected conflict
- [ ] Choosing "shift slot" or "draft email" produces the corresponding
      proposal object and appends a resolution to `state["resolutions"]`;
      choosing "skip" appends a resolution with no tool call
- [ ] No tool call that would send an email or create/patch a calendar
      event exists anywhere in this phase's code path

**Verification:**
- [ ] Manual CLI run demonstrating greeting → free-text input → conflict →
      menu → proposal, for at least one conflict pattern
- [ ] `tests/test_graph.py` asserts the graph halts at the menu interrupt
      without a chosen remedy, and asserts each menu branch's resulting
      state shape

**Dependencies:** Task 7

**Files likely touched:** `src/agentic_secretary/graph.py`,
`src/agentic_secretary/cli.py`

**Estimated scope:** Medium

---

### Task 9: LangSmith tracing verification

**Description:** Wire `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`,
`LANGCHAIN_PROJECT` via `config.py`/`.env`, and confirm a CLI run produces a
visible trace in the LangSmith UI showing the full node path.

**Acceptance criteria:**
- [ ] A documented CLI run appears in the LangSmith project dashboard with
      all graph nodes visible in the trace

**Verification:**
- [ ] Manual check in the LangSmith UI

**Dependencies:** Task 8

**Files likely touched:** `src/agentic_secretary/config.py`, `README.md`

**Estimated scope:** Small

---

### Task 10: Full test suite + lint pass

**Description:** Close any remaining test gaps (tool edge cases, additional
conflict false-positive/negative cases) and run Ruff format + check across
the repo.

**Acceptance criteria:**
- [ ] `uv run pytest` passes with no failures
- [ ] `uv run ruff check .` passes with no errors

**Verification:**
- [ ] Both commands exit `0`

**Dependencies:** Tasks 4-8

**Files likely touched:** `tests/*`, `src/agentic_secretary/*`

**Estimated scope:** Small

---

### Task 11: README + demo walkthrough notes

**Description:** Document setup (OAuth, `.env`, seeding), how to run the
demo, and a short "what this demonstrates" section for recruiters, including
the synthetic-data disclosure note from the intent doc.

**Acceptance criteria:**
- [ ] Someone unfamiliar with the project can follow `README.md` from clone
      to a working demo run
- [ ] README discloses that demo data is synthetic/seeded, not real
      correspondence

**Verification:**
- [ ] Manual read-through / dry run of the documented steps

**Dependencies:** Task 9, Task 10

**Files likely touched:** `README.md`

**Estimated scope:** Small

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Google OAuth consent/scopes friction (unverified app warnings, scope approval) | Medium | Use a personal/testing-mode OAuth client with minimal scopes (`gmail.readonly`, `gmail.compose`, `calendar`); validate this early in Task 2 rather than discovering it late |
| LLM conflict-detection misses an ambiguous fixture wording | Medium | Keep seed data scenarios explicit and unambiguous (Task 3 acceptance criteria); back deterministic time-overlap checks with code, not just LLM judgment |
| LangGraph interrupt/human-in-the-loop API unfamiliarity | Low-Medium | Prototype the interrupt pattern early in Task 8; consult LangGraph docs via `source-driven-development` if behavior doesn't match expectations |
| Scope creep toward RAG/multi-persona mid-implementation | Low | Explicitly out of scope per `docs/spec/ai-secretary.md` boundaries; flag and defer if tempted mid-task |

## Open Questions

None outstanding for this plan. Resolved 2026-07-13: milestone 1 moved from
a fixed no-input pipeline to a chat loop, with conflict response as a
human-chosen remedy menu rather than a bot-authored draft (see Architecture
Decisions above and `docs/spec/ai-secretary.md`'s Conflict Response
Behavior section). Note: Tasks 2, 5, 8 (manual check), and 9
depend on you completing Google Cloud OAuth client setup and having burner
account access ready — these are prerequisites outside what can be automated
and should be confirmed before implementation begins.
