# Implementation Plan: AI Secretary — Milestone 1 (Planner)

Implements [`docs/spec/ai-secretary.md`](../docs/spec/ai-secretary.md).

## Overview

Build a LangGraph-orchestrated planner agent that reads a burner Gmail inbox
and Google Calendar, detects scheduling action items against seeded
synthetic data, and — via a chat loop rather than a fixed no-input
pipeline — resolves them autonomously using bound tools before presenting
one summary for human review and correction, instead of gating each
decision behind a pre-approval menu. Every resolution stays propose-only
(never auto-sends/auto-books). Claude Haiku 4.5 is the default model;
LangSmith traces the reasoning path.

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
                            └── Autonomous resolution + review interrupt (Task 8)
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
  `docs/spec/ai-secretary.md`'s Action Resolution Behavior section) — the
  agent greets the user and waits for free text (e.g. "check for
  conflicts") rather than running `fetch_emails → check_calendar →
  detect_actions` unconditionally.
- **Autonomous resolution, review-after** (revised 2026-07-15, superseding
  the original menu design below) — after `detect_actions`, the agent gets
  all action items and real `@tool`-bound `propose_event`/`draft_reply`
  functions in one agentic loop (`agent` ⇄ `tools`), resolves everything it
  can on its own judgment — tool calls execute immediately, no
  pre-execution gate, since both tools are already propose-only by
  construction — then presents one summary for human review. Corrections
  are handled conversationally: the agent has full memory of what it
  already did, so a follow-up like "move Client Sync to 2pm instead" is
  resolved against its own prior tool call, not treated as a fresh request.
  Replaces an earlier design where each conflict got a human-chosen remedy
  via a fixed pre-approval menu (shift slot / draft email / skip) — that
  design gated every decision individually, which added interaction
  friction without adding real safety beyond the propose-only tool
  boundary itself. Both `propose_event`/`draft_reply` stay propose-only — no
  write-capable tool is added for milestone 1.

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
- [x] Task 7: Conflict-detection node
- [ ] Task 8: Autonomous resolution + review interrupt
  - [ ] Task 8.0: Detection-layer typing foundation (`ActionNeeded` union)
  - [ ] Task 8.1: `@tool`-annotate `propose_event`/`draft_reply`
  - [ ] Task 8.2: `agent` + `tools` loop
  - [ ] Task 8.3: `review` node (interrupt + routing)
  - [ ] Task 8.4: Deterministic collision annotation
  - [ ] Task 8.5: Graph wiring + CLI
  - [ ] Task 8.6: System prompt
  - [ ] Task 8.7: Live verification

### Checkpoint: Core Agent Flow
- [ ] End-to-end CLI run against the seeded account: greet → free-text
      "check for conflicts" → fetch → detect seeded action items → agent
      resolves them autonomously → one review summary
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
- [x] Every function has a typed signature and docstring
- [x] `draft_reply` calls Gmail's draft-create endpoint, never `send`
- [x] `propose_event` returns a structured proposal object, never calls
      Calendar's `events.insert`

**Verification:**
- [x] `tests/test_tools.py` mocks the Google API client; asserts `list_*`
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
- [x] Running the script populates the burner Gmail + Calendar with all
      seeded scenarios (confirmed indirectly: Task 6's 2026-07-13 live CLI
      run fetched 4 emails + 4 calendar events from the burner account,
      which only exist there because this script seeded them)
- [x] Relative times resolve correctly relative to "now" at seed time
      (covered by `tests/test_seed_demo_data.py`: offset formats `-2h`/`+30m`/
      `-1d` and the day+clock-time format `+1d 09:00`, plus invalid-format
      rejection)

**Verification:**
- [x] Manual run + visual check in Gmail/Calendar web UI (live-API action,
      not part of the automated suite) — confirmed via Task 6's live fetch
      results rather than a separate visual check

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
- [x] `uv run python -m agentic_secretary.cli` (bare-bones CLI) runs the
      graph and prints fetched emails + calendar events from the seeded
      burner account (verified live 2026-07-13: 4 emails + 4 calendar
      events fetched successfully)

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
- [x] Given the seeded fixture data, `detect_conflicts` identifies at least
      one real conflict of each of the 4 pattern types
- [x] A conflict-free fixture produces no false-positive conflicts

**Verification:**
- [x] `tests/test_conflicts.py` exercises `detect_conflicts` against fixture
      state for each pattern and asserts conflicts are found, plus a
      negative case for the conflict-free fixture (verified live 2026-07-14
      against the real Anthropic API, repeated runs: no crashes, all 4
      patterns classified correctly, no false positives)

**Dependencies:** Task 6

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_conflicts.py`

**Estimated scope:** Medium

**Implementation note:** `calendar_overlap`/`back_to_back` are deterministic
(direct time-range comparison); `email_conflict`/`reschedule` use an
LLM-assisted `_analyze_email` call per email. That call uses a Pydantic
`BaseModel` schema (`_EmailIntent`) with `with_structured_output(...,
method="json_schema")` rather than a `TypedDict` with the default
`function_calling` method — live testing showed `function_calling` doesn't
guarantee every schema field is populated (Claude can omit a key entirely,
causing a `KeyError` downstream), while `json_schema` uses Claude's
constrained-decoding structured-outputs feature to guarantee schema
conformance. A `model_validator` additionally zeroes fields whose paired
boolean is false, since live probing showed the model can still attach a
real, valid event id to an email that references no event at all. `pydantic`
was added as an explicit dependency (previously only transitive via
`langchain`/`langgraph`). Future LLM-calling nodes (e.g. Task 8's draft
generation) should follow this same pattern.

---

### Task 8: Autonomous resolution + review interrupt

Supersedes an earlier "chat loop + conflict remedy menu" design (human
picks a remedy from a fixed menu per conflict, pre-approval-gated). Revised
2026-07-15: the propose-only tool boundary (`propose_event`/`draft_reply`
never call `insert`/`patch`/`send`) is already the real safety guarantee,
so gating every individual decision behind pre-approval added interaction
friction without adding real safety. New design: the agent resolves all
detected action items autonomously in one pass using real `@tool`-bound
functions, then presents one summary; the human reviews the *result* and
can request corrections conversationally. See `docs/spec/ai-secretary.md`'s
Action Resolution Behavior section.

Split into sub-tasks along the node/module boundaries the design
introduces (`state.py`/`detection.py`/`resolution.py`/`review.py`/
`graph.py` — see Project Structure in the spec), each independently
testable and committable per the `/build` TDD loop.

---

#### Task 8.0: Detection-layer typing foundation

**Description:** Rebuild `Conflict` as the Pydantic `ActionNeeded`
discriminated union (`CalendarOverlapConflict`/`BackToBackConflict`/
`EmailConflict`/`RescheduleRequest`) with correct per-kind field arity
(e.g. overlaps always carry exactly 2 events; `EmailConflict` can
reference more than one existing event, which the current `Conflict`
TypedDict doesn't enforce). Detection logic itself
(`_find_calendar_overlaps`, `_find_back_to_back`, `_analyze_email`) is
unchanged — only the container type gets stricter. Also moves detection
code out of `graph.py` into the new `detection.py` module, and introduces
`state.py` for `PlannerState` (breaks the import cycle: `graph.py` imports
node functions from `detection.py`/`resolution.py`/`review.py`, which each
need `PlannerState`). Since `PlannerState` itself needs `ActionNeeded` for
its `action_items` field's type hint, `ActionNeeded` and its 4 variants
live in `state.py` alongside `PlannerState` (not in `detection.py`, which
would recreate the same cycle one level down) — `detection.py` imports
both from `state.py`. Adds `messages: Annotated[list[AnyMessage],
add_messages]` to `PlannerState` here too, even though the chat loop that
first populates it doesn't exist until Task 8.5 — Task 8.2's `agent` node
needs the field to exist on `PlannerState` before then.

**Acceptance criteria:**
- [ ] `ActionNeeded` and its 4 variants live in `state.py`; `PlannerState`
      gains `action_items: list[ActionNeeded]` and `messages`
- [ ] `ActionNeeded` is a Pydantic discriminated union with 4 variants,
      each enforcing its own field arity
- [ ] `detect_actions` (renamed from `detect_conflicts`) returns
      `list[ActionNeeded]` and lives in `detection.py`
- [ ] All Task 7 detection behavior is unchanged for the 4 conflict
      patterns

**Verification:**
- [ ] `tests/test_detection.py` (renamed from `test_conflicts.py`) passes
      against the same fixtures as Task 7, plus a new case asserting
      `EmailConflict` can reference more than one event

**Dependencies:** Task 7

**Files likely touched:** `src/agentic_secretary/detection.py` (new),
`src/agentic_secretary/state.py` (new), `src/agentic_secretary/graph.py`
(detection code removed), `tests/test_detection.py` (renamed)

**Estimated scope:** Medium

---

#### Task 8.1: `@tool`-annotate `propose_event`/`draft_reply`

**Description:** The point where this project's tools become real
LangChain tools for the first time. Docstrings become the "which tool
applies to which item" logic (the LLM reads them directly instead of a
hand-coded registry), so they need to be precise: `propose_event`'s
docstring must state that omitting `existing_event_id` proposes a
brand-new event (for accepting a meeting request) while setting it
proposes moving that existing event.

**Acceptance criteria:**
- [ ] `propose_event`/`draft_reply` decorated with LangChain's `@tool`
- [ ] Docstrings specify applicability precisely, including the
      `existing_event_id` semantics above

**Verification:**
- [ ] `tests/test_tools.py` still passes unchanged (decoration doesn't
      change behavior/callability as plain functions)
- [ ] New assertion: each tool exposes `.name`/`.description` suitable for
      `bind_tools`

**Dependencies:** None (independent of 8.0)

**Files likely touched:** `src/agentic_secretary/tools.py`,
`tests/test_tools.py`

**Estimated scope:** Small

---

#### Task 8.2: `agent` + `tools` loop

**Description:** Standard LangGraph pattern: `agent` node calls
`ChatAnthropic(...).bind_tools([propose_event, draft_reply])` with the
system prompt (Task 8.6) + all `action_items` + message history;
`tools_condition` routes to a `ToolNode` when the response has tool calls,
looping back to `agent`, or onward to `review` once the agent responds
with plain text. Tool calls execute immediately, no interrupt — they're
already non-destructive by construction. `action_items` are serialized as
structured, id-bearing text (same convention as `_analyze_email`'s
existing `events_context` block — `- id=... title=... start=... end=...`,
or `model_dump_json()`), not the human-readable `description` prose: the
LLM needs to copy exact ids/times verbatim into tool calls, not infer them
back out of a sentence.

Carries forward a live-testing finding from Task 7: `_analyze_email`
miscomputed a relative-day reference by one day when left to infer it.
`agent` faces the same risk computing target datetimes for `propose_event`
(from an email's relative phrasing, or a human's follow-up like "next
Tuesday") — give it the anchor date's weekday name explicitly in context
rather than making it infer one.

**Acceptance criteria:**
- [ ] `agent` node wired with bound tools + system prompt + serialized
      action items + message history
- [ ] `tools_condition` loop terminates correctly (tool calls → `tools` →
      back to `agent`; plain text → `review`)
- [ ] Anchor-date weekday name is included in context explicitly

**Verification:**
- [ ] `tests/test_resolution.py` — mocked `ChatAnthropic.bind_tools(...)`
      responses; assert the right tool is called with the right args, and
      the loop terminates once the mock stops returning tool calls

**Dependencies:** Task 8.0, Task 8.1

**Files likely touched:** `src/agentic_secretary/resolution.py` (new),
`tests/test_resolution.py` (new)

**Estimated scope:** Medium

---

#### Task 8.3: `review` node

**Description:** `interrupt()` showing the agent's final summary text.
Deterministic (not LLM) routing on the reply: a small fixed set of exit
phrases ("done", "no", "nothing else", "that's all", "bye") routes to
`END`; anything else is appended as a `HumanMessage` and loops back to
`agent`, which re-engages with full memory of everything done so far.

**Acceptance criteria:**
- [ ] `review` calls `interrupt()` with the agent's final text
- [ ] Exit phrases route to `END`; anything else loops back to `agent`

**Verification:**
- [ ] `tests/test_review.py` covers both routing branches deterministically

**Dependencies:** Task 8.2

**Files likely touched:** `src/agentic_secretary/review.py` (new),
`tests/test_review.py` (new)

**Estimated scope:** Small

---

#### Task 8.4: Deterministic collision annotation

**Description:** Scans the `EventProposal` results from the tool calls
just made, adapts them into `CalendarEvent`-shaped values
(`end = start + timedelta(minutes=duration_minutes)`), and calls the
*existing* `_find_calendar_overlaps` (imported from `detection.py`)
directly against those plus the untouched `calendar_events` — one
definition of "overlap" in the codebase, not a parallel implementation.
Appends a plain FYI note to what `review` displays when a collision is
found. Advisory only, computed after the fact, not a gate — the one piece
of "don't trust the LLM with computable facts" carried forward from every
prior design, relocated from a pre-generation warning to a post-hoc
annotation.

**Acceptance criteria:**
- [ ] Reuses `_find_calendar_overlaps` rather than reimplementing overlap
      math
- [ ] Appends an FYI note only when a real collision is found; silent
      otherwise
- [ ] Never blocks or gates tool execution

**Verification:**
- [ ] `tests/test_review.py` — synthetic overlapping/non-overlapping
      `EventProposal` fixtures

**Dependencies:** Task 8.3

**Files likely touched:** `src/agentic_secretary/review.py`

**Estimated scope:** Small

---

#### Task 8.5: Chat entry + graph wiring + CLI

**Description:** `greet`/`classify_intent` don't exist anywhere yet on
this branch (Task 6/7 only ever built `fetch_emails → check_calendar →
detect_conflicts`, no chat loop) — this task writes them, not just wires
them in. `greet` is a static opening message; `classify_intent` is a small
LLM-classified routing step (free text → "check for conflicts" routes
onward to `fetch_emails`, anything else loops back for another turn). Real
logic (an LLM call), so it belongs in its own module, not `graph.py`, per
this task's own "wiring only" principle — new `chat.py`. `build_graph`
then assembles `greet → classify_intent → fetch_emails → check_calendar →
detect_actions → agent ⇄ tools → review ⇄ agent → END`, importing node
functions from `chat.py`/`detection.py`/`resolution.py`/`review.py` plus
LangGraph's `ToolNode` for `tools` — `graph.py` itself contains only
imports, `add_node`/`add_edge`/conditional edges, checkpointer, and
`compile`. `cli.py`'s final print is just the last `AIMessage.content`
from the finished state — the same summary text `review` already showed
the human on the last turn, not a second, independently-derived
description of the same outcome (avoids reintroducing a small version of
the plan/summary drift this whole redesign eliminated).

**Acceptance criteria:**
- [ ] `greet` and `classify_intent` exist in `chat.py` and route correctly
      (free-text "check for conflicts"-style intent → `fetch_emails`;
      anything else loops back for another chat turn)
- [ ] `build_graph` wires `greet → classify_intent → fetch_emails →
      check_calendar → detect_actions → agent ⇄ tools → review ⇄ agent →
      END`
- [ ] `graph.py` contains no chat/detection/resolution/review logic, only
      wiring
- [ ] `cli.py` prints the last `AIMessage.content`, not a separately
      derived summary

**Verification:**
- [ ] `tests/test_chat.py` (new) — mocked LLM classification, asserts
      routing for an in-scope vs. out-of-scope free-text turn
- [ ] `tests/test_graph.py` asserts wiring/state shape (mocked nodes)

**Dependencies:** Task 8.0, Task 8.1, Task 8.2, Task 8.3, Task 8.4

**Files likely touched:** `src/agentic_secretary/chat.py` (new),
`src/agentic_secretary/graph.py`, `src/agentic_secretary/cli.py`,
`tests/test_chat.py` (new)

**Estimated scope:** Medium

---

#### Task 8.6: System prompt

**Description:** This project's first real system prompt. States:
propose-only, never send/book (redundant with the tool-layer guarantee,
but explicit defense in depth); work through every given action item
using reasonable judgment; ask directly (plain text, no tool call) when
something is genuinely unclear rather than guessing; treat a correction
given after seeing results as amending the specific thing it refers to,
not a fresh unrelated request.

**Acceptance criteria:**
- [ ] System prompt covers all four points above
- [ ] Anchor-date context (current date + weekday name) is passed
      explicitly, per Task 8.2's relative-date consideration

**Verification:** covered by Task 8.2's mocked tests (wiring) and Task
8.7 (live prompt-quality check) — prompt wording itself is a live/eval
concern, not a unit-testable one

**Dependencies:** Task 8.2

**Files likely touched:** `src/agentic_secretary/resolution.py`

**Estimated scope:** Small

---

#### Task 8.7: Live verification

**Description:** Re-run representative scenarios against the finished
design: multiple action items resolved in one autonomous pass, an
ambiguous item the agent should ask about instead of guessing, a
correction after seeing results ("move X to 2pm instead"), and a
shared-event scenario (same event referenced by two different action
items) that should self-resolve via the agent's own conversation memory
alone, with no special-case code.

**Acceptance criteria:**
- [ ] Multiple action items resolved in a single pass with one review
      summary — no per-item approval friction
- [ ] A correction after review is understood in context, not treated as
      a fresh request
- [ ] Genuine ambiguity produces a direct question, not a bad guess

**Verification:**
- [ ] Manual CLI walkthrough against the seeded burner account, documented

**Dependencies:** Task 8.0–8.6

**Files likely touched:** none (manual verification)

**Estimated scope:** Small

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
| LangGraph interrupt/human-in-the-loop API unfamiliarity | Low-Medium | Prototype the interrupt pattern early in Task 8.3; consult LangGraph docs via `source-driven-development` if behavior doesn't match expectations |
| Agent misjudges which tool/timing applies to an item (autonomous-resolution design shifts more judgment onto the LLM and the Task 8.6 system prompt than the old fixed-menu design did) | Medium | Mitigated by consequence, not prevention: nothing the agent calls is destructive (no `insert`/`send`), so a bad judgment call is a one-turn correction via `review`, not a real mistake. Task 8.7's live verification specifically checks judgment quality, since it isn't unit-testable |
| Scope creep toward RAG/multi-persona mid-implementation | Low | Explicitly out of scope per `docs/spec/ai-secretary.md` boundaries; flag and defer if tempted mid-task |

## Open Questions

None outstanding for this plan. Resolved 2026-07-13: milestone 1 moved from
a fixed no-input pipeline to a chat loop. Resolved 2026-07-15: conflict
response moved from a human-chosen remedy menu to autonomous
resolution-with-review (see Architecture Decisions above and
`docs/spec/ai-secretary.md`'s Action Resolution Behavior section) — the
menu design's per-decision pre-approval gate added interaction friction
without adding real safety beyond the propose-only tool boundary itself.
Note: Tasks 2, 5, 8.7 (manual check), and 9
depend on you completing Google Cloud OAuth client setup and having burner
account access ready — these are prerequisites outside what can be automated
and should be confirmed before implementation begins.
