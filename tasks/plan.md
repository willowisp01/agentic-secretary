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
- [x] `uv sync` succeeds, `uv run ruff check .` clean
- [x] Manual OAuth smoke test obtains a working token against the burner account
- [x] Seed fixture YAML parses and covers all 4 conflict patterns from the spec
- [x] Review with human before proceeding

### Phase 2: Data Layer

- [x] Task 4: Gmail/Calendar tool wrappers
- [x] Task 5: Seeding script

### Checkpoint: Data Layer
- [x] Burner account seeded successfully with all scenarios (manual visual check)
- [x] Tool tests pass against mocks; no live API calls in the automated test suite
- [x] Review with human before proceeding to the reasoning layer

### Phase 3: Agent Reasoning

- [x] Task 6: PlannerState + graph skeleton
- [x] Task 7: Conflict-detection node
- [x] Task 8: Autonomous resolution + review interrupt
  - [x] Task 8.0: Detection-layer typing foundation (`ActionNeeded` union)
  - [x] Task 8.1: `@tool`-annotate `propose_event`/`draft_reply`
  - [x] Task 8.2: `agent` + `tools` loop
  - [x] Task 8.3: `review` node (interrupt + routing)
  - [x] Task 8.4: Deterministic collision annotation
  - [x] Task 8.5: Graph wiring + CLI
  - [x] Task 8.6: System prompt
  - [x] Task 8.7: Live verification

### Checkpoint: Core Agent Flow
- [x] End-to-end CLI run against the seeded account: greet → free-text
      "check for conflicts" → fetch → detect seeded action items → agent
      resolves them autonomously → one review summary
- [x] All conflict-pattern tests pass against fixtures
- [x] Review with human before observability/polish phase

### Phase 4: Observability and Polish

- [x] Task 9: LangSmith tracing verification
- [x] Task 10: Full test suite + lint pass
- [x] Task 11: README + demo walkthrough notes

### Checkpoint: Complete
- [x] All success criteria in `docs/spec/ai-secretary.md` are met
- [x] `uv run pytest` and `uv run ruff check .` both pass
- [x] LangSmith trace confirmed for a full run
- [x] README walkthrough verified end-to-end
- [x] Ready for human review / demo recording

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
- [x] `ActionNeeded` and its 4 variants live in `state.py`; `PlannerState`
      gains `action_items: list[ActionNeeded]` and `messages`
- [x] `ActionNeeded` is a Pydantic discriminated union with 4 variants,
      each enforcing its own field arity
- [x] `detect_actions` (renamed from `detect_conflicts`) returns
      `list[ActionNeeded]` and lives in `detection.py`
- [x] All Task 7 detection behavior is unchanged for the 4 conflict
      patterns

**Verification:**
- [x] `tests/test_detection.py` (renamed from `test_conflicts.py`) passes
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

`propose_event` has no service dependency, so it can be wrapped directly:
`propose_event_tool = tool(propose_event)` (functional form, not the
`@tool` decorator, so the plain function stays exactly as-is for direct
calls/existing tests). `draft_reply` can't be decorated in place — it
takes `service: Resource` as its first argument, and `Resource` isn't
JSON-schema-able, so a directly-decorated version would either fail schema
generation or expose an argument the LLM can never legally fill in. Fix:
`draft_reply` stays a plain function unchanged; add a
`make_draft_reply_tool(service) -> BaseTool` factory that closes over
`service` and exposes a `@tool`-decorated inner function taking only
`to`/`subject`/`body`/`thread_id`.

**Acceptance criteria:**
- [x] `propose_event_tool = tool(propose_event)` — plain `propose_event`
      unchanged
- [x] `make_draft_reply_tool(service)` returns a bindable tool exposing
      only `to`/`subject`/`body`/`thread_id`; plain `draft_reply`
      unchanged
- [x] Docstrings specify applicability precisely, including the
      `existing_event_id` semantics above

**Verification:**
- [x] `tests/test_tools.py` still passes unchanged (the plain functions
      keep their existing signatures and direct-call behavior)
- [x] New assertions: `propose_event_tool` and
      `make_draft_reply_tool(service)` each expose `.name`/`.description`
      suitable for `bind_tools`, and invoking the bound `draft_reply` tool
      calls through to the same service instance

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
- [x] `agent` node wired with bound tools + system prompt + serialized
      action items + message history
- [x] `tools_condition` loop terminates correctly (tool calls → `tools` →
      back to `agent`; plain text → `review`)
- [x] Anchor-date weekday name is included in context explicitly

**Verification:**
- [x] `tests/test_resolution.py` — mocked `ChatAnthropic.bind_tools(...)`
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
- [x] `review` calls `interrupt()` with the agent's final text
- [x] Exit phrases route to `END`; anything else loops back to `agent`

**Verification:**
- [x] `tests/test_review.py` covers both routing branches deterministically

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
- [x] Reuses `_find_calendar_overlaps` rather than reimplementing overlap
      math
- [x] Appends an FYI note only when a real collision is found; silent
      otherwise
- [x] Never blocks or gates tool execution

**Verification:**
- [x] `tests/test_review.py` — synthetic overlapping/non-overlapping
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
- [x] `greet` and `classify_intent` exist in `chat.py` and route correctly
      (free-text "check for conflicts"-style intent → `fetch_emails`;
      anything else loops back for another chat turn)
- [x] `build_graph` wires `greet → classify_intent → fetch_emails →
      check_calendar → detect_actions → agent ⇄ tools → review ⇄ agent →
      END`
- [x] `graph.py` contains no chat/detection/resolution/review logic, only
      wiring
- [x] `cli.py` prints the last `AIMessage.content`, not a separately
      derived summary

**Verification:**
- [x] `tests/test_chat.py` (new) — mocked LLM classification, asserts
      routing for an in-scope vs. out-of-scope free-text turn
- [x] `tests/test_graph.py` asserts wiring/state shape (mocked nodes)

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
- [x] System prompt covers all four points above
- [x] Anchor-date context (current date + weekday name) is passed
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
- [x] Multiple action items resolved in a single pass with one review
      summary — no per-item approval friction
- [x] A correction after review is understood in context, not treated as
      a fresh request
- [x] Genuine ambiguity produces a direct question, not a bad guess

**Verification:**
- [x] Manual CLI walkthrough against the seeded burner account, documented

**Dependencies:** Task 8.0–8.6

**Files likely touched:** none (manual verification)

**Estimated scope:** Small

---

### Task 9: LangSmith tracing verification ✅

**Description:** Wire `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`,
`LANGCHAIN_PROJECT` via `config.py`/`.env`, and confirm a CLI run produces a
visible trace in the LangSmith UI showing the full node path.

Note: those are the pre-rebrand env var names -- the actual LangSmith SDK
uses `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` (and
optionally `LANGSMITH_ENDPOINT` for self-hosted/regional deployments), which
is what got wired.

**Acceptance criteria:**
- [x] A documented CLI run appears in the LangSmith project dashboard with
      all graph nodes visible in the trace -- confirmed live: a real CLI
      run's trace showed the full `greet` -> `classify_intent` ->
      `fetch_emails` -> `check_calendar` -> `detect_actions` node path
      (Turn 17 of an ongoing thread), each with its own nested
      `ChatAnthropic` calls. Documented in README.md's "LangSmith tracing"
      section.

**Verification:**
- [x] Manual check in the LangSmith UI

**Dependencies:** Task 8

**Files likely touched:** `src/agentic_secretary/config.py`, `README.md`

**Estimated scope:** Small

**Beyond scope -- eval dataset built on top of this:** the original task
only asked for tracing wiring/verification, but while working through it we
also built a real LangSmith eval suite around `resolution.agent()`:

- `evals/agent_examples.py` -- 8 hand-authored single-step examples, each
  scripting either a first-turn `action_items` list or a full prior-turn
  message history for scenarios needing context, plus a structured
  `expected` block per example.
- `tests/test_agent_examples.py` -- mocked, free, CI-safe check that every
  example is well-formed and `agent()` runs on it without raising.
- `tests/test_agent_examples_eval.py` -- runs the real thing against
  Claude, marked `llm_eval` and excluded from CI (costs real API calls).
  Combines two evaluation layers on every run:
  - **Code-based evals**: deterministic checks (`tool_calls_include`,
    `content_must_not_contain`) for properties that have an objectively
    correct answer.
  - **LLM-as-a-judge eval**: a LangSmith-bound evaluator (`agent-evaluator`,
    feedback key `satisfies-rubric`) scores each example's `rubric` field --
    the softer, judgment-call properties a fixed assertion can't capture
    (e.g. "asked a clarifying question instead of guessing"). Synced
    automatically via `@pytest.mark.langsmith` on every test run, no
    separate dataset-upload step.
  - **Alignment loop**: the judge's prompt was refined after a live
    disagreement (a false negative on an "acknowledge without overclaiming"
    case) surfaced a real gap in its instructions -- confirmed fixed by
    re-running the full sample and checking agreement, not just trusting
    the fix.

---

### Task 10: Full test suite + lint pass ✅

**Description:** Close any remaining test gaps (tool edge cases, additional
conflict false-positive/negative cases) and run Ruff format + check across
the repo.

Closed 7 genuine coverage gaps (all characterization tests on already-correct
defensive code -- no bugs found, just previously-unverified behavior):
- `tools.py`: `_extract_plain_text_body` returns `""` for an HTML-only email
  (no `text/plain` part anywhere in the tree); `list_recent_emails`/
  `list_upcoming_events` return `[]` cleanly for an empty inbox/calendar.
- `detection.py`: `_find_calendar_overlaps` doesn't flag events that only
  touch at the boundary (false-positive guard, the overlap-side analog of
  the existing back-to-back threshold tests) and correctly finds every
  pairwise overlap among three mutually-overlapping events;
  `_find_back_to_back` doesn't also flag a pair that's genuinely
  overlapping (would be a confusing duplicate signal alongside the overlap
  conflict); `detect_actions` silently skips (doesn't raise `KeyError` on)
  a reschedule request whose `references_event_id` doesn't match any real
  calendar event.

**Acceptance criteria:**
- [x] `uv run pytest -m "not llm_eval"` passes with no failures (114 passed)
      -- matches what CI actually runs; the literal `uv run pytest` would
      also attempt `tests/test_agent_examples_eval.py`'s real Anthropic
      calls, which are deliberately excluded from routine runs (see Task 9's
      beyond-scope section) and weren't re-run just for this task
- [x] `uv run ruff check .` passes with no errors

**Verification:**
- [x] Both commands exit `0`

**Dependencies:** Tasks 4-8

**Files likely touched:** `tests/*`, `src/agentic_secretary/*`

**Estimated scope:** Small

---

### Task 11: README + demo walkthrough notes ✅

**Description:** Document setup (OAuth, `.env`, seeding) and how to run the
demo, including the synthetic-data disclosure note from the intent doc.

**Acceptance criteria:**
- [x] Someone unfamiliar with the project can follow `README.md` from clone
      to a working demo run -- Setup (deps, `.env`, OAuth, optional
      LangSmith) -> Seed demo data -> Run the demo -> Testing, in that order
- [x] README discloses that demo data is synthetic/seeded, not real
      correspondence -- "All seed content is synthetic" note under "Seed
      demo data", sourced from `docs/intent/ai-secretary.md`

**Verification:**
- [x] Manual read-through / dry run of the documented steps -- read through
      against the actual scripts (`scripts/seed_demo_data.py`'s `[y/N]`
      confirmation prompt, `cli.py`'s `if __name__ == "__main__"` entry
      point) rather than assumed; live CLI/seeding run not re-executed for
      this task (would touch the real burner account for no new signal
      beyond what Task 8.7 and Task 9 already confirmed live)

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

---

# Implementation Plan: AI Secretary — Milestone 1.5 (Resilience & UX Hardening)

## Overview

Before adding RAG, close two gaps discovered by re-examining Milestone 1's
finished graph: none of its API calls (three `ChatAnthropic` call sites,
four Google API call sites) have any failure handling — an exception
crashes the whole CLI process — and `route_after_detection` silently loops
back to `greet` when no action items are found, giving the human no signal
a check even ran. Both are fixed here, ahead of Milestone 2, so the new
RAG-related API calls (OpenAI, Chroma Cloud, the local BGE reranker) can
follow the same established pattern instead of inventing a second one.

**Guiding principle: 0 retries everywhere, fail fast, catch and fall back
immediately.** No exponential backoff, no attempt counting. The
conversational loop already gives the human a natural retry path (just ask
again), so a second automatic-retry layer underneath it would trade latency
for a benefit the chat loop already provides for free — simplest thing that
works, not the most sophisticated thing that could work.

Each of the three existing LLM call sites fails differently, because each
is a structurally different kind of function:

- **`classify_intent`** (a router, not a node — only returns a routing
  string) — catch, `print()` a plain diagnostic line directly to stdout
  (bypassing the conversational `messages`/`interrupt()` machinery
  entirely, since a router can't inject a message into state), fall back to
  routing `"greet"`.
- **`_analyze_email`** (one call per email, inside `detect_actions`'s loop)
  — catch per-email, skip that email (contributes no action item),
  continue the rest of the batch. Consistent with the existing precedent
  in this same function (a malformed reschedule reference is already
  silently skipped, per Milestone 1 Task 10). Lower stakes than the agent
  case below: nothing has been drafted/proposed yet at this stage, so
  there's no already-completed side effect to misrepresent — the risk here
  is a missed detection, not a hidden action.
- **`resolution.agent()`** (the main node, which may have already executed
  several real tool calls — real Gmail drafts, real proposals — before a
  later call in the same turn fails) — catch, and instead of a canned
  "sorry, try again" message, build an honest status report from whatever
  `review.py`'s existing `_latest_proposals` helper finds already in
  `state["messages"]` (reused, not duplicated), describing what was
  actually completed before the failure. A canned message would be
  dishonest here in a way it isn't for the other two: real, persisted side
  effects (a Gmail draft genuinely created via `drafts.create`) could
  already exist, and telling the human "nothing happened, try again" would
  hide that — the same category of risk `review.py`'s fixed disclaimer
  already exists to prevent, just triggered by a different cause. True
  rollback (deleting the already-created drafts) was considered and
  rejected: it would require the first destructive capability
  (`drafts.delete`) anywhere in `tools.py`, to undo something that's
  otherwise harmless to leave behind (a stray, propose-only draft the human
  can delete themselves) — not worth it against the simpler honest-report
  alternative. The resulting plain-text `AIMessage` flows through the
  *already-existing* `tools_condition`/`review` path with zero new graph
  wiring (no tool calls attached → routes to `review` exactly like a normal
  end-of-turn).

## Dependency Graph

```
no_action_items node (Task 12) ──────┐
                                       │ (independent of each other,
fetch_failed node + atomicity          │  both just add nodes/edges
  (Task 13)                            │  to the existing graph)
                                       │
classify_intent failure handling ─────┤
  (Task 14)                           │
                                       ├──► Checkpoint: hardening complete
_analyze_email failure handling ──────┤      (all four independent fixes
  (Task 15)                           │       verified together)
                                       │
resolution.agent() failure handling ──┘
  (Task 16, depends on review.py's
   existing _latest_proposals helper)
```

## Task List

### Phase A: Graph UX + Atomicity

- [x] Task 12: `no_action_items` node
- [x] Task 13: Fetch-stage atomicity (`fetch_failed`)

### Phase B: API failure handling

- [x] Task 14: `classify_intent` failure handling
- [ ] Task 15: `_analyze_email` failure handling
- [ ] Task 16: `resolution.agent()` failure handling

### Checkpoint: Milestone 1.5 complete
- [ ] A "check for conflicts" turn that finds nothing shows a real
      acknowledgment, not a silent loop back to `greet`
- [ ] A simulated Google API failure during `fetch_emails`/`check_calendar`
      aborts the whole turn (never reaches `detect_actions` with partial
      data) and shows a clear message
- [ ] A simulated `classify_intent` failure prints a diagnostic line and
      falls back to `greet` without crashing
- [ ] A simulated `_analyze_email` failure for one email doesn't stop the
      rest of the batch from being classified normally
- [ ] A simulated `resolution.agent()` failure after some tool calls
      already succeeded produces an honest report of what was actually
      done, not a generic "nothing happened" message
- [ ] `uv run pytest -m "not llm_eval"` and `uv run ruff check .` both pass
- [ ] Review with human before starting Milestone 2's RAG work

## Task Details

### Task 12: `no_action_items` node

**Description:** `route_after_detection` currently returns `"greet"` when
`state["action_items"]` is empty, silently reusing `greet`'s fixed
opening/reprompt message with no acknowledgment a check ran. Add a new
`no_action_items(state) -> dict` node (in `graph.py`, alongside the other
inline node closures) that shows a real message via `interrupt()` ("I
checked your calendar and inbox — nothing to report right now.") and
captures the reply. Routes through `classify_intent` next (not back through
`greet`, which would re-show its own fixed prompt and double up) — same
principle already applied to `answer_policy_question`/`fetch_failed`
elsewhere in this plan.

**Acceptance criteria:**
- [x] `route_after_detection` routes to `"no_action_items"` instead of
      `"greet"` when `action_items` is empty
- [x] `no_action_items` shows a specific, real message via `interrupt()`,
      not a generic fallback
- [x] Routes to `classify_intent` next, not back through `greet`

**Verification:**
- [x] `tests/test_graph.py` — a fixture with empty `action_items` routes
      to `no_action_items`, and its outgoing edge goes to `classify_intent`

**Dependencies:** None

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Small

---

### Task 13: Fetch-stage atomicity (`fetch_failed`)

**Description:** `fetch_emails`/`check_calendar` currently have no error
handling — an exception crashes the graph. Wrap each Google API call in a
plain try/except (0 retries) that, on failure, returns `{"status": "error",
"error_message": str(e)}` instead of letting the exception propagate. Add
two conditional edges (`fetch_emails → {check_calendar, fetch_failed}`,
`check_calendar → {detect_actions, fetch_failed}`) checking that status,
both routing to one shared `fetch_failed` node on failure. `fetch_failed`
shows the error via `interrupt()` ("Sorry, I couldn't complete that check —
[reason]. Want to try again?") and routes to `classify_intent` next.
Critically: on either step's failure, `detect_actions` is never reached
this turn — partial data (e.g. emails fetched but no calendar events) never
gets cross-referenced, avoiding a confidently-wrong result (a real conflict
silently missed because only half the data was available).

**Acceptance criteria:**
- [x] A `fetch_emails` failure routes to `fetch_failed`, never to
      `check_calendar`
- [x] A `check_calendar` failure routes to `fetch_failed`, never to
      `detect_actions`
- [x] `detect_actions` is never invoked with partial (emails-only or
      calendar-only) data in the same turn
- [x] `fetch_failed` shows a specific message and routes to
      `classify_intent` next

**Verification:**
- [x] `tests/test_graph.py` / `tests/test_tools.py` — mocked Google API
      clients raising on `fetch_emails` and separately on `check_calendar`
      each route to `fetch_failed` and never reach `detect_actions`

**Dependencies:** None

**Files likely touched:** `src/agentic_secretary/graph.py`,
`src/agentic_secretary/state.py`, `tests/test_graph.py`

**Estimated scope:** Medium

**Implementation note:** Added `error_message: NotRequired[str]` to
`PlannerState` rather than reusing `status`'s existing string for both the
verdict and the reason. `fetch_emails`/`check_calendar` now set `status`
fresh on every invocation (`"error"`, or `"fetching"`/`"done"` on success)
rather than only on failure -- needed so a stale `"error"` from an earlier
turn's failure never leaks into this turn's routing check after a
subsequent successful fetch.

---

### Task 14: `classify_intent` failure handling

**Description:** Wrap `classify_intent`'s `structured_llm.invoke(...)` call
in a plain try/except (0 retries). On failure: `print()` a plain line
directly to stdout (matching the existing `print(f"\n{...}\n")` formatting
already used in `cli.py`'s interrupt-rendering loop, so it doesn't look
visually inconsistent), then return `"greet"`. Noted limitation, accepted
rather than engineered around: this message is a bare terminal side effect,
not part of `state["messages"]` — it won't appear in a LangSmith trace and
wouldn't be visible to any future non-CLI frontend. Acceptable for this
CLI-only project.

**Acceptance criteria:**
- [x] A failed classification call prints a clear diagnostic line and
      falls back to routing `"greet"`, without raising
- [x] Formatting matches the existing interrupt-message style in `cli.py`

**Verification:**
- [x] `tests/test_chat.py` — a mocked `ChatAnthropic`/`invoke` raising
      results in a `"greet"` route and no unhandled exception (assert on
      the print via `capsys` or equivalent)

**Dependencies:** None

**Files likely touched:** `src/agentic_secretary/chat.py`,
`tests/test_chat.py`

**Estimated scope:** Small

---

### Task 15: `_analyze_email` failure handling

**Description:** Wrap the per-email `structured_llm.invoke(prompt)` call
(inside `_find_email_conflicts`'s loop, called from `detect_actions`) in a
plain try/except (0 retries). On failure for a given email, skip it
(contribute no action item from it) and continue processing the remaining
emails — same shape as the existing skip-on-malformed-reference precedent
already in this function.

Revised from the original silent-skip design: skipping without any visible
note has the same "confidently wrong from partial data" shape Task 13
exists to prevent (there, a half-fetched turn never reaching
`detect_actions`; here, a half-analyzed batch reported as if it were
complete) — cheap enough to fix that there's no real reason to accept the
risk. `detect_actions` collects each failing email's subject into a new
`failed_emails: list[str]` field on `PlannerState` (`state.py`).
`detection.py` exposes `_failed_email_note(failed_emails: list[str]) -> str
| None` (`None` when the list is empty) so the formatting logic exists
once, not twice. Both terminal nodes for a turn append it when present:
`review.py`'s `review()`, alongside its existing `_collision_note()`, before
the fixed disclaimer; and `graph.py`'s `no_action_items()`, appended to its
otherwise-static message — covering the all-emails-failed-and-nothing-else-
found case, which wouldn't reach `review` at all. Both already end in an
`interrupt(...)` call whose value is what `cli.py:55` prints, so no new
print call is needed anywhere.

**Acceptance criteria:**
- [ ] A failure classifying one email doesn't raise and doesn't stop the
      rest of the batch from being processed normally
- [ ] The failed email contributes zero entries to `action_items`, but its
      subject is recorded in `state["failed_emails"]`
- [ ] `review()`'s display includes a note naming the failed email(s) when
      `action_items` is non-empty; `no_action_items()`'s message includes
      the same note when it is empty
- [ ] No note is appended, in either node, when `failed_emails` is empty

**Verification:**
- [ ] `tests/test_detection.py` — a mocked `_analyze_email` call raising
      for one email in a multi-email fixture still returns correct action
      items for the others, and that email's subject appears in
      `failed_emails`
- [ ] `tests/test_review.py` — `review()`'s display includes the note when
      `failed_emails` is non-empty, and omits it when empty
- [ ] `tests/test_graph.py` — a fixture with empty `action_items` and a
      non-empty `failed_emails` routes to `no_action_items` and its
      interrupt value includes the note

**Dependencies:** None (uses the existing `no_action_items` node from Task
12 and the existing `_collision_note`-style append pattern in `review.py`
from Task 8.4; doesn't block on either)

**Files likely touched:** `src/agentic_secretary/detection.py`,
`src/agentic_secretary/state.py`, `src/agentic_secretary/review.py`,
`src/agentic_secretary/graph.py`, `tests/test_detection.py`,
`tests/test_review.py`, `tests/test_graph.py`

**Estimated scope:** Medium

---

### Task 16: `resolution.agent()` failure handling

**Description:** Wrap `agent()`'s `llm_with_tools.invoke(...)` call in a
plain try/except (0 retries). On failure, instead of letting the exception
propagate: call `review.py`'s existing `_latest_proposals(state["messages"])`
helper (exported/reused, not duplicated) to find whatever proposals/drafts
already exist from earlier tool calls this turn, and construct a plain-text
`AIMessage` describing them honestly (e.g. "Before running into an error, I
managed to: propose moving Client Sync to 2pm, draft a reply to Alex. I
couldn't get to the rest — want me to try again?"), or a simpler "I ran into
an error and wasn't able to do anything this turn — want to try again?" if
nothing had been done yet. This message carries no tool calls, so it flows
through the *already-existing* `tools_condition` → `review` path exactly
like a normal end-of-turn summary — zero new graph nodes or edges needed
for this fix.

**Acceptance criteria:**
- [ ] A failure after zero prior tool calls this turn produces a plain "I
      ran into an error, nothing was done" message
- [ ] A failure after one or more successful tool calls this turn produces
      a message that accurately lists what was actually completed, reusing
      `_latest_proposals` rather than re-deriving that logic
- [ ] The resulting message has no tool calls attached, so it flows to
      `review` through the existing `tools_condition` routing unchanged

**Verification:**
- [ ] `tests/test_resolution.py` — a mocked LLM call raising after 0 and
      after 2+ prior successful tool-call rounds each produce the
      corresponding message shape, verified against `_latest_proposals`'s
      actual output rather than a hand-duplicated check

**Dependencies:** None (uses `review.py`'s existing helper as a dependency,
not a new one being built here)

**Files likely touched:** `src/agentic_secretary/resolution.py`,
`src/agentic_secretary/review.py` (export `_latest_proposals` if not
already accessible), `tests/test_resolution.py`

**Estimated scope:** Medium

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 0 retries means a genuinely transient blip (rare, but possible) surfaces as a visible failure instead of being silently absorbed | Low | Accepted tradeoff — the conversational retry loop already covers it at zero extra engineering cost; revisit only if live use shows transient failures are frequent enough to be annoying |
| `classify_intent`'s bare `print()` diverges from the state-based messaging pattern everywhere else | Low | Documented explicitly here rather than silently inconsistent; acceptable because this project has no non-CLI frontend today |
| `_analyze_email` failure hiding a real, important email if it happens to be the one that fails | Resolved | Originally accepted as a Low-Medium silent-skip risk (Task 10 precedent: malformed items already skip quietly); superseded before implementation once the same "confidently wrong from partial data" concern that motivates Task 13 was recognized here too. Task 15 now surfaces a `failed_emails` note via `review`/`no_action_items` instead of staying silent, so a missed email is visible to the human every turn rather than an accepted theoretical gap |

## Open Questions

None outstanding — all four fixes' designs were settled in discussion
before this plan was written, including the specific reasoning for why
each of the three LLM call sites is handled differently rather than
sharing one generic wrapper.

---

# Implementation Plan: AI Secretary — Milestone 2 (RAG: Policy Knowledge Base)

## Overview

Adds a retrieval-augmented capability to the AI secretary, building on
Milestone 1.5's hardened graph: a synthetic corpus of advisory scheduling
policies (leave types, meeting norms, expense categories), sized and
written so some topics are deliberately adjacent/overlapping — large enough
that plain cosine-similarity retrieval starts confusing similar sections,
the same gap identified in a prior tutorial-style RAG project
(`client-onboarding-rag-demo`) that used only top-n cosine search despite a
47-page corpus that structurally called for more.

**Explicitly out of scope:** policies that are themselves scheduling
constraints (e.g. "must work in-office on Fridays"). Those aren't prose to
retrieve and cite — they're deterministic, checkable facts (day-of-week
cross-referenced against real event dates) that would need a structured
constraint model and a new deterministic detection-layer check, the same
"don't trust the LLM with computable facts" principle already enforced
elsewhere in this codebase (the weekday-name ban in `resolution.py`'s
system prompt; deterministic overlap checks in `detection.py`). Building
that properly would roughly double this milestone's scope and shift its
focus from demonstrating retrieval technique to building a rules engine —
a clean candidate for a future milestone, not this one.

This milestone demonstrates **both** RAG control-flow architectures from
the field's standard taxonomy (2-Step / Agentic / Hybrid), sharing one
retrieval engine underneath:

- **Agentic RAG** — `search_policies` is bound into `resolution.agent()`
  alongside `propose_event`/`draft_reply`/`withdraw_proposal`; the agent
  decides mid-reasoning whether an action item warrants a policy check.
  Guaranteed to actually exercise, not just possibly exercise, via a new
  seeded email pattern (`policy_question`, Task 18/21): an email that asks
  a question requiring policy knowledge to answer, in both a
  found-a-relevant-policy and no-relevant-policy-exists variant.
- **2-Step RAG** — a direct chat question ("what's our policy on X?") is
  classified by `classify_intent` (already hardened in Milestone 1.5's Task
  14) and routed straight to a new `answer_policy_question` node: retrieve,
  generate, done.

Retrieval is **hybrid** (dense + sparse, fused) plus a **reranking** pass —
deliberately built because the corpus is sized to actually need them, not
left as documented-but-unexercised choices.

OpenAI's `text-embedding-3-small` supplies dense embeddings (Anthropic has
no embeddings endpoint); Claude remains the only LLM used for
reasoning/generation — OpenAI stays scoped to embeddings only.

## Dependency Graph

```
[Milestone 1.5 complete] ──────────────────────────────────────┐
                                                                 │
Config + deps (Task 17) ───────────────────┐                   │
                                             │                   │
Policy corpus + policy_question email       │                   │
  scenarios (Task 18)                       │                   │
    │                                       │                   │
    └── Ingestion module (Task 19) ◄────────┘                   │
            │                                                    │
            └── search_policies: hybrid + BGE rerank (Task 20)   │
                    │                                            │
                    ├── Live retrieval smoke test (Checkpoint)   │
                    │                                            │
                    ├── PolicyQuestionEmail detection (Task 21) ◄─┘
                    │       │
                    │       └── Bind into resolution.agent() —
                    │             agentic RAG (Task 22)
                    │
                    └── 3-way classify_intent + answer_policy_question
                          — 2-step RAG (Task 23, builds on Task 14)
                                    │
                                    └── Live verification, both paths (Task 24)
                                            │
                                            ├── Resilience: 0-retry catch/fallback
                                            │     for OpenAI/Chroma/BGE (Task 25)
                                            ├── Retrieval eval: hybrid+rerank
                                            │     vs. cosine-only baseline (Task 26)
                                            ├── Agent + chat eval scenarios (Task 27)
                                            ├── Full test/lint pass (Task 28)
                                            └── README/spec update (Task 29)
```

## Architecture Decisions

- **Chroma Cloud, not self-hosted Chroma or Weaviate** — Chroma Cloud ships
  real native hybrid search (BM25 sparse indexing computed server-side via
  a `Bm25EmbeddingFunction`, fused with dense OpenAI embeddings via a
  `Rrf()` ranking function) with zero local infrastructure — just
  `CHROMA_API_KEY`/`CHROMA_TENANT`/`CHROMA_DATABASE` and
  `chromadb.CloudClient()`, the same API-key pattern already used for
  Anthropic/OpenAI/LangSmith. Self-hosted Chroma doesn't have this feature
  today (confirmed: *"Search API is available in Chroma Cloud only. Future
  support on single-node Chroma is planned"*) — hybrid search there would
  mean hand-rolling BM25 plus manual Reciprocal Rank Fusion. Weaviate offers
  the same native hybrid search fully self-hosted, but only via Docker
  Compose (Weaviate Embedded, which avoids Docker, is Linux/macOS only,
  unsupported on this project's Windows environment) — more operational
  surface for less benefit than Chroma Cloud's API-key-only setup.
- **Deliberate, scoped exception to "local-only"** — the policy corpus
  (text + embeddings) lives on Chroma's managed cloud service. Scoped
  *only* to the vector store: Gmail/Calendar access, the burner account,
  and all LLM reasoning stay exactly as local as Milestone 1 left them.
  Acceptable because the corpus is synthetic policy text, not sensitive.
- **BGE-Reranker-v2-m3 (local, via `FlagEmbedding`), not a Claude-based
  LLM reranker** — chosen for latency and cost: a local cross-encoder
  avoids a network round-trip and per-call API cost on every rerank, at
  the cost of new, heavier dependencies (`torch`, `transformers`/
  `FlagEmbedding`) and a one-time ~1-2GB model weight download. No
  reranking capability exists in Chroma (self-hosted or Cloud) or in
  OpenAI's API (confirmed — no `/rerank` endpoint anywhere in their API
  reference), so this is hand-built regardless of provider choice.
- **Corpus deliberately sized to need hybrid + rerank** — includes at
  least two pairs of genuinely adjacent topics (e.g. two different leave
  policies) so the retrieval eval (Task 26) can show a measured precision
  difference between cosine-only and hybrid+rerank, demonstrating the
  technique rather than just describing it.
- **Chunking: one file = one chunk, no splitter** — each policy doc is
  authored short and single-topic, so the file boundary already is the
  correct semantic boundary. No fixed-size splitting (would cut a policy
  statement in half or fragment an intentionally-adjacent pair into noise
  the retrieval eval didn't intend to test), no overlap parameter (nothing
  to lose context across, since there's no arbitrary cut point), no
  hierarchical/parent-child chunking (no larger "parent" exists beyond the
  already-short document itself) and no category/section metadata (doesn't
  help disambiguate the deliberately-overlapping pairs, since both members
  of a pair would share the same category — that disambiguation is hybrid
  search + reranking's job specifically).
- **Two RAG architectures, one retrieval engine** — `rag.py` owns
  chunking, hybrid retrieval, and reranking as shared logic; both
  `resolution.agent()` (agentic) and `answer_policy_question` (2-step) call
  into it rather than each reimplementing retrieval.
- **New `ActionNeeded` variant: `PolicyQuestionEmail`** — a fifth pattern
  alongside `calendar_overlap`/`back_to_back`/`email_conflict`/
  `reschedule`/`mentions`, for an email that asks a question a policy
  document would answer. Fields: `kind`, `description`, `email` — no
  `events`, since it's not about calendar state. Detected by extending the
  existing `_analyze_email` LLM classification with a fifth category, not
  a separate detection pass. Seeded in both a found-relevant-policy and
  no-relevant-policy variant (Task 18), so the agent's grounding/refusal
  behavior is guaranteed to be exercised, not merely possible — mirroring
  how Milestone 1's four conflict patterns were each guaranteed real
  fixture coverage rather than left to chance.
- **`classify_intent` becomes a 3-way classification** — `_Intent` changes
  from `wants_conflict_check: bool` to `intent: Literal["check_conflicts",
  "policy_question", "other"]`, matching this codebase's existing
  preference for Pydantic discriminated types over ambiguous independent
  booleans (`state.py`'s `ActionNeeded` union). The same `classify_intent`
  function is attached as the conditional-edge router for `greet`,
  `answer_policy_question`, `fetch_failed`, and `no_action_items` — every
  node that ends a turn by capturing a fresh reply routes through it
  directly, never back through `greet`'s fixed prompt (which would
  double-prompt). Builds on Milestone 1.5 Task 14's already-hardened
  version of this function — the 0-retry catch/print/fallback behavior
  carries forward unchanged under the new 3-way schema.
- **New dependencies:** `langchain-openai`, `langchain-chroma` + `chromadb`
  (Cloud client), `torch` + `transformers`/`FlagEmbedding` (BGE reranker).
  Flagging per the Milestone 1 spec's "ask first before adding dependencies
  beyond the tech stack" boundary.
- **New external prerequisite:** a Chroma Cloud account, database, and API
  key — same category of manual setup step as Milestone 1's Google Cloud
  OAuth client and burner account.
- **Resilience follows Milestone 1.5's rule exactly:** 0 retries, plain
  try/except per call site (OpenAI embedding calls, Chroma Cloud network
  calls, the local BGE inference call), immediate fallback — no second
  resilience pattern invented for the new provider surface.

## Task List

### Phase C: RAG Foundation

- [ ] Task 17: Config + dependencies
- [ ] Task 18: Synthetic policy corpus + `policy_question` email scenarios

### Checkpoint: Foundation
- [ ] `uv sync` succeeds with all new dependencies; `uv run ruff check .`
      clean
- [ ] `uv run python -c "from agentic_secretary.config import settings; print(settings.openai_api_key is not None, settings.chroma_api_key is not None)"` runs against a real `.env`
- [ ] Manual: a Chroma Cloud database exists and is reachable via
      `chromadb.CloudClient()`
- [ ] Policy corpus includes at least two genuinely adjacent/overlapping
      topic pairs; `seed_data/emails.yaml`/`relations.yaml` include both a
      found-relevant-policy and no-relevant-policy `policy_question` email
- [ ] Review with human before proceeding

### Phase D: Retrieval Walking Skeleton

- [ ] Task 19: Ingestion module (`rag.py`)
- [ ] Task 20: `search_policies` — hybrid retrieval + BGE rerank

### Checkpoint: Retrieval proven end-to-end
- [ ] Live smoke test (manual, real Chroma Cloud + OpenAI + local BGE): an
      ambiguous query matching one of the two overlapping policy docs
      returns the correct one
- [ ] `tests/test_rag.py` passes with mocked embeddings/Chroma
      client/reranker — no live network calls or model downloads in the
      automated suite
- [ ] Review with human before agent/chat integration

### Phase E: Detection + Agent + Chat Integration (both RAG architectures)

- [ ] Task 21: `PolicyQuestionEmail` detection
- [ ] Task 22: Bind `search_policies` into `resolution.agent()` (agentic RAG)
- [ ] Task 23: 3-way `classify_intent` + `answer_policy_question` (2-step RAG)
- [ ] Task 24: Live verification, both paths

### Checkpoint: Core RAG flow
- [ ] Seeded `policy_question` email (found variant): agent calls
      `search_policies`, drafts a reply citing the policy
- [ ] Seeded `policy_question` email (not-found variant): agent correctly
      states it found no relevant policy rather than fabricating an answer
- [ ] Direct chat policy question routes to `answer_policy_question`
      without touching the conflict-detection pipeline at all
- [ ] Review with human before hardening phase

### Phase F: Production-Grade Hardening

- [ ] Task 25: Resilience for OpenAI/Chroma/BGE calls
- [ ] Task 26: Retrieval evaluation — hybrid+rerank vs. cosine-only baseline
- [ ] Task 27: Extend eval suites

### Checkpoint: Hardening complete
- [ ] Retrieval eval shows a measured precision difference between
      cosine-only and hybrid+rerank on the ambiguous-topic queries
- [ ] A failure-injection test (mocked Chroma/embedding/reranker call
      raising) proves immediate, clean fallback on both the agentic and
      2-step paths, consistent with Milestone 1.5's 0-retry rule
- [ ] Review with human before final polish

### Phase G: Polish

- [ ] Task 28: Full test suite + lint pass
- [ ] Task 29: README + spec update

### Checkpoint: Complete
- [ ] All acceptance criteria in this plan are met
- [ ] `uv run pytest -m "not llm_eval"` and `uv run ruff check .` both pass
- [ ] README walkthrough for Milestone 2 verified end-to-end
- [ ] Ready for human review / demo

## Task Details

### Task 17: Config + dependencies

**Description:** Add `langchain-openai`, `langchain-chroma`, `chromadb`,
`torch`, `transformers` (or `FlagEmbedding`) to `pyproject.toml`. Extend
`config.py`'s `Settings` with `openai_api_key: str | None`,
`embedding_model_name: str` (default `"text-embedding-3-small"`),
`chroma_api_key: str | None`, `chroma_tenant: str | None`,
`chroma_database: str | None`, `reranker_model_name: str` (default
`"BAAI/bge-reranker-v2-m3"`). Add all new env vars to `.env.example`.

**Acceptance criteria:**
- [ ] `pyproject.toml` lists all new dependencies
- [ ] `Settings` exposes all new fields, reading from the corresponding
      env vars
- [ ] `.env.example` documents each with a comment on where to get it

**Verification:**
- [ ] `uv sync` succeeds
- [ ] `tests/test_config.py` covers the new settings fields (default + env
      override)

**Dependencies:** None

**Files likely touched:** `pyproject.toml`, `src/agentic_secretary/config.py`,
`.env.example`, `tests/test_config.py`

**Estimated scope:** Small

---

### Task 18: Synthetic policy corpus + `policy_question` email scenarios

**Description:** Author `seed_data/policies/*.md` (8-15 short,
single-topic, advisory-only documents — explicitly no time/day-of-week
scheduling constraints), including at least two pairs of genuinely
adjacent topics distinguishable only by a specific checkable detail. Add a
new `policy_question` relation kind to `relations.yaml` (fields: `email`
only, zero events) and two new emails to `emails.yaml`: one asking
something the corpus actually covers, one asking something entirely
outside the corpus.

**Acceptance criteria:**
- [ ] At least two adjacent/overlapping document pairs exist, each
      distinguishable only by a specific detail, not by an obviously
      different subject
- [ ] At least a few documents remain clearly distinct
- [ ] One `policy_question` email is answerable from the corpus; one is not
- [ ] `relations.yaml`'s `policy_question` kind is validated by the
      existing loader (`seed_data.py`) alongside the four Milestone 1 kinds

**Verification:**
- [ ] `tests/test_seed_data.py` extended: `policy_question` kind parses and
      validates like the existing five kinds
- [ ] Manual read-through confirming the overlapping pairs and the
      found/not-found email pair are genuinely unambiguous test cases

**Dependencies:** None (parallel with Task 17)

**Files likely touched:** `seed_data/policies/*.md`, `seed_data/emails.yaml`,
`seed_data/relations.yaml`, `tests/test_seed_data.py`

**Estimated scope:** Medium

---

### Task 19: Ingestion module (`rag.py`)

**Description:** New `src/agentic_secretary/rag.py`. `build_policy_index()`
lazily gets-or-creates a Chroma Cloud collection (`chromadb.CloudClient()`)
configured with a dense embedding function (`OpenAIEmbeddings`) and a
sparse `Bm25EmbeddingFunction`, then loads each `seed_data/policies/*.md`
file as exactly one chunk (no splitter) and upserts by a stable id derived
from filename (idempotent — re-running doesn't duplicate). No
Chroma/OpenAI-touching object constructed at import time, matching the
lazy-construction convention already used for `ChatAnthropic`.

**Acceptance criteria:**
- [ ] `build_policy_index()` returns a Chroma collection populated from the
      real corpus, one chunk per file, configured for both dense and
      sparse indexing
- [ ] Re-running ingestion doesn't create duplicate entries
- [ ] Each chunk retains its source filename as metadata

**Verification:**
- [ ] `tests/test_rag.py` mocks the Chroma client and `OpenAIEmbeddings`;
      asserts one chunk per file, metadata carries the filename, upsert
      uses stable ids, no real network call happens

**Dependencies:** Task 17, Task 18

**Files likely touched:** `src/agentic_secretary/rag.py` (new),
`tests/test_rag.py` (new)

**Estimated scope:** Medium

---

### Task 20: `search_policies` — hybrid retrieval + BGE rerank

**Description:** `search_policies(query: str, k: int = 3) -> str` lazily
builds/caches the collection, runs Chroma Cloud's hybrid search (dense
`Knn` + sparse `Knn`, fused via `Rrf`) to pull ~10-20 candidates, reranks
via a lazily-loaded `FlagReranker(settings.reranker_model_name)` down to the
final top-k, formats results with source citations
(`"[source: sick-leave.md] ..."`), or returns a distinct `"No relevant
policy found."` string below a relevance threshold. Wrapped as
`search_policies_tool = tool(search_policies)`.

**Acceptance criteria:**
- [ ] Hybrid retrieval pulls a wider candidate set than `k`, combining
      dense + sparse via `Rrf`
- [ ] BGE reranking narrows/reorders to the final top-k
- [ ] Results include source document names
- [ ] Below-threshold case returns the sentinel string, never an exception
      or empty string
- [ ] `search_policies_tool` exposes `.name`/`.description` precise enough
      for the agent to know when to call it

**Verification:**
- [ ] `tests/test_rag.py` extended: mocked hybrid results + mocked reranker
      score; assert citation formatting and the no-match sentinel path
- [ ] Live smoke test (manual, real services) per this phase's checkpoint

**Dependencies:** Task 19

**Files likely touched:** `src/agentic_secretary/rag.py`, `tests/test_rag.py`

**Estimated scope:** Medium

---

### Task 21: `PolicyQuestionEmail` detection

**Description:** Add `PolicyQuestionEmail(kind, description, email)` to
`state.py`'s `ActionNeeded` union. Extend `_analyze_email` in `detection.py`
with a fifth classification category (alongside `email_conflict`/
`reschedule`/`mentions`) recognizing an email that asks a question a policy
document would answer. Milestone 1.5's Task 15 per-email failure handling
(skip on error) applies unchanged to this new category too.

**Acceptance criteria:**
- [ ] `PolicyQuestionEmail` is a discriminated variant with `email` only,
      no `events` field
- [ ] `_analyze_email` correctly classifies both Task 18 seeded emails
      (found and not-found variants) as `policy_question`
- [ ] Existing four-pattern detection behavior is unchanged

**Verification:**
- [ ] `tests/test_detection.py` extended: both seeded `policy_question`
      emails are correctly detected; existing conflict-pattern tests still
      pass unchanged

**Dependencies:** Task 18

**Files likely touched:** `src/agentic_secretary/state.py`,
`src/agentic_secretary/detection.py`, `tests/test_detection.py`

**Estimated scope:** Medium

---

### Task 22: Bind `search_policies` into `resolution.agent()` (agentic RAG)

**Description:** Add `search_policies_tool` to `_make_bound_tools`. Extend
`SYSTEM_PROMPT`: for a `PolicyQuestionEmail` item, call `search_policies`
and draft a reply grounded in what it finds (or state plainly that no
policy applies, per the not-found seeded scenario); for any other action
item whose resolution could plausibly be affected by a policy, consult it
before deciding and cite it in the summary.

**Acceptance criteria:**
- [ ] `search_policies_tool` bound alongside the three existing tools
- [ ] System prompt covers `PolicyQuestionEmail` handling explicitly, plus
      the general "consult if plausibly relevant" case for other item
      kinds
- [ ] Existing Milestone 1 propose/draft/withdraw behavior unchanged

**Verification:**
- [ ] `tests/test_resolution.py` extended: a `PolicyQuestionEmail` item in
      context leads to a mocked `search_policies` call routed correctly
      through the existing `tools_condition` loop

**Dependencies:** Task 20, Task 21

**Files likely touched:** `src/agentic_secretary/resolution.py`,
`tests/test_resolution.py`

**Estimated scope:** Small

---

### Task 23: 3-way `classify_intent` + `answer_policy_question` (2-step RAG)

**Description:** Change `_Intent` from `wants_conflict_check: bool` to
`intent: Literal["check_conflicts", "policy_question", "other"]`, on top
of Task 14's already-hardened (0-retry, print, fallback) version of
`classify_intent`. New `answer_policy_question(state) -> dict` in
`chat.py`: calls `search_policies`, generates a grounded answer (citing the
source, or stating no policy applies), shows it via `interrupt()`, returns
`{"messages": [HumanMessage(content=reply)]}`. `graph.py` adds the node and
attaches `classify_intent` as the router for both `greet` and
`answer_policy_question`.

**Acceptance criteria:**
- [ ] `_Intent` is a 3-way discriminated classification
- [ ] `classify_intent` routes all three cases correctly, preserving Task
      14's failure-handling behavior unchanged
- [ ] `answer_policy_question` shows a real generated (not fixed) answer
      via `interrupt()`
- [ ] `graph.py` wires the node and router correctly, without re-entering
      `greet`'s own interrupt in between

**Verification:**
- [ ] `tests/test_chat.py` extended: mocked classification covers all
      three branches; `answer_policy_question` tested with mocked
      retrieval/generation
- [ ] `tests/test_graph.py` asserts the new node/edges

**Dependencies:** Task 20

**Files likely touched:** `src/agentic_secretary/chat.py`,
`src/agentic_secretary/graph.py`, `tests/test_chat.py`, `tests/test_graph.py`

**Estimated scope:** Medium

---

### Task 24: Live verification (both paths)

**Description:** Re-run representative scenarios against the finished
integration.

**Acceptance criteria:**
- [ ] Agentic path: the found-variant `policy_question` email produces a
      drafted reply citing the correct policy; the not-found variant
      produces a reply plainly stating no policy applies
- [ ] Agentic-path control: a scheduling conflict with no applicable
      policy is resolved on judgment alone, correctly reporting no policy
      consulted
- [ ] 2-step path: a direct policy question (including an
      ambiguous/overlapping-topic query) is answered correctly with a
      citation, without touching `fetch_emails`/`check_calendar`/
      `detect_actions`

**Verification:**
- [ ] Manual CLI walkthrough against the seeded burner account and the
      real Chroma Cloud collection, documented

**Dependencies:** Task 22, Task 23

**Files likely touched:** none (manual verification)

**Estimated scope:** Small

---

### Task 25: Resilience for OpenAI/Chroma/BGE calls

**Description:** Same rule as Milestone 1.5: 0 retries, plain try/except
per call site, immediate fallback — no backoff, no new resilience pattern.
Wrap the OpenAI embedding calls, Chroma Cloud network calls (upsert +
hybrid search), and the local BGE inference call. On failure:
index-build/upsert failure raises loudly (the feature is entirely
unavailable this run); a query-time failure inside `search_policies`
(retrieval or reranking) returns a degradation string ("Policy search is
temporarily unavailable; proceeding without it.") instead of raising, so
the agent's turn or the 2-step answer path continues.

**Acceptance criteria:**
- [ ] Index-build/upsert failure raises a clear error
- [ ] Query-time failure (retrieval or reranking) returns a degradation
      string instead of raising, on both the agentic and 2-step paths
- [ ] "No relevant policy found" and "search unavailable" are textually
      distinguishable

**Verification:**
- [ ] `tests/test_rag.py` — mocked Chroma/OpenAI/BGE clients that raise on
      every call prove both failure paths behave as specified
- [ ] `tests/test_chat.py`/`tests/test_resolution.py` — a mocked
      `search_policies` failure doesn't crash either call path

**Dependencies:** Task 19, Task 20, Task 23

**Files likely touched:** `src/agentic_secretary/rag.py`, `tests/test_rag.py`

**Estimated scope:** Small-Medium

---

### Task 26: Retrieval evaluation — hybrid+rerank vs. cosine-only

**Description:** A golden dataset (`evals/policy_retrieval_examples.py`)
of query → expected source document, weighted toward the
overlapping-topic pairs. `tests/test_policy_retrieval_eval.py` (marked
`llm_eval`, excluded from CI) runs each query through both a cosine-only
baseline (dense `Knn` alone) and the real hybrid+rerank pipeline, reporting
precision for each — the concrete before/after this milestone demonstrates.

**Acceptance criteria:**
- [ ] Every overlapping-topic pair has at least one query only the
      specific-detail distinction resolves correctly
- [ ] Eval reports precision separately for cosine-only vs. hybrid+rerank

**Verification:**
- [ ] `uv run pytest -m llm_eval -k policy_retrieval` passes and its
      precision comparison is documented in Task 29's spec update

**Dependencies:** Task 20

**Files likely touched:** `evals/policy_retrieval_examples.py` (new),
`tests/test_policy_retrieval_eval.py` (new)

**Estimated scope:** Medium

---

### Task 27: Extend eval suites

**Description:** Add 2-3 examples to `evals/agent_examples.py` covering the
agentic path (both `policy_question` variants plus a general
conflict-with-policy-context case). Extend `tests/test_chat.py`'s mocked
coverage for the 2-step path with a couple of direct-question scenarios.
Update the LangSmith judge rubric to capture "did it correctly
apply/dismiss the relevant policy."

**Acceptance criteria:**
- [ ] At least 2 new agentic-path examples
- [ ] 2-step path has mocked-CI coverage plus at least one live-verified
      scenario per Task 24

**Verification:**
- [ ] `uv run pytest -m "not llm_eval"` includes and passes the new
      mocked-example checks
- [ ] `tests/test_agent_examples_eval.py` (live, not CI) run once manually

**Dependencies:** Task 22, Task 23

**Files likely touched:** `evals/agent_examples.py`,
`tests/test_agent_examples.py`, `tests/test_chat.py`

**Estimated scope:** Small-Medium

---

### Task 28: Full test suite + lint pass

**Description:** Close any remaining gaps and run Ruff format + check.

**Acceptance criteria:**
- [ ] `uv run pytest -m "not llm_eval"` passes with no failures
- [ ] `uv run ruff check .` passes with no errors

**Verification:**
- [ ] Both commands exit `0`

**Dependencies:** Tasks 17-27

**Files likely touched:** `tests/*`, `src/agentic_secretary/*`

**Estimated scope:** Small

---

### Task 29: README + spec update

**Description:** Document Chroma Cloud + OpenAI setup as new prerequisites
alongside Milestone 1's Google OAuth/burner account. Document how to demo
both RAG paths (a policy-grounded email reply, a direct policy question).
Add a "Milestone 2: RAG" section to `docs/spec/ai-secretary.md` recording:
success criteria, the deliberate local-only exception for the vector
store, the explicit out-of-scope call on time-based policies, and the Task
26 precision comparison.

**Acceptance criteria:**
- [ ] README documents setup and both demo paths
- [ ] Spec doc records success criteria, the local-only exception, the
      time-based-policy scoping decision, and the Task 26 results

**Verification:**
- [ ] Manual read-through / dry run of the documented steps

**Dependencies:** Task 28

**Files likely touched:** `README.md`, `docs/spec/ai-secretary.md`

**Estimated scope:** Small

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chroma Cloud account/database setup friction | Low-Medium | Confirm account + database creation early (Task 17's checkpoint), same treatment as Milestone 1's Google OAuth prerequisite |
| BGE reranker's `torch`/model-download footprint is heavier than anything else in this project | Low-Medium | Deliberate, named tradeoff (latency/cost over install weight) — documented in Architecture Decisions rather than discovered later |
| Corpus's deliberately-overlapping pairs turn out not to actually confuse cosine-only retrieval | Medium | Task 26's eval directly measures this — a clean baseline result is a real, reportable finding, not a failure |
| Sending the policy corpus to Chroma Cloud is a real, scoped departure from "local-only" | Low (documented) | Explicit in Architecture Decisions and Task 29; corpus is synthetic, not sensitive |
| Agent over-relies on `search_policies` for items that don't need it | Low-Medium | System prompt (Task 22) scopes it to plausibly-relevant items; Task 24's control scenario checks against fabricated relevance |

## Open Questions

- Exact wording/topics of the 8-15 policy documents and which pairs are
  made deliberately overlapping (Task 18) — left to implementation time.
- Chroma Cloud pricing tier / free-tier limits for this project's scale —
  confirm during Task 17 setup.
- Whether Task 26's before/after comparison is compelling enough for the
  README's headline demo, or better left as spec-doc detail — decide once
  real numbers exist.
