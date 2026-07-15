# Implementation Plan: AI Secretary — Milestone 1 (Planner)

Implements [`docs/spec/ai-secretary.md`](../docs/spec/ai-secretary.md).

## Overview

Build a LangGraph-orchestrated planner agent that reads a burner Gmail inbox
and Google Calendar, detects scheduling action items against seeded
synthetic data, and — via a chat loop rather than a fixed no-input
pipeline — presents each action item in an open-text turn where the human
(or the agent, if asked to decide) composes a remedy plan — shift the slot,
draft a reply, accept a proposed meeting, any combination of those, or
skip — that's confirmed by the human before anything is generated. Every
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
                    └── Action-detection node (Task 7)
                            │
                            └── Chat remedy loop (Task 8: 8.1–8.8)
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
- **Chat loop with open-turn remedy selection, not a fixed pipeline or a
  numbered menu** (decided before Task 6+, revised 2026-07-15 — see
  `docs/spec/ai-secretary.md`'s Action Response Behavior section) — the
  agent greets the user and waits for free text (e.g. "check for
  conflicts") rather than running `fetch_emails → check_calendar →
  detect_actions` unconditionally. After `detect_actions`, each action item
  gets an open-text turn (`present_item`) offering suggested remedies as
  plain text rather than a forced numbered choice; an LLM (`propose_plan`)
  turns the human's reply — including "you decide" — into a remedy
  *subset* (multiple remedies per item are allowed, e.g. shift **and**
  draft together) plus an optional explicit target time; a deterministic
  (non-LLM) overlap check and human confirmation (`confirm_plan`) happen
  *before* any generation, not after. All remedies stay propose-only — no
  write-capable tool is added for milestone 1. Revised from the original
  fixed-menu design after live testing showed it couldn't express "shift
  AND draft" for one item, and silently produced duplicate, contradictory
  Gmail drafts when a single email conflicted with two calendar events (see
  Task 8.1).
- **`EmailConflict` is multi-event** — an email's proposed meeting can
  overlap more than one existing calendar event (e.g. a proposed 9:15am
  slot overlapping both a 9:00 standup and a 9:15 client call);
  `EmailConflict` carries `events: list[CalendarEvent]`, matching
  `CalendarOverlapConflict`/`BackToBackConflict`'s existing list-of-events
  shape, instead of one `EmailConflict` per overlapping event. Confirmed
  necessary by a live CLI run producing two separate, contradictory Gmail
  drafts on the same email thread under the earlier singular-`event`
  design.

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
- [ ] Task 8: Chat remedy loop (open-turn + confirm-before-generate)
  - [ ] Task 8.1: `EmailConflict` becomes multi-event
  - [ ] Task 8.2: State shape for the open-turn remedy flow
  - [ ] Task 8.3: `present_item` node (open-text turn)
  - [ ] Task 8.4: `propose_plan` node (LLM plan + multi-remedy)
  - [ ] Task 8.5: `confirm_plan` node (deterministic overlap warning + confirmation gate)
  - [ ] Task 8.6: `content_generation` rewrite (queue processing + `accept_meeting`)
  - [ ] Task 8.7: Graph wiring + CLI display
  - [ ] Task 8.8: Spec + plan documentation update

### Checkpoint: Core Agent Flow
- [ ] End-to-end CLI run against the seeded account: greet → free-text
      "check for conflicts" → fetch → detect action items → open-text
      remedy turn per item, confirmed before generation
- [ ] All conflict-pattern tests pass against fixtures, including the
      multi-event `EmailConflict` case (one email overlapping two events)
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
it's triggered (from a chat turn) and what happens with its output.

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
`langchain`/`langgraph`). LLM-calling nodes added in Task 8 follow this same
pattern.

**Amendment (Task 8.1):** `_find_email_actions`'s original implementation
appended one `EmailConflict` per overlapping calendar event, discovered via
live testing to produce duplicate/contradictory Gmail drafts when one email
overlapped two events. See Task 8.1 below — `EmailConflict` is now
multi-event (`events: list[...]`), grouped per email.

---

### Task 8.1: `EmailConflict` becomes multi-event

**Description:** Fixes a live-confirmed bug: `_find_email_actions`
currently appends one `EmailConflict` per calendar event a proposed meeting
overlaps (a `for event in calendar_events` loop inside the
`proposes_new_meeting` branch), so a single email overlapping two events
produces two independent `EmailConflict` items. A live CLI run against the
seeded fixtures ("Quick sync tomorrow?" overlapping both "Team Standup" and
"Client Sync") confirmed this produces two separate, contradictory Gmail
drafts on the same email thread once both got resolved via `draft_reply`.
`EmailConflict.event: CalendarEvent` (singular) becomes
`events: list[CalendarEvent]`, matching the list-of-events shape
`CalendarOverlapConflict`/`BackToBackConflict` already use;
`_find_email_actions` collects all overlapping events for an email before
appending a single `EmailConflict`.

**Acceptance criteria:**
- [ ] `EmailConflict` has an `events: list[tools.CalendarEvent]` field, not
      `event`
- [ ] An email whose proposed meeting overlaps two or more calendar events
      produces exactly one `EmailConflict` action item listing all of them,
      not one item per overlap
- [ ] `_event_by_id`'s multi-event `isinstance` check includes
      `EmailConflict` alongside `CalendarOverlapConflict | BackToBackConflict`

**Verification:**
- [ ] `tests/test_actions.py` gains a case with two overlapping calendar
      events against one proposed-meeting email, asserting exactly one
      `EmailConflict` with `len(events) == 2`
- [ ] Existing `EmailConflict`-related tests updated for the field rename;
      `uv run pytest` passes

**Dependencies:** None (amends Task 7's already-shipped code)

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_actions.py`

**Estimated scope:** Small

---

### Task 8.2: State shape for the open-turn remedy flow

**Description:** Replace `PlannerState`'s `pending_resolution:
ActionResolution | None` with the fields the new multi-remedy queue needs:
`pending_remedies: list[RemedyLiteral]`, `pending_shift_event_ids:
list[str]`, `pending_explicit_time: datetime | None`,
`pending_plan_summary: str`. Extend the remedy `Literal` type (shared by
`ActionResolution.remedy` and the new fields) with `"accept_meeting"`.
Extend `_applicable_remedies` so `EmailConflict` returns `{shift_slot,
draft_reply, accept_meeting, skip}` — the other three kinds are unchanged
(`CalendarOverlapConflict`/`BackToBackConflict`: `{shift_slot, skip}`;
`RescheduleRequest`: `{shift_slot, draft_reply, skip}`) since only an
`EmailConflict` carries a proposed new meeting to accept.

**Acceptance criteria:**
- [ ] `PlannerState.__annotations__` reflects the new fields, no
      `pending_resolution`
- [ ] `_applicable_remedies` returns the correct remedy set per
      `ActionNeeded` kind, including `accept_meeting` for `EmailConflict`
      only

**Verification:**
- [ ] Update `test_planner_state_has_expected_fields` for the new field set
- [ ] New `tests/test_graph.py` cases for `_applicable_remedies` per kind

**Dependencies:** Task 8.1

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Small

---

### Task 8.3: `present_item` node (open-text turn)

**Description:** Replaces `present_menu`'s forced numbered-choice
`interrupt()` (and its second conditional "which event to shift"
sub-interrupt) with an open turn: shows the action item's description and
its applicable remedies as plain human-readable text (not a numbered list),
and accepts a free-text reply (e.g. "shift and draft", "you decide", "shift
it to 3pm"). This node does no interpretation — it only displays the item
and captures the raw reply for `propose_plan`. Unconditional edge to
`propose_plan`.

**Acceptance criteria:**
- [ ] Interrupting on an action item shows its description and the
      applicable remedies as text, not a numbered menu
- [ ] The free-text reply is captured into state for `propose_plan` to read
- [ ] `present_item` performs no remedy interpretation itself (no branching
      on the reply's content)

**Verification:**
- [ ] `tests/test_graph.py` case asserting the interrupt payload contains
      the item description and remedy labels (including "accept the
      meeting" only when the item is an `EmailConflict`)

**Dependencies:** Task 8.2

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Small

---

### Task 8.4: `propose_plan` node (LLM plan + multi-remedy)

**Description:** New LLM node, following the existing structured-output
pattern (`_EmailIntent`/`_ChatIntent`/`_ShiftProposal`/`_ReplyDraft`, all
`with_structured_output(Model, method="json_schema")`). A new
`_ProposedPlan` model captures: `remedies: list[RemedyLiteral]` (the chosen
subset — multi-remedy allowed), `shift_event_ids: list[str]` (which of the
item's event(s) to shift, when `shift_slot` is chosen), `explicit_time:
datetime | None` (a specific time named in the free text, if any),
`summary: str` (plain-text plan description for `confirm_plan` to show).
The prompt gets the item's description, its applicable remedies, and the
human's free-text reply (including a "decide for me" case). The LLM's
`remedies` output is deterministically validated against
`_applicable_remedies(item)` afterward — anything not actually applicable
is stripped, never trusted blindly.

Disambiguation default differs by kind: `CalendarOverlapConflict`/
`BackToBackConflict` are exactly-2-arity pairwise conflicts where shifting
resolves the overlap by moving either one — if the reply doesn't name one,
default to the first event and let `confirm_plan`'s summary make the choice
visible for correction. `EmailConflict` can have more than two events; if
`shift_slot` is chosen and the reply doesn't disambiguate, default to *all*
of the item's events (moving every conflicting event out of the way is what
"shift to make room" means when several things are in the way). Unconditional
edge to `confirm_plan`.

**Acceptance criteria:**
- [ ] A reply naming multiple remedies (e.g. "shift and draft") produces
      both in `pending_remedies`
- [ ] "You decide" produces a plan chosen from the item's applicable
      remedies
- [ ] A reply naming a specific time populates `pending_explicit_time`
- [ ] `remedies` is filtered against `_applicable_remedies(item)` before
      being stored — an LLM response naming an inapplicable remedy never
      reaches `pending_remedies`
- [ ] `shift_event_ids` defaults per kind as described above when the reply
      doesn't disambiguate

**Verification:**
- [ ] `tests/test_graph.py` cases mocking the LLM call (same `@patch`
      pattern as `_classify_intent`/`_analyze_email`) for: multi-remedy
      reply, "you decide", explicit-time extraction, and the
      invalid-remedy-filtered-out case

**Dependencies:** Task 8.3

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Medium — split `shift_event_ids` disambiguation into a
follow-up task if the prompt/validation logic and test matrix grow past a
single focused session

---

### Task 8.5: `confirm_plan` node (deterministic overlap warning + confirmation gate)

**Description:** Displays `pending_plan_summary` via `interrupt()`. If
`pending_explicit_time` is set, deterministically (no LLM call) checks it
against `state["calendar_events"]` and any shift proposals already in
`state["resolutions"]` this session, reusing the same overlap comparison
`_find_calendar_overlaps`/`_busy_times_context` already use, and appends a
plain f-string warning to the displayed summary if it overlaps — advisory
only, never blocking. The human's reply either confirms (routes to
`content_generation`) or rejects/edits (clears all `pending_*` fields,
routes back to `present_item` to re-show the same item).

**Acceptance criteria:**
- [ ] A plan with no explicit time, or an explicit time that doesn't
      overlap anything, shows no warning
- [ ] A plan with an explicit time that overlaps a known event or an
      already-proposed shift shows a warning but can still be confirmed
- [ ] Rejecting/editing clears `pending_*` state and re-enters
      `present_item` for the same item (not the next one)

**Verification:**
- [ ] `tests/test_graph.py` cases for: no-warning path,
      warning-shown-but-confirmed path, rejected-plan-loops-back path

**Dependencies:** Task 8.4

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Small

---

#### Sub-checkpoint: Interaction path (Tasks 8.1–8.5)
- [ ] `uv run pytest` passes for all of `test_actions.py`/`test_graph.py`
- [ ] A full round-trip (`present_item` → `propose_plan` → `confirm_plan`)
      is exercisable via `graph.invoke`/`Command(resume=...)`, even though
      `content_generation` doesn't yet consume the new queue shape —
      acceptable at this checkpoint since Task 8.6 finishes the wiring
- [ ] Review with human before generation logic

---

### Task 8.6: `content_generation` rewrite (queue processing + `accept_meeting`)

**Description:** Currently handles exactly one remedy per call, reading
`state["pending_resolution"]`. Rewritten to pop one remedy at a time off
`state["pending_remedies"]`: `shift_slot` uses the existing
`_generate_shift_proposal`, iterating `pending_shift_event_ids` (one
`ActionResolution` per shifted event, since `ActionResolution` is already
shaped around a single `proposal`); `draft_reply` uses the existing
`_generate_reply_body`/`tools.draft_reply`, unchanged; `accept_meeting` is
new — calls `tools.propose_event(title=item.email.subject,
start=item.proposed_start, duration_minutes=item.proposed_duration_minutes,
existing_event_id=None)` directly, no LLM call, only reachable for
`EmailConflict` items; `skip` is the existing no-op short-circuit,
unchanged. Loops back to itself while `pending_remedies` is non-empty (same
item); once empty, advances `pending_action_index`, clears all `pending_*`
fields, and routes to `present_item` (more items remain) or `END`
(exhausted).

**Acceptance criteria:**
- [ ] A plan with `[shift_slot, draft_reply]` produces two resolutions for
      the same action item
- [ ] `shift_slot` on an `EmailConflict` with multiple `shift_event_ids`
      produces one resolution per shifted event
- [ ] `accept_meeting` produces an `EventProposal` with
      `existing_event_id=None`, using the email's already-known
      `proposed_start`/`proposed_duration_minutes`, and makes no LLM call
- [ ] `skip` still makes no LLM or tool call

**Verification:**
- [ ] `tests/test_graph.py` cases: multi-remedy queue produces multiple
      resolutions; `accept_meeting` resolution asserted via
      `tools.propose_event` call args plus an LLM-call mock's
      `assert_not_called()`; queue-exhaustion routing (mirrors the existing
      `test_content_generation_advances_to_end_when_action_items_exhausted`)

**Dependencies:** Task 8.5

**Files likely touched:** `src/agentic_secretary/graph.py`,
`tests/test_graph.py`

**Estimated scope:** Medium

---

### Task 8.7: Graph wiring + CLI display

**Description:** `build_graph`: remove `present_menu`'s node/edges; add
`present_item`, `propose_plan`, `confirm_plan`; update
`content_generation`'s conditional edges for the new
self-loop-on-nonempty-queue / `present_item` / `END` routing. `cli.py`:
extend its remedy-label display mapping (mirroring `_REMEDY_LABELS` in
`graph.py`) to include `accept_meeting` so resolutions display correctly.

**Acceptance criteria:**
- [ ] `uv run python -m agentic_secretary.cli` runs the full open-turn flow
      end-to-end against the seeded burner account, including the
      multi-event `EmailConflict` scenario (Alex's email vs. Standup +
      Client Sync)
- [ ] Choosing `accept_meeting` on that scenario proposes Alex's meeting as
      a new calendar entry (not just clearing space)
- [ ] No duplicate/contradictory Gmail drafts are created on the same email
      thread

**Verification:**
- [ ] Full `uv run pytest` passes
- [ ] Manual CLI run against seeded data confirming the three acceptance
      criteria above

**Dependencies:** Task 8.6

**Files likely touched:** `src/agentic_secretary/graph.py`,
`src/agentic_secretary/cli.py`

**Estimated scope:** Small

---

### Task 8.8: Spec + plan documentation update

**Description:** Rewrite `docs/spec/ai-secretary.md`'s "Action Response
Behavior" section to describe the open-turn + confirm-before-generate flow
(superseding the fixed-menu description), and update "Success Criteria" to
match (open text instead of "menu of remedies", confirm-before-generate,
multi-remedy, `accept_meeting`). Record the three deferred items below in
the spec's Open Questions.

**Acceptance criteria:**
- [ ] Spec accurately describes the shipped behavior; no stale references
      to a numbered menu remain
- [ ] Deferred items (shared-event-linking, no-thread-to-reply-to for pure
      calendar conflicts, the 10-event fetch window bound) are recorded as
      open/deferred, not silently dropped

**Verification:**
- [ ] Manual read-through of the spec against the actual `graph.py`
      behavior

**Dependencies:** Task 8.7

**Files likely touched:** `docs/spec/ai-secretary.md`

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
| LangGraph interrupt/human-in-the-loop API unfamiliarity | Low-Medium | Prototype the interrupt pattern early in Task 8; consult LangGraph docs via `source-driven-development` if behavior doesn't match expectations |
| Scope creep toward RAG/multi-persona mid-implementation | Low | Explicitly out of scope per `docs/spec/ai-secretary.md` boundaries; flag and defer if tempted mid-task |
| `propose_plan`'s LLM-based remedy/disambiguation interpretation misreads free text | Medium | `remedies` deterministically filtered against `_applicable_remedies` before use (Task 8.4); `confirm_plan`'s human gate (Task 8.5) catches a wrong interpretation before any generation happens |
| Multi-event `shift_slot` on an `EmailConflict` moves more events than the human intended | Low-Medium | Per-kind default documented in Task 8.4; `confirm_plan`'s summary always lists which events are being shifted before confirmation |

## Open Questions

Resolved 2026-07-13: milestone 1 moved from a fixed no-input pipeline to a
chat loop. Resolved 2026-07-15: action response moved again, from a fixed
numbered remedy menu to an open-text turn with LLM-interpreted,
human-confirmed multi-remedy plans (see Architecture Decisions above and
`docs/spec/ai-secretary.md`'s Action Response Behavior section) — live
testing found the fixed-menu design couldn't express "shift AND draft" and
silently produced duplicate, contradictory Gmail drafts when one email
conflicted with two calendar events.

Deferred, not blocking Task 8:
- **Shared-event-linking across different `ActionNeeded` kinds** — e.g. a
  `BackToBackConflict` pair where one event is also independently named in
  a `RescheduleRequest`. Cheap to add later (thread sibling-item context
  into `propose_plan`'s prompt, no schema change needed) — not scheduled.
- **No way to originate a fresh email for a pure calendar-calendar
  conflict** — `CalendarOverlapConflict`/`BackToBackConflict` have no
  attached email/thread, and `tools.draft_reply` requires an existing
  `thread_id` (reply-only, not fresh-compose). Good to have, not needed for
  milestone 1.
- **`confirm_plan`'s overlap check is bounded by the 10-event fetch
  window** — `state["calendar_events"]` comes from one
  `list_upcoming_events(max_results=10)` call at session start and is never
  refreshed; a target date beyond that window would pass the check with
  false confidence on a busier calendar. Good to have, not needed for this
  demo's scale.

Note: Tasks 2, 5, 8 (manual check), and 9 depend on you completing Google
Cloud OAuth client setup and having burner account access ready — these are
prerequisites outside what can be automated and should be confirmed before
implementation begins.
