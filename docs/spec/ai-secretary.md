# Spec: AI Secretary (Portfolio Project) — Milestone 1: Planner

See [`docs/intent/ai-secretary.md`](../intent/ai-secretary.md) for the confirmed
project intent this spec implements. This file also covers
[Milestone 1.5: Resilience & UX Hardening](#milestone-15-resilience--ux-hardening),
which hardens this same graph, and
[Milestone 2: RAG (Policy Knowledge Base)](#milestone-2-rag-policy-knowledge-base),
which adds retrieval on top of it.

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
schedule and inbox, (b) identifies at least one real action item — a time
conflict or a meeting-request/reschedule email — and (c) resolves what it
can on its own judgment using bound tools (propose a new event time, or
draft a reply), then presents one summary of what it did for the human to
review and correct — rather than gating each individual decision behind a
pre-approval menu. Every resolution stays propose-only (a Gmail draft or a
structured event proposal); actually sending the email or booking the slot
is deferred to a later milestone. The full reasoning trace is visible in
LangSmith.

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
│   ├── state.py                # PlannerState TypedDict + ActionNeeded union
│   ├── chat.py                  # greet + classify_intent nodes
│   ├── detection.py            # detect_actions node
│   ├── resolution.py           # agent node: system prompt + bound-tool LLM loop
│   ├── review.py               # review node: summary interrupt + sanity annotation
│   └── graph.py                # LangGraph wiring only: nodes + edges + compile
├── tests/                     # one test module per src/scripts module above
└── docs/
    ├── intent/ai-secretary.md
    └── spec/ai-secretary.md
```

Node/tool files stay single-file (not split into subpackages) while the node
count is small. Task 8 splits what was a single `graph.py` (types +
detection + resolution + wiring) into `state.py`/`detection.py`/
`resolution.py`/`review.py`/`graph.py` along node boundaries, once the node
count reached six (`fetch_emails`, `check_calendar`, `detect_actions`,
`agent`, `tools`, `review`) and `graph.py` was heading toward also holding
the system prompt and the tool-calling/review-interrupt logic on top of
detection — sooner than the originally-anticipated "revisit at milestone 2"
point, but for the same underlying reason (file size/readability), not a
scope change.

## Code Style

- Type-hint all function signatures; no bare `Any` for tool inputs/outputs.
- LangGraph node functions take and return the shared graph state
  (`TypedDict` or Pydantic model), e.g.:

```python
class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # chat turns + the
                                                           # agent's own tool-calling history
    emails: list[EmailSummary]
    calendar_events: list[CalendarEvent]
    action_items: list[ActionNeeded]

def check_calendar(state: PlannerState) -> dict:
    events = tools.list_upcoming_events(calendar_service)
    return {"calendar_events": events}
```

Nodes return a partial dict of only the keys they update — not a full-state
spread — so each field's reducer (default: replace; `add_messages`:
append) decides how it merges. `PlannerState` grows incrementally with the
graph: the Task 6 skeleton (`fetch_emails → check_calendar`) only needs
`emails`/`calendar_events`/`status`; `messages` and `action_items` are added
once the chat loop and the detection/resolution nodes that use them exist
(Task 7/8). There's no separate `resolutions` field or per-item pending
index — what the agent did lives in `messages` (its own tool calls and
results), not a hand-maintained parallel list.

- Tools are thin wrappers around the Google API clients — no business logic
  in the tool layer; reasoning belongs in graph nodes / prompts.
- No inline secrets or hardcoded burner-account identifiers — everything
  sensitive comes from `.env` via `config.py`.

## Testing Strategy

- `pytest` for unit tests.
- `tests/test_tools.py`: mock the Google API clients (no live network calls
  in unit tests) — verify tool functions parse/shape data correctly.
- `tests/test_graph.py`: run the compiled LangGraph app against fixture
  state (not live APIs) and assert on graph wiring/state shape — since
  Task 8, `graph.py` is wiring only, so this file stays scoped to node/edge
  structure, not node internals.
- `tests/test_chat.py` (Task 8): `classify_intent` against a mocked LLM
  response — asserts routing for an in-scope ("check for conflicts") vs.
  out-of-scope free-text turn.
- `tests/test_detection.py` (Task 8 renames `tests/test_conflicts.py` to
  match the `detection.py` split): exercise `detect_actions` (deterministic
  overlap/back-to-back math, plus the LLM-assisted email patterns with
  `_analyze_email` mocked — no live Anthropic calls in the automated suite,
  same "no live API calls" rule as the Google clients) against fixture data
  loaded from the real `seed_data/*.yaml`, not hand-duplicated lookalikes.
- `tests/test_resolution.py` (Task 8): the `agent`/`tools` loop against a
  mocked `ChatAnthropic.bind_tools(...)` response — assert the right tool
  is called with the right (id-bearing) args, and that the loop terminates
  once the mocked response stops returning tool calls.
- `tests/test_review.py` (Task 8): deterministic tests only — the
  exit-vs-continue routing on a fixed set of reply phrases, and the
  collision-annotation helper against synthetic overlapping/non-overlapping
  `EventProposal` fixtures. No LLM involved in this file.
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
  committing node/tool logic changes; handle a new LLM/API call site with
  0 retries, fail-fast, catch-and-fall-back-immediately (see Milestone
  1.5's Failure Handling Behavior) rather than adding a bespoke
  retry/backoff scheme per call site.
- **Ask first:** adding new dependencies beyond the tech stack above;
  changing the default model tier (e.g., away from Haiku) in a way that
  changes cost profile; expanding scope into RAG/vector DB or multi-persona
  support (explicitly milestone 2+, per intent doc).
- **Never do:** commit real credentials, `.env`, `token.json`, or
  `credentials.json`; auto-send emails or auto-book events without a human
  approval step; deploy this publicly/hosted (local-only per intent doc).

## Action Resolution Behavior

Milestone 1 does not prescribe per-pattern draft content (i.e. no fixed
template like "for a calendar-calendar overlap, always draft X"). Instead,
once `detect_actions` finds action items, the agent resolves all of them in
one autonomous pass, using its own judgment about which tool applies to
which item, before presenting a single summary:

- **`propose_event(...)`** — used both to shift an existing event
  (`existing_event_id` set) and to accept a meeting request as a new event
  (`existing_event_id` omitted). Produces a structured `EventProposal`.
  Never calls Calendar's `insert`/`patch`.
- **`draft_reply(...)`** — produces a Gmail draft via `drafts.create`. Never
  calls `send`.
- **Skip** — the agent simply doesn't call a tool for that item, and says so
  in its summary if there's a reason worth surfacing.

Tool calls execute immediately as the agent decides to make them — there is
no pre-execution approval gate, because both tools are already propose-only
by construction (see Boundaries). The human reviews the *result*, not each
individual decision: once the agent has acted on everything it can, it
presents one summary, and the human can request corrections
conversationally (e.g. "move Client Sync to 2pm instead"). The agent
retains full memory of what it already did, so a correction is resolved
against its own prior tool call rather than treated as an unrelated new
request. If the agent is genuinely unsure how to handle an item, it asks
directly instead of guessing, in that same summary-and-reply turn.

This is a deliberate change from an earlier "human picks a remedy from a
fixed menu per conflict" design (see Open Questions): that design gated
every decision behind pre-approval, which added interaction friction
without adding real safety — the propose-only tool boundary is already the
actual safety guarantee, not the human's pre-approval step.

## Success Criteria

- [x] `uv run python scripts/seed_demo_data.py` populates the burner Gmail +
      Calendar with the seeded synthetic scenarios (including at least one
      deliberate time conflict, per the conflict-seeding patterns below).
- [x] `uv run python -m agentic_secretary.cli` opens a chat session against
      the seeded account: the agent greets the user, the user asks it to
      check for conflicts, the agent fetches emails/calendar, detects the
      seeded action items, and resolves them autonomously (proposing event
      shifts/new events or drafting replies as appropriate) before
      presenting one summary for review, per the Action Resolution Behavior
      above.
- [x] No tool the agent calls ever sends an email or creates/patches a
      calendar event — every resolution only ever produces a proposal
      (draft or structured event, never a send/insert/patch call) — and the
      human reviews the full set of resolutions each run before ending the
      session.
- [x] A LangSmith trace exists for the run and shows the node-by-node
      reasoning path.
- [x] `uv run pytest` passes, covering tool-parsing and conflict-detection
      logic against fixture data (no live API calls in the test suite) --
      run as `uv run pytest -m "not llm_eval"`, the marker that excludes
      the one test file that does make live Anthropic calls by design
      (`tests/test_agent_examples_eval.py`, a separate eval suite, not part
      of this criterion's "no live API calls" scope).

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
point is a chat loop rather than a fixed no-input pipeline. Action
resolution was originally spec'd as a human-chosen remedy via menu per
conflict; revised 2026-07-15 to autonomous-resolution-with-review (see
Action Resolution Behavior above) after Task 8 prototyping showed the
menu's per-decision pre-approval gate added interaction friction without
adding real safety beyond what the propose-only tool boundary already
guarantees.

---

# Milestone 1.5: Resilience & UX Hardening

Backfilled after the fact — this section documents design already settled
and largely implemented via `tasks/plan.md`'s Milestone 1.5 tasks (12-16)
before this spec section was written; see that file's Task Details for the
full per-task rationale.

## Objective

Milestone 1's finished graph had two gaps, found by re-examining it rather
than by a live failure: none of its API calls (three `ChatAnthropic` call
sites, four Google API call sites) had any failure handling — an exception
crashed the whole CLI process — and `route_after_detection` silently
looped back to `greet` when no action items were found, giving the human
no signal a check even ran. Both are fixed here, ahead of Milestone 2, so
Milestone 2's new API calls (OpenAI, Chroma Cloud, a local reranker) can
follow the same established pattern instead of inventing a second one.

**Guiding principle: 0 retries everywhere, fail fast, catch and fall back
immediately.** No exponential backoff, no attempt counting. The
conversational loop already gives the human a natural retry path (just ask
again), so a second automatic-retry layer underneath it would trade
latency for a benefit the chat loop already provides for free.

## Failure Handling Behavior

Each failure-prone call site is handled differently, because each is a
structurally different kind of function — there is no single generic
wrapper:

- **`classify_intent`** (a router, not a node — only returns a routing
  string) — catches the failure, prints a plain diagnostic line directly
  to stdout (matching `cli.py`'s existing interrupt-rendering format,
  since a router can't inject a message into `state["messages"]`), and
  falls back to routing `"greet"`. Accepted limitation: this message is a
  bare terminal side effect — it won't appear in a LangSmith trace or any
  future non-CLI frontend.
- **`fetch_emails`/`check_calendar`** — each wraps its Google API call and,
  on failure, sets `{"status": "error", "error_message": str(e)}` instead
  of letting the exception propagate. Both set `status` fresh on *every*
  invocation (`"fetching"`/`"done"` on success, `"error"` on failure), not
  only on failure, so a stale `"error"` from an earlier turn's failure
  never leaks into a later turn's routing decision. Either step's failure
  routes to a shared `fetch_failed` node, which shows the error via
  `interrupt()` and routes to `classify_intent` next — critically,
  `detect_actions` is never reached with partial (emails-only or
  calendar-only) data in the same turn, avoiding a confidently-wrong
  result (a real conflict silently missed because only half the data was
  available).
- **`_analyze_email`** (one call per email, inside `detect_actions`'s
  detection loop) — catches per-email, skips that email (contributes no
  action item from it), and continues the rest of the batch. Unlike the
  original design, the failure is **not** silent: the failing email's
  subject is recorded in a new `failed_emails: list[str]` field on
  `PlannerState`, and whichever node actually talks to the human that turn
  — `review()` (alongside its existing collision note) or
  `no_action_items()` (for the all-failed-and-nothing-else-found case) —
  appends a note naming the failed email(s). Reversed from an initial
  silent-skip design once it was recognized as the same "confidently wrong
  from partial data" risk that motivates the `fetch_failed` design above,
  just one email wide instead of a whole fetch.
- **`resolution.agent()`** (the main node, which may have already executed
  several real tool calls — real Gmail drafts, real proposals — before a
  later call in the same turn fails) — catches, and instead of a canned
  "sorry, try again" message, builds an honest status report from whatever
  `review.py`'s existing `_latest_proposals` helper finds already in
  `state["messages"]`, describing what was actually completed before the
  failure. A canned "nothing happened" message would be dishonest here in
  a way it isn't for the other call sites: real, persisted side effects
  (a Gmail draft genuinely created via `drafts.create`) could already
  exist. True rollback (a `drafts.delete` capability) was considered and
  rejected — not worth introducing the first destructive tool capability
  to undo something otherwise harmless to leave behind. The resulting
  plain-text `AIMessage` carries no tool calls, so it flows through the
  *already-existing* `tools_condition` → `review` path with zero new graph
  wiring.

## Project Structure Additions

- `PlannerState` (`state.py`) gains two `NotRequired` fields:
  `error_message: str` (set alongside `status: "error"`, read by
  `fetch_failed`) and `failed_emails: list[str]` (set by `detect_actions`,
  read by `review`/`no_action_items`). `NotRequired` so every existing
  `PlannerState` literal across the test suite doesn't need updating just
  to add a field most turns never populate.
- `graph.py` gains two new nodes, both inline closures alongside the
  existing ones (`fetch_emails`, `check_calendar`, `detect_actions` are
  the only real logic; wiring stays in `graph.py` per the Task 8.5
  "wiring only" principle): `fetch_failed` (shows the fetch-stage error,
  routes to `classify_intent`) and `no_action_items` (shows the "nothing
  to report" acknowledgment, routes to `classify_intent`). Both reuse the
  same `classify_intent`-routing edge map `greet` already used, rather
  than a new one per node.

## Success Criteria

- [x] A "check for conflicts" turn that finds nothing shows a real
      acknowledgment (`no_action_items`), not a silent loop back to
      `greet`.
- [x] A simulated Google API failure during `fetch_emails`/`check_calendar`
      aborts the whole turn (never reaches `detect_actions` with partial
      data) and shows a clear message via `fetch_failed`.
- [x] A simulated `classify_intent` failure prints a diagnostic line and
      falls back to routing `"greet"` without crashing.
- [x] A simulated `_analyze_email` failure for one email doesn't stop the
      rest of the batch from being classified normally, and the human is
      told which email couldn't be analyzed.
- [x] A simulated `resolution.agent()` failure after some tool calls
      already succeeded produces an honest report of what was actually
      done, not a generic "nothing happened" message.
- [x] `uv run pytest -m "not llm_eval"` and `uv run ruff check .` both pass.

## Open Questions

None outstanding — all four failure-handling designs (including reversing
`_analyze_email`'s original silent-skip choice) were settled in discussion
before/during implementation; see `tasks/plan.md`'s Milestone 1.5 Risks
table for the accepted-vs-resolved tradeoffs.

---

# Milestone 2: RAG (Policy Knowledge Base)

Written before implementation begins (Tasks 17-29 in `tasks/plan.md`),
unlike Milestone 1.5's section above which was backfilled after the fact —
this time the spec leads the build. See that file's Milestone 2 Overview,
Architecture Decisions, and Task Details for full task-by-task rationale;
this section records the parts of that design a spec should own: scope,
external contracts, and boundaries.

## Objective

Add a retrieval-augmented capability on top of Milestone 1.5's hardened
graph: a synthetic corpus of advisory scheduling policies (leave types,
meeting norms, expense categories), sized deliberately so some topics are
adjacent/overlapping enough that plain cosine-similarity retrieval starts
confusing them — the same structural gap a prior tutorial-style RAG
project left unaddressed despite a corpus large enough to need it.

**Explicitly out of scope:** policies that are themselves scheduling
constraints (e.g. "must work in-office on Fridays"). Those are
deterministic, checkable facts — not prose to retrieve and cite — and
would need a structured constraint model plus a new deterministic
detection-layer check, the same "don't trust the LLM with computable
facts" principle this codebase already enforces elsewhere (the
weekday-name ban in `resolution.py`'s system prompt; deterministic overlap
checks in `detection.py`). Out of scope for this milestone; a clean
candidate for a future one.

**Demonstrates both RAG control-flow architectures** against one shared
retrieval engine:
- **Agentic RAG** — `search_policies` bound into `resolution.agent()`
  alongside the existing three tools; the agent decides mid-reasoning
  whether an action item warrants a policy check. Guaranteed exercised
  (not just possible) via a seeded `policy_question` email pattern in both
  a found-relevant-policy and no-relevant-policy variant.
- **2-Step RAG** — a direct chat question ("what's our policy on X?") is
  classified by a now-3-way `classify_intent` and routed straight to a new
  `answer_policy_question` node: retrieve, generate, done.

**Success looks like:** a seeded `policy_question` email gets a drafted
reply grounded in the correct policy document (or a plain statement that
none applies, for the not-found variant); a direct chat policy question —
including one that only an adjacent-pair-aware retrieval can resolve
correctly — is answered with a citation without ever touching
`fetch_emails`/`check_calendar`/`detect_actions`; and a scheduling item
with no applicable policy is still resolved on judgment alone, correctly
reporting that no policy was consulted.

## Tech Stack Additions

- `langchain-openai` — OpenAI `text-embedding-3-small` supplies dense
  embeddings (Anthropic has no embeddings endpoint). Claude remains the
  only LLM used for reasoning/generation; OpenAI is scoped to embeddings
  only.
- `langchain-chroma` + `chromadb` (Cloud client) — hybrid retrieval (dense
  `Knn` + sparse `Bm25EmbeddingFunction`, fused via `Rrf`), all computed
  server-side. Chosen over self-hosted Chroma (native hybrid search is
  Cloud-only today) and over Weaviate (same capability self-hosted, but
  Docker-only — Weaviate Embedded is Linux/macOS-only, unsupported on this
  project's Windows environment).
- `torch` + `transformers`/`FlagEmbedding` — local BGE-Reranker-v2-m3
  cross-encoder reranking. Chosen over an LLM-based reranker for latency
  and per-call cost, at the cost of a heavier local footprint (~1-2GB
  model download). Neither Chroma nor OpenAI expose a rerank endpoint, so
  this is hand-built regardless of provider choice.

Flagged per this doc's existing "ask first before adding dependencies
beyond the tech stack" boundary.

## Project Structure Additions

- `src/agentic_secretary/rag.py` (new) — owns chunking, hybrid retrieval,
  and reranking as shared logic; both `resolution.agent()` (agentic) and
  `answer_policy_question` (2-step) call into it rather than each
  reimplementing retrieval. Exposes `build_policy_index()` (ingestion) and
  `search_policies(query, k) -> str` (retrieval, wrapped as
  `search_policies_tool = tool(search_policies)`).
- `seed_data/policies/*.md` (new) — 8-15 short, single-topic, advisory-only
  documents, including at least two pairs of genuinely adjacent topics
  distinguishable only by a specific checkable detail. **Chunking: one file
  = one chunk, no splitter** — each doc is authored short and single-topic,
  so the file boundary is already the correct semantic boundary; no
  fixed-size splitting, overlap, hierarchical chunking, or category
  metadata (would blur the deliberately-overlapping pairs that hybrid
  search + reranking exist specifically to disambiguate).
- `state.py` gains a fifth `ActionNeeded` variant, `PolicyQuestionEmail`
  (`kind`, `description`, `email` — no `events`, since it isn't about
  calendar state). Detected by extending `_analyze_email`'s existing
  classification with a fifth category, not a separate detection pass.
- `chat.py` gains `answer_policy_question`; `_Intent` changes from
  `wants_conflict_check: bool` to `intent: Literal["check_conflicts",
  "policy_question", "other"]`, matching this codebase's existing
  preference for discriminated types over ambiguous booleans. The same
  `classify_intent` is attached as the router for `greet`,
  `answer_policy_question`, `fetch_failed`, and `no_action_items` — every
  node that ends a turn by capturing a fresh reply routes through it
  directly, never back through `greet`'s own fixed prompt.
- `config.py` gains `openai_api_key`, `embedding_model_name` (default
  `text-embedding-3-small`), `chroma_api_key`, `chroma_tenant`,
  `chroma_database`, `reranker_model_name` (default
  `BAAI/bge-reranker-v2-m3`).

## Failure Handling Behavior (extends Milestone 1.5)

Same rule, no new pattern: 0 retries, plain try/except per call site,
immediate fallback. Applied to the three new provider surfaces (OpenAI
embedding calls, Chroma Cloud network calls, the local BGE inference
call), split by *when* the failure happens rather than by provider:

- **Index-build/upsert failure** (ingestion time) raises loudly — the
  retrieval feature is entirely unavailable for this run, and there's no
  partial-index state worth silently continuing with.
- **Query-time failure** inside `search_policies` (retrieval or reranking)
  returns a degradation string ("Policy search is temporarily unavailable;
  proceeding without it.") instead of raising, so the agent's turn or the
  2-step answer path continues rather than crashing. Textually distinct
  from the legitimate "No relevant policy found." sentinel — one means the
  search ran and found nothing, the other means the search didn't run.

## Boundaries Additions

- **Ask first:** already-flagged new dependencies above; any change to the
  hybrid-retrieval or reranking approach once implemented (e.g. dropping
  the reranker for cost reasons) would change a load-bearing part of this
  milestone's demonstrated technique, not just an implementation detail.
- **New external prerequisite:** a Chroma Cloud account, database, and API
  key — same category of manual setup step as Milestone 1's Google Cloud
  OAuth client and burner account. Confirm reachability
  (`chromadb.CloudClient()`) and free-tier limits for this project's scale
  during Task 17, before building on top of it.
- **Deliberate, scoped exception to "local-only":** the policy corpus
  (text + embeddings) lives on Chroma's managed cloud service. Scoped
  *only* to the vector store — Gmail/Calendar access, the burner account,
  and all LLM reasoning stay exactly as local as Milestone 1 left them.
  Acceptable because the corpus is synthetic policy text, not sensitive.
- **Never do:** let the agent fabricate a policy citation when
  `search_policies` returns the no-match sentinel or the degradation
  string — the system prompt must state plainly that no policy applies
  rather than guess, same "don't guess, ask/state directly" principle
  already governing Milestone 1's action resolution.

## Testing Strategy Additions

- `tests/test_rag.py` (new) — mocks the Chroma client, `OpenAIEmbeddings`,
  and the reranker; no live network calls or model downloads in the
  automated suite. Covers ingestion (one chunk per file, stable ids,
  filename metadata), retrieval (hybrid fusion, rerank narrowing, citation
  formatting, the no-match sentinel), and both failure-handling paths
  (index-build raises, query-time degrades).
- `evals/policy_retrieval_examples.py` + `tests/test_policy_retrieval_eval.py`
  (new, marked `llm_eval`, excluded from CI) — a golden query → expected
  source dataset weighted toward the overlapping-topic pairs, reporting
  precision separately for a cosine-only baseline vs. the real
  hybrid+rerank pipeline. The concrete before/after this milestone exists
  to demonstrate.
- `tests/test_detection.py`, `tests/test_resolution.py`, `tests/test_chat.py`,
  `tests/test_graph.py` extended with mocked coverage for
  `PolicyQuestionEmail` detection, the agentic tool-binding path, the
  3-way `classify_intent` split, and `answer_policy_question`.
- `evals/agent_examples.py` extended with both `policy_question` variants
  plus a general conflict-with-policy-context case; LangSmith judge rubric
  updated to score "did it correctly apply/dismiss the relevant policy."

## Success Criteria

- [ ] Seeded `policy_question` email (found variant): agent calls
      `search_policies` and drafts a reply citing the correct policy.
- [ ] Seeded `policy_question` email (not-found variant): agent states
      plainly that no policy applies rather than fabricating an answer.
- [ ] A scheduling conflict with no applicable policy is resolved on
      judgment alone, correctly reporting no policy was consulted.
- [ ] A direct chat policy question — including one only the
      overlapping-pair distinction resolves correctly — is answered with a
      citation via `answer_policy_question`, without touching
      `fetch_emails`/`check_calendar`/`detect_actions`.
- [ ] Retrieval eval (Task 26) shows a measured precision difference
      between cosine-only and hybrid+rerank on the ambiguous-topic
      queries — recorded here once real numbers exist.
- [ ] A failure-injection test (mocked Chroma/embedding/reranker call
      raising) proves immediate, clean fallback on both the agentic and
      2-step paths, consistent with Milestone 1.5's 0-retry rule.
- [ ] `uv run pytest -m "not llm_eval"` and `uv run ruff check .` both
      pass.
- [ ] README documents Chroma Cloud + OpenAI setup and both demo paths.

## Open Questions

- Exact wording/topics of the 8-15 policy documents and which pairs are
  made deliberately overlapping — left to Task 18 implementation time, not
  spec-blocking.
- Chroma Cloud pricing tier / free-tier limits for this project's scale —
  confirm during Task 17 setup.
- Whether Task 26's before/after comparison is compelling enough for the
  README's headline demo, or better left as spec-doc detail — decide once
  real numbers exist (Task 29).
