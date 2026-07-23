# Task List: AI Secretary — Milestone 1 (Planner) ✅ Complete

Full detail (acceptance criteria, verification, files, dependencies) in
[`tasks/plan.md`](plan.md). This file tracks status only.

## Phase 1: Foundation
- [x] Task 1: Project scaffolding and config
- [x] Task 2: Google OAuth flow
- [x] Task 3: Seed data fixtures (conflict patterns)

**Checkpoint:** deps installed + lint clean, OAuth smoke test works against
burner account, seed fixtures parse and cover all 4 conflict patterns.

## Phase 2: Data Layer
- [x] Task 4: Gmail/Calendar tool wrappers
- [x] Task 5: Seeding script

**Checkpoint:** burner account seeded and visually verified, tool tests pass
against mocks (no live calls in the test suite).

## Phase 3: Agent Reasoning
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

**Checkpoint:** end-to-end CLI run detects seeded action items, resolves
them autonomously (proposals/drafts as appropriate), presents one review
summary; conflict-pattern tests pass.

## Phase 4: Observability and Polish
- [x] Task 9: LangSmith tracing verification
- [x] Task 10: Full test suite + lint pass
- [x] Task 11: README + demo walkthrough notes

**Checkpoint (final):** all spec success criteria met, tests + lint pass,
LangSmith trace confirmed, README walkthrough verified — ready for demo
recording.

## Prerequisites (not automatable — needed before implementation starts)
- [x] Google Cloud project + OAuth client created, Gmail + Calendar APIs
      enabled
- [x] Burner Gmail/Calendar account ready and accessible
- [x] `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` available for `.env`

---

# Task List: AI Secretary — Milestone 1.5 (Resilience & UX Hardening)

Full detail in [`tasks/plan.md`](plan.md). 0 retries everywhere — fail
fast, catch, fall back immediately; no backoff. Each of the three existing
LLM call sites (router, per-email loop, main agent node) fails differently
because each is a structurally different kind of function — see plan.md's
Overview for why.

## Phase A: Graph UX + Atomicity
- [ ] Task 12: `no_action_items` node (real acknowledgment instead of a
      silent loop back to `greet` when nothing's found)
- [ ] Task 13: Fetch-stage atomicity (`fetch_failed` node — abort the whole
      turn on a `fetch_emails`/`check_calendar` failure, never run
      `detect_actions` on partial data)

## Phase B: API failure handling
- [ ] Task 14: `classify_intent` — print a diagnostic line, fall back to
      `greet`
- [ ] Task 15: `_analyze_email` — skip the failed email silently, continue
      the batch
- [ ] Task 16: `resolution.agent()` — honest status report (reusing
      `review.py`'s `_latest_proposals`) instead of a canned message, since
      real tool calls may have already executed this turn

**Checkpoint:** all four fixes verified together (empty-result
acknowledgment, atomicity, and all three failure-handling behaviors); full
test/lint pass; review with human before starting Milestone 2.

---

# Task List: AI Secretary — Milestone 2 (RAG: Policy Knowledge Base)

Full detail in [`tasks/plan.md`](plan.md). Builds on Milestone 1.5's
hardened graph. Demonstrates both agentic RAG (`search_policies` bound
into `resolution.agent()`) and 2-step RAG (`answer_policy_question` via a
3-way `classify_intent`), sharing one hybrid-search + BGE-rerank retrieval
engine. Explicitly out of scope: time/day-of-week-based scheduling
policies (would need a deterministic constraint engine, not RAG — see
plan.md's Overview).

## Phase C: RAG Foundation
- [ ] Task 17: Config + dependencies (OpenAI, Chroma Cloud, BGE reranker)
- [x] Task 18: Synthetic policy corpus (sized for deliberate topic overlap)
      + two new `policy_question` email scenarios (found / not-found)

**Checkpoint:** deps installed, Chroma Cloud reachable, corpus + seed
emails in place; review with human.

## Phase D: Retrieval Walking Skeleton
- [ ] Task 19: Ingestion module (`rag.py` — header-based H1/H2 chunks,
      header path prepended before embedding, upsert into a Chroma Cloud
      collection configured for hybrid search)
- [ ] Task 20: `search_policies` — hybrid retrieval (dense + BM25, `Rrf`)
      + BGE-Reranker-v2-m3 rerank

**Checkpoint:** live smoke test proves hybrid+rerank correctly disambiguates
an overlapping-topic pair; mocked tests pass with no live calls; review
with human.

## Phase E: Detection + Agent + Chat Integration
- [ ] Task 21: `PolicyQuestionEmail` detection (5th `_analyze_email`
      category, new `ActionNeeded` variant)
- [ ] Task 22: Bind `search_policies` into `resolution.agent()` — agentic RAG
- [ ] Task 23: 3-way `classify_intent` + `answer_policy_question` node —
      2-step RAG
- [ ] Task 24: Live verification, both paths (found/not-found policy
      emails, direct chat question, control scenario)

**Checkpoint:** both RAG architectures demonstrated end-to-end against the
seeded burner account; review with human before hardening.

## Phase F: Production-Grade Hardening
- [ ] Task 25: Resilience for OpenAI/Chroma/BGE calls (same 0-retry rule)
- [ ] Task 26: Retrieval evaluation — hybrid+rerank vs. cosine-only
      baseline, measured precision comparison
- [ ] Task 27: Extend eval suites (agentic + 2-step scenarios)

**Checkpoint:** retrieval eval shows a real measured difference; a
failure-injection test proves clean fallback, not a crash; review with
human.

## Phase G: Polish
- [ ] Task 28: Full test suite + lint pass
- [ ] Task 29: README + spec update (Chroma Cloud/OpenAI setup, both RAG
      demo paths, local-only exception documented, out-of-scope
      time-based-policy call documented)

**Checkpoint (final):** all acceptance criteria met, tests + lint pass,
README walkthrough verified — ready for demo recording.

## Prerequisites (not automatable — needed before Milestone 2 implementation)
- [ ] Chroma Cloud account + database created, API key available
- [ ] `OPENAI_API_KEY` available for `.env`
- [ ] Confirm local disk space / first-run time budget for the BGE
      reranker's one-time model download
