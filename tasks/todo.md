# Task List: AI Secretary — Milestone 1 (Planner)

Full detail (acceptance criteria, verification, files, dependencies) in
[`tasks/plan.md`](plan.md). This file tracks status only.

## Phase 1: Foundation
- [x] Task 1: Project scaffolding and config
- [x] Task 2: Google OAuth flow
- [x] Task 3: Seed data fixtures (conflict patterns)

**Checkpoint:** deps installed + lint clean, OAuth smoke test works against
burner account, seed fixtures parse and cover all 4 conflict patterns.

## Phase 2: Data Layer
- [ ] Task 4: Gmail/Calendar tool wrappers
- [ ] Task 5: Seeding script

**Checkpoint:** burner account seeded and visually verified, tool tests pass
against mocks (no live calls in the test suite).

## Phase 3: Agent Reasoning
- [ ] Task 6: PlannerState + graph skeleton
- [ ] Task 7: Conflict-detection node
- [ ] Task 8: Draft-response node + human-review interrupt

**Checkpoint:** end-to-end CLI run detects seeded conflicts, drafts a
response, pauses for human confirmation; conflict-pattern tests pass.

## Phase 4: Observability and Polish
- [ ] Task 9: LangSmith tracing verification
- [ ] Task 10: Full test suite + lint pass
- [ ] Task 11: README + demo walkthrough notes

**Checkpoint (final):** all spec success criteria met, tests + lint pass,
LangSmith trace confirmed, README walkthrough verified — ready for demo
recording.

## Prerequisites (not automatable — needed before implementation starts)
- [x] Google Cloud project + OAuth client created, Gmail + Calendar APIs
      enabled
- [x] Burner Gmail/Calendar account ready and accessible
- [ ] `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` available for `.env`
