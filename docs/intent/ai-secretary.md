# Intent: AI Secretary (Portfolio Project)

Confirmed via interview on 2026-07-04.

- **Outcome:** A resume/portfolio project — an AI "planner" agent built with
  LangChain + LangGraph that reads a (burner) Gmail inbox and Google Calendar,
  reasons about scheduling (conflicts, meeting requests, reschedules), and
  defaults to drafting replies/events rather than auto-sending. RAG/vector
  database work is layered in later, once the planner core works. LangSmith
  provides observability.
- **User:** A single seeded persona — "busy working professional" — populated
  with realistic, synthetically-generated scenarios (external meeting
  requests, reschedules, internal digest emails, a calendar with recurring
  standups plus occasional conflicts). The persona is fictional test data,
  not drawn from the author's own work history.
- **Why now:** To demonstrate applied AI/agent-engineering skills (LangGraph
  orchestration, live third-party API integration, RAG, observability) to
  recruiters/hiring managers who recognize this class of problem from their
  own inboxes.
- **Success:** A working local demo — the agent reads inbox/calendar, detects
  conflicts, and drafts sensible responses/events — shown via recorded demo
  or live screen-share, not a public/hosted link. Real Gmail/Calendar access
  runs through a burner account so the demo never touches the author's real
  schedule or a stranger's credentials.
- **Constraint:** Budget-conscious (Claude API, Haiku-tier by default for
  cost), burner Gmail/Calendar account, local-only runtime (no hosted/public
  deployment so no one but the author can operate it), LangSmith for tracing.
- **Out of scope (for now):** Multi-persona support, RAG/vector database
  (milestone 2), autonomous auto-send/auto-book without human review, and
  public/hosted deployment.

## Repo conventions agreed alongside this

- Planning artifacts (this file, specs, task plans) are committed to the
  repo — they're a visible signal of process for anyone reviewing the
  project.
- Tool-local scratch/config (e.g. `.claude/settings.local.json`) is
  gitignored — it's machine-specific, not project content.
