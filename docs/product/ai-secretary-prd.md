# AI Secretary — PRD

## Purpose
The AI Secretary reads a user's Gmail inbox and Google Calendar, detects scheduling problems (overlaps, tight back-to-backs, meeting-request emails, reschedule emails), and resolves what it can using its own judgment — always as a proposal or draft for the human to review, never as a final booked/sent action.

## Tools
- **`propose_event`** — propose a new calendar event time, or propose moving an existing one (`existing_event_id` set to move, omitted for a brand-new event). Never calls Calendar's `insert`/`patch`.
- **`draft_reply_tool`** — draft a reply email to the sender of an email-related action item. Never calls Gmail's `send`.
- **`withdraw_proposal`** — retract a previously made `propose_event` call for the same target when the agent changes its mind mid-turn.
- (Upstream, not agent-callable) **`list_recent_emails`** / **`list_upcoming_events`** — feed the detection layer that decides which action items reach the agent at all.

## Behavior Rules
- Every tool call only ever produces a proposal or draft — never act as though something is final.
- If a draft reply commits to a specific time, also call `propose_event` for that time — an email-only commitment is invisible to the collision check.
- If abandoning a previously proposed time (e.g. declining it for alternatives instead), call `withdraw_proposal` — otherwise the collision check keeps treating it as live.
- It's fine to skip an item and say why, rather than force a resolution that doesn't make sense.
- If genuinely unsure how to handle an item, ask a direct question instead of guessing.
- After acting on everything addressable, end the turn with a plain-text summary (never another tool call) for human review.
- A correction after the summary (e.g. "move it to 2pm instead") amends the specific prior tool call it refers to — not a new unrelated request.
- A plain acknowledgment ("ok", "approved") gets a warm reply with **no claim that anything was booked or sent** — nothing happens automatically beyond the tool calls already made.
- Use the given current date as the anchor for relative time phrases; never mention a day-of-week name anywhere in any response.

## Scenarios and Expected Behavior

### Conflict Detection Queries (calendar-only)
| Input / Seeded State | Expected Behavior |
|---|---|
| "Team Standup" 09:00–09:30 directly overlaps "Client Sync" 09:15–10:00 | `detect_actions` flags `calendar_overlap` → agent proposes moving one via `propose_event` (or explains why it's leaving both) |
| "Lunch" ends 13:00 exactly as "Design Review" starts | `detect_actions` flags `back_to_back` (zero buffer) → agent proposes a buffer shift or explains why it's leaving it |
| Two events with a normal gap between them | No action item generated — nothing proposed |

### Email-Driven Queries (requires reading email + comparing to calendar)
| Input | Expected Behavior |
|---|---|
| Email: "are you free tomorrow at 9:15am for 30 min?" (overlaps Standup + Client Sync) | `email_conflict` detected → agent drafts a reply addressing the overlap, and calls `propose_event` for the same 9:15 time since the draft commits to it |
| Email: "can we push our client sync from tomorrow to Thursday instead?" | `reschedule` detected → agent proposes moving Client Sync (`existing_event_id` set) and/or drafts a confirmation reply — or asks for clarification if the target date is genuinely ambiguous (e.g. "Thursday" when today already is Thursday) rather than guessing |
| Email casually mentions an existing event ("see you at the client call!") with no request | No `reschedule`/`email_conflict` raised — email is informational only, not an action item |

### Mixed Queries (both calendar + email reasoning in one pass)
| Input | Expected Behavior |
|---|---|
| A calendar overlap and an unrelated reschedule email both present in the same run | Agent resolves both independently (right tool for each), then presents **one combined summary** covering everything it did |
| An email proposes a new time that itself collides with a *different* existing event | Agent calls `propose_event` for the requested time (making the draft's commitment visible to the collision check) and surfaces the resulting conflict via the review node's collision note |

### Out-of-Scope / Edge Cases
| Input | Expected Behavior |
|---|---|
| "i love carrot" / other non-scheduling chit-chat | `classify_intent` routes back to `greet` — not treated as a conflict-check request |
| Weekly digest email with no scheduling content | Ignored by `_find_email_conflicts` — no action item |
| Human replies "ok" / "looks good" after the summary | Warm acknowledgment only — never implies booking/sending happened |
| Human replies "move it to 2pm instead" after the summary | Treated as amending the specific prior proposal, using the agent's own tool-call history as context |
