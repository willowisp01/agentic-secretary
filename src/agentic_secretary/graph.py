from datetime import timedelta
from typing import TypedDict

from googleapiclient.discovery import Resource
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agentic_secretary import tools

# Gap threshold under which two adjacent events count as "back-to-back, no
# buffer" (soft conflict) rather than just a normally-spaced schedule.
NO_BUFFER_THRESHOLD = timedelta(minutes=15)


class Conflict(TypedDict):
    kind: str
    description: str
    events: list[tools.CalendarEvent]
    email: tools.EmailSummary | None


class PlannerState(TypedDict):
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    conflicts: list[Conflict]
    status: str


def _find_calendar_overlaps(events: list[tools.CalendarEvent]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for i, a in enumerate(sorted_events):
        for b in sorted_events[i + 1 :]:
            if a.start < b.end and b.start < a.end:
                conflicts.append(
                    Conflict(
                        kind="calendar_overlap",
                        description=f"{a.title!r} overlaps with {b.title!r}",
                        events=[a, b],
                        email=None,
                    )
                )
    return conflicts


def _find_back_to_back(events: list[tools.CalendarEvent]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sorted_events = sorted(events, key=lambda e: e.start)
    for a, b in zip(sorted_events, sorted_events[1:]):
        gap = b.start - a.end
        if timedelta(0) <= gap <= NO_BUFFER_THRESHOLD:
            conflicts.append(
                Conflict(
                    kind="back_to_back",
                    description=f"{a.title!r} ends right as {b.title!r} starts, no buffer",
                    events=[a, b],
                    email=None,
                )
            )
    return conflicts


def detect_conflicts(state: PlannerState) -> dict:
    calendar_events = state["calendar_events"]
    conflicts = _find_calendar_overlaps(calendar_events) + _find_back_to_back(calendar_events)
    return {"conflicts": conflicts}


def build_graph(gmail_service: Resource, calendar_service: Resource):
    def fetch_emails(state: PlannerState) -> dict:
        return {"emails": tools.list_recent_emails(gmail_service)}

    def check_calendar(state: PlannerState) -> dict:
        return {
            "calendar_events": tools.list_upcoming_events(calendar_service),
            "status": "done",
        }

    builder = StateGraph(PlannerState)
    builder.add_node("fetch_emails", fetch_emails)
    builder.add_node("check_calendar", check_calendar)
    builder.add_node("detect_conflicts", detect_conflicts)
    builder.add_edge(START, "fetch_emails")
    builder.add_edge("fetch_emails", "check_calendar")
    builder.add_edge("check_calendar", "detect_conflicts")
    builder.add_edge("detect_conflicts", END)
    return builder.compile(checkpointer=InMemorySaver())
