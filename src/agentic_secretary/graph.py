from typing import TypedDict

from googleapiclient.discovery import Resource
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agentic_secretary import tools


class PlannerState(TypedDict):
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    status: str


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
    builder.add_edge(START, "fetch_emails")
    builder.add_edge("fetch_emails", "check_calendar")
    builder.add_edge("check_calendar", END)
    return builder.compile(checkpointer=InMemorySaver())
