from googleapiclient.discovery import Resource
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import interrupt

from agentic_secretary import tools
from agentic_secretary.chat import classify_intent, greet
from agentic_secretary.detection import detect_actions
from agentic_secretary.resolution import make_agent_node, make_tools_node
from agentic_secretary.review import review, route_after_review
from agentic_secretary.state import (
    BackToBackConflict,
    CalendarOverlapConflict,
    EmailConflict,
    PlannerState,
    RescheduleRequest,
)

# Every custom type that ends up inside PlannerState -- LangGraph's default
# checkpoint serializer warns (and, in a future version, will refuse) to
# deserialize any type outside its own built-in allowlist, since every
# interrupt/resume round-trip goes through the checkpointer. Re-solves a gap
# already fixed once on master (before this redesign branched) for the
# now-superseded state shape; the types have since moved to state.py/tools.py.
_CHECKPOINT_ALLOWED_TYPES = (
    tools.EmailSummary,
    tools.CalendarEvent,
    tools.EventProposal,
    tools.DraftResult,
    CalendarOverlapConflict,
    BackToBackConflict,
    EmailConflict,
    RescheduleRequest,
)


def build_graph(gmail_service: Resource, calendar_service: Resource):
    def fetch_emails(state: PlannerState) -> dict:
        try:
            emails = tools.list_recent_emails(gmail_service)
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
        return {"emails": emails, "status": "fetching"}

    def check_calendar(state: PlannerState) -> dict:
        try:
            events = tools.list_upcoming_events(calendar_service)
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
        return {"calendar_events": events, "status": "done"}

    def fetch_failed(state: PlannerState) -> dict:
        reply = interrupt(
            "Sorry, I couldn't complete that check -- "
            f"{state['error_message']}. Want to try again?"
        )
        return {"messages": [HumanMessage(content=reply)]}

    def no_action_items(state: PlannerState) -> dict:
        reply = interrupt(
            "I checked your calendar and inbox -- nothing to report right now."
        )
        return {"messages": [HumanMessage(content=reply)]}

    def route_after_fetch_emails(state: PlannerState) -> str:
        return "fetch_failed" if state["status"] == "error" else "check_calendar"

    def route_after_check_calendar(state: PlannerState) -> str:
        return "fetch_failed" if state["status"] == "error" else "detect_actions"

    def route_after_detection(state: PlannerState) -> str:
        return "agent" if state["action_items"] else "no_action_items"

    builder = StateGraph(PlannerState)
    builder.add_node("greet", greet)
    builder.add_node("fetch_emails", fetch_emails)
    builder.add_node("check_calendar", check_calendar)
    builder.add_node("fetch_failed", fetch_failed)
    builder.add_node("detect_actions", detect_actions)
    builder.add_node("no_action_items", no_action_items)
    builder.add_node("agent", make_agent_node(gmail_service))
    builder.add_node("tools", make_tools_node(gmail_service))
    builder.add_node("review", review)

    builder.add_edge(START, "greet")
    _post_greet_routes = {"fetch_emails": "fetch_emails", "greet": "greet"}
    builder.add_conditional_edges("greet", classify_intent, _post_greet_routes)
    builder.add_conditional_edges(
        "fetch_emails",
        route_after_fetch_emails,
        {"check_calendar": "check_calendar", "fetch_failed": "fetch_failed"},
    )
    builder.add_conditional_edges(
        "check_calendar",
        route_after_check_calendar,
        {"detect_actions": "detect_actions", "fetch_failed": "fetch_failed"},
    )
    builder.add_conditional_edges("fetch_failed", classify_intent, _post_greet_routes)
    builder.add_conditional_edges(
        "detect_actions",
        route_after_detection,
        {"agent": "agent", "no_action_items": "no_action_items"},
    )
    builder.add_conditional_edges(
        "no_action_items", classify_intent, _post_greet_routes
    )
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": "review"}
    )
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges(
        "review", route_after_review, {"agent": "agent", END: END}
    )

    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES)
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
