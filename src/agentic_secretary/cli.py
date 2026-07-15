from googleapiclient.discovery import build
from langgraph.types import Command

from agentic_secretary.auth import get_credentials
from agentic_secretary.graph import build_graph

_INITIAL_STATE = {
    "messages": [],
    "emails": [],
    "calendar_events": [],
    "action_items": [],
    "status": "pending",
}


def main() -> None:
    credentials = get_credentials()
    gmail_service = build("gmail", "v1", credentials=credentials)
    calendar_service = build("calendar", "v3", credentials=credentials)
    graph = build_graph(gmail_service, calendar_service)
    config = {"configurable": {"thread_id": "cli"}}

    resume_value = None
    while True:
        if resume_value is None:
            result = graph.invoke(_INITIAL_STATE, config=config)
        else:
            result = graph.invoke(Command(resume=resume_value), config=config)

        interrupts = result.get("__interrupt__")
        if not interrupts:
            break

        print(f"\n{interrupts[0].value}\n")
        resume_value = input("> ")


if __name__ == "__main__":
    main()
