import sys

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


def _read_non_blank_input(prompt: str) -> str:
    # Live-discovered: a blank reply becomes an empty HumanMessage, which
    # eventually reaches a real Anthropic API call and gets rejected
    # outright ("user messages must have non-empty content"), crashing the
    # whole process. Don't let blank input reach the graph at all.
    while True:
        value = input(prompt)
        if value.strip():
            return value


def main() -> None:
    # LLM output routinely includes characters (arrows, checkmarks, smart
    # punctuation) outside legacy Windows console codepages (cp1252 etc).
    # Live-discovered: printing such a character crashed the whole process
    # with UnicodeEncodeError when stdout wasn't already UTF-8 -- which
    # depends on the invoking environment, not anything this code controls.
    # Force it explicitly rather than relying on the ambient console setup.
    sys.stdout.reconfigure(encoding="utf-8")

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
        resume_value = _read_non_blank_input("> ")


if __name__ == "__main__":
    main()
