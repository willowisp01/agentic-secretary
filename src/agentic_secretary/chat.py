from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agentic_secretary.config import settings
from agentic_secretary.state import PlannerState

_OPENING_MESSAGE = (
    "Hi! I'm your AI secretary. Ask me to check for conflicts and I'll take a "
    "look at your calendar and inbox."
)
_REPROMPT_MESSAGE = 'Anything else? (e.g. "check for conflicts")'


class _Intent(BaseModel):
    wants_conflict_check: bool = Field(
        description="True if the message asks to check for scheduling "
        "conflicts, meeting requests, or reschedules."
    )


def greet(state: PlannerState) -> dict:
    prompt = _OPENING_MESSAGE if not state["messages"] else _REPROMPT_MESSAGE
    reply = interrupt(prompt)
    return {"messages": [HumanMessage(content=reply)]}


def classify_intent(state: PlannerState) -> str:
    last_message = state["messages"][-1]
    llm = ChatAnthropic(
        model_name=settings.model_name, api_key=settings.anthropic_api_key
    )
    structured_llm = llm.with_structured_output(_Intent, method="json_schema")
    intent = structured_llm.invoke(
        "Does this message ask to check for scheduling conflicts, meeting "
        f"requests, or reschedules?\n\nMessage: {last_message.content}"
    )
    return "fetch_emails" if intent.wants_conflict_check else "greet"
