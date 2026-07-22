from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agentic_secretary import tools


class CalendarOverlapConflict(BaseModel):
    kind: Literal["calendar_overlap"] = "calendar_overlap"
    description: str
    events: tuple[tools.CalendarEvent, tools.CalendarEvent]


class BackToBackConflict(BaseModel):
    kind: Literal["back_to_back"] = "back_to_back"
    description: str
    events: tuple[tools.CalendarEvent, tools.CalendarEvent]


class EmailConflict(BaseModel):
    kind: Literal["email_conflict"] = "email_conflict"
    description: str
    email: tools.EmailSummary
    events: list[tools.CalendarEvent] = Field(min_length=1)


class RescheduleRequest(BaseModel):
    kind: Literal["reschedule"] = "reschedule"
    description: str
    email: tools.EmailSummary
    event: tools.CalendarEvent


ActionNeeded = Annotated[
    CalendarOverlapConflict | BackToBackConflict | EmailConflict | RescheduleRequest,
    Field(discriminator="kind"),
]


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    emails: list[tools.EmailSummary]
    calendar_events: list[tools.CalendarEvent]
    action_items: list[ActionNeeded]
    status: str
    # Only meaningful when status == "error" -- set fresh by fetch_emails/
    # check_calendar on every invocation (success or failure) so a stale
    # value from an earlier turn's failure never leaks into this turn's
    # routing decision. NotRequired since existing PlannerState literals
    # (initial states, fixtures) predate this field and never set it.
    error_message: NotRequired[str]
