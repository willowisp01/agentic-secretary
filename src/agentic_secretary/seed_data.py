from dataclasses import dataclass
from pathlib import Path

import yaml

RELATION_ARITY = {
    "calendar_overlap": {"min_events": 2},
    "back_to_back": {"exact_events": 2},
    "email_conflict": {"requires_email": True, "min_events": 1},
    "reschedule": {"requires_email": True, "requires_event": True},
    "mentions": {"requires_email": True, "requires_event": True},
}


@dataclass(frozen=True)
class Email:
    id: str
    from_: str
    to: str
    subject: str
    body: str
    sent_relative: str


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    title: str
    start_relative: str
    duration_minutes: int


@dataclass(frozen=True)
class Relation:
    kind: str
    email: str | None = None
    event: str | None = None
    events: list[str] | None = None


def load_emails(path: Path) -> list[Email]:
    data = yaml.safe_load(path.read_text())
    return [
        Email(
            id=e["id"],
            from_=e["from"],
            to=e["to"],
            subject=e["subject"],
            body=e["body"],
            sent_relative=e["sent_relative"],
        )
        for e in data["emails"]
    ]


def load_calendar_events(path: Path) -> list[CalendarEvent]:
    data = yaml.safe_load(path.read_text())
    return [
        CalendarEvent(
            id=e["id"],
            title=e["title"],
            start_relative=e["start_relative"],
            duration_minutes=e["duration_minutes"],
        )
        for e in data["events"]
    ]


def load_relations(path: Path) -> list[Relation]:
    data = yaml.safe_load(path.read_text())
    return [
        Relation(
            kind=r["kind"],
            email=r.get("email"),
            event=r.get("event"),
            events=r.get("events"),
        )
        for r in data["relations"]
    ]


def validate_relations(
    relations: list[Relation], email_ids: set[str], event_ids: set[str]
) -> None:
    for r in relations:
        rule = RELATION_ARITY.get(r.kind)
        if rule is None:
            raise ValueError(f"Unknown relation kind: {r.kind!r}")

        if rule.get("requires_email"):
            if not r.email:
                raise ValueError(f"Relation {r.kind!r} requires an 'email' field")
            if r.email not in email_ids:
                raise ValueError(f"Relation references unknown email id: {r.email!r}")

        if rule.get("requires_event"):
            if not r.event:
                raise ValueError(f"Relation {r.kind!r} requires an 'event' field")
            if r.event not in event_ids:
                raise ValueError(f"Relation references unknown event id: {r.event!r}")

        if "min_events" in rule or "exact_events" in rule:
            if not r.events:
                raise ValueError(f"Relation {r.kind!r} requires an 'events' list")
            unknown = [e for e in r.events if e not in event_ids]
            if unknown:
                raise ValueError(f"Relation references unknown event ids: {unknown}")
            min_events = rule.get("min_events")
            if min_events is not None and len(r.events) < min_events:
                raise ValueError(
                    f"Relation {r.kind!r} requires at least {min_events} events"
                )
            exact_events = rule.get("exact_events")
            if exact_events is not None and len(r.events) != exact_events:
                raise ValueError(
                    f"Relation {r.kind!r} requires exactly {exact_events} events"
                )
