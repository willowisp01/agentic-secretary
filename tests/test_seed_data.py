from pathlib import Path

import pytest

from agentic_secretary.seed_data import (
    Email,
    CalendarEvent,
    Relation,
    load_calendar_events,
    load_emails,
    load_relations,
    validate_relations,
)

SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"
MIN_FIXTURE_EMAILS = 4
MIN_FIXTURE_EVENTS = 4


def test_loads_emails():
    emails = load_emails(SEED_DATA_DIR / "emails.yaml")
    assert len(emails) >= MIN_FIXTURE_EMAILS
    assert all(isinstance(e, Email) for e in emails)
    assert all(e.sent_relative for e in emails)


def test_loads_calendar_events():
    events = load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
    assert len(events) >= MIN_FIXTURE_EVENTS
    assert all(isinstance(e, CalendarEvent) for e in events)
    assert all(e.start_relative for e in events)


def test_loads_relations():
    relations = load_relations(SEED_DATA_DIR / "relations.yaml")
    assert all(isinstance(r, Relation) for r in relations)
    kinds = {r.kind for r in relations}
    assert kinds == {
        "calendar_overlap",
        "back_to_back",
        "email_conflict",
        "reschedule",
        "mentions",
    }


def test_real_fixtures_pass_validation():
    emails = load_emails(SEED_DATA_DIR / "emails.yaml")
    events = load_calendar_events(SEED_DATA_DIR / "calendar_events.yaml")
    relations = load_relations(SEED_DATA_DIR / "relations.yaml")

    validate_relations(
        relations,
        email_ids={e.id for e in emails},
        event_ids={e.id for e in events},
    )


@pytest.mark.parametrize(
    "relation,expected_message",
    [
        (Relation(kind="back_to_back", events=["a", "b", "c"]), "exactly 2"),
        (Relation(kind="calendar_overlap", events=["a"]), "at least 2"),
        (Relation(kind="email_conflict", email=None, events=["a"]), "requires an 'email'"),
        (Relation(kind="reschedule", email="e1", event=None), "requires an 'event'"),
        (Relation(kind="mentions", email="e1", event="unknown"), "unknown event"),
        (Relation(kind="not_a_real_kind"), "Unknown relation kind"),
    ],
)
def test_validate_relations_rejects_bad_fixtures(relation, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        validate_relations([relation], email_ids={"e1"}, event_ids={"a", "b", "c"})
