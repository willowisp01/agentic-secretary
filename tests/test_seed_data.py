import re
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
EXPECTED_POLICY_COUNT = 15
MIN_H2_PER_POLICY = 3
MAX_H2_PER_POLICY = 5


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
        "policy_question",
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
        (
            Relation(kind="email_conflict", email=None, events=["a"]),
            "requires an 'email'",
        ),
        (Relation(kind="reschedule", email="e1", event=None), "requires an 'event'"),
        (Relation(kind="mentions", email="e1", event="unknown"), "unknown event"),
        (
            Relation(kind="policy_question", email=None),
            "requires an 'email'",
        ),
        (Relation(kind="not_a_real_kind"), "Unknown relation kind"),
    ],
)
def test_validate_relations_rejects_bad_fixtures(relation, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        validate_relations([relation], email_ids={"e1"}, event_ids={"a", "b", "c"})


POLICY_CORPUS_PATH = SEED_DATA_DIR / "policies.md"
H1_PATTERN = re.compile(r"^# (?!#)", re.MULTILINE)
H2_PATTERN = re.compile(r"^## (?!#)", re.MULTILINE)


def test_policy_corpus_has_fifteen_h1_sections():
    text = POLICY_CORPUS_PATH.read_text(encoding="utf-8")
    h1_titles = [line for line in text.splitlines() if line.startswith("# ")]
    assert len(h1_titles) == EXPECTED_POLICY_COUNT
    assert len(set(h1_titles)) == EXPECTED_POLICY_COUNT, "H1 titles must be unique"


def test_policy_corpus_sections_have_three_to_five_h2s():
    text = POLICY_CORPUS_PATH.read_text(encoding="utf-8")
    h1_starts = [m.start() for m in H1_PATTERN.finditer(text)] + [len(text)]
    for start, end in zip(h1_starts, h1_starts[1:]):
        section = text[start:end]
        h2_count = len(H2_PATTERN.findall(section))
        assert MIN_H2_PER_POLICY <= h2_count <= MAX_H2_PER_POLICY, (
            f"section starting {section.splitlines()[0]!r} has {h2_count} H2s"
        )
