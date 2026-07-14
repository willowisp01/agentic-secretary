from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Command

from agentic_secretary.cli import main
from agentic_secretary.graph import ActionResolution, CalendarOverlapConflict
from agentic_secretary.tools import CalendarEvent, DraftResult, EventProposal


def test_main_exits_before_side_effects_when_anthropic_api_key_missing():
    # Reproduces a live-discovered gap: detect_actions now runs on every
    # invocation and constructs ChatAnthropic per email, which previously
    # crashed deep inside the graph -- after real Gmail/Calendar API calls
    # had already run -- if ANTHROPIC_API_KEY was unset. Fail fast instead,
    # before any side effect happens.
    with (
        patch("agentic_secretary.cli.settings") as mock_settings,
        patch("agentic_secretary.cli.get_credentials") as mock_get_credentials,
    ):
        mock_settings.anthropic_api_key = None

        with pytest.raises(SystemExit):
            main()

        mock_get_credentials.assert_not_called()


def test_main_prints_detected_action_items(capsys):
    # detect_actions populates PlannerState["action_items"] but main() never
    # surfaced it -- a user running the CLI had no visibility into whether
    # any action items were found at all.
    standup = CalendarEvent(
        id="evt_standup",
        title="Team Standup",
        start=datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc),
    )
    client_sync = CalendarEvent(
        id="evt_client_sync",
        title="Client Sync",
        start=datetime(2026, 7, 14, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc),
    )
    fake_result = {
        "emails": [],
        "calendar_events": [],
        "action_items": [
            CalendarOverlapConflict(
                description="'Team Standup' overlaps with 'Client Sync'",
                events=[standup, client_sync],
            )
        ],
        "resolutions": [],
        "status": "done",
    }

    with (
        patch("agentic_secretary.cli.settings") as mock_settings,
        patch("agentic_secretary.cli.get_credentials"),
        patch("agentic_secretary.cli.build"),
        patch("agentic_secretary.cli.build_graph") as mock_build_graph,
    ):
        mock_settings.anthropic_api_key = "fake-key"
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_result
        mock_build_graph.return_value = mock_graph

        main()

    captured = capsys.readouterr()
    assert "calendar_overlap" in captured.out
    assert "'Team Standup' overlaps with 'Client Sync'" in captured.out


def test_main_prints_resolutions(capsys):
    # A shift-slot proposal has no persisted artifact anywhere (unlike a
    # Gmail draft) -- the CLI transcript is the only place it's ever
    # visible, so it needs to show the actual proposed new time, not just
    # a generic "resolved" message.
    standup = CalendarEvent(
        id="evt_standup",
        title="Team Standup",
        start=datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc),
    )
    client_sync = CalendarEvent(
        id="evt_client_sync",
        title="Client Sync",
        start=datetime(2026, 7, 14, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc),
    )
    item = CalendarOverlapConflict(
        description="'Team Standup' overlaps with 'Client Sync'",
        events=[standup, client_sync],
    )
    proposal = EventProposal(
        title="Client Sync",
        start=datetime(2026, 7, 14, 11, 0, tzinfo=timezone.utc),
        duration_minutes=45,
        existing_event_id="evt_client_sync",
    )
    fake_result = {
        "emails": [],
        "calendar_events": [],
        "action_items": [item],
        "resolutions": [
            ActionResolution(
                action_item=item,
                remedy="shift_slot",
                shift_event_id="evt_client_sync",
                proposal=proposal,
            ),
            ActionResolution(action_item=item, remedy="draft_reply", proposal=DraftResult("d1", "th1")),
        ],
        "status": "done",
    }

    with (
        patch("agentic_secretary.cli.settings") as mock_settings,
        patch("agentic_secretary.cli.get_credentials"),
        patch("agentic_secretary.cli.build"),
        patch("agentic_secretary.cli.build_graph") as mock_build_graph,
    ):
        mock_settings.anthropic_api_key = "fake-key"
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_result
        mock_build_graph.return_value = mock_graph

        main()

    captured = capsys.readouterr()
    assert "2026-07-14 11:00:00+00:00" in captured.out
    assert "2026-07-14 11:45:00+00:00" in captured.out
    assert "'Client Sync'" in captured.out
    assert "d1" in captured.out


@patch("builtins.input", return_value="my answer")
def test_main_resumes_on_interrupt_until_the_graph_finishes(mock_input):
    # main() used to do a single one-shot invoke with no interrupt handling
    # at all -- any node that pauses via interrupt() (classify_intent,
    # present_menu, ...) would have nothing driving the resume.
    with (
        patch("agentic_secretary.cli.settings") as mock_settings,
        patch("agentic_secretary.cli.get_credentials"),
        patch("agentic_secretary.cli.build"),
        patch("agentic_secretary.cli.build_graph") as mock_build_graph,
    ):
        mock_settings.anthropic_api_key = "fake-key"
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = [
            {"__interrupt__": (SimpleNamespace(value="Check for conflicts?"),)},
            {"emails": [], "calendar_events": [], "action_items": [], "resolutions": [], "status": "done"},
        ]
        mock_build_graph.return_value = mock_graph

        main()

    assert mock_graph.invoke.call_count == 2
    mock_input.assert_called_once_with("Check for conflicts?> ")
    resume_call_args = mock_graph.invoke.call_args_list[1]
    assert resume_call_args.args[0] == Command(resume="my answer")
