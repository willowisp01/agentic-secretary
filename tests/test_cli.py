from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agentic_secretary.cli import main
from agentic_secretary.graph import CalendarOverlapConflict
from agentic_secretary.tools import CalendarEvent


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
