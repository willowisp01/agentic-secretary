from unittest.mock import patch

from agentic_secretary.cli import _read_non_blank_input


def test_read_non_blank_input_returns_the_first_non_blank_value():
    with patch("builtins.input", return_value="check for conflicts"):
        assert _read_non_blank_input("> ") == "check for conflicts"


def test_read_non_blank_input_reprompts_on_blank_or_whitespace_only_input():
    # Live-discovered: a blank reply becomes an empty HumanMessage that
    # eventually crashes a real Anthropic API call ("user messages must
    # have non-empty content"). Blank input should never reach the graph.
    with patch(
        "builtins.input", side_effect=["", "   ", "check for conflicts"]
    ) as mock_input:
        assert _read_non_blank_input("> ") == "check for conflicts"
    assert mock_input.call_count == 3
