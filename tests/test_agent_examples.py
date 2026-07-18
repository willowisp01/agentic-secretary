from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_examples import EXAMPLES
from agentic_secretary.resolution import make_agent_node


def _llm_returning(*invoke_results):
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.side_effect = invoke_results
    llm.bind_tools.return_value = bound
    return llm


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
@patch("agentic_secretary.resolution.ChatAnthropic")
def test_example_state_is_well_formed(mock_chat_anthropic, example):
    # Not checking `expected` here -- that needs a real LLM call, which is a
    # separate, not-yet-built evaluator. This only proves the scripted state
    # is schema-valid and agent() runs on it without raising.
    mock_chat_anthropic.return_value = _llm_returning(AIMessage(content="stub reply"))

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(example.state)

    assert result["messages"]
