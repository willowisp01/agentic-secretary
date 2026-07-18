"""Runs `evals/agent_examples.py`'s examples against the real Anthropic model
and checks each `Expected` block.

Unlike `test_agent_examples.py` (mocked, free, runs in CI), this hits a real
`ChatAnthropic` call per example -- costs real money, needs ANTHROPIC_API_KEY.
Marked `llm_eval` so it can be excluded from the default CI run once that
marker is wired into pyproject.toml/ci.yml; for now, run explicitly:

    uv run pytest tests/test_agent_examples_eval.py

`@pytest.mark.langsmith` syncs each parametrized case to a LangSmith dataset
(one per test file, per resolution.py's SYSTEM_PROMPT-and-tools-bound agent)
and records an experiment run -- no separate JSONL/upload step needed.
"""

import dataclasses
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langsmith import testing as t

from agent_examples import EXAMPLES
from agentic_secretary.resolution import make_agent_node


def _args_match(call_args: dict, args_subset: dict) -> bool:
    for key, expected in args_subset.items():
        actual = call_args.get(key)
        # A set/frozenset value means "any of these" rather than an exact
        # match -- e.g. when either of two events could legitimately be the
        # one the agent chooses to move.
        if isinstance(expected, (set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _check_tool_calls(response: AIMessage, expectations) -> list[str]:
    failures = []
    for expectation in expectations:
        matches = [
            call
            for call in response.tool_calls
            if call["name"] == expectation.name
            and _args_match(call["args"], expectation.args_subset)
        ]
        if not matches:
            failures.append(
                f"expected a {expectation.name!r} tool call with args including "
                f"{expectation.args_subset!r}; got tool_calls={response.tool_calls!r}"
            )
    return failures


def _check_forbidden_content(response: AIMessage, forbidden) -> list[str]:
    content = str(response.content).lower()
    return [word for word in forbidden if word.lower() in content]


@pytest.mark.llm_eval
@pytest.mark.langsmith
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_agent_example_against_real_llm(example):
    t.log_inputs(
        {
            "action_items": [
                item.model_dump(mode="json") for item in example.state["action_items"]
            ],
            "messages": convert_to_openai_messages(example.state["messages"]),
        }
    )
    t.log_reference_outputs(dataclasses.asdict(example.expected))

    agent = make_agent_node(MagicMock(name="gmail_service"))
    result = agent(example.state)
    response: AIMessage = result["messages"][-1]

    t.log_outputs({"content": response.content, "tool_calls": response.tool_calls})

    failures = [
        *_check_tool_calls(response, example.expected.tool_calls_include),
        *_check_forbidden_content(response, example.expected.content_must_not_contain),
    ]
    assert not failures, "\n".join(failures)
