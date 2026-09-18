"""The complete request must fit before any provider call is made."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from memorizz.llms.streaming import ProviderStreamError, tool_response
from memorizz.memagent.utils.prompt_budget import fit_prompt


def test_evicts_complete_old_turns_for_context_and_tool_definitions():
    messages = [
        {"role": "system", "content": "Keep the current question."},
        {"role": "user", "content": "old question " * 120},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "old", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": "old evidence " * 120},
        {"role": "assistant", "content": "old answer " * 120},
        {"role": "developer", "content": "Keep this instruction."},
        {"role": "user", "content": "CURRENT QUESTION with current context"},
    ]
    before = deepcopy(messages)
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "schema " * 400},
        }
    ]
    selected = fit_prompt(messages, tools, 2048)
    assert selected == [messages[0], messages[5], messages[6]]
    assert messages == before


@pytest.mark.parametrize("streaming", [False, True])
def test_tool_surface_is_budgeted_at_the_provider_boundary(
    memagent_with_mocks, streaming
):
    agent = memagent_with_mocks
    agent._context_window_tokens = 2048
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "old " * 800},
        {"role": "assistant", "content": "old answer " * 200},
        {"role": "user", "content": "current question"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "schema " * 500},
        }
    ]
    captured = []

    def generate(request, **kwargs):
        captured.append(deepcopy(request))
        return "answer"

    def generate_stream(request, **kwargs):
        captured.append(deepcopy(request))
        yield {"type": "content", "content": "answer"}
        yield {"type": "done", "content": "answer"}

    agent.model.generate = generate
    agent.model.generate_stream = generate_stream
    if streaming:
        list(
            agent._generate_stream_with_trace(
                messages, tools=tools, iteration=1, stage="test"
            )
        )
    else:
        agent._generate_with_trace(messages, tools=tools, iteration=1, stage="test")
    assert captured == [[messages[0], messages[-1]]]


def test_oversized_history_is_discarded_but_current_query_is_kept(memagent_with_mocks):
    agent = memagent_with_mocks
    agent._context_window_tokens = 8192
    messages = agent._build_prompt_messages(
        "system",
        "current query",
        {"conversation_history": [{"role": "assistant", "content": "fact " * 12000}]},
    )
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current query"},
    ]


def test_oversized_current_context_fails_without_sending_or_caching(
    memagent_with_mocks, monkeypatch
):
    agent = memagent_with_mocks
    agent._context_window_tokens = 8192
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    agent.cache_manager.cache_response = Mock()
    with pytest.raises(ProviderStreamError, match="context_window_exceeded"):
        agent.run("current query", context={"document": "context " * 10000})
    agent.model.generate.assert_not_called()
    agent.cache_manager.cache_response.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_growing_tool_evidence_is_checked_before_next_iteration(
    memagent_with_mocks, monkeypatch, streaming
):
    agent = memagent_with_mocks
    agent._context_window_tokens = 8192
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(
        agent,
        "_build_llm_tools",
        lambda *a, **k: [{"type": "function", "function": {"name": "lookup"}}],
    )
    agent.cache_manager.cache_response = Mock()
    response = tool_response(
        [{"id": "lookup-1", "name": "lookup", "arguments": "{}"}], None
    )

    def execute(call, messages, *a, **k):
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": "result " * 10000}
        )

    monkeypatch.setattr(agent, "_execute_and_record_tool_call", execute)
    agent.model.generate = Mock(return_value=response)
    agent.model.generate_stream = Mock(
        return_value=iter([{"type": "tool_calls", "response": response}])
    )
    if streaming:
        events = list(agent.run_stream_events("Look up the status."))
        assert events[-1]["status"] == "error"
        assert events[-1]["error_code"] == "context_window_exceeded"
        assert not any(e["type"] == "answer.done" for e in events)
        agent.model.generate_stream.assert_called_once()
    else:
        with pytest.raises(ProviderStreamError, match="context_window_exceeded"):
            agent.run("Look up the status.")
        agent.model.generate.assert_called_once()
    agent.cache_manager.cache_response.assert_not_called()


def test_required_tool_pair_is_never_partially_removed():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current question"},
        {"role": "assistant", "tool_calls": [{"id": "current"}]},
        {"role": "tool", "tool_call_id": "current", "content": "result " * 10000},
    ]
    original = deepcopy(messages)
    with pytest.raises(ProviderStreamError, match="context_window_exceeded"):
        fit_prompt(messages, None, 8192)
    assert messages == original
