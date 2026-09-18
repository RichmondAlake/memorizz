"""Provider terminal states apply equally to streaming and synchronous runs."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.llms.streaming import ProviderStreamError, tool_response


@pytest.fixture
def agent(memagent_with_mocks, monkeypatch):
    agent = memagent_with_mocks
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(agent, "_build_llm_tools", lambda *a, **k: [])
    agent.cache_manager.cache_response = Mock()
    agent._record_interaction = Mock()
    return agent


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "text, reason, expected",
    [
        ("", "stop", "empty_response"),
        ("  \n", "stop", "empty_response"),
        ("The first step is", "length", "provider_length"),
        ("The first step is", "max_tokens", "provider_length"),
        ("The first step is", "max_output_tokens", "provider_length"),
    ],
)
def test_incomplete_output_never_completes_or_is_cached(
    agent, streaming, text, reason, expected
):
    agent.model.get_last_response_metadata = lambda: {"finish_reason": reason}
    agent.model.generate = lambda *a, **k: text

    def stream(*a, **k):
        yield {"type": "content", "content": text}
        yield {"type": "done", "content": text}

    agent.model.generate_stream = stream
    if streaming:
        events = list(agent.run_stream_events("Answer the current question."))
        assert events[-1]["status"] == "error"
        assert events[-1]["error_code"] == expected
        assert not any(e["type"] == "answer.done" for e in events)
        assert (
            "".join(e["delta"] for e in events if e["type"] == "answer.delta") == text
        )
    else:
        with pytest.raises(ProviderStreamError, match=expected):
            agent.run("Answer the current question.")
    agent.cache_manager.cache_response.assert_not_called()
    agent._record_interaction.assert_not_called()
    assert not agent.completion_policy_report()["accepted"]


def test_none_is_not_a_successful_answer(agent):
    agent.model.generate = lambda *a, **k: None
    with pytest.raises(ProviderStreamError, match="empty_response"):
        agent.run("hello")


def test_tool_only_response_can_complete_after_evidence(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_build_llm_tools",
        lambda *a, **k: [{"type": "function", "function": {"name": "lookup"}}],
    )
    tool = tool_response(
        [{"id": "lookup-1", "name": "lookup", "arguments": "{}"}], None
    )
    agent.model.generate = Mock(side_effect=[tool, "complete answer"])
    monkeypatch.setattr(
        agent,
        "_execute_and_record_tool_call",
        lambda call, messages, *a, **k: messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": "evidence"}
        ),
    )
    assert agent.run("Look it up.") == "complete answer"


def test_truncated_tool_arguments_are_not_executed(agent, monkeypatch):
    agent.model.get_last_response_metadata = lambda: {"finish_reason": "length"}
    agent.model.generate = Mock(
        return_value=tool_response(
            [{"id": "1", "name": "lookup", "arguments": '{"incomplete":'}], None
        )
    )
    execute = Mock()
    monkeypatch.setattr(agent, "_execute_and_record_tool_call", execute)
    with pytest.raises(ProviderStreamError, match="provider_length"):
        agent.run("Look it up.")
    execute.assert_not_called()


def test_cli_exits_with_actionable_error_instead_of_empty_success(agent, monkeypatch):
    from memorizz.cli import streaming

    agent.model.generate_stream = lambda *a, **k: iter(
        [{"type": "done", "content": ""}]
    )
    session = SimpleNamespace(agent=agent, memory_id=None, thread_id=None, user_id=None)
    monkeypatch.setattr(streaming, "save_stream_session", lambda session: "written")
    stdout, stderr = StringIO(), StringIO()
    assert streaming.consume_stream(session, "hello", stdout=stdout, stderr=stderr) == 1
    assert "model returned no answer" in stderr.getvalue()
    assert not stdout.getvalue().strip()


def test_nonstreaming_cli_reports_empty_answer_without_traceback(agent, monkeypatch):
    import importlib

    from typer.testing import CliRunner

    cli = importlib.import_module("memorizz.cli.app")
    agent.model.generate = lambda *a, **k: ""
    session = SimpleNamespace(agent=agent, memory_id=None, thread_id=None, user_id=None)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli, "_build_or_wizard", lambda *a, **k: session)
    result = CliRunner().invoke(cli.app, ["run", "--no-stream", "hello"])
    assert result.exit_code == 1
    assert "empty_response" in result.stderr
    assert "model returned no answer" in result.stderr
    assert "Traceback" not in result.output
