"""Model selection through the canonical /models command and its old alias."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document
from rich.console import Console

from memorizz.cli import commands
from memorizz.cli.agent_factory import Session
from memorizz.cli.repl import SlashCompleter

pytestmark = pytest.mark.unit


@pytest.fixture
def session():
    output = StringIO()
    config = {"provider": "ollama", "model": "old-model", "host": "http://local"}
    agent = SimpleNamespace(
        model=object(),
        llm_config=dict(config),
        _initialize_context_window_tokens=Mock(return_value=8192),
        memory_ids=["saved-memory"],
    )
    return (
        Session(
            agent=agent,
            provider=object(),
            llm_config=config,
            memory_id="saved-memory",
            thread_id="saved-thread",
            console=Console(file=output, color_system=None, width=140),
        ),
        output,
    )


def test_models_lists_current_and_installed_models(session, monkeypatch):
    active, output = session
    monkeypatch.setattr(
        commands.ollama_probe,
        "list_models",
        lambda: ["qwen3.5:latest", "gemma4:latest"],
    )
    assert commands.dispatch("/models", active) is True
    text = output.getvalue()
    assert "Model: old-model" in text
    assert "Available Ollama models:" in text
    assert "qwen3.5:latest" in text and "gemma4:latest" in text
    assert "Usage: /models <model-name>" in text


@pytest.mark.parametrize("name", ["models", "model", "mo"])
def test_models_switch_keeps_provider_and_conversation(session, monkeypatch, name):
    from memorizz.llms import llm_factory

    active, output = session
    replacement = object()
    factory = Mock(return_value=replacement)
    monkeypatch.setattr(llm_factory, "create_llm_provider", factory)

    assert commands.dispatch(f"/{name} qwen3.5:latest", active) is True
    factory.assert_called_once_with(
        {"provider": "ollama", "model": "qwen3.5:latest", "host": "http://local"}
    )
    assert active.agent.model is replacement
    assert active.model_name == "qwen3.5:latest"
    assert active.agent.llm_config == active.llm_config
    assert active.agent._context_window_tokens == 8192
    assert (active.memory_id, active.thread_id) == ("saved-memory", "saved-thread")
    assert active.agent.memory_ids == ["saved-memory"]
    assert "Model → qwen3.5:latest" in output.getvalue()


def test_help_and_completion_offer_models(session):
    active, output = session
    commands.dispatch("/help", active)
    text = output.getvalue()
    assert "/models [name]" in text
    assert "/provider [name]" in text
    assert "/model " not in text

    completer = SlashCompleter(commands.command_completions())
    offered = {item.text for item in completer.get_completions(Document("/mo"), None)}
    assert offered == {"/models"}
