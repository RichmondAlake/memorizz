"""Ollama must receive the same context budget that MemAgent uses."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.llms.ollama import OllamaLLM

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_ollama(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Mock()))
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 8192),
        ({"context_window_tokens": 16384}, 16384),
        ({"additional_config": {"num_ctx": 12288}}, 12288),
        (
            {"context_window_tokens": 16384, "additional_config": {"num_ctx": 8192}},
            16384,
        ),
    ],
)
def test_context_budget_is_sent_to_ollama_and_survives_reload(config, expected):
    provider = OllamaLLM(model="qwen2.5:7b", **config)
    for instance in (provider, OllamaLLM(**provider.get_config())):
        assert instance.get_context_window_tokens() == expected
        request = instance._chat_kwargs(messages=[{"role": "user", "content": "hello"}])
        assert request["options"]["num_ctx"] == expected


def test_environment_context_is_honored_unless_explicitly_overridden(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "16384")
    provider = OllamaLLM()
    assert provider.get_context_window_tokens() == 16384
    assert provider._chat_kwargs()["options"]["num_ctx"] == 16384
    assert OllamaLLM(context_window_tokens=8192).get_context_window_tokens() == 8192


@pytest.mark.parametrize("value", [0, -1, "invalid", True])
def test_invalid_context_is_rejected_instead_of_silently_disagreeing(value):
    with pytest.raises(ValueError, match="context"):
        OllamaLLM(context_window_tokens=value)


def test_context_selection_preserves_generation_options():
    provider = OllamaLLM(
        temperature=0,
        num_predict=256,
        additional_config={"num_ctx": 16384, "repeat_penalty": 1.1},
    )
    assert provider._chat_kwargs()["options"] == {
        "temperature": 0,
        "num_predict": 256,
        "num_ctx": 16384,
        "repeat_penalty": 1.1,
    }


def test_cli_model_switch_refreshes_the_agent_history_budget():
    from memorizz import MemAgent
    from memorizz.cli.commands import _swap_model

    agent = MemAgent(memory_provider=False, auto_register=False)
    agent._context_window_tokens = 128_000
    session = SimpleNamespace(agent=agent, llm_config={})
    try:
        _swap_model(
            session, {"provider": "ollama", "model": "qwen2.5:7b"}, carry_key=False
        )
        assert agent._context_window_tokens == 8192
        assert agent.model._chat_kwargs()["options"]["num_ctx"] == 8192
    finally:
        agent.close()


def test_cli_resume_replaces_the_legacy_128k_history_budget(monkeypatch, tmp_path):
    from memorizz import MemAgent
    from memorizz.cli import agent_factory

    agent = MemAgent(memory_provider=False, auto_register=False)
    agent._context_window_tokens = 128_000
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "none")
    monkeypatch.setattr(MemAgent, "load", lambda *a, **k: agent)
    monkeypatch.setattr(
        agent_factory.cfg, "load_state", lambda: {"agent_id": "saved-agent"}
    )
    monkeypatch.setattr(agent_factory, "make_internet_provider", lambda: None)
    try:
        session = agent_factory.build_session_agent(
            llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
            memory_provider=object(),
        )
        assert session.agent is agent
        assert agent._context_window_tokens == 8192
        assert agent.model._chat_kwargs()["options"]["num_ctx"] == 8192
    finally:
        agent.close()
