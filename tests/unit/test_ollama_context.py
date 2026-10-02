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
    # Context lengths are cached per process; a real daemon must not leak in.
    monkeypatch.setattr("memorizz.llms.ollama._MODEL_PROFILES", {})


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


def _client_reporting(length, **layout):
    import ollama

    info = SimpleNamespace(modelinfo={"qwen2.context_length": length, **layout})
    ollama.Client = Mock(return_value=Mock(show=Mock(return_value=info)))


QWEN_7B = {  # 56 KB of cache per token
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.embedding_length": 3584,
}


def _ram(monkeypatch, gib):
    from memorizz.llms import ollama as provider_module

    monkeypatch.setattr(
        provider_module,
        "_local_memory_bytes",
        lambda host: None if "gpu-box" in str(host) else gib * 2**30,
    )


def test_unconfigured_window_is_the_models_length_when_it_fits(monkeypatch):
    from memorizz.llms import ollama as provider_module

    _ram(monkeypatch, 16)
    _client_reporting(32768, **QWEN_7B)
    assert OllamaLLM(model="qwen2.5:7b").get_context_window_tokens() == 32768

    # 131k of 56 KB/token is ~7.3 GB: more than a quarter of 16 GB, so the
    # largest standard size that fits. 64 GB fits all of it.
    monkeypatch.setattr(provider_module, "_MODEL_PROFILES", {})
    _client_reporting(131072, **QWEN_7B)
    assert OllamaLLM(model="big").get_context_window_tokens() == 65536
    monkeypatch.setattr(provider_module, "_MODEL_PROFILES", {})
    _ram(monkeypatch, 64)
    assert OllamaLLM(model="big").get_context_window_tokens() == 131072

    monkeypatch.setattr(provider_module, "_MODEL_PROFILES", {})
    _client_reporting(4096, **QWEN_7B)
    assert OllamaLLM(model="tiny").get_context_window_tokens() == 4096


def test_unknown_layout_or_remote_daemon_keeps_the_16k_default(monkeypatch):
    from memorizz.llms import ollama as provider_module

    _ram(monkeypatch, 64)
    _client_reporting(131072)  # no layout metadata
    assert OllamaLLM(model="m").get_context_window_tokens() == 16384

    monkeypatch.setattr(provider_module, "_MODEL_PROFILES", {})
    _client_reporting(131072, **QWEN_7B)
    remote = OllamaLLM(model="m", host="http://gpu-box:11434")
    assert remote.get_context_window_tokens() == 16384


def test_cache_estimate_follows_sliding_shared_and_state_space_layers():
    from memorizz.llms.ollama import _kv_cache_bytes

    gemma = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 12,
        "gemma4.attention.head_count": 8,
        "gemma4.attention.head_count_kv": 2,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
        "gemma4.attention.shared_kv_layers": 6,
    }
    # Six layers keep their own cache: five sliding (fixed), one global.
    assert _kv_cache_bytes(gemma) == (2 * 1024 * 2, 5 * 2 * 512 * 2 * 512)
    hybrid = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 4,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": [0, 0, 0, 4],
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
    }
    assert _kv_cache_bytes(hybrid) == (4 * 512 * 2, 0)
    assert _kv_cache_bytes({"general.architecture": "x"}) is None


def test_default_window_is_not_saved_as_a_setting():
    _client_reporting(32768)
    config = OllamaLLM(model="qwen2.5:7b").get_config()
    assert "context_window_tokens" not in config
    assert "num_ctx" not in config["additional_config"]

    pinned = OllamaLLM(model="qwen2.5:7b", context_window_tokens=12288).get_config()
    assert pinned["context_window_tokens"] == 12288


def test_saved_agents_keep_only_explicit_caps():
    from memorizz import MemAgent
    from memorizz.memagent import persistence

    assert (
        persistence._config_context_window(
            {"provider": "ollama", "additional_config": {"num_ctx": 32768}}
        )
        == 32768
    )
    assert persistence._config_context_window({"provider": "openai"}) is None

    derived = MemAgent(memory_provider=False, auto_register=False)
    capped = MemAgent(
        memory_provider=False, auto_register=False, context_window_tokens=6000
    )
    try:
        # A budget derived from the model is not a cap the user chose.
        assert derived._context_window_cap is None
        assert capped._context_window_cap == 6000
    finally:
        derived.close()
        capped.close()


def test_only_a_daemon_on_this_machine_uses_its_memory():
    from memorizz.llms.ollama import _local_memory_bytes

    for host in ("http://localhost:11434", "127.0.0.1:11434", "http://[::1]:11434"):
        assert _local_memory_bytes(host) > 0
    assert _local_memory_bytes("http://gpu-box:11434") is None
