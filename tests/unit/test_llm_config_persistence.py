"""Effective model settings survive JSON persistence and explicit overrides."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz import MemAgent
from memorizz.llms.huggingface import HuggingFaceLLM
from memorizz.llms.llm_factory import create_llm_provider
from memorizz.llms.openai import OpenAI
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


@pytest.fixture(autouse=True)
def clients(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", Mock())
    monkeypatch.setattr("openai.AzureOpenAI", Mock())
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Mock()))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Mock()))
    tokenizer = SimpleNamespace(model_max_length=32768)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=Mock(return_value=(object(), tokenizer))),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            pipeline=Mock(return_value=SimpleNamespace(tokenizer=tokenizer))
        ),
    )
    monkeypatch.setattr(
        HuggingFaceLLM, "_resolve_device_and_dtype", lambda *a: ("cpu", None)
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("OPENAI_API_VERSION", "test-version")


@pytest.mark.parametrize(
    "settings",
    [
        {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "temperature": 0,
            "seed": 42,
            "num_predict": 256,
            "timeout": 90,
            "additional_config": {
                "repeat_penalty": 1.1,
                "stop": ["END"],
                "api_key": "private-test-key",
            },
        },
        {
            "provider": "openai",
            "model": "local-model",
            "api_key": "private-test-key",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0,
            "max_tokens": 256,
            "seed": 42,
            "additional_config": {"response_format": {"type": "json_object"}},
        },
        {
            "provider": "anthropic",
            "api_key": "private-test-key",
            "temperature": 0,
            "max_tokens": 256,
            "enable_prompt_caching": False,
            "additional_config": {"stop_sequences": ["END"]},
        },
        {"provider": "azure", "deployment_name": "custom-deployment"},
        {
            "provider": "huggingface",
            "auth_token": "private-test-key",
            "max_new_tokens": 256,
            "temperature": 0,
            "local_files_only": False,
        },
        {"provider": "mlx", "max_new_tokens": 256, "temperature": 0},
    ],
)
def test_effective_settings_roundtrip_without_credentials(settings):
    original = create_llm_provider({**settings, "context_window_tokens": 8192})
    serialized = json.dumps(original.get_config())
    assert "private-test-key" not in serialized
    restored = create_llm_provider(json.loads(serialized))
    assert restored.get_context_window_tokens() == 8192
    assert restored.get_config() == original.get_config()
    for attribute in (
        "_request_options",
        "_max_tokens",
        "_enable_prompt_caching",
        "max_new_tokens",
        "temperature",
        "_timeout",
    ):
        if hasattr(original, attribute):
            assert getattr(restored, attribute) == getattr(original, attribute)
    if settings["provider"] == "ollama":
        assert restored._options == {
            key: value for key, value in original._options.items() if key != "api_key"
        }


def test_config_snapshot_does_not_share_mutable_generation_options():
    model = create_llm_provider(
        {"provider": "ollama", "additional_config": {"stop": ["END"]}}
    )
    snapshot = model.get_config()
    snapshot["additional_config"]["stop"].append("MUTATED")
    assert model._options["stop"] == ["END"]


@pytest.fixture
def saved_agent(tmp_path):
    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    model = OpenAI(
        model="original-model",
        base_url="http://localhost:1234/v1",
        context_window_tokens=8192,
    )
    agent = MemAgent(
        model=model,
        memory_provider=memory,
        context_window_tokens=4096,
        auto_register=False,
    )
    agent.save()
    yield agent, memory
    agent.close()


def test_save_load_keeps_agent_cap_and_provider_window(saved_agent):
    original, memory = saved_agent
    loaded = MemAgent.load(
        original.agent_id, memory_provider=memory, auto_register=False
    )
    assert loaded._context_window_tokens == 4096
    assert loaded.model.context_window_tokens == 8192


def test_llm_config_override_replaces_runtime_and_reported_configuration(saved_agent):
    original, memory = saved_agent
    config = {
        "provider": "openai",
        "model": "replacement",
        "base_url": "http://localhost:4321/v1",
        "context_window_tokens": 16384,
    }
    loaded = MemAgent.load(
        original.agent_id,
        memory_provider=memory,
        llm_config=config,
        auto_register=False,
    )
    assert loaded.llm_model == loaded.model.model == "replacement"
    assert loaded.model.base_url == config["base_url"]
    assert loaded._context_window_tokens == loaded.model.context_window_tokens == 16384


def test_explicit_budget_cap_can_accompany_model_override(saved_agent):
    original, memory = saved_agent
    replacement = OpenAI(model="replacement", context_window_tokens=16384)
    loaded = MemAgent.load(
        original.agent_id,
        memory_provider=memory,
        model=replacement,
        context_window_tokens=2048,
        auto_register=False,
    )
    assert loaded.model is replacement
    assert loaded._context_window_tokens == 2048


def test_budget_cannot_exceed_effective_provider_window():
    model = create_llm_provider({"provider": "ollama", "context_window_tokens": 8192})
    agent = MemAgent(
        model=model,
        context_window_tokens=128000,
        memory_provider=False,
        auto_register=False,
    )
    assert agent._context_window_tokens == 8192


def test_failed_override_never_falls_back_to_saved_model(saved_agent):
    original, memory = saved_agent
    loaded = MemAgent.load(
        original.agent_id,
        memory_provider=memory,
        llm_config={"provider": "invalid-provider", "model": "replacement"},
        auto_register=False,
    )
    assert loaded.model is None
    assert loaded.llm_config["provider"] == "invalid-provider"
    assert loaded._llm_init_error
