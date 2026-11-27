"""OpenRouter and other model gateways: routing, keys and what is saved."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memorizz import ModelGateway, ModelRouting, OpenRouter
from memorizz.llms import gateway as gateway_module
from memorizz.llms.llm_factory import create_llm_provider


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        gateway_module, "openrouter_context_windows", lambda *a, **k: {"a/big": 900_000}
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")


def _answer(text="hello", model="a/b"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
        model=model,
    )


@pytest.mark.unit
def test_routing_becomes_the_fields_openrouter_takes():
    routing = ModelRouting(
        fallback_models=["c/d", "c/d", "e/f"],
        providers="anthropic",
        ignore=["slowco"],
        allow_fallbacks=False,
        sort="Throughput",
        data_collection="deny",
        zero_data_retention=True,
        max_price={"prompt": 1, "completion": 2},
    )
    model = OpenRouter("a/b", routing=routing)

    assert model._request_body() == {
        "usage": {"include": True},
        "models": ["a/b", "c/d", "e/f"],
        "provider": {
            "order": ["anthropic"],
            "ignore": ["slowco"],
            "allow_fallbacks": False,
            "sort": "throughput",
            "data_collection": "deny",
            "zdr": True,
            "max_price": {"prompt": 1.0, "completion": 2.0},
        },
    }
    assert OpenRouter("a/b")._request_body() == {"usage": {"include": True}}
    assert not ModelRouting()


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        {"sort": "cheapest"},
        {"data_collection": "maybe"},
        {"only": ["x"], "ignore": ["x"]},
        {"provider_order": ["x"]},
    ],
)
def test_routing_refuses_settings_it_does_not_know(bad):
    with pytest.raises(ValueError):
        ModelRouting.coerce(bad)


@pytest.mark.unit
def test_every_request_carries_the_routing_and_the_openrouter_key():
    model = OpenRouter("a/b", routing={"fallback_models": ["c/d"], "sort": "price"})
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = _answer()

    assert model.generate([{"role": "user", "content": "hi"}]) == "hello"
    # Plain text goes down the same road, so the rules are not skipped.
    assert model.generate_text("hi", instructions="Be brief.") == "hello"

    calls = model.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["model"] == "a/b"
        assert call.kwargs["extra_body"] == {
            "usage": {"include": True},
            "models": ["a/b", "c/d"],
            "provider": {"sort": "price"},
        }
    assert calls[1].kwargs["messages"][0] == {"role": "system", "content": "Be brief."}
    model.client.responses.create.assert_not_called()

    real = OpenRouter("a/b")
    assert real.client.api_key == "sk-or-test"
    assert str(real.client.base_url).startswith("https://openrouter.ai/api/v1")
    assert real.client.default_headers["X-OpenRouter-Title"] == "MemoRizz"


@pytest.mark.unit
def test_the_saved_config_rebuilds_the_model_and_holds_no_key():
    model = OpenRouter(
        "a/big",
        routing=ModelRouting(fallback_models=["c/d"], data_collection="deny"),
        temperature=0.2,
    )

    config = model.get_config()

    assert config == {
        "provider": "openrouter",
        "model": "a/big",
        "context_window_tokens": 900_000,
        "additional_config": {"temperature": 0.2},
        "routing": {"fallback_models": ["c/d"], "data_collection": "deny"},
    }
    assert "sk-or-test" not in str(config)
    rebuilt = create_llm_provider(config)
    assert isinstance(rebuilt, OpenRouter)
    assert rebuilt.get_config() == config


@pytest.mark.unit
def test_a_gateway_never_borrows_the_openai_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouter("a/b")

    local = ModelGateway("served-model", base_url="http://10.0.0.5:8000/v1/")
    assert local.client.api_key == "sk-local"
    assert local.get_config() == {
        "provider": "gateway",
        "model": "served-model",
        "context_window_tokens": 128_000,
        "base_url": "http://10.0.0.5:8000/v1",
    }

    monkeypatch.setenv("TEAM_GATEWAY_KEY", "team-key")
    keyed = create_llm_provider(
        {
            "provider": "gateway",
            "model": "served-model",
            "base_url": "https://gateway.example/v1",
            "api_key_env": "TEAM_GATEWAY_KEY",
            "extra_body": {"tier": "fast"},
        }
    )
    assert keyed.client.api_key == "team-key"
    assert keyed.get_config()["api_key_env"] == "TEAM_GATEWAY_KEY"
    assert "team-key" not in str(keyed.get_config())
    assert keyed._request_body() == {"tier": "fast"}
    with pytest.raises(ValueError, match="base_url"):
        ModelGateway("served-model")


@pytest.mark.unit
def test_routing_is_set_on_the_agent_and_saved_with_it(tmp_path):
    from memorizz.memagent import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    try:
        agent = MemAgent(model=OpenRouter("a/b"), memory_provider=provider)
        assert agent.get_model_routing() == {}

        same = agent.with_model_routing(
            ModelRouting(fallback_models=["c/d"], sort="latency")
        )
        assert same is agent
        agent.save()

        loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
        assert isinstance(loaded.model, OpenRouter)
        assert loaded.get_model_routing() == {
            "fallback_models": ["c/d"],
            "sort": "latency",
        }
        assert loaded.with_model_routing(None).get_model_routing() == {}

        plain = MemAgent(
            model=create_llm_provider({"provider": "openai", "model": "gpt-4o"}),
            memory_provider=provider,
            auto_register=False,
        )
        with pytest.raises(ValueError, match="gateway"):
            plain.with_model_routing({"sort": "price"})
    finally:
        provider.close()


@pytest.mark.unit
def test_the_charge_openrouter_reports_becomes_the_calls_cost(tmp_path):
    from memorizz.memagent import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.observability.pricing import DEFAULT_PRICING

    model = OpenRouter("some/unlisted-model")
    answer = _answer()
    answer.usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        prompt_tokens_details=None,
        cost=0.00425,
    )
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = answer

    model.generate([{"role": "user", "content": "hi"}])

    assert model.get_last_usage()["cost_usd"] == 0.00425
    # No rate card exists for the model; the reported charge is used as is.
    quote = DEFAULT_PRICING.quote(
        {
            "provider": "openrouter",
            "model": "some/unlisted-model",
            "provider_cost_usd": 0.00425,
        }
    )
    assert (quote["cost_usd"], quote["cost_status"]) == ("0.00425", "provider_reported")
    assert (
        DEFAULT_PRICING.quote(
            {"provider": "openrouter", "model": "some/unlisted-model"}
        )["cost_status"]
        == "unknown"
    )
    free = DEFAULT_PRICING.quote({"provider": "openrouter", "provider_cost_usd": 0})
    assert (free["cost_usd"], free["cost_status"]) == ("0", "provider_reported")

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    try:
        agent = MemAgent(
            model=model, memory_provider=provider, auto_register=False, streaming=False
        )
        agent.run("hi")
        usage = agent.last_run_usage if hasattr(agent, "last_run_usage") else None
        metadata = agent._model_trace_metadata()
        agent._price_call(metadata)
        assert metadata["cost_status"] == "provider_reported"
        assert "provider_cost_usd" not in metadata
        assert usage is None or usage.get("unpriced_calls", 0) == 0
    finally:
        provider.close()

    stream = OpenRouter("a/b")
    stream.client = MagicMock()
    stream._create_chat_completion({"model": "a/b", "messages": [], "stream": True})
    sent = stream.client.chat.completions.create.call_args.kwargs
    assert sent["stream_options"] == {"include_usage": True}


@pytest.mark.unit
def test_routing_from_a_console_form_is_checked_before_it_is_saved():
    from memorizz.ui.helpers import _parse_routing
    from memorizz.ui.routers.agents_crud import _parse_llm_config

    assert _parse_routing("") == (None, None)
    assert _parse_routing("{}") == (None, None)
    assert _parse_routing('{"sort": "price", "fallback_models": ["a/b"]}') == (
        {"fallback_models": ["a/b"], "sort": "price"},
        None,
    )
    assert _parse_routing('{"sort": "cheapest"}')[1].startswith("Routing: sort")
    assert _parse_routing("{not json")[1] == "The routing settings could not be read."

    config, error = _parse_llm_config(
        "openrouter", "a/b", '{"routing": {"data_collection": "deny"}}'
    )
    assert error is None and config["routing"] == {"data_collection": "deny"}
    # Another provider never carries OpenRouter's rules.
    config, error = _parse_llm_config(
        "openai", "gpt-4o", '{"routing": {"data_collection": "deny"}}'
    )
    assert error is None and "routing" not in config
    assert _parse_llm_config("openrouter", "a/b", '{"routing": {"sort": "x"}}')[1]
