"""Google Gemini as a model provider: the key, the request, the thought
signature a tool call carries, and where the console and CLI offer it."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memorizz import Gemini, OpenAI
from memorizz.llms import model_lists
from memorizz.llms.gateway import GEMINI_NO_SIGNATURE
from memorizz.llms.llm_factory import create_llm_provider
from memorizz.llms.model_catalog import DEFAULT_LLM_MODEL_BY_PROVIDER, LLM_MODEL_CATALOG
from memorizz.llms.streaming import chat_events
from memorizz.memagent.core import MemAgent

pytestmark = pytest.mark.unit

SIGNED = {"google": {"thought_signature": "sig-from-gemini"}}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    model_lists.clear_cache()
    yield
    model_lists.clear_cache()


def _call(call_id, name="lookup", extra=None):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }
    if extra:
        call["extra_content"] = extra
    return call


def _answer(text="hello"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
        model="gemini-4-hydrogen",
    )


def test_the_factory_builds_gemini_on_googles_endpoint_with_its_own_key():
    model = create_llm_provider(
        {"provider": "gemini", "model": "models/gemini-4-neon", "temperature": 0.1}
    )

    assert isinstance(model, Gemini)
    # The model list's "models/" prefix is not part of a request.
    assert model.model == "gemini-4-neon"
    assert str(model.client.base_url).startswith(
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert model.client.api_key == "gemini-test-key"
    assert model.api_mode == "chat_completions"
    assert model.get_context_window_tokens() == 1_048_576
    # What is saved holds the provider and the model, never the key.
    saved = model.get_config()
    assert saved["provider"] == "gemini" and saved["model"] == "gemini-4-neon"
    assert "api_key" not in saved and "base_url" not in saved
    assert isinstance(create_llm_provider(saved), Gemini)
    # "google" names the same provider; no model means the default one.
    assert create_llm_provider({"provider": "google"}).model == "gemini-4-hydrogen"


def test_the_key_falls_back_to_google_api_key_and_never_to_openais(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Gemini("gemini-4-hydrogen")

    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    assert Gemini("gemini-4-hydrogen").client.api_key == "google-test-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    assert Gemini("gemini-4-hydrogen").client.api_key == "gemini-test-key"
    assert Gemini("gemini-4-hydrogen", api_key="given").client.api_key == "given"


def test_a_tool_calling_step_goes_back_with_its_signature_or_a_stand_in():
    model = Gemini("gemini-4-hydrogen", reasoning_effort="low")
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = _answer()
    history = [
        {"role": "user", "content": "look it up"},
        # Two calls another model made: only the first needs a signature.
        {"role": "assistant", "content": None, "tool_calls": [_call("a"), _call("b")]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("c", extra=SIGNED)],
        },
        {"role": "tool", "tool_call_id": "c", "content": "3"},
    ]

    assert model.generate(history) == "hello"

    sent = model.client.chat.completions.create.call_args.kwargs
    first, second = sent["messages"][1]["tool_calls"]
    assert first["extra_content"] == {
        "google": {"thought_signature": GEMINI_NO_SIGNATURE}
    }
    assert "extra_content" not in second
    # Gemini's own signature goes back exactly as it came.
    assert sent["messages"][4]["tool_calls"][0]["extra_content"] == SIGNED
    assert sent["reasoning_effort"] == "low"
    # The caller's history is not changed.
    assert "extra_content" not in history[1]["tool_calls"][0]


def test_a_streamed_answer_asks_for_its_token_counts():
    model = Gemini("gemini-4-hydrogen")
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = iter(())

    with pytest.raises(Exception):
        list(model.generate_stream([{"role": "user", "content": "hi"}]))

    sent = model.client.chat.completions.create.call_args.kwargs
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


def test_other_providers_are_not_sent_geminis_fields():
    model = OpenAI(model="gpt-5", api_key="sk-test")
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = _answer()
    history = [
        {"role": "user", "content": "look it up"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("a", extra=SIGNED)],
        },
        {"role": "tool", "tool_call_id": "a", "content": "1"},
    ]

    model.generate(history)

    sent = model.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[1]["tool_calls"] == [_call("a")]
    assert history[1]["tool_calls"][0]["extra_content"] == SIGNED


def _chunk(*, calls=None, finish=None, content=None):
    delta = SimpleNamespace(content=content, tool_calls=calls, refusal=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=None)


def test_a_streamed_tool_call_keeps_its_signature_into_the_history():
    model = Gemini("gemini-4-hydrogen")
    stream = [
        _chunk(
            calls=[
                SimpleNamespace(
                    index=0,
                    id="call-1",
                    function=SimpleNamespace(name="lookup", arguments='{"q":'),
                    extra_content=SIGNED,
                )
            ]
        ),
        _chunk(
            calls=[
                SimpleNamespace(
                    index=0,
                    id=None,
                    function=SimpleNamespace(name=None, arguments='"x"}'),
                    extra_content=None,
                )
            ]
        ),
        _chunk(finish="tool_calls"),
    ]

    # What generate_stream sets before it reads the stream.
    model._last_usage, model._last_response_metadata = None, {}
    events = list(chat_events(model, iter(stream)))

    message = events[-1]["response"].choices[0].message
    assert events[-1]["type"] == "tool_calls"
    assert message.tool_calls[0].extra_content == SIGNED
    # The agent loop writes it onto the assistant's tool-call message.
    history: list = []
    MemAgent._append_assistant_tool_calls(history, message)
    assert history[0]["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            "extra_content": SIGNED,
        }
    ]
    # A call with nothing beside it is written as before.
    plain = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call-2", function=SimpleNamespace(name="lookup", arguments="{}")
            )
        ],
    )
    MemAgent._append_assistant_tool_calls(history, plain)
    assert history[1]["tool_calls"] == [_call("call-2")]


def test_the_catalogue_lists_gemini_models_with_a_default():
    values = [row["value"] for row in LLM_MODEL_CATALOG["gemini"]]

    assert values[:4] == [
        "gemini-4-neon",
        "gemini-4-hydrogen",
        "gemini-4-carbon",
        "gemini-4-argon",
    ]
    assert "gemini-3.8-flash" in values
    assert DEFAULT_LLM_MODEL_BY_PROVIDER["gemini"] in values
    assert len(values) == len(set(values))
    # Retired models and the ones that cannot chat are not offered.
    assert not [v for v in values if "image" in v or v.startswith("gemini-2.")]


def test_the_live_list_keeps_the_models_that_chat_newest_first(monkeypatch):
    asked = {}

    def listing(url, headers, timeout=4.0):
        asked.update(url=url, headers=headers)
        ids = [
            "models/gemini-3.8-flash",
            "models/gemini-4-neon",
            "models/gemini-3.1-flash-image",
            "models/gemini-3.8-flash-tts",
            "models/gemini-3.8-live",
            "models/gemini-embedding-2",
            "models/gemini-nano-banana-2.1",
            "models/gemini-omni-1.2-flash",
            "models/gemini-4-hydrogen",
            "models/lyria-3.5",
            "gemini-3.5-flash-lite",
        ]
        return {"data": [{"id": name} for name in ids]}

    monkeypatch.setattr(model_lists, "_get_json", listing)

    assert model_lists.latest_models("gemini") == [
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert asked["url"].endswith("/v1beta/openai/models")
    assert asked["headers"] == {"Authorization": "Bearer gemini-test-key"}
    assert "gemini" in model_lists.available_providers()
    # No key: nothing is asked and nothing is offered.
    model_lists.clear_cache()
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr(model_lists, "_get_json", lambda *a, **k: 1 / 0)
    assert model_lists.latest_models("gemini") == []
    assert "gemini" not in model_lists.available_providers()


def test_the_cli_picks_gemini_when_its_key_is_the_one_set(monkeypatch):
    from memorizz.cli import agent_factory

    for name in (
        "MEMORIZZ_DEFAULT_LLM_PROVIDER",
        "MEMORIZZ_DEFAULT_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert agent_factory.detect_llm_config() == {
        "provider": "gemini",
        "model": "gemini-4-hydrogen",
    }
    from memorizz.cli.commands import _LOGIN_PROVIDERS
    from memorizz.mcp_server.runtime import _CREATABLE_LLM_PROVIDERS

    assert ("gemini", "GEMINI_API_KEY") in [row[:2] for row in _LOGIN_PROVIDERS]
    assert "gemini" in _CREATABLE_LLM_PROVIDERS


@pytest.fixture()
def console(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_ENV_FILE", str(tmp_path / "home" / ".env"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            yield TestClient(create_app(), follow_redirects=False), provider
    finally:
        provider.close()


def test_settings_take_the_gemini_key_and_offer_it_as_the_default(console):
    client, _ = console
    from memorizz.ui.app import _build_settings_sections

    fields = {
        field["env"]: field
        for section in _build_settings_sections()
        for field in section["fields"]
    }
    key = fields["GEMINI_API_KEY"]
    assert key["label"] == "Google Gemini API key" and key["is_set"] is True
    # A key is never sent back to the page.
    assert key["current_value"] == ""
    providers = fields["MEMORIZZ_DEFAULT_LLM_PROVIDER"]["options"]
    assert {"value": "gemini", "label": "Google Gemini"} in providers

    html = client.get("/settings").text
    assert 'name="GEMINI_API_KEY"' in html
    assert "gemini-test-key" not in html
    # The default-model dropdown is filled from the same catalogue.
    assert '"gemini-4-hydrogen"' in html


def test_an_agent_is_created_on_gemini_from_the_form(console):
    client, provider = console

    form = client.get("/agents/new").text
    assert '<option value="gemini" >Google Gemini</option>' in form
    assert '<option value="gemini">Google Gemini</option>' in form
    assert "aistudio.google.com/apikey" in form

    created = client.post(
        "/agents/new",
        data={
            "agent_name": "Gem",
            "instruction": "Be brief.",
            "llm_provider": "gemini",
            "llm_model": "gemini-4-neon",
            "background_model": "gemini:gemini-3.5-flash-lite",
            "application_mode": "assistant",
        },
    )
    assert created.status_code in (302, 303), created.text[:300]
    (agent,) = provider.list_memagents()
    assert agent.name == "Gem"
    assert agent.llm_config == {"provider": "gemini", "model": "gemini-4-neon"}
    assert agent.background_llm_config == {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
    }
    # The playground's settings panel offers the provider too.
    page = client.get(f"/agents/{agent.agent_id}/playground").text
    assert '<option value="gemini" selected>Google Gemini</option>' in page


def test_a_first_agent_on_gemini_is_ready_with_either_key(console, monkeypatch):
    from memorizz.ui.helpers import first_agent_model

    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("MEMORIZZ_DEFAULT_LLM_MODEL", raising=False)
    chosen = first_agent_model()
    assert chosen["provider_title"] == "Google Gemini"
    assert chosen["model"] == "gemini-4-hydrogen" and chosen["ready"] is True

    monkeypatch.delenv("GEMINI_API_KEY")
    assert first_agent_model()["ready"] is False
    assert first_agent_model()["key_env"] == "GEMINI_API_KEY"
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    assert first_agent_model()["ready"] is True
    assert os.environ.get("GEMINI_API_KEY") is None


def test_a_turn_with_no_key_says_which_key_to_set(console, monkeypatch):
    client, provider = console
    from memorizz.memagent.models import MemAgentModel

    provider.store_memagent(
        MemAgentModel(
            agent_id="gem",
            name="Gem",
            llm_config={"provider": "gemini", "model": "gemini-4-neon"},
        )
    )
    monkeypatch.delenv("GEMINI_API_KEY")

    stream = client.post("/agents/gem/playground/stream", data={"query": "hello"})

    assert stream.status_code == 200
    done = [
        line
        for line in stream.text.splitlines()
        if line.startswith("data:") and '"type": "run.done"' in line
    ]
    assert len(done) == 1
    import json

    event = json.loads(done[0][len("data:") :])
    assert event["status"] == "error"
    assert event["error_code"] == "model_not_configured"
    # The page shows this sentence instead of a bare "Run error".
    assert "set GEMINI_API_KEY" in event["error_message"]
    assert "gemini-test-key" not in stream.text


def test_streamed_parallel_tool_calls_stay_separate_calls():
    # Google's stream numbers no tool call: each arrives whole under its own
    # id, and only the first carries the signature.
    model = Gemini("gemini-3.5-flash-lite")
    stream = [
        _chunk(
            calls=[
                SimpleNamespace(
                    id="call-lagos",
                    function=SimpleNamespace(
                        name="get_weather", arguments='{"city":"Lagos"}'
                    ),
                    extra_content=SIGNED,
                )
            ]
        ),
        _chunk(
            calls=[
                SimpleNamespace(
                    id="call-oslo",
                    function=SimpleNamespace(
                        name="get_weather", arguments='{"city":"Oslo"}'
                    ),
                )
            ]
        ),
        _chunk(finish="stop"),
    ]

    model._last_usage, model._last_response_metadata = None, {}
    events = list(chat_events(model, iter(stream)))

    calls = events[-1]["response"].choices[0].message.tool_calls
    assert [(c.id, c.function.name, c.function.arguments) for c in calls] == [
        ("call-lagos", "get_weather", '{"city":"Lagos"}'),
        ("call-oslo", "get_weather", '{"city":"Oslo"}'),
    ]
    assert calls[0].extra_content == SIGNED
    assert calls[1].extra_content is None


def _refusal(status, message):
    """The error the client raises for Google's answer, which is a list."""
    import json

    import httpx
    import openai

    response = httpx.Response(
        status,
        text=json.dumps([{"error": {"code": status, "message": message}}]),
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )
    kind = openai.RateLimitError if status == 429 else openai.InternalServerError
    return kind(f"Error code: {status}", response=response, body=None)


FREE_TIER_NONE = (
    "You exceeded your current quota, please check your plan and billing "
    "details. \n* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 0, model: gemini-4-neon, "
    "description: Generate content requests per day limit exceeded on Free Tier."
)
FREE_TIER_USED = FREE_TIER_NONE.replace("limit: 0", "limit: 5").replace(
    "per day", "per minute"
)


@pytest.mark.parametrize(
    "status, said, shown",
    [
        (429, FREE_TIER_NONE, "does not offer gemini-4-neon on the Gemini API free"),
        (429, FREE_TIER_USED, "(5: generate content requests per minute)"),
        (503, "This model is currently experiencing high demand.", "high demand"),
    ],
)
def test_a_refusal_from_google_says_why(status, said, shown):
    model = Gemini("gemini-4-neon")

    with patch.object(
        OpenAI, "_create_chat_completion", side_effect=_refusal(status, said)
    ):
        with pytest.raises(Exception) as caught:
            model.generate([{"role": "user", "content": "hello"}])

    assert caught.value.status_code == status
    assert shown in caught.value.detail
    assert shown in str(caught.value)
    assert "gemini-test-key" not in str(caught.value)


def test_the_playground_shows_why_google_refused_a_model(console):
    client, provider = console
    from memorizz.memagent.models import MemAgentModel

    provider.store_memagent(
        MemAgentModel(
            agent_id="gem",
            name="Gem",
            llm_config={"provider": "gemini", "model": "gemini-4-neon"},
        )
    )

    with patch.object(
        OpenAI, "_create_chat_completion", side_effect=_refusal(429, FREE_TIER_NONE)
    ):
        stream = client.post("/agents/gem/playground/stream", data={"query": "hello"})

    import json

    done = [
        json.loads(line[len("data:") :])
        for line in stream.text.splitlines()
        if line.startswith("data:") and '"type": "run.done"' in line
    ]
    assert len(done) == 1
    assert done[0]["status"] == "error"
    assert done[0]["error_code"] == "provider_error"
    assert "Turn on billing" in done[0]["error_message"]
