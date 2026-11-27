"""The playground is home. With no agent in the store it is a first-run
view: a first message makes the first agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture()
def empty(tmp_path: Path, monkeypatch):
    """A console on a store with no agents, defaulting to an OpenAI model."""
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
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


def test_an_empty_store_opens_on_a_first_run_view(empty):
    client, _ = empty
    assert client.get("/").headers["location"] == "/playground"
    html = client.get("/playground").text

    assert "Start with your first agent" in html
    # A composer that makes the agent, with the model it will answer with.
    assert 'id="first-run-form" action="/playground/first-agent" method="post"' in html
    assert '<span class="mono">gpt-test</span> · OpenAI' in html
    # Other ways in, and what follows.
    assert html.count('class="chat-home-card first-run-card"') >= 3
    assert 'href="/agents/new?template=research"' in html
    assert html.count('class="first-run-step-number"') == 5
    assert ">Create agent</a>" in html
    # None of the page for a store that has agents.
    assert 'id="chat-home-form"' not in html and 'id="pg-table"' not in html
    assert "No agents yet" not in html


def test_without_the_models_key_the_page_says_what_to_set(empty, monkeypatch):
    client, _ = empty
    monkeypatch.delenv("OPENAI_API_KEY")
    html = client.get("/playground").text

    assert 'id="first-run-form"' not in html
    assert "<code>OPENAI_API_KEY</code>" in html
    assert 'href="/settings">Open Settings</a>' in html
    assert "Create an agent on another model" in html
    # A model on this machine needs no key.
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "qwen2.5:7b")
    local = client.get("/playground").text
    assert 'id="first-run-form"' in local and "· Ollama" in local


def test_a_first_message_makes_the_first_agent_and_opens_the_conversation(empty):
    client, provider = empty

    sent = client.post(
        "/playground/first-agent", data={"prompt": "  Help me plan my week & more  "}
    )
    assert sent.status_code == 303
    target = urlparse(sent.headers["location"])
    (agent,) = provider.list_memagents()
    agent_id = agent.agent_id if hasattr(agent, "agent_id") else agent["agent_id"]
    assert target.path == f"/agents/{agent_id}/playground"
    # The message goes with it, to be sent once the page opens.
    assert parse_qs(target.query) == {"prompt": ["Help me plan my week & more"]}

    stored = provider.retrieve_memagent(agent_id)
    assert stored.name == "Assistant"
    assert stored.llm_config == {"provider": "openai", "model": "gpt-test"}
    assert "long-term memory" in stored.instruction
    assert "entity_memory" in [str(value).lower() for value in stored.memory_types]

    # Sent again (a double press, another tab): the same agent, not a second.
    again = client.post("/playground/first-agent", data={"prompt": "Again"})
    assert urlparse(again.headers["location"]).path == target.path
    assert len(provider.list_memagents()) == 1
    # Now the playground is the chat home for the agents there are.
    home = client.get("/playground").text
    assert 'id="chat-home-form"' in home and "Start with your first agent" not in home


def test_no_message_still_opens_the_new_agent(empty):
    client, provider = empty
    sent = client.post("/playground/first-agent", data={"prompt": "   "})
    assert sent.status_code == 303
    assert "?" not in sent.headers["location"]
    assert len(provider.list_memagents()) == 1


def test_a_read_only_console_makes_nothing(empty):
    client, provider = empty
    with patch.dict(state._state, {"read_only": True}):
        html = client.get("/playground").text
        assert "this console is read-only" in html
        assert 'id="first-run-form"' not in html and ">Create agent</a>" not in html
        refused = client.post("/playground/first-agent", data={"prompt": "Hello"})
        assert refused.status_code == 403
    assert provider.list_memagents() == []


def test_an_existing_agent_takes_the_message_instead(empty):
    client, provider = empty
    provider.store_memagent(
        MemAgentModel(
            agent_id="vera",
            name="Vera",
            llm_config={"provider": "openai", "model": "m"},
        )
    )
    sent = client.post("/playground/first-agent", data={"prompt": "Hello"})
    assert sent.headers["location"] == "/agents/vera/playground?prompt=Hello"
    assert len(provider.list_memagents()) == 1
