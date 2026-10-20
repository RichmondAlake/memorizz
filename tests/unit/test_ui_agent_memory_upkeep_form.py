"""The agent form's Memory upkeep group: background model, sealed
conversations and background consolidation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers.agents_crud import (
    _parse_background_model,
    _retrieval_policy_with_privacy,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "ui-upkeep", lazy_vector_indexes=True
        )
    )
    patcher = patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    )
    patcher.start()
    yield provider
    patcher.stop()


def _form(**overrides):
    form = {
        "agent_name": "Upkeep Agent",
        "instruction": "You remember things.",
        "application_mode": "assistant",
        "max_steps": "10",
        "tool_access": "private",
        "llm_provider": "ollama",
        "llm_model": "gemma4",
        "llm_config_json": '{"base_url": "http://127.0.0.1:11434"}',
    }
    form.update(overrides)
    return form


def test_a_background_model_is_a_model_of_the_chat_provider_or_provider_model():
    chat = {
        "provider": "ollama",
        "model": "gemma4",
        "base_url": "http://h",
        "api_key": "k",
    }
    assert _parse_background_model("qwen2.5:3b", chat) == {
        "provider": "ollama",
        "base_url": "http://h",
        "model": "qwen2.5:3b",
    }
    assert _parse_background_model("ollama:qwen2.5:3b", chat)["model"] == "qwen2.5:3b"
    assert _parse_background_model("openai:gpt-4o-mini", chat) == {
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
    assert _parse_background_model("  ", chat) is None


def test_unsealing_only_undoes_a_seal():
    sealed = _retrieval_policy_with_privacy({"scoring": {"alpha_recency": 1.0}}, True)
    assert sealed["conversation_scope"] == sealed["entity_scope"] == "thread"
    opened = _retrieval_policy_with_privacy(sealed, False)
    assert opened == {"scoring": {"alpha_recency": 1.0}, "conversation_scope": "memory"}
    # A thread-only conversation scope someone chose is left as it is.
    assert _retrieval_policy_with_privacy({"conversation_scope": "thread"}, False) == {
        "conversation_scope": "thread"
    }


def test_create_and_edit_round_trip_the_memory_upkeep_group(client, connected):
    response = client.post(
        "/agents/new",
        data=_form(
            background_model="qwen2.5:3b",
            sealed_conversations="on",
            memory_consolidation="on",
            consolidation_every_n_turns="12",
        ),
    )
    assert response.status_code in (302, 303), response.text[:300]
    agent = connected.list_memagents()[0]
    assert agent.background_llm_config["model"] == "qwen2.5:3b"
    assert agent.background_llm_config["provider"] == "ollama"
    assert agent.retrieval_policy["entity_scope"] == "thread"
    assert agent.memory_consolidation["every_n_turns"] == 12
    assert agent.memory_consolidation["auto_apply"] is False  # box left unticked

    html = client.get(f"/agents/{agent.agent_id}/edit").text
    assert 'name="background_model" value="ollama:qwen2.5:3b"' in html
    assert 'name="consolidation_every_n_turns" value="12"' in html

    response = client.post(f"/agents/{agent.agent_id}/edit", data=_form())
    assert response.status_code in (302, 303), response.text[:300]
    agent = connected.retrieve_memagent(agent.agent_id)
    assert agent.background_llm_config is None
    assert agent.memory_consolidation is None
    assert "entity_scope" not in agent.retrieval_policy


def test_the_new_agent_form_shows_the_group(client, connected):
    html = client.get("/agents/new").text
    assert "Conversation privacy and memory upkeep" in html
    assert 'name="background_model" value=""' in html
    assert 'id="background_provider"' in html and 'name="entity_scope"' in html


def test_entity_facts_can_be_kept_per_conversation_without_sealing(client, connected):
    response = client.post("/agents/new", data=_form(entity_scope="thread"))
    assert response.status_code in (302, 303), response.text[:300]
    agent = connected.list_memagents()[0]
    assert agent.retrieval_policy["entity_scope"] == "thread"
    assert agent.retrieval_policy.get("conversation_scope", "memory") == "memory"
    html = client.get(f"/agents/{agent.agent_id}/edit").text
    assert '<option value="thread" selected>' in html

    client.post(f"/agents/{agent.agent_id}/edit", data=_form(entity_scope="memory"))
    assert "entity_scope" not in (
        connected.retrieve_memagent(agent.agent_id).retrieval_policy or {}
    )
