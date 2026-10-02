"""Every MemAgentModel field survives a store and reload on each backend."""

from __future__ import annotations

import pytest

from memorizz.memagent.models import MemAgentModel

# One non-default value for every stored field.
STORED = {
    "name": "Round trip",
    "application_id": "app-1",
    "instruction": "Be precise.",
    "application_mode": "workflow",
    "max_steps": 7,
    "tool_access": "global",
    "knowledge_base_ids": ["kb-1"],
    "delegates": ["agent-2"],
    "embedding_config": {"dimensions": 3},
    "semantic_cache": True,
    "semantic_cache_config": {"threshold": 0.9},
    "tool_result_policy": {"max_chars": 100},
    "context_policy": {"sticky_tool_limit": 2},
    "completion_policy": {"mode": "final"},
    "retrieval_policy": {"top_k": 4},
    "delegation_config": {"enabled": True},
    "skill_retrieval": True,
    "skill_retrieval_config": {"limit": 2},
    "semantic_layer_config": {"enabled": True},
    "context_window_tokens": 32_000,
    "is_favorite": True,
    "internet_access_provider": "tavily",
    "sandbox_provider": "e2b",
    "meta_harness": True,
    "meta_harness_mode": "delegate",
    "default_harness": "codex",
    "harness_config": {"workspace": "/w"},
    "skill_paths": ["/skills"],
    "mcp_servers": [{"name": "docs"}],
    "self_aware": True,
    "self_aware_config": {"allow_deletes": False},
    "continual_learning": True,
    "continual_learning_config": {"cadence": "daily"},
    "learning_control_plane": True,
    "learning_control_plane_config": {"shadow": True},
    "automations_enabled": False,
    "default_timezone": "Europe/London",
    "whatsapp_enabled": True,
    "whatsapp_config": {"number": "x"},
}


@pytest.mark.unit
def test_from_document_restores_every_field_and_defaults() -> None:
    agent = MemAgentModel.from_document({**STORED, "_id": "abc", "memory_ids": None})
    for name, value in STORED.items():
        assert getattr(agent, name) == value, name
    assert agent.agent_id == "abc" and agent.memory_ids == []
    bare = MemAgentModel.from_document({"_id": 5, "max_steps": None, "is_favorite": 1})
    assert bare.agent_id == "5"
    assert bare.max_steps == MemAgentModel().max_steps
    assert bare.is_favorite is True and bare.application_mode == "assistant"


def _roundtrip(provider) -> MemAgentModel:
    provider.store_memagent(MemAgentModel(**STORED, agent_id="round-trip"))
    listed = {agent.agent_id: agent for agent in provider.list_memagents()}
    loaded = provider.retrieve_memagent("round-trip")
    for name, value in STORED.items():
        assert getattr(loaded, name) == value, name
        assert getattr(listed[loaded.agent_id], name) == value, name
    return loaded


@pytest.mark.unit
def test_filesystem_round_trip(tmp_path) -> None:
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    try:
        _roundtrip(provider)
    finally:
        provider.close()


@pytest.mark.unit
def test_mongodb_round_trip() -> None:
    mongomock = pytest.importorskip("mongomock")
    from memorizz.enums import MemoryType
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    # Same in-memory construction as test_mongodb_provider_dispatch.py.
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = mongomock.MongoClient()["roundtrip"]
    provider.memagent_collection = provider.db[MemoryType.MEMAGENT.value]
    provider._sync_agent_tools_to_toolbox = lambda *a, **k: None
    loaded = _roundtrip(provider)
    assert loaded.agent_id


@pytest.mark.unit
def test_ui_favorite_toggle_keeps_every_other_setting(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.store_memagent(MemAgentModel(**STORED, agent_id="starred"))
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            response = client.post(
                "/agents/starred/favorite", data={"is_favorite": "0"}
            )
            assert response.status_code == 302
        loaded = provider.retrieve_memagent("starred")
        assert loaded.is_favorite is False
        # Starring used to rebuild the model by hand and drop these.
        for name in (
            "meta_harness_mode",
            "default_harness",
            "harness_config",
            "completion_policy",
            "application_id",
        ):
            assert getattr(loaded, name) == STORED[name], name
    finally:
        provider.close()
