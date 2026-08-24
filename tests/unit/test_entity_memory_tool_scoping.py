"""Security and observability coverage for MemAgent entity-memory tools."""

import inspect
from unittest.mock import Mock

import pytest

from memorizz.memagent.core import MemAgent


def _assistant(mock_memory_provider):
    return MemAgent(
        instruction="Entity scope test",
        memory_provider=mock_memory_provider,
        application_mode="assistant",
        memory_ids=["primary-user-a"],
    )


def test_lookup_binds_current_memory_and_user_scope(mock_memory_provider):
    agent = _assistant(mock_memory_provider)
    expected = {
        "matches": [{"entity_id": "entity-a", "name": "user"}],
        "retrieval": {
            "retrieval_mode": "exact_fallback",
            "degraded": True,
            "degraded_reason": "vector_index_missing",
        },
    }
    lookup = Mock(return_value=expected)
    agent.entity_memory_manager.lookup_entities_with_diagnostics = lookup
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_lookup"]["function"]
    result = tool(query="user profile")

    assert result == expected
    lookup.assert_called_once_with(
        entity_id=None,
        name=None,
        query="user profile",
        limit=5,
        memory_id="primary-user-a",
        user_id="user-a",
    )


def test_upsert_persists_current_user_scope(mock_memory_provider):
    agent = _assistant(mock_memory_provider)
    upsert = Mock(return_value="entity-a")
    agent.entity_memory_manager.upsert_entity_from_tool = upsert
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    result = tool(
        name="user",
        entity_type="person",
        attributes={"role": "developer experience leader"},
    )

    assert result["entity_id"] == "entity-a"
    assert result["storage"] == {
        "memory_id_bound": True,
        "user_id_bound": True,
        "scope_override_rejected": False,
    }
    upsert.assert_called_once_with(
        entity_id=None,
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "developer experience leader"}],
        relations=None,
        metadata=None,
        memory_id="primary-user-a",
        user_id="user-a",
    )


def test_upsert_schema_does_not_expose_memory_scope(mock_memory_provider):
    agent = _assistant(mock_memory_provider)
    upsert = Mock(return_value="entity-a")
    agent.entity_memory_manager.upsert_entity_from_tool = upsert
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    assert "memory_id" not in inspect.signature(tool).parameters
    with pytest.raises(TypeError, match="memory_id"):
        tool(
            name="user",
            attributes={"role": "attacker-selected"},
            memory_id="primary-user-b",
        )

    upsert.assert_not_called()
