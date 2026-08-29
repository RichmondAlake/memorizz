"""Security and observability coverage for MemAgent entity-memory tools."""

import inspect
import json
from unittest.mock import Mock

import pytest

from memorizz import capabilities
from memorizz.memagent.core import MemAgent


def _assistant(mock_memory_provider):
    return MemAgent(
        instruction="Entity scope test",
        memory_provider=mock_memory_provider,
        application_mode="assistant",
        memory_ids=["primary-user-a"],
    )


def test_structured_entity_tool_capability_is_advertised():
    feature = capabilities()["features"]["structured_entity_memory_tools"]

    assert feature == {
        "available": True,
        "nested_input_schema": True,
        "attribute_alias_normalization": True,
    }


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
        attributes=[
            {
                "name": "role",
                "value": "developer experience leader",
                "confidence": 0.8,
            }
        ],
        relations=None,
        metadata=None,
        memory_id="primary-user-a",
        user_id="user-a",
    )


def test_upsert_normalizes_production_attribute_alias_payload(mock_memory_provider):
    agent = _assistant(mock_memory_provider)
    upsert = Mock(return_value="entity-a")
    agent.entity_memory_manager.upsert_entity_from_tool = upsert
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    result = tool(
        name="Richmond Alake",
        entity_type="person",
        attributes=[
            {"attribute": "name", "value": "Richmond Alake"},
            {"attribute": "role", "value": "AI Memory Engineer"},
        ],
    )

    assert result["entity_id"] == "entity-a"
    assert upsert.call_args.kwargs["attributes"] == [
        {"name": "name", "value": "Richmond Alake", "confidence": 0.8},
        {"name": "role", "value": "AI Memory Engineer", "confidence": 0.8},
    ]


def test_upsert_normalizes_list_shorthand_attribute_maps(mock_memory_provider):
    agent = _assistant(mock_memory_provider)
    upsert = Mock(return_value="entity-a")
    agent.entity_memory_manager.upsert_entity_from_tool = upsert
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    result = tool(
        name="project-a",
        entity_type="project",
        attributes=[{"status": "verified", "priority": 1}],
    )

    assert result["entity_id"] == "entity-a"
    assert upsert.call_args.kwargs["attributes"] == [
        {"name": "status", "value": "verified", "confidence": 0.8},
        {"name": "priority", "value": "1", "confidence": 0.8},
    ]


def test_upsert_tool_schema_exposes_required_nested_attribute_fields(
    mock_memory_provider,
):
    agent = _assistant(mock_memory_provider)
    metadata = agent.tool_manager.tools["entity_memory_upsert"]["metadata"]
    attribute_schema = metadata["input_schema"]["properties"]["attributes"]
    serialized = json.dumps(attribute_schema)

    array_schema = next(
        option for option in attribute_schema["anyOf"] if option.get("type") == "array"
    )
    item_schema = array_schema["items"]
    assert item_schema["required"] == ["name", "value"]
    assert set(item_schema["properties"]) == {"name", "value", "confidence", "source"}
    assert item_schema["additionalProperties"] is False
    assert "$ref" not in serialized
    assert "$defs" not in serialized


def test_upsert_schema_does_not_expose_host_scope_or_identity_authority(
    mock_memory_provider,
):
    agent = _assistant(mock_memory_provider)
    upsert = Mock(return_value="entity-a")
    agent.entity_memory_manager.upsert_entity_from_tool = upsert
    agent._current_memory_id = "primary-user-a"
    agent._current_user_id = "user-a"

    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    assert "memory_id" not in inspect.signature(tool).parameters
    assert "identity_key" not in inspect.signature(tool).parameters
    with pytest.raises(TypeError, match="memory_id"):
        tool(
            name="user",
            attributes={"role": "attacker-selected"},
            memory_id="primary-user-b",
        )

    upsert.assert_not_called()
    with pytest.raises(TypeError, match="identity_key"):
        tool(
            name="user",
            attributes={"role": "attacker-selected"},
            identity_key="authenticated_user",
        )

    upsert.assert_not_called()
