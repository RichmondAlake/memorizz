"""Unit tests for MemAgent manager components."""
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from memorizz.enums import MemoryType, Role
from memorizz.memagent.managers import (
    CacheManager,
    MemoryManager,
    PersonaManager,
    ToolManager,
)
from tests.conftest import assert_memory_unit_valid


class TestMemoryManager:
    """Test MemoryManager functionality."""

    @pytest.mark.unit
    @pytest.mark.memory
    def test_memory_manager_initialization(self, mock_memory_provider):
        """Test memory manager initialization."""
        manager = MemoryManager(mock_memory_provider)

        assert manager.memory_provider == mock_memory_provider
        assert manager._conversation_memory_cache == {}

    @pytest.mark.unit
    @pytest.mark.memory
    def test_load_conversation_history(self, mock_memory_provider):
        """Test loading conversation history."""
        manager = MemoryManager(mock_memory_provider)

        # Mock the memory provider method
        mock_memory_provider.retrieve_conversation_history_ordered_by_timestamp.return_value = [
            {
                "content": {"role": "user", "content": "Hello"},
                "timestamp": "2023-01-01T12:00:00",
            },
            {
                "content": {"role": "assistant", "content": "Hi there!"},
                "timestamp": "2023-01-01T12:00:01",
            },
        ]

        history = manager.load_conversation_history("test_memory", limit=10)

        assert isinstance(history, list)
        assert len(history) >= 0
        mock_memory_provider.retrieve_conversation_history_ordered_by_timestamp.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.memory
    def test_load_conversation_history_is_thread_scoped(self):
        class Provider:
            def retrieve_conversation_history_ordered_by_timestamp(
                self,
                memory_id,
                memory_type=None,
                limit=None,
                user_id=None,
            ):
                # Deliberately lacks a native thread_id parameter: the manager
                # must fetch the whole scope and apply its compatibility filter.
                return [
                    {
                        "memory_id": memory_id,
                        "thread_id": "thread-a",
                        "role": "user",
                        "content": "from a",
                        "user_id": user_id,
                        "timestamp": "2026-01-01T12:00:00",
                    },
                    {
                        "memory_id": memory_id,
                        "thread_id": "thread-b",
                        "role": "user",
                        "content": "from b",
                        "user_id": user_id,
                        "timestamp": "2026-01-01T12:00:01",
                    },
                ]

        manager = MemoryManager(Provider())

        history = manager.load_conversation_history(
            "memory-1",
            limit=10,
            user_id=None,
            thread_id="thread-b",
        )

        assert [row["content"] for row in history] == ["from b"]

    @pytest.mark.unit
    @pytest.mark.memory
    def test_save_memory_unit(self, mock_memory_provider):
        """Test saving memory units."""
        manager = MemoryManager(mock_memory_provider)

        memory_unit = Mock()
        unit_id = manager.save_memory_unit(memory_unit, "test_memory")

        assert unit_id is not None
        mock_memory_provider.store.assert_called_once_with(
            memory_id="test_memory", memory_unit=memory_unit
        )

    @pytest.mark.unit
    @pytest.mark.memory
    def test_list_tool_logs_forwards_anonymous_scope_to_native_provider(self):
        sentinel = object()

        class Provider:
            def __init__(self):
                self.received_user_id = sentinel

            def list_tool_logs(
                self, memory_id=None, user_id=sentinel, limit=20, thread_id=None
            ):
                self.received_user_id = user_id
                return []

        provider = Provider()
        manager = MemoryManager(provider)

        assert manager.list_tool_logs("memory-1", user_id=None) == []
        assert provider.received_user_id is None

    @pytest.mark.unit
    @pytest.mark.memory
    def test_retrieve_relevant_memories(self, mock_memory_provider):
        """Test retrieving relevant memories."""
        manager = MemoryManager(mock_memory_provider)

        mock_memory_provider.retrieve_by_query.side_effect = None
        mock_memory_provider.retrieve_by_query.return_value = [
            {
                "content": "relevant memory 1",
                "memory_id": "test_memory",
                "user_id": None,
            },
            {
                "content": "relevant memory 2",
                "memory_id": "test_memory",
                "user_id": None,
            },
        ]

        memories = manager.retrieve_relevant_memories(
            query="test query",
            memory_type=MemoryType.CONVERSATION_MEMORY,
            memory_id="test_memory",
            limit=5,
        )

        assert isinstance(memories, list)
        assert len(memories) == 2
        mock_memory_provider.retrieve_by_query.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.memory
    def test_create_conversation_memory_unit(self, mock_memory_provider):
        """Test creating conversation memory units."""
        manager = MemoryManager(mock_memory_provider)

        memory_unit = manager.create_conversation_memory_unit(
            role=Role.USER,
            content="Test message",
            thread_id="conv_123",
            memory_id="mem_123",
        )

        assert memory_unit is not None
        assert memory_unit.role == Role.USER.value
        assert memory_unit.content == "Test message"
        assert memory_unit.thread_id == "conv_123"
        assert memory_unit.memory_id == "mem_123"

    @pytest.mark.unit
    @pytest.mark.memory
    def test_clear_conversation_cache(self, mock_memory_provider):
        """Test clearing conversation cache."""
        manager = MemoryManager(mock_memory_provider)

        # Add something to cache
        manager._conversation_memory_cache["test_memory"] = ["cached_item"]

        # Clear specific memory
        manager.clear_conversation_cache("test_memory")
        assert "test_memory" not in manager._conversation_memory_cache

        # Add and clear all
        manager._conversation_memory_cache["mem1"] = ["item1"]
        manager._conversation_memory_cache["mem2"] = ["item2"]

        manager.clear_conversation_cache()
        assert len(manager._conversation_memory_cache) == 0


class TestToolManager:
    """Test ToolManager functionality."""

    @pytest.mark.unit
    def test_tool_manager_initialization(self, mock_memory_provider):
        """Test tool manager initialization."""
        manager = ToolManager(mock_memory_provider)

        assert manager.memory_provider == mock_memory_provider
        assert manager.tools == {}
        assert manager.toolbox is None

    @pytest.mark.unit
    def test_add_function_tool(self, mock_memory_provider):
        """Test adding a function as a tool."""
        manager = ToolManager(mock_memory_provider)

        def sample_tool(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        success = manager.add_tool(sample_tool, persist=False)

        assert success is True
        assert "sample_tool" in manager.tools
        assert manager.tools["sample_tool"]["type"] == "function"

    @pytest.mark.unit
    def test_execute_tool_function(self, mock_memory_provider):
        """Test executing a function tool."""
        manager = ToolManager(mock_memory_provider)

        def multiply_tool(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        manager.add_tool(multiply_tool)

        result, outcome = manager.execute_tool("multiply_tool", {"a": 3.0, "b": 4.0})

        assert result == 12.0
        assert outcome is None  # No workflow outcome for simple functions

    @pytest.mark.unit
    def test_execute_tool_function_with_null_arguments(self, mock_memory_provider):
        """Tool execution should treat null arguments as an empty mapping."""
        manager = ToolManager(mock_memory_provider)

        def ping_tool() -> str:
            return "pong"

        manager.add_tool(ping_tool)

        result, outcome = manager.execute_tool("ping_tool", None)

        assert result == "pong"
        assert outcome is None

    @pytest.mark.unit
    def test_execute_tool_function_with_json_null_string(self, mock_memory_provider):
        """JSON null argument payloads should not raise mapping errors."""
        manager = ToolManager(mock_memory_provider)

        def ping_tool() -> str:
            return "pong"

        manager.add_tool(ping_tool)

        result, outcome = manager.execute_tool("ping_tool", "null")

        assert result == "pong"
        assert outcome is None

    @pytest.mark.unit
    def test_execute_nonexistent_tool(self, mock_memory_provider):
        """Test executing a tool that doesn't exist."""
        manager = ToolManager(mock_memory_provider)

        result, outcome = manager.execute_tool("nonexistent_tool", {})

        assert "not found" in result.lower()
        assert outcome is None

    @pytest.mark.unit
    def test_get_tool_metadata(self, mock_memory_provider):
        """Test getting tool metadata."""
        manager = ToolManager(mock_memory_provider)

        def documented_tool(param1: str, param2: int = 5) -> str:
            """A well-documented tool.

            Args:
                param1: First parameter
                param2: Second parameter with default
            """
            return f"{param1}_{param2}"

        manager.add_tool(documented_tool)

        # Get metadata for specific tool
        metadata = manager.get_tool_metadata("documented_tool")

        assert isinstance(metadata, dict)
        assert metadata["name"] == "documented_tool"
        assert "documented" in metadata["description"].lower()
        assert "param1" in metadata["parameters"]
        assert "param2" in metadata["parameters"]

        # Get metadata for all tools
        all_metadata = manager.get_tool_metadata()
        assert isinstance(all_metadata, list)
        assert len(all_metadata) == 1

    @pytest.mark.unit
    def test_remove_tool(self, mock_memory_provider):
        """Test removing a tool."""
        manager = ToolManager(mock_memory_provider)

        def temp_tool():
            return "temp"

        manager.add_tool(temp_tool)
        assert "temp_tool" in manager.tools

        success = manager.remove_tool("temp_tool")

        assert success is True
        assert "temp_tool" not in manager.tools

        # Try removing non-existent tool
        success = manager.remove_tool("nonexistent")
        assert success is False

    @pytest.mark.unit
    def test_list_tools(self, mock_memory_provider):
        """Test listing all tools."""
        manager = ToolManager(mock_memory_provider)

        def tool1():
            pass

        def tool2():
            pass

        manager.add_tool(tool1)
        manager.add_tool(tool2)

        tool_names = manager.list_tools()

        assert isinstance(tool_names, list)
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert len(tool_names) == 2


class TestCacheManager:
    """Test CacheManager functionality."""

    @pytest.mark.unit
    def test_cache_manager_initialization_disabled(self):
        """Test cache manager initialization when disabled."""
        manager = CacheManager(enabled=False)

        assert manager.enabled is False
        assert manager.cache_instance is None

    @pytest.mark.unit
    def test_cache_manager_initialization_enabled(self):
        """Test cache manager initialization when enabled."""
        config = {"similarity_threshold": 0.85, "scope": "local"}

        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_class.return_value = mock_cache_instance

            manager = CacheManager(
                enabled=True,
                config=config,
                agent_id="test_agent",
                memory_id="test_memory",
            )

            assert manager.enabled is True
            assert manager.cache_instance == mock_cache_instance
            mock_cache_class.assert_called_once()

    @pytest.mark.unit
    def test_get_cached_response_disabled(self):
        """Test getting cached response when cache is disabled."""
        manager = CacheManager(enabled=False)

        response = manager.get_cached_response("test query", "session_123")

        assert response is None

    @pytest.mark.unit
    def test_get_cached_response_enabled(self):
        """Test getting cached response when cache is enabled."""
        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_instance.get.return_value = "cached response"
            mock_cache_class.return_value = mock_cache_instance

            manager = CacheManager(enabled=True, agent_id="test_agent")

            response = manager.get_cached_response("test query", "session_123")

            assert response == "cached response"
            # ``user_id`` is threaded through by default (None == legacy scope).
            mock_cache_instance.get.assert_called_once_with(
                query="test query", session_id="session_123", user_id=None
            )

    @pytest.mark.unit
    def test_cache_response(self):
        """Test caching a response."""
        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_class.return_value = mock_cache_instance

            manager = CacheManager(enabled=True, agent_id="test_agent")

            success = manager.cache_response(
                "test query", "test response", "session_123"
            )

            assert success is True
            mock_cache_instance.set.assert_called_once_with(
                query="test query",
                response="test response",
                session_id="session_123",
                user_id=None,
            )

    @pytest.mark.unit
    def test_clear_cache(self):
        """Test clearing the cache."""
        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_class.return_value = mock_cache_instance

            manager = CacheManager(enabled=True, agent_id="test_agent")

            # Clear entire cache
            manager.clear_cache()
            mock_cache_instance.clear.assert_called_once()

            # Clear session-specific cache — the cache manager now delegates
            # to ``clear(session_id=..., user_id=...)`` rather than a
            # non-existent ``clear_session`` helper.
            manager.clear_cache("session_123")
            mock_cache_instance.clear.assert_called_with(
                session_id="session_123", user_id=None
            )


class TestPersonaManager:
    """Test PersonaManager functionality."""

    @pytest.mark.unit
    def test_persona_manager_initialization(self, mock_memory_provider):
        """Test persona manager initialization."""
        manager = PersonaManager(mock_memory_provider)

        assert manager.memory_provider == mock_memory_provider
        assert manager.current_persona is None
        assert manager._persona_cache == {}

    @pytest.mark.unit
    def test_set_persona(self, mock_memory_provider, sample_persona):
        """Test setting a persona."""
        manager = PersonaManager(mock_memory_provider)

        success = manager.set_persona(sample_persona, "agent_123", save=False)

        assert success is True
        assert manager.current_persona == sample_persona
        assert "agent_123" in manager._persona_cache

    @pytest.mark.unit
    def test_export_persona(self, mock_memory_provider, sample_persona):
        """Test exporting persona."""
        manager = PersonaManager(mock_memory_provider)
        manager.set_persona(sample_persona, "agent_123", save=False)

        exported = manager.export_persona()

        assert isinstance(exported, dict)
        assert "name" in exported
        assert "role" in exported
        assert exported["name"] == sample_persona.name

    @pytest.mark.unit
    def test_delete_persona(self, mock_memory_provider, sample_persona):
        """Test deleting persona."""
        manager = PersonaManager(mock_memory_provider)
        manager.set_persona(sample_persona, "agent_123", save=False)

        success = manager.delete_persona("agent_123", save=False)

        assert success is True
        assert manager.current_persona is None
        assert "agent_123" not in manager._persona_cache

    @pytest.mark.unit
    def test_get_persona_prompt(self, mock_memory_provider, sample_persona):
        """Test generating persona prompt."""
        manager = PersonaManager(mock_memory_provider)
        manager.set_persona(sample_persona, "agent_123", save=False)

        prompt = manager.get_persona_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert sample_persona.name in prompt
