"""Backward compatibility tests for the refactored MemAgent."""
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from memorizz.memagent import MemAgent, MemAgentModel
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider, MockPersona


class TestBackwardCompatibilityAPI:
    """Test that the refactored MemAgent maintains API compatibility."""

    @pytest.mark.compatibility
    def test_original_constructor_signature(self):
        """Test that original constructor parameters still work."""
        # Test all the original constructor parameters
        llm_provider = MockLLMProvider(["Test response"])
        memory_provider = MockMemoryProvider()

        # This should work exactly as before the refactor
        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful assistant.",
            max_steps=15,
            agent_id="compat_test_agent",
            memory_ids=["mem1", "mem2"],
            semantic_cache=False,
        )

        assert_agent_state_valid(agent)
        assert agent.instruction == "You are a helpful assistant."
        assert agent.max_steps == 15
        assert agent.agent_id == "compat_test_agent"
        assert agent.memory_ids == ["mem1", "mem2"]
        assert agent.model == llm_provider
        assert agent.memory_provider == memory_provider

    @pytest.mark.compatibility
    def test_legacy_llm_config_parameter(self):
        """Test that llm_config parameter still works."""
        with patch("memorizz.memagent.core.create_llm_provider") as mock_create:
            mock_llm = Mock()
            mock_create.return_value = mock_llm

            # Legacy way of creating agent with llm_config
            agent = MemAgent(
                llm_config={
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "test_key",
                },
                instruction="Test with llm_config",
            )

            mock_create.assert_called_once()
            assert agent.model == mock_llm

    @pytest.mark.compatibility
    def test_tools_parameter_backward_compatibility(self):
        """Test that tools parameter works as before."""

        def legacy_tool(x: int, y: int) -> int:
            """A legacy tool function."""
            return x + y

        def another_legacy_tool(text: str) -> str:
            """Another legacy tool."""
            return text.upper()

        # Test single tool
        agent1 = MemAgent(
            model=MockLLMProvider(["Tool added"]),
            tools=legacy_tool,
            instruction="Test single tool",
        )

        # Test list of tools
        agent2 = MemAgent(
            model=MockLLMProvider(["Tools added"]),
            tools=[legacy_tool, another_legacy_tool],
            instruction="Test multiple tools",
        )

        # Verify tools were added
        assert "legacy_tool" in agent1.tool_manager.list_tools()
        assert "legacy_tool" in agent2.tool_manager.list_tools()
        assert "another_legacy_tool" in agent2.tool_manager.list_tools()

    @pytest.mark.compatibility
    def test_persona_parameter_backward_compatibility(self):
        """Test that persona parameter works as before."""
        persona = MockPersona(
            name="Legacy Assistant",
            role="Helper",
            traits=["helpful", "friendly"],
            expertise=["general assistance"],
        )

        agent = MemAgent(
            model=MockLLMProvider(["I'm Legacy Assistant"]),
            persona=persona,
            instruction="You are a legacy assistant.",
        )

        # Verify persona was set
        assert agent.persona_manager.current_persona == persona

    @pytest.mark.compatibility
    def test_memory_types_parameter_compatibility(self):
        """Test that memory_types parameter is handled."""
        # This parameter might have been in the original but deprecated
        agent = MemAgent(
            model=MockLLMProvider(["Memory types handled"]),
            memory_provider=MockMemoryProvider(),
            memory_types=["conversation", "semantic"],  # Legacy parameter
            instruction="Test memory types",
        )

        assert_agent_state_valid(agent)
        # Should not break even if the parameter is not used

    @pytest.mark.compatibility
    def test_delegates_parameter_compatibility(self):
        """Test that delegates parameter doesn't break initialization."""
        delegate1 = MemAgent(
            model=MockLLMProvider(["Delegate 1"]),
            instruction="I am delegate 1",
            agent_id="delegate_1",
        )

        delegate2 = MemAgent(
            model=MockLLMProvider(["Delegate 2"]),
            instruction="I am delegate 2",
            agent_id="delegate_2",
        )

        # Main agent with delegates
        main_agent = MemAgent(
            model=MockLLMProvider(["I coordinate delegates"]),
            delegates=[delegate1, delegate2],
            instruction="I coordinate other agents",
        )

        assert_agent_state_valid(main_agent)
        # Delegates parameter should be accepted without error

    @pytest.mark.compatibility
    def test_verbose_parameter_compatibility(self):
        """Test that verbose parameter is accepted."""
        agent = MemAgent(
            model=MockLLMProvider(["Verbose response"]),
            instruction="Test verbose parameter",
            verbose=True,  # Legacy parameter
        )

        assert_agent_state_valid(agent)
        # Should not break initialization

    @pytest.mark.compatibility
    def test_embedding_parameters_compatibility(self):
        """Test that embedding-related parameters don't break initialization."""
        agent = MemAgent(
            model=MockLLMProvider(["Embedding test"]),
            memory_provider=MockMemoryProvider(),
            instruction="Test embedding parameters",
            embedding_provider="test_provider",
            embedding_config={"model": "test-embedding"},
        )

        assert_agent_state_valid(agent)
        # Should handle embedding parameters gracefully


class TestBackwardCompatibilityMethods:
    """Test that all original methods still work."""

    @pytest.mark.compatibility
    def test_run_method_signature(self):
        """Test that run method signature is unchanged."""
        agent = MemAgent(
            model=MockLLMProvider(["Run method works"]),
            memory_provider=MockMemoryProvider(),
            instruction="Test run method",
        )

        # Test all run method parameter combinations
        response1 = agent.run("Basic query")
        assert_agent_response_valid(response1)

        response2 = agent.run("Query with memory", memory_id="test_memory")
        assert_agent_response_valid(response2)

        response3 = agent.run("Query with conversation", thread_id="test_conv")
        assert_agent_response_valid(response3)

        response4 = agent.run("Full query", memory_id="test_mem", thread_id="test_conv")
        assert_agent_response_valid(response4)

    @pytest.mark.compatibility
    def test_load_conversation_history_method(self):
        """Test that load_conversation_history method works as before."""
        memory_provider = MockMemoryProvider()
        agent = MemAgent(
            model=MockLLMProvider(["History loaded"]),
            memory_provider=memory_provider,
            instruction="Test history loading",
            memory_ids=["test_memory"],
        )

        # Test without memory_id (should use default)
        history1 = agent.load_conversation_history()
        assert isinstance(history1, list)

        # Test with explicit memory_id
        history2 = agent.load_conversation_history("specific_memory")
        assert isinstance(history2, list)

    @pytest.mark.compatibility
    def test_add_tool_method(self):
        """Test that add_tool method works as before."""
        agent = MemAgent(
            model=MockLLMProvider(["Tool added"]), instruction="Test add_tool method"
        )

        def new_tool(x: str) -> str:
            return f"Processed: {x}"

        # Test adding tool (should return True for success)
        result = agent.add_tool(new_tool)
        assert result is True

        # Test with persist parameter
        def persistent_tool(y: int) -> int:
            return y * 3

        result2 = agent.add_tool(persistent_tool, persist=False)
        assert result2 is True

        # Verify tools were added
        assert "new_tool" in agent.tool_manager.list_tools()
        assert "persistent_tool" in agent.tool_manager.list_tools()

    @pytest.mark.compatibility
    def test_set_persona_method(self):
        """Test that set_persona method works as before."""
        agent = MemAgent(
            model=MockLLMProvider(["Persona set"]),
            instruction="Test set_persona method",
        )

        persona = MockPersona(
            name="Test Persona",
            role="Tester",
            traits=["thorough"],
            expertise=["testing"],
        )

        # Test setting persona (should return True for success)
        result = agent.set_persona(persona)
        assert result is True

        # Test with save parameter
        result2 = agent.set_persona(persona, save=False)
        assert result2 is True

        # Verify persona was set
        assert agent.persona_manager.current_persona == persona


class TestBackwardCompatibilityModels:
    """Test that model classes maintain compatibility."""

    @pytest.mark.compatibility
    def test_memagent_model_class(self):
        """Test that MemAgentModel class works as before."""
        # Test default initialization
        model = MemAgentModel()
        assert model.instruction is not None
        assert model.max_steps > 0
        assert model.tool_access is not None
        assert model.semantic_cache == False
        assert model.application_mode is not None

        # Test custom initialization
        custom_model = MemAgentModel(
            instruction="Custom instruction",
            max_steps=25,
            agent_id="custom_agent",
            semantic_cache=True,
            memory_ids=["mem1", "mem2"],
            application_mode="agent",
        )

        assert custom_model.instruction == "Custom instruction"
        assert custom_model.max_steps == 25
        assert custom_model.agent_id == "custom_agent"
        assert custom_model.semantic_cache == True
        assert custom_model.memory_ids == ["mem1", "mem2"]
        assert custom_model.application_mode == "agent"

    @pytest.mark.compatibility
    def test_memagent_model_serialization(self):
        """Test that model serialization still works."""
        model = MemAgentModel(
            instruction="Serialization test", max_steps=30, semantic_cache=True
        )

        # Should be able to serialize to dict
        model_dict = model.model_dump()
        assert isinstance(model_dict, dict)
        assert model_dict["instruction"] == "Serialization test"
        assert model_dict["max_steps"] == 30
        assert model_dict["semantic_cache"] == True


class TestBackwardCompatibilityBehavior:
    """Test that behavior remains the same after refactoring."""

    @pytest.mark.compatibility
    def test_conversation_flow_behavior(self):
        """Test that conversation flow behavior is unchanged."""
        memory_provider = MockMemoryProvider()
        llm_provider = MockLLMProvider(
            [
                "Hello! How can I help you?",
                "I understand you need assistance with Python.",
                "Based on our conversation, here's what I recommend.",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful Python assistant.",
        )

        memory_id = "compat_conversation"
        thread_id = "compat_session"

        # Simulate the same conversation flow as pre-refactor
        response1 = agent.run("Hi there!", memory_id=memory_id, thread_id=thread_id)
        response2 = agent.run(
            "I need help with Python",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        response3 = agent.run(
            "What do you recommend?",
            memory_id=memory_id,
            thread_id=thread_id,
        )

        # All responses should be valid
        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)
        assert_agent_response_valid(response3)

        # Conversation should be stored in memory
        stored_memories = memory_provider.storage[memory_id]
        assert len(stored_memories) == 6  # 3 user + 3 assistant messages

        # LLM should have been called for each turn
        assert llm_provider.call_count == 3

    @pytest.mark.compatibility
    def test_tool_usage_behavior(self):
        """Test that tool usage behavior is unchanged."""

        def calculator(a: float, b: float, operation: str = "add") -> float:
            """Calculator tool that works as before."""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b if b != 0 else float("inf")
            return 0

        agent = MemAgent(
            model=MockLLMProvider(["Calculation completed"]),
            tools=[calculator],
            instruction="You help with calculations.",
        )

        # Test tool execution (should work exactly as before)
        result, outcome = agent.tool_manager.execute_tool(
            "calculator", {"a": 10, "b": 5, "operation": "multiply"}
        )

        assert result == 50
        assert outcome is None  # Simple functions don't return workflow outcomes

        # Test different operations
        add_result, _ = agent.tool_manager.execute_tool(
            "calculator", {"a": 3, "b": 7, "operation": "add"}
        )
        assert add_result == 10

        div_result, _ = agent.tool_manager.execute_tool(
            "calculator", {"a": 15, "b": 3, "operation": "divide"}
        )
        assert div_result == 5

    @pytest.mark.compatibility
    def test_memory_behavior_consistency(self):
        """Test that memory behavior remains consistent."""
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=MockLLMProvider(["Memory test response"]),
            memory_provider=memory_provider,
            instruction="You have good memory.",
            memory_ids=["consistent_memory"],
        )

        # Store some information
        agent.run("Remember that I like coffee", memory_id="consistent_memory")
        agent.run("Also remember I work in tech", memory_id="consistent_memory")

        # Load conversation history (should work as before)
        history = agent.load_conversation_history("consistent_memory")
        assert isinstance(history, list)

        # Memory should contain the interactions
        stored_memories = memory_provider.storage["consistent_memory"]
        assert len(stored_memories) == 4  # 2 user + 2 assistant messages

        # New query should have access to previous context
        response = agent.run(
            "What do you know about me?", memory_id="consistent_memory"
        )
        assert_agent_response_valid(response)

    @pytest.mark.compatibility
    def test_error_handling_consistency(self):
        """Test that error handling behavior is consistent."""
        # Test with failing LLM (should handle gracefully as before)
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("LLM service error")

        agent = MemAgent(
            model=failing_llm,
            memory_provider=MockMemoryProvider(),
            instruction="Error handling test",
        )

        response = agent.run("This will cause an error")

        # Should get error response, not crash (same as pre-refactor)
        assert isinstance(response, str)
        assert len(response) > 0
        assert "error" in response.lower()

    @pytest.mark.compatibility
    def test_agent_id_behavior(self):
        """Test that agent_id behavior is consistent."""
        # Test automatic ID generation
        agent1 = MemAgent(
            model=MockLLMProvider(["Auto ID test"]), instruction="Auto ID test"
        )

        assert agent1.agent_id is not None
        assert len(agent1.agent_id) > 0

        # Test explicit ID setting
        agent2 = MemAgent(
            model=MockLLMProvider(["Explicit ID test"]),
            instruction="Explicit ID test",
            agent_id="explicit_test_id",
        )

        assert agent2.agent_id == "explicit_test_id"

        # Different agents should have different IDs
        agent3 = MemAgent(
            model=MockLLMProvider(["Another agent"]), instruction="Another agent"
        )

        assert agent1.agent_id != agent3.agent_id

    @pytest.mark.compatibility
    def test_semantic_cache_behavior(self):
        """Test that semantic cache behavior is consistent."""
        memory_provider = MockMemoryProvider()
        llm_provider = MockLLMProvider(
            ["Cache test response 1", "Cache test response 2"]
        )

        # Test with cache disabled (default behavior)
        agent_no_cache = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="No cache test",
            semantic_cache=False,
        )

        assert agent_no_cache.cache_manager.enabled == False

        # Test with cache enabled
        agent_with_cache = MemAgent(
            model=MockLLMProvider(["Cached response"]),
            memory_provider=MockMemoryProvider(),
            instruction="Cache test",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.8},
        )

        assert agent_with_cache.cache_manager.enabled == True

        # Basic functionality should work regardless of cache setting
        response1 = agent_no_cache.run("Test without cache")
        response2 = agent_with_cache.run("Test with cache")

        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)
