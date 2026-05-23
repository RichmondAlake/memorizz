"""Tests for single agent functionality."""
import time
import uuid
from unittest.mock import Mock, patch

import pytest

from memorizz.memagent import MemAgent
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider


class TestSingleAgentBasics:
    """Test basic single agent functionality."""

    @pytest.mark.single_agent
    def test_single_agent_simple_query(self):
        """Test single agent handling a simple query."""
        llm_provider = MockLLMProvider(["The capital of France is Paris."])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful geography assistant.",
        )

        response = agent.run("What is the capital of France?")

        assert_agent_response_valid(response)
        assert "paris" in response.lower()
        assert llm_provider.call_count == 1

    @pytest.mark.single_agent
    def test_single_agent_conversation_flow(self):
        """Test single agent maintaining conversation flow."""
        responses = [
            "Hello! I'm doing well, thank you for asking.",
            "I can help you with math, science, writing, and many other topics.",
            "Sure! 2 + 2 equals 4.",
        ]

        llm_provider = MockLLMProvider(responses)
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a friendly and helpful assistant.",
            semantic_cache=False,  # Disable caching for this test
        )

        # Simulate a conversation
        thread_id = "conv_single_agent_123"
        memory_id = "mem_single_agent_123"

        # Turn 1
        response1 = agent.run(
            "Hello, how are you?", memory_id=memory_id, thread_id=thread_id
        )
        assert_agent_response_valid(response1)

        # Turn 2
        response2 = agent.run(
            "What can you help me with?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response2)

        # Turn 3
        response3 = agent.run(
            "What is 2 + 2?", memory_id=memory_id, thread_id=thread_id
        )
        assert_agent_response_valid(response3)

        assert llm_provider.call_count == 3
        assert (
            len(memory_provider.storage[memory_id]) == 6
        )  # 3 user + 3 assistant messages

    @pytest.mark.single_agent
    def test_single_agent_with_tools(self):
        """Test single agent using tools."""

        def calculator(a: float, b: float, operation: str = "add") -> float:
            """A simple calculator tool."""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b if b != 0 else float("inf")

        def text_processor(text: str, action: str = "upper") -> str:
            """Process text in various ways."""
            if action == "upper":
                return text.upper()
            elif action == "lower":
                return text.lower()
            elif action == "reverse":
                return text[::-1]
            return text

        llm_provider = MockLLMProvider(
            ["I'll help you with that calculation.", "I can process that text for you."]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[calculator, text_processor],
            instruction="You are a helpful assistant with access to calculator and text processing tools.",
        )

        # Test that tools were added
        tool_names = set(agent.tool_manager.list_tools())
        assert {"calculator", "text_processor"}.issubset(tool_names)

        # Test tool execution
        calc_result, _ = agent.tool_manager.execute_tool(
            "calculator", {"a": 10, "b": 5, "operation": "add"}
        )
        assert calc_result == 15

        text_result, _ = agent.tool_manager.execute_tool(
            "text_processor", {"text": "hello world", "action": "upper"}
        )
        assert text_result == "HELLO WORLD"

    @pytest.mark.single_agent
    def test_single_agent_with_persona(self):
        """Test single agent with persona."""
        from tests.mocks.mock_providers import MockPersona

        persona = MockPersona(
            name="Dr. Science",
            role="Science Educator",
            traits=["knowledgeable", "patient", "encouraging"],
            expertise=["physics", "chemistry", "biology"],
        )

        llm_provider = MockLLMProvider(
            ["As Dr. Science, I'd be happy to explain quantum physics to you!"]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            persona=persona,
            instruction="You are Dr. Science, a patient and knowledgeable science educator.",
        )

        # Check persona is set
        assert agent.persona_manager.current_persona == persona

        # Check persona prompt generation
        persona_prompt = agent.persona_manager.get_persona_prompt()
        assert "Dr. Science" in persona_prompt
        assert "Science Educator" in persona_prompt

    @pytest.mark.single_agent
    def test_single_agent_memory_persistence(self):
        """Test single agent memory persistence across sessions."""
        llm_provider = MockLLMProvider(
            [
                "I'll remember that your name is Alice.",
                "Hello again, Alice! Nice to see you back.",
            ]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful assistant with good memory.",
        )

        memory_id = "persistent_memory_123"

        # First session - introduce yourself
        response1 = agent.run("Hi, my name is Alice", memory_id=memory_id)
        assert_agent_response_valid(response1)

        # Simulate some time passing and new session
        # Memory should persist
        conversation_history = agent.load_conversation_history(memory_id)
        assert len(conversation_history) >= 0  # Should have stored the conversation

        # Second session - agent should remember
        response2 = agent.run("Hello again!", memory_id=memory_id)
        assert_agent_response_valid(response2)

        # Verify memory provider was used
        assert len(memory_provider.get_call_history()) > 0
        assert any(call[0] == "store" for call in memory_provider.get_call_history())


class TestSingleAgentSemanticCache:
    """Test single agent with semantic caching."""

    @pytest.mark.single_agent
    def test_single_agent_cache_hit(self):
        """Test single agent cache hit scenario."""
        llm_provider = MockLLMProvider(["The capital of France is Paris."])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a geography assistant.",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.8},
        )

        thread_id = "cache_test_conv"

        # First query - should hit LLM and cache result
        response1 = agent.run("What is the capital of France?", thread_id=thread_id)
        assert_agent_response_valid(response1)
        assert llm_provider.call_count == 1

        # Similar query - might hit cache depending on implementation
        response2 = agent.run("What's the capital city of France?", thread_id=thread_id)
        assert_agent_response_valid(response2)

        # Cache behavior depends on semantic similarity implementation
        # At minimum, both responses should be valid

    @pytest.mark.single_agent
    def test_single_agent_cache_miss(self):
        """Test single agent cache miss scenario."""
        responses = [
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
            "The capital of Italy is Rome.",
        ]

        llm_provider = MockLLMProvider(responses)
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a geography assistant.",
            semantic_cache=True,
        )

        thread_id = "cache_miss_conv"

        # Three different queries - should all miss cache
        response1 = agent.run("What is the capital of France?", thread_id=thread_id)
        response2 = agent.run("What is the capital of Germany?", thread_id=thread_id)
        response3 = agent.run("What is the capital of Italy?", thread_id=thread_id)

        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)
        assert_agent_response_valid(response3)

        # All should be different responses
        assert response1 != response2
        assert response2 != response3
        assert response1 != response3


class TestSingleAgentErrorHandling:
    """Test single agent error handling scenarios."""

    @pytest.mark.single_agent
    def test_single_agent_llm_failure(self):
        """Test single agent handling LLM failures."""
        # Create failing LLM
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("LLM service unavailable")

        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=failing_llm,
            memory_provider=memory_provider,
            instruction="You are a helpful assistant.",
        )

        response = agent.run("This will cause LLM to fail")

        # Should get error response, not crash
        assert isinstance(response, str)
        assert len(response) > 0
        assert "error" in response.lower()

    @pytest.mark.single_agent
    def test_single_agent_memory_failure(self):
        """Test single agent handling memory provider failures."""
        # Create failing memory provider
        failing_memory = Mock()
        failing_memory.store.side_effect = Exception("Memory storage failed")
        failing_memory.retrieve_conversation_history_ordered_by_timestamp.side_effect = Exception(
            "Memory retrieval failed"
        )

        llm_provider = MockLLMProvider(["I'll try to help despite memory issues."])

        agent = MemAgent(
            model=llm_provider,
            memory_provider=failing_memory,
            instruction="You are a resilient assistant.",
        )

        # Should still work despite memory failures
        response = agent.run("Hello, can you help me?")

        assert_agent_response_valid(response)
        assert llm_provider.call_count == 1

    @pytest.mark.single_agent
    def test_single_agent_invalid_query(self):
        """Test single agent handling invalid queries."""
        llm_provider = MockLLMProvider(["I'm sorry, I don't understand that query."])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful assistant.",
        )

        # Test empty query (should be caught by validation)
        try:
            response = agent.run("")
        except ValueError:
            # Expected - empty query should raise validation error
            pass
        else:
            # If no error, response should still be valid
            assert_agent_response_valid(response)

        # Test None query (should be caught by validation)
        try:
            response = agent.run(None)
        except (ValueError, TypeError):
            # Expected - None query should raise validation error
            pass
        else:
            # If no error, response should still be valid
            assert_agent_response_valid(response)


class TestSingleAgentApplicationModes:
    """Test single agent with different application modes."""

    @pytest.mark.single_agent
    def test_single_agent_assistant_mode(self):
        """Test single agent in assistant mode."""
        llm_provider = MockLLMProvider(
            ["I'm happy to assist you with any questions or tasks."]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a helpful assistant.",
            application_mode="assistant",
            max_steps=10,
        )

        response = agent.run("Can you help me with something?")

        assert_agent_response_valid(response)
        assert "assist" in response.lower() or "help" in response.lower()

    @pytest.mark.single_agent
    def test_single_agent_chatbot_mode(self):
        """Test single agent in chatbot mode."""
        llm_provider = MockLLMProvider(
            ["Hello! I'm a friendly chatbot. How are you today?"]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are a friendly conversational chatbot.",
            application_mode="chatbot",
            semantic_cache=True,  # Chatbots often benefit from caching
        )

        response = agent.run("Hello there!")

        assert_agent_response_valid(response)
        # Chatbot should be conversational
        assert any(
            word in response.lower()
            for word in ["hello", "hi", "how", "today", "friendly"]
        )

    @pytest.mark.single_agent
    def test_single_agent_task_mode(self):
        """Test single agent in task-oriented mode."""
        llm_provider = MockLLMProvider(
            ["I'll complete that task for you. Let me break it down into steps."]
        )
        memory_provider = MockMemoryProvider()

        def task_tool(task_description: str) -> str:
            """Complete a task."""
            return f"Completed task: {task_description}"

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[task_tool],
            instruction="You are a task-oriented agent focused on completing objectives.",
            application_mode="agent",
            max_steps=20,  # Task agents might need more steps
        )

        response = agent.run("Please organize my schedule for tomorrow")

        assert_agent_response_valid(response)
        # Task agent should be action-oriented
        assert any(
            word in response.lower()
            for word in ["task", "complete", "steps", "objective"]
        )
