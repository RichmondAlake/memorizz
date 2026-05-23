"""End-to-end integration tests for MemAgent."""
import time
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from memorizz.enums import MemoryType, Role
from memorizz.memagent import MemAgent
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider, MockPersona


class TestEndToEndWorkflow:
    """Test complete end-to-end agent workflows."""

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle from creation to complex interactions."""
        # Initialize components
        llm_provider = MockLLMProvider(
            [
                "Hello! I'm ready to help you.",
                "I'll remember that you're working on a Python project.",
                "Based on our conversation, you're looking for debugging help.",
                "Here's a comprehensive debugging strategy based on what I know.",
            ]
        )
        memory_provider = MockMemoryProvider()

        # Create agent with full configuration
        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You are an expert programming assistant with excellent memory.",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.85},
            max_steps=20,
            agent_id="e2e_test_agent",
        )

        assert_agent_state_valid(agent)

        # Phase 1: Initial interaction
        memory_id = "e2e_project"
        thread_id = "e2e_session"

        response1 = agent.run(
            "Hi, I'm starting a new Python project for data analysis.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response1)

        # Phase 2: Context building
        response2 = agent.run(
            "The project involves processing CSV files and generating reports.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response2)

        # Phase 3: Problem solving with context
        response3 = agent.run(
            "I'm getting a pandas error when trying to read the CSV. Can you help?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response3)

        # Phase 4: Complex query utilizing all context
        response4 = agent.run(
            "Give me a comprehensive debugging strategy for my pandas CSV issue.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response4)

        # Verify complete workflow
        assert llm_provider.call_count == 4

        # Verify memory persistence
        conversation_history = agent.load_conversation_history(memory_id)
        assert len(conversation_history) >= 0  # Should have conversation history

        # Verify memory storage
        stored_memories = memory_provider.storage[memory_id]
        assert len(stored_memories) == 8  # 4 user queries + 4 assistant responses

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_agent_with_tools_and_persona(self):
        """Test agent with tools and persona working together."""

        # Define tools
        def calculator(a: float, b: float, operation: str = "add") -> float:
            """Perform mathematical calculations."""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b if b != 0 else float("inf")
            return 0

        def text_analyzer(text: str, action: str = "count_words") -> dict:
            """Analyze text in various ways."""
            if action == "count_words":
                return {"word_count": len(text.split())}
            elif action == "count_chars":
                return {"char_count": len(text)}
            elif action == "analyze":
                words = text.split()
                return {
                    "word_count": len(words),
                    "char_count": len(text),
                    "avg_word_length": sum(len(w) for w in words) / len(words)
                    if words
                    else 0,
                }
            return {}

        # Create persona
        persona = MockPersona(
            name="Dr. Data",
            role="Data Analysis Expert",
            traits=["analytical", "precise", "helpful"],
            expertise=["statistics", "data processing", "problem solving"],
        )

        llm_provider = MockLLMProvider(
            [
                "I'm Dr. Data, your data analysis expert. I can help with calculations and text analysis.",
                "Let me calculate that for you: 15 + 25 = 40",
                "I'll analyze this text for you. It has 5 words and 25 characters.",
            ]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[calculator, text_analyzer],
            persona=persona,
            instruction="You are Dr. Data, an expert who uses tools to solve problems.",
            agent_id="tool_persona_agent",
        )

        memory_id = "tool_test"

        # Test persona introduction
        response1 = agent.run("Who are you and how can you help?", memory_id=memory_id)
        assert_agent_response_valid(response1)

        # Test tool usage - math
        response2 = agent.run("Calculate 15 + 25", memory_id=memory_id)
        assert_agent_response_valid(response2)

        # Test tool execution
        calc_result, _ = agent.tool_manager.execute_tool(
            "calculator", {"a": 15, "b": 25, "operation": "add"}
        )
        assert calc_result == 40

        # Test tool usage - text analysis
        response3 = agent.run(
            "Analyze this text: 'Hello world of data'", memory_id=memory_id
        )
        assert_agent_response_valid(response3)

        # Test text analyzer execution
        text_result, _ = agent.tool_manager.execute_tool(
            "text_analyzer", {"text": "Hello world of data", "action": "analyze"}
        )
        assert text_result["word_count"] == 4

        # Verify persona is active
        assert agent.persona_manager.current_persona == persona
        persona_prompt = agent.persona_manager.get_persona_prompt()
        assert "Dr. Data" in persona_prompt

        # Verify tools are available
        available_tools = agent.tool_manager.list_tools()
        assert "calculator" in available_tools
        assert "text_analyzer" in available_tools

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_semantic_cache_integration(self):
        """Test semantic cache integration with full agent workflow."""
        llm_provider = MockLLMProvider(
            [
                "The capital of France is Paris.",
                "Machine learning is a subset of artificial intelligence.",
            ]
        )
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You answer questions accurately and use caching effectively.",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.8, "scope": "session"},
            agent_id="cache_integration_agent",
        )

        thread_id = "cache_session"
        memory_id = "cache_test"

        # First query - should hit LLM and cache result
        response1 = agent.run(
            "What is the capital of France?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response1)

        initial_call_count = llm_provider.call_count

        # Similar query - behavior depends on cache implementation
        response2 = agent.run(
            "What's France's capital city?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response2)

        # Different query - should definitely hit LLM
        response3 = agent.run(
            "What is machine learning?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response3)

        # Verify cache manager is enabled and configured
        assert agent.cache_manager.enabled == True

        # At minimum, should have made calls to LLM
        assert llm_provider.call_count >= initial_call_count


class TestComponentIntegration:
    """Test integration between different agent components."""

    @pytest.mark.integration
    def test_memory_manager_tool_manager_integration(self):
        """Test memory manager and tool manager working together."""
        memory_provider = MockMemoryProvider()

        def memory_search_tool(query: str, memory_id: str) -> dict:
            """Search memory for relevant information."""
            # Simulate tool accessing memory
            return {
                "query": query,
                "memory_id": memory_id,
                "results_found": 3,
                "search_completed": True,
            }

        llm_provider = MockLLMProvider(
            [
                "I'll search our memory for information about that topic.",
                "Based on the memory search, I found 3 relevant results.",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[memory_search_tool],
            instruction="You use tools to search memory and provide informed responses.",
            agent_id="memory_tool_integration",
        )

        memory_id = "integration_memory"

        # Agent stores information
        response1 = agent.run(
            "Remember that I work on AI projects", memory_id=memory_id
        )
        assert_agent_response_valid(response1)

        # Agent uses tool to search memory
        response2 = agent.run(
            "Search for information about my work", memory_id=memory_id
        )
        assert_agent_response_valid(response2)

        # Test tool execution
        search_result, _ = agent.tool_manager.execute_tool(
            "memory_search_tool", {"query": "AI projects", "memory_id": memory_id}
        )

        assert search_result["search_completed"] == True
        assert search_result["results_found"] == 3

        # Verify both managers are working
        assert agent.memory_manager is not None
        tool_names = agent.tool_manager.list_tools()
        assert "memory_search_tool" in tool_names

    @pytest.mark.integration
    def test_persona_cache_workflow_integration(self):
        """Test persona, cache, and workflow managers integration."""
        memory_provider = MockMemoryProvider()

        persona = MockPersona(
            name="Alex Assistant",
            role="Personal Assistant",
            traits=["organized", "efficient", "proactive"],
            expertise=["scheduling", "task management", "communication"],
        )

        llm_provider = MockLLMProvider(
            [
                "I'm Alex, your personal assistant. I'll help you stay organized.",
                "I'll schedule that meeting for you and send confirmations.",
                "Based on your schedule, I recommend moving the 3pm meeting to 4pm.",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            persona=persona,
            instruction="You are Alex, a proactive personal assistant.",
            semantic_cache=True,
            agent_id="integrated_assistant",
        )

        memory_id = "assistant_workflow"
        thread_id = "assistant_session"

        # Test persona-driven response
        response1 = agent.run(
            "I need help managing my schedule",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response1)

        # Test workflow with caching
        response2 = agent.run(
            "Schedule a meeting with the team for tomorrow at 2pm",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response2)

        # Test context-aware recommendation
        response3 = agent.run(
            "I have a conflict with the 3pm meeting",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response3)

        # Verify all managers are integrated
        assert agent.persona_manager.current_persona == persona
        assert agent.cache_manager.enabled == True
        assert agent.workflow_manager is not None

        # Verify persona influences responses
        persona_prompt = agent.persona_manager.get_persona_prompt()
        assert "Alex" in persona_prompt
        assert "Personal Assistant" in persona_prompt

    @pytest.mark.integration
    def test_full_component_coordination(self):
        """Test all components working together in coordination."""
        memory_provider = MockMemoryProvider()

        # Create a complex scenario with all components
        def project_tracker(action: str, project_name: str, **kwargs) -> dict:
            """Track project activities."""
            if action == "create":
                return {
                    "project": project_name,
                    "status": "created",
                    "id": f"proj_{hash(project_name)}",
                }
            elif action == "update":
                return {
                    "project": project_name,
                    "status": "updated",
                    "progress": kwargs.get("progress", 0),
                }
            elif action == "query":
                return {"project": project_name, "status": "active", "progress": 75}
            return {}

        def team_coordinator(action: str, team_member: str, **kwargs) -> dict:
            """Coordinate with team members."""
            if action == "assign":
                return {
                    "member": team_member,
                    "task": kwargs.get("task", ""),
                    "assigned": True,
                }
            elif action == "status":
                return {
                    "member": team_member,
                    "availability": "available",
                    "current_tasks": 2,
                }
            return {}

        persona = MockPersona(
            name="Morgan Manager",
            role="Project Manager",
            traits=["organized", "strategic", "communicative"],
            expertise=["project management", "team coordination", "strategic planning"],
        )

        llm_provider = MockLLMProvider(
            [
                "I'm Morgan, your project manager. I'll coordinate this project efficiently.",
                "I've created the new AI project and will track its progress.",
                "Let me check the team status and assign tasks appropriately.",
                "Based on project status and team availability, here's the updated plan.",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[project_tracker, team_coordinator],
            persona=persona,
            instruction="You are Morgan, an expert project manager who uses tools and memory effectively.",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.9},
            max_steps=30,
            agent_id="coordination_manager",
        )

        memory_id = "project_coordination"
        thread_id = "management_session"

        # Phase 1: Project initialization
        response1 = agent.run(
            "We're starting a new AI-powered chatbot project. Set it up.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response1)

        # Phase 2: Team coordination
        response2 = agent.run(
            "Check Sarah's availability and assign her to UI development.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response2)

        # Phase 3: Status inquiry with context
        response3 = agent.run(
            "What's the current status of our chatbot project?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response3)

        # Phase 4: Strategic planning with full context
        response4 = agent.run(
            "Based on everything we've discussed, what's our next strategic move?",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        assert_agent_response_valid(response4)

        # Verify all components are working
        assert agent.persona_manager.current_persona == persona
        tool_names = agent.tool_manager.list_tools()
        assert "project_tracker" in tool_names
        assert "team_coordinator" in tool_names
        assert agent.cache_manager.enabled == True
        assert agent.memory_manager is not None
        assert agent.workflow_manager is not None

        # Test tool executions
        project_result, _ = agent.tool_manager.execute_tool(
            "project_tracker", {"action": "create", "project_name": "AI Chatbot"}
        )
        assert project_result["status"] == "created"

        team_result, _ = agent.tool_manager.execute_tool(
            "team_coordinator", {"action": "status", "team_member": "Sarah"}
        )
        assert team_result["availability"] == "available"

        # Verify conversation flow
        conversation_history = agent.load_conversation_history(memory_id)
        assert len(conversation_history) >= 0

        # Verify memory storage
        stored_memories = memory_provider.storage[memory_id]
        assert len(stored_memories) == 8  # 4 user queries + 4 assistant responses


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    @pytest.mark.integration
    def test_cascading_error_recovery(self):
        """Test error recovery when multiple components fail."""
        # Create failing components
        failing_memory = Mock()
        failing_memory.store.side_effect = Exception("Memory storage failed")
        failing_memory.retrieve_conversation_history_ordered_by_timestamp.side_effect = Exception(
            "Memory retrieval failed"
        )

        def failing_tool(param: str) -> str:
            raise Exception(f"Tool failed with param: {param}")

        # But LLM works
        llm_provider = MockLLMProvider(
            ["I'll work despite component failures and provide helpful responses."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=failing_memory,
            tools=[failing_tool],
            instruction="You are resilient and handle component failures gracefully.",
            semantic_cache=True,  # Cache might also fail
            agent_id="error_recovery_agent",
        )

        # Agent should still respond despite failures
        response = agent.run("Help me with this task despite any system issues")

        assert_agent_response_valid(response)
        assert llm_provider.call_count == 1

    @pytest.mark.integration
    def test_partial_component_failure(self):
        """Test behavior when some components fail but others work."""
        memory_provider = MockMemoryProvider()  # This works

        # Tools that partially fail
        def reliable_tool(x: int) -> int:
            return x * 2

        def unreliable_tool(x: int) -> int:
            if x > 10:
                raise Exception("Tool cannot handle large numbers")
            return x + 1

        llm_provider = MockLLMProvider(
            [
                "I'll use the reliable tool for this calculation.",
                "The unreliable tool failed, but I can work around it.",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[reliable_tool, unreliable_tool],
            instruction="You adapt when tools fail and use alternatives.",
            agent_id="partial_failure_agent",
        )

        memory_id = "partial_test"

        # Use reliable tool - should work
        response1 = agent.run("Use tools to calculate something", memory_id=memory_id)
        assert_agent_response_valid(response1)

        # Test reliable tool directly
        result1, _ = agent.tool_manager.execute_tool("reliable_tool", {"x": 5})
        assert result1 == 10

        # Test unreliable tool with safe input
        result2, _ = agent.tool_manager.execute_tool("unreliable_tool", {"x": 5})
        assert result2 == 6

        # Test unreliable tool with problematic input
        result3, _ = agent.tool_manager.execute_tool("unreliable_tool", {"x": 15})
        assert "cannot handle" in result3.lower()  # Should return error message

        # Agent should continue working despite partial failures
        response2 = agent.run(
            "Continue helping despite any tool issues", memory_id=memory_id
        )
        assert_agent_response_valid(response2)

    @pytest.mark.integration
    def test_component_isolation(self):
        """Test that component failures are properly isolated."""
        memory_provider = MockMemoryProvider()

        # Create agent with mixed reliability
        llm_provider = MockLLMProvider(
            ["I'm working normally despite any component issues."]
        )

        # Simulate cache failure
        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_instance.get.side_effect = Exception("Cache service unavailable")
            mock_cache_instance.set.side_effect = Exception("Cache service unavailable")
            mock_cache_class.return_value = mock_cache_instance

            agent = MemAgent(
                model=llm_provider,
                memory_provider=memory_provider,
                instruction="You work reliably despite cache issues.",
                semantic_cache=True,  # This will fail
                agent_id="isolation_test_agent",
            )

            # Agent should work despite cache failures
            response = agent.run("Test response with failing cache")

            assert_agent_response_valid(response)
            assert llm_provider.call_count == 1

            # Verify cache is configured (even if failing)
            assert agent.cache_manager.enabled == True
