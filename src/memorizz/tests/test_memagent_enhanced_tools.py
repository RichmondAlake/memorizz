# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""
Test the enhanced tool functionality for MemAgent.
This test demonstrates the new ability to add decorated functions directly.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

from memorizz.coordination.shared_memory.shared_memory import SharedMemory
from memorizz.llms.openai import OpenAI
from memorizz.memagent import MemAgent
from memorizz.memory_provider import MemoryProvider
from memorizz.memory_provider.mongodb.provider import MongoDBProvider
from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestMemAgentEnhancedTools(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_memory_provider = MagicMock(spec=MemoryProvider)
        self.mock_model = MagicMock(spec=OpenAI)

        # Create agent with mocked dependencies
        self.agent = MemAgent(
            model=self.mock_model, memory_provider=self.mock_memory_provider
        )

    def test_add_decorated_function_ephemeral(self):
        """Test adding a decorated function without persistence."""

        def sample_tool(message: str, count: int = 1) -> str:
            """
            Repeat a message a specified number of times.

            Args:
                message: The message to repeat
                count: Number of times to repeat (default: 1)

            Returns:
                The repeated message
            """
            return message * count

        # Add the function without persistence
        result = self.agent.add_tool(sample_tool, persist=False)

        # Verify the function was added successfully
        self.assertTrue(result)
        self.assertIn("sample_tool", self.agent.tool_manager.tools)

        # Verify the tool metadata is correctly generated
        tool = self.agent.tool_manager.get_tool_metadata("sample_tool")
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "sample_tool")
        self.assertIn("Repeat a message", tool["description"])

        # Verify parameters are correctly extracted
        params = tool["parameters"]
        self.assertEqual(len(params), 2)

        # Check parameter types
        self.assertEqual(params["message"]["type"], "string")
        self.assertEqual(params["count"]["type"], "integer")

        # Verify required list includes all function params
        required_params = tool["required"]
        self.assertIn("message", required_params)
        # Parameters with defaults are optional in tool schemas.
        self.assertNotIn("count", required_params)

    def test_add_decorated_function_persistent(self):
        """Test adding a decorated function with persistence."""

        def persistent_tool(data: str) -> str:
            """Process some data."""
            return f"Processed: {data}"

        # Mock the memory provider store method
        self.mock_memory_provider.store.return_value = "test-tool-id-123"

        # Add the function with persistence
        result = self.agent.add_tool(persistent_tool, persist=True)

        # Verify the function was added successfully
        self.assertTrue(result)
        self.assertIn("persistent_tool", self.agent.tool_manager.tools)

        # Verify memory provider was called to store the tool
        self.mock_memory_provider.store.assert_called_once()

        # Verify the stored tool has the correct _id from memory provider
        tool = self.agent.tool_manager.get_tool_metadata("persistent_tool")
        self.assertEqual(tool.get("name"), "persistent_tool")

    def test_add_multiple_functions(self):
        """Test adding multiple functions at once."""

        def tool_one(x: int) -> int:
            """First tool."""
            return x * 2

        def tool_two(y: str) -> str:
            """Second tool."""
            return y.upper()

        def tool_three(z: float) -> float:
            """Third tool."""
            return z / 2

        functions = [tool_one, tool_two, tool_three]

        initial_count = len(self.agent.tool_manager.tools)

        # Add all functions at once
        results = [self.agent.add_tool(func, persist=False) for func in functions]

        # Verify all functions were added successfully
        self.assertTrue(all(results))
        self.assertEqual(
            len(self.agent.tool_manager.tools), initial_count + len(functions)
        )

        # Verify each tool was added correctly
        tool_names = self.agent.tool_manager.list_tools()
        self.assertIn("tool_one", tool_names)
        self.assertIn("tool_two", tool_names)
        self.assertIn("tool_three", tool_names)

    def test_update_existing_function(self):
        """Test updating an existing function with the same name."""

        def original_tool(message: str) -> str:
            """Original version."""
            return message

        def updated_tool(message: str) -> str:
            """Updated version with better functionality."""
            return f"Enhanced: {message}"

        initial_count = len(self.agent.tool_manager.tools)

        # Add original function
        self.agent.add_tool(original_tool, persist=False)
        self.assertEqual(len(self.agent.tool_manager.tools), initial_count + 1)
        original_description = self.agent.tool_manager.get_tool_metadata(
            "original_tool"
        )["description"]

        # Add updated function with same name
        updated_tool.__name__ = "original_tool"
        self.agent.add_tool(updated_tool, persist=False)

        # Verify only one tool exists (updated, not duplicated)
        self.assertEqual(len(self.agent.tool_manager.tools), initial_count + 1)
        updated_description = self.agent.tool_manager.get_tool_metadata(
            "original_tool"
        )["description"]

        # Verify the description was updated
        self.assertNotEqual(original_description, updated_description)
        self.assertIn("better functionality", updated_description)

    def test_error_handling_invalid_function(self):
        """Test error handling for invalid function input."""

        # Missing tool IDs should fail to load
        self.mock_memory_provider.retrieve_by_id.return_value = None
        result = self.agent.add_tool("missing_tool")
        self.assertFalse(result)

        # Unsupported types should return False
        result = self.agent.add_tool(None)
        self.assertFalse(result)

    def test_backward_compatibility(self):
        """Test that existing functionality still works."""

        # Mock toolbox and existing methods should still work
        from memorizz.long_term_memory.procedural.toolbox import Toolbox

        mock_toolbox = MagicMock(spec=Toolbox)
        mock_toolbox.list_tools.return_value = []

        # This should not raise an error
        mock_toolbox.tools = {"example": {"metadata": {"name": "example"}}}

        tools_loaded = self.agent.tool_manager.initialize_from_toolbox(mock_toolbox)

        self.assertEqual(tools_loaded, 1)
        self.assertIn("example", self.agent.tool_manager.list_tools())


class TestMultiAgentMemoryManagement(unittest.TestCase):
    """Test multi-agent memory management fixes."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock memory provider
        self.mock_memory_provider = Mock(spec=MongoDBProvider)
        self.mock_memory_provider.update_memagent_memory_ids = Mock(return_value=True)

        # Create root agent
        self.root_agent = MemAgent(
            agent_id="root_agent_001",
            memory_provider=self.mock_memory_provider,
            memory_ids=["existing_memory_001"],
        )

        # Create delegate agents
        self.delegate1 = MemAgent(
            agent_id="delegate_001",
            memory_provider=self.mock_memory_provider,
            memory_ids=["delegate_memory_001"],
        )

        self.delegate2 = MemAgent(
            agent_id="delegate_002",
            memory_provider=self.mock_memory_provider,
            memory_ids=[],
        )

        self.delegates = [self.delegate1, self.delegate2]

    @patch("memorizz.multi_agent_orchestrator.SharedMemory")
    @patch("memorizz.multi_agent_orchestrator.TaskDecomposer")
    def test_shared_memory_id_added_to_root_agent(
        self, mock_task_decomposer, mock_shared_memory_class
    ):
        """Test that shared memory ID is added to root agent's memory_ids array."""

        # Mock shared memory instance
        mock_shared_memory = Mock(spec=SharedMemory)
        mock_shared_memory.create_shared_session.return_value = "shared_memory_123"
        mock_shared_memory.find_active_session_for_agent.return_value = None
        mock_shared_memory.add_blackboard_entry.return_value = True
        mock_shared_memory.update_session_status.return_value = True
        mock_shared_memory_class.return_value = mock_shared_memory

        # Mock task decomposer
        mock_task_decomposer_instance = Mock()
        mock_task_decomposer_instance.decompose_task.return_value = (
            []
        )  # No tasks to simulate fallback
        mock_task_decomposer.return_value = mock_task_decomposer_instance

        # Mock root agent run method for fallback
        self.root_agent.run = Mock(return_value="Fallback response")

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(self.root_agent, self.delegates)

        # Verify initial state
        initial_memory_ids = self.root_agent.memory_ids.copy()
        self.assertNotIn("shared_memory_123", initial_memory_ids)

        # Execute multi-agent workflow
        result = orchestrator.execute_multi_agent_workflow("Test query")

        # Verify shared memory ID was added to root agent
        self.assertIn("shared_memory_123", self.root_agent.memory_ids)

        # Verify memory provider was called to persist the update
        self.mock_memory_provider.update_memagent_memory_ids.assert_called()

        # Verify the call was made with the correct parameters
        called_args = (
            self.mock_memory_provider.update_memagent_memory_ids.call_args_list
        )
        root_agent_call = next(
            (call for call in called_args if call[0][0] == "root_agent_001"), None
        )
        self.assertIsNotNone(root_agent_call)
        self.assertIn("shared_memory_123", root_agent_call[0][1])

    @patch("memorizz.multi_agent_orchestrator.SharedMemory")
    @patch("memorizz.multi_agent_orchestrator.TaskDecomposer")
    def test_shared_memory_id_added_to_delegate_agents(
        self, mock_task_decomposer, mock_shared_memory_class
    ):
        """Test that shared memory ID is added to delegate agents' memory_ids arrays."""

        # Mock shared memory instance
        mock_shared_memory = Mock(spec=SharedMemory)
        mock_shared_memory.create_shared_session.return_value = "shared_memory_456"
        mock_shared_memory.find_active_session_for_agent.return_value = None
        mock_shared_memory.add_blackboard_entry.return_value = True
        mock_shared_memory.update_session_status.return_value = True
        mock_shared_memory_class.return_value = mock_shared_memory

        # Mock task decomposer
        mock_task_decomposer_instance = Mock()
        mock_task_decomposer_instance.decompose_task.return_value = (
            []
        )  # No tasks to simulate fallback
        mock_task_decomposer.return_value = mock_task_decomposer_instance

        # Mock root agent run method for fallback
        self.root_agent.run = Mock(return_value="Fallback response")

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(self.root_agent, self.delegates)

        # Verify initial state
        self.assertNotIn("shared_memory_456", self.delegate1.memory_ids)
        self.assertNotIn("shared_memory_456", self.delegate2.memory_ids)

        # Execute multi-agent workflow
        result = orchestrator.execute_multi_agent_workflow("Test query")

        # Verify shared memory ID was added to both delegates
        self.assertIn("shared_memory_456", self.delegate1.memory_ids)
        self.assertIn("shared_memory_456", self.delegate2.memory_ids)

        # Verify memory provider was called for both delegates
        called_args = (
            self.mock_memory_provider.update_memagent_memory_ids.call_args_list
        )

        delegate1_call = next(
            (call for call in called_args if call[0][0] == "delegate_001"), None
        )
        self.assertIsNotNone(delegate1_call)
        self.assertIn("shared_memory_456", delegate1_call[0][1])

        delegate2_call = next(
            (call for call in called_args if call[0][0] == "delegate_002"), None
        )
        self.assertIsNotNone(delegate2_call)
        self.assertIn("shared_memory_456", delegate2_call[0][1])

    @patch("memorizz.multi_agent_orchestrator.SharedMemory")
    @patch("memorizz.multi_agent_orchestrator.TaskDecomposer")
    def test_no_duplicate_shared_memory_ids(
        self, mock_task_decomposer, mock_shared_memory_class
    ):
        """Test that shared memory ID is not duplicated if already present."""

        # Pre-add shared memory ID to root agent
        shared_memory_id = "shared_memory_789"
        self.root_agent.memory_ids.append(shared_memory_id)
        initial_count = self.root_agent.memory_ids.count(shared_memory_id)

        # Mock shared memory instance
        mock_shared_memory = Mock(spec=SharedMemory)
        mock_shared_memory.create_shared_session.return_value = shared_memory_id
        mock_shared_memory.find_active_session_for_agent.return_value = None
        mock_shared_memory.add_blackboard_entry.return_value = True
        mock_shared_memory.update_session_status.return_value = True
        mock_shared_memory_class.return_value = mock_shared_memory

        # Mock task decomposer
        mock_task_decomposer_instance = Mock()
        mock_task_decomposer_instance.decompose_task.return_value = []
        mock_task_decomposer.return_value = mock_task_decomposer_instance

        # Mock root agent run method for fallback
        self.root_agent.run = Mock(return_value="Fallback response")

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(self.root_agent, self.delegates)

        # Execute multi-agent workflow
        result = orchestrator.execute_multi_agent_workflow("Test query")

        # Verify no duplication occurred
        final_count = self.root_agent.memory_ids.count(shared_memory_id)
        self.assertEqual(initial_count, final_count)


if __name__ == "__main__":
    unittest.main()
