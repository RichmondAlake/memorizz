"""Test save/load functionality for MemAgent."""
import json
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from memorizz.memagent import MemAgent, MemAgentModel
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider, MockPersona


class TestMemAgentSaveFunctionality:
    """Test MemAgent save functionality."""

    @pytest.mark.save_load
    def test_save_basic_agent(self):
        """Test saving a basic MemAgent configuration."""
        memory_provider = MockMemoryProvider()

        # Mock the memory provider methods for saving
        memory_provider.store_memagent = Mock(return_value={"_id": "test_agent_123"})
        memory_provider.retrieve_memagent = Mock(
            return_value=None
        )  # Agent doesn't exist yet

        llm_provider = MockLLMProvider(["Test response"])
        llm_provider.get_config = Mock(
            return_value={"provider": "test", "model": "test-model"}
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="Test save functionality",
            max_steps=25,
            agent_id="test_agent_123",
        )

        # Test saving the agent
        result = agent.save()

        # Verify save was successful
        assert result == agent  # Should return self for chaining
        assert memory_provider.store_memagent.called

        # Check the saved model structure
        call_args = memory_provider.store_memagent.call_args[0][0]
        assert isinstance(call_args, MemAgentModel)
        assert call_args.instruction == "Test save functionality"
        assert call_args.max_steps == 25
        assert call_args.agent_id == "test_agent_123"
        assert call_args.llm_config == {"provider": "test", "model": "test-model"}

    @pytest.mark.save_load
    def test_save_agent_with_tools(self):
        """Test saving an agent with tools."""
        memory_provider = MockMemoryProvider()
        memory_provider.store_memagent = Mock(return_value={"_id": "tool_agent_123"})

        def test_tool(x: int, y: int) -> int:
            """A test tool for saving."""
            return x + y

        def another_tool(text: str) -> str:
            """Another test tool."""
            return text.upper()

        llm_provider = MockLLMProvider(["Tool response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[test_tool, another_tool],
            instruction="Agent with tools",
            agent_id="tool_agent_123",
        )

        # Test saving
        result = agent.save()

        assert result == agent
        assert memory_provider.store_memagent.called

        # Verify tools were serialized
        saved_model = memory_provider.store_memagent.call_args[0][0]
        assert saved_model.tools is not None

        # Check tool metadata was preserved
        tool_names = [tool["name"] for tool in saved_model.tools]
        assert "test_tool" in tool_names
        assert "another_tool" in tool_names

    @pytest.mark.save_load
    def test_save_agent_with_persona(self):
        """Test saving an agent with persona."""
        memory_provider = MockMemoryProvider()
        memory_provider.store_memagent = Mock(return_value={"_id": "persona_agent_123"})

        persona = MockPersona(
            name="Test Persona",
            role="Test Assistant",
            traits=["helpful", "knowledgeable"],
            expertise=["testing", "debugging"],
        )

        llm_provider = MockLLMProvider(["Persona response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            persona=persona,
            instruction="Agent with persona",
            agent_id="persona_agent_123",
        )

        # Test saving
        result = agent.save()

        assert result == agent
        assert memory_provider.store_memagent.called

        # Verify persona was saved
        saved_model = memory_provider.store_memagent.call_args[0][0]
        assert saved_model.persona == persona

    @pytest.mark.save_load
    def test_save_agent_with_semantic_cache(self):
        """Test saving an agent with semantic cache configuration."""
        memory_provider = MockMemoryProvider()
        memory_provider.store_memagent = Mock(return_value={"_id": "cache_agent_123"})

        llm_provider = MockLLMProvider(["Cache response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        with patch(
            "memorizz.memagent.managers.cache_manager.SemanticCache"
        ) as mock_cache_class:
            from types import SimpleNamespace

            mock_config = SimpleNamespace(
                similarity_threshold=0.8, scope=SimpleNamespace(value="local")
            )

            mock_cache_instance = Mock()
            mock_cache_instance.config = mock_config
            mock_cache_class.return_value = mock_cache_instance

            agent = MemAgent(
                model=llm_provider,
                memory_provider=memory_provider,
                instruction="Agent with cache",
                semantic_cache=True,
                semantic_cache_config={"similarity_threshold": 0.8},
                agent_id="cache_agent_123",
            )

            # Test saving
            result = agent.save()

            assert result == agent
            assert memory_provider.store_memagent.called

            # Verify semantic cache config was saved
            saved_model = memory_provider.store_memagent.call_args[0][0]
            assert saved_model.semantic_cache == True
            assert saved_model.semantic_cache_config is not None

    @pytest.mark.save_load
    def test_save_agent_with_skills_and_mcp_servers(self, tmp_path):
        """Test saving skill paths and MCP server configs."""
        memory_provider = MockMemoryProvider()
        memory_provider.store_memagent = Mock(return_value={"_id": "skill_mcp_agent"})
        memory_provider.retrieve_memagent = Mock(return_value=None)
        memory_provider.store = Mock(return_value="toolbox_mcp_doc")

        skill_file = tmp_path / "assistant.skills.md"
        skill_file.write_text(
            "# Assistant Skill\n\nUse this skill.\n", encoding="utf-8"
        )

        llm_provider = MockLLMProvider(["Skill/MCP response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="Agent with skills and MCPs",
            agent_id="skill_mcp_agent",
            memory_ids=["mcp_memory"],
            skills_marketplace_provider="skillsmp",
            skills_marketplace_config={"api_key": "sk_test_skillsmp"},
            skill_paths=[str(skill_file)],
            mcp_servers=[
                {
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                }
            ],
        )

        agent.save()

        saved_model = memory_provider.store_memagent.call_args[0][0]
        assert saved_model.skills_marketplace_provider == "skillsmp"
        assert saved_model.skills_marketplace_config is not None
        assert (
            saved_model.skills_marketplace_config.get("api_key") == "sk_test_skillsmp"
        )
        assert saved_model.skill_paths == [str(skill_file)]
        assert saved_model.mcp_servers is not None
        assert saved_model.mcp_servers[0]["name"] == "filesystem"
        assert memory_provider.store.called

    @pytest.mark.save_load
    def test_save_includes_self_aware_configuration(self):
        """Save should persist self-aware flags and normalized config."""
        memory_provider = MockMemoryProvider()
        memory_provider.store_memagent = Mock(return_value={"_id": "self_aware_agent"})
        memory_provider.retrieve_memagent = Mock(return_value=None)

        llm_provider = MockLLMProvider(["Self-aware response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="Self-aware save",
            agent_id="self_aware_agent",
            self_aware=True,
            self_aware_config={
                "root_paths": ["."],
                "allow_writes": True,
                "allow_deletes": False,
            },
        )

        agent.save()
        saved_model = memory_provider.store_memagent.call_args[0][0]
        assert saved_model.self_aware is True
        assert isinstance(saved_model.self_aware_config, dict)
        assert saved_model.self_aware_config.get("allow_writes") is True

    @pytest.mark.save_load
    def test_save_without_memory_provider(self):
        """Test saving fails gracefully without memory provider."""
        agent = MemAgent(
            model=MockLLMProvider(["No provider"]),
            instruction="No memory provider",
            memory_provider=False,
        )

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Cannot save MemAgent: no memory provider configured"
        ):
            agent.save()

    @pytest.mark.save_load
    def test_save_updates_existing_agent(self):
        """Test that saving updates an existing agent."""
        memory_provider = MockMemoryProvider()

        # Mock existing agent
        existing_agent = Mock()
        memory_provider.retrieve_memagent = Mock(return_value=existing_agent)
        memory_provider.update_memagent = Mock(
            return_value={"_id": "existing_agent_123"}
        )
        memory_provider.store_memagent = Mock()

        llm_provider = MockLLMProvider(["Update response"])
        llm_provider.get_config = Mock(return_value={"provider": "test"})

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="Update test",
            agent_id="existing_agent_123",
        )

        # Test saving (should update, not create new)
        result = agent.save()

        assert result == agent
        assert memory_provider.retrieve_memagent.called
        assert memory_provider.update_memagent.called
        assert (
            not memory_provider.store_memagent.called
        )  # Should not call store for update


class TestMemAgentLoadFunctionality:
    """Test MemAgent load functionality."""

    @pytest.mark.save_load
    def test_load_basic_agent(self):
        """Test loading a basic MemAgent."""
        memory_provider = MockMemoryProvider()

        # Mock saved agent data
        saved_agent_data = Mock()
        saved_agent_data.llm_config = {"provider": "test", "model": "test-model"}
        saved_agent_data.instruction = "Loaded instruction"
        saved_agent_data.max_steps = 30
        saved_agent_data.memory_ids = ["loaded_memory"]
        saved_agent_data.persona = None
        saved_agent_data.tools = None
        saved_agent_data.semantic_cache = False
        saved_agent_data.semantic_cache_config = None
        saved_agent_data.delegates = None
        saved_agent_data.skills_marketplace_provider = "skillsmp"
        saved_agent_data.skills_marketplace_config = {"api_key": "sk_test_skillsmp"}
        saved_agent_data.skill_paths = ["skills/support.skills.md"]
        saved_agent_data.mcp_servers = [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            }
        ]

        memory_provider.retrieve_memagent = Mock(return_value=saved_agent_data)

        with patch("memorizz.memagent.core.create_llm_provider") as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm

            # Test loading
            loaded_agent = MemAgent.load("test_agent_123", memory_provider)

            # Verify agent was loaded correctly
            assert loaded_agent.agent_id == "test_agent_123"
            assert loaded_agent.instruction == "Loaded instruction"
            assert loaded_agent.max_steps == 30
            assert loaded_agent.memory_ids == ["loaded_memory"]
            assert loaded_agent.model == mock_llm
            assert loaded_agent.llm_config == {
                "provider": "test",
                "model": "test-model",
            }
            assert loaded_agent.llm_provider == "test"
            assert loaded_agent.llm_model == "test-model"
            assert loaded_agent.memory_provider == memory_provider
            assert loaded_agent.get_skills_marketplace_provider_name() == "skillsmp"
            skills_cfg = loaded_agent.get_skills_marketplace_config() or {}
            assert skills_cfg.get("api_key") == "sk_test_skillsmp"
            assert loaded_agent.skill_paths == ["skills/support.skills.md"]
            assert loaded_agent.mcp_servers[0]["name"] == "filesystem"

            mock_create_llm.assert_called_once_with(
                {"provider": "test", "model": "test-model"}
            )

    @pytest.mark.save_load
    def test_load_agent_with_overrides(self):
        """Test loading agent with parameter overrides."""
        memory_provider = MockMemoryProvider()

        saved_agent_data = Mock()
        saved_agent_data.llm_config = {"provider": "test"}
        saved_agent_data.instruction = "Original instruction"
        saved_agent_data.max_steps = 20
        saved_agent_data.memory_ids = ["original_memory"]
        saved_agent_data.persona = None
        saved_agent_data.tools = None
        saved_agent_data.semantic_cache = False
        saved_agent_data.semantic_cache_config = None
        saved_agent_data.delegates = None

        memory_provider.retrieve_memagent = Mock(return_value=saved_agent_data)

        override_llm = MockLLMProvider(["Override response"])

        with patch("memorizz.memagent.core.create_llm_provider") as create_provider:
            # Test loading with overrides
            loaded_agent = MemAgent.load(
                "test_agent_123",
                memory_provider,
                model=override_llm,
                instruction="Override instruction",
                max_steps=50,
            )

            # Verify overrides were applied
            assert loaded_agent.model == override_llm
            assert loaded_agent.llm_config == {
                "provider": "mock",
                "model": "mock-model",
            }
            assert loaded_agent.instruction == "Override instruction"
            assert loaded_agent.max_steps == 50
            create_provider.assert_not_called()

    @pytest.mark.save_load
    def test_load_legacy_agent_defaults_self_awareness_safely(self):
        """Legacy payloads without self-aware fields should load with safe defaults."""
        from types import SimpleNamespace

        memory_provider = MockMemoryProvider()

        saved_agent_data = SimpleNamespace(
            llm_config={"provider": "test"},
            instruction="Legacy instruction",
            max_steps=20,
            memory_ids=["legacy_memory"],
            persona=None,
            tools=None,
            semantic_cache=False,
            semantic_cache_config=None,
            delegates=None,
        )

        memory_provider.retrieve_memagent = Mock(return_value=saved_agent_data)

        with patch("memorizz.memagent.core.create_llm_provider") as mock_create_llm:
            mock_create_llm.return_value = Mock()
            loaded_agent = MemAgent.load("legacy_agent", memory_provider)

        assert loaded_agent.has_self_awareness() is False
        cfg = loaded_agent.get_self_aware_config()
        assert cfg.get("allow_writes") is False
        assert cfg.get("allow_deletes") is False

    @pytest.mark.save_load
    def test_load_self_awareness_overrides_are_applied(self):
        """Load overrides should replace stored self-aware settings."""
        memory_provider = MockMemoryProvider()

        saved_agent_data = Mock()
        saved_agent_data.llm_config = {"provider": "test"}
        saved_agent_data.instruction = "Original instruction"
        saved_agent_data.max_steps = 20
        saved_agent_data.memory_ids = ["original_memory"]
        saved_agent_data.persona = None
        saved_agent_data.tools = None
        saved_agent_data.semantic_cache = False
        saved_agent_data.semantic_cache_config = None
        saved_agent_data.delegates = None
        saved_agent_data.self_aware = False
        saved_agent_data.self_aware_config = {"root_paths": ["."]}

        memory_provider.retrieve_memagent = Mock(return_value=saved_agent_data)

        with patch("memorizz.memagent.core.create_llm_provider") as mock_create_llm:
            mock_create_llm.return_value = Mock()
            loaded_agent = MemAgent.load(
                "override_agent",
                memory_provider,
                self_aware=True,
                self_aware_config={
                    "root_paths": ["."],
                    "allow_writes": True,
                    "allow_deletes": False,
                },
            )

        assert loaded_agent.has_self_awareness() is True
        cfg = loaded_agent.get_self_aware_config()
        assert cfg.get("allow_writes") is True
        assert cfg.get("allow_deletes") is False

    @pytest.mark.save_load
    def test_load_nonexistent_agent(self):
        """Test loading a non-existent agent raises error."""
        memory_provider = MockMemoryProvider()
        memory_provider.retrieve_memagent = Mock(return_value=None)

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="MemAgent with agent id nonexistent not found"
        ):
            MemAgent.load("nonexistent", memory_provider)

    @pytest.mark.save_load
    def test_load_with_delegates(self):
        """Test loading agent with delegate agents."""
        memory_provider = MockMemoryProvider()

        # Mock main agent data
        saved_agent_data = Mock()
        saved_agent_data.llm_config = {"provider": "test"}
        saved_agent_data.instruction = "Main agent"
        saved_agent_data.max_steps = 20
        saved_agent_data.memory_ids = ["main_memory"]
        saved_agent_data.persona = None
        saved_agent_data.tools = None
        saved_agent_data.semantic_cache = False
        saved_agent_data.semantic_cache_config = None
        saved_agent_data.delegates = ["delegate_1", "delegate_2"]

        # Mock delegate data
        delegate_1_data = Mock()
        delegate_1_data.llm_config = {"provider": "test"}
        delegate_1_data.instruction = "Delegate 1"
        delegate_1_data.max_steps = 10
        delegate_1_data.memory_ids = []
        delegate_1_data.persona = None
        delegate_1_data.tools = None
        delegate_1_data.semantic_cache = False
        delegate_1_data.semantic_cache_config = None
        delegate_1_data.delegates = None

        delegate_2_data = Mock()
        delegate_2_data.llm_config = {"provider": "test"}
        delegate_2_data.instruction = "Delegate 2"
        delegate_2_data.max_steps = 15
        delegate_2_data.memory_ids = []
        delegate_2_data.persona = None
        delegate_2_data.tools = None
        delegate_2_data.semantic_cache = False
        delegate_2_data.semantic_cache_config = None
        delegate_2_data.delegates = None

        # Setup mock to return different data based on agent_id
        def mock_retrieve(agent_id):
            if agent_id == "main_agent":
                return saved_agent_data
            elif agent_id == "delegate_1":
                return delegate_1_data
            elif agent_id == "delegate_2":
                return delegate_2_data
            return None

        memory_provider.retrieve_memagent = Mock(side_effect=mock_retrieve)

        with patch("memorizz.memagent.core.create_llm_provider") as mock_create_llm:
            mock_create_llm.return_value = Mock()

            # Test loading main agent with delegates
            loaded_agent = MemAgent.load("main_agent", memory_provider)

            assert loaded_agent.agent_id == "main_agent"
            assert memory_provider.retrieve_memagent.call_count == 3
            retrieved_ids = {
                call.args[0]
                for call in memory_provider.retrieve_memagent.call_args_list
            }
            assert retrieved_ids == {"main_agent", "delegate_1", "delegate_2"}

    @pytest.mark.save_load
    def test_load_without_memory_provider(self):
        """Test loading without specifying memory provider."""
        # Should try to create default memory provider
        with patch("memorizz.memory_provider.MemoryProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.retrieve_memagent = Mock(return_value=None)
            mock_provider_class.return_value = mock_provider

            with pytest.raises(ValueError, match="not found in the memory provider"):
                MemAgent.load("test_agent")


class TestMemAgentRefreshFunctionality:
    """Test MemAgent refresh functionality."""

    @pytest.mark.save_load
    def test_refresh_agent_configuration(self):
        """Test refreshing agent configuration from memory provider."""
        memory_provider = MockMemoryProvider()

        # Create agent
        agent = MemAgent(
            model=MockLLMProvider(["Original"]),
            memory_provider=memory_provider,
            instruction="Original instruction",
            max_steps=20,
            agent_id="refresh_test_agent",
        )

        # Mock updated configuration in memory provider
        updated_config = Mock()
        updated_config.instruction = "Updated instruction"
        updated_config.max_steps = 30
        updated_config.memory_ids = ["updated_memory"]
        updated_config.persona = MockPersona(
            name="Updated Persona",
            role="Updated Role",
            traits=["updated"],
            expertise=["refresh"],
        )

        memory_provider.retrieve_memagent = Mock(return_value=updated_config)

        # Test refresh
        result = agent.refresh()

        # Verify refresh was successful
        assert result == agent
        assert agent.instruction == "Updated instruction"
        assert agent.max_steps == 30
        assert agent.memory_ids == ["updated_memory"]

        # Verify memory provider was called
        memory_provider.retrieve_memagent.assert_called_once_with("refresh_test_agent")

    @pytest.mark.save_load
    def test_refresh_without_memory_provider(self):
        """Test refresh fails gracefully without memory provider."""
        agent = MemAgent(
            model=MockLLMProvider(["No provider"]),
            instruction="No memory provider",
            memory_provider=False,
        )

        result = agent.refresh()
        assert result == False

    @pytest.mark.save_load
    def test_refresh_without_agent_id(self):
        """Test refresh fails without agent_id."""
        agent = MemAgent(
            model=MockLLMProvider(["No ID"]),
            memory_provider=MockMemoryProvider(),
            instruction="No agent ID",
            agent_id=None,
        )

        result = agent.refresh()
        assert result == False

    @pytest.mark.save_load
    def test_refresh_nonexistent_agent(self):
        """Test refresh handles non-existent agent gracefully."""
        memory_provider = MockMemoryProvider()
        memory_provider.retrieve_memagent = Mock(return_value=None)

        agent = MemAgent(
            model=MockLLMProvider(["Nonexistent"]),
            memory_provider=memory_provider,
            instruction="Nonexistent agent",
            agent_id="nonexistent_agent",
        )

        result = agent.refresh()
        assert result == False


class TestSaveLoadIntegration:
    """Integration tests for save/load functionality."""

    @pytest.mark.save_load
    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip."""
        memory_provider = MockMemoryProvider()

        # Create original agent
        def test_tool(x: int) -> int:
            return x * 2

        persona = MockPersona(
            name="Test Persona",
            role="Tester",
            traits=["thorough"],
            expertise=["testing"],
        )

        llm_provider = MockLLMProvider(["Original response"])
        llm_provider.get_config = Mock(
            return_value={"provider": "test", "model": "test-model", "temperature": 0.7}
        )

        original_agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            tools=[test_tool],
            persona=persona,
            instruction="Original test agent",
            max_steps=25,
            memory_ids=["test_memory"],
            agent_id="roundtrip_agent",
            semantic_cache=True,
        )

        # Mock save functionality
        saved_data = None

        def mock_store(agent_model):
            nonlocal saved_data
            saved_data = agent_model
            return {"_id": "roundtrip_agent"}

        memory_provider.store_memagent = Mock(side_effect=mock_store)
        memory_provider.retrieve_memagent = Mock(return_value=None)  # New agent

        # Save the agent
        original_agent.save()

        # Verify data was saved
        assert saved_data is not None
        assert saved_data.instruction == "Original test agent"
        assert saved_data.max_steps == 25

        # Mock load functionality
        memory_provider.retrieve_memagent = Mock(return_value=saved_data)

        with patch("memorizz.memagent.core.create_llm_provider") as mock_create_llm:
            mock_create_llm.return_value = llm_provider

            # Load the agent
            loaded_agent = MemAgent.load("roundtrip_agent", memory_provider)

            # Verify loaded agent matches original
            assert loaded_agent.agent_id == "roundtrip_agent"
            assert loaded_agent.instruction == "Original test agent"
            assert loaded_agent.max_steps == 25
            assert loaded_agent.memory_ids == ["test_memory"]
            assert loaded_agent.memory_provider == memory_provider
