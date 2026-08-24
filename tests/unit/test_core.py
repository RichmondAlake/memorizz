"""Unit tests for MemAgent core functionality."""
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS
from memorizz.memagent.core import MemAgent
from memorizz.memagent.models import MemAgentConfig, MemAgentModel
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid


class TestMemAgentCore:
    """Test the core MemAgent class."""

    @pytest.mark.unit
    def test_memagent_initialization_minimal(self):
        """Test MemAgent initialization with minimal parameters."""
        agent = MemAgent(instruction="Test instruction")

        assert_agent_state_valid(agent)
        assert agent.instruction == "Test instruction"
        assert agent.max_steps == DEFAULT_MAX_STEPS
        assert MemoryType.SUMMARIES in agent.active_memory_types
        assert agent.agent_id is not None
        assert len(agent.agent_id) > 0

    @pytest.mark.unit
    def test_default_context_tools_include_autosummarize(self):
        """Built-in context toolset should expose on-demand autosummarization."""
        agent = MemAgent(instruction="Tool availability test")
        assert "summarize_conversation" in agent.tool_manager.tools

        result, _ = agent.tool_manager.execute_tool("summarize_conversation", {})
        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert "llm model" in result.get("error", "").lower()

    @pytest.mark.unit
    def test_memagent_initialization_full(
        self, mock_llm_provider, mock_memory_provider, sample_tools, sample_persona
    ):
        """Test MemAgent initialization with all parameters."""
        agent = MemAgent(
            model=mock_llm_provider,
            llm_config={"provider": "test", "model": "test-model"},
            tools=sample_tools,
            persona=sample_persona,
            instruction="Full test instruction",
            application_mode="agent",
            max_steps=25,
            memory_provider=mock_memory_provider,
            memory_ids=["test_memory_1", "test_memory_2"],
            agent_id="test_agent_123",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.9},
        )

        assert_agent_state_valid(agent)
        assert agent.instruction == "Full test instruction"
        assert agent.max_steps == 25
        assert agent.agent_id == "test_agent_123"
        assert agent.memory_ids == ["test_memory_1", "test_memory_2"]
        assert agent.model == mock_llm_provider
        assert agent.memory_provider == mock_memory_provider

    @pytest.mark.unit
    def test_tool_iteration_limit_uses_agent_max_steps(self):
        """Tool-calling loop limit should follow configured max_steps."""
        agent = MemAgent(instruction="Tool iteration test", max_steps=20)
        assert agent._get_tool_iteration_limit() == 20

        agent.max_steps = 7
        assert agent._get_tool_iteration_limit() == 7

    @pytest.mark.unit
    def test_tool_iteration_limit_has_safe_bounds(self):
        """Tool-calling loop limit should enforce sane min/max defaults."""
        agent = MemAgent(instruction="Tool iteration bounds test")

        agent.max_steps = -1
        assert agent._get_tool_iteration_limit() == 1

        agent.max_steps = -9
        assert agent._get_tool_iteration_limit() == 1

        agent.max_steps = "not-a-number"
        assert agent._get_tool_iteration_limit() == DEFAULT_MAX_STEPS

        agent.max_steps = 5000
        assert agent._get_tool_iteration_limit() == 1000

    @pytest.mark.unit
    def test_memagent_managers_initialization(self, mock_memory_provider):
        """Test that all manager components are properly initialized."""
        agent = MemAgent(
            instruction="Test",
            memory_provider=mock_memory_provider,
            semantic_cache=True,
        )

        # Check that managers exist
        assert hasattr(agent, "memory_manager")
        assert hasattr(agent, "tool_manager")
        assert hasattr(agent, "cache_manager")
        assert hasattr(agent, "persona_manager")
        assert hasattr(agent, "internet_access_manager")

        # Check memory manager
        assert agent.memory_manager is not None
        assert agent.memory_manager.memory_provider == mock_memory_provider

        # Check cache manager
        assert agent.cache_manager is not None
        assert agent.cache_manager.enabled == True

        # Check other managers
        assert agent.tool_manager is not None
        assert agent.persona_manager is not None
        assert agent.internet_access_manager is not None
        assert agent.internet_access_manager.is_enabled() is False

    @pytest.mark.unit
    def test_memagent_defaults_to_filesystem_memory_provider(self):
        """A bare MemAgent should be durable and memory-first by default."""
        from memorizz.memory_provider import FileSystemProvider

        agent = MemAgent(instruction="Test with default memory")

        assert_agent_state_valid(agent)
        assert isinstance(agent.memory_provider, FileSystemProvider)
        assert agent.uses_default_memory_provider is True
        assert agent.memory_manager is not None

    @pytest.mark.unit
    def test_default_filesystem_provider_honors_root_and_is_process_shared(
        self, tmp_path, monkeypatch
    ):
        """Default agents share one provider for the configured durable root."""
        configured_root = tmp_path / "default-memory"
        monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(configured_root))

        first = MemAgent(instruction="First default agent")
        second = MemAgent(instruction="Second default agent")

        assert first.memory_provider is second.memory_provider
        assert first.memory_provider.root_path == configured_root.resolve()

    @pytest.mark.unit
    def test_memagent_allows_explicit_stateless_opt_out(self):
        """memory_provider=False preserves intentional stateless execution."""
        agent = MemAgent(instruction="Test without memory", memory_provider=False)

        assert_agent_state_valid(agent)
        assert agent.memory_provider is None
        assert agent.uses_default_memory_provider is False
        assert agent.memory_manager is None
        assert agent._init_workflow_capture("use a tool", "test-user") is None

    @pytest.mark.unit
    def test_explicit_empty_memory_types_make_agent_stateless(self):
        """An explicit empty memory list must not silently restore defaults."""
        agent = MemAgent(
            instruction="Stateless agent",
            memory_types=[],
        )

        assert agent.active_memory_types == []

    @pytest.mark.unit
    def test_workflow_persistence_is_noop_without_provider(self, caplog):
        """A provider-less tool run must not log a failed workflow write."""
        agent = MemAgent(instruction="Provider-less agent", memory_provider=False)
        workflow = MagicMock()
        workflow.steps = {"Step 1": {"result": "ok"}}

        assert agent._persist_workflow_run(workflow) is False
        workflow.store_workflow.assert_not_called()
        assert "Error storing workflow" not in caplog.text

    @pytest.mark.unit
    def test_memagent_llm_config_loading(self):
        """Test LLM configuration loading."""
        with patch("memorizz.memagent.core.create_llm_provider") as mock_create:
            mock_llm = Mock()
            mock_create.return_value = mock_llm

            agent = MemAgent(
                instruction="Test LLM config",
                llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            )

            mock_create.assert_called_once_with(
                {"provider": "openai", "model": "gpt-3.5-turbo"}
            )
            assert agent.model == mock_llm

    @pytest.mark.unit
    def test_memagent_tool_initialization(self, sample_tools):
        """Test tool initialization."""
        agent = MemAgent(instruction="Test tools", tools=sample_tools)

        assert_agent_state_valid(agent)
        # Verify tools were added to tool manager
        assert agent.tool_manager is not None
        # Note: Detailed tool testing is in test_tool_manager.py

    @pytest.mark.unit
    def test_memagent_persona_initialization(self, sample_persona):
        """Test persona initialization."""
        agent = MemAgent(instruction="Test persona", persona=sample_persona)

        assert_agent_state_valid(agent)
        assert agent.persona_manager is not None
        # The persona should be set in the persona manager
        # Note: Detailed persona testing is in test_persona_manager.py

    @pytest.mark.unit
    def test_with_entity_memory_toggle(self, mock_memory_provider):
        """Ensure with_entity_memory toggles tools and state."""
        agent = MemAgent(
            instruction="Entity toggle", memory_provider=mock_memory_provider
        )

        assert agent._entity_memory_enabled is False
        agent.with_entity_memory(True)
        assert agent._entity_memory_enabled is True
        assert "entity_memory_lookup" in agent.tool_manager.tools

        agent.with_entity_memory(False)
        assert agent._entity_memory_enabled is False
        assert "entity_memory_lookup" not in agent.tool_manager.tools

    @pytest.mark.unit
    def test_assistant_mode_enables_entity_memory(self, mock_memory_provider):
        """Assistant mode should activate entity memory by default."""
        agent = MemAgent(
            instruction="Assistant with entity memory",
            memory_provider=mock_memory_provider,
            application_mode="assistant",
        )

        assert agent._entity_memory_enabled is True
        assert "entity_memory_lookup" in agent.tool_manager.tools

    @pytest.mark.unit
    def test_builder_entity_memory_toggle(self, mock_memory_provider):
        """Builder helper should pass through entity memory preference."""
        from memorizz.memagent.builders import MemAgentBuilder

        builder = (
            MemAgentBuilder()
            .with_instruction("Test builder entity memory")
            .with_memory_provider(mock_memory_provider)
            .with_entity_memory(True)
        )
        agent = builder.build()

        assert agent._entity_memory_enabled is True

    @pytest.mark.unit
    def test_self_awareness_disabled_by_default(self):
        """Self-awareness should remain opt-in by default."""
        agent = MemAgent(instruction="Self-aware default test")

        assert agent.has_self_awareness() is False
        tool_names = set(agent.tool_manager.list_tools())
        assert "self_aware_list_files" not in tool_names

    @pytest.mark.unit
    def test_self_awareness_tools_register_when_enabled(self):
        """Self-aware toolset should register when enabled in config."""
        agent = MemAgent(
            instruction="Self-aware tools test",
            self_aware=True,
            self_aware_config={"root_paths": ["."], "allow_writes": False},
        )

        assert agent.has_self_awareness() is True
        tool_names = set(agent.tool_manager.list_tools())
        assert "self_aware_list_roots" in tool_names
        assert "self_aware_list_files" in tool_names
        assert "self_aware_read_file" in tool_names
        assert "self_aware_search_files" in tool_names
        assert "self_aware_write_file" in tool_names
        assert "self_aware_delete_path" in tool_names
        assert "self_aware_run_command" in tool_names

    @pytest.mark.unit
    def test_self_awareness_toggle_registers_and_unregisters_tools(self):
        """Runtime toggle should add/remove self-aware tools."""
        agent = MemAgent(instruction="Self-aware toggle test")
        assert "self_aware_list_files" not in set(agent.tool_manager.list_tools())

        agent.with_self_aware(True, {"root_paths": ["."]})
        assert "self_aware_list_files" in set(agent.tool_manager.list_tools())

        agent.with_self_aware(False)
        assert "self_aware_list_files" not in set(agent.tool_manager.list_tools())

    @pytest.mark.unit
    def test_system_prompt_includes_self_awareness_section_only_when_enabled(self):
        """Prompt should describe self-aware policy only when enabled."""
        agent = MemAgent(instruction="Prompt section test")
        prompt_without = agent._build_system_prompt()
        assert "Self-awareness host codebase tools" not in prompt_without

        agent.with_self_aware(True, {"root_paths": ["."]})
        prompt_with = agent._build_system_prompt()
        assert "Self-awareness host codebase tools" in prompt_with

    @pytest.mark.unit
    def test_memagent_loads_skill_paths_and_registers_skill_tools(self, tmp_path):
        """Skill markdown files should load and expose skill tools."""
        skill_file = tmp_path / "support.skills.md"
        skill_file.write_text(
            "# Support Skill\n\nHandle support flows.\n\n```python\nprint('ok')\n```\n",
            encoding="utf-8",
        )

        agent = MemAgent(
            instruction="Skill test",
            skill_paths=[str(skill_file)],
        )

        assert len(agent.skills) == 1
        assert agent.skills[0]["name"] == "Support Skill"

        tool_names = set(agent.tool_manager.list_tools())
        assert "list_skills" in tool_names
        assert "read_skill" in tool_names
        assert "run_skill_code" in tool_names
        assert "run_skill_script" in tool_names

    @pytest.mark.unit
    def test_run_skill_code_requires_sandbox(self, tmp_path):
        """Skill code execution should fail when sandbox is not configured."""
        skill_file = tmp_path / "math.skills.md"
        skill_file.write_text(
            "# Math Skill\n\n```python\nprint(1 + 1)\n```\n",
            encoding="utf-8",
        )

        agent = MemAgent(
            instruction="Skill execution test",
            skill_paths=[str(skill_file)],
        )
        result, _ = agent.tool_manager.execute_tool(
            "run_skill_code", {"skill_name": "Math Skill"}
        )
        assert isinstance(result, dict)
        assert "execution" in result
        assert result["execution"].get("ok") is False
        assert "Sandbox provider is required" in result["execution"].get("error", "")

    @pytest.mark.unit
    def test_memagent_registers_mcp_tools(self):
        """MCP config should expose MCP helper tools."""
        agent = MemAgent(
            instruction="MCP test",
            mcp_servers=[
                {
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                }
            ],
        )

        tool_names = set(agent.tool_manager.list_tools())
        assert "list_mcp_servers" in tool_names
        assert "mcp_list_tools" in tool_names
        assert "mcp_call_tool" in tool_names
        assert "mcp_list_resources" in tool_names
        assert "mcp_list_resource_templates" in tool_names
        assert "mcp_read_resource" in tool_names
        assert "mcp_list_prompts" in tool_names
        assert "mcp_get_prompt" in tool_names

        result, _ = agent.tool_manager.execute_tool("list_mcp_servers", {})
        assert isinstance(result, dict)
        assert result.get("servers")
        assert result["servers"][0]["name"] == "filesystem"

        agent.with_mcp_servers([])
        assert "mcp_call_tool" not in set(agent.tool_manager.list_tools())

    @pytest.mark.unit
    def test_memagent_registers_skills_marketplace_tool(self):
        """Skills marketplace config should expose the marketplace search tool."""
        agent = MemAgent(
            instruction="Marketplace test",
            skills_marketplace_provider="skillsmp",
            skills_marketplace_config={"api_key": "sk_test_skillsmp"},
        )

        tool_names = set(agent.tool_manager.list_tools())
        assert "skills_marketplace_search" in tool_names

        with patch.object(
            agent,
            "_run_skills_marketplace_request",
            return_value={
                "ok": True,
                "status_code": 200,
                "response": {"data": {"skills": [{"name": "SEO Assistant"}]}},
            },
        ):
            result, _ = agent.tool_manager.execute_tool(
                "skills_marketplace_search",
                {"q": "SEO"},
            )
        assert isinstance(result, dict)
        assert result.get("ok") is True
        assert result.get("count") == 1

    @pytest.mark.unit
    def test_skills_marketplace_cloudflare_error_is_normalized(self):
        """Cloudflare HTML blocks should map to a concise actionable tool error."""
        agent = MemAgent(
            instruction="Marketplace Cloudflare test",
            skills_marketplace_provider="skillsmp",
            skills_marketplace_config={"api_key": "sk_test_skillsmp"},
        )
        with patch.object(
            agent,
            "_run_skills_marketplace_request",
            return_value={
                "ok": False,
                "status_code": 403,
                "error_code": None,
                "error": "<!doctype html><title>Access denied | skillsmp.com used Cloudflare to restrict access</title><h1>Error 1010</h1>",
                "response": None,
            },
        ):
            result, _ = agent.tool_manager.execute_tool(
                "skills_marketplace_search",
                {"q": "spotify playlist url"},
            )

        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert result.get("status_code") == 403
        assert "CLOUDFLARE_ACCESS_DENIED" in str(result.get("error", ""))
        assert "backend runtime" in str(result.get("error", "")).lower()

    @pytest.mark.unit
    def test_memagent_graalpy_missing_executable_degrades_gracefully(self):
        """Selecting GraalPy without executable should degrade gracefully."""
        missing_binary = f"/tmp/graalpy-missing-{uuid.uuid4().hex}"

        agent = MemAgent(
            instruction="Sandbox validation test",
            sandbox_provider={
                "provider": "graalpy",
                "graalpy_path": missing_binary,
            },
        )

        # Agent should initialize successfully but without sandbox
        assert agent.sandbox_manager is None
        assert not agent.has_sandbox()


class TestMemAgentRun:
    """Test the MemAgent run method."""

    @pytest.mark.unit
    def test_run_basic_query(self, memagent_with_mocks):
        """Test basic query execution."""
        agent = memagent_with_mocks

        response = agent.run("What is 2+2?")

        assert_agent_response_valid(response)
        # Verify LLM was called
        assert agent.model.generate.called

    @pytest.mark.unit
    def test_run_with_memory_id(self, memagent_with_mocks):
        """Test query execution with specific memory ID."""
        agent = memagent_with_mocks

        response = agent.run("Remember this conversation", memory_id="test_memory_123")

        assert_agent_response_valid(response)
        # Verify memory operations were attempted
        # (Detailed memory testing in test_memory_manager.py)

    @pytest.mark.unit
    def test_run_with_thread_id(self, memagent_with_mocks):
        """Test query execution with thread ID."""
        agent = memagent_with_mocks

        response = agent.run("Continue our chat", thread_id="conv_456")

        assert_agent_response_valid(response)

    @pytest.mark.unit
    def test_resume_thread_updates_active_memory_and_thread(self, memagent_with_mocks):
        agent = memagent_with_mocks

        resumed = agent.resume_thread("memory-resume", "thread-resume")

        assert resumed == "thread-resume"
        assert agent.get_current_memory_id() == "memory-resume"
        assert agent.get_current_thread_id() == "thread-resume"
        assert agent._thread_ids_by_memory["memory-resume"] == "thread-resume"

    @pytest.mark.unit
    def test_run_error_handling(self, mock_memory_provider):
        """Test error handling in run method."""
        # Create agent with failing LLM
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("LLM failed")

        agent = MemAgent(
            model=failing_llm,
            memory_provider=mock_memory_provider,
            instruction="Test error handling",
        )

        response = agent.run("This will fail")

        # Should get error response, not crash
        assert isinstance(response, str)
        assert "error" in response.lower()

    @pytest.mark.unit
    def test_run_without_llm(self, mock_memory_provider):
        """Test run method without LLM model."""
        agent = MemAgent(
            instruction="Test without LLM", memory_provider=mock_memory_provider
        )

        response = agent.run("Test query")

        # Should get error response about missing model
        assert isinstance(response, str)
        assert "no llm model" in response.lower()

    @pytest.mark.unit
    def test_run_context_building(self, memagent_with_mocks):
        """Test that context is properly built for queries."""
        agent = memagent_with_mocks

        # Add some mock conversation history
        memory_id = "test_memory_context"
        agent.memory_ids = [memory_id]

        response = agent.run("Test context", memory_id=memory_id)

        assert_agent_response_valid(response)
        # Verify memory manager was called for context
        assert agent.memory_manager.load_conversation_history.called

    @pytest.mark.unit
    def test_run_context_uses_extended_history_limit(self, memagent_with_mocks):
        """Context builder should fetch more than a tiny fixed history window."""
        agent = memagent_with_mocks

        agent._build_context("Keep context", "memory_for_history_limit")

        assert agent.memory_manager.load_conversation_history.called
        _, kwargs = agent.memory_manager.load_conversation_history.call_args
        assert int(kwargs.get("limit", 0)) > 10

    @pytest.mark.unit
    def test_run_context_uses_only_the_active_thread(self, memagent_with_mocks):
        agent = memagent_with_mocks
        agent.resume_thread("memory-resume", "thread-resume")

        agent._build_context("Continue this conversation", "memory-resume")

        _, kwargs = agent.memory_manager.load_conversation_history.call_args
        assert kwargs["thread_id"] == "thread-resume"

    @pytest.mark.unit
    def test_build_prompt_messages_keeps_recent_history_beyond_five(
        self, memagent_with_mocks
    ):
        """Prompt assembly should not truncate thread history to only five messages."""
        agent = memagent_with_mocks
        agent._context_window_tokens = None

        history = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"history-message-{i}",
            }
            for i in range(12)
        ]

        prompt_messages = agent._build_prompt_messages(
            "system prompt",
            "current query",
            {"conversation_history": history},
        )

        history_in_prompt = prompt_messages[1:-1]
        assert len(history_in_prompt) == 12
        assert history_in_prompt[0]["content"] == "history-message-0"
        assert history_in_prompt[-1]["content"] == "history-message-11"

    @pytest.mark.unit
    def test_prepare_history_messages_respects_budget(self, memagent_with_mocks):
        """History assembly should trim for tight context windows while keeping recency."""
        agent = memagent_with_mocks
        agent._context_window_tokens = 256

        history = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"long-history-{i}-" + ("x" * 220),
            }
            for i in range(18)
        ]

        selected = agent._prepare_history_messages(
            history, "system prompt for budget", "query"
        )

        assert 0 < len(selected) < len(history)
        assert selected[-1]["content"] == history[-1]["content"]

    @pytest.mark.unit
    def test_run_isolates_thread_state_per_memory(self, memagent_with_mocks):
        """Switching memory_ids should keep per-thread thread IDs isolated."""
        agent = memagent_with_mocks

        response_a = agent.run("Thread A message", memory_id="thread_a")
        assert_agent_response_valid(response_a)
        conv_a = agent.get_current_thread_id()
        assert conv_a

        response_b = agent.run("Thread B message", memory_id="thread_b")
        assert_agent_response_valid(response_b)
        conv_b = agent.get_current_thread_id()
        assert conv_b
        assert conv_b != conv_a

        response_a_2 = agent.run("Thread A follow-up", memory_id="thread_a")
        assert_agent_response_valid(response_a_2)
        assert agent.get_current_thread_id() == conv_a

        assert agent._thread_ids_by_memory.get("thread_a") == conv_a
        assert agent._thread_ids_by_memory.get("thread_b") == conv_b


class TestMemAgentMethods:
    """Test additional MemAgent methods."""

    @pytest.mark.unit
    def test_load_conversation_history(self, memagent_with_mocks):
        """Test loading conversation history."""
        agent = memagent_with_mocks

        history = agent.load_conversation_history("test_memory")

        # Should return a list (empty or with items)
        assert isinstance(history, list)

    @pytest.mark.unit
    def test_add_tool(self, memagent_with_mocks):
        """Test adding a tool to the agent."""
        agent = memagent_with_mocks

        def new_tool(x: int) -> int:
            return x * 2

        result = agent.add_tool(new_tool)

        # Should return True for successful addition
        assert result is True

    @pytest.mark.unit
    def test_set_persona(self, memagent_with_mocks, sample_persona):
        """Test setting agent persona."""
        agent = memagent_with_mocks

        result = agent.set_persona(sample_persona)

        # Should return True for successful setting
        assert result is True

    @pytest.mark.unit
    def test_persona_tools_registered_on_init_with_persona(self):
        """Constructing an agent with a persona should register the
        update_persona / read_persona tools so the LLM can evolve its own
        persona. Uses a dict payload (not SimpleNamespace) so the
        PersonaManager rehydrates it via Persona.from_dict without
        triggering an embedding network call."""
        persona_payload = {
            "name": "Hype Coach",
            "role": "general",
            "goals": "keep momentum high",
            "background": "motivator/hype-man",
            "embedding": None,
        }
        agent = MemAgent(instruction="Persona init test", persona=persona_payload)

        assert agent._persona_tools_registered is True
        assert "update_persona" in agent.tool_manager.tools
        assert "read_persona" in agent.tool_manager.tools

    @pytest.mark.unit
    def test_no_persona_tools_without_persona(self):
        """Agents built without a persona must not expose the persona
        evolution tools — current behavior is intentional (no identity to
        evolve)."""
        agent = MemAgent(instruction="No persona test")

        assert agent._persona_tools_registered is False
        assert "update_persona" not in agent.tool_manager.tools
        assert "read_persona" not in agent.tool_manager.tools

    @pytest.mark.unit
    def test_set_persona_wrapper_registers_tools(self):
        """The public set_persona wrapper (core.py:4925) must register persona
        tools when attaching a persona after construction."""
        agent = MemAgent(instruction="Late persona attach")
        assert "update_persona" not in agent.tool_manager.tools

        result = agent.set_persona(
            {"name": "Hype Coach", "role": "general", "embedding": None},
            save=False,
        )

        assert result is True
        assert agent._persona_tools_registered is True
        assert "update_persona" in agent.tool_manager.tools
        assert "read_persona" in agent.tool_manager.tools

    @pytest.mark.unit
    def test_refresh_registers_persona_tools_when_persona_added(
        self, mock_memory_provider
    ):
        """Regression: refresh() used to call persona_manager.set_persona
        directly, bypassing the public wrapper, so a persona added via the
        UI wouldn't get the update_persona / read_persona tools until the
        agent was fully reloaded. After the fix, refresh() must go through
        the wrapper and register the tools in-place."""
        agent = MemAgent(
            instruction="Refresh persona test",
            memory_provider=mock_memory_provider,
            agent_id="agent_refresh_persona",
        )
        assert "update_persona" not in agent.tool_manager.tools

        saved = MemAgentModel(
            agent_id="agent_refresh_persona",
            instruction="Refresh persona test",
            persona={
                "name": "Hype Coach",
                "role": "general",
                "embedding": None,
            },
        )
        mock_memory_provider.retrieve_memagent = MagicMock(return_value=saved)

        result = agent.refresh()

        assert result is agent
        assert agent._persona_tools_registered is True
        assert "update_persona" in agent.tool_manager.tools
        assert "read_persona" in agent.tool_manager.tools

    @pytest.mark.unit
    def test_register_persona_tools_is_idempotent(self):
        """_register_persona_tools is called unconditionally on construction
        AND via the set_persona wrapper. Calling it twice must not double-
        register or error — the guard flag prevents re-entry."""
        persona_payload = {
            "name": "Hype Coach",
            "role": "general",
            "embedding": None,
        }
        agent = MemAgent(instruction="Idempotent persona", persona=persona_payload)
        assert agent._persona_tools_registered is True

        # A second call should be a no-op (guarded by _persona_tools_registered).
        agent._register_persona_tools()

        assert "update_persona" in agent.tool_manager.tools
        assert "read_persona" in agent.tool_manager.tools


class TestMemAgentConfig:
    """Test MemAgentConfig class."""

    @pytest.mark.unit
    def test_config_initialization_default(self):
        """Test default config initialization."""
        config = MemAgentConfig()

        assert config.instruction == DEFAULT_INSTRUCTION
        assert config.max_steps == DEFAULT_MAX_STEPS
        assert config.tool_access == "private"
        assert config.semantic_cache == False

    @pytest.mark.unit
    def test_config_initialization_custom(self):
        """Test custom config initialization."""
        config = MemAgentConfig(
            instruction="Custom instruction",
            max_steps=50,
            semantic_cache=True,
            custom_param="custom_value",
        )

        assert config.instruction == "Custom instruction"
        assert config.max_steps == 50
        assert config.semantic_cache == True
        assert config.custom_param == "custom_value"

    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = MemAgentConfig(instruction="Test", max_steps=15, semantic_cache=True)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["instruction"] == "Test"
        assert config_dict["max_steps"] == 15
        assert config_dict["semantic_cache"] == True


class TestMemAgentModel:
    """Test MemAgentModel class."""

    @pytest.mark.unit
    def test_model_initialization_default(self):
        """Test default model initialization."""
        model = MemAgentModel()

        assert model.instruction == DEFAULT_INSTRUCTION
        assert model.max_steps == DEFAULT_MAX_STEPS
        assert model.tool_access == "private"
        assert model.semantic_cache == False
        assert model.application_mode == "assistant"
        assert model.self_aware is False
        assert model.self_aware_config is None

    @pytest.mark.unit
    def test_model_initialization_custom(self):
        """Test custom model initialization."""
        model = MemAgentModel(
            instruction="Custom model instruction",
            max_steps=30,
            agent_id="custom_agent",
            semantic_cache=True,
            memory_ids=["mem1", "mem2"],
        )

        assert model.instruction == "Custom model instruction"
        assert model.max_steps == 30
        assert model.agent_id == "custom_agent"
        assert model.semantic_cache == True
        assert model.memory_ids == ["mem1", "mem2"]

    @pytest.mark.unit
    def test_model_validation(self):
        """Test model validation."""
        # Should not raise validation errors
        model = MemAgentModel(max_steps=10, instruction="Valid instruction")

        assert model.max_steps == 10
        assert model.instruction == "Valid instruction"

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test model serialization."""
        model = MemAgentModel(
            instruction="Test serialization", max_steps=20, semantic_cache=True
        )

        # Should be able to convert to dict
        model_dict = model.model_dump()

        assert isinstance(model_dict, dict)
        assert model_dict["instruction"] == "Test serialization"
        assert model_dict["max_steps"] == 20
        assert model_dict["semantic_cache"] == True
