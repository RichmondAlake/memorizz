# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Saving and reloading an agent keeps what was stored, and only that.

A run that changes settings for itself (a scheduled job, a harness run, a
playground message) never saves them; reloaded tools come from code; the
entity-memory switch, WhatsApp and embedding settings round-trip; refresh
applies every stored field; agent-wide knowledge and skills serve every user
while a user's own stay private.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.enums import ApplicationMode, ApplicationModeConfig, MemoryType
from memorizz.memagent import MemAgent, MemAgentModel, persistence
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

LLM = {"provider": "ollama", "model": "qwen2.5:7b"}


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


def _saved(provider, **settings) -> MemAgent:
    agent = MemAgent(llm_config=LLM, memory_provider=provider, **settings)
    agent.save()
    return agent


def _edit_record(provider, agent_id, **fields) -> None:
    """Change the stored record as the UI does, behind the agent's back."""
    record = provider.retrieve_memagent(agent_id)
    provider.store_memagent(record.model_copy(update=fields))


# ---------------------------------------------------------------------------
# Run-only overrides are never saved
# ---------------------------------------------------------------------------


def test_a_loaded_agent_registers_new_memories_without_rewriting_its_record(
    provider, monkeypatch
):
    agent = _saved(provider, memory_ids=["m1"], instruction="own")
    loaded = MemAgent.load(
        agent.agent_id, memory_provider=provider, instruction="for this run"
    )
    rewrites = []
    monkeypatch.setattr(persistence, "save_agent", rewrites.append)

    loaded._ensure_agent_registered()  # the stored memories: nothing to write
    loaded.memory_ids.append("m2")
    loaded._ensure_agent_registered()  # a new memory: memory_ids only

    assert rewrites == []
    record = provider.retrieve_memagent(agent.agent_id)
    assert record.memory_ids == ["m1", "m2"]
    assert record.instruction == "own"


def test_a_scheduled_run_leaves_the_agents_automations_on(provider, monkeypatch):
    from memorizz.automation import runner
    from memorizz.automation.models import AutomationJob

    agent = _saved(provider)
    stored_tools = provider.retrieve_memagent(agent.agent_id).tools
    assert "automation_create_job" in [tool["name"] for tool in stored_tools]

    def run(self, query, **kwargs):
        self.save()  # a tool that saves the agent mid-run (update_persona)
        return "done"

    monkeypatch.setattr(MemAgent, "run", run)
    job = AutomationJob(
        job_id="j1",
        agent_id=agent.agent_id,
        name="n",
        schedule_type="interval",
        interval_seconds=60,
        timezone="UTC",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        action_type="agent_query",
        action_config={"query_template": "hi"},
    )
    result = runner.execute_job_action(
        job, scheduled_for_utc=job.next_run_at, memory_provider=provider
    )

    assert result["response"] == "done"
    record = provider.retrieve_memagent(agent.agent_id)
    assert record.automations_enabled is True
    assert record.tools == stored_tools


def test_a_native_harness_run_leaves_the_meta_harness_and_model(
    provider, tmp_path, monkeypatch
):
    from memorizz.metaharness import adapters
    from memorizz.metaharness.base import AdapterOutcome
    from memorizz.metaharness.models import HarnessTask

    provider.store_memagent(
        MemAgentModel(
            agent_id="a1",
            instruction="x",
            llm_config=LLM,
            meta_harness=True,
            meta_harness_mode="delegate",
            default_harness="codex",
        )
    )

    def run(self, task, **kwargs):
        self.agent.save()  # e.g. a new memory registered during the run
        return AdapterOutcome(final_response="ok")

    monkeypatch.setattr(adapters.NativeMemAgentHarness, "run", run)
    outcome = adapters.PersistedMemAgentHarness(provider).run(
        HarnessTask(
            task="x",
            workspace=str(tmp_path),
            harness="memagent",
            agent_id="a1",
            model="ollama/llama3.2",
        ),
        workspace=tmp_path,
        context_pack=SimpleNamespace(rendered="", source_ids=[]),
        emit=lambda event: None,
        cancel_event=SimpleNamespace(is_set=lambda: False),
    )

    assert outcome.final_response == "ok", outcome.error
    record = provider.retrieve_memagent("a1")
    assert record.meta_harness is True
    assert record.meta_harness_mode == "delegate"
    assert record.llm_config == LLM


def test_playground_overrides_are_not_saved_with_the_agent(provider, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    provider.store_memagent(
        MemAgentModel(agent_id="desk", name="Desk", instruction="own", llm_config=LLM)
    )
    ran = []

    class _Stream:
        def poll(self, timeout=None):
            raise StopIteration

        def close(self):
            pass

    def fake_stream(*_args, _prepare=None, **_kwargs):
        with ExitStack() as stack:
            session = SimpleNamespace(
                stack=stack, emit=lambda *a, **k: None, persistence={}
            )
            agent, _ = _prepare(session)
        agent.save()  # a save during the run
        ran.append(agent)
        return _Stream()

    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    with patch.dict(state._state, values), patch(
        "memorizz.streaming.agent_event_stream", fake_stream
    ):
        client = TestClient(create_app(), follow_redirects=False)
        response = client.post(
            "/agents/desk/playground/stream",
            data={"query": "hi", "instruction": "just once", "llm_model": "other"},
        )

    assert response.status_code == 200, response.text
    assert ran[0].instruction == "just once"
    assert ran[0].model.get_config()["model"] == "other"
    record = provider.retrieve_memagent("desk")
    assert record.instruction == "own"
    assert record.llm_config == LLM


# ---------------------------------------------------------------------------
# Tools after a reload come from code
# ---------------------------------------------------------------------------


def test_reloaded_built_in_tools_describe_themselves_from_code(provider):
    agent = _saved(provider)
    record = provider.retrieve_memagent(agent.agent_id)
    tools = [dict(tool) for tool in record.tools]
    for tool in tools:
        if tool["name"] == "knowledge_base_lookup":
            # As saved by a release whose docstring and policy differed.
            tool["description"] = tool["docstring"] = "Docstring before an upgrade."
            tool["tool_policy"] = {"requires_approval": True}
    _edit_record(provider, agent.agent_id, tools=tools)

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
    metadata = loaded.tool_manager.get_tool_metadata("knowledge_base_lookup")

    assert metadata["description"].startswith("Semantic search over knowledge-base")
    assert metadata["tool_policy"] == agent.tool_manager.get_tool_policy(
        "knowledge_base_lookup"
    )
    loaded.save()
    resaved = {
        tool["name"]: tool for tool in provider.retrieve_memagent(agent.agent_id).tools
    }
    assert resaved["knowledge_base_lookup"]["description"] == metadata["description"]


def test_saved_tools_without_a_function_neither_linger_nor_block_built_ins(provider):
    agent = _saved(provider)
    assert agent.tool_manager.get_tool_callable("request_capability")
    stale = [
        {"name": name, "description": "saved copy", "parameters": {}}
        for name in ("request_capability", "list_mcp_servers", "mcp_call_tool")
    ]
    record = provider.retrieve_memagent(agent.agent_id)
    # MCP servers were removed in the UI; their facade tools stayed saved.
    _edit_record(
        provider, agent.agent_id, tools=list(record.tools) + stale, mcp_servers=None
    )

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
    manager = loaded.tool_manager

    assert manager.get_tool_callable("request_capability")
    assert "list_mcp_servers" not in manager.tools
    assert "mcp_call_tool" not in manager.tools
    assert all(manager.get_tool_callable(name) for name in manager.tools)


# ---------------------------------------------------------------------------
# The entity-memory switch round-trips
# ---------------------------------------------------------------------------


def test_entity_memory_switched_off_stays_off_after_reload(provider):
    agent = MemAgent(
        llm_config=LLM, memory_provider=provider, application_mode="assistant"
    )
    assert agent._entity_memory_enabled is True
    agent.with_entity_memory(False)
    agent.save()

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)

    assert loaded._entity_memory_enabled is False
    assert "entity_memory_lookup" not in loaded.tool_manager.tools
    # The mode's defaults are shared by every agent and stay as they were.
    assert MemoryType.ENTITY_MEMORY in ApplicationModeConfig.get_memory_types(
        ApplicationMode.ASSISTANT
    )


def test_entity_memory_switched_on_in_workflow_mode_stays_on_after_reload(provider):
    agent = MemAgent(
        llm_config=LLM, memory_provider=provider, application_mode="workflow"
    )
    agent.with_entity_memory(True)
    agent.save()

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)

    assert loaded._entity_memory_enabled is True
    assert loaded.tool_manager.get_tool_callable("entity_memory_lookup")


def test_an_agents_memory_types_never_change_the_modes_defaults(provider):
    defaults = list(ApplicationModeConfig.get_memory_types(ApplicationMode.WORKFLOW))
    MemAgent(
        llm_config=LLM,
        memory_provider=provider,
        application_mode="workflow",
        continual_learning=True,
    )
    assert ApplicationModeConfig.get_memory_types(ApplicationMode.WORKFLOW) == defaults


def test_builder_clone_keeps_every_setting():
    builder = MemAgentBuilder().with_entity_memory(False).with_tool(len)
    clone = builder.clone()

    assert clone._entity_memory_enabled is False
    assert {k for k in vars(clone)} == {k for k in vars(builder)}
    clone.with_tool(abs)  # containers are copies
    assert builder._tools == [len]


# ---------------------------------------------------------------------------
# WhatsApp and embedding settings round-trip
# ---------------------------------------------------------------------------


def test_whatsapp_settings_survive_a_save(provider):
    agent = _saved(provider)
    _edit_record(
        provider,
        agent.agent_id,
        whatsapp_enabled=True,
        whatsapp_config={"welcome_message": "Hi there"},
    )

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
    loaded.save()

    record = provider.retrieve_memagent(agent.agent_id)
    assert record.whatsapp_enabled is True
    assert record.whatsapp_config == {"welcome_message": "Hi there"}


def test_embedding_settings_are_saved_and_restored(provider):
    agent = _saved(
        provider,
        embedding_provider="openai",
        embedding_config={"model": "text-embedding-3-small"},
    )
    record = provider.retrieve_memagent(agent.agent_id)
    assert record.embedding_provider == "openai"

    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
    assert loaded.embedding_provider == "openai"
    assert loaded.embedding_config == {"model": "text-embedding-3-small"}


# ---------------------------------------------------------------------------
# Refresh applies every stored field
# ---------------------------------------------------------------------------


def test_refresh_then_save_keeps_edits_made_elsewhere(provider, tmp_path):
    from memorizz.mcp import MCPClientManager
    from memorizz.retrieval import RetrievalPolicy
    from memorizz.task_decomposition import normalize_delegation_config
    from memorizz.tool_cache import ToolCacheConfig
    from memorizz.tooling import ContextPolicy, ToolResultPolicy

    provider.store_memagent(MemAgentModel(agent_id="helper", llm_config=LLM))
    agent = _saved(provider, memory_ids=["m1"])
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    edits = {
        "name": "Renamed",
        "instruction": "Edited in the UI",
        "max_steps": 7,
        "application_mode": "workflow",
        "memory_types": ["conversation_memory", "summaries"],
        "context_policy": ContextPolicy.from_value({"tool_top_k": 3}).to_dict(),
        "tool_result_policy": ToolResultPolicy.from_value(
            {"offload_above_chars": 600}
        ).to_dict(),
        "retrieval_policy": RetrievalPolicy.from_value(
            {"knowledge_base_scope": "disabled"}
        ).to_dict(),
        "tool_cache_config": ToolCacheConfig().to_dict(),
        "skill_retrieval": True,
        "skill_retrieval_config": {"top_k": 3},
        "delegation_config": normalize_delegation_config(
            {"enabled": True, "mode": "deterministic", "plan": []},
            for_persistence=True,
        ),
        "delegates": ["helper"],
        "skill_paths": [str(skills_dir)],
        "mcp_servers": MCPClientManager(
            owner_id=agent.agent_id,
            servers=[{"name": "docs", "transport": "stdio", "command": "echo"}],
        ).server_dicts(),
        "default_timezone": "Europe/London",
        "automations_enabled": False,
        "knowledge_base_ids": ["kb-1"],
        "thread_titles": {"m1": "Planning"},
        "is_favorite": True,
        "whatsapp_enabled": True,
        "whatsapp_config": {"welcome_message": "Hi"},
        "embedding_provider": "openai",
    }
    _edit_record(
        provider,
        agent.agent_id,
        llm_config={"provider": "ollama", "model": "llama3.2"},
        **edits,
    )

    assert agent.refresh() is agent
    agent.save()

    record = provider.retrieve_memagent(agent.agent_id)
    for name, value in edits.items():
        assert getattr(record, name) == value, name
    assert record.llm_config["model"] == "llama3.2"
    # Applied to the live agent, not just carried through.
    assert agent.model.get_config()["model"] == "llama3.2"
    assert agent.application_mode == ApplicationMode.WORKFLOW
    assert agent.semantic_tool_router.top_k == 3
    assert [delegate.agent_id for delegate in agent.delegates] == ["helper"]
    assert "automation_create_job" not in agent.tool_manager.tools
    assert agent.tool_manager.get_tool_callable("list_mcp_servers")


def test_refresh_applies_every_setting_load_restores():
    settings = persistence._saved_settings(MemAgentModel())
    stepped = [key for keys, _ in persistence._REFRESH_STEPS for key in keys]

    assert sorted(stepped) == sorted(settings)
    assert len(stepped) == len(set(stepped))


def test_every_saved_field_is_restored():
    read = set()
    defaults = MemAgentModel()

    class _Record:
        def __getattr__(self, name):
            read.add(name)
            return getattr(defaults, name)

    persistence._saved_settings(_Record())

    # The model instance and the id are restored by load_agent itself.
    assert set(MemAgentModel.model_fields) - read == {"model", "agent_id"}


# ---------------------------------------------------------------------------
# Agent-wide knowledge and skills serve every user; a user's own stay private
# ---------------------------------------------------------------------------


_KB_ROWS = [
    {"knowledge_base_id": "kb-1", "content": "handbook", "user_id": None},
    {"knowledge_base_id": "kb-1", "content": "alice notes", "user_id": "alice"},
    {"knowledge_base_id": "kb-1", "content": "bob notes", "user_id": "bob"},
]


class _KnowledgeBase:
    """A store that cannot scope by user, so the agent has to."""

    def __init__(self, memory_provider):
        self.memory_provider = memory_provider

    def retrieve_knowledge_by_query(self, query, namespace=None, limit=5):
        return list(_KB_ROWS[:limit])


def test_knowledge_lookup_shows_agent_wide_and_own_rows_only(monkeypatch, provider):
    monkeypatch.setattr(
        "memorizz.long_term.semantic.knowledge_base.KnowledgeBase", _KnowledgeBase
    )
    agent = MemAgent(memory_provider=provider, auto_register=False)
    agent.knowledge_base_ids = ["kb-1"]

    agent._current_user_id = "alice"
    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "notes"}
    )
    assert [m["content"] for m in result["matches"]] == ["handbook", "alice notes"]

    agent._current_user_id = None
    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "notes"}
    )
    assert [m["content"] for m in result["matches"]] == ["handbook"]


def test_learned_skills_of_another_user_stay_private(provider):
    from memorizz.long_term.procedural.skillbox import Skill, SkillStatus

    skills = [
        Skill(name=name, description=name, content=f"{name} steps", user_id=user)
        for name, user in (("shared", None), ("mine", "alice"), ("theirs", "bob"))
    ]
    for skill in skills:
        skill.status = SkillStatus.ACTIVE

    class _Skillbox:
        def list_skills(self, statuses=None):
            return list(skills)

    agent = MemAgent(memory_provider=provider, auto_register=False)
    agent.skillbox = _Skillbox()
    agent._register_skill_tools()
    agent._current_user_id = "alice"

    listed, _ = agent.tool_manager.execute_tool("list_skills", {})
    assert [s["name"] for s in listed["skills"]] == ["shared", "mine"]
    theirs, _ = agent.tool_manager.execute_tool("read_skill", {"skill_name": "theirs"})
    assert "error" in theirs
    mine, _ = agent.tool_manager.execute_tool("read_skill", {"skill_name": "mine"})
    assert mine["content"] == "mine steps"


def test_an_agent_without_tools_saves_an_empty_list_so_old_rows_are_cleared():
    """None means "this save carries no tools", so removing the last tool
    left its mirrored rows in place."""
    from types import SimpleNamespace

    from memorizz.memagent.persistence import _serialize_tools_for_save

    agent = SimpleNamespace(
        agent_id="a", tool_manager=SimpleNamespace(get_tool_metadata=lambda: [])
    )
    assert _serialize_tools_for_save(agent) == []
    assert _serialize_tools_for_save(SimpleNamespace(tool_manager=None)) is None
