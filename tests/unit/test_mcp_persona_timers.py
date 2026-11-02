# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""MCP clients can set a persona, turn on persona learning and use timers.

A persona's history holds what users wrote, so MCP never returns it, and timers
belong to the person at this machine, so a remote caller can't reach them.
"""

from __future__ import annotations

import json

import pytest

from memorizz import timers
from memorizz.approval import SQLiteApprovalStore
from memorizz.long_term.semantic.persona.persona import Persona
from memorizz.mcp_server import MemorizzMCPServerConfig, MemorizzRuntime
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES, READ_SCOPE, parse_api_key_grants
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memagent import MemAgent, MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

LLM = {"provider": "ollama", "model": "qwen2.5:7b"}


def _identity(*scopes):
    return RequestIdentity(
        principal="alice", scopes=frozenset(scopes or ALL_SCOPES), authenticated=True
    )


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding",
        lambda text, **kwargs: [0.0] * 8,
    )


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


def _runtime(provider, tmp_path, **settings):
    settings.setdefault("transport", "stdio")
    settings.setdefault("allow_writes", True)
    settings.setdefault("exposed_agent_ids", None)
    return MemorizzRuntime(
        MemorizzMCPServerConfig(**settings),
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )


def _remote(provider, tmp_path, **settings):
    """A server on the network, as ``memorizz mcp serve`` runs one."""
    return _runtime(
        provider,
        tmp_path,
        transport="streamable-http",
        public_url="https://memory.example.com",
        api_key_grants=parse_api_key_grants(
            json.dumps({"alice": "alice-token-that-is-long-enough"})
        ),
        **settings,
    )


def _agent(provider, **fields):
    record = MemAgentModel(agent_id="agent-1", name="Agent", instruction="x", **fields)
    provider.store_memagent(record)
    return record


# -- reading an agent --------------------------------------------------------


def test_a_client_sees_the_persona_and_learning_status(provider, tmp_path):
    _agent(
        provider,
        persona=Persona(name="Zed", role="Harbour pilot", goals="Help ships."),
        persona_learning={"enabled": True, "every_n_turns": 7, "auto_apply": False},
        persona_learning_state={
            "pending": {"addition": "x"},
            "reviews": [
                {"at": "2026-11-02", "outcome": "proposed", "code": "proposed"}
            ],
        },
    )
    agent = _runtime(provider, tmp_path).get_agent("agent-1", _identity())["agent"]

    assert agent["persona"]["name"] == "Zed"
    assert agent["persona"]["version"] == 1
    status = agent["persona_learning"]
    assert status["every_n_turns"] == 7
    assert status["auto_apply"] is False
    assert status["change_waiting"] is True
    assert status["last_review"]["code"] == "proposed"


def test_what_users_wrote_never_reaches_a_client(provider, tmp_path):
    persona = Persona(name="Zed")
    persona.evolution_history = [
        {
            "version": 2,
            "changes": {"goals": {"old": "a", "new": "b"}},
            "change_trigger": {
                "reason": "r",
                "evidence": [{"quote": "PRIVATE-QUOTE"}],
            },
        }
    ]
    _agent(
        provider,
        persona=persona,
        persona_learning={"enabled": True},
        persona_learning_state={
            "pending": {"addition": "y", "evidence": [{"quote": "PRIVATE-WAITING"}]},
            "declined": ["PRIVATE-DECLINED"],
        },
    )
    runtime = _runtime(provider, tmp_path)
    identity = _identity()
    for result in (
        runtime.get_agent("agent-1", identity),
        runtime.list_agents(identity),
    ):
        text = json.dumps(result, default=str)
        assert "PRIVATE" not in text
        assert "evolution_history" not in text


def test_an_agent_with_neither_shows_neither(provider, tmp_path):
    _agent(provider)
    agent = _runtime(provider, tmp_path).get_agent("agent-1", _identity())["agent"]
    assert "persona" not in agent
    assert "persona_learning" not in agent


# -- creating ----------------------------------------------------------------


def test_create_with_a_persona_that_learns(provider, tmp_path):
    runtime = _runtime(provider, tmp_path)
    created = runtime.create_agent(
        "Pilot",
        _identity(),
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        persona_name="Zed",
        persona_role="Harbour pilot",
        persona_goals="Help ships.",
        persona_learning=True,
        persona_learning_auto_apply=False,
    )["agent"]

    assert created["persona"]["name"] == "Zed"
    assert created["persona_learning"]["auto_apply"] is False
    loaded = MemAgent.load(created["agent_id"], memory_provider=provider)
    assert loaded.persona_learning.enabled is True
    assert "You are Zed" in loaded._build_system_prompt()


def test_learning_needs_a_persona(provider, tmp_path):
    with pytest.raises(MemorizzServerError, match="no persona to grow"):
        _runtime(provider, tmp_path).create_agent(
            "Pilot", _identity(), persona_learning=True
        )


def test_a_persona_needs_a_name(provider, tmp_path):
    with pytest.raises(MemorizzServerError, match="persona_name is required"):
        _runtime(provider, tmp_path).create_agent(
            "Pilot", _identity(), persona_goals="Help ships."
        )


def test_a_persona_works_without_embeddings(provider, tmp_path, monkeypatch):
    def refuse(text, **kwargs):
        raise RuntimeError("no embedding provider")

    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding", refuse
    )
    _agent(provider)
    agent = _runtime(provider, tmp_path).update_agent(
        "agent-1", _identity(), persona_name="Zed"
    )["agent"]
    assert agent["persona"]["name"] == "Zed"


# -- updating ----------------------------------------------------------------


def test_update_gives_an_agent_a_persona_and_turns_learning_on(provider, tmp_path):
    _agent(provider)
    agent = _runtime(provider, tmp_path).update_agent(
        "agent-1", _identity(), persona_name="Zed", persona_learning=True
    )["agent"]

    assert agent["persona"]["name"] == "Zed"
    assert agent["persona_learning"]["auto_apply"] is True


def test_changing_a_persona_keeps_its_version_and_history(provider, tmp_path):
    persona = Persona(name="Zed", goals="Help ships.")
    _agent(provider, persona=persona)
    runtime = _runtime(provider, tmp_path)
    runtime.update_agent("agent-1", _identity(), persona_goals="Help tugboats.")

    stored = provider.retrieve_memagent("agent-1").persona
    assert stored.persona_id == persona.persona_id
    assert stored.goals == "Help tugboats."
    assert stored.version == 2
    assert stored.evolution_history[-1]["change_trigger"]["source_type"] == "tool_call"


def test_a_refused_call_leaves_the_persona_alone(provider, tmp_path):
    _agent(provider, persona=Persona(name="Zed", goals="Help ships."))
    runtime = _runtime(provider, tmp_path)
    with pytest.raises(MemorizzServerError):
        runtime.update_agent(
            "agent-1",
            _identity(),
            persona_goals="Help tugboats.",
            llm_provider="no-such-provider",
        )
    assert provider.retrieve_memagent("agent-1").persona.version == 1


def test_a_call_that_cant_turn_learning_on_changes_nothing(provider, tmp_path):
    _agent(provider, persona=Persona(name="Zed"), multi_tenant=True)
    with pytest.raises(MemorizzServerError, match="several users"):
        _runtime(provider, tmp_path).update_agent(
            "agent-1", _identity(), persona_goals="New.", persona_learning=True
        )
    stored = provider.retrieve_memagent("agent-1")
    assert stored.persona.version == 1
    assert stored.persona_learning is None


def test_learning_turns_off_and_keeps_what_it_did(provider, tmp_path):
    state = {"declined": ["x"], "last_review_at": 1.0}
    _agent(
        provider,
        persona=Persona(name="Zed"),
        persona_learning={"enabled": True},
        persona_learning_state=state,
    )
    _runtime(provider, tmp_path).update_agent(
        "agent-1", _identity(), persona_learning=False
    )
    stored = provider.retrieve_memagent("agent-1")
    assert stored.persona_learning is None
    assert stored.persona_learning_state == state


def test_review_mode_alone_needs_learning_on(provider, tmp_path):
    _agent(provider, persona=Persona(name="Zed"))
    with pytest.raises(MemorizzServerError, match="off for this agent"):
        _runtime(provider, tmp_path).update_agent(
            "agent-1", _identity(), persona_learning_auto_apply=False
        )


def test_review_mode_changes_on_its_own_when_learning_is_on(provider, tmp_path):
    _agent(
        provider,
        persona=Persona(name="Zed"),
        persona_learning={"enabled": True, "every_n_turns": 9},
    )
    agent = _runtime(provider, tmp_path).update_agent(
        "agent-1", _identity(), persona_learning_auto_apply=False
    )["agent"]
    assert agent["persona_learning"]["auto_apply"] is False
    assert agent["persona_learning"]["every_n_turns"] == 9


def test_an_update_over_a_remote_transport_is_refused(provider, tmp_path):
    _agent(provider)
    runtime = _remote(provider, tmp_path, exposed_agent_ids={"agent-1"})
    with pytest.raises(MemorizzServerError, match="local stdio"):
        runtime.update_agent("agent-1", _identity(), persona_name="Zed")


# -- timers ------------------------------------------------------------------


def test_start_list_and_cancel(provider, tmp_path):
    runtime = _runtime(provider, tmp_path)
    identity = _identity()
    started = runtime.start_timer(identity, minutes=25, label="Focus")["timer"]
    assert started["label"] == "Focus"
    assert started["length"] == "25:00"

    [listed] = runtime.list_timers(identity)["timers"]
    assert listed["timer_id"] == started["timer_id"]
    assert listed["state"] == "running"
    # The CLI and the UI read the same timers.
    assert timers.running()[0]["timer_id"] == started["timer_id"]

    assert runtime.cancel_timer(identity)["cancelled"]["label"] == "Focus"
    assert runtime.list_timers(identity)["timers"] == []


def test_a_bad_length_is_explained(provider, tmp_path):
    with pytest.raises(MemorizzServerError, match="at least one second"):
        _runtime(provider, tmp_path).start_timer(_identity())


def test_several_timers_need_an_id_to_cancel(provider, tmp_path):
    runtime = _runtime(provider, tmp_path)
    identity = _identity()
    runtime.start_timer(identity, minutes=5)
    runtime.start_timer(identity, minutes=9)
    with pytest.raises(MemorizzServerError, match="pass timer_id"):
        runtime.cancel_timer(identity)


def test_reading_timers_needs_only_read_access(provider, tmp_path):
    runtime = _runtime(provider, tmp_path)
    reader = _identity(READ_SCOPE)
    assert runtime.list_timers(reader)["ok"] is True
    with pytest.raises(MemorizzServerError):
        runtime.start_timer(reader, minutes=5)


def test_a_server_without_writes_starts_no_timer(provider, tmp_path):
    runtime = _runtime(provider, tmp_path, allow_writes=False)
    with pytest.raises(MemorizzServerError):
        runtime.start_timer(_identity(), minutes=5)
    assert timers.running() == []


@pytest.mark.parametrize(
    "call",
    [
        lambda runtime, who: runtime.start_timer(who, minutes=5),
        lambda runtime, who: runtime.list_timers(who),
        lambda runtime, who: runtime.cancel_timer(who, timer_id="x"),
    ],
)
def test_timers_stay_on_this_machine(provider, tmp_path, call):
    timers.start(300, "Private label")
    runtime = _remote(provider, tmp_path)
    with pytest.raises(MemorizzServerError, match="local stdio"):
        call(runtime, _identity())


def _saved_agent(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider, memory_ids=["m1"])
    agent.save()
    return agent


def test_an_agent_run_for_a_remote_caller_has_no_timer_tools(provider, tmp_path):
    agent_id = _saved_agent(provider).agent_id
    runtime = _remote(
        provider,
        tmp_path,
        exposed_agent_ids={agent_id},
        allow_agent_execution=True,
    )
    identity = _identity()
    # Inspecting first caches the agent, and the run must not reuse it as is.
    runtime._load_agent_for_inspection(agent_id, identity)
    agent = runtime._load_runnable_agent(agent_id, identity)

    assert not {"start_timer", "list_timers", "cancel_timer"} & set(
        agent.tool_manager.list_tools()
    )
    assert "start_timer" not in agent._build_system_prompt()


def test_a_local_agent_keeps_its_timer_tools(provider, tmp_path):
    agent_id = _saved_agent(provider).agent_id
    runtime = _runtime(provider, tmp_path)
    agent = runtime._load_runnable_agent(agent_id, _identity())
    assert "start_timer" in agent.tool_manager.list_tools()
    assert "start_timer" in agent._build_system_prompt()


def test_a_connected_app_gets_no_timer_tools():
    from memorizz.mcp_server.server import MEMORY_TOOLS

    assert not {
        "memorizz_start_timer",
        "memorizz_list_timers",
        "memorizz_cancel_timer",
    } & set(MEMORY_TOOLS)
