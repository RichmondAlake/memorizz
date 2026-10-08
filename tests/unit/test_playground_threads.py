"""Renaming and deleting playground conversations, and choosing the window."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")

from memorizz import MemAgent  # noqa: E402
from memorizz.enums import MemoryType  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", Mock())
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )


def _saved_agent(memory, **fields):
    from memorizz.llms.openai import OpenAI

    agent = MemAgent(
        model=OpenAI(model="gpt-test"), memory_provider=memory, auto_register=False
    )
    for key, value in fields.items():
        setattr(agent, key, value)
    agent.save()
    agent_id = agent.agent_id
    agent.close()
    return agent_id


def test_playground_context_link_selects_the_requested_conversation(ui, memory):
    agent_id = _saved_agent(memory, memory_ids=["one", "two"])
    response = ui.get(
        f"/agents/{agent_id}/playground?memory_id=two&inspector=context&turn_id=selected"
    )
    assert response.status_code == 200
    assert 'id="pg-memory-id" type="hidden" value="two"' in response.text
    assert (
        ui.get(f"/agents/{agent_id}/playground?memory_id=someone-else").status_code
        == 404
    )


@pytest.mark.parametrize(
    "ids,expected,compacted",
    [
        ([], "No compaction performed.", False),
        (["summary"], "Context compacted: 1 summary created.", True),
        (["s1", "s2"], "Context compacted: 2 summaries created.", True),
    ],
)
def test_compaction_message_reports_only_actual_changes(
    ui, memory, monkeypatch, ids, expected, compacted
):
    agent_id = _saved_agent(memory, memory_ids=["m"])
    instance = SimpleNamespace(
        context_policy=SimpleNamespace(keep_recent_messages=6),
        _current_user_id=None,
        _current_thread_id=None,
        generate_summaries=Mock(return_value=ids),
    )
    monkeypatch.setattr(MemAgent, "load", lambda *a, **k: instance)
    response = ui.post(
        f"/agents/{agent_id}/playground/compact", data={"memory_id": "m"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"].startswith(expected)
    assert payload["compacted"] is compacted
    assert payload["summary_count"] == len(ids)


def test_context_turn_history_uses_real_requests_and_marks_missing_turns(ui, memory):
    from memorizz.observability.context_snapshots import ContextSnapshots
    from memorizz.observability.store import ObservabilityStore

    agent_id = _saved_agent(memory, memory_ids=["m", "other"])
    snapshots = ContextSnapshots(memory)
    saved = snapshots.capture(
        identity={
            "agent_id": agent_id,
            "memory_id": "m",
            "turn_id": "new",
            "timestamp": "2026-10-07T12:00:00Z",
        },
        span_id="new",
        messages=[{"role": "user", "content": "actual question"}],
        tools=[],
        model="local",
        window_tokens=1024,
        iteration=1,
        stage="answer",
    )
    for turn, events in (
        ("old", [{"trace_kind": "model_call"}]),
        ("cache", [{"trace_kind": "cache_decision"}]),
        ("envelope", [{"run_id": "harness-run", "type": "model_call", "data": {}}]),
    ):
        ObservabilityStore(memory)._put(
            {
                "record_id": "bundle-" + turn,
                "record_type": "observability_trace_bundle",
                "agent_id": agent_id,
                "memory_id": "m",
                "turn_id": turn,
                "events": events,
            }
        )
    url = f"/agents/{agent_id}/playground/context-history"
    response = ui.get(url, params={"memory_id": "m"})
    assert response.status_code == 200, response.text
    turns = {t["turn_id"]: t for t in response.json()["turns"]}
    assert turns["new"]["calls"][0]["record_id"] == saved["record_id"]
    assert turns["old"]["status"] == "unavailable"
    assert turns["cache"]["status"] == "no_model_request"
    assert "envelope" not in turns
    page = ui.get(url, params={"memory_id": "m", "limit": 1})
    assert "next_cursors" in page.json()
    detail = ui.get(url + "/" + saved["record_id"], params={"memory_id": "m"})
    assert detail.json()["snapshot"]["messages"] == [
        {"role": "user", "content": "actual question"}
    ]
    assert (
        ui.get(
            url + "/" + saved["record_id"], params={"memory_id": "other"}
        ).status_code
        == 404
    )
    assert ui.get(url, params={"memory_id": "unknown"}).status_code == 404
    ui.delete(f"/api/agents/{agent_id}/threads/m")
    assert snapshots.get(saved["record_id"], agent_id=agent_id) is None


def test_captured_request_respects_content_policy_and_tenant_scope(
    ui, memory, monkeypatch
):
    from memorizz.observability.context_snapshots import ContextSnapshots
    from memorizz.ui.routers import memory_history as routes

    agent_id = _saved_agent(memory, memory_ids=["m"])
    snapshot = ContextSnapshots(memory).capture(
        identity={"agent_id": agent_id, "memory_id": "m", "user_id": "alice"},
        span_id="private-span",
        messages=[{"role": "user", "content": "private request"}],
        tools=[],
        model="local",
        window_tokens=1024,
        iteration=1,
        stage="answer",
    )
    url = f"/agents/{agent_id}/playground/context-history/{snapshot['record_id']}"
    assert ui.get(url, params={"memory_id": "m", "user_id": "bob"}).status_code == 404
    assert ui.get(url, params={"memory_id": "m", "user_id": "alice"}).status_code == 200
    monkeypatch.setattr(routes, "trace_content_mode", lambda: "metadata")
    payload = ui.get(url, params={"memory_id": "m"}).json()
    assert payload["content_mode"] == "metadata"
    assert "messages" not in payload["snapshot"] and "tools" not in payload["snapshot"]
    monkeypatch.setattr(
        routes, "scoped_trace_filters", lambda *_args: {"user_id": "bob"}
    )
    assert ui.get(url, params={"memory_id": "m"}).status_code == 404


def test_thread_titles_survive_save_and_load(memory):
    agent_id = _saved_agent(memory, thread_titles={"t1": "Trip plans"})
    loaded = MemAgent.load(agent_id, memory_provider=memory, auto_register=False)
    try:
        assert loaded.thread_titles == {"t1": "Trip plans"}
    finally:
        loaded.close()


def test_loading_agent_preserves_runtime_capture_overrides(memory):
    agent_id = _saved_agent(memory)
    loaded = MemAgent.load(
        agent_id,
        memory_provider=memory,
        auto_register=False,
        capture_context_snapshots=True,
        capture_memory_history=True,
    )
    try:
        assert loaded.capture_context_snapshots is True
        assert loaded.capture_memory_history is True
        record_id = memory.store(
            {"content": "tracked change"}, MemoryType.KNOWLEDGE_BASE
        )
        assert any(
            event["target_record_id"] == record_id
            for event in loaded.memory_history.timeline()["events"]
        )
    finally:
        loaded.close()


def _message(agent_id, memory_id, index, owner=None):
    return {
        "role": "user" if index % 2 == 0 else "assistant",
        "content": f"{memory_id} message {index}",
        "memory_id": memory_id,
        "agent_id": owner or agent_id,
        "user_id": None,
        "timestamp": f"2026-09-29T10:00:{index:02d}",
    }


@pytest.fixture
def ui(memory, monkeypatch):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    client = TestClient(create_app())
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", memory)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})
        yield client


def test_rename_and_delete_a_conversation(ui, memory):
    from memorizz.ui.helpers import _build_agent_threads, _load_agent

    agent_id = _saved_agent(memory, memory_ids=["t1", "t2"])
    for memory_id in ("t1", "t2"):
        for index in range(3):
            memory.store(
                _message(agent_id, memory_id, index), MemoryType.CONVERSATION_MEMORY
            )
    # Another agent's rows in the same conversation are not ours to delete.
    memory.store(
        _message(agent_id, "t1", 9, owner="someone-else"),
        MemoryType.CONVERSATION_MEMORY,
    )
    memory.store(
        {
            "memory_id": "t1",
            "agent_id": agent_id,
            "content": "Gist.",
            "summary_type": "compaction",
            "created_at": "2026-09-29T10:01:00",
        },
        MemoryType.SUMMARIES,
    )
    memory.store(
        {
            "memory_id": "t2",
            "agent_id": agent_id,
            "content": "Other conversation.",
            "summary_type": "compaction",
            "created_at": "2026-09-29T10:02:00",
        },
        MemoryType.SUMMARIES,
    )
    memory.store(
        {
            "tool_log_id": "log-1",
            "memory_id": "t1",
            "agent_id": agent_id,
            "tool_name": "lookup",
            "timestamp": "2026-09-29T10:00:05",
        },
        MemoryType.TOOL_LOG,
    )

    url = f"/api/agents/{agent_id}/threads"
    renamed = ui.post(f"{url}/t1", json={"title": "  Trip   plans  "})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == "Trip plans"
    threads = {
        row["memory_id"]: row for row in _build_agent_threads(_load_agent(agent_id))
    }
    assert threads["t1"]["title"] == "Trip plans"
    assert ui.post(f"{url}/missing", json={"title": "x"}).status_code == 404
    assert len(ui.post(f"{url}/t2", json={"title": "y" * 200}).json()["title"]) == 80

    deleted = ui.delete(f"{url}/t1")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] == {"messages": 3, "summaries": 1, "tool_logs": 1}
    agent = _load_agent(agent_id)
    assert list(agent.memory_ids) == ["t2"]
    assert "t1" not in (agent.thread_titles or {})
    left = memory.retrieve_conversation_history_ordered_by_timestamp(memory_id="t1")
    assert [row["agent_id"] for row in left] == ["someone-else"]
    kept = memory.retrieve_conversation_history_ordered_by_timestamp(memory_id="t2")
    assert len(kept) == 3
    others = [row["memory_id"] for row in memory.list_all(MemoryType.SUMMARIES)]
    assert others == ["t2"]
    assert ui.delete(f"{url}/t1").status_code == 404


def test_settings_window_replaces_the_agent_cap(ui, memory):
    from memorizz.ui.helpers import _load_agent
    from memorizz.ui.routers.playground import _agent_context_window

    agent_id = _saved_agent(memory)
    record = _load_agent(agent_id)
    record.llm_config = {"provider": "openai", "model": "gpt-test"}
    record.context_window_tokens = 4096
    record.context_window_source = "explicit"
    memory.store_memagent(record)

    form = {"llm_provider": "openai", "llm_model": "gpt-test", "max_steps": "20"}
    saved = ui.post(
        f"/agents/{agent_id}/playground/config",
        data={**form, "context_window_tokens": "32768"},
        follow_redirects=False,
    )
    assert saved.status_code == 302 and "config_error" not in saved.headers["location"]
    record = _load_agent(agent_id)
    assert record.llm_config["context_window_tokens"] == 32768
    assert record.context_window_tokens is None
    assert _agent_context_window(record) == 32768

    ui.post(
        f"/agents/{agent_id}/playground/config",
        data={**form, "context_window_tokens": ""},
        follow_redirects=False,
    )
    assert "context_window_tokens" not in _load_agent(agent_id).llm_config


def test_legacy_record_cap_gives_way_to_the_configured_window():
    from memorizz.memagent.models import MemAgentModel
    from memorizz.ui.routers.playground import _agent_context_window

    legacy = MemAgentModel(
        instruction="x",
        context_window_tokens=8192,
        llm_config={"provider": "openai", "model": "m", "context_window_tokens": 32768},
    )
    assert _agent_context_window(legacy) == 32768
    legacy.context_window_source = "explicit"
    assert _agent_context_window(legacy) == 8192


def test_context_length_endpoint(monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Mock()))
    monkeypatch.setattr(
        "memorizz.llms.ollama._model_context_length",
        lambda client, host, model: 131072 if model == "gemma4" else None,
    )
    monkeypatch.setattr(
        "memorizz.llms.ollama._auto_context_window",
        lambda client, host, model: 65536 if model == "gemma4" else None,
    )
    client = TestClient(create_app())
    body = client.get("/api/ollama/context-length?model=gemma4").json()
    assert body == {
        "ok": True,
        "model": "gemma4",
        "context_length": 131072,
        "default_window": 65536,
    }
    assert client.get("/api/ollama/context-length").status_code == 400


def test_stream_reports_the_model_that_answers():
    from memorizz.llms.streaming import streaming_capabilities

    provider = SimpleNamespace(model="gemma4:latest")
    assert streaming_capabilities(provider)["model"] == "gemma4:latest"


def test_unscoped_tool_log_listing_returns_every_users_rows(memory):
    memory.store(
        {"tool_log_id": "log-1", "memory_id": "t1", "user_id": None, "timestamp": "1"},
        MemoryType.TOOL_LOG,
    )
    memory.store(
        {"tool_log_id": "log-2", "memory_id": "t1", "user_id": "u2", "timestamp": "2"},
        MemoryType.TOOL_LOG,
    )
    rows = memory.list_tool_logs(memory_id="t1")
    assert [row["tool_log_id"] for row in rows] == ["log-2", "log-1"]
    scoped = memory.list_tool_logs(memory_id="t1", user_id="u2")
    assert [row["tool_log_id"] for row in scoped] == ["log-2"]


def test_panel_names_the_model_and_its_longest_window(monkeypatch):
    from memorizz.llms.ollama import OllamaLLM
    from memorizz.ui.routers.playground import _build_token_stats

    monkeypatch.setattr(
        "memorizz.llms.ollama._model_context_length", lambda *args: 131072
    )
    model = OllamaLLM.__new__(OllamaLLM)
    model.model, model.client, model._host = "gemma4:latest", object(), "local"
    model.get_context_window_tokens = lambda: 16384
    agent = MemAgent(
        memory_provider=False, auto_register=False, context_window_tokens=16384
    )
    agent.model = model
    try:
        stats = _build_token_stats(agent, [], toolbox_memory=[])
        assert stats["model_name"] == "gemma4:latest"
        assert stats["model_context_tokens"] == 131072
        assert stats["context_window_tokens"] == 16384
    finally:
        agent.model = None
        agent.close()


def test_thread_count_matches_the_list_not_run_traces(ui, memory):
    import json

    from memorizz.ui.helpers import _build_agent_threads, _load_agent

    agent_id = _saved_agent(memory, memory_ids=["t1"])
    for index in range(2):
        memory.store(_message(agent_id, "t1", index), MemoryType.CONVERSATION_MEMORY)
    memory.store(
        {
            "role": "tool",
            "content": json.dumps({"type": "trace_bundle", "events": []}),
            "memory_id": "t1",
            "agent_id": agent_id,
            "timestamp": "2026-09-29T10:00:05",
        },
        MemoryType.CONVERSATION_MEMORY,
    )
    listed = _build_agent_threads(_load_agent(agent_id))[0]["message_count"]
    payload = ui.get(f"/agents/{agent_id}/playground/thread?memory_id=t1").json()
    assert any(row.get("message_type") == "trace_bundle" for row in payload["messages"])
    assert payload["message_count"] == 2
    assert listed == 2
