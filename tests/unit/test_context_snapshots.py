"""Captures freeze actual fitted inputs and enforce exact identity scopes."""

import copy

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability.context_snapshots import ContextSnapshots

pytestmark = pytest.mark.unit


@pytest.fixture
def snapshots(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    return ContextSnapshots(provider)


def _capture(store, span="span", **identity):
    return store.capture(
        identity={
            "agent_id": "a",
            "memory_id": "m",
            "turn_id": "turn",
            "user_id": "alice",
            **identity,
        },
        span_id=span,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
        model="local",
        window_tokens=4096,
        iteration=1,
        stage="answer",
    )


def test_capture_is_copied_and_first_write_wins(snapshots):
    messages = [{"role": "user", "content": "original"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    first = snapshots.capture(
        identity={"agent_id": "a", "memory_id": "m"},
        span_id="span",
        messages=messages,
        tools=tools,
        model="local",
        window_tokens=4096,
        iteration=1,
        stage="answer",
    )
    messages[0]["content"] = "mutated"
    tools[0]["function"]["name"] = "mutated"
    _capture(snapshots)
    saved = snapshots.get(first["record_id"], agent_id="a", memory_id="m")
    assert saved["messages"][0]["content"] == "original"
    assert saved["tools"][0]["function"]["name"] == "lookup"
    assert "messages" not in snapshots.list(agent_id="a")[0]


def test_snapshot_reads_enforce_agent_namespace_and_tenant(snapshots):
    first = _capture(snapshots, application_id="app")
    assert snapshots.get(
        first["record_id"],
        agent_id="a",
        memory_id="m",
        user_id="alice",
        application_id="app",
    )
    for filters in (
        {"agent_id": "b"},
        {"agent_id": "a", "memory_id": "other"},
        {"agent_id": "a", "user_id": "bob"},
        {"agent_id": "a", "application_id": "other"},
    ):
        assert snapshots.get(first["record_id"], **filters) is None
    assert snapshots.list(agent_id="a", memory_id="other") == []
    assert snapshots.list(agent_id="a", user_id=None) == []


def test_memagent_captures_after_fitting_before_provider_mutates(
    memagent_with_mocks, snapshots, monkeypatch
):
    agent = memagent_with_mocks
    agent.memory_provider = snapshots.provider
    agent.capture_context_snapshots = True
    agent._context_window_tokens = 512
    monkeypatch.setattr(agent, "_get_model_context_window", lambda: None)
    agent._current_memory_id = "m"
    agent._current_thread_id = "thread"
    agent._begin_trace_turn("alice")
    seen = []

    def generate(messages, tools=None):
        seen.append(copy.deepcopy(messages))
        messages[0]["content"] = "provider mutation"
        return "done"

    agent.model.generate.side_effect = generate
    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "old" * 400},
        {"role": "assistant", "content": "old answer" * 200},
        {"role": "user", "content": "Current question"},
    ]
    agent._generate_with_trace(messages, tools=[], iteration=1, stage="answer")
    metadata = snapshots.list(agent_id=agent.agent_id, memory_id="m")
    assert len(metadata) == 1
    saved = snapshots.get(
        metadata[0]["record_id"], agent_id=agent.agent_id, user_id="alice"
    )
    assert saved["messages"] == seen[0]
    assert len(saved["messages"]) < len(messages)
    assert saved["messages"][-1]["content"] == "Current question"


def test_capture_failure_does_not_fail_model_response(memagent_with_mocks, monkeypatch):
    agent = memagent_with_mocks
    agent.capture_context_snapshots = True
    monkeypatch.setattr(
        ContextSnapshots,
        "capture",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = agent._generate_with_trace(
        [{"role": "user", "content": "question"}], tools=[], iteration=1, stage="answer"
    )
    assert result == agent.model.generate.return_value


def test_namespace_filter_happens_before_pagination_and_cursor_traversal(snapshots):
    for i, namespace in enumerate(("target", "other", "target", "other", "other")):
        snapshots.capture(
            identity={"agent_id": "a", "memory_id": namespace},
            span_id=str(i),
            messages=[{"role": "user", "content": str(i)}],
            tools=[],
            model="local",
            window_tokens=1024,
            iteration=1,
            stage="answer",
        )
    first = snapshots.page(agent_id="a", memory_id="target", limit=1)
    assert len(first["items"]) == 1 and first["next_cursor"]
    second = snapshots.page(
        agent_id="a", memory_id="target", limit=1, cursor=first["next_cursor"]
    )
    assert len(second["items"]) == 1 and second["next_cursor"] is None
    assert first["items"][0]["record_id"] != second["items"][0]["record_id"]


def test_oracle_style_trace_payloads_page_by_event_time_instead_of_random_id():
    from memorizz.memory_provider.base import MemoryProvider

    class OracleShape:
        query_observability_records = MemoryProvider.query_observability_records

        def list_all(self, kind):
            return [
                {
                    "memory_id": record,
                    "content": {
                        "record_type": "observability_trace_bundle",
                        "memory_id": "thread",
                        "agent_id": "a",
                        "started_at": f"2026-10-07T10:00:0{i}Z",
                    },
                }
                for i, record in enumerate(("z-earliest", "m-middle", "a-latest"))
            ]

    provider = OracleShape()
    filters = dict(
        memory_ids=["thread"], record_type="observability_trace_bundle", limit=1
    )
    first = provider.query_observability_records(MemoryType.SHARED_MEMORY, **filters)
    assert first["items"][0]["memory_id"] == "a-latest"
    second = provider.query_observability_records(
        MemoryType.SHARED_MEMORY, cursor=first["next_cursor"], **filters
    )
    assert second["items"][0]["memory_id"] == "m-middle"
