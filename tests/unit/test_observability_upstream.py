import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz import MemAgent
from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore, analyze_trace_events
from memorizz.ui.security import ReadOnlyProviderProxy, redact_trace_events


class _DeterministicToolModel:
    def __init__(self):
        self.calls = 0
        self.model = "test-model"
        self._usage = {}

    def get_config(self):
        return {"provider": "test", "model": self.model}

    def get_last_usage(self):
        return self._usage

    def generate(self, _messages, tools=None):
        self.calls += 1
        self._usage = {
            "prompt_tokens": 10 * self.calls,
            "completion_tokens": 3,
            "total_tokens": 10 * self.calls + 3,
        }
        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="lookup", arguments='{"item":"url"}'),
            )
            message = SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        return "completed"


@pytest.fixture()
def fs_provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "observability-store")
    )


@pytest.mark.unit
def test_named_application_agent_has_stable_identity_and_auto_upserts(fs_provider):
    first = MemAgent(
        name="chat-assistant",
        application_id="openspeech",
        memory_provider=fs_provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
    )
    second = MemAgent(
        name="chat-assistant",
        application_id="openspeech",
        memory_provider=fs_provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
    )

    assert first.agent_id == second.agent_id
    first._resolve_execution_state("memory-1", "thread-1")

    stored = fs_provider.retrieve_memagent(first.agent_id)
    assert stored is not None
    assert stored.application_id == "openspeech"
    assert stored.memory_ids == ["memory-1"]


@pytest.mark.unit
def test_nonstreaming_turn_persists_model_and_tool_spans_with_identity(fs_provider):
    def lookup(item: str):
        """Look up an item."""
        return {"ok": True, "item": item}

    agent = MemAgent(
        name="span-agent",
        application_id="test-app",
        model=_DeterministicToolModel(),
        tools=[lookup],
        memory_provider=fs_provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        context_policy={"progressive_tool_disclosure": False},
    )
    # Keep this unit test independent from embedding-provider configuration.
    agent._schedule_conversation_embedding_backfill = lambda _rows: None

    assert (
        agent.run(
            "remember this url",
            memory_id="memory-1",
            thread_id="thread-1",
            user_id="u1",
        )
        == "completed"
    )

    conversation_rows = fs_provider.list_all(MemoryType.CONVERSATION_MEMORY)
    assert len(conversation_rows) == 2
    rows = fs_provider.list_all(MemoryType.SHARED_MEMORY)
    bundles = []
    for row in rows:
        try:
            payload = json.loads(row.get("content") or "")
        except (TypeError, ValueError):
            continue
        if payload.get("type") == "trace_bundle":
            bundles.append(payload)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["version"] == 2
    assert bundle["agent_id"] == agent.agent_id
    assert bundle["memory_id"] == "memory-1"
    assert bundle["thread_id"] == "thread-1"
    assert bundle["user_id"] == "u1"
    assert bundle["run_id"] and bundle["turn_id"] and bundle["root_trace_id"]

    events = bundle["events"]
    kinds = {event["trace_kind"] for event in events}
    assert {
        "turn_start",
        "model_call",
        "model_result",
        "tool_call",
        "tool_result",
        "turn_result",
    }.issubset(kinds)
    for event in events:
        assert event["agent_id"] == agent.agent_id
        assert event["run_id"] == bundle["run_id"]
        assert event["turn_id"] == bundle["turn_id"]
        assert event["root_trace_id"] == bundle["root_trace_id"]
        assert event.get("span_id")
    model_results = [row for row in events if row["trace_kind"] == "model_result"]
    assert all(row["provider"] == "test" for row in model_results)
    assert all(row["model"] == "test-model" for row in model_results)
    assert [row["input_tokens"] for row in model_results] == [10, 20]
    tool_result = next(row for row in events if row["trace_kind"] == "tool_result")
    assert tool_result["success"] is True
    assert tool_result["duration_ms"] >= 0


@pytest.mark.unit
def test_context_and_cache_provenance_are_traceable_without_raw_request_content(
    fs_provider,
):
    secret = "private page text and token sk-live-do-not-persist"

    def lookup(item: str):
        return {"ok": True, "item": item}

    agent = MemAgent(
        name="context-provenance-agent",
        application_id="openspeech",
        model=_DeterministicToolModel(),
        tools=[lookup],
        memory_provider=fs_provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        context_policy={"progressive_tool_disclosure": False},
    )
    agent._schedule_conversation_embedding_backfill = lambda _rows: None

    assert (
        agent.run(
            "use the current page",
            memory_id="memory-1",
            thread_id="analysis-a-1",
            user_id="u1",
            context={
                "current_page": {"type": "analysis", "id": "a-1"},
                "current_page_content": {"excerpts": [secret]},
            },
            observability_context={
                "request_id": "request-1",
                "client_page_type": "analysis",
                "client_page_id": "a-1",
                "canonical_page_type": "analysis",
                "canonical_page_id": "a-1",
                "thread_binding_status": "matched",
                "expected_thread_id": "analysis-a-1",
                "ownership_verified": True,
                "grounding_status": "ready",
                "grounding_source": "stored_text",
                "grounding_excerpt_count": 2,
                "raw_prompt": secret,
            },
        )
        == "completed"
    )

    bundles = []
    for row in fs_provider.list_all(MemoryType.SHARED_MEMORY):
        try:
            payload = json.loads(row.get("content") or "")
        except (TypeError, ValueError):
            continue
        if payload.get("type") == "trace_bundle":
            bundles.append(payload)
    assert len(bundles) == 1
    serialized = json.dumps(bundles[0])
    assert secret not in serialized
    assert "raw_prompt" not in serialized

    context_event = next(
        event
        for event in bundles[0]["events"]
        if event["trace_kind"] == "context_provenance"
    )
    assert context_event["thread_binding_status"] == "matched"
    assert context_event["grounding_status"] == "ready"
    assert context_event["request_context_fingerprint"]
    assert any(
        event["trace_kind"] == "cache_decision"
        and event["cache_decision"] == "disabled"
        for event in bundles[0]["events"]
    )


@pytest.mark.unit
def test_recommendation_reviews_create_versioned_evalground_experiments(fs_provider):
    store = ObservabilityStore(fs_provider)
    report = analyze_trace_events(
        [
            {
                "kind": "tool_result",
                "title": "Tool Result: ingest",
                "content": '{"ok":false}',
                "success": False,
                "thread_id": "t1",
            }
        ],
        agent_id="agent-1",
    )
    recommendations = store.sync_recommendations(
        report,
        evidence_refs=[{"root_trace_id": "trace-1", "turn_id": "turn-1"}],
    )
    recommendation = next(
        row for row in recommendations if row["insight"]["id"] == "tool_failure:ingest"
    )
    first_review = store.review_recommendation(
        recommendation["recommendation_id"],
        decision="accepted",
        reviewer_id="operator",
        baseline_config={"instruction": "v1"},
    )
    assert first_review["experiment"]["experiment_version"] == 1
    assert first_review["experiment"]["status"] == "draft"

    changed_report = dict(report)
    changed_report["insights"] = [dict(row) for row in report["insights"]]
    changed_insight = next(
        row for row in changed_report["insights"] if row["id"] == "tool_failure:ingest"
    )
    changed_insight["recommendation"] += " Validate a typed failure contract."
    updated = store.sync_recommendations(changed_report)
    updated_recommendation = next(
        row for row in updated if row["insight"]["id"] == "tool_failure:ingest"
    )
    assert updated_recommendation["revision"] == 2
    second_review = store.review_recommendation(
        updated_recommendation["recommendation_id"],
        decision="accepted",
        reviewer_id="operator",
        baseline_config={"instruction": "v1"},
    )
    assert second_review["experiment"]["experiment_version"] == 2
    assert len(store.list_experiments(agent_id="agent-1")) == 2
    assert len(second_review["recommendation"]["review_history"]) == 2


@pytest.mark.unit
def test_verified_feedback_and_outcomes_join_trace_analysis_without_raw_comment(
    fs_provider,
):
    store = ObservabilityStore(fs_provider)
    context = {
        "agent_id": "agent-1",
        "root_trace_id": "trace-1",
        "run_id": "run-1",
        "turn_id": "turn-1",
        "thread_id": "thread-1",
    }
    secret_comment = "wrong answer for private-user@example.com"
    store.record_feedback(
        trace_context=context,
        rating=-1,
        verified=True,
        label="thumbs_down",
        comment=secret_comment,
    )
    store.record_outcome(
        trace_context=context,
        status="failure",
        verified=True,
        score=0,
    )
    signals = store.list_signals(root_trace_ids=["trace-1"], agent_id="agent-1")
    report = analyze_trace_events([], agent_id="agent-1", signals=signals)

    assert report["summary"]["verified_negative_feedback"] == 1
    assert report["summary"]["verified_outcome_failures"] == 1
    assert {row["id"] for row in report["insights"]}.issuperset(
        {"verified_negative_feedback", "verified_task_failure"}
    )
    assert secret_comment not in json.dumps(
        fs_provider.list_all(MemoryType.SHARED_MEMORY)
    )


@pytest.mark.unit
def test_read_only_proxy_blocks_mutations_and_redaction_masks_secrets(fs_provider):
    read_only = ReadOnlyProviderProxy(fs_provider)
    assert read_only.list_all(MemoryType.SHARED_MEMORY) == []
    with pytest.raises(PermissionError):
        read_only.store({"content": "x"}, MemoryType.SHARED_MEMORY)

    events = redact_trace_events(
        [
            {
                "user_id": "private-user@example.com",
                "content": (
                    '{"authorization":"Bearer very-secret-token",'
                    '"url":"mongodb://operator:password@host/db"}'
                ),
            }
        ],
        mode="redacted",
    )
    serialized = json.dumps(events)
    assert "private-user@example.com" not in serialized
    assert "very-secret-token" not in serialized
    assert "operator:password" not in serialized
