"""Pins for the helpers that ``run``/``run_stream`` and the tool loops share.

Each test checks a helper against the inline logic it replaced, so an edit
that drifts from the original behaviour fails here rather than in a
production turn: the single token estimator, the per-turn cache lookup and
finish, the once-per-turn cache metadata, the stored persona length, the
tool-loop bookkeeping, and the dead code that was removed.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.completion import CompletionDecision, CompletionRejectedError
from memorizz.llms.streaming import ProviderStreamError
from memorizz.memagent.core import _PROMPT_BUFFER_TOKENS, _PROMPT_WINDOW_RATIO, MemAgent
from memorizz.memagent.managers.cache_manager import CacheManager
from memorizz.memagent.utils.prompt_budget import (
    MESSAGE_OVERHEAD_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    estimate_message_tokens,
    estimate_tokens,
    fit_prompt,
)

# Scope caplog to the core logger: other suites leave the ``memorizz`` logger
# at WARNING, so raising the root level alone would not deliver INFO records.
CORE_LOGGER = "memorizz.memagent.core"

# ---------------------------------------------------------------------------
# 1. One token estimator (prompt_budget) for history, compaction and fitting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_estimate_message_tokens_prices_content_plus_framing():
    assert MESSAGE_OVERHEAD_TOKENS == 6
    assert REQUEST_OVERHEAD_TOKENS == 16
    assert estimate_message_tokens({"role": "user", "content": "héllo wörld"}) == (
        estimate_tokens("héllo wörld") + 6
    )
    assert estimate_message_tokens({"role": "assistant"}) == 6
    assert estimate_message_tokens({"content": None}) == 6
    assert estimate_message_tokens("plain text") == estimate_tokens("plain text") + 6


@pytest.mark.unit
def test_fit_prompt_budget_arithmetic_is_unchanged():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "first question " * 4},
        {"role": "assistant", "content": "first answer " * 4},
        {"role": "user", "content": "second question"},
    ]
    total = (
        sum(estimate_tokens(m) + MESSAGE_OVERHEAD_TOKENS for m in messages)
        + REQUEST_OVERHEAD_TOKENS
    )
    fits = next(w for w in range(total, 4 * total) if int(w * 0.8) == total)
    assert fit_prompt(messages, None, fits) == messages
    short = next(w for w in range(fits, 0, -1) if int(w * 0.8) == total - 1)
    assert fit_prompt(messages, None, short) == [messages[0], messages[3]]


@pytest.mark.unit
def test_prepare_history_messages_uses_the_shared_estimator(memagent_with_mocks):
    """The history window is priced with ``estimate_message_tokens`` and is
    evicted from the oldest end in chunks; the newest message always stays."""
    agent = memagent_with_mocks
    agent._context_window_tokens = 1024
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"turn-{i}-" + "x" * 220,
        }
        for i in range(18)
    ]

    selected = agent._prepare_history_messages(history, "system prompt", "query")

    budget = int(1024 * _PROMPT_WINDOW_RATIO) - (
        estimate_tokens("system prompt")
        + estimate_tokens("query")
        + _PROMPT_BUFFER_TOKENS
    )
    consumed, start = 0, len(history)
    for index in range(len(history) - 1, -1, -1):
        cost = estimate_message_tokens(history[index])
        if consumed + cost > budget:
            break
        consumed += cost
        start = index
        if consumed >= budget:
            break
    start = max(start, len(history) - agent._get_conversation_history_limit())
    expected = history[agent._quantize_history_start(start, history) :]
    assert selected == expected
    assert 0 < len(selected) < len(history)
    assert selected[-1] == history[-1]


@pytest.mark.unit
def test_compaction_estimate_matches_the_shared_estimator(memagent_with_mocks):
    agent = memagent_with_mocks
    history = [
        {"role": "user", "content": "first question"},
        {"content": {"role": "assistant", "content": "nested answer"}},
        {"role": "tool", "content": "tool rows never count"},
    ]
    messages = agent._compaction_messages(history)
    assert [m["content"] for m in messages] == ["first question", "nested answer"]
    assert agent._compaction_estimate(history, "system", "q") == (
        estimate_tokens("system")
        + estimate_tokens("q")
        + sum(estimate_message_tokens(m) for m in messages)
        + agent._estimated_tool_tokens()
    )


@pytest.mark.unit
def test_chars_based_estimator_is_gone():
    assert not hasattr(MemAgent, "_estimate_text_tokens")


# ---------------------------------------------------------------------------
# 2. run() / run_stream() share the per-turn cache lookup and finish
# ---------------------------------------------------------------------------


def _capture_cache_traces(agent):
    traces = []
    agent._emit_cache_decision_trace = lambda decision, metadata, bypass_reason=None: (
        traces.append((decision, bypass_reason))
    )
    return traces


@pytest.mark.unit
def test_begin_turn_cache_lookup_resets_turn_state_and_traces_disabled(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    agent._turn_had_side_effects = True
    agent._turn_had_nondeterministic_tools = True
    agent._turn_cache_domains = {"stale"}
    agent._cache_bypass_reason = "stale"
    agent._interaction_error_code = "stale"
    agent._last_completion_decisions = ["stale"]
    traces = _capture_cache_traces(agent)
    agent.cache_manager.enabled = False

    metadata, cached = agent._begin_turn_cache_lookup(
        "hello", "thread-1", None, {"cache_domain": "billing"}
    )

    assert cached is None
    assert metadata == agent._semantic_cache_metadata({"cache_domain": "billing"})
    assert metadata["domains"] == ["billing"]
    assert traces == [("disabled", None)]
    assert agent._turn_had_side_effects is False
    assert agent._turn_had_nondeterministic_tools is False
    assert agent._turn_cache_domains == set()
    assert agent._cache_bypass_reason is None
    assert agent._interaction_error_code is None
    assert agent._last_completion_decisions == []


@pytest.mark.unit
def test_begin_turn_cache_lookup_preflight_bypass_records_the_miss(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value=None)
    agent._semantic_cache_preflight_bypass = lambda query, user_id=None: "no_cache"
    agent.learning_control_plane = Mock()
    traces = _capture_cache_traces(agent)

    metadata, cached = agent._begin_turn_cache_lookup("q", "thread-1", "alice", None)

    assert cached is None
    # A bypassed lookup is traced once; no hit/miss trace follows it.
    assert traces == [("bypassed", "no_cache")]
    record = agent.learning_control_plane.record_cache
    record.assert_called_once()
    assert record.call_args.kwargs["hit"] is False
    assert record.call_args.kwargs["reason"] == "no_cache"
    lookup = agent.cache_manager.get_cached_response.call_args.kwargs
    assert lookup["metadata"] == metadata
    assert lookup["bypass_reason"] == "no_cache"
    assert lookup["user_id"] == "alice"


@pytest.mark.unit
def test_begin_turn_cache_lookup_revalidates_hits(memagent_with_mocks, caplog):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value="stale answer")
    traces = _capture_cache_traces(agent)
    agent._evaluate_completion_candidate = Mock(
        return_value=SimpleNamespace(accepted=False, code="missing_citation")
    )

    with caplog.at_level(logging.INFO, logger=CORE_LOGGER):
        _, cached = agent._begin_turn_cache_lookup(
            "q", "thread-1", None, None, stream=True
        )
    assert cached is None
    assert traces == [("hit", None), ("rejected", "missing_citation")]
    assert (
        "Cached stream response rejected by completion policy (missing_citation); "
        "continuing with a fresh model turn"
    ) in caplog.text
    candidate = agent._evaluate_completion_candidate.call_args.kwargs
    assert candidate == {
        "query": "q",
        "response": "stale answer",
        "iteration": 0,
        "tool_call_count": 0,
    }

    caplog.clear()
    traces.clear()
    agent._evaluate_completion_candidate = Mock(
        return_value=SimpleNamespace(accepted=True, code="ok")
    )
    with caplog.at_level(logging.INFO, logger=CORE_LOGGER):
        metadata, cached = agent._begin_turn_cache_lookup("q", "thread-1", None, None)
    assert cached == "stale answer"
    assert traces == [("hit", None)]
    assert "rejected by completion policy" not in caplog.text
    assert agent.cache_manager.get_cached_response.call_args.kwargs["metadata"] == (
        metadata
    )


@pytest.mark.unit
def test_run_and_run_stream_share_the_cache_hit_bookkeeping(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value="cached reply")
    agent.cache_manager.cache_response = Mock()
    agent.model.generate_stream = lambda *_a, **_k: iter([])
    agent.learning_control_plane = Mock()
    recorded = []
    agent._record_interaction = (
        lambda query, response, memory_id, thread_id, user_id=None: recorded.append(
            (query, response, user_id)
        )
    )

    assert agent.run("hello", user_id="alice") == "cached reply"
    assert list(agent.run_stream("hello again", user_id="alice")) == ["cached reply"]

    assert recorded == [
        ("hello", "cached reply", "alice"),
        ("hello again", "cached reply", "alice"),
    ]
    hits = agent.learning_control_plane.record_cache.call_args_list
    assert [c.kwargs["hit"] for c in hits] == [True, True]
    assert {c.kwargs["reason"] for c in hits} == {"exact_or_semantic_cache_hit"}
    agent.model.generate.assert_not_called()
    agent.cache_manager.cache_response.assert_not_called()


@pytest.mark.unit
def test_turn_failure_code_logs_the_turn_kind(memagent_with_mocks, caplog):
    agent = memagent_with_mocks
    agent._interaction_error_code = None
    assert agent._turn_failure_code() is None

    agent._interaction_error_code = "iteration_limit"
    with caplog.at_level(logging.WARNING, logger=CORE_LOGGER):
        assert agent._turn_failure_code() == "iteration_limit"
        assert agent._turn_failure_code(stream=True) == "iteration_limit"
    assert (
        f"MemAgent {agent.agent_id} turn failed (iteration_limit); "
        "reply not cached or recorded"
    ) in caplog.text
    assert (
        f"MemAgent {agent.agent_id} stream failed (iteration_limit); "
        "reply not cached or recorded"
    ) in caplog.text


@pytest.mark.unit
def test_run_neither_caches_nor_records_a_failure_reply(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.cache_response = Mock()
    agent._record_interaction = Mock()

    def failing(*_args, **_kwargs):
        agent._mark_interaction_error("provider_error")
        return "I encountered an error while processing your request"

    agent._execute_llm_interaction = failing

    assert agent.run("hello").startswith("I encountered an error")
    agent.cache_manager.cache_response.assert_not_called()
    agent._record_interaction.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Cache metadata computed once per turn; persona rendered once per turn
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_with_turn_cache_domains_matches_recomputing_metadata(memagent_with_mocks):
    """Re-merging the turn's tool domains into the lookup metadata gives the
    same write metadata the second ``_semantic_cache_metadata`` call did."""
    agent = memagent_with_mocks
    context = {
        "cache_domains": ["billing"],
        "cache_tags": ["b", "a"],
        "cache_data_version": "v3",
    }
    agent._turn_cache_domains = set()
    before = agent._semantic_cache_metadata(context)
    assert before["domain"] == "billing" and before["domains"] == ["billing"]

    agent._turn_cache_domains = {"inventory", "billing"}
    merged = agent._with_turn_cache_domains(before)
    assert merged == agent._semantic_cache_metadata(context)
    assert merged["domains"] == ["billing", "inventory"] and merged["domain"] is None
    assert merged["tags"] == ["a", "b"]
    # The lookup metadata itself is not mutated.
    assert before["domain"] == "billing" and before["domains"] == ["billing"]

    agent._turn_cache_domains = {"inventory"}
    assert agent._with_turn_cache_domains(agent._semantic_cache_metadata({})) == {
        **agent._semantic_cache_metadata({}),
        "domain": "inventory",
        "domains": ["inventory"],
    }


@pytest.mark.unit
def test_finish_turn_writes_the_cache_with_turn_domains_and_records(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.cache_response = Mock()
    agent._emit_memory_reference_trace = Mock()
    recorded = []
    agent._record_interaction = lambda *args, **kwargs: recorded.append((args, kwargs))
    metadata, _ = agent._begin_turn_cache_lookup(
        "q", "thread-1", "alice", {"cache_domain": "billing"}
    )
    agent._turn_cache_domains.add("inventory")
    agent._turn_had_side_effects = True
    agent._turn_had_nondeterministic_tools = True
    agent._cache_bypass_reason = "side_effecting_tool"
    session = SimpleNamespace(persistence={})

    agent._finish_turn(
        "q", "answer", "memory-1", "thread-1", "alice", metadata, session=session
    )

    agent._emit_memory_reference_trace.assert_called_once_with("answer")
    write = agent.cache_manager.cache_response
    write.assert_called_once()
    assert write.call_args.args == ("q", "answer", "thread-1")
    kwargs = write.call_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["metadata"]["domains"] == ["billing", "inventory"]
    assert kwargs["metadata"]["fingerprints"] == metadata["fingerprints"]
    assert kwargs["deterministic"] is False
    assert kwargs["read_only"] is False
    assert kwargs["bypass_reason"] == "side_effecting_tool"
    assert session.persistence == {"cache": "unknown"}
    assert recorded == [(("q", "answer", "memory-1", "thread-1"), {"user_id": "alice"})]

    # A disabled cache writes nothing and sets no persistence flag; the
    # exchange is still recorded.
    agent.cache_manager.enabled = False
    write.reset_mock()
    session = SimpleNamespace(persistence={})
    agent._finish_turn(
        "q", "answer", "memory-1", "thread-1", "alice", metadata, session=session
    )
    write.assert_not_called()
    assert session.persistence == {}
    assert len(recorded) == 2


@pytest.mark.unit
def test_run_merges_tool_domains_into_the_cache_write(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value=None)
    agent.cache_manager.cache_response = Mock()
    agent._record_interaction = Mock()

    def answer(*_args, **_kwargs):
        # A tool declared its cache domain while the turn ran.
        agent._turn_cache_domains.add("inventory")
        return "42 units"

    agent._execute_llm_interaction = answer

    assert agent.run("stock?", context={"cache_domain": "billing"}) == "42 units"

    lookup = agent.cache_manager.get_cached_response.call_args.kwargs["metadata"]
    write = agent.cache_manager.cache_response.call_args.kwargs["metadata"]
    assert lookup["domains"] == ["billing"]
    assert write["domains"] == ["billing", "inventory"]
    assert write["fingerprints"] == lookup["fingerprints"]
    agent._record_interaction.assert_called_once()


@pytest.mark.unit
def test_memory_context_trace_reads_the_rendered_persona_length(memagent_with_mocks):
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(events.append)
    snapshot = {"persona_id": "account", "version": 3, "goals": "Use examples."}
    with agent.persona_manager.use_snapshot(snapshot):
        agent._begin_trace_turn("account")
        agent._build_system_prompt()
        rendered = len(
            agent.persona_manager.get_persona_prompt(
                include_history=True, history_limit=5
            )
        )
        assert agent._rendered_persona_chars == rendered > 0
        agent._emit_memory_context_trace({})
    supply = next(e for e in events if e.get("trace_kind") == "memory_context")
    assert supply["memory_injected_chars"] == rendered
    assert json.loads(supply["content"])["injected_char_count"] == rendered

    # Without a persona reference the stored length is not counted.
    with agent.persona_manager.use_snapshot(None):
        agent._emit_memory_context_trace({})
    assert events[-1]["memory_injected_chars"] == 0


# ---------------------------------------------------------------------------
# 3. Tool loops: shared turn setup, tool-call turn and rejection bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prepare_turn_request_builds_the_shared_turn_inputs(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.model.set_prompt_cache_key = Mock()

    workflow, messages, tools = agent._prepare_turn_request(
        "system", "query", {"conversation_history": []}, "alice", None
    )

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user" and "query" in messages[-1]["content"]
    assert tools and all(t["type"] == "function" for t in tools)
    agent.model.set_prompt_cache_key.assert_called_once()
    assert workflow is None or hasattr(workflow, "steps")


@pytest.mark.unit
def test_prepare_turn_request_refuses_streamed_tools_before_pinning_scope(
    memagent_with_mocks, monkeypatch
):
    agent = memagent_with_mocks
    agent.model.set_prompt_cache_key = Mock()
    monkeypatch.setattr(
        "memorizz.llms.streaming.streaming_capabilities",
        lambda provider: {"tool_calls": False, "text_deltas": True},
    )

    with pytest.raises(ProviderStreamError) as excinfo:
        agent._prepare_turn_request(
            "system", "query", {}, None, None, streaming_tools=True
        )
    assert excinfo.value.code == "provider_tools_unsupported"
    agent.model.set_prompt_cache_key.assert_not_called()

    monkeypatch.setattr(
        "memorizz.llms.streaming.streaming_capabilities",
        lambda provider: {"tool_calls": True, "text_deltas": True},
    )
    _, _, tools = agent._prepare_turn_request(
        "system", "query", {}, None, None, streaming_tools=True
    )
    assert isinstance(tools, list) and tools
    agent.model.set_prompt_cache_key.assert_called_once()


@pytest.mark.unit
def test_run_tool_turn_appends_the_assistant_turn_and_counts_calls(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    executed = []
    agent._execute_turn_tool_calls = lambda calls, messages, workflow, user_id, **kw: (
        executed.append((list(calls), workflow, user_id, kw))
    )
    call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="echo", arguments="{}")
    )
    message = SimpleNamespace(content=None, tool_calls=[call, call])
    messages = [{"role": "user", "content": "q"}]

    count = agent._run_tool_turn(
        message, messages, "wf", "alice", streaming=True, query="q"
    )

    assert count == 2
    assert messages[-1]["role"] == "assistant"
    assert [tc["id"] for tc in messages[-1]["tool_calls"]] == ["call-1", "call-1"]
    assert executed == [
        ([call, call], "wf", "alice", {"streaming": True, "query": "q"})
    ]


@pytest.mark.unit
def test_note_completion_rejection_counts_appends_and_raises(memagent_with_mocks):
    agent = memagent_with_mocks
    limit = agent.completion_policy.max_rejections
    decision = CompletionDecision(accepted=False, code="needs_evidence", reason="cite")
    messages = []
    rejections = 0
    for expected in range(1, limit + 1):
        rejections = agent._note_completion_rejection(
            messages, f"draft {expected}", decision, rejections
        )
        assert rejections == expected
        assert messages[-2] == {"role": "assistant", "content": f"draft {expected}"}
        assert messages[-1] == {
            "role": "developer",
            "content": agent.completion_policy.retry_message(decision),
        }
    with pytest.raises(CompletionRejectedError):
        agent._note_completion_rejection(messages, "draft", decision, rejections)
    assert len(messages) == 2 * limit


@pytest.mark.unit
def test_direct_run_stream_still_takes_the_legacy_loop(memagent_with_mocks):
    """``run_stream()`` called outside a stream session is deprecated but
    still served by the legacy loop; removing it is a deliberate decision."""
    agent = memagent_with_mocks

    def stream(messages, tools=None, **_):
        yield {"type": "content", "content": "the answer"}
        yield {"type": "done", "content": "the answer"}

    agent.model.generate_stream = stream
    seen = []
    legacy = agent._execute_llm_interaction_stream_legacy

    def spy(*args, **kwargs):
        seen.append(True)
        return legacy(*args, **kwargs)

    agent._execute_llm_interaction_stream_legacy = spy
    with pytest.warns(DeprecationWarning):
        assert "".join(agent.run_stream("hello")) == "the answer"
    assert seen == [True]


# ---------------------------------------------------------------------------
# Dead code removed: cache bypass contract, conversation unit field
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_lookup_returns_none_under_a_bypass_even_on_an_exact_hit():
    """``delegate()`` relies on the cache honouring ``bypass_reason`` itself;
    the former ``if cache_bypass_reason: cached_response = None`` guard is
    gone."""
    from memorizz.short_term_memory.semantic_cache import (
        SemanticCache,
        SemanticCacheConfig,
    )

    class _Embeddings:
        def get_embedding(self, text):
            return [1.0, 0.0]

    cache = SemanticCache(
        config=SemanticCacheConfig(
            similarity_threshold=0.9, enable_memory_provider_sync=False
        ),
        embedding_manager=_Embeddings(),
        agent_id="agent-1",
        memory_id="memory-1",
    )
    metadata = {
        "domain": "inventory",
        "tags": [],
        "fingerprints": {
            "model": "m",
            "prompt": "p",
            "tool_schema": "t",
            "data_version": "v1",
        },
        "admission": {"read_only": True, "deterministic": True},
    }
    assert cache.set(
        "inventory for SKU-7", "12 units", user_id="alice", metadata=metadata
    )
    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=metadata)
        == "12 units"
    )
    assert (
        cache.get(
            "inventory for SKU-7",
            user_id="alice",
            lookup_metadata=metadata,
            bypass_reason="runtime_delegation_contract",
        )
        is None
    )
    assert cache.statistics()["bypass_reasons"]["runtime_delegation_contract"] == 1

    manager = CacheManager(enabled=True)
    manager.cache_instance = Mock(get=Mock(return_value=None))
    manager.get_cached_response("q", "s", bypass_reason="invalid_explicit_plan")
    assert manager.cache_instance.get.call_args.kwargs["bypass_reason"] == (
        "invalid_explicit_plan"
    )


@pytest.mark.unit
def test_conversation_memory_unit_has_no_recall_recency_but_loads_old_rows():
    from memorizz.long_term.episodic.conversational_memory_unit import (
        ConversationMemoryUnit,
    )

    assert "recall_recency" not in ConversationMemoryUnit.model_fields
    unit = ConversationMemoryUnit(
        role="user",
        content="hi",
        timestamp="2026-10-08T00:00:00",
        memory_id="memory-1",
        thread_id="thread-1",
        recall_recency=0.5,  # a row persisted before the field was dropped
    )
    assert "recall_recency" not in unit.model_dump()
