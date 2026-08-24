# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tests for the cache-friendly context assembly and pre-inference dedup.

Covers the invariants introduced by the context-efficiency work:

1. The system prompt is byte-stable across turns (prompt-cache prefix).
2. The history window start moves only at eviction-chunk boundaries.
3. Retrieved memories are deduplicated (exact, vs-history, near-dup) and
   rendered in the volatile tail, never the system prompt.
4. The Anthropic provider attaches cache_control breakpoints; the OpenAI
   provider forwards prompt_cache_key and surfaces cached_tokens.
5. Duplicate (query, response) pairs are not double-written.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.long_term.procedural.skillbox import SkillInjectionRole
from memorizz.memagent import MemAgent
from memorizz.memagent.managers.continual_learning_manager import (
    ContinualLearningManager,
)
from memorizz.memagent.utils.context_dedup import (
    candidate_text,
    content_fingerprint,
    cosine_similarity,
    dedupe_and_select,
    filter_skill_covered_workflows,
    normalize_text,
)

# ---------------------------------------------------------------------------
# context_dedup unit behavior
# ---------------------------------------------------------------------------


def test_normalize_and_fingerprint_ignore_cosmetic_diffs():
    assert normalize_text("  Hello   WORLD \n") == "hello world"
    assert content_fingerprint("Hello  world") == content_fingerprint("hello world")
    assert content_fingerprint("hello world") != content_fingerprint("hello mars")


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) is None
    assert cosine_similarity([1.0], [1.0, 0.0]) is None


def test_candidate_text_handles_provider_row_shapes():
    assert candidate_text({"content": "plain"}) == "plain"
    assert (
        candidate_text({"content": {"role": "user", "content": "nested"}}) == "nested"
    )
    assert candidate_text({"text": "pre-shaped"}) == "pre-shaped"
    assert candidate_text(None) == ""


def test_exact_duplicates_collapse_across_sources():
    rows = [
        ("episodic", {"content": "The user prefers Python.", "_id": "a"}),
        ("knowledge_base", {"content": "The user prefers  python.", "_id": "b"}),
    ]
    out = dedupe_and_select(rows)
    assert len(out) == 1


def test_exact_duplicate_from_stronger_query_variant_keeps_higher_score():
    rows = [
        (
            "knowledge_base",
            {"content": "Same memory", "source_id": "weak", "score": 0.2},
        ),
        (
            "knowledge_base",
            {"content": "Same memory", "source_id": "strong", "score": 0.9},
        ),
    ]
    out = dedupe_and_select(rows)
    assert out[0]["id"] == "strong"


def test_candidates_already_in_history_are_dropped():
    rows = [
        ("episodic", {"content": "Budget is $500", "_id": "a"}),
        ("episodic", {"content": "Fresh fact never mentioned", "_id": "b"}),
    ]
    out = dedupe_and_select(
        rows, history_texts=["We agreed the budget is $500 for this project."]
    )
    texts = [item["text"] for item in out]
    assert "Fresh fact never mentioned" in texts
    # "Budget is $500" is contained in a history message → dropped.
    assert all("Budget" not in t for t in texts)


def test_near_duplicates_dropped_by_embedding_similarity():
    rows = [
        (
            "episodic",
            {
                "content": "likes hiking",
                "_id": "a",
                "embedding": [1.0, 0.0],
                "score": 0.9,
            },
        ),
        (
            "knowledge_base",
            {
                "content": "enjoys hiking trips",
                "_id": "b",
                "embedding": [0.999, 0.001],
                "score": 0.8,
            },
        ),
        (
            "knowledge_base",
            {
                "content": "allergic to peanuts",
                "_id": "c",
                "embedding": [0.0, 1.0],
                "score": 0.7,
            },
        ),
    ]
    out = dedupe_and_select(rows)
    texts = [item["text"] for item in out]
    # Near-dup pair collapses to the higher-scored copy.
    assert "likes hiking" in texts
    assert "enjoys hiking trips" not in texts
    assert "allergic to peanuts" in texts


def test_parent_source_dedup_is_explicit_and_keeps_best_ranked_copy():
    rows = [
        (
            "knowledge_base",
            {
                "content": "Original statement",
                "source_id": "turn-1",
                "parent_source_id": "turn-1",
                "score": 0.6,
            },
        ),
        (
            "knowledge_base",
            {
                "content": "Semantic constraint derived from the statement",
                "source_id": "turn-1#semantic-1",
                "parent_source_id": "turn-1",
                "score": 0.9,
            },
        ),
    ]
    assert len(dedupe_and_select(rows)) == 2
    selected = dedupe_and_select(rows, dedupe_parent_sources=True)
    assert len(selected) == 1
    assert selected[0]["text"].startswith("Semantic constraint")


def test_parent_source_dedup_prefers_multi_source_provenance():
    rows = [
        (
            "knowledge_base",
            {
                "content": "One statement",
                "source_id": "cue:1#semantic",
                "parent_source_id": "cue:1",
                "score": 0.95,
            },
        ),
        (
            "knowledge_base",
            {
                "content": "Statement plus related advice",
                "source_id": "cue-group#semantic",
                "parent_source_id": "cue:1",
                "linked_source_ids": ["cue:1", "cue:2"],
                "score": 0.90,
            },
        ),
    ]
    selected = dedupe_and_select(rows, dedupe_parent_sources=True)
    assert selected[0]["linked_source_ids"] == ["cue:1", "cue:2"]


def test_selection_respects_max_items_and_orders_chronologically():
    rows = [
        ("episodic", {"content": f"fact {i}", "_id": f"id{i}", "timestamp": 1000 + i})
        for i in range(10)
    ]
    out = dedupe_and_select(rows, max_items=3)
    assert len(out) == 3
    stamps = [item["timestamp"] for item in out]
    assert stamps == sorted(stamps)


def test_output_order_is_deterministic_across_calls():
    rows = [
        ("episodic", {"content": "alpha", "_id": "1", "timestamp": 5}),
        ("knowledge_base", {"content": "beta", "_id": "2", "timestamp": 3}),
        ("episodic", {"content": "gamma", "_id": "3", "timestamp": 4}),
    ]
    first = dedupe_and_select(list(rows))
    second = dedupe_and_select(list(rows))
    assert first == second


def test_dedupe_preserves_workflow_suppression_provenance():
    out = dedupe_and_select(
        [
            (
                "workflow_memory",
                {
                    "content": "raw refund run",
                    "_id": "record-1",
                    "workflow_id": "workflow-1",
                    "canonical_hash": "hash-1",
                    "promoted_skill_id": "skill-1",
                    "memory_type": "workflow_memory",
                },
            )
        ]
    )

    assert out[0]["workflow_id"] == "workflow-1"
    assert out[0]["canonical_hash"] == "hash-1"
    assert out[0]["promoted_skill_id"] == "skill-1"
    assert out[0]["memory_type"] == "workflow_memory"


def test_skill_covered_workflows_are_removed_at_context_boundary():
    skill = SimpleNamespace(
        skill_id="skill-1",
        source_canonical_hash="hash-1",
        source_workflow_ids=["workflow-1"],
        exemplar_workflow_id="workflow-exemplar",
    )
    scored = SimpleNamespace(skill=skill, similarity=0.99)
    memories = [
        {
            "source": "workflow_memory",
            "text": "matched by promotion stamp",
            "promoted_skill_id": "skill-1",
        },
        {
            "source": "workflow",
            "text": "matched by canonical hash",
            "canonical_hash": "hash-1",
        },
        {
            "source": "procedural_workflow",
            "text": "matched by source workflow id",
            "workflow_id": "workflow-1",
        },
        {
            "text": "unlabelled direct provider workflow",
            "workflow_id": "workflow-exemplar",
        },
        {
            "source": "workflow_memory",
            "text": "cannot prove this raw run is distinct",
        },
        {
            "source": "workflow_memory",
            "text": "known unrelated workflow",
            "canonical_hash": "hash-other",
            "workflow_id": "workflow-other",
        },
        {
            "source": "episodic",
            "text": "ordinary memory remains",
            "canonical_hash": "hash-1",
        },
    ]

    filtered = filter_skill_covered_workflows(memories, [scored])
    assert [item["text"] for item in filtered] == [
        "known unrelated workflow",
        "ordinary memory remains",
    ]
    assert filter_skill_covered_workflows(memories, []) == memories


# ---------------------------------------------------------------------------
# Prompt assembly invariants
# ---------------------------------------------------------------------------


def test_programmatic_agent_config_normalizes_and_validates_skill_authority():
    agent = MemAgent(
        model=MagicMock(),
        continual_learning=False,
        continual_learning_config={
            "require_shadow": True,
            "skill_injection_role": SkillInjectionRole.DEVELOPER,
        },
    )
    assert agent.continual_learning_config["skill_injection_role"] == "developer"

    with pytest.raises(ValueError, match="require_shadow=True"):
        MemAgent(
            model=MagicMock(),
            continual_learning=False,
            continual_learning_config={
                "require_shadow": False,
                "skill_injection_role": "developer",
            },
        )


def test_system_prompt_is_byte_stable_across_turns(memagent_with_mocks):
    """The static system prompt must not change turn-to-turn — it is the
    prompt-cache prefix."""
    agent = memagent_with_mocks
    first = agent._build_system_prompt()
    second = agent._build_system_prompt()
    assert first == second
    # Volatile markers must not appear in the static prompt.
    assert "Recent tool outputs" not in first
    assert "Entity memory facts:" not in first


def test_system_prompt_does_not_duplicate_tool_descriptions(memagent_with_mocks):
    """Tool schemas ride in the request's tools parameter; the system prompt
    must not repeat every description (double token cost)."""
    agent = memagent_with_mocks
    prompt = agent._build_system_prompt()
    assert "Available tools (" not in prompt


def test_retrieved_memories_render_in_final_user_message(memagent_with_mocks):
    agent = memagent_with_mocks
    context = {
        "conversation_history": [],
        "retrieved_memories": [
            {
                "source": "episodic",
                "text": "user is vegetarian",
                "id": "x",
                "timestamp": None,
            }
        ],
    }
    messages = agent._build_prompt_messages("SYS", "plan dinner", context)
    assert len(messages) == 2
    assert messages[0]["content"] == "SYS"
    assert "user is vegetarian" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("plan dinner")


def test_prompt_never_renders_skill_with_its_source_workflow(memagent_with_mocks):
    agent = memagent_with_mocks
    skill = SimpleNamespace(
        skill_id="skill-1",
        source_canonical_hash="hash-1",
        source_workflow_ids=["workflow-1"],
        exemplar_workflow_id=None,
    )
    scored = SimpleNamespace(skill=skill, similarity=0.99)
    agent.continual_learning_manager = SimpleNamespace(
        format_skills_prompt_section=lambda _: "LEARNED REFUND SKILL"
    )
    context = {
        "conversation_history": [],
        "activated_skills": [scored],
        "retrieved_memories": [
            {
                "source": "workflow_memory",
                "text": "DUPLICATE RAW WORKFLOW",
                "canonical_hash": "hash-1",
                "workflow_id": "workflow-1",
            },
            {
                "source": "episodic",
                "text": "customer preference",
                "id": "memory-1",
            },
        ],
    }

    prompt = agent._build_prompt_messages("SYS", "refund order", context)[-1]["content"]
    assert "LEARNED REFUND SKILL" in prompt
    assert "customer preference" in prompt
    assert "DUPLICATE RAW WORKFLOW" not in prompt


def test_prompt_separates_user_and_developer_skill_authority(memagent_with_mocks):
    agent = memagent_with_mocks
    developer_skill = SimpleNamespace(
        skill_id="skill-dev",
        injection_role="developer",
        source_canonical_hash="hash-dev",
        source_workflow_ids=["workflow-dev"],
        exemplar_workflow_id=None,
    )
    user_skill = SimpleNamespace(
        skill_id="skill-user",
        injection_role="user",
        source_canonical_hash="hash-user",
        source_workflow_ids=["workflow-user"],
        exemplar_workflow_id=None,
    )
    developer_scored = SimpleNamespace(skill=developer_skill, similarity=0.99)
    user_scored = SimpleNamespace(skill=user_skill, similarity=0.98)
    agent.continual_learning_manager = SimpleNamespace(
        format_skills_prompt_section=lambda skills, injection_role=None: (
            "DEVELOPER SKILL" if injection_role == "developer" else "USER SKILL"
        )
    )

    messages = agent._build_prompt_messages(
        "SYSTEM POLICY",
        "refund order",
        {
            "conversation_history": [],
            "activated_skills": [developer_scored, user_scored],
            "retrieved_memories": [
                {
                    "source": "workflow_memory",
                    "text": "RAW DEV WORKFLOW",
                    "canonical_hash": "hash-dev",
                    "workflow_id": "workflow-dev",
                },
                {
                    "source": "workflow_memory",
                    "text": "RAW USER WORKFLOW",
                    "canonical_hash": "hash-user",
                    "workflow_id": "workflow-user",
                },
                {
                    "source": "episodic",
                    "text": "ordinary memory",
                    "id": "memory-1",
                },
            ],
        },
    )

    assert [message["role"] for message in messages] == [
        "system",
        "developer",
        "user",
    ]
    assert "DEVELOPER SKILL" in messages[1]["content"]
    assert "USER SKILL" not in messages[1]["content"]
    assert "USER SKILL" in messages[2]["content"]
    assert "DEVELOPER SKILL" not in messages[2]["content"]
    assert "ordinary memory" in messages[2]["content"]
    assert "RAW DEV WORKFLOW" not in messages[2]["content"]
    assert "RAW USER WORKFLOW" not in messages[2]["content"]


def test_automatic_context_does_not_retrieve_raw_workflow_memory(
    memagent_with_mocks,
):
    """Workflow memory remains a capture/evidence store, not prompt retrieval."""
    agent = memagent_with_mocks
    agent.active_memory_types = [
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SKILLBOX,
    ]
    agent.continual_learning_manager = SimpleNamespace(
        retrieve_skills_for_query=lambda query, user_id=None: []
    )
    agent.memory_manager.retrieve_relevant_memories = MagicMock(return_value=[])

    agent._build_context("refund order", "memory-1")

    agent.memory_manager.retrieve_relevant_memories.assert_not_called()


def test_legacy_include_exemplar_option_does_not_inject_raw_run():
    manager = ContinualLearningManager.__new__(ContinualLearningManager)
    manager.config = SimpleNamespace(include_exemplar=True)
    manager.memory_provider = MagicMock()
    scored = SimpleNamespace(
        similarity=0.98,
        skill=SimpleNamespace(
            name="Refund flow",
            version=1,
            content="Use the approved refund procedure.",
            exemplar_workflow_id="workflow-1",
        ),
    )

    section = manager.format_skills_prompt_section([scored])
    assert "Use the approved refund procedure." in section
    assert "Reference run:" not in section
    manager.memory_provider.retrieve_by_id.assert_not_called()


def test_tools_are_sorted_deterministically(memagent_with_mocks):
    agent = memagent_with_mocks
    tools = agent._build_llm_tools()
    if not tools:
        pytest.skip("fixture registered no tools")
    names = [t.get("function", {}).get("name", "") for t in tools]
    assert names == sorted(names)


def test_history_window_start_is_chunk_quantized(memagent_with_mocks):
    """The first message of the rendered history window may only change at
    eviction-chunk boundaries — a per-turn sliding window breaks the prompt
    cache every turn."""
    agent = memagent_with_mocks
    agent._context_window_tokens = 4000  # force a tight budget

    def build_history(n):
        return [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"message {i} " + "x" * 200,
            }
            for i in range(n)
        ]

    starts = []
    for total in range(60, 80):
        window = agent._prepare_history_messages(build_history(total), "SYS", "q")
        starts.append(window[0]["content"])

    # Across 20 consecutive turns the window start changes at most a few
    # times (chunked eviction), never every turn.
    changes = sum(1 for a, b in zip(starts, starts[1:]) if a != b)
    assert changes <= 3, f"window start churned {changes} times across 20 turns"


def test_quantize_history_start_boundaries(memagent_with_mocks):
    agent = memagent_with_mocks
    rows = [{"role": "user", "content": str(i)} for i in range(100)]
    assert agent._quantize_history_start(0, rows) == 0
    assert agent._quantize_history_start(1, rows) == 20
    assert agent._quantize_history_start(20, rows) == 20
    assert agent._quantize_history_start(21, rows) == 40
    # Never quantizes away the whole window.
    assert agent._quantize_history_start(99, rows) == 99


# ---------------------------------------------------------------------------
# Write-path dedup
# ---------------------------------------------------------------------------


def test_duplicate_interaction_pairs_are_not_double_written(memagent_with_mocks):
    agent = memagent_with_mocks
    if not agent.memory_manager:
        pytest.skip("no memory manager on fixture")

    agent.model.generate = lambda messages, tools=None, **_: "same answer"

    # Prime the conversation cache slot: the guard inspects the cached tail,
    # which save_memory_unit appends to (the fixture's mock provider can't
    # serve a real load_conversation_history).
    agent.memory_manager._conversation_memory_cache[("m-dedup", None)] = []

    agent.run("same question", memory_id="m-dedup", thread_id="t-1")
    saves = MagicMock(wraps=agent.memory_manager.save_memory_unit)
    agent.memory_manager.save_memory_unit = saves
    agent.run("same question", memory_id="m-dedup", thread_id="t-1")

    # The second identical (query, response) pair must be skipped.
    assert saves.call_count == 0


# ---------------------------------------------------------------------------
# Tool-result context fidelity
# ---------------------------------------------------------------------------


def test_tool_log_placeholder_preserves_scalar_decision_fields():
    from memorizz.memagent.utils.tool_log import _build_tool_log_placeholder

    placeholder = _build_tool_log_placeholder(
        tool_name="lookup_order",
        tool_log_id="log-1",
        arguments={"order_id": "R-1001"},
        result={
            "ok": True,
            "order_id": "R-1001",
            "status": "completed",
            "amount": 49.0,
        },
    )

    assert "ok=True" in placeholder
    assert "order_id=R-1001" in placeholder
    assert "status=completed" in placeholder
    assert "amount=49.0" in placeholder


# ---------------------------------------------------------------------------
# Anthropic provider cache_control
# ---------------------------------------------------------------------------


def _make_anthropic(monkeypatch):
    anthropic_mod = pytest.importorskip("anthropic")
    from memorizz.llms.anthropic import Anthropic as MemorizzAnthropic

    provider = MemorizzAnthropic.__new__(MemorizzAnthropic)
    provider.model = "claude-sonnet-4-5"
    provider.context_window_tokens = 200_000
    provider._last_usage = None
    provider._max_tokens = 512
    provider._enable_prompt_caching = True
    provider._request_options = {}
    provider.client = MagicMock()
    return provider


def test_anthropic_apply_cache_control_marks_system_and_last_message(monkeypatch):
    provider = _make_anthropic(monkeypatch)
    kwargs = {
        "system": "STATIC SYSTEM",
        "messages": [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "reply one"},
            {"role": "user", "content": "turn two"},
        ],
    }
    provider._apply_cache_control(kwargs)

    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    # Last message: within-turn (tool loop) read point.
    last = kwargs["messages"][-1]
    assert isinstance(last["content"], list)
    assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Second-to-last message: cross-turn read point (stable history).
    second_last = kwargs["messages"][-2]
    assert isinstance(second_last["content"], list)
    assert second_last["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Earlier messages untouched.
    assert kwargs["messages"][0]["content"] == "turn one"


def test_anthropic_apply_cache_control_handles_block_content(monkeypatch):
    provider = _make_anthropic(monkeypatch)
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            }
        ]
    }
    provider._apply_cache_control(kwargs)
    block = kwargs["messages"][-1]["content"][-1]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_apply_cache_control_disabled_is_noop(monkeypatch):
    provider = _make_anthropic(monkeypatch)
    provider._enable_prompt_caching = False
    kwargs = {"system": "S", "messages": [{"role": "user", "content": "q"}]}
    provider._apply_cache_control(kwargs)
    assert kwargs["system"] == "S"
    assert kwargs["messages"][0]["content"] == "q"


def test_anthropic_usage_includes_cache_fields(monkeypatch):
    provider = _make_anthropic(monkeypatch)
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=1500,
        cache_creation_input_tokens=50,
    )
    extracted = provider._usage_to_dict(usage)
    # prompt_tokens = uncached + cache_read + cache_creation
    assert extracted["prompt_tokens"] == 1650
    assert extracted["cached_tokens"] == 1500
    assert extracted["cache_creation_input_tokens"] == 50
    assert extracted["total_tokens"] == 1670


# ---------------------------------------------------------------------------
# OpenAI provider cache options
# ---------------------------------------------------------------------------


def _make_openai():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    provider = MemorizzOpenAI.__new__(MemorizzOpenAI)
    provider.model = "gpt-4o"
    provider.base_url = None
    provider.context_window_tokens = 128_000
    provider._last_usage = None
    provider._prompt_cache_key = None
    provider._prompt_cache_retention = None
    provider._request_options = {}
    provider.client = MagicMock()
    return provider


def test_openai_prompt_cache_key_is_attached():
    provider = _make_openai()
    provider.set_prompt_cache_key("memorizz:a:m:t")
    kwargs = {"model": "gpt-4o", "messages": []}
    provider._apply_cache_options(kwargs)
    assert kwargs["prompt_cache_key"] == "memorizz:a:m:t"


def test_openai_cache_options_skipped_for_local_base_url():
    provider = _make_openai()
    provider.base_url = "http://127.0.0.1:8080/v1"
    provider.set_prompt_cache_key("memorizz:a:m:t")
    provider._prompt_cache_retention = "24h"
    kwargs = {"model": "gpt-4o", "messages": []}
    provider._apply_cache_options(kwargs)
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs


def test_openai_prompt_cache_key_hashes_values_over_api_limit():
    provider = _make_openai()

    provider.set_prompt_cache_key("tenant:agent:memory:thread:" + "x" * 100)

    assert len(provider._prompt_cache_key) == 64
    assert provider._prompt_cache_key == (
        "b7724a8ded18f921d68fa4a1ec7bb689d0aa94ad71c30dfd3e30d70502ab0d7b"
    )


def test_openai_usage_extracts_cached_tokens():
    provider = _make_openai()
    usage = SimpleNamespace(
        prompt_tokens=2006,
        completion_tokens=300,
        total_tokens=2306,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1920),
    )
    extracted = provider._extract_usage(SimpleNamespace(usage=usage))
    assert extracted["cached_tokens"] == 1920
    assert extracted["prompt_tokens"] == 2006


def test_openai_old_sdk_without_cache_params_falls_back():
    provider = _make_openai()
    provider.set_prompt_cache_key("key")

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "prompt_cache_key" in kwargs:
            raise TypeError("unexpected keyword argument 'prompt_cache_key'")
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="hi"))
            ],
        )

    provider.client.chat.completions.create = fake_create
    kwargs = {"model": "gpt-4o", "messages": []}
    provider._apply_cache_options(kwargs)
    result = provider._create_chat_completion(kwargs)
    assert result is not None
    assert len(calls) == 2
    assert "prompt_cache_key" not in calls[-1]


def test_openai_reasoning_effort_is_forwarded_and_persisted():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    with patch("memorizz.llms.openai.openai.OpenAI"):
        provider = MemorizzOpenAI(
            api_key="test-key",
            model="gpt-5.6",
            reasoning_effort="none",
        )

    captured = {}

    def fake_create(kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ready", tool_calls=None)
                )
            ],
            usage=None,
        )

    provider._create_chat_completion = fake_create
    assert provider.generate([{"role": "user", "content": "ready?"}]) == "ready"
    assert captured["reasoning_effort"] == "none"
    assert provider.get_config()["reasoning_effort"] == "none"


def test_openai_gpt_5_6_translates_legacy_max_tokens():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    with patch("memorizz.llms.openai.openai.OpenAI"):
        provider = MemorizzOpenAI(
            api_key="test-key",
            model="gpt-5.6-terra",
            max_tokens=32_000,
        )

    assert provider._request_options["max_completion_tokens"] == 32_000
    assert "max_tokens" not in provider._request_options
    assert provider.get_context_window_tokens() == 1_050_000


def test_openai_gpt_5_5_text_generation_uses_responses_options_and_usage():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    with patch("memorizz.llms.openai.openai.OpenAI"):
        provider = MemorizzOpenAI(
            api_key="test-key",
            model="gpt-5.5",
            max_tokens=512,
            reasoning_effort="low",
        )

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_text="Paris",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=50),
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
        )

    provider.client.responses.create = fake_create
    assert provider.generate_text("Capital?", instructions="Answer briefly") == "Paris"
    assert captured["model"] == "gpt-5.5"
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["max_output_tokens"] == 512
    assert captured["store"] is False
    assert provider.get_context_window_tokens() == 1_050_000
    assert provider.get_last_usage()["cached_tokens"] == 50


def test_openai_explicit_max_completion_tokens_wins_for_gpt_5_6():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    with patch("memorizz.llms.openai.openai.OpenAI"):
        provider = MemorizzOpenAI(
            api_key="test-key",
            model="gpt-5.6-terra",
            max_tokens=16_000,
            max_completion_tokens=8_000,
        )

    assert provider._request_options["max_completion_tokens"] == 8_000
    assert "max_tokens" not in provider._request_options


def test_openai_responses_mode_translates_function_tools_and_usage():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    with patch("memorizz.llms.openai.openai.OpenAI"):
        provider = MemorizzOpenAI(
            api_key="test-key",
            model="gpt-5.6-terra",
            max_tokens=32_000,
            reasoning_effort="max",
            api_mode="responses",
        )

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="resp_1",
            output_text="",
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="terminal_exec",
                    arguments='{"command":"pwd"}',
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=64),
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
        )

    provider.client.responses.create = fake_create
    response = provider.generate(
        [{"role": "user", "content": "inspect"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "terminal_exec",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    assert response.choices[0].message.tool_calls[0].id == "call_1"
    assert captured["reasoning"] == {"effort": "max"}
    assert captured["max_output_tokens"] == 32_000
    assert captured["tools"][0]["name"] == "terminal_exec"
    assert captured["tools"][0]["strict"] is False
    assert provider.get_last_usage()["cached_tokens"] == 64
    assert provider.get_last_usage()["reasoning_tokens"] == 10
    assert provider.get_config()["api_mode"] == "responses"


def test_openai_responses_input_replays_tool_calls_and_outputs():
    from memorizz.llms.openai import OpenAI as MemorizzOpenAI

    items = MemorizzOpenAI._responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal_exec",
                            "arguments": '{"command":"pwd"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "terminal_exec",
                "content": '{"return_code":0,"stdout":"/app"}',
            },
        ]
    )

    assert items[0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "terminal_exec",
        "arguments": '{"command":"pwd"}',
    }
    assert items[1]["type"] == "function_call_output"
    assert items[1]["call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Embedding LRU cache
# ---------------------------------------------------------------------------


def test_embedding_manager_memoizes_repeat_calls():
    from memorizz.embeddings import EmbeddingManager

    manager = EmbeddingManager.__new__(EmbeddingManager)
    import threading
    from collections import OrderedDict

    manager._embedding_cache = OrderedDict()
    manager._embedding_cache_lock = threading.Lock()

    calls = []

    class FakeProvider:
        def get_embedding(self, text, **kwargs):
            calls.append(text)
            return [0.1, 0.2, 0.3]

    manager._provider = FakeProvider()

    first = manager.get_embedding("same query")
    second = manager.get_embedding("same query")
    third = manager.get_embedding("different query")

    assert first == second == [0.1, 0.2, 0.3]
    assert calls == ["same query", "different query"]
    # Cached copies are independent lists (no shared mutation).
    first.append(999.0)
    assert manager.get_embedding("same query") == [0.1, 0.2, 0.3]
