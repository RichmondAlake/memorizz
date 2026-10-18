# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Fixes for issues reported against 0.16.6 on a real workload.

The per-turn tool-output digest leaves out MemoRizz's own memory tools;
recalled entities are printed as facts, not raw attribute records; the memory
supply snapshot counts the evidence pack; attached knowledge bases are
recalled whatever memory_id they were ingested under, and survive a rebuild;
a reload takes governed tools back with their policies.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz import governed_tool
from memorizz.learning.evidence import _row_content
from memorizz.learning.models import EvidenceItem, EvidencePack
from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

LLM = {"provider": "ollama", "model": "qwen2.5:7b"}


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


# ---------------------------------------------------------------------------
# 1. The recent tool-output digest
# ---------------------------------------------------------------------------


def _log(tool_name, index):
    return {
        "tool_log_id": f"tl-{index}",
        "tool_name": tool_name,
        "arguments": {"name": f"call-{index}"},
        "result": '{"ok": true, "entity_id": "4f1c1c1e-9a43-4c6b-a9d1-3d0f4c2b7e10"}',
    }


def _digest(rows, limit=3):
    host = SimpleNamespace(
        memory_manager=SimpleNamespace(list_tool_logs=lambda **kwargs: list(rows)),
        _current_memory_id="m1",
        _current_user_id=None,
        _current_thread_id="t1",
    )
    return MemAgent._format_recent_tool_logs_digest(host, limit=limit)


def test_the_tool_output_digest_leaves_out_memorizz_memory_tools():
    rows = [
        _log("entity_memory_upsert", 1),
        _log("search_orders", 2),
        _log("entity_memory_upsert", 3),
        _log("update_persona", 4),
        _log("refund_order", 5),
    ]

    digest = _digest(rows)

    assert "tool=search_orders" in digest and "tool=refund_order" in digest
    assert "entity_memory_upsert" not in digest
    assert "update_persona" not in digest


def test_the_digest_still_shows_up_to_its_limit_of_real_calls():
    rows = [_log("entity_memory_upsert", index) for index in range(8)]
    rows += [_log(f"tool_{index}", 10 + index) for index in range(4)]

    lines = [
        line for line in _digest(rows, limit=3).splitlines() if line.startswith("•")
    ]

    assert [line.split("tool=")[1].split()[0] for line in lines] == [
        "tool_0",
        "tool_1",
        "tool_2",
    ]


def test_only_memory_tools_leave_no_digest():
    assert _digest([_log("entity_memory_upsert", 1)]) == ""


# ---------------------------------------------------------------------------
# 2. Recalled entities as facts
# ---------------------------------------------------------------------------


def test_a_recalled_entity_is_printed_as_its_facts():
    row = {
        "entity_id": "e-1",
        "name": "Ada",
        "entity_type": "person",
        "attributes": [
            {
                "name": "role",
                "value": "migration lead",
                "confidence": 0.6,
                "source": "conversation",
                "updated_at": "2026-10-01T09:00:00+00:00",
            },
            {
                "name": "role",
                "value": "CTO",
                "confidence": 0.9,
                "source": "conversation",
                "updated_at": "2026-10-02T09:00:00+00:00",
            },
            {"name": "cutover", "value": "Friday", "confidence": 0.8},
        ],
        "relations": [{"entity_id": "e-2", "relation_type": "manages"}],
    }

    text = _row_content("entity_memory", row)

    assert text.splitlines() == [
        "Ada (person): role = CTO; cutover = Friday",
        "relations: manages → e-2",
    ]
    for noise in ("confidence", "source", "updated_at", "2026-10", "migration lead"):
        assert noise not in text


def test_an_entity_with_a_mapping_of_attributes_prints_the_same_way():
    row = {"name": "Lagos office", "attributes": {"city": "Lagos", "desks": 40}}

    assert (
        _row_content("entity_memory", row) == "Lagos office: city = Lagos; desks = 40"
    )


# ---------------------------------------------------------------------------
# 3. Attached knowledge bases are recalled whatever memory_id they carry
# ---------------------------------------------------------------------------


def test_an_attached_knowledge_base_ingested_without_a_memory_id_is_recalled(
    provider,
):
    kb_id = KnowledgeBase(provider).ingest_knowledge(
        "The Oracle fixes ship on Friday.", namespace="plans", embeddings="off"
    )
    agent = MemAgent(
        llm_config=LLM,
        memory_provider=provider,
        name="recall",
        memory_ids=["agent-memory"],
        application_mode="assistant",
    )
    agent.knowledge_base_ids = [kb_id]

    context = agent._build_context("When do the Oracle fixes ship?", "agent-memory")

    recalled = context.get("retrieved_memories") or []
    assert [(row["source"], row["text"]) for row in recalled] == [
        ("knowledge_base", "The Oracle fixes ship on Friday.")
    ]


def test_attached_knowledge_bases_an_agent_cannot_recall_are_reported(provider, caplog):
    agent = MemAgent(
        llm_config=LLM,
        memory_provider=provider,
        name="no-recall",
        memory_types=["conversation_memory"],
    )
    agent.knowledge_base_ids = ["kb-1"]

    with caplog.at_level(logging.WARNING, logger="memorizz.memagent.core"):
        agent._build_context("anything", "agent-memory")
        agent._build_context("again", "agent-memory")

    warnings = [r for r in caplog.records if "not recalled automatically" in r.message]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# 4. A rebuild under the same name keeps the stored knowledge bases
# ---------------------------------------------------------------------------


def _builder(provider):
    return (
        MemAgentBuilder()
        .with_llm_config(LLM)
        .with_memory_provider(provider)
        .with_name("rebuilt")
    )


def test_rebuilding_an_agent_keeps_its_knowledge_bases(provider):
    first = _builder(provider).with_knowledge_base("kb-1").build_and_save()

    second = _builder(provider).build_and_save()

    assert second.agent_id == first.agent_id
    assert second.knowledge_base_ids == ["kb-1"]
    assert provider.retrieve_memagent(second.agent_id).knowledge_base_ids == ["kb-1"]


def test_a_rebuild_that_names_knowledge_bases_replaces_or_detaches_them(provider):
    _builder(provider).with_knowledge_base(["kb-1", "kb-2"]).build_and_save()

    replaced = _builder(provider).with_knowledge_base("kb-3").build_and_save()
    assert replaced.knowledge_base_ids == ["kb-3"]

    detached = _builder(provider).with_knowledge_base([]).build_and_save()
    assert detached.knowledge_base_ids == []
    assert not provider.retrieve_memagent(detached.agent_id).knowledge_base_ids


# ---------------------------------------------------------------------------
# 5. Reloading takes governed tools back with their policies
# ---------------------------------------------------------------------------


@governed_tool(
    deterministic=False,
    side_effects=True,
    requires_approval=True,
    approval_reason="Creates a customer",
    domains=("crm",),
)
def create_customer(name: str) -> dict:
    """Create one CRM customer after host approval."""
    return {"ok": True, "name": name}


def test_a_reload_with_its_tools_restores_their_code_and_policy(provider):
    saved = MemAgent(
        llm_config=LLM, memory_provider=provider, name="crm", tools=[create_customer]
    )
    saved.save()

    loaded = MemAgent.load(
        saved.agent_id, memory_provider=provider, tools=[create_customer]
    )

    entry = loaded.tool_manager.tools["create_customer"]
    assert callable(entry["function"])
    policy = loaded.tool_manager.get_tool_policy("create_customer")
    assert policy["requires_approval"] is True and policy["side_effects"] is True


# ---------------------------------------------------------------------------
# 6. The memory supply snapshot counts the evidence pack
# ---------------------------------------------------------------------------


def _item(index, source_type, content):
    return EvidenceItem(
        evidence_id=f"ev-{index}",
        source_type=source_type,
        source_id=f"row-{index}",
        content=content,
        relevance=0.9 - index / 10,
        token_estimate=5,
        content_hash=f"hash-{index}",
        reason="semantic",
    )


def test_last_memory_context_evidence_counts_the_evidence_pack(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider, name="evidence")
    pack = EvidencePack(
        pack_id="pack-1",
        query="When do the fixes ship?",
        items=(
            _item(0, "knowledge_base", "The Oracle fixes ship on Friday."),
            _item(1, "entity_memory", "Ada (person): role = CTO"),
            _item(2, "knowledge_base", "Cutover is at noon."),
        ),
        created_at="2026-10-18T09:00:00+00:00",
        token_budget=1000,
        tokens_used=15,
        candidate_tokens=40,
        candidate_count=7,
        rejected_count=4,
        scope={},
        query_hash="q",
        policy_hash="p",
        latency_ms=1.0,
    )

    agent._emit_memory_context_trace({"evidence_pack": pack})
    evidence = agent.last_memory_context_evidence()

    assert evidence["source_counts"]["evidence_items"] == 3
    assert evidence["supplied_count"] >= 3
    assert evidence["evidence_source_counts"] == {
        "knowledge_base": 2,
        "entity_memory": 1,
    }
    assert evidence["retrieved_source_counts"]["evidence_pack"] == 7
    assert evidence["retrieved_selected_source_counts"]["evidence_pack"] == 3
    assert [ref["source"] for ref in evidence["evidence_refs"]] == [
        "knowledge_base",
        "entity_memory",
        "knowledge_base",
    ]
    # Content-free: hashed references only.
    assert "Friday" not in str(evidence)


# ---------------------------------------------------------------------------
# Reported on 0.17.1
# ---------------------------------------------------------------------------


def test_deleting_a_knowledge_base_that_does_not_exist_reports_false(provider):
    knowledge = KnowledgeBase(provider)
    kb_id = knowledge.ingest_knowledge("Kept.", namespace="notes", embeddings="off")

    assert knowledge.delete_knowledge("no-such-knowledge-base") is False
    assert knowledge.delete_knowledge(kb_id) is True
    assert knowledge.delete_knowledge(kb_id) is False


def test_reloading_a_continual_learning_agent_does_not_warn(provider, caplog):
    built = (
        MemAgentBuilder()
        .with_llm_config(LLM)
        .with_memory_provider(provider)
        .with_name("learner")
        .with_continual_learning(True)
        .build_and_save()
    )

    with caplog.at_level(logging.WARNING, logger="memorizz"):
        loaded = MemAgent.load(built.agent_id, memory_provider=provider)
        MemAgent.load(built.agent_id, memory_provider=provider)

    assert not [r for r in caplog.records if "omit workflow_memory" in r.message]
    assert "workflow_memory" in {t.value for t in loaded.active_memory_types}
