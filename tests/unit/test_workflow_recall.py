"""RetrievalPolicy(workflow_scope=...): past workflows recalled into the
prompt, inside the memory and user they ran for."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.workflow.workflow import (
    Workflow,
    WorkflowOutcome,
    workflow_recall_text,
)
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy
from tests.unit.test_memagent_core_fixes import _DummyEmbeddingProvider, _StubModel

pytestmark = pytest.mark.unit

REQUEST = "Will it rain in Lisbon during my trip?"


@pytest.fixture
def provider():
    root = Path(tempfile.mkdtemp(prefix="workflow-recall-"))
    try:
        yield FileSystemProvider(
            FileSystemConfig(
                root_path=root / "memory",
                embedding_provider=_DummyEmbeddingProvider(),
                use_faiss=False,
                lazy_vector_indexes=True,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _store(
    provider,
    *,
    thread_id: str,
    user_id: Optional[str] = "traveller",
    promoted: Optional[str] = None,
    sealed: bool = False,
) -> str:
    workflow = Workflow(
        name="weather check",
        description="Checked the weather",
        steps={
            "Step 1: weather_outlook": {
                "arguments": {"place": "Lisbon", "start_date": "2026-11-20"},
                "result": {"days": ["…a long result that is not recalled…"]},
                "error": None,
                "tool_outcome": {"ok": True},
            },
            "Step 2: exchange_rate": {
                "arguments": {"base": "GBP", "quote": "EUR"},
                "result": None,
                "error": "ECB feed timed out",
                "tool_outcome": {"ok": False},
            },
        },
        memory_id="m1",
        outcome=WorkflowOutcome.FAILURE,
        user_query=REQUEST,
        user_id=user_id,
        promoted_skill_id=promoted,
        generate_embedding=False,
        embedding=[1.0, 1.0, 0.0],
        thread_id=thread_id,
    )
    record_id = workflow.store_workflow(provider)
    if sealed:
        provider.update_by_id(
            record_id, {"retention_state": "sealed"}, MemoryType.WORKFLOW_MEMORY
        )
    return record_id


def _prompt_after_turn(provider, policy: RetrievalPolicy, *, thread_id: str) -> str:
    model = _StubModel(responses=["ok"])
    agent = MemAgent(
        model=model,
        memory_provider=provider,
        memory_ids=["m1"],
        memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.WORKFLOW_MEMORY],
        retrieval_policy=policy,
        auto_register=False,
    )
    try:
        agent.run(REQUEST, memory_id="m1", thread_id=thread_id, user_id="traveller")
    finally:
        agent.close()
    # Recalled memories ride on the user turn (the system prompt stays
    # cacheable), so read the whole request.
    return "\n".join(
        str(message.get("content"))
        for message in model.requests[0]
        if isinstance(message, dict)
    )


def test_the_recalled_text_names_each_call_and_how_it_went():
    row = {
        "user_query": REQUEST,
        "outcome": "failure",
        "steps": {
            "Step 1: weather_outlook": {
                "arguments": {"place": "Lisbon"},
                "result": "rain",
                "error": None,
            },
            "Step 2: exchange_rate": {
                "arguments": {"base": "GBP"},
                "error": "ECB feed timed out",
            },
        },
    }
    assert workflow_recall_text(row) == (
        f'Earlier task "{REQUEST}": weather_outlook({{"place": "Lisbon"}}) worked; '
        'exchange_rate({"base": "GBP"}) failed: ECB feed timed out. '
        "Outcome: failure."
    )


def test_workflows_stay_out_of_the_prompt_by_default(provider):
    _store(provider, thread_id="t1")
    prompt = _prompt_after_turn(provider, RetrievalPolicy(), thread_id="t1")
    assert "Earlier task" not in prompt


def test_memory_scope_recalls_a_workflow_from_another_thread(provider):
    _store(provider, thread_id="t1")
    prompt = _prompt_after_turn(
        provider, RetrievalPolicy(workflow_scope="memory"), thread_id="t2"
    )
    assert f'Earlier task "{REQUEST}"' in prompt
    assert "exchange_rate" in prompt and "ECB feed timed out" in prompt
    assert "a long result that is not recalled" not in prompt


def test_thread_scope_recalls_only_this_threads_workflows(provider):
    _store(provider, thread_id="t1")
    policy = RetrievalPolicy(workflow_scope="thread")
    assert "Earlier task" in _prompt_after_turn(provider, policy, thread_id="t1")
    assert "Earlier task" not in _prompt_after_turn(provider, policy, thread_id="t2")


def test_a_sealed_policy_keeps_workflows_in_their_thread(provider):
    _store(provider, thread_id="t1")
    sealed = RetrievalPolicy.sealed(workflow_scope="memory")
    assert "Earlier task" not in _prompt_after_turn(provider, sealed, thread_id="t2")


def test_a_sealed_threads_workflow_is_not_recalled_elsewhere(provider):
    _store(provider, thread_id="t1", sealed=True)
    policy = RetrievalPolicy(workflow_scope="memory")
    assert "Earlier task" not in _prompt_after_turn(provider, policy, thread_id="t2")
    assert "Earlier task" in _prompt_after_turn(provider, policy, thread_id="t1")


def test_another_users_and_skill_covered_workflows_are_not_recalled(provider):
    _store(provider, thread_id="t1", user_id="someone-else")
    _store(provider, thread_id="t1", promoted="skill-1")
    prompt = _prompt_after_turn(
        provider, RetrievalPolicy(workflow_scope="memory"), thread_id="t1"
    )
    assert "Earlier task" not in prompt


def test_the_policy_saves_its_workflow_scope():
    policy = RetrievalPolicy(workflow_scope="memory")
    assert policy.to_dict()["workflow_scope"] == "memory"
    assert RetrievalPolicy.from_value(policy.to_dict()) == policy
    assert "workflow_scope" not in RetrievalPolicy().to_dict()
    with pytest.raises(ValueError, match="workflow_scope"):
        RetrievalPolicy(workflow_scope="everywhere")


def test_a_sealed_turn_marks_its_workflow(provider):
    model = _StubModel(responses=["ok"])
    agent = MemAgent(
        model=model,
        memory_provider=provider,
        memory_ids=["m1"],
        memory_types=[MemoryType.WORKFLOW_MEMORY],
        retrieval_policy=RetrievalPolicy.sealed(),
        auto_register=False,
    )
    marked: List[Dict[str, Any]] = []
    agent._mark_sealed = lambda memory_type, ids: marked.append(
        {"type": memory_type, "ids": list(ids)}
    )
    workflow = Workflow(
        name="w",
        steps={"Step 1: t": {"arguments": {}, "error": None}},
        memory_id="m1",
        generate_embedding=False,
        embedding=[1.0, 0.0, 0.0],
    )
    try:
        assert agent._persist_workflow_run(workflow)
    finally:
        agent.close()
    assert marked and marked[0]["type"] == MemoryType.WORKFLOW_MEMORY
