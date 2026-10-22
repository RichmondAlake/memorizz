# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Run tags reach the work a tagged turn starts: a delegate agent's turn and
a harness task carry the parent's tags, and they don't leak afterwards."""

from typing import Any, Dict, List

import pytest

from memorizz.memagent import MemAgent
from memorizz.observability.tags import current_trace_tags
from tests.unit.test_memagent_core_fixes import (  # noqa: F401  (fixtures)
    _fs_provider,
    _offline_embeddings,
    _StubModel,
    _tool_call_response,
    work_dir,
)


def _starts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("trace_kind") == "turn_start"]


@pytest.mark.unit
def test_a_turn_started_inside_a_tagged_turn_keeps_its_tags(work_dir):
    provider = _fs_provider(work_dir)
    child = MemAgent(
        model=_StubModel(responses=["child answer"]),
        memory_provider=provider,
        memory_ids=["memory-1"],
        auto_register=False,
    )
    child_events: List[Dict[str, Any]] = []
    child.set_stream_event_callback(lambda e: child_events.append(dict(e)))

    def ask_delegate(question: str) -> str:
        """Ask the delegate agent."""
        return child.run(question, memory_id="memory-1", thread_id="child")

    parent = MemAgent(
        model=_StubModel(
            responses=[
                _tool_call_response([("c1", "ask_delegate", {"question": "q"})]),
                "parent answer",
            ]
        ),
        tools=[ask_delegate],
        memory_provider=provider,
        memory_ids=["memory-1"],
        auto_register=False,
    )
    parent.run(
        "go",
        memory_id="memory-1",
        thread_id="parent",
        trace_tags={"experiment": "lme-v2"},
    )
    assert _starts(child_events)[-1]["trace_tags"] == {"experiment": "lme-v2"}
    assert current_trace_tags.get() == {}, "the tags end with the turn"

    child.run("again", memory_id="memory-1", thread_id="child")
    assert "trace_tags" not in _starts(child_events)[-1], "no stale tags later"
    parent.close()
    child.close()


@pytest.mark.unit
def test_a_harness_task_takes_the_running_turns_tags():
    from memorizz.metaharness.models import HarnessTask

    token = current_trace_tags.set({"experiment": "lme-v2", "question": "q17"})
    try:
        task = HarnessTask(
            task="fix it",
            workspace="/tmp/w",
            metadata={"trace_tags": {"question": "q18", "stage": "build"}},
        )
    finally:
        current_trace_tags.reset(token)
    assert task.metadata["trace_tags"] == {
        "question": "q18",
        "stage": "build",
        "experiment": "lme-v2",
    }, "its own tags first, then the parent's"
    assert "trace_tags" not in HarnessTask(task="t", workspace="/tmp/w").metadata
    with pytest.raises(ValueError):
        HarnessTask(task="t", workspace="/tmp/w", metadata={"trace_tags": {"a": "x=y"}})


@pytest.mark.unit
def test_a_harness_runs_trace_carries_its_tags(tmp_path):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.enums import MemoryType
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.metaharness.models import HarnessTask
    from memorizz.metaharness.service import MetaHarness
    from memorizz.metaharness.store import SQLiteHarnessRunStore
    from memorizz.observability.normalization import query_trace_events
    from tests.unit.test_metaharness import FakeHarness

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[FakeHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        token = current_trace_tags.set({"experiment": "lme-v2"})
        try:
            task = HarnessTask(task="Inspect", workspace=str(workspace), harness="fake")
        finally:
            current_trace_tags.reset(token)
        assert service.run(task).ok
        events = query_trace_events(
            provider,
            MemoryType.SHARED_MEMORY,
            limit=100,
            record_type="observability_trace_bundle",
        )["items"]
        assert events and all(
            e.get("trace_tags") == {"experiment": "lme-v2"} for e in events
        )
    finally:
        service.close()
        provider.close()
