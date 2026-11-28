"""A coordinator's turn is a run of its own, beside its delegates' runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memorizz import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit


class _Model:
    """Plans one task for the delegate it is told about, and otherwise says
    who is answering."""

    model = "fake"

    def __init__(self, name, delegate_id=None):
        self.name, self.delegate_id = name, delegate_id

    def get_config(self):
        return {"provider": "openai", "model": "fake"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return {"input_tokens": 120, "output_tokens": 30}

    def generate(self, messages, tools=None, **kwargs):
        asked = " ".join(str(m.get("content")) for m in messages)
        if self.delegate_id and "decompose this task" in asked:
            plan = {
                "task_id": "t1",
                "description": "Write a slogan",
                "assigned_agent_id": self.delegate_id,
            }
            return json.dumps([plan])
        return f"{self.name} answered"

    def generate_stream(self, messages, tools=None, **kwargs):
        text = self.generate(messages, tools=tools)
        yield {"type": "content", "content": text}
        yield {"type": "done", "content": text}


def _team(tmp_path: Path, delegates=True):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    writer = MemAgent(
        model=_Model("Ann"), memory_provider=provider, auto_register=False
    )
    writer.name = "Ann"
    lead = MemAgent(
        model=_Model("Lead", writer.agent_id),
        memory_provider=provider,
        auto_register=False,
        delegates=[writer] if delegates else None,
    )
    lead.name = "Lead"
    return provider, lead, writer


def test_a_coordinators_turn_is_recorded_as_its_own_run(tmp_path):
    provider, lead, writer = _team(tmp_path)
    recorded = []
    real = lead._record_stream_trace_bundle

    def keep(memory_id, thread_id, user_id=None):
        recorded.append((memory_id, thread_id, list(lead._stream_trace_events)))
        return real(memory_id, thread_id, user_id=user_id)

    lead._record_stream_trace_bundle = keep

    answer = lead.run("A slogan for the planter", memory_id="team", thread_id="t")

    assert "answered" in str(answer)
    ((memory_id, thread_id, events),) = recorded
    assert (memory_id, thread_id) == ("team", "t")
    kinds = [(e.get("trace_kind"), e.get("stage")) for e in events]
    assert kinds[0] == ("turn_start", None)
    assert kinds[-1] == ("turn_result", None)
    # The coordinator's own model calls are in its run: the plan, then the
    # combining of what its delegate returned.
    stages = [stage for kind, stage in kinds if kind == "model_result"]
    assert stages[0] == "planning" and "combining" in stages
    planning = next(e for e in events if e.get("stage") == "planning")
    assert planning["input_tokens"] == 120 and planning["duration_ms"] >= 0
    assert events[-1]["status"] == "success"
    # The trace is closed, so a later call is counted on its own.
    assert lead._stream_trace_events is None


def test_a_coordinator_that_fails_closes_its_run_as_failed(tmp_path):
    provider, lead, writer = _team(tmp_path)
    closed = []
    real = lead._finish_trace_turn

    def keep(status, **more):
        closed.append((status, more.get("error_code")))
        return real(status, **more)

    lead._finish_trace_turn = keep

    def broken(*args, **kwargs):
        raise RuntimeError("the delegates are out")

    lead.delegate = broken
    with pytest.raises(RuntimeError):
        lead.run("A slogan", memory_id="team", thread_id="t")

    assert closed == [("error", "RuntimeError")]
    assert lead._stream_trace_events is None


def test_a_call_counted_outside_a_turn_adds_usage_and_no_trace(tmp_path):
    provider, lead, writer = _team(tmp_path, delegates=False)
    lead._run_usage = {}

    lead.count_model_call(stage="planning", duration_ms=12.5)

    assert lead._run_usage.get("input_tokens") == 120
    assert lead._stream_trace_events is None


def _hierarchy(tmp_path: Path):
    """Ona gives work to Lena, a lead who gives it to Cole: two levels on the
    real file store."""
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )

    def agent(name, delegate=None):
        made = MemAgent(
            model=_Model(name, delegate.agent_id if delegate else None),
            memory_provider=provider,
            auto_register=False,
            delegates=[delegate] if delegate else None,
        )
        made.name = name
        return made

    cole = agent("Cole")
    lena = agent("Lena", cole)
    ona = agent("Ona", lena)
    return provider, ona, lena, cole


def test_a_lead_under_another_lead_writes_to_the_same_board(tmp_path):
    """On the file store a session's record ID and its own memory_id were two
    different IDs. The lower lead joined by the second, which no record was
    filed under, so none of its entries or sub-agents were saved."""
    from memorizz.coordination import SharedMemory

    provider, ona, lena, cole = _hierarchy(tmp_path)

    report = ona.delegate(
        "A slogan for the planter", memory_id="team", thread_id="t", return_report=True
    )

    assert report["ok"], report.get("error")
    assert report.get("recording_warnings", []) == []
    shared = SharedMemory(provider)
    board = report["shared_memory_id"]
    session = provider.retrieve_by_id(board, "shared_memory")
    assert session["memory_id"] == session["_id"] == board
    writers = {entry["agent_id"] for entry in shared.get_blackboard_entries(board)}
    # Both levels wrote: the root's plan, and the lower lead's own.
    assert {ona.agent_id, lena.agent_id} <= writers
    hierarchy = shared.get_agent_hierarchy(board)
    assert cole.agent_id in json.dumps(hierarchy)


def test_the_file_store_keys_a_shared_session_by_its_own_id(tmp_path):
    from memorizz.coordination import SharedMemory
    from memorizz.enums import MemoryType

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    shared = SharedMemory(provider)

    created = shared.create_shared_session("root", ["lead"], workflow_id="wf-1")
    found = shared.find_active_session_for_agent("lead", workflow_id="wf-1")

    assert str(found.get("memory_id")) == str(found.get("_id")) == created
    assert shared.add_blackboard_entry(created, "lead", {"n": 1}, "task_start")
    assert shared.register_sub_agents(created, "lead", ["cole", "sam"])
    # A record that brings its own ID keeps it (the change-history rows do).
    kept = provider.store(
        {"_id": "mem-change-1", "memory_id": "team", "content": "{}"},
        MemoryType.SHARED_MEMORY,
    )
    assert kept == "mem-change-1"


def test_a_delegated_job_reports_what_every_run_used(tmp_path):
    """Run usage is kept per context and each delegate runs in its own, so the
    coordinator saw only its own calls and nobody saw a total."""
    provider, ona, lena, cole = _hierarchy(tmp_path)
    # Left over from an earlier direct run; not part of this job.
    ona._run_usage = {"calls": 9, "input_tokens": 9000, "output_tokens": 900}

    report = ona.delegate(
        "A slogan for the planter", memory_id="team", thread_id="t", return_report=True
    )

    usage = report["usage"]
    (task,) = report["tasks"]
    # Every fake call reports 120 in and 30 out.
    per_call = {"input_tokens": 120, "output_tokens": 30}
    assert task["usage"]["calls"] >= 1
    assert usage["coordinator"]["calls"] == 2  # the plan, and the combining
    # Lena's run is her own two calls and Cole's one, counted through her.
    assert task["usage"]["calls"] == usage["delegates"]["calls"] == 3
    assert task["usage"]["delegated"]["calls"] == 1
    assert usage["total"]["calls"] == 5
    for key, each in per_call.items():
        assert usage["total"][key] == 5 * each
    # The coordinator's own run usage now answers for the whole job.
    assert ona.get_last_run_usage()["calls"] == 5
    assert ona.get_last_run_usage()["delegated"]["calls"] == 3
