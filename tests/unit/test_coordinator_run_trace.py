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
