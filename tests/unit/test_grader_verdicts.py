# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Grader verdicts joined to a turn: record_task_outcome(turn_id=..., verdict=...),
the Overview's graded_failures attention item, and verdicts on the turn in
Traces."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore
from memorizz.observability.overview import build_operations_overview
from memorizz.observability.store import resolve_outcome_status
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.dashboard import link_attention, load_verdicts


class _Model:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 10}

    def generate(self, messages, tools=None):
        return "answer"


@pytest.fixture()
def provider(tmp_path):
    instance = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield instance
    instance.close()


def _agent(provider):
    agent = MemAgent(
        model=_Model(),
        memory_provider=provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name="graded",
        context_policy={"progressive_tool_disclosure": False},
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    return agent


def _outcomes(provider):
    return ObservabilityStore(provider).list_signals(verified_only=False)["outcomes"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "status, verdict, expected",
    [
        ("failure", None, "failure"),
        (None, "fail", "failure"),
        (None, "pass", "success"),
        (None, "PARTIAL", "partial"),
        ("success", "pass", "success"),
        (None, "success", "success"),
    ],
)
def test_status_and_verdict_resolve_to_one_outcome(status, verdict, expected):
    assert resolve_outcome_status(status, verdict) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "status, verdict",
    [(None, None), ("done", None), (None, "maybe"), ("success", "fail")],
)
def test_bad_or_conflicting_outcomes_are_rejected(status, verdict):
    with pytest.raises(ValueError):
        resolve_outcome_status(status, verdict)


@pytest.mark.unit
def test_a_verdict_joins_the_named_turn(provider):
    agent = _agent(provider)
    agent.run("first", memory_id="m1", thread_id="t1", user_id="u1")
    first = agent.get_trace_context()
    agent.run("second", memory_id="m1", thread_id="t2", user_id="u1")

    record = agent.record_task_outcome(
        turn_id=first["turn_id"],
        verdict="fail",
        score=0.0,
        grader="llm-judge",
        reason_code="wrong_answer",
    )
    assert record["status"] == "failure"
    assert record["turn_id"] == first["turn_id"]
    assert record["root_trace_id"] == first["root_trace_id"]
    assert record["thread_id"] == "t1"
    assert record["grader"] == "llm-judge"
    assert record["reason_code"] == "wrong_answer"
    assert record["source"] == "grader"
    assert record["score"] == 0.0
    assert record["verified"] is True

    (stored,) = _outcomes(provider)
    assert stored["reason_code"] == "wrong_answer" and stored["grader"] == "llm-judge"

    # The positional status form keeps working and joins the last turn.
    legacy = agent.record_task_outcome("success", metrics={"completed": True})
    assert legacy["turn_id"] != first["turn_id"]
    assert legacy["source"] == "application" and "grader" not in legacy


@pytest.mark.unit
def test_a_turn_from_an_earlier_process_is_found_in_its_trace(provider):
    agent = _agent(provider)
    agent.run("question", memory_id="m1", thread_id="t1")
    context = agent.get_trace_context()
    # A new agent object with the same id has no turns in memory.
    reloaded = _agent(provider)
    reloaded.agent_id = agent.agent_id
    record = reloaded.record_task_outcome(
        turn_id=context["turn_id"], verdict="pass", grader="exact-match"
    )
    assert record["root_trace_id"] == context["root_trace_id"]
    assert record["memory_id"] == "m1" and record["thread_id"] == "t1"
    with pytest.raises(ValueError, match="No trace was found"):
        reloaded.record_task_outcome(turn_id="no-such-turn", verdict="fail")


@pytest.mark.unit
def test_verdict_labels_and_turn_disagreements_are_rejected(provider):
    agent = _agent(provider)
    agent.run("question", memory_id="m1", thread_id="t1")
    turn_id = agent.get_trace_context()["turn_id"]
    for bad in ("two  spaces", "x" * 81, "someone@example.com", ""):
        with pytest.raises(ValueError):
            agent.record_task_outcome(turn_id=turn_id, verdict="fail", reason_code=bad)
    with pytest.raises(ValueError, match="different turns"):
        agent.record_task_outcome(
            turn_id=turn_id,
            verdict="fail",
            trace_context={**agent.get_trace_context(), "turn_id": "another"},
        )
    assert _outcomes(provider) == []


@pytest.mark.unit
def test_learning_control_plane_path_keeps_grader_fields(provider):
    from memorizz.learning import LearningControlPlane

    agent = _agent(provider)
    agent.learning_control_plane = LearningControlPlane(
        provider,
        agent_id=agent.agent_id,
        config={"enabled": True, "compile_async": False, "compile_every_n_events": 0},
    )
    agent.run("question", memory_id="m1", thread_id="t1")
    turn_id = agent.get_trace_context()["turn_id"]
    record = agent.record_task_outcome(
        turn_id=turn_id, verdict="fail", grader="llm-judge", reason_code="wrong_answer"
    )
    assert record["grader"] == "llm-judge"
    assert record["learning_event"]["payload"]["reason_code"] == "wrong_answer"
    (stored,) = _outcomes(provider)
    assert stored["turn_id"] == turn_id and stored["status"] == "failure"


# --- Overview -----------------------------------------------------------------

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _run(turn, agent, minutes_ago, thread="t"):
    at = NOW - timedelta(minutes=minutes_ago)
    return [
        {
            "trace_kind": "turn_start",
            "turn_id": turn,
            "root_trace_id": f"root-{turn}",
            "agent_id": agent,
            "thread_id": thread,
            "timestamp": at.isoformat(),
        },
        {
            "trace_kind": "turn_result",
            "turn_id": turn,
            "root_trace_id": f"root-{turn}",
            "agent_id": agent,
            "thread_id": thread,
            "status": "success",
            "timestamp": (at + timedelta(seconds=2)).isoformat(),
        },
    ]


def _verdict(turn, status, *, grader="llm-judge", reason=None, minutes_ago=1):
    return {
        "record_type": "observability_outcome",
        "turn_id": turn,
        "root_trace_id": f"root-{turn}",
        "agent_id": "a1",
        "thread_id": "t",
        "status": status,
        "grader": grader,
        "reason_code": reason,
        "verified": True,
        "updated_at": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
    }


@pytest.mark.unit
def test_failing_verdicts_need_attention_and_link_the_newest_failing_turn():
    events = [*_run("t1", "a1", 30), *_run("t2", "a1", 20), *_run("t3", "a1", 10)]
    verdicts = [
        _verdict("t1", "failure", reason="wrong_answer", minutes_ago=25),
        _verdict("t2", "failure", reason="wrong_answer", minutes_ago=15),
        # A re-grade passes t2 for one grader; another grader still fails it.
        _verdict("t2", "success", minutes_ago=5),
        _verdict(
            "t2", "failure", grader="exact-match", reason="missing", minutes_ago=4
        ),
        _verdict("t3", "success", minutes_ago=3),
    ]
    overview = build_operations_overview(
        events, window="24h", now=NOW, verdicts=verdicts
    )
    assert overview["totals"]["graded_runs"] == 3
    assert overview["totals"]["graded_failures"] == 2
    (item,) = [i for i in overview["attention"] if i["kind"] == "graded_failures"]
    assert item["title"] == "2 turns failed grading"
    assert item["run"]["turn_id"] == "t2", "the newest failing turn"
    assert "wrong_answer" in item["detail"] and "exact-match" in item["detail"]
    grades = {run["turn_id"]: run.get("grade") for run in overview["recent"]}
    assert grades == {"t1": "fail", "t2": "fail", "t3": "pass"}

    (linked,) = [
        i
        for i in link_attention(overview["attention"], {"a1"})
        if i["kind"] == "graded_failures"
    ]
    assert linked["actions"] == [
        {
            "label": "Open the newest failing turn",
            "href": "/traces?agent_id=a1&thread_id=t&root_trace_id=root-t2"
            "&turn_id=t2#trace-timeline",
        }
    ]


@pytest.mark.unit
def test_passing_verdicts_raise_nothing_and_old_ones_are_ignored():
    events = _run("t1", "a1", 10)
    overview = build_operations_overview(
        events,
        window="24h",
        now=NOW,
        verdicts=[
            _verdict("t1", "success"),
            # A failure for a turn outside the window, recorded long ago.
            _verdict("old", "failure", minutes_ago=60 * 24 * 3),
        ],
    )
    assert overview["totals"]["graded_failures"] == 0
    assert not [i for i in overview["attention"] if i["kind"] == "graded_failures"]

    # A failure recorded in the window for a turn whose trace isn't loaded
    # still needs attention, linked from the verdict's own record.
    overview = build_operations_overview(
        [], window="24h", now=NOW, verdicts=[_verdict("gone", "failure")]
    )
    (item,) = overview["attention"]
    assert item["kind"] == "graded_failures"
    assert item["run"]["turn_id"] == "gone" and item["run"]["agent_id"] == "a1"


# --- UI ---------------------------------------------------------------------


@pytest.fixture()
def ui(provider, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)


@pytest.mark.unit
def test_verdicts_show_on_the_turn_and_on_the_overview(provider, ui):
    agent = _agent(provider)
    agent.run("question", memory_id="m1", thread_id="t1")
    context = agent.get_trace_context()
    agent.record_task_outcome(
        turn_id=context["turn_id"],
        verdict="fail",
        score=0.0,
        grader="llm-judge",
        reason_code="wrong_answer",
    )
    assert [row["turn_id"] for row in load_verdicts()] == [context["turn_id"]]

    page = ui.get(agent.trace_url("http://testserver").replace("http://testserver", ""))
    assert page.status_code == 200, page.text
    assert "Verdict: fail · llm-judge · wrong answer · score 0.0" in page.text

    overview = ui.get("/dashboard?window=24h")
    assert overview.status_code == 200, overview.text
    assert "1 turn failed grading" in overview.text
    assert "Open the failing turn" in overview.text
