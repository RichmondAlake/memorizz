"""Answer judging: opt-in, durable evidence and separate execution measurements."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessRun,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome
from memorizz.metaharness.judging import (
    HarnessJudge,
    _api_cost,
    judge_config,
    parse_verdict,
)

pytestmark = pytest.mark.unit


def _run(run_id="a", answer="Paris", **kwargs):
    return HarnessRun(
        run_id=run_id,
        harness="codex",
        status=HarnessStatus.SUCCEEDED,
        task={
            "task": "What is the capital of France?",
            "metadata": {"source": "plugin"},
        },
        result={"final_response": answer, "cost_usd": 0.3, "latency_ms": 1400},
        **kwargs,
    )


def _wait(judge, judgment_id):
    deadline = time.monotonic() + 4
    while True:
        value = judge.get(judgment_id)
        if value["status"] not in {"queued", "running"}:
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.01)


class _Model:
    def __init__(self, calls, response=None):
        self.calls = calls
        self.response = response

    def generate_text(self, prompt, instructions):
        self.calls.append((json.loads(prompt), instructions))
        if self.response is not None:
            return self.response
        correct = json.loads(prompt)["candidate_answer"] == "Paris"
        return json.dumps(
            {
                "score": 100 if correct else 20,
                "rationale": "Matches reference" if correct else "Wrong capital",
                "issues": [] if correct else ["Capital is Paris"],
            }
        )


def test_judgment_survives_import_and_restart_and_detects_changed_answer(tmp_path):
    path = tmp_path / "runs.sqlite3"
    store = SQLiteHarnessRunStore(path)
    calls = []
    judge = HarnessJudge(store, model_factory=lambda _: _Model(calls))
    store.create(_run())
    config = judge_config({"reference": "Paris"})
    judge.settings(config)
    job = judge.start([store.get("a").to_dict()], config)
    value = _wait(judge, job["judgment_id"])
    assert value["status"] == "completed"
    assert value["results"]["a"]["score"] == 100
    assert value["results"]["a"]["cost_usd"] == 0
    assert value["results"]["a"]["cost_basis"] == "no_external_api_charge"
    assert calls[0][0] == {
        "task": "What is the capital of France?",
        "candidate_answer": "Paris",
        "reference": "Paris",
    }
    assert "codex" not in calls[0][1].lower()
    assert "instructions to you" in calls[0][1]
    assert "inputs" not in value and "owner_pid" not in value
    store.replace(_run(), [])
    judge.close()
    store.close()
    store = SQLiteHarnessRunStore(path)
    judge = HarnessJudge(store)
    try:
        assert judge.settings() == config
        annotated = judge.annotate([store.get("a").to_dict()])[0]
        assert annotated["judgment"]["score"] == 100
        assert annotated["result"]["cost_usd"] == 0.3
        assert annotated["result"]["latency_ms"] == 1400
        store.replace(_run(answer="London"), [])
        current = judge.annotate([store.get("a").to_dict()])[0]
        assert current["judgment"]["status"] == "stale"
        assert current["judgment"]["score"] is None
        store.delete(["a"])
        assert judge.get(job["judgment_id"]) is None
    finally:
        judge.close()
        store.close()


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": true, "rationale": "ok"}',
        '{"score": "95", "rationale": "ok"}',
        '{"score": NaN, "rationale": "ok"}',
        '{"score": 101, "rationale": "ok"}',
        '{"score": -1, "rationale": "ok"}',
        '{"score": 90}',
        '{"score": 90, "rationale": "ok", "issues": "none"}',
        "garbage",
        "[]",
    ],
)
def test_invalid_verdicts_never_become_scores(raw):
    with pytest.raises(ValueError):
        parse_verdict(raw)


def test_fenced_json_and_zero_score_are_valid():
    assert (
        parse_verdict('```json\n{"score":0,"rationale":"Incorrect"}\n```')["score"] == 0
    )


@pytest.mark.parametrize(
    "config",
    [
        {"provider": "unknown"},
        {"provider": "openai"},
        {"model": ""},
        {"prompt": ""},
        {"pass_score": float("nan")},
        {"pass_score": True},
        {"reference": "x" * 16001},
        {"api_key": "do-not-accept"},
        [],
    ],
)
def test_config_validation(config):
    with pytest.raises(ValueError):
        judge_config(config)


def test_a_failed_judge_has_no_score_and_no_paid_fallback(tmp_path):
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    store.create(_run())
    seen = []

    def broken(config):
        seen.append(config["provider"])
        raise RuntimeError("provider failed with secret details")

    judge = HarnessJudge(store, model_factory=broken)
    try:
        job = judge.start([store.get("a").to_dict()])
        value = _wait(judge, job["judgment_id"])
        assert value["status"] == "failed"
        assert value["results"]["a"]["score"] is None
        assert "secret" not in value["results"]["a"]["error"]
        assert seen == ["ollama"]
    finally:
        judge.close()
        store.close()


def test_shutdown_interrupts_pending_jobs_without_late_writes(tmp_path):
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    store.create(_run())
    store.create(_run("b"))
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Blocking(_Model):
        def generate_text(self, prompt, instructions):
            entered.set()
            assert release.wait(4)
            return super().generate_text(prompt, instructions)

    judge = HarnessJudge(store, model_factory=lambda _: Blocking(calls))
    first = judge.start([store.get("a").to_dict()])
    assert entered.wait(2)
    second = judge.start([store.get("b").to_dict()])
    judge.close()
    assert judge.get(first["judgment_id"])["status"] == "interrupted"
    assert judge.get(second["judgment_id"])["status"] == "interrupted"
    store.close()
    release.set()
    judge._thread.join(2)
    assert not judge._thread.is_alive()
    assert len(calls) == 1


def test_orphaned_job_is_interrupted_without_reexecution(tmp_path):
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    store.create(_run())
    judge = HarnessJudge(store)
    job = dict(
        judgment_id="orphan",
        created_at="2026-10-03T12:00:00+00:00",
        status="running",
        owner_pid=99999999,
        config=judge_config(),
        config_hash="x",
        run_ids=["a"],
        inputs={},
        results={},
    )
    store.create_judgment(job)
    try:
        assert judge.get("orphan")["status"] == "interrupted"
    finally:
        judge.close()
        store.close()


def test_remote_evaluation_cost_uses_shared_model_and_service_tier_rates():
    config = judge_config({"provider": "openai", "model": "gpt-6.1-sol"})
    measured = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "cached_tokens": 500,
            "cache_write_tokens": 0,
        },
        "response_metadata": {"service_tier": "priority"},
    }
    assert _api_cost(config, measured)["cost_usd"] == pytest.approx(0.0041)
    assert _api_cost(config, {})["cost_usd"] is None
    assert (
        _api_cost(
            judge_config({"provider": "openai", "model": "unpriced-model"}), measured
        )["cost_usd"]
        is None
    )


def test_oversized_automatic_evaluation_reports_failure_without_failing_run(tmp_path):
    service = MetaHarness(
        adapters=[_Harness("one", "x" * 48_001)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    calls = []
    service._judge_service().model_factory = lambda _: _Model(calls)
    try:
        result = service.run(
            HarnessTask(
                task="Explain",
                workspace=str(tmp_path),
                harness="one",
                metadata={"judge": judge_config()},
            )
        )
        run = service.get_run(result.run_id)
        assert run["status"] == "succeeded"
        assert run["judgment"]["status"] == "failed"
        assert "48000 characters" in run["judgment"]["error"]
        assert calls == []
    finally:
        service.close()


class _Harness(AgentHarness):
    def __init__(self, name, answer="Paris"):
        self.name, self.answer = name, answer

    def probe(self):
        return HarnessCapabilities(name=self.name, available=True, mcp=True)

    def run(self, task, **kwargs):
        return AdapterOutcome(final_response=self.answer, cost_usd=0.02, exit_code=0)


def test_automatic_comparison_judges_every_answer_without_changing_run_cost(tmp_path):
    service = MetaHarness(
        adapters=[_Harness("one"), _Harness("two", "London")],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    calls = []
    service._judge_service().model_factory = lambda _: _Model(calls)
    try:
        task = HarnessTask(
            task="What is the capital of France?",
            workspace=str(tmp_path),
            metadata={"judge": judge_config({"reference": "Paris"})},
        )
        workflow = service.start_compare(task, ["one", "two"])
        deadline = time.monotonic() + 5
        while True:
            runs = service.list_runs()
            if len(runs) == 2 and all(
                (run.get("judgment") or {}).get("status") == "completed" for run in runs
            ):
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert sorted(run["judgment"]["score"] for run in runs) == [20, 100]
        assert len({run["judgment"]["config_hash"] for run in runs}) == 1
        assert all(run["result"]["cost_usd"] == 0.02 for run in runs)
        assert all(run["status"] == "succeeded" for run in runs)
        assert (
            service.get_orchestration(workflow["orchestration_id"])["task"]["metadata"][
                "judge"
            ]["reference"]
            == "Paris"
        )
        assert len(calls) == 2
        plain = service.run(
            HarnessTask(task="Another question", workspace=str(tmp_path), harness="one")
        )
        assert not service.get_run(plain.run_id).get("judgment")
        assert len(calls) == 2
    finally:
        service.close()


def test_judge_api_settings_manual_results_and_invalid_selection(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    service = MetaHarness(
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    service.run_store.create(_run())
    service.run_store.create(_run("b", "London"))
    service._judge_service().model_factory = lambda _: _Model([])
    provider = object()
    try:
        with patch.dict(
            state._state,
            {
                "provider": provider,
                "meta_harness": service,
                "meta_harness_provider": provider,
            },
        ), patch(
            "memorizz.llms.model_lists.latest_models", return_value=["qwen2.5:3b"]
        ):
            client = TestClient(create_app())
            assert (
                client.get("/api/harness-judge/settings").json()["config"]["provider"]
                == "ollama"
            )
            config = judge_config(
                {
                    "prompt": "Evaluate correctness against the reference.",
                    "reference": "Paris",
                    "pass_score": 90,
                }
            )
            assert (
                client.put("/api/harness-judge/settings", json=config).status_code
                == 200
            )
            assert client.get("/api/harness-judge/settings").json()["config"] == config
            assert (
                client.post(
                    "/api/harness-judgments", json={"run_ids": ["missing"]}
                ).status_code
                == 404
            )
            assert (
                client.post("/api/harness-judgments", json={"run_ids": "a"}).status_code
                == 400
            )
            assert (
                client.post(
                    "/api/harness-judgments",
                    json={"run_ids": ["a"], "config": {"pass_score": -1}},
                ).status_code
                == 400
            )
            response = client.post(
                "/api/harness-judgments", json={"run_ids": ["a", "b"]}
            )
            assert response.status_code == 200, response.text
            value = _wait(
                service._judge_service(), response.json()["judgment"]["judgment_id"]
            )
            assert value["config"] == config
            assert value["results"]["b"]["score"] == 20
            assert (
                client.get("/api/harness-runs/a").json()["run"]["judgment"]["score"]
                == 100
            )
            assert client.get("/api/harness-judgments/unknown").status_code == 404
            with patch.dict("os.environ", {"MEMORIZZ_UI_READ_ONLY": "true"}):
                read_only = TestClient(create_app())
                assert (
                    read_only.post(
                        "/api/harness-judgments", json={"run_ids": ["a"]}
                    ).status_code
                    == 403
                )
                assert (
                    read_only.put(
                        "/api/harness-judge/settings", json=config
                    ).status_code
                    == 403
                )
                assert read_only.get("/api/harness-runs/a").status_code == 200
    finally:
        service.close()
