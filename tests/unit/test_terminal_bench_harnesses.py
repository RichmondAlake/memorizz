"""Terminal-Bench 4.0 in Evalground: harness agents, the runner and the routes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("harbor")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.benchmarks import terminal_bench as tb  # noqa: E402
from memorizz.benchmarks import terminal_bench_runner as runner  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers import evalground  # noqa: E402

pytestmark = pytest.mark.unit


def _trial(job: Path, name: str, **fields) -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    payload = {
        "task_name": f"terminal-bench/{name.split('__')[0]}",
        "trial_name": name,
        "agent_execution": {
            "started_at": "2026-09-30T10:00:00+00:00",
            "finished_at": "2026-09-30T10:05:30+00:00",
        },
        **fields,
    }
    (trial / "result.json").write_text(json.dumps(payload))
    return trial


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "job-1"
    job.mkdir(parents=True)
    passed = _trial(
        job,
        "music-harmony__a1",
        verifier_result={"rewards": {"reward": 1.0}},
        agent_result={
            "n_input_tokens": 1_000_000,
            "n_cache_tokens": 800_000,
            "n_output_tokens": 20_000,
            "cost_usd": None,
        },
    )
    (passed / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "task"},
                    {
                        "source": "agent",
                        "message": "Tuned the voicing rules; tests pass.",
                    },
                    {"source": "agent", "message": "[tool call: exec]"},
                ]
            }
        )
    )
    (passed / "agent" / tb.MEMORY_CONTEXT_FILENAME).write_text(
        json.dumps({"source_ids": ["kb-1", "kb-2"]})
    )
    _trial(
        job,
        "cargo-flight-dispatch__b2",
        verifier_result={"rewards": {"reward": 0.0}},
        agent_result={"n_input_tokens": 10, "n_output_tokens": 2, "cost_usd": 0.5},
    )
    _trial(
        job,
        "sound-change-cascade__c3",
        exception_info={
            "exception_type": "AgentTimeoutError",
            "exception_message": "Agent ran out of time",
        },
    )
    (job / "result.json").write_text(
        json.dumps({"stats": {"evals": {"x": {"pass_at_k": {"2": 0.5}}}}})
    )
    return job


def test_a_harbor_job_becomes_an_evalground_result(tmp_path: Path) -> None:
    result = runner.summarize(
        _job(tmp_path),
        harness="codex",
        model="openai/gpt-5.6-terra",
        attempts=1,
        memory_id="terminal-bench",
    )
    assert result["overall_accuracy"] == pytest.approx(1 / 3)
    assert result["category_results"]["Media"] == {
        "total": 1,
        "passed": 1,
        "accuracy": 1.0,
    }
    assert result["category_results"]["Operations"]["passed"] == 0
    cases = {case["task"]: case for case in result["cases"]}
    harmony = cases["music-harmony"]
    assert harmony["passed"] and harmony["duration_s"] == 330.0
    assert harmony["memory_sources"] == 2
    assert harmony["summary"] == "Tuned the voicing rules; tests pass."
    # Harbor gave no cost, so it is priced at list rates: 200k new input at
    # $2/M, 800k cached at $0.20/M and 20k output at $12/M.
    assert harmony["cost_usd"] == pytest.approx(0.4 + 0.16 + 0.24)
    assert result["cost_is_estimate"] is True
    assert cases["cargo-flight-dispatch"]["cost_usd"] == 0.5
    timeout = cases["sound-change-cascade"]
    assert timeout["reward"] is None and timeout["error"].startswith(
        "AgentTimeoutError"
    )
    meta = result["metadata"]
    assert meta["benchmark"] == "terminal-bench" and meta["num_trials"] == 3
    assert meta["pass_at_k"] == {"2": 0.5}
    assert meta["external_api_cost_usd"] == result["external_api_cost_usd"]


def test_the_command_carries_memory_budgets_and_sign_in(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--harness",
            "memagent",
            "--tasks",
            "music-harmony,cargo-flight-dispatch",
            "--memory-root",
            str(tmp_path),
            "--agent-timeout-multiplier",
            "0.1",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    command = runner.build_command(args)
    joined = " ".join(command)
    assert "-a memorizz.benchmarks.terminal_bench:MemorizzHarborAgent" in joined
    assert "-m openai/gpt-5.6-terra" in joined
    assert (
        "-i terminal-bench/music-harmony -i terminal-bench/cargo-flight-dispatch"
        in joined
    )
    assert f"--ak memory_root={tmp_path}" in joined
    assert "--ak max_wall_time_seconds=2820" in joined  # 48 minutes less a minute
    assert "--ak max_cost_usd=4.0" in joined

    codex = runner.build_command(
        runner.parse_args(
            [
                "--harness",
                "codex",
                "--codex-auth",
                "chatgpt",
                "--tasks",
                "music-harmony",
                "--output",
                str(tmp_path / "o.json"),
            ]
        )
    )
    assert "CODEX_FORCE_AUTH_JSON=1" in codex and "memory_root" not in " ".join(codex)
    oracle = runner.build_command(
        runner.parse_args(
            [
                "--harness",
                "oracle",
                "--tasks",
                "music-harmony",
                "--memory-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "o.json"),
            ]
        )
    )
    assert "-m" not in oracle and "--ak" not in oracle

    for bad in (
        ["--harness", "codex", "--tasks", "fp8-rmsnorm-gemm"],
        ["--harness", "codex", "--tasks", "no-such-task"],
        [
            "--harness",
            "memagent",
            "--model",
            "anthropic/claude-opus-5-5",
            "--tasks",
            "music-harmony",
        ],
        ["--harness", "codex", "--tasks", "music-harmony", "--learn"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args([*bad, "--output", str(tmp_path / "o.json")])


def test_lessons_come_back_for_other_tasks_but_never_their_own(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    cases = [
        {
            "task": "music-harmony",
            "category": "Media",
            "passed": True,
            "reward": 1.0,
            "error": None,
            "summary": "Check the voice-leading tests before writing chords.",
        },
        {
            "task": "sound-change-cascade",
            "category": "Science",
            "passed": False,
            "reward": None,
            "error": "AgentTimeoutError: x",
            "summary": "",
        },
    ]
    saved = runner.store_lessons(
        cases,
        memory_root=str(root),
        memory_id="terminal-bench",
        user_id="terminal-bench",
        harness="codex",
        model="openai/gpt-5.6-terra",
        job="j1",
    )
    assert saved == 1  # the timed-out trial taught nothing

    own, report = tb.build_memory_context(
        "Write chords that pass the voice-leading tests.",
        memory_root=root,
        task_name="music-harmony",
    )
    assert own == "Write chords that pass the voice-leading tests."
    assert report["excluded_count"] == 1 and report["source_ids"] == []

    other, report = tb.build_memory_context(
        "Write chords that pass the voice-leading tests.",
        memory_root=root,
        task_name="cargo-flight-dispatch",
    )
    assert "Terminal-Bench task music-harmony (Media) passed" in other
    assert other.endswith(
        "--- Task ---\nWrite chords that pass the voice-leading tests."
    )
    assert len(report["source_ids"]) == 1

    baseline, report = tb.build_memory_context("Do it.", memory_root=None)
    assert baseline == "Do it." and report["memory_root"] is None


def test_container_agents_name_themselves_and_write_their_memory_report(
    tmp_path: Path,
) -> None:
    agent = tb.MemorizzCodexAgent(
        logs_dir=tmp_path / "agent",
        model_name="openai/gpt-5.6-terra",
        memory_root=str(tmp_path / "memory"),
        memory_id="m1",
    )
    assert agent.name() == "codex"  # Harbor's own install behaviour stays
    assert agent.to_agent_info().name == "memorizz-codex"
    text = agent._with_memory(
        "Fix it.", SimpleNamespace(environment_name="music-harmony")
    )
    assert text == "Fix it."  # an empty memory adds nothing
    report = json.loads((tmp_path / "agent" / tb.MEMORY_CONTEXT_FILENAME).read_text())
    assert report["harness"] == "codex" and report["memory_id"] == "m1"
    assert report["task_name"] == "music-harmony"
    claude = tb.MemorizzClaudeCodeAgent(
        logs_dir=tmp_path / "c", model_name="anthropic/claude-opus-5-5"
    )
    assert (
        claude.to_agent_info().name == "memorizz-claude-code"
        and claude._memory_root is None
    )
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        tb.MemorizzCodexAgent(logs_dir=tmp_path / "x", model_name="openai/m", bogus=1)


# ------------------------------------------------------------------ routes


@pytest.fixture()
def client(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "store", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"path": str(tmp_path / "store")},
            "read_only": False,
        },
    ):
        yield TestClient(create_app(), follow_redirects=False), provider
    provider.close()


def _ready(**keys):
    return {
        "harbor": {"ok": True, "version": "0.23.0", "detail": "Harbor 0.23.0"},
        "docker": {"ok": True, "detail": "Docker 28"},
        "keys": {"openai": False, "anthropic": False, "codex_chatgpt": False, **keys},
        "memory_root": "/stores/ui",
    }


def test_the_start_route_checks_what_the_run_needs(client) -> None:
    app, _ = client
    post = lambda **data: app.post(
        "/evalground/terminal-bench/runs", data=data
    )  # noqa: E731
    with patch.object(evalground, "_terminal_bench_status", return_value=_ready()):
        assert (
            "ANTHROPIC_API_KEY"
            in post(harness="claude-code", tasks="music-harmony").json()["error"]
        )
        assert (
            "OPENAI_API_KEY"
            in post(harness="codex", tasks="music-harmony").json()["error"]
        )
        assert "GPU" in post(harness="oracle", tasks="jax-speedrun-gpu").json()["error"]
        assert "at least one" in post(harness="oracle").json()["error"]
    down = _ready()
    down["docker"] = {"ok": False, "detail": "Docker is installed but not running."}
    with patch.object(evalground, "_terminal_bench_status", return_value=down):
        assert (
            "Start Docker"
            in post(harness="oracle", tasks="music-harmony").json()["error"]
        )

    # Patch the module's reference, not threading.Thread itself: the route
    # checks the machine through asyncio.to_thread, which needs real threads.
    with patch.object(
        evalground, "_terminal_bench_status", return_value=_ready(anthropic=True)
    ), patch.object(evalground, "threading") as threading_module:
        thread = threading_module.Thread
        response = app.post(
            "/evalground/terminal-bench/runs",
            data={
                "harness": "claude-code",
                "tasks": ["music-harmony", "cargo-flight-dispatch"],
                "memory": "on",
                "memory_id": "tb-lessons",
                "learn": "on",
                "attempts": "2",
            },
        )
    assert response.status_code == 200, response.text
    thread.return_value.start.assert_called_once()
    run = evalground._get_eval_run_snapshot(response.json()["run_id"])
    assert run["benchmark"] == "terminal-bench" and run["harness"] == "claude-code"
    assert run["model"] == "anthropic/claude-opus-5-5"
    assert run["memory_root"] == "/stores/ui" and run["learn"] is True
    command = evalground._terminal_bench_command(
        run, Path("/r/out.json"), {"results_dir": Path("/r")}
    )
    assert command[1:4] == ["-u", "-m", "memorizz.benchmarks.terminal_bench_runner"]
    joined = " ".join(command)
    assert "--tasks music-harmony,cargo-flight-dispatch" in joined
    assert "--memory-root /stores/ui --memory-id tb-lessons --learn" in joined
    assert "--attempts 2" in joined and "--model anthropic/claude-opus-5-5" in joined


def test_a_finished_run_shows_its_tasks_and_joins_the_run_library(
    client, tmp_path: Path
) -> None:
    app, _ = client
    result = runner.summarize(
        _job(tmp_path),
        harness="codex",
        model="openai/gpt-5.6-terra",
        attempts=1,
        memory_id=None,
    )
    run_id = evalground._create_eval_run(
        {
            "benchmark": "terminal-bench",
            "num_samples": 3,
            "model": "openai/gpt-5.6-terra",
            "evaluation_mode": "harness",
        }
    )
    evalground._update_eval_run(
        run_id,
        harness="codex",
        tasks=["music-harmony"],
        status="completed",
        eval_results=result,
    )
    page = app.get(f"/evalground?run_id={run_id}")
    assert page.status_code == 200, page.text
    assert 'Terminal-Bench 4.0 · <span class="mono">codex</span>' in page.text
    assert "Tasks passed" in page.text and "sound-change-cascade" in page.text
    assert 'id="terminal-bench"' in page.text and "Smoke set · 5 tasks" in page.text

    library = app.get("/evalground/run-library").json()["runs"]
    row = next(r for r in library if r["id"] == run_id)
    assert row["kind"] == "harness" and row["quality_min"] == pytest.approx(
        1 / 3, abs=1e-3
    )
    assert row["name"] == "codex · Terminal-Bench 4.0"


def test_stopping_gives_harbor_time_to_remove_containers() -> None:
    process = SimpleNamespace(pid=4321, wait=lambda timeout=None: 0)
    sent = []
    with patch.object(
        evalground.os, "killpg", lambda pid, sig: sent.append((pid, sig))
    ):
        evalground._stop_terminal_bench("r1", process)
    assert sent == [(4321, evalground.signal.SIGINT)]


def test_the_bundled_catalog_matches_the_dataset() -> None:
    catalog = runner.task_catalog()
    assert catalog["version"] == "4.0.0" and len(catalog["tasks"]) == 66
    names = {task["name"] for task in catalog["tasks"]}
    assert set(catalog["smoke"]) <= names
    assert {t["name"] for t in catalog["tasks"] if t["gpus"]} == {
        "fp8-rmsnorm-gemm",
        "jax-speedrun-gpu",
        "math-eval-grader",
    }
    assert len(runner.runnable_tasks(catalog)) == 63
