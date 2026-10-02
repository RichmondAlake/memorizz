"""Run a harness on Terminal-Bench 4.0 with Harbor and write an Evalground result.

    python -m memorizz.benchmarks.terminal_bench_runner --harness codex \
        --tasks music-harmony,cargo-flight-dispatch --output results.json

Harbor runs each task in its own Docker container and grades it with the
task's verifier. ``memagent`` drives the container from the host; ``codex``
and ``claude-code`` are installed inside it; ``oracle`` runs the reference
solutions, which checks Docker and grading for free. With ``--memory-root``
each harness gets MemoRizz memory in front of the task, and ``--learn`` saves
one lesson per finished trial to that memory for later runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DATASET = "terminal-bench/terminal-bench@4.0.0"
TASK_ORG = "terminal-bench"
AGENTS = {
    "memagent": "memorizz.benchmarks.terminal_bench:MemorizzHarborAgent",
    "codex": "memorizz.benchmarks.terminal_bench:MemorizzCodexAgent",
    "claude-code": "memorizz.benchmarks.terminal_bench:MemorizzClaudeCodeAgent",
    "oracle": "oracle",
}
DEFAULT_MODELS = {
    "memagent": "openai/gpt-5.6-terra",
    "codex": "openai/gpt-5.6-terra",
    "claude-code": "anthropic/claude-opus-5-5",
}
# Every Terminal-Bench 4.0 task allows its agent eight hours.
TASK_AGENT_TIMEOUT_SECONDS = 28_800
LESSON_CHARS = 1_200


# ------------------------------------------------------------------ catalog


def task_catalog() -> Dict[str, Any]:
    """The bundled Terminal-Bench 4.0 task list with categories and resources."""
    source = resources.files("memorizz.benchmarks") / "data" / "terminal_bench_4.json"
    return json.loads(source.read_text(encoding="utf-8"))


def runnable_tasks(catalog: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Tasks that run without a GPU."""
    catalog = catalog or task_catalog()
    return [task for task in catalog["tasks"] if not task.get("gpus")]


def _env_value(name: str) -> str:
    """A key from the process or ~/.memorizz/.env, without exporting it."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from dotenv import dotenv_values

        from memorizz._env_io import resolve_env_file

        path = resolve_env_file()
        if path.exists():
            return str(dotenv_values(path).get(name) or "")
    except Exception:
        pass
    return ""


def codex_auth_file() -> Path:
    """Where a ChatGPT sign-in for Codex is kept."""
    home = os.environ.get("CODEX_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".codex") / "auth.json"


def environment_status(memory_root: Optional[str] = None) -> Dict[str, Any]:
    """Whether this machine can run Terminal-Bench 4.0: Harbor 0.23 or newer,
    a running Docker, model keys and a Codex sign-in. Spends nothing."""
    harbor_version = None
    try:
        from importlib.metadata import version

        harbor_version = version("harbor")
    except Exception:
        pass
    docker: Dict[str, Any] = {"ok": False, "detail": "Docker is not installed."}
    try:
        probe = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "{{.ServerVersion}} {{.NCPU}} {{.MemTotal}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode == 0:
            server, cpus, memory = (probe.stdout.split() + ["", "", ""])[:3]
            docker = {
                "ok": True,
                "detail": f"Docker {server}",
                "cpus": int(cpus) if cpus.isdigit() else None,
                "memory_gb": round(int(memory) / 2**30, 1)
                if memory.isdigit()
                else None,
            }
        else:
            docker = {"ok": False, "detail": "Docker is installed but not running."}
    except (OSError, subprocess.TimeoutExpired):
        pass
    harbor_ok = bool(harbor_version) and tuple(
        int(part) for part in harbor_version.split(".")[:2] if part.isdigit()
    ) >= (0, 23)
    return {
        "harbor": {
            "ok": harbor_ok,
            "version": harbor_version,
            "detail": (
                f"Harbor {harbor_version}"
                if harbor_ok
                else "Install the benchmark extra: pip install 'memorizz[terminal-bench]'"
            ),
        },
        "docker": docker,
        "keys": {
            "openai": bool(_env_value("OPENAI_API_KEY")),
            "anthropic": bool(_env_value("ANTHROPIC_API_KEY")),
            "codex_chatgpt": codex_auth_file().exists(),
        },
        "memory_root": memory_root,
    }


def _short(task_name: str) -> str:
    return str(task_name or "").rsplit("/", 1)[-1]


# ------------------------------------------------------------------ command


def harbor_command() -> List[str]:
    beside = Path(sys.executable).with_name("harbor")
    if beside.exists():
        return [str(beside)]
    found = shutil.which("harbor")
    if found:
        return [found]
    return [sys.executable, "-c", "from harbor.cli.main import app; app()"]


def build_command(args: argparse.Namespace) -> List[str]:
    harness = args.harness
    command = [
        *harbor_command(),
        "run",
        "-d",
        DATASET,
        "-a",
        AGENTS[harness],
        "-e",
        "docker",
        "-n",
        str(args.n_concurrent),
        "-k",
        str(args.attempts),
        "-o",
        str(args.jobs_dir),
        "--job-name",
        args.job_name,
        "--agent-timeout-multiplier",
        str(args.agent_timeout_multiplier),
        "-y",
    ]
    if harness != "oracle":
        command.extend(["-m", args.model])
    for task in args.tasks:
        command.extend(["-i", f"{TASK_ORG}/{task}"])
    agent_kwargs: List[str] = []
    if args.memory_root and harness != "oracle":
        agent_kwargs += [
            f"memory_root={args.memory_root}",
            f"memory_id={args.memory_id}",
            f"user_id={args.user_id}",
        ]
    if harness == "memagent":
        # The MemAgent paces itself to finish before Harbor's hard stop.
        budget = TASK_AGENT_TIMEOUT_SECONDS * args.agent_timeout_multiplier
        agent_kwargs += [
            f"max_wall_time_seconds={max(120, int(budget) - 60)}",
            f"max_cost_usd={args.max_cost_per_task}",
        ]
    for value in agent_kwargs:
        command.extend(["--ak", value])
    if harness == "codex" and args.codex_auth == "chatgpt":
        command.extend(["--ae", "CODEX_FORCE_AUTH_JSON=1"])
    if args.dry_run:
        command.append("--dry-run")
    return command


def _redacted(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def run_harbor(command: List[str]) -> int:
    """Run Harbor, passing its output through line by line."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"},
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line.strip():
                print(line, flush=True)
        return process.wait()
    except KeyboardInterrupt:
        # Evalground stops a run with SIGINT: Harbor cancels its trials and
        # takes their containers down before it exits.
        print("Stopping Harbor and its task containers…", flush=True)
        try:
            return process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


# ------------------------------------------------------------------ results


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _seconds(timing: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(timing, dict):
        return None
    try:
        start = datetime.fromisoformat(str(timing["started_at"]))
        end = datetime.fromisoformat(str(timing["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return round((end - start).total_seconds(), 1)


def _final_message(agent_dir: Path) -> str:
    """The agent's last words in the trial: its summary of what it did."""
    trajectory = _read_json(agent_dir / "trajectory.json") or {}
    for step in reversed(trajectory.get("steps") or []):
        if step.get("source") != "agent":
            continue
        message = step.get("message")
        if isinstance(message, list):
            message = " ".join(
                str(part.get("text") or "")
                for part in message
                if isinstance(part, dict)
            )
        if (
            message
            and str(message).strip()
            and not str(message).startswith("[tool call")
        ):
            return str(message).strip()
    report = _read_json(agent_dir / "memorizz-run.json") or {}
    return str(report.get("response") or "").strip()


def _list_cost(
    model: Optional[str], tokens: Dict[str, Optional[int]]
) -> Optional[float]:
    """A trial's cost at list rates, from the tokens Harbor summed over its calls.

    The sum spans many requests, so no single-request surcharge (long-context
    rates) applies. Harbor reports cache reads but not cache writes; writes
    are priced as plain input, so the estimate can run slightly low.
    """
    if not model or "/" not in model or tokens.get("input_tokens") is None:
        return None
    from memorizz.observability.pricing import DEFAULT_PRICING

    provider, name = model.split("/", 1)
    card = DEFAULT_PRICING.cards.get((provider.lower(), name, "default"))
    if card is None:
        return None
    inputs = int(tokens.get("input_tokens") or 0)
    cached = min(int(tokens.get("cached_tokens") or 0), inputs)
    outputs = int(tokens.get("output_tokens") or 0)
    cost = (
        (inputs - cached) * float(card.input_per_million)
        + cached * float(card.cached_input_per_million)
        + outputs * float(card.output_per_million)
    ) / 1_000_000
    return cost


def read_trials(job_dir: Path) -> List[Dict[str, Any]]:
    trials = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        data = _read_json(result_path)
        if not data or "task_name" not in data:
            continue
        data["_dir"] = str(result_path.parent)
        trials.append(data)
    return trials


def summarize(
    job_dir: Path,
    *,
    harness: str,
    model: Optional[str],
    attempts: int,
    memory_id: Optional[str],
    catalog: Optional[Dict[str, Any]] = None,
    harbor_exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Turn a Harbor job folder into an Evalground results document."""
    catalog = catalog or task_catalog()
    by_name = {task["name"]: task for task in catalog["tasks"]}
    cases: List[Dict[str, Any]] = []
    estimated_cost = False
    for trial in read_trials(job_dir):
        task = _short(trial["task_name"])
        meta = by_name.get(task, {})
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        agent = trial.get("agent_result") or {}
        tokens = {
            "input_tokens": agent.get("n_input_tokens"),
            "cached_tokens": agent.get("n_cache_tokens"),
            "output_tokens": agent.get("n_output_tokens"),
        }
        cost = agent.get("cost_usd")
        if cost is None and harness != "oracle":
            cost = _list_cost(model, tokens)
            estimated_cost = estimated_cost or cost is not None
        error = trial.get("exception_info") or None
        agent_dir = Path(trial["_dir"]) / "agent"
        memory = _read_json(agent_dir / "memorizz-context.json") or {}
        cases.append(
            {
                "task": task,
                "trial": trial.get("trial_name"),
                "category": meta.get("category") or "Other",
                "subcategory": meta.get("subcategory"),
                "difficulty": meta.get("difficulty"),
                "reward": None if reward is None else float(reward),
                "passed": reward is not None and float(reward) >= 1.0,
                "duration_s": _seconds(trial.get("agent_execution"))
                or _seconds(
                    {
                        "started_at": trial.get("started_at"),
                        "finished_at": trial.get("finished_at"),
                    }
                ),
                "cost_usd": None if cost is None else round(float(cost), 6),
                **tokens,
                "memory_sources": len(memory.get("source_ids") or []),
                "error": (
                    f"{error.get('exception_type')}: "
                    f"{str(error.get('exception_message') or '')[:300]}"
                    if error
                    else None
                ),
                "summary": _final_message(agent_dir)[:LESSON_CHARS],
            }
        )
    cases.sort(key=lambda case: (case["category"], case["task"], case["trial"] or ""))
    passed = sum(1 for case in cases if case["passed"])
    categories: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        row = categories.setdefault(case["category"], {"total": 0, "passed": 0})
        row["total"] += 1
        row["passed"] += int(case["passed"])
    for row in categories.values():
        row["accuracy"] = row["passed"] / row["total"] if row["total"] else 0.0
    costs = [case["cost_usd"] for case in cases if case["cost_usd"] is not None]
    job = _read_json(job_dir / "result.json") or {}
    pass_at_k: Dict[str, float] = {}
    for stats in ((job.get("stats") or {}).get("evals") or {}).values():
        pass_at_k.update(
            {str(k): float(v) for k, v in (stats.get("pass_at_k") or {}).items()}
        )
    total_cost = (
        round(sum(costs), 6) if costs else (0.0 if harness == "oracle" else None)
    )
    return {
        "overall_accuracy": passed / len(cases) if cases else 0.0,
        "category_results": categories,
        "external_api_cost_usd": total_cost,
        "cost_is_estimate": estimated_cost,
        "cases": cases,
        "metadata": {
            "benchmark": "terminal-bench",
            "dataset": DATASET,
            "evaluation_mode": "harness",
            "harness": harness,
            "model": model,
            "num_samples": len({case["task"] for case in cases}),
            "num_trials": len(cases),
            "passed": passed,
            "n_attempts": attempts,
            "pass_at_k": pass_at_k,
            "memory_id": memory_id,
            "external_api_cost_usd": total_cost,
            "job_dir": str(job_dir),
            "harbor_exit_code": harbor_exit_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ------------------------------------------------------------------ memory


def store_lessons(
    cases: List[Dict[str, Any]],
    *,
    memory_root: str,
    memory_id: str,
    user_id: str,
    harness: str,
    model: Optional[str],
    job: str,
) -> int:
    """Save one lesson per finished trial so later runs can recall them."""
    from memorizz.enums.memory_type import MemoryType
    from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(memory_root).expanduser(), lazy_vector_indexes=True
        )
    )
    saved = 0
    try:
        for case in cases:
            if case["error"] and case["reward"] is None:
                continue  # the trial never ran far enough to teach anything
            outcome = "passed" if case["passed"] else "failed"
            text = (
                f"Terminal-Bench task {case['task']} ({case['category']}) {outcome} "
                f"with {harness}"
                + (f" on {model}" if model else "")
                + ".\n"
                + (
                    f"What the agent reported: {case['summary']}"
                    if case["summary"]
                    else ""
                )
            ).strip()
            try:
                provider.store(
                    {
                        "content": text,
                        "namespace": "terminal-bench",
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "task_name": case["task"],
                        "reward": case["reward"],
                        "harness": harness,
                        "model": model,
                        "job": job,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    memory_store_type=MemoryType.KNOWLEDGE_BASE,
                )
                saved += 1
            except Exception as exc:
                print(
                    f"Could not save the lesson for {case['task']}: {exc}", flush=True
                )
    finally:
        try:
            provider.close()
        except Exception:
            pass
    return saved


# ------------------------------------------------------------------ main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--harness", choices=sorted(AGENTS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=0.1)
    parser.add_argument("--max-cost-per-task", type=float, default=4.0)
    parser.add_argument("--memory-root", default=None)
    parser.add_argument("--memory-id", default="terminal-bench")
    parser.add_argument("--user-id", default="terminal-bench")
    parser.add_argument("--learn", action="store_true")
    parser.add_argument(
        "--codex-auth", choices=("api_key", "chatgpt"), default="api_key"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs-dir", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    catalog = task_catalog()
    known = {task["name"]: task for task in catalog["tasks"]}
    args.tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown = [name for name in args.tasks if name not in known]
    if unknown:
        parser.error("unknown Terminal-Bench 4.0 tasks: " + ", ".join(unknown))
    gpu = [name for name in args.tasks if known[name].get("gpus")]
    if gpu:
        parser.error("these tasks need a GPU: " + ", ".join(gpu))
    if not args.tasks:
        parser.error("choose at least one task")
    if args.harness != "oracle":
        args.model = args.model or DEFAULT_MODELS[args.harness]
        if "/" not in args.model:
            parser.error("--model must be provider/model, e.g. openai/gpt-5.6-terra")
    else:
        args.model = None
    if args.harness == "memagent" and not args.model.startswith("openai/"):
        parser.error("memagent on Terminal-Bench supports openai/<model> models")
    if not 0 < args.agent_timeout_multiplier <= 1:
        parser.error("--agent-timeout-multiplier must be above 0 and at most 1")
    args.n_concurrent = max(1, args.n_concurrent)
    args.attempts = max(1, args.attempts)
    output = Path(args.output).expanduser()
    args.jobs_dir = (
        Path(args.jobs_dir).expanduser()
        if args.jobs_dir
        else output.parent / "terminal-bench-jobs"
    )
    args.job_name = args.job_name or (
        f"memorizz-{args.harness}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if args.learn and not args.memory_root:
        parser.error("--learn needs --memory-root")
    return args


def _load_keys() -> None:
    """Model keys from ~/.memorizz/.env, without overriding the environment.

    Only MemoRizz's own file: a project .env in the working folder may hold
    unrelated credentials that have no business in a benchmark.
    """
    try:
        from dotenv import load_dotenv

        from memorizz._env_io import resolve_env_file
    except ImportError:  # pragma: no cover - python-dotenv is a base dependency
        return
    path = resolve_env_file()
    if path.exists():
        load_dotenv(path, override=False)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _load_keys()
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args)
    print(
        f"Terminal-Bench 4.0 · {args.harness}"
        + (f" · {args.model}" if args.model else "")
        + f" · {len(args.tasks)} task(s) × {args.attempts} attempt(s)"
        + (f" · memory {args.memory_id}" if args.memory_root else " · no memory"),
        flush=True,
    )
    print("$ " + _redacted(command), flush=True)
    started = time.monotonic()
    code = run_harbor(command)
    if args.dry_run:
        print(f"Dry run finished with exit code {code}.", flush=True)
        return code
    job_dir = args.jobs_dir / args.job_name
    if not read_trials(job_dir):
        print(
            f"Harbor exited with code {code} and recorded no trials in {job_dir}.",
            flush=True,
        )
        return code or 1
    result = summarize(
        job_dir,
        harness=args.harness,
        model=args.model,
        attempts=args.attempts,
        memory_id=args.memory_id if args.memory_root else None,
        harbor_exit_code=code,
    )
    result["metadata"]["total_processing_time"] = round(time.monotonic() - started, 1)
    if args.learn:
        saved = store_lessons(
            result["cases"],
            memory_root=args.memory_root,
            memory_id=args.memory_id,
            user_id=args.user_id,
            harness=args.harness,
            model=args.model,
            job=args.job_name,
        )
        result["metadata"]["lessons_saved"] = saved
        print(f"Saved {saved} lesson(s) to memory {args.memory_id}.", flush=True)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    temporary.replace(output)
    meta = result["metadata"]
    cost = result["external_api_cost_usd"]
    print(
        f"Passed {meta['passed']} of {meta['num_trials']} trial(s) "
        f"({result['overall_accuracy'] * 100:.0f}%)"
        + (f", ${cost:.2f}" if cost is not None else "")
        + ".",
        flush=True,
    )
    if code:
        print(
            f"Harbor exited with code {code}; the results above are what it recorded.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
