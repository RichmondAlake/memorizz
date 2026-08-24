#!/usr/bin/env python3
"""Generate and officially grade a MemoRizz SWE-bench Lite prediction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memorizz._env_io import load_layered_env  # noqa: E402
from memorizz.benchmarks.swe_bench_lite import run_swe_bench_instance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded MemoRizz SWE-bench Lite evaluation."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance-id", default="sympy__sympy-20590")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument("--skip-grade", action="store_true")
    parser.add_argument(
        "--memory-backend", choices=("filesystem", "oracle"), default="filesystem"
    )
    parser.add_argument("--legacy-memory-path", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    args = parse_args()
    load_layered_env([PROJECT_ROOT / ".env"])
    official_root = args.official_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (official_root / "swebench" / "harness" / "run_evaluation.py").exists():
        raise FileNotFoundError(f"Not an official SWE-bench checkout: {official_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))

    from swebench.harness.utils import load_swebench_dataset  # type: ignore

    dataset_name = "SWE-bench/SWE-bench_Lite"
    full_dataset = load_swebench_dataset(dataset_name, "test")
    selected = [
        row for row in full_dataset if row.get("instance_id") == args.instance_id
    ]
    if not selected:
        raise ValueError(f"Instance not found in SWE-bench Lite: {args.instance_id}")
    instance = dict(selected[0])
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(dataset_name).sha
    except Exception:
        pass
    run_id = (
        "memorizz-lite-"
        + args.instance_id.lower().replace("__", "-")
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    )
    _write_json(
        output_dir / "memorizz-evaluation-manifest.json",
        {
            "benchmark": "SWE-bench Lite",
            "official_repository": "https://github.com/SWE-bench/SWE-bench",
            "official_revision": _git_revision(official_root),
            "dataset": dataset_name,
            "dataset_revision": revision,
            "dataset_test_count": len(full_dataset),
            "instance_id": args.instance_id,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "run_id": run_id,
            "memory_backend": args.memory_backend,
            "learning_control_plane": not args.legacy_memory_path,
            "paper_comparable": False,
            "non_comparability_reasons": [
                "one-instance smoke subset",
                "local Apple Silicon Docker execution is experimental upstream",
            ],
            "gold_patch_exposed_to_agent": False,
            "test_patch_exposed_to_agent": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    run_swe_bench_instance(
        instance,
        output_dir=output_dir,
        model_name=args.model,
        reasoning_effort=args.reasoning_effort,
        max_steps=args.max_steps,
        max_cost_usd=args.max_cost_usd,
        memory_backend=args.memory_backend,
        learning_control_plane=not args.legacy_memory_path,
    )

    if args.skip_grade:
        return
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        dataset_name,
        "-s",
        "test",
        "-i",
        args.instance_id,
        "-p",
        str(output_dir / "predictions.jsonl"),
        "--max_workers",
        "1",
        "-t",
        "1800",
        "-id",
        run_id,
        "--report_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(official_root), str(SRC_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(command, cwd=output_dir, env=env, check=False)
    _write_json(
        output_dir / "official-grader-process.json",
        {
            "command": command,
            "return_code": completed.returncode,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Official SWE-bench grader exited with {completed.returncode}"
        )


if __name__ == "__main__":
    main()
