#!/usr/bin/env python3
"""Run MemoRizz on an official raw-data MemBench smoke matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memorizz._env_io import load_layered_env  # noqa: E402
from memorizz.benchmarks.membench import (  # noqa: E402
    evaluate_membench_samples,
    load_official_raw_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MemoRizz on MemBench.")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-track", type=int, default=1)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--summary-batch-size", type=int, default=24)
    parser.add_argument("--embedding-dimensions", type=int, default=256)
    parser.add_argument(
        "--memory-backend", choices=("filesystem", "oracle"), default="filesystem"
    )
    parser.add_argument("--legacy-memory-path", action="store_true")
    return parser.parse_args()


def _revision(root: Path) -> str | None:
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
    data_root = official_root / "MemData"
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Not an official MemBench checkout: {official_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_official_raw_samples(
        data_root, samples_per_track=max(1, args.samples_per_track)
    )
    (output_dir / "memorizz-evaluation-manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "MemBench",
                "official_repository": "https://github.com/import-myself/Membench",
                "official_revision": _revision(official_root),
                "track": "official_raw_categorical_smoke",
                "paper_comparable": False,
                "sample_count": len(samples),
                "model": args.model,
                "memory_backend": args.memory_backend,
                "learning_control_plane": not args.legacy_memory_path,
                "embedding_dimensions": args.embedding_dimensions,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluate_membench_samples(
        samples,
        output_dir=output_dir,
        model_name=args.model,
        top_k=args.top_k,
        summary_batch_size=args.summary_batch_size,
        memory_backend=args.memory_backend,
        learning_control_plane=not args.legacy_memory_path,
        embedding_dimensions=args.embedding_dimensions,
    )


if __name__ == "__main__":
    main()
