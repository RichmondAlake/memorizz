"""Run one training job in its own process.

``python -m memorizz.finetune.worker <run folder>`` reads the folder's
``config.json`` and ``data.jsonl`` (the snapshot taken when the run was
started), trains, and leaves the model, its live events and a final
``state.json`` in the folder. It needs no database connection: the process
that started it records the outcome in the memory provider.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

from .models import utcnow
from .store import RunFiles
from .training.common import Canceled, Reporter


def load_snapshot(files: RunFiles) -> Dict[str, List[Dict[str, Any]]]:
    data: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    with open(files.data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                data.setdefault(row.get("split") or "train", []).append(row)
    return data


def trainer_for(config: Dict[str, Any]) -> Callable:
    kind, backbone = config["kind"], config.get("backbone")
    if config.get("backend") == "unsloth":
        from .training.unsloth_backend import train as unsloth_train

        return unsloth_train
    if kind == "decision" and backbone == "lm":
        from .training.decision_lm import train

        return train
    if kind == "decision":
        from .training.decision_embed import train

        return train
    if kind == "embedding":
        from .training.retrieval import train_embedding

        return train_embedding
    if kind == "reranker":
        from .training.retrieval import train_reranker

        return train_reranker
    raise ValueError(f"No trainer for {kind}/{backbone}")


def run(folder: str | Path) -> int:
    files = RunFiles(Path(folder))
    config = json.loads(files.config.read_text(encoding="utf-8"))
    files.write_state({"phase": "starting", "pid": os.getpid(), "started_at": utcnow()})
    report = Reporter(files)
    started = utcnow()
    try:
        data = load_snapshot(files)
        report.status(
            f"{len(data['train'])} training, {len(data['validation'])} validation, "
            f"{len(data['test'])} test examples"
        )
        result = trainer_for(config)(config, data, files.root, report)
        files.write_state(
            {
                "phase": "succeeded",
                "pid": os.getpid(),
                "started_at": started,
                "finished_at": utcnow(),
                "result": result,
            }
        )
        files.append_event("finished", {"status": "succeeded"})
        return 0
    except Canceled:
        files.write_state(
            {
                "phase": "canceled",
                "pid": os.getpid(),
                "started_at": started,
                "finished_at": utcnow(),
            }
        )
        files.append_event("finished", {"status": "canceled"})
        return 1
    except Exception as exc:  # recorded for the UI, CLI and MCP
        from ..metaharness.security import redact

        error = str(redact(f"{type(exc).__name__}: {exc}"))[:2000]
        traceback.print_exc()
        files.write_state(
            {
                "phase": "failed",
                "pid": os.getpid(),
                "started_at": started,
                "finished_at": utcnow(),
                "error": error,
            }
        )
        files.append_event("finished", {"status": "failed", "error": error})
        return 1


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m memorizz.finetune.worker <run folder>", file=sys.stderr)
        return 2
    return run(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())
