"""Where fine-tuning records live.

Datasets, their examples and the record of each training run are kept in
the memory provider MemoRizz is connected to (filesystem, MongoDB, Oracle),
like the learning control plane's records: private ``SHARED_MEMORY`` rows
partitioned by ``record_type`` and scoped to their dataset, so the training
data sits with the memories it came from and moves with them.

What only makes sense on the machine that trains stays in that run's
folder under ``MEMORIZZ_HOME/finetune/runs/<run_id>``: the snapshot of the
data the run trained on, the model it produced, its live metrics
(``events.jsonl``), its worker state and a cancel flag.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .._env_io import memorizz_home
from .._json import read_json_object
from ..enums.memory_type import MemoryType
from .models import Dataset, Example, TrainingRun, utcnow

PARTITION = "memorizz-finetune"  # the agent_id these private records carry
DATASET_RECORD = "finetune_dataset"
EXAMPLE_RECORD = "finetune_example"
RUN_RECORD = "finetune_run"
SETTING_RECORD = "finetune_setting"
EVAL_RECORD = "finetune_eval"


def finetune_home() -> Path:
    return memorizz_home() / "finetune"


class FineTuneStore:
    """Fine-tuning records in a MemoRizz memory provider.

    Listings are kept in memory: a page that shows every dataset's counts
    would otherwise read every example on each load. Any write through this
    store clears them at once, so its own changes show immediately. A
    listing older than ``LIST_TTL`` seconds is still served while it is read
    again in the background, so another process's writes show on a later
    load."""

    LIST_TTL = 30.0

    def __init__(self, provider: Any, *, home: Optional[str | Path] = None):
        if provider is None:
            raise ValueError("Fine-tuning needs a memory provider to hold its data")
        self.provider = provider
        self.home = Path(home).expanduser() if home else finetune_home()
        self._cache: Dict[tuple, tuple] = {}
        self._refreshing: set = set()
        self._cache_lock = threading.Lock()

    def _forget(self, record_type: Optional[str] = None) -> None:
        """Drop cached listings of ``record_type`` (all when None): a run's
        progress, saved while it trains, keeps the examples cached."""
        with self._cache_lock:
            for key in list(self._cache):
                if record_type is None or key[0] == record_type:
                    del self._cache[key]

    # -- provider rows ------------------------------------------------------

    def _put(
        self, record_type: str, record_id: str, scope: str, value: Dict[str, Any]
    ) -> None:
        payload = {
            **value,
            "record_id": record_id,
            "record_type": record_type,
            "agent_id": PARTITION,
            "trace_memory_id": scope,
            "timestamp": value.get("updated_at") or utcnow(),
        }
        self._forget(record_type)
        document = {
            "_id": record_id,
            "memory_id": record_id,
            "content": json.dumps(payload, ensure_ascii=False, default=str),
            "memory_type": MemoryType.SHARED_MEMORY.value,
            "scope": "private",
            "owner_agent_id": PARTITION,
            "record_type": record_type,
            "agent_id": PARTITION,
            "trace_memory_id": scope,
            "status": value.get("status"),
            "timestamp": payload["timestamp"],
        }
        self.provider.store(
            {k: v for k, v in document.items() if v is not None},
            MemoryType.SHARED_MEMORY,
        )

    @staticmethod
    def _decode(row: Mapping[str, Any], record_type: str) -> Optional[Dict[str, Any]]:
        content = row.get("content")
        payload = content if isinstance(content, dict) else read_json_object(content)
        if not payload or payload.get("record_type") != record_type:
            return None
        if payload.get("agent_id") != PARTITION:
            return None
        return payload

    def _get(self, record_type: str, record_id: str) -> Optional[Dict[str, Any]]:
        try:
            row = self.provider.retrieve_by_id(str(record_id), MemoryType.SHARED_MEMORY)
        except (NotImplementedError, TypeError, ValueError):
            row = None
        if isinstance(row, Mapping):
            payload = self._decode(row, record_type)
            if payload and payload.get("record_id") == str(record_id):
                return payload
        return None

    def _indexed(self) -> bool:
        from ..memory_provider.base import MemoryProvider

        method = getattr(type(self.provider), "query_observability_records", None)
        return callable(method) and method is not MemoryProvider.__dict__.get(
            "query_observability_records"
        )

    def _list(
        self, record_type: str, scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Every record of one type (in one dataset when ``scope`` is set)."""
        key = (record_type, scope)
        with self._cache_lock:
            cached = self._cache.get(key)
            stale = cached is not None and time.monotonic() - cached[0] >= self.LIST_TTL
            refresh = stale and key not in self._refreshing
            if refresh:
                self._refreshing.add(key)
        if cached is not None:
            if refresh:
                # Serve what we have; read it again in the background.
                threading.Thread(target=self._refresh, args=(key,), daemon=True).start()
            return copy.deepcopy(cached[1])
        out = self._list_uncached(record_type, scope)
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), out)
        return copy.deepcopy(out)

    def _refresh(self, key: tuple) -> None:
        try:
            out = self._list_uncached(*key)
            with self._cache_lock:
                if key in self._cache:  # not cleared by a write meanwhile
                    self._cache[key] = (time.monotonic(), out)
        except Exception:  # the next read tries again
            pass
        finally:
            with self._cache_lock:
                self._refreshing.discard(key)

    def _list_uncached(
        self, record_type: str, scope: Optional[str]
    ) -> List[Dict[str, Any]]:
        rows: List[Mapping[str, Any]] = []
        if self._indexed():
            cursor = None
            for _ in range(1000):
                page = self.provider.query_observability_records(
                    MemoryType.SHARED_MEMORY,
                    agent_ids=None if scope else [PARTITION],
                    memory_ids=[scope] if scope else None,
                    record_type=record_type,
                    limit=1000,
                    cursor=cursor,
                )
                rows.extend(page.get("items") or [])
                cursor = page.get("next_cursor")
                if not cursor:
                    break
        else:
            rows = self.provider.list_all(MemoryType.SHARED_MEMORY) or []
        out = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            payload = self._decode(row, record_type)
            if payload and (scope is None or payload.get("trace_memory_id") == scope):
                out.append(payload)
        return out

    def _delete(self, record_id: str) -> bool:
        self._forget()
        try:
            return bool(
                self.provider.delete_by_id(str(record_id), MemoryType.SHARED_MEMORY)
            )
        except (NotImplementedError, ValueError):
            return False

    @staticmethod
    def _fields(payload: Dict[str, Any], cls: Any) -> Dict[str, Any]:
        names = cls.__dataclass_fields__
        return {k: v for k, v in payload.items() if k in names}

    # -- datasets -----------------------------------------------------------

    def save_dataset(self, dataset: Dataset) -> Dataset:
        dataset.updated_at = utcnow()
        self._put(
            DATASET_RECORD, dataset.dataset_id, dataset.dataset_id, dataset.to_dict()
        )
        return dataset

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        payload = self._get(DATASET_RECORD, dataset_id)
        return Dataset(**self._fields(payload, Dataset)) if payload else None

    def list_datasets(self) -> List[Dataset]:
        rows = self._list(DATASET_RECORD)
        datasets = [Dataset(**self._fields(row, Dataset)) for row in rows]
        return sorted(datasets, key=lambda d: d.updated_at, reverse=True)

    def delete_dataset(self, dataset_id: str) -> bool:
        for example in self.examples(dataset_id):
            self._delete(example.example_id)
        for run in self.list_runs(dataset_id):
            self._delete(run.run_id)
        for result in self.list_evals(dataset_id):
            self.delete_eval(result["eval_id"])
        return self._delete(dataset_id)

    def counts(
        self, dataset_id: str, examples: Optional[List[Example]] = None
    ) -> Dict[str, Any]:
        examples = self.examples(dataset_id) if examples is None else examples
        return self._tally(examples)

    def all_counts(self) -> Dict[str, Dict[str, Any]]:
        """Counts for every dataset from one listing of the examples (not one
        per dataset)."""
        grouped: Dict[str, List[Example]] = {}
        for row in self._list(EXAMPLE_RECORD):
            example = Example(**self._fields(row, Example))
            grouped.setdefault(example.dataset_id, []).append(example)
        return {key: self._tally(rows) for key, rows in grouped.items()}

    @staticmethod
    def _tally(examples: List[Example]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"total": len(examples), "status": {}, "split": {}}
        for e in examples:
            out["status"][e.status] = out["status"].get(e.status, 0) + 1
            if e.status == "accepted":
                out["split"][e.split] = out["split"].get(e.split, 0) + 1
        return out

    # -- examples -----------------------------------------------------------

    def add_examples(self, examples: Iterable[Example]) -> int:
        count = 0
        for example in examples:
            self._put(
                EXAMPLE_RECORD,
                example.example_id,
                example.dataset_id,
                example.to_dict(),
            )
            count += 1
        return count

    def update_example(self, example: Example) -> Example:
        example.updated_at = utcnow()
        self.add_examples([example])
        return example

    def get_example(self, example_id: str) -> Optional[Example]:
        payload = self._get(EXAMPLE_RECORD, example_id)
        return Example(**self._fields(payload, Example)) if payload else None

    def examples(
        self,
        dataset_id: str,
        *,
        status: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Example]:
        rows = self._list(EXAMPLE_RECORD, dataset_id)
        examples = [Example(**self._fields(row, Example)) for row in rows]
        examples = [
            e
            for e in examples
            if (status is None or e.status == status)
            and (split is None or e.split == split)
        ]
        return sorted(examples, key=lambda e: (e.created_at, e.example_id))

    def delete_examples(self, example_ids: Iterable[str]) -> int:
        return sum(1 for example_id in example_ids if self._delete(example_id))

    # -- runs ---------------------------------------------------------------

    def save_run(self, run: TrainingRun) -> TrainingRun:
        run.updated_at = utcnow()
        self._put(RUN_RECORD, run.run_id, run.dataset_id, run.to_dict())
        return run

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        payload = self._get(RUN_RECORD, run_id)
        return TrainingRun(**self._fields(payload, TrainingRun)) if payload else None

    def list_runs(self, dataset_id: Optional[str] = None) -> List[TrainingRun]:
        rows = self._list(RUN_RECORD, dataset_id)
        runs = [TrainingRun(**self._fields(row, TrainingRun)) for row in rows]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def delete_run(self, run_id: str) -> bool:
        return self._delete(run_id)

    # -- evaluations --------------------------------------------------------

    def save_eval(self, result: Dict[str, Any]) -> None:
        self._put(EVAL_RECORD, result["eval_id"], result["dataset_id"], result)

    def list_evals(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self._list(EVAL_RECORD, dataset_id)
        return sorted(rows, key=lambda r: r.get("created_at") or "")

    def delete_eval(self, eval_id: str) -> bool:
        return self._delete(eval_id)

    # -- settings -----------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        payload = self._get(SETTING_RECORD, f"finetune-setting-{key}")
        return payload.get("value", default) if payload else default

    def set_setting(self, key: str, value: Any) -> None:
        record_id = f"finetune-setting-{key}"
        if value is None:
            self._delete(record_id)
        else:
            self._put(
                SETTING_RECORD,
                record_id,
                PARTITION,
                {"key": key, "value": value, "updated_at": utcnow()},
            )

    # -- the run's own folder -----------------------------------------------

    def run_files(self, run_id: str) -> "RunFiles":
        return RunFiles(self.home / "runs" / run_id)


class RunFiles:
    """One run's folder: data snapshot, model, live events and worker state."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def ensure(self) -> "RunFiles":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def data(self) -> Path:
        return self.root / "data.jsonl"

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def log(self) -> Path:
        return self.root / "worker.log"

    def append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        line = json.dumps(
            {"timestamp": utcnow(), "type": event_type, "data": data}, default=str
        )
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def events(self, *, after: int = 0, limit: int = 2000) -> List[Dict[str, Any]]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if number <= after:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    break  # a line still being written
                out.append({"sequence": number, **event})
                if len(out) >= limit:
                    break
        return out

    def write_state(self, state: Dict[str, Any]) -> None:
        temporary = self.root / "state.json.tmp"
        temporary.write_text(json.dumps(state, default=str), encoding="utf-8")
        os.replace(temporary, self.root / "state.json")

    def state(self) -> Dict[str, Any]:
        try:
            return json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def request_cancel(self) -> None:
        self.ensure()
        (self.root / "cancel").write_text(utcnow(), encoding="utf-8")

    @property
    def cancel_requested(self) -> bool:
        return (self.root / "cancel").exists()


__all__ = [
    "DATASET_RECORD",
    "EVAL_RECORD",
    "EXAMPLE_RECORD",
    "FineTuneStore",
    "PARTITION",
    "RUN_RECORD",
    "RunFiles",
    "finetune_home",
]
