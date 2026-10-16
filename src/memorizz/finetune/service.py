"""Fine-tuning in MemoRizz: one API for the SDK, CLI, MCP server and UI.

    from memorizz.finetune import FineTuning
    ft = FineTuning(memory_provider)
    dataset = ft.create_dataset("decision", "memory_routing", name="Routing")
    ft.generate(dataset.dataset_id, count=60)          # teacher LLM, checked blind
    ft.import_memories(dataset.dataset_id)              # your stored memories, labelled by store
    ft.review([...], "accepted")                         # mark what to train on
    run = ft.start_run(dataset.dataset_id, base_model="Qwen/Qwen3-Embedding-0.6B")
    ft.run(run.run_id)                                   # progress, metrics before/after
    ft.decide(run.run_id, "Priya prefers calls on Tuesdays")
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..enums.memory_type import MemoryType
from .bases import base_models, default_base, describe_format, resolve_base
from .formats import default_format, render, to_jsonl, unsloth_bundle
from .models import (
    FINAL_RUN_STATES,
    SPLITS,
    Dataset,
    Example,
    ExampleStatus,
    RunStatus,
    TrainingRun,
    utcnow,
)
from .store import PARTITION, FineTuneStore
from .synthetic import (
    Teacher,
    default_teacher,
    generate_corpus,
    generate_decision_examples,
    generate_pairs,
    make_verifier,
    split_for,
    stratified_splits,
)
from .tasks import (
    MEMORY_ROUTING_OPTIONS,
    TASKS,
    answer_for,
    only_question,
    options,
    task_template,
)

logger = logging.getLogger(__name__)

Progress = Optional[Callable[[int, int, str], None]]
MIN_TRAIN, MIN_HELD_OUT = 8, 2
# Retrieval and reranking evaluations recorded since scoring matched the
# trainer's (whole-dataset corpus; reranker pools with 9 sampled memories;
# an untrained base reads as many tokens as training let it; ties averaged;
# per-query scores kept for paired significance tests).
# Older ones used a smaller candidate pool and are not compared with these.
RANKING_METHOD = "trainer-v3"
USES = {
    "memory_routing": "decision",
    "relevance": "decision",
    "embedding": "embedding",
    "reranker": "reranker",
}
# Fields that hold a memory's text, per store, in the order to try.
_TEXT_FIELDS = (
    "content",
    "text",
    "summary",
    "description",
    "document",
    "body",
    "skill_md",
    "docstring",
)


class FineTuningError(ValueError):
    """A request the fine-tuning API cannot carry out, said plainly."""


def _memory_text(store: str, row: Dict[str, Any]) -> str:
    """A stored memory as the state a router would see when it was written.

    A tool is its name, signature and description: its description alone
    reads like a fact and routes to the knowledge base."""
    if store == "toolbox" and row.get("name"):
        signature = str(row.get("signature") or "").strip()
        description = str(row.get("description") or row.get("docstring") or "").strip()
        return f"Tool {row['name']}{signature}: {description}"
    if store == "personas":
        parts = [
            row.get("name"),
            row.get("role"),
            row.get("background"),
            row.get("goals"),
        ]
        return " · ".join(str(p) for p in parts if p)
    if store == "entity_memory":
        attrs = row.get("attributes")
        if isinstance(attrs, (list, dict)) and attrs:
            return f"{row.get('name') or row.get('entity_name') or ''}: {json.dumps(attrs, ensure_ascii=False)[:1500]}"
    if store == "conversation_memory" and row.get("content"):
        return f"{row.get('role') or 'message'}: {row['content']}"
    for field in _TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    name = row.get("name")
    return str(name) if name else ""


def environment() -> Dict[str, Any]:
    """What this machine can train with."""
    import importlib.util

    info: Dict[str, Any] = {"packages": {}, "missing": []}
    for name, module in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("unsloth", "unsloth"),
    ):
        found = importlib.util.find_spec(module) is not None
        info["packages"][name] = found
        if not found and name != "unsloth":
            info["missing"].append(name)
    info["device"] = None
    if info["packages"]["torch"]:
        from .training.common import pick_device

        info["device"] = pick_device()
    from .training.unsloth_backend import available

    info["unsloth"] = (
        available()
        if info["packages"]["torch"]
        else {"available": False, "reason": "PyTorch is not installed"}
    )
    info["ready"] = not info["missing"]
    info["install"] = 'pip install "memorizz[finetune]"'
    return info


_CACHED_MODELS: Dict[str, Any] = {"at": 0.0, "models": {}}


def _cached_models() -> Dict[str, Dict[str, Any]]:
    """Base models already in the Hugging Face cache, with their size
    (looked up at most once a minute)."""
    import time as _time

    if _time.monotonic() - _CACHED_MODELS["at"] < 60:
        return _CACHED_MODELS["models"]
    models: Dict[str, Dict[str, Any]] = {}
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type == "model":
                models[repo.repo_id] = {
                    "on_disk": True,
                    "size_mb": round(repo.size_on_disk / 1e6, 1),
                }
    except Exception:  # no cache yet, or huggingface_hub missing
        pass
    _CACHED_MODELS.update(at=_time.monotonic(), models=models)
    return models


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class FineTuning:
    def __init__(self, provider: Any, *, home: Optional[str] = None):
        self.store = FineTuneStore(provider, home=home)
        self.provider = provider
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- catalogue -----------------------------------------------------------

    @staticmethod
    def tasks() -> List[Dict[str, Any]]:
        return [
            {
                "kind": t.kind,
                "task": t.task,
                "title": t.title,
                "description": t.description,
                "metrics": list(t.metrics),
                "spec": t.default_spec(),
            }
            for t in TASKS.values()
        ]

    @staticmethod
    def base_models(kind: Optional[str] = None) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in base_models(kind)]

    @staticmethod
    def environment() -> Dict[str, Any]:
        return environment()

    # -- datasets ------------------------------------------------------------

    def create_dataset(
        self,
        kind: str,
        task: str,
        *,
        name: str = "",
        spec: Optional[Dict[str, Any]] = None,
        description: str = "",
        owner: str = "",
    ) -> Dataset:
        try:
            template = task_template(kind, task)
        except ValueError as exc:
            raise FineTuningError(str(exc)) from None
        spec = dict(spec or template.default_spec())
        self._validate_spec(kind, spec)
        dataset = Dataset(
            name=name or template.title,
            kind=kind,
            task=task,
            spec=spec,
            description=description,
            owner=owner,
        )
        return self.store.save_dataset(dataset)

    @staticmethod
    def _validate_spec(kind: str, spec: Dict[str, Any]) -> None:
        if kind == "decision":
            try:
                only_question(spec)
            except ValueError as exc:
                raise FineTuningError(str(exc)) from None

    def datasets(self, *, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every dataset with its example counts and a summary of its runs
        (two listings in all, whatever the number of datasets)."""
        empty = {"total": 0, "status": {}, "split": {}}
        counts = self.store.all_counts()
        runs: Dict[str, List[Dict[str, Any]]] = {}
        for run in self.runs():
            runs.setdefault(run["dataset_id"], []).append(run)
        out = []
        for d in self.store.list_datasets():
            if owner is not None and d.owner != owner:
                continue
            mine = runs.get(d.dataset_id, [])
            done = [r for r in mine if r["status"] == RunStatus.SUCCEEDED.value]
            latest = done[0] if done else None
            metric = self._primary_metric(d.kind, d.task)
            out.append(
                {
                    **d.to_dict(),
                    "counts": counts.get(d.dataset_id, empty),
                    "runs": {
                        "total": len(mine),
                        "succeeded": len(done),
                        "running": sum(
                            1
                            for r in mine
                            if r["status"]
                            in (RunStatus.RUNNING.value, RunStatus.QUEUED.value)
                        ),
                        "latest": {
                            "run_id": latest["run_id"],
                            "label": latest["label"],
                            "metric": metric,
                            "before": (latest["metrics_before"] or {}).get(metric),
                            "after": (latest["metrics_after"] or {}).get(metric),
                        }
                        if latest
                        else None,
                    },
                }
            )
        return out

    @staticmethod
    def _primary_metric(kind: str, task: str) -> str:
        return "accuracy" if kind == "decision" and task != "relevance" else "ndcg@10"

    def owner_of(self, dataset_id: str) -> str:
        return self._dataset(dataset_id).owner

    def dataset(self, dataset_id: str) -> Dict[str, Any]:
        dataset = self._dataset(dataset_id)
        examples = self.store.examples(dataset_id)
        out = {**dataset.to_dict(), "counts": self.store.counts(dataset_id, examples)}
        out["labels"] = self._label_counts(dataset, examples)
        out["sources"] = {}
        for e in examples:
            out["sources"][e.source] = out["sources"].get(e.source, 0) + 1
        return out

    def _dataset(self, dataset_id: str) -> Dataset:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise FineTuningError(f"No dataset {dataset_id}")
        return dataset

    def _label_counts(
        self, dataset: Dataset, examples: List[Example]
    ) -> Dict[str, Dict[str, int]]:
        """Examples per label and status (decision datasets)."""
        if dataset.kind != "decision":
            return {}
        qid, question = only_question(dataset.spec)
        from .tasks import gold_key

        counts: Dict[str, Dict[str, int]] = {key: {} for key, _ in options(question)}
        for e in examples:
            try:
                key = gold_key(question, e.label[qid])
            except (KeyError, TypeError, ValueError):
                continue
            bucket = counts.setdefault(key, {})
            bucket[e.status] = bucket.get(e.status, 0) + 1
        return counts

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        test_set: Optional[bool] = None,
    ) -> Dataset:
        """Change a dataset. ``test_set=True`` freezes it as a test set: it is
        then only scored on, never trained on or replayed."""
        dataset = self._dataset(dataset_id)
        if spec is not None:
            self._validate_spec(dataset.kind, spec)
            dataset.spec = spec
        if name is not None:
            if not name.strip():
                raise FineTuningError("A dataset needs a name")
            dataset.name = name.strip()
        if description is not None:
            dataset.description = description
        if test_set is not None:
            dataset.test_set = bool(test_set)
        return self.store.save_dataset(dataset)

    def delete_dataset(self, dataset_id: str) -> bool:
        return self.store.delete_dataset(dataset_id)

    # -- examples ------------------------------------------------------------

    def examples(
        self,
        dataset_id: str,
        *,
        status: Optional[str] = None,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        rows = self.store.examples(dataset_id, status=status, split=split)
        if limit is not None:
            rows = rows[offset : offset + int(limit)]
        return [e.to_dict() for e in rows]

    def add_example(
        self,
        dataset_id: str,
        *,
        input: Dict[str, Any],
        label: Optional[Dict[str, Any]] = None,
        split: Optional[str] = None,
        status: str = ExampleStatus.ACCEPTED.value,
        source: str = "manual",
    ) -> Example:
        dataset = self._dataset(dataset_id)
        example = Example(
            dataset_id=dataset_id,
            input=dict(input),
            label=dict(label or {}),
            status=status,
            source=source,
        )
        example.split = split or split_for(example.example_id)
        self._validate_example(dataset, example)
        self.store.add_examples([example])
        return example

    def _validate_example(self, dataset: Dataset, example: Example) -> None:
        if example.split not in SPLITS:
            raise FineTuningError(f"split must be one of {', '.join(SPLITS)}")
        if example.status not in {s.value for s in ExampleStatus}:
            raise FineTuningError("status must be pending, accepted or rejected")
        if dataset.kind == "decision":
            qid, question = only_question(dataset.spec)
            if "state" not in example.input:
                raise FineTuningError("A decision example needs input.state")
            answer = example.label.get(qid)
            if not isinstance(answer, dict):
                raise FineTuningError(f"A decision example needs label.{qid}")
            from .tasks import gold_key

            try:
                key = gold_key(question, answer)
            except (KeyError, TypeError, ValueError):
                raise FineTuningError(
                    f"label.{qid} does not answer a {question['type']} question"
                ) from None
            if key not in {k for k, _ in options(question)}:
                raise FineTuningError(f"{key!r} is not an option of {qid}")
        elif not example.input.get("query") or not example.input.get("positive"):
            raise FineTuningError(
                "A retrieval example needs input.query and input.positive"
            )

    def update_example(
        self,
        example_id: str,
        *,
        label: Optional[Dict[str, Any]] = None,
        split: Optional[str] = None,
        status: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
    ) -> Example:
        example = self.store.get_example(example_id)
        if example is None:
            raise FineTuningError(f"No example {example_id}")
        if label is not None:
            example.label = label
        if split is not None:
            example.split = split
        if status is not None:
            example.status = status
        if input is not None:
            example.input = input
        self._validate_example(self._dataset(example.dataset_id), example)
        return self.store.update_example(example)

    def relabel(self, example_id: str, key: str) -> Example:
        """Set a decision example's answer to the option ``key``."""
        example = self.store.get_example(example_id)
        if example is None:
            raise FineTuningError(f"No example {example_id}")
        qid, question = only_question(self._dataset(example.dataset_id).spec)
        return self.update_example(example_id, label={qid: answer_for(question, key)})

    def review(self, example_ids: Iterable[str], status: str) -> int:
        if status not in {s.value for s in ExampleStatus}:
            raise FineTuningError("status must be pending, accepted or rejected")
        changed = 0
        for example_id in example_ids:
            example = self.store.get_example(example_id)
            if example is not None and example.status != status:
                example.status = status
                self.store.update_example(example)
                changed += 1
        return changed

    def accept_verified(self, dataset_id: str) -> int:
        """Accept every pending example a blind check agreed with."""
        pending = self.store.examples(dataset_id, status=ExampleStatus.PENDING.value)
        return self.review(
            [e.example_id for e in pending if e.verified], ExampleStatus.ACCEPTED.value
        )

    def import_examples(
        self,
        dataset_id: str,
        rows: Iterable[Dict[str, Any]],
        *,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add examples exported with ``export(fmt="memorizz")`` (from this or
        another memory provider). New ids are given; labels are checked."""
        dataset = self._dataset(dataset_id)
        seen = {self._example_text(e) for e in self.store.examples(dataset_id)}
        added, skipped, errors = [], 0, []
        for number, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or not isinstance(row.get("input"), dict):
                errors.append(f"row {number}: no input")
                continue
            example = Example(
                dataset_id=dataset_id,
                input=row["input"],
                label=dict(row.get("label") or {}),
                status=status or row.get("status") or ExampleStatus.PENDING.value,
                split=row.get("split") or "train",
                source=row.get("source") or "import",
                group=row.get("group") or "",
                case_type=row.get("case_type") or "",
                verified=row.get("verified"),
                teacher=row.get("teacher") or "",
                meta=dict(row.get("meta") or {}),
            )
            if self._example_text(example) in seen:
                skipped += 1
                continue
            try:
                self._validate_example(dataset, example)
            except FineTuningError as exc:
                errors.append(f"row {number}: {exc}")
                continue
            seen.add(self._example_text(example))
            added.append(example)
        self.store.add_examples(added)
        return {
            "added": len(added),
            "skipped_duplicates": skipped,
            "errors": errors[:20],
        }

    def delete_examples(self, example_ids: Iterable[str]) -> int:
        return self.store.delete_examples(example_ids)

    # -- data: synthetic and from memory -------------------------------------

    def generate(
        self,
        dataset_id: str,
        *,
        count: int = 40,
        teacher: Optional[Dict[str, Any]] = None,
        context: str = "",
        topic: str = "",
        from_memory: bool = False,
        memory_types: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        verify: bool = True,
        verifier: Optional[Dict[str, Any]] = None,
        progress: Progress = None,
        teacher_model: Any = None,
    ) -> Dict[str, Any]:
        """Write synthetic examples (left pending for review).

        Decision tasks: the teacher writes states for a balanced plan of
        labels and a blind call checks each one. Retrieval tasks (and the
        relevance decision): questions and hard negatives for memory
        passages, from your stored memories (``from_memory``) or written on
        ``topic``."""
        dataset = self._dataset(dataset_id)
        count = max(1, min(int(count), 2000))
        coach = Teacher(teacher or default_teacher(), model=teacher_model)
        try:
            checker = make_verifier(verifier, coach)
        except ValueError as exc:
            raise FineTuningError(str(exc)) from None
        existing = [self._example_text(e) for e in self.store.examples(dataset_id)]
        if dataset.kind == "decision" and dataset.task != "relevance":
            if labels:
                known = {key for key, _ in options(only_question(dataset.spec)[1])}
                unknown = [label for label in labels if label not in known]
                if unknown:
                    raise FineTuningError(
                        f"Not options of this question: {', '.join(unknown)}"
                    )
            examples, report = generate_decision_examples(
                dataset_id,
                dataset.spec,
                coach,
                count=count,
                context=context,
                labels=labels or None,
                existing=existing,
                progress=progress,
                verify=verify,
                verifier=checker,
            )
        else:
            if from_memory:
                passages = self.memory_passages(memory_types, limit=count)
                if not passages:
                    raise FineTuningError(
                        "No stored memories to write questions for; add memories or give a topic"
                    )
            else:
                if progress:
                    progress(
                        0,
                        1,
                        f"writing {count} memories about {topic or context or 'everyday work'}",
                    )
                passages = generate_corpus(
                    topic
                    or context
                    or "an assistant's everyday work: people, projects, preferences and procedures",
                    coach,
                    count=count,
                )
            triples, report = generate_pairs(
                passages,
                coach,
                verify=verify,
                existing=existing,
                progress=progress,
                verifier=checker,
            )
            examples = self._examples_from_triples(
                dataset,
                triples,
                source="memory" if from_memory else "synthetic",
                teacher=coach.label,
            )
        coach.release()
        self.store.add_examples(examples)
        report["added"] = len(examples)
        report["teacher"] = coach.label
        return report

    def _examples_from_triples(
        self,
        dataset: Dataset,
        triples: List[Dict[str, Any]],
        *,
        source: str,
        teacher: str,
    ) -> List[Example]:
        examples = []
        for t in triples:
            split = split_for(t["group"])
            common = dict(
                dataset_id=dataset.dataset_id,
                split=split,
                source=source,
                group=t["group"],
                verified=t.get("verified"),
                teacher=teacher,
                meta={"source_id": t.get("source_id")},
            )
            if dataset.kind == "decision":  # pointwise relevance decisions
                qid, question = only_question(dataset.spec)
                candidates = [
                    (t["positive"], "true"),
                    *((n, "false") for n in t["negatives"]),
                ]
                for text, key in candidates:
                    examples.append(
                        Example(
                            input={"state": {"query": t["query"], "candidate": text}},
                            label={qid: answer_for(question, key)},
                            case_type="positive" if key == "true" else "hard_negative",
                            **common,
                        )
                    )
            else:
                examples.append(
                    Example(
                        input={
                            "query": t["query"],
                            "positive": t["positive"],
                            "negatives": t["negatives"],
                        },
                        **common,
                    )
                )
        return examples

    @staticmethod
    def _example_text(example: Example) -> str:
        if "state" in example.input:
            state = example.input["state"]
            return (
                state if isinstance(state, str) else json.dumps(state, sort_keys=True)
            )
        return str(example.input.get("query") or "")

    def _store_rows(self, store: str) -> List[Dict[str, Any]]:
        try:
            rows = self.provider.list_all(MemoryType(store)) or []
        except Exception:
            return []
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if store == "shared_memory" and (
                row.get("scope") == "private" or row.get("record_type")
            ):
                continue  # control-plane records, including these datasets
            out.append(row)
        return out

    def memory_passages(
        self, memory_types: Optional[List[str]] = None, *, limit: int = 200
    ) -> List[Dict[str, str]]:
        stores = memory_types or [
            "knowledge_base",
            "entity_memory",
            "summaries",
            "conversation_memory",
            "workflow_memory",
        ]
        passages: List[Dict[str, str]] = []
        for store in stores:
            for row in self._store_rows(store):
                text = _memory_text(store, row).strip()
                if len(text) >= 20:
                    passages.append(
                        {
                            # The row's own id: memory_id names a whole
                            # conversation or project, so examples from it
                            # would all share one split.
                            "id": str(
                                row.get("_id")
                                or row.get("id")
                                or row.get("memory_id")
                                or len(passages)
                            ),
                            "text": text[:2000],
                            "store": store,
                        }
                    )
                if len(passages) >= limit:
                    return passages
        return passages

    def import_memories(
        self,
        dataset_id: str,
        *,
        memory_types: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Your stored memories as routing examples, each labelled with the
        store it lives in (left pending for review)."""
        dataset = self._dataset(dataset_id)
        if dataset.kind != "decision" or dataset.task == "relevance":
            raise FineTuningError(
                "Import labels memories by their store; use generate(from_memory=True) for retrieval data"
            )
        qid, question = only_question(dataset.spec)
        keys = {key for key, _ in options(question)}
        stores = [
            s
            for s in (memory_types or list(MEMORY_ROUTING_OPTIONS))
            if s in keys and s != "do_not_store"
        ]
        seen = {self._example_text(e) for e in self.store.examples(dataset_id)}
        examples, per_store = [], {}
        for store in stores:
            for row in self._store_rows(store):
                text = _memory_text(store, row).strip()[:2000]
                if len(text) < 8 or text in seen:
                    continue
                seen.add(text)
                source_id = str(
                    row.get("memory_id") or row.get("_id") or row.get("id") or ""
                )
                example = Example(
                    dataset_id=dataset_id,
                    input={"state": text},
                    label={qid: answer_for(question, store)},
                    source="memory",
                    group=f"memory:{source_id or text[:40]}",
                    meta={"store": store, "source_id": source_id},
                )
                example.split = split_for(example.group)
                examples.append(example)
                per_store[store] = per_store.get(store, 0) + 1
                if len(examples) >= limit:
                    break
            if len(examples) >= limit:
                break
        stratified_splits(examples, lambda e: e.meta.get("store", ""))
        self.store.add_examples(examples)
        return {"added": len(examples), "by_store": per_store}

    # -- formats ---------------------------------------------------------------

    def preview(
        self,
        dataset_id: str,
        *,
        base_model: Optional[str] = None,
        fmt: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        dataset = self._dataset(dataset_id)
        base = resolve_base(
            base_model or default_base(dataset.kind).model_id, dataset.kind
        )
        fmt = fmt or default_format(dataset.kind, base)
        rows = self.store.examples(
            dataset_id, status=ExampleStatus.ACCEPTED.value
        ) or self.store.examples(dataset_id)
        rendered = []
        for example in rows[: max(1, int(limit))]:
            rendered.extend(
                render(fmt, dataset.kind, dataset.spec, base, example.to_dict())
            )
        return {
            "base_model": base.to_dict(),
            "format": fmt,
            "how": describe_format(base, dataset.kind, dataset.spec),
            "rows": rendered,
        }

    def export(
        self,
        dataset_id: str,
        *,
        fmt: str = "memorizz",
        base_model: Optional[str] = None,
        status: Optional[str] = "accepted",
        split: Optional[str] = None,
    ) -> str:
        dataset = self._dataset(dataset_id)
        base = resolve_base(
            base_model or default_base(dataset.kind).model_id, dataset.kind
        )
        rows = []
        for example in self.store.examples(dataset_id, status=status, split=split):
            for row in render(fmt, dataset.kind, dataset.spec, base, example.to_dict()):
                rows.append(
                    {**row, "split": example.split} if fmt not in ("memorizz",) else row
                )
        return to_jsonl(rows)

    def unsloth_bundle(
        self, dataset_id: str, *, base_model: str = "unsloth/Qwen3.5-4B"
    ) -> Dict[str, str]:
        dataset = self._dataset(dataset_id)
        if dataset.kind != "decision":
            raise FineTuningError("Unsloth bundles are for decision datasets")
        files = unsloth_bundle(base_model)
        files["data.jsonl"] = self.export(
            dataset_id, fmt="systemone", base_model=base_model
        )
        return files

    # -- training --------------------------------------------------------------

    def _snapshot(
        self, dataset: Dataset, replay: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        for item in replay or []:
            source = self._dataset(str(item.get("dataset_id")))
            if source.test_set:
                raise FineTuningError(
                    f"{source.name} is a test set; replaying it would train on it"
                )
        data = self._own_snapshot(dataset)
        for item in replay or []:
            source = self._dataset(str(item.get("dataset_id")))
            if source.kind != dataset.kind:
                raise FineTuningError(
                    f"Replay dataset {source.name} is for a {source.kind} model"
                )
            if dataset.kind == "decision":
                ours = [k for k, _ in options(only_question(dataset.spec)[1])]
                theirs = [k for k, _ in options(only_question(source.spec)[1])]
                if set(ours) != set(theirs):
                    raise FineTuningError(
                        f"Replay dataset {source.name} has different options"
                    )
            fraction = max(0.0, min(1.0, float(item.get("fraction", 0.5))))
            rows = [
                e.to_dict()
                for e in self.store.examples(
                    source.dataset_id,
                    status=ExampleStatus.ACCEPTED.value,
                    split="train",
                )
            ]
            import random as _random

            _random.Random(7).shuffle(rows)
            chosen = rows[: int(round(len(rows) * fraction))]
            if dataset.kind == "decision":
                # The same options can sit under another question id there.
                qid = only_question(dataset.spec)[0]
                theirs = only_question(source.spec)[0]
                chosen = [
                    {**row, "label": {qid: row["label"].get(theirs)}} for row in chosen
                ]
            data["train"].extend({**row, "replay": source.dataset_id} for row in chosen)
        self._check_labels(dataset, [r for rows in data.values() for r in rows])
        return data

    @staticmethod
    def _check_labels(dataset: Dataset, rows: Iterable[Dict[str, Any]]) -> None:
        """Every decision example must answer the question as it is now: an
        option renamed or removed after labelling would otherwise stop
        training or scoring with an unhelpful error."""
        if dataset.kind != "decision":
            return
        from .tasks import gold_key

        qid, question = only_question(dataset.spec)
        keys = {key for key, _ in options(question)}
        unknown: Dict[str, int] = {}
        for row in rows:
            try:
                key = gold_key(question, (row.get("label") or {})[qid])
            except (KeyError, TypeError, ValueError):
                key = "(no answer)"
            if key not in keys:
                unknown[key] = unknown.get(key, 0) + 1
        if unknown:
            listed = ", ".join(f"{k} ×{n}" for k, n in sorted(unknown.items()))
            raise FineTuningError(
                f"{sum(unknown.values())} accepted examples are labelled with options "
                f"the question no longer has ({listed}). Relabel or delete them, or "
                "put the options back in the question."
            )

    def _own_snapshot(self, dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
        accepted = self.store.examples(
            dataset.dataset_id, status=ExampleStatus.ACCEPTED.value
        )
        data: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
        for e in accepted:
            data[e.split if e.split in data else "train"].append(e.to_dict())
        held_out = len(data["validation"]) + len(data["test"])
        # A before/after on one or two examples says nothing: hold out at
        # least a tenth of a dataset (up to 20 examples).
        needed = max(MIN_HELD_OUT, min(20, len(accepted) // 10))
        if held_out < needed and len(accepted) >= MIN_TRAIN + needed:
            # Everything was marked train: hold some out so the before/after
            # comparison means something. By group first (related examples
            # stay together); if the groups are too few, a share of each label.
            data = {split: [] for split in SPLITS}
            for e in accepted:
                data[split_for(e.group or e.example_id)].append(e.to_dict())
            if (
                len(data["validation"]) + len(data["test"]) < needed
                or len(data["train"]) < MIN_TRAIN
            ):
                copies = [Example(**{**e.to_dict()}) for e in accepted]
                stratified_splits(copies, lambda e: json.dumps(e.label, sort_keys=True))
                data = {
                    split: [e.to_dict() for e in copies if e.split == split]
                    for split in SPLITS
                }
        if len(data["train"]) < MIN_TRAIN:
            raise FineTuningError(
                f"Accept at least {MIN_TRAIN} training examples first ({len(data['train'])} so far)"
            )
        if not data["test"] and not data["validation"]:
            raise FineTuningError(
                "Hold out a few accepted examples as test or validation"
            )
        if not data["test"] and len(data["validation"]) >= 2:
            # Calibration is fitted on validation; scoring on those same
            # examples would flatter it. Score on half of them instead,
            # keeping each group whole.
            rows = sorted(data["validation"], key=lambda r: r.get("group") or "")
            half = len(rows) // 2
            while (
                0 < half < len(rows)
                and rows[half].get("group")
                and rows[half].get("group") == rows[half - 1].get("group")
            ):
                half += 1
            if 0 < half < len(rows):
                data["validation"], data["test"] = rows[:half], rows[half:]
        return data

    def start_run(
        self,
        dataset_id: str,
        *,
        base_model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        backend: str = "auto",
        name: str = "",
        detach: bool = True,
        from_run: Optional[str] = None,
        replay: Optional[List[Dict[str, Any]]] = None,
        in_process: bool = False,
        label: str = "",
    ) -> TrainingRun:
        """Train on the dataset's accepted examples. ``from_run`` continues
        from that run's model (continual training): same backbone and input
        format, with the lineage recorded.

        Training runs in its own worker process: in the background
        (``detach``, the default) or waited for (``detach=False``). Native
        libraries a memory provider loads (FAISS's OpenMP on macOS) can crash
        PyTorch in the same process; ``in_process=True`` trains here anyway."""
        dataset = self._dataset(dataset_id)
        if dataset.test_set:
            raise FineTuningError(
                f"{dataset.name} is a test set: it is only scored on. "
                "Unmark it as a test set to train on it."
            )
        env = environment()
        if not env["ready"]:
            raise FineTuningError(
                "Training needs " + ", ".join(env["missing"]) + ": " + env["install"]
            )
        parent = None
        if from_run:
            parent = self.store.get_run(from_run)
            if parent is None or parent.kind != dataset.kind:
                raise FineTuningError(
                    f"Run {from_run} did not train a {dataset.kind} model"
                )
            parent_config = json.loads(
                self._files(parent).config.read_text(encoding="utf-8")
            )
            base = resolve_base(parent_config["base_model"], dataset.kind)
            base = base.__class__(
                **{
                    **base.to_dict(),
                    "kinds": tuple(base.kinds),
                    "recommended": tuple(base.recommended),
                    "model_id": self.model_dir(from_run),
                    "family": parent_config["family"],
                    "backbone": parent_config["backbone"],
                }
            )
        else:
            base = resolve_base(
                base_model or default_base(dataset.kind).model_id, dataset.kind
            )
        if dataset.kind not in base.kinds:
            raise FineTuningError(
                f"{base.model_id} cannot be trained as a {dataset.kind} model"
            )
        if backend == "auto":
            backend = (
                "unsloth"
                if base.backbone == "lm" and env["unsloth"]["available"]
                else "local"
            )
        if backend == "unsloth" and not env["unsloth"]["available"]:
            raise FineTuningError(
                env["unsloth"]["reason"]
                + "; use the local backend or export an Unsloth bundle"
            )
        data = self._snapshot(dataset, replay)
        run = TrainingRun(
            dataset_id=dataset_id,
            kind=dataset.kind,
            task=dataset.task,
            base_model=base.model_id,
            backbone=base.backbone,
            backend=backend,
            params=dict(params or {}),
            name=name or f"{dataset.name} · {base.label}",
            label=label.strip()[:40],
            progress={
                "host": socket.gethostname(),
                "counts": {k: len(v) for k, v in data.items()},
            },
            parent_run_id=parent.run_id if parent else "",
        )
        if replay:
            run.progress["replay"] = [
                {
                    "dataset_id": item["dataset_id"],
                    "fraction": item.get("fraction", 0.5),
                    "examples": sum(
                        1
                        for r in data["train"]
                        if r.get("replay") == item["dataset_id"]
                    ),
                }
                for item in replay
            ]
        if parent:
            run.base_model = parent.base_model  # the original base, for display
            run.progress["lineage"] = [
                *(parent.progress.get("lineage") or []),
                parent.run_id,
            ]
        files = self.store.run_files(run.run_id).ensure()
        run.output_dir = str(files.root)
        config = {
            "kind": dataset.kind,
            "task": dataset.task,
            "spec": dataset.spec,
            "base_model": base.model_id,
            "original_base_model": run.base_model,
            "parent_run_id": run.parent_run_id,
            "family": base.family,
            "backbone": base.backbone,
            "backend": backend,
            "params": run.params,
            "dataset_id": dataset_id,
            "run_id": run.run_id,
        }
        files.config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        with open(files.data, "w", encoding="utf-8") as handle:
            for split, rows in data.items():
                for row in rows:
                    handle.write(
                        json.dumps({**row, "split": split}, ensure_ascii=False) + "\n"
                    )
        run.status = RunStatus.RUNNING.value
        run.started_at = utcnow()
        self.store.save_run(run)
        if in_process:
            from .worker import run as run_worker

            run_worker(files.root)
            return self._sync(run)
        environ = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Apple GPU: run the few operations MPS lacks on the CPU.
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
        command = [sys.executable, "-m", "memorizz.finetune.worker", str(files.root)]
        with open(files.log, "ab") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=detach,
                env=environ,
            )
        run.pid = process.pid
        self.store.save_run(run)
        if not detach:
            process.wait()
        return self._sync(run)

    def _sync(self, run: TrainingRun) -> TrainingRun:
        """Bring the provider's record of a run up to date with its folder."""
        if run.status in FINAL_RUN_STATES:
            return run
        files = self._files(run)
        state = files.state()
        phase = state.get("phase")
        changed = False
        if state.get("pid") and state["pid"] != run.pid:
            run.pid, changed = state["pid"], True
        progress = {
            k: state[k]
            for k in (
                "step",
                "total",
                "epoch",
                "loss",
                "eta_seconds",
                "elapsed_seconds",
                "message",
            )
            if k in state
        }
        if progress and any(run.progress.get(k) != v for k, v in progress.items()):
            run.progress.update(progress)
            changed = True
        if phase in ("succeeded", "failed", "canceled"):
            result = state.get("result") or {}
            run.status = phase
            run.metrics_before = result.get("metrics_before") or {}
            run.metrics_after = result.get("metrics_after") or {}
            run.device = result.get("device") or run.device
            run.error = state.get("error") or ""
            run.finished_at = state.get("finished_at") or utcnow()
            if result.get("model_dir"):
                run.progress["model_dir"] = result["model_dir"]
            changed = True
        elif run.progress.get("host") == socket.gethostname() and not _pid_alive(
            run.pid
        ):
            run.status = RunStatus.INTERRUPTED.value
            run.error = "The training process stopped without finishing"
            run.finished_at = utcnow()
            changed = True
        if changed:
            self.store.save_run(run)
        return run

    def runs(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Runs, newest first, each with ``job`` (what its model does),
        ``version`` (v1, v2, … per job, in the order they were started) and
        ``label`` (its given name, else "v3")."""
        jobs = {d.dataset_id: self._job_of(d) for d in self.store.list_datasets()}
        rows = [self._sync(run).to_dict() for run in self.store.list_runs()]
        for row in rows:
            job = jobs.get(row["dataset_id"]) or self._job_of(
                Dataset(name="", kind=row["kind"], task=row["task"])
            )
            row["job"], row["job_title"] = job["key"], job["title"]
        rows = self._labelled(rows)
        return [r for r in rows if dataset_id is None or r["dataset_id"] == dataset_id]

    @staticmethod
    def _job_of(dataset: Dataset) -> Dict[str, str]:
        """The job a dataset's model does: a memory job (routing, embeddings,
        reranking, whatever the model type) or, for a custom decision, its
        own question, so unrelated questions are numbered apart."""
        kind, task = dataset.kind, dataset.task
        if kind == "decision" and task == "memory_routing":
            return {"key": "memory_routing", "title": "Memory routing"}
        if kind == "embedding":
            return {"key": "embedding", "title": "Memory embeddings"}
        if kind == "reranker" or (kind == "decision" and task == "relevance"):
            return {"key": "reranking", "title": "Memory reranking"}
        import hashlib

        try:
            qid, question = only_question(dataset.spec)
        except (ValueError, TypeError):
            return {
                "key": f"{kind}:{task}",
                "title": task.replace("_", " ").capitalize(),
            }
        keys = sorted(key for key, _ in options(question))
        signature = hashlib.sha1(f"{qid}|{'|'.join(keys)}".encode()).hexdigest()[:10]
        instructions = str(question.get("instructions") or qid).strip().rstrip(".")
        return {
            "key": f"custom:{signature}",
            "title": f"{instructions} ({len(keys)} options)",
        }

    def run(self, run_id: str) -> Dict[str, Any]:
        for row in self.runs():
            if row["run_id"] == run_id:
                return row
        raise FineTuningError(f"No training run {run_id}")

    @staticmethod
    def _labelled(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failed = {
            RunStatus.FAILED.value,
            RunStatus.CANCELED.value,
            RunStatus.INTERRUPTED.value,
        }
        numbers: Dict[tuple, int] = {}
        for row in sorted(rows, key=lambda r: (r["created_at"], r["run_id"])):
            if row["status"] in failed:
                row["version"] = None
                continue
            job = row.get("job") or (row["kind"], row["task"])
            numbers[job] = numbers.get(job, 0) + 1
            row["version"] = numbers[job]
        names = {}
        for row in rows:
            given = str(row.get("label") or "").strip()
            row["label"] = given or (
                f"v{row['version']}"
                if row["version"]
                else f"failed run {row['run_id'][:6]}"
            )
            names[row["run_id"]] = row["label"]
        for row in rows:
            parent = row.get("parent_run_id")
            replay = (row.get("progress") or {}).get("replay")
            row["parent_label"] = names.get(parent, parent) if parent else ""
            parts = [f"from {names.get(parent, parent)}"] if parent else []
            if replay:
                parts.append("replay")
            row["lineage"] = " + ".join(parts)
            row["short_base"] = str(row["base_model"]).rstrip("/").split("/")[-1]
        return rows

    def rename_run(self, run_id: str, label: str) -> Dict[str, Any]:
        """Give a version a short name (empty puts back "v3")."""
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        run.label = str(label or "").strip()[:40]
        self.store.save_run(run)
        return self.run(run_id)

    def _files(self, run: TrainingRun):
        """A run's folder, wherever it was trained to (not only this home)."""
        from pathlib import Path

        from .store import RunFiles

        return (
            RunFiles(Path(run.output_dir))
            if run.output_dir
            else self.store.run_files(run.run_id)
        )

    def events(
        self, run_id: str, *, after: int = 0, limit: int = 2000
    ) -> List[Dict[str, Any]]:
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        return self._files(run).events(after=after, limit=limit)

    def curves(self, run_id: str) -> Dict[str, Any]:
        """How a run trained: its loss by step (at most 400 points), its
        validation score per epoch, its score before and after, and a plain
        reading of convergence."""
        run = self.run(run_id)
        metric = self._primary_metric(run["kind"], run["task"])
        events = self.events(run_id, limit=100000)
        steps = [e["data"] for e in events if e["type"] == "step"]
        stride = max(1, -(-len(steps) // 400))
        thinned = steps[::stride] + (
            [steps[-1]] if steps and (len(steps) - 1) % stride else []
        )

        def compact(metrics: Dict[str, Any]) -> Dict[str, Any]:
            return {
                k: v
                for k, v in (metrics or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }

        validation = [
            {
                "epoch": e["data"].get("epoch"),
                "step": e["data"].get("step"),
                "metrics": compact(e["data"].get("metrics")),
            }
            for e in events
            if e["type"] == "eval" and e["data"].get("phase") == "validation"
        ]
        phases = {
            e["data"].get("phase"): compact(e["data"].get("metrics"))
            for e in events
            if e["type"] == "eval" and e["data"].get("phase") in ("before", "after")
        }
        return {
            "run_id": run_id,
            "label": run["label"],
            "metric": metric,
            "steps": [
                {
                    k: d.get(k)
                    for k in (
                        "step",
                        "epoch",
                        "loss",
                        "learning_rate",
                        "batch_accuracy",
                    )
                    if d.get(k) is not None
                }
                for d in thinned
            ],
            "total_steps": steps[-1].get("total") if steps else None,
            "validation": validation,
            "before": phases.get("before") or compact(run.get("metrics_before")),
            "after": phases.get("after") or compact(run.get("metrics_after")),
            "convergence": self._convergence(steps, validation, metric),
        }

    @staticmethod
    def _convergence(
        steps: List[Dict[str, Any]], validation: List[Dict[str, Any]], metric: str
    ) -> List[str]:
        """Plain notes on how training went: how far the loss fell and
        whether it was still falling, and when validation peaked."""
        notes: List[str] = []
        losses = [float(s["loss"]) for s in steps if s.get("loss") is not None]
        if len(losses) >= 10:
            tenth = max(1, len(losses) // 10)
            start = sum(losses[:tenth]) / tenth
            end = sum(losses[-tenth:]) / tenth
            notes.append(
                f"Loss fell from {start:.3g} to {end:.3g} ({(end - start) / start * 100:+.0f}%)."
                if start
                else f"Loss ended at {end:.3g}."
            )
            # Still falling: the last fifth of training cut the loss by more
            # than 5% of the whole fall (a tiny loss can halve without
            # mattering).
            fifth = max(2, len(losses) // 5)
            late = losses[-fifth:]
            half = len(late) // 2
            earlier = sum(late[:half]) / half
            later = sum(late[half:]) / (len(late) - half)
            fall = start - end
            if (
                fall > 0
                and earlier - later > 0.05 * fall
                and (earlier - later) / earlier > 0.05
            ):
                notes.append("It was still falling at the end: more epochs may help.")
            elif fall > 0:
                share = max(0.0, earlier - later) / fall
                notes.append(
                    f"It had levelled off by the end: the last fifth of training "
                    f"took off {share:.0%} of the whole fall (the log-scale chart "
                    "magnifies small losses)."
                )
            else:
                notes.append("It had levelled off by the end.")
        scores = [
            (v["epoch"], v["metrics"].get(metric, v["metrics"].get("accuracy")))
            for v in validation
            if v["metrics"].get(metric, v["metrics"].get("accuracy")) is not None
        ]
        if len(scores) >= 2:
            best_epoch, best = max(scores, key=lambda x: x[1])
            last_epoch, last = scores[-1]
            if last < best - 0.02:
                notes.append(
                    f"Validation peaked at epoch {best_epoch} ({best:.1%}) and ended at {last:.1%}: "
                    "later epochs may be overfitting; fewer epochs could do as well."
                )
            elif scores[-1][1] > scores[-2][1] + 0.02:
                notes.append(
                    f"Validation was still rising at the last epoch ({last:.1%})."
                )
            else:
                notes.append(
                    f"Validation settled at {last:.1%} (best {best:.1%}, epoch {best_epoch})."
                )
        elif not scores:
            notes.append(
                "No validation score was recorded during training (runs before 13 Oct 2026 scored embeddings and rerankers only before and after)."
            )
        return notes

    def run_data(
        self,
        run_id: str,
        *,
        split: Optional[str] = None,
        source: Optional[str] = None,
        offset: int = 0,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """The examples a run trained and was scored on: its own snapshot
        (taken when it started), by split, with replayed examples marked,
        and how the dataset has changed since."""
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        files = self._files(run)
        if not files.data.exists():
            raise FineTuningError("This run's data snapshot is not on this machine")
        rows = [
            json.loads(line)
            for line in open(files.data, encoding="utf-8")
            if line.strip()
        ]
        names = {d.dataset_id: d.name for d in self.store.list_datasets()}
        qid = None
        if run.kind == "decision":
            try:
                qid = only_question(self._dataset(run.dataset_id).spec)[0]
            except FineTuningError:
                qid = None

        def label_of(row: Dict[str, Any]) -> str:
            answer = (row.get("label") or {}).get(qid) if qid else None
            if isinstance(answer, dict):
                return str(
                    answer.get("choice", answer.get("noul", answer.get("level", "")))
                )
            return ""

        def text_of(row: Dict[str, Any]) -> Dict[str, Any]:
            data = row.get("input") or {}
            if "state" in data:
                state = data["state"]
                return {
                    "state": state
                    if isinstance(state, str)
                    else json.dumps(state, ensure_ascii=False)
                }
            return {
                "query": data.get("query"),
                "positive": data.get("positive"),
                "negatives": data.get("negatives") or [],
            }

        summary: Dict[str, Any] = {"splits": {}, "labels": {}, "replayed": {}}
        for row in rows:
            split_name = row.get("split") or "train"
            summary["splits"][split_name] = summary["splits"].get(split_name, 0) + 1
            if row.get("replay"):
                key = names.get(row["replay"], row["replay"])
                summary["replayed"][key] = summary["replayed"].get(key, 0) + 1
            if qid:
                per = summary["labels"].setdefault(label_of(row), {})
                per[split_name] = per.get(split_name, 0) + 1
        chosen = [
            r
            for r in rows
            if (split is None or (r.get("split") or "train") == split)
            and (
                source is None
                or (source == "replay" and r.get("replay"))
                or (source == "own" and not r.get("replay"))
            )
        ]
        snapshot_ids = {r.get("example_id") for r in rows if not r.get("replay")}
        current = {
            e.example_id
            for e in self.store.examples(
                run.dataset_id, status=ExampleStatus.ACCEPTED.value
            )
        }
        return {
            "run_id": run_id,
            "dataset_name": names.get(run.dataset_id, run.dataset_id),
            "total": len(chosen),
            "summary": summary,
            "since": {
                "added": len(current - snapshot_ids),
                "removed": len(snapshot_ids - current),
            },
            "rows": [
                {
                    "example_id": r.get("example_id"),
                    "split": r.get("split") or "train",
                    "label": label_of(r),
                    "replayed_from": names.get(r["replay"], r["replay"])
                    if r.get("replay")
                    else "",
                    **text_of(r),
                }
                for r in chosen[offset : offset + max(1, min(limit, 200))]
            ],
        }

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        if run.status not in FINAL_RUN_STATES:
            self._files(run).request_cancel()
        return self.run(run_id)

    def delete_run(self, run_id: str, *, files: bool = True) -> bool:
        import shutil

        run = self.store.get_run(run_id)
        if (
            run is not None
            and run.status == RunStatus.RUNNING.value
            and _pid_alive(run.pid)
        ):
            raise FineTuningError("Cancel the run before deleting it")
        for use, record in self.active().items():
            if record.get("run_id") == run_id:
                raise FineTuningError(
                    f"This version is in use for {use.replace('_', ' ')}: swap "
                    "back or swap in another version before deleting it"
                )
        if files and run is not None:
            shutil.rmtree(self._files(run).root, ignore_errors=True)
        for key, value in list(self.active().items()):
            if value.get("run_id") == run_id:
                self.store.set_setting(f"active:{key}", None)
        return self.store.delete_run(run_id)

    # -- using a model ---------------------------------------------------------

    def model_dir(self, run_id: str) -> str:
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        run = self._sync(run)
        path = run.progress.get("model_dir") or os.path.join(run.output_dir, "model")
        if run.status != RunStatus.SUCCEEDED.value or not os.path.isdir(path):
            raise FineTuningError("This run has no finished model on this machine")
        return path

    def decide(
        self, run_id: str, state: Any, questions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._decision_model(run_id).decide(state, questions)

    def rank(
        self, run_id: str, query: str, passages: List[str]
    ) -> List[Dict[str, Any]]:
        from .inference import rank_with_embedding, rank_with_reranker

        run = self.store.get_run(run_id)
        path = self.model_dir(run_id)
        if run.kind == "decision":
            answers = [
                self.decide(run_id, {"query": query, "candidate": p}) for p in passages
            ]
            scores = [next(iter(a.values())).get("noul", 0.0) for a in answers]
        elif run.kind == "embedding":
            scores = rank_with_embedding(path, query, passages)
        else:
            scores = rank_with_reranker(path, query, passages)
        order = sorted(range(len(passages)), key=lambda i: -scores[i])
        return [
            {"index": i, "score": round(float(scores[i]), 6), "passage": passages[i]}
            for i in order
        ]

    def _decision_model(self, run_id: str):
        from .inference import DecisionModel

        path = self.model_dir(run_id)
        with self._lock:
            model = self._models.get(path)
            if model is None:
                model = self._models[path] = DecisionModel(path)
                while len(self._models) > 2:
                    self._models.pop(next(iter(self._models)))
        return model

    def evaluate(
        self,
        dataset_id: str,
        *,
        run_id: Optional[str] = None,
        base_model: Optional[str] = None,
        jev_model: Optional[str] = None,
        in_use: bool = False,
        split: str = "test",
        record: bool = True,
        max_length: Optional[int] = None,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Score one model on a dataset's accepted held-out examples.

        The model is a trained run, an untrained base model, Jev
        (``jev_model``, e.g. "jev-latest", decision datasets) or, for
        retrieval and reranking datasets, the embedding model in use now
        (``in_use``). ``split`` is test, validation, held_out (both) or all.
        Results are kept in the memory provider, so every version of a model
        can be compared on the same frozen sets: the history continual
        training is judged by."""
        import time as _time

        from .evaluate import softmax
        from .models import new_id
        from .scoring import decision_metrics
        from .tasks import gold_key

        dataset = self._dataset(dataset_id)
        if dataset.kind != "decision":
            return self._evaluate_ranking(
                dataset,
                run_id=run_id,
                base_model=base_model,
                in_use=in_use,
                split=split,
                record=record,
                max_length=max_length,
                dimensions=dimensions,
            )
        if in_use:
            raise FineTuningError("in_use compares retrieval or reranking models")
        qid, question = only_question(dataset.spec)
        wanted = {
            "test": {"test"},
            "validation": {"validation"},
            "held_out": {"test", "validation"},
            "all": set(SPLITS),
        }.get(split)
        if wanted is None:
            raise FineTuningError("split must be test, validation, held_out or all")
        rows = [
            e
            for e in self.store.examples(
                dataset_id, status=ExampleStatus.ACCEPTED.value
            )
            if e.split in wanted
        ]
        if not rows:
            raise FineTuningError(f"No accepted {split} examples to evaluate on")
        self._check_labels(dataset, [e.to_dict() for e in rows])
        keys = [key for key, _ in options(question)]
        states = [e.input["state"] for e in rows]
        gold = [keys.index(gold_key(question, e.label[qid])) for e in rows]
        started = _time.monotonic()
        extra: Dict[str, Any] = {}
        abstained = None
        if run_id:
            run = self.store.get_run(run_id)
            if run is None or run.kind != "decision":
                raise FineTuningError(f"Run {run_id} is not a decision model")
            probabilities = self._decision_model(run_id).probabilities_many(
                states, question
            )
            model = {
                "type": "run",
                "run_id": run_id,
                "label": f"{run.base_model.split('/')[-1]} · {run_id}",
                "parent_run_id": run.parent_run_id,
            }
        elif jev_model:
            from .synthetic import JevVerifier

            try:
                verifier = JevVerifier(model=jev_model, min_confidence=0.0)
            except ValueError as exc:
                raise FineTuningError(str(exc)) from None
            verdicts = verifier.decide_many(question, states)
            uniform = [1 / len(keys)] * len(keys)
            probabilities = [
                [(v.get("probabilities") or {}).get(k, 0.0) for k in keys]
                if v and v.get("probabilities")
                else uniform
                for v in verdicts
            ]
            # A call that returned nothing counts as a wrong answer.
            abstained = [sum(row) <= 0 or row is uniform for row in probabilities]
            probabilities = [row if sum(row) > 0 else uniform for row in probabilities]
            extra["usage"] = verifier.usage()
            model = {"type": "jev", "label": f"Jev ({jev_model}), zero-shot"}
        else:
            from .inference import base_decision_logits

            base = resolve_base(
                base_model or default_base("decision").model_id, "decision"
            )
            logits = base_decision_logits(
                base.model_id,
                base.family,
                base.backbone,
                question,
                states,
                max_length=max_length,
            )
            probabilities = [softmax(row) for row in logits]
            model = {
                "type": "base",
                "base_model": base.model_id,
                "label": f"{base.label}, untrained",
            }
        metrics = decision_metrics(
            [e.to_dict() for e in rows],
            probabilities,
            gold,
            keys,
            question["type"],
            abstained=abstained,
        )
        result = {
            "eval_id": "fteval-" + new_id(),
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "split": split,
            "count": len(rows),
            "model": model,
            "metrics": metrics,
            "seconds": round(_time.monotonic() - started, 1),
            "created_at": utcnow(),
            **extra,
        }
        if record:
            self.store.save_eval(result)
        return result

    def _evaluate_ranking(
        self,
        dataset: Dataset,
        *,
        run_id: Optional[str],
        base_model: Optional[str],
        in_use: bool,
        split: str,
        record: bool,
        max_length: Optional[int] = None,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Score retrieval with ``scoring.rank_held_out``, the code training
        scores a run with, so a comparison here and a run's own before/after
        agree: an embedding model searches every memory in the dataset for
        each held-out query; a reranker reorders the query's memory, its
        look-alikes and 9 other memories (a fixed sample). MRR, nDCG and
        Recall, with each query's nDCG@10 in ``per_item``."""
        import time as _time

        from .models import new_id
        from .scoring import memory_pool, rank_held_out

        wanted = {
            "test": {"test"},
            "validation": {"validation"},
            "held_out": {"test", "validation"},
            "all": set(SPLITS),
        }.get(split)
        if wanted is None:
            raise FineTuningError("split must be test, validation, held_out or all")
        accepted = [
            e.to_dict()
            for e in self.store.examples(
                dataset.dataset_id, status=ExampleStatus.ACCEPTED.value
            )
            if isinstance(e.input, dict)
            and e.input.get("query")
            and e.input.get("positive")
        ]
        rows = [e for e in accepted if e["split"] in wanted]
        if not rows:
            raise FineTuningError(f"No accepted {split} examples to evaluate on")
        by_split: Dict[str, List[Dict[str, Any]]] = {}
        for e in accepted:
            by_split.setdefault(
                e["split"] if e["split"] in SPLITS else "train", []
            ).append(e)
        pool = memory_pool(by_split)
        started = _time.monotonic()
        score, model = self._ranker(
            dataset,
            run_id=run_id,
            base_model=base_model,
            in_use=in_use,
            max_length=max_length,
            dimensions=dimensions,
        )
        metrics = rank_held_out(dataset.kind, rows, pool, score)
        result = {
            "eval_id": "fteval-" + new_id(),
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "split": split,
            "count": len(rows),
            "candidates": len(pool) if dataset.kind != "reranker" else None,
            "method": RANKING_METHOD,
            "model": model,
            "metrics": metrics,
            "seconds": round(_time.monotonic() - started, 1),
            "created_at": utcnow(),
        }
        if record:
            self.store.save_eval(result)
        return result

    def _ranker(
        self,
        dataset: Dataset,
        *,
        run_id,
        base_model,
        in_use,
        max_length=None,
        dimensions=None,
    ):
        """A ``(query, passages) -> scores`` function and its description.

        An untrained base model reads at most ``max_length`` tokens (the
        trainer's default unless given), as it did when training scored it;
        a trained model keeps the length it was saved with. An embedding
        model scored at ``dimensions`` has its vectors cut to that size, as
        a store of that size keeps them."""
        if in_use:
            from ..embeddings import get_embedding_manager
            from ..memory_provider.vectors import cosine

            manager = get_embedding_manager()
            info = manager.get_provider_info()
            vectors: Dict[str, Any] = {}

            def by_embedding(query, passages):
                q = manager.get_embedding(query, input_type="query")
                missing = [p for p in dict.fromkeys(passages) if p not in vectors]
                if missing:
                    vectors.update(zip(missing, manager.get_embeddings(missing)))
                return [cosine(q, vectors[p]) or 0.0 for p in passages]

            short = str(info.get("model")).rstrip("/").split("/")[-1]
            label = (
                f"Vector search only ({short})" if dataset.kind == "reranker" else short
            )
            return by_embedding, {
                "type": "in_use",
                "label": label,
                "model": str(info.get("model")),
            }
        if run_id:
            run = self.store.get_run(run_id)
            if run is None or run.kind != dataset.kind:
                raise FineTuningError(f"Run {run_id} is not a {dataset.kind} model")
            path, name = self.model_dir(run_id), run.base_model
            model = {
                "type": "run",
                "run_id": run_id,
                "label": f"{name.split('/')[-1]} · {run_id}",
                "parent_run_id": run.parent_run_id,
            }
        else:
            base = resolve_base(
                base_model or default_base(dataset.kind).model_id, dataset.kind
            )
            path, name = base.model_id, base.model_id
            model = {
                "type": "base",
                "base_model": base.model_id,
                "label": f"{base.label}, untrained",
            }
        from .scoring import cross_encoder_scorer, embedding_scorer
        from .training.common import pick_device
        from .training.retrieval import EMBEDDING_DEFAULTS, RERANKER_DEFAULTS

        device = pick_device(None)
        if dataset.kind == "reranker":
            from sentence_transformers.cross_encoder import CrossEncoder

            options = {}
            if not run_id:
                options["max_length"] = int(
                    max_length or RERANKER_DEFAULTS["max_length"]
                )
                model["max_length"] = options["max_length"]
            encoder = CrossEncoder(str(path), device=device, **options)
            return cross_encoder_scorer(encoder), model
        from .training.decision_embed import load_encoder
        from .training.retrieval import prompts_for

        family = resolve_base(name, "embedding").family
        encoder = load_encoder(str(path), family, device)
        if not run_id:
            encoder.max_seq_length = int(max_length or EMBEDDING_DEFAULTS["max_length"])
            model["max_length"] = encoder.max_seq_length
        prompts = dict(getattr(encoder, "prompts", {}) or {})
        if run_id is None or not prompts.get("query"):
            prompts = prompts_for(family, str(dataset.spec.get("instruction") or ""))
        if dimensions:
            model["dimensions"] = int(dimensions)
            model["label"] += f" @ {int(dimensions)} dims"
        return embedding_scorer(encoder, prompts, dimensions), model

    def evaluations(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recorded scores, oldest first. A fine-tuned model is named by its
        version ("v3 · EmbeddingGemma-2"). Retrieval and reranking scores
        from before scoring matched training are left out: they used a
        smaller candidate pool, so they are not comparable."""
        runs = {r["run_id"]: r for r in self.runs()}
        kinds = {d.dataset_id: d.kind for d in self.store.list_datasets()}
        out = []
        for e in self.store.list_evals(dataset_id):
            if (
                kinds.get(e.get("dataset_id")) in ("embedding", "reranker")
                and e.get("method") != RANKING_METHOD
            ):
                continue
            model = dict(e.get("model") or {})
            run = runs.get(model.get("run_id") or "")
            if model.get("type") == "run" and run:
                model["label"] = f"{run['label']} · {run['short_base']}"
            out.append({**e, "model": model})
        return out

    def activate(self, run_id: str, use: str) -> Dict[str, Any]:
        """Record a decision model as the one for ``use`` (memory_routing or
        relevance): ``route`` and the MCP decide tool read it. Embedding
        models and rerankers are put to work with ``swap``, which also
        re-embeds the stored memories or sets the reranker."""
        if use in ("embedding", "reranker"):
            card = "embedding" if use == "embedding" else "reranking"
            raise FineTuningError(
                f"Put a {use} model to work with swap (memorizz finetune swap {card} <run_id>); "
                "activating alone would not change what MemoRizz uses"
            )
        return self._record_active(run_id, use)

    def _record_active(self, run_id: str, use: str) -> Dict[str, Any]:
        if use not in USES:
            raise FineTuningError(f"use must be one of {', '.join(USES)}")
        run = self.store.get_run(run_id)
        if run is None or run.kind != USES[use]:
            raise FineTuningError(f"A {use} model must come from a {USES[use]} run")
        record = {
            "run_id": run_id,
            "model_dir": self.model_dir(run_id),
            "base_model": run.base_model,
            "host": socket.gethostname(),
            "activated_at": utcnow(),
        }
        self.store.set_setting(f"active:{use}", record)
        return record

    # -- the simple view: is the fine-tuned model better, and swap it in --------

    # One card per job a model does in the memory layer, and the uses
    # (activation slots) whose models can do it.
    CARDS = {
        "memory_routing": ("memory_routing",),
        "embedding": ("embedding",),
        "reranking": ("reranker", "relevance"),
    }
    CARD_TITLES = {
        "memory_routing": "Memory routing",
        "embedding": "Memory embeddings",
        "reranking": "Memory reranking",
    }
    CARD_NOTES = {
        "memory_routing": 'Decides which store a new memory belongs in when a write asks for memory_type "auto".',
        "embedding": "Embeds every memory and query; swapping re-embeds the stored memories.",
        "reranking": "Rescores the memories vector search found, so the best evidence comes first.",
    }
    SPLIT_WORDS = {
        "test": "test",
        "validation": "validation",
        "held_out": "test and validation",
        "all": "all",
    }

    def _use_of(self, run: Dict[str, Any]) -> Optional[str]:
        if run["kind"] == "decision":
            return (
                run["task"] if run["task"] in ("memory_routing", "relevance") else None
            )
        return run["kind"] if run["kind"] in ("embedding", "reranker") else None

    def _card_of(self, use: Optional[str]) -> Optional[str]:
        for card, uses in self.CARDS.items():
            if use in uses:
                return card
        return None

    def _evidence(
        self,
        run: Dict[str, Any],
        evaluations: List[Dict[str, Any]],
        names: Dict[str, str],
        in_use_run: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The fairest comparison available for a run, pooled over every
        fixed set it and the model it is compared with were both scored on:
        each example (or query) counts once, so a gain on one set and a loss
        on another both show, and the paired test sees all of them. A set
        counts when nothing was trained on it: a test set, or a dataset's
        test (or validation) examples, never its "all" split. The run is
        compared with the fine-tuned version in use (``in_use_run``) when
        that was scored on the same sets, else with what is in use now (an
        external model), else with its untrained base. Without such scores,
        its own before/after from training, scored the same way."""
        metric = self._primary_metric(run["kind"], run["task"])
        test_sets = {d.dataset_id for d in self.store.list_datasets() if d.test_set}

        def latest_of(match) -> Dict[tuple, Dict[str, Any]]:
            out: Dict[tuple, Dict[str, Any]] = {}
            for r in evaluations:  # oldest first: the newest on a set wins
                fair = r["split"] in ("test", "validation", "held_out") or (
                    r["dataset_id"] in test_sets
                )
                if fair and match(r):
                    out[(r["dataset_id"], r["split"])] = r
            return out

        # An embedding model is scored at the size a swap would store it at
        # (cut to the store's size, or its full size); the rest at their own.
        cut = self._scored_size(run)

        def sized(r) -> bool:
            return (r["model"].get("dimensions") or None) == cut

        own = latest_of(
            lambda r: r["model"].get("run_id") == run["run_id"] and sized(r)
        )
        current = (
            latest_of(lambda r: r["model"].get("run_id") == in_use_run and sized(r))
            if in_use_run and in_use_run != run["run_id"]
            else {}
        )
        external = latest_of(lambda r: r["model"].get("type") == "in_use")
        exact = latest_of(
            lambda r: r["model"].get("type") == "base"
            and r["model"].get("base_model") == run["base_model"]
            and sized(r)
        )
        base = {
            **latest_of(lambda r: r["model"].get("type") == "base" and sized(r)),
            **exact,
        }
        jev = latest_of(lambda r: r["model"].get("type") == "jev")
        compared = None
        for kind, table in (
            ("current", current),
            ("external", external),
            ("base", base),
        ):
            shared = [k for k in own if k in table]
            if shared:
                compared = (kind, table, shared)
                break
        if compared:
            kind, table, shared = compared
            key = metric if metric in own[shared[0]]["metrics"] else "accuracy"
            itemised = [
                k
                for k in shared
                if isinstance(own[k]["metrics"].get("per_item"), dict)
                and isinstance(table[k]["metrics"].get("per_item"), dict)
            ]
            sets = itemised or [max(shared, key=lambda k: own[k]["count"] or 0)]

            def pooled(rows) -> Optional[Dict[str, float]]:
                if not all(k in rows for k in sets):
                    return None
                if not all(
                    isinstance(rows[k]["metrics"].get("per_item"), dict) for k in sets
                ):
                    return None
                return {
                    f"{k[0]}:{k[1]}:{i}": float(v)
                    for k in sets
                    for i, v in rows[k]["metrics"]["per_item"].items()
                }

            def value(rows) -> Optional[float]:
                """A model's score on exactly these sets (None if it was not
                scored on all of them)."""
                if not all(k in rows for k in sets):
                    return None
                items = pooled(rows)
                if items is not None and ids:
                    shared_ids = [i for i in ids if i in items]
                    if len(shared_ids) == len(ids):
                        return sum(items[i] for i in shared_ids) / len(shared_ids)
                total = sum(rows[k]["count"] or 0 for k in sets)
                if not total:
                    return None
                return (
                    sum(
                        (rows[k]["metrics"].get(key) or 0) * (rows[k]["count"] or 0)
                        for k in sets
                    )
                    / total
                )

            mine, theirs = pooled(own), pooled(table)
            ids = sorted(set(mine) & set(theirs)) if mine and theirs else []
            fine_tuned, against = value(own), value(table)
            significance = None
            if ids:
                from .evaluate import paired_test

                significance = paired_test(
                    {i: mine[i] for i in ids}, {i: theirs[i] for i in ids}
                )
            newest = table[sets[-1]]
            if kind == "current":
                in_use_entry = {
                    "label": f"{newest['model']['label']} (in use now)",
                    "model": newest["model"]["label"],
                    "when": "in use now",
                    "value": against,
                }
            elif kind == "external":
                in_use_entry = {**self._in_use_label(newest["model"]), "value": against}
            else:
                in_use_entry = None
            reference = value(jev) if jev else None
            set_names = [own[k]["dataset_name"] for k in sets]
            splits = {k[1] for k in sets}
            return {
                "significance": significance,
                "metric": key,
                "fine_tuned": fine_tuned,
                "base": against if kind == "base" else value(base),
                "base_label": f"{run['short_base']}, untrained",
                "base_kind": "untrained",
                "in_use": in_use_entry,
                "reference": {
                    "label": jev[sets[0]]["model"]["label"],
                    "value": reference,
                }
                if reference is not None
                else None,
                "count": len(ids) if ids else own[sets[0]]["count"],
                "candidates": own[sets[0]].get("candidates")
                if len(sets) == 1
                else None,
                "dataset_id": sets[0][0],
                "set_name": set_names[0]
                if len(sets) == 1
                else ", ".join(f"“{n}”" for n in set_names),
                "sets": [
                    {
                        "dataset_id": k[0],
                        "name": own[k]["dataset_name"],
                        "split": k[1],
                        "count": own[k]["count"],
                    }
                    for k in sets
                ],
                "split": next(iter(splits)) if len(splits) == 1 else "held-out",
                "set": " + ".join(f"{n}" for n in set_names),
                "frozen": True,
                "dimensions": cut,
            }
        before, after = run.get("metrics_before") or {}, run.get("metrics_after") or {}
        key = metric if metric in after else "accuracy"
        counts = (run.get("progress") or {}).get("counts") or {}
        split = "test" if counts.get("test") else "validation"
        name = names.get(run["dataset_id"], "its dataset")
        # A continued run's "before" is the version it started from.
        started = run.get("parent_label")
        return {
            "significance": self._significance(after, before),
            "metric": key,
            "fine_tuned": after.get(key),
            "base": before.get(key),
            "base_label": started or f"{run['short_base']}, untrained",
            "base_kind": "parent" if started else "untrained",
            "in_use": None,
            "reference": None,
            "count": after.get("count") or after.get("queries") or counts.get(split),
            "candidates": None,
            "dataset_id": run["dataset_id"],
            "set_name": name,
            "split": split,
            "set": f"{name} ({split}), measured in training",
            "frozen": False,
        }

    @staticmethod
    def _significance(
        mine: Dict[str, Any], other: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Could mine − other be chance? The paired sign-flip test
        (``evaluate.paired_test``) over the examples or queries both were
        scored on; None without per-example scores (recorded before they
        were kept) or with fewer than 5 shared."""
        from .evaluate import paired_test

        a, b = mine.get("per_item"), other.get("per_item")
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        return paired_test(a, b)

    @staticmethod
    def _in_use_label(model: Dict[str, Any]) -> Dict[str, str]:
        """An "in use" score is about the setup at the time it was measured;
        after a swap, say so."""
        from .reranking import RERANKER_ENV

        label = str(model.get("label") or "").replace(" (in use now)", "")
        if label.startswith("No reranker: "):  # recorded before the wording changed
            label = f"Vector search only ({label[len('No reranker: '):]})"
        same_embedding = str(model.get("model") or "") == os.environ.get(
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL", ""
        )
        no_reranker = label.startswith("Vector search only")
        unchanged = same_embedding and not (
            no_reranker and os.environ.get(RERANKER_ENV, "").strip()
        )
        when = "in use now" if unchanged else "before the swap"
        return {"label": f"{label} ({when})", "model": label, "when": when}

    def _in_use(
        self, card: str, active: Dict[str, Any], labels: Dict[str, str]
    ) -> Dict[str, Any]:
        """What does this job now, in words."""

        def fine_tuned(record):
            name = labels.get(record["run_id"], record["run_id"])
            return {
                "run_id": record["run_id"],
                "label": f"{name} · {record['base_model'].rstrip('/').split('/')[-1]} (fine-tuned)",
            }

        if card == "memory_routing":
            record = active.get("memory_routing")
            if record:
                return fine_tuned(record)
            return {"run_id": None, "label": "No router: each write names its store"}
        if card == "embedding":
            record = active.get("embedding")
            model = os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "")
            if record and model and Path(model) == Path(record["model_dir"]):
                return fine_tuned(record)
            provider = (
                os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "") or "default"
            )
            return {"run_id": None, "label": f"{provider} · {model or 'default model'}"}
        from .reranking import RERANKER_ENV

        path = os.environ.get(RERANKER_ENV, "").strip()
        for use in self.CARDS["reranking"]:
            record = active.get(use)
            if record and path and Path(path) == Path(record["model_dir"]):
                return fine_tuned(record)
        if path:
            return {"run_id": None, "label": f"Reranker at {path}"}
        return {"run_id": None, "label": "Vector search only (no reranker)"}

    def _candidate(
        self, run: Dict[str, Any], evidence: Dict[str, Any], in_use: Dict[str, Any]
    ) -> Dict[str, Any]:
        """A fine-tuned model, its comparison and a plain verdict.

        ``verdict`` is "better" or "worse" only when a paired test says the
        difference is unlikely to be chance (p < 0.05), "unclear" when it
        could be, "untested" when no per-example scores were kept to test
        it, and "none" with nothing to compare."""
        against = (
            evidence["in_use"]["value"]
            if evidence.get("in_use")
            else evidence.get("base")
        )
        gain = (
            None
            if against is None or evidence.get("fine_tuned") is None
            else round((evidence["fine_tuned"] - against) * 100, 1)
        )
        return {
            "run_id": run["run_id"],
            "dataset_id": run["dataset_id"],
            "kind": run["kind"],
            "version": run["label"],
            "label": f"{run['label']} · {run['short_base']}",
            "short_base": run["short_base"],
            "lineage": run["lineage"],
            "base_model": run["base_model"],
            "evidence": evidence,
            "gain_points": gain,
            "compared_with": (
                (
                    "the model in use now"
                    if evidence["in_use"]["when"] == "in use now"
                    else "the setup before the swap"
                )
                if evidence.get("in_use")
                else "the version it started from"
                if evidence.get("base_kind") == "parent"
                else "the untrained base model"
            ),
            "speed": self.store.get_setting(f"speed:{run['run_id']}"),
            "verdict": self._verdict(gain, evidence.get("significance")),
            "better": self._verdict(gain, evidence.get("significance")) == "better",
            "in_use": in_use.get("run_id") == run["run_id"],
        }

    @staticmethod
    def _verdict(gain: Optional[float], test: Optional[Dict[str, float]]) -> str:
        if gain is None:
            return "none"
        if test is None:
            return "untested"
        if test["p"] < 0.05 and gain > 0:
            return "better"
        if test["p"] < 0.05 and gain < 0:
            return "worse"
        return "unclear"

    def overview(self) -> List[Dict[str, Any]]:
        """One card per job: what does it now, the best fine-tuned model for
        it, how the two compare on held-out examples, and a plain verdict."""
        everything = self.runs()
        labels = {r["run_id"]: r["label"] for r in everything}
        runs = [r for r in everything if r["status"] == RunStatus.SUCCEEDED.value]
        evaluations = self.evaluations()
        names = {d.dataset_id: d.name for d in self.store.list_datasets()}
        active = self.active()
        out = []
        for card, uses in self.CARDS.items():
            mine = [r for r in runs if self._use_of(r) in uses]
            in_use = self._in_use(card, active, labels)
            scored = []
            for run in mine:
                evidence = self._evidence(run, evaluations, names, in_use.get("run_id"))
                if evidence.get("fine_tuned") is None:
                    continue
                scored.append((run, self._candidate(run, evidence, in_use)))
            # The best version: scored on a fixed set, next to what runs now
            # (or itself in use), then by its score.
            scored.sort(
                key=lambda item: (
                    item[1]["evidence"]["frozen"],
                    item[1]["evidence"]["in_use"] is not None or item[1]["in_use"],
                    item[1]["evidence"]["fine_tuned"],
                    item[0]["created_at"],
                ),
                reverse=True,
            )
            out.append(
                {
                    "use": card,
                    "title": self.CARD_TITLES[card],
                    "note": self.CARD_NOTES[card],
                    "in_use": in_use,
                    "candidate": scored[0][1] if scored else None,
                    "runs": len(mine),
                    "can_swap_back": bool(in_use.get("run_id")),
                }
            )
        return out

    def verdict(self, run_id: str) -> Dict[str, Any]:
        """The comparison a run's card would show, for any finished run: the
        same numbers wherever the run appears."""
        everything = self.runs()
        run = next((r for r in everything if r["run_id"] == run_id), None)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        names = {d.dataset_id: d.name for d in self.store.list_datasets()}
        card = self._card_of(self._use_of(run))
        in_use = (
            self._in_use(
                card, self.active(), {r["run_id"]: r["label"] for r in everything}
            )
            if card
            else {"run_id": None, "label": ""}
        )
        evidence = self._evidence(run, self.evaluations(), names, in_use.get("run_id"))
        return {
            **self._candidate(run, evidence, in_use),
            # Its own before/after in training, tested the same way.
            "training_significance": self._significance(
                run.get("metrics_after") or {}, run.get("metrics_before") or {}
            ),
            "card": card,
            "card_title": self.CARD_TITLES.get(card or ""),
            "in_use_now": in_use,
        }

    # -- per-agent choices and what a swap affects ----------------------------

    def agent_models(self) -> List[Dict[str, Any]]:
        """Every agent in the store with its own choice per job (memory
        routing, reranking), or None where it follows the store's."""
        from . import agent_models as per_agent
        from .store import SETTING_RECORD

        chosen = {
            row.get("key"): row.get("value")
            for row in self.store._list(SETTING_RECORD)
            if str(row.get("key") or "").startswith("agent:")
        }
        out = []
        for agent in self.provider.list_memagents() or []:
            data = agent if isinstance(agent, dict) else {}
            agent_id = str(data.get("agent_id") or getattr(agent, "agent_id", "") or "")
            if not agent_id:
                continue
            out.append(
                {
                    "agent_id": agent_id,
                    "name": data.get("name")
                    or getattr(agent, "name", None)
                    or agent_id,
                    "memory_ids": list(
                        data.get("memory_ids")
                        or getattr(agent, "memory_ids", None)
                        or []
                    ),
                    "choices": {
                        job: chosen.get(per_agent.setting_key(agent_id, job))
                        for job in per_agent.JOBS
                    },
                }
            )
        return sorted(out, key=lambda a: str(a["name"]).lower())

    def set_agent_model(
        self, agent_id: str, job: str, choice: Optional[str]
    ) -> Dict[str, Any]:
        """Give one agent its own memory router or reranker: a run id, "off"
        for none, or "default" (None) to follow the store again. Embeddings
        are one model per store, so they cannot be chosen per agent."""
        from . import agent_models as per_agent

        if job not in per_agent.JOBS:
            raise FineTuningError(
                "An agent can have its own memory_routing or reranking model; "
                "embeddings are one model per store"
            )
        if not any(a["agent_id"] == agent_id for a in self.agent_models()):
            raise FineTuningError(f"No agent {agent_id} in this memory store")
        key = per_agent.setting_key(agent_id, job)
        if choice in (None, "", "default"):
            self.store.set_setting(key, None)
        elif choice == "off":
            self.store.set_setting(key, {"mode": "off"})
        else:
            run = self.run(choice)
            if run["job"] != job:
                raise FineTuningError(
                    f"{run['label']} is a {run['job_title'].lower()} model, not one for {job.replace('_', ' ')}"
                )
            self.store.set_setting(
                key,
                {
                    "mode": "run",
                    "run_id": run["run_id"],
                    "model_dir": self.model_dir(run["run_id"]),
                    "label": f"{run['label']} · {run['short_base']}",
                },
            )
        per_agent.forget()
        return next(a for a in self.agent_models() if a["agent_id"] == agent_id)

    def swap_impact(self, card: str) -> Dict[str, Any]:
        """What swapping the model for ``card`` touches: the store, the agents
        in it (and those that keep their own choice), and the other MemoRizz
        processes on this machine using the same store, which must restart
        to pick up an embedding model (or they search re-embedded memories
        with the old one) or a reranker."""
        from .._env_io import memorizz_home
        from . import clients, staging

        if card not in self.CARDS:
            raise FineTuningError(f"Swap one of: {', '.join(self.CARDS)}")
        store = clients.store_identity(self.provider)
        agents = self.agent_models()
        own = [
            {"agent_id": a["agent_id"], "name": a["name"], "choice": a["choices"][card]}
            for a in agents
            if card in ("memory_routing", "reranking") and a["choices"].get(card)
        ]
        others = [c for c in clients.clients(store["id"]) if c["pid"] != os.getpid()]
        return {
            "card": card,
            "store": store,
            "home": str(memorizz_home()),
            "agents": {
                "total": len(agents),
                "follow_store": len(agents) - len(own),
                "own_choice": own,
                "names": [a["name"] for a in agents][:60],
            },
            "processes": others,
            "unregistered": clients.unregistered(store),
            "server": clients._server(store),
            # The router is read from the store on every write; the embedding
            # model and the reranker are read when a process starts.
            "restart_needed": card in ("embedding", "reranking"),
            "reembeds": card == "embedding",
            # Embeddings: re-embedded beside the live vectors (then one
            # switch), and the vectors a swap back would reuse.
            "staged": card == "embedding" and staging.supported(self.provider),
            "kept": self._kept_vectors() if card == "embedding" else None,
        }

    def _kept_vectors(self) -> Optional[Dict[str, Any]]:
        from . import staging

        if not staging.supported(self.provider):
            return None
        try:
            return self.provider.vector_previous()
        except Exception:
            return None

    def registry(self) -> Dict[str, Any]:
        """Every model MemoRizz knows, per job: what runs now, each
        fine-tuned version (lineage, data, settings, training result with its
        significance, every fixed-set score, files on disk) and the base
        models to start from. A view over the run and evaluation records
        already kept in the memory provider, not a second store."""
        everything = self.runs()
        evaluations = self.evaluations()
        datasets = {d.dataset_id: d for d in self.store.list_datasets()}
        cards = {c["use"]: c for c in self.overview()}
        cached = _cached_models()
        from .store import SETTING_RECORD

        speeds = {
            str(row.get("key"))[len("speed:") :]: row.get("value")
            for row in self.store._list(SETTING_RECORD)
            if str(row.get("key") or "").startswith("speed:")
        }
        jobs: Dict[str, Dict[str, Any]] = {}
        for key in self.CARDS:
            card = cards.get(key) or {}
            jobs[key] = {
                "key": key,
                "title": self.CARD_TITLES[key],
                "note": self.CARD_NOTES[key],
                "memory_job": True,
                "in_use": card.get("in_use"),
                "best_run_id": (card.get("candidate") or {}).get("run_id"),
                "kinds": sorted({USES[u] for u in self.CARDS[key]}),
                "versions": [],
                "failed": [],
            }
        for run in sorted(everything, key=lambda r: r["created_at"]):
            job = jobs.setdefault(
                run["job"],
                {
                    "key": run["job"],
                    "title": run["job_title"],
                    "note": "Not used by MemoRizz's memory layer: ask it from Python or the CLI.",
                    "memory_job": False,
                    "in_use": None,
                    "best_run_id": None,
                    "kinds": [run["kind"]],
                    "versions": [],
                    "failed": [],
                },
            )
            entry = self._registry_entry(run, evaluations, datasets, job)
            entry["speed"] = speeds.get(run["run_id"])
            job["failed" if run["version"] is None else "versions"].append(entry)
        agents = self.agent_models()
        for key in ("memory_routing", "reranking"):
            jobs[key]["agents"] = [
                {
                    "agent_id": a["agent_id"],
                    "name": a["name"],
                    "choice": a["choices"].get(key),
                }
                for a in agents
            ]
        jobs["embedding"]["agents_total"] = len(agents)
        for job in jobs.values():
            job["versions"].sort(key=lambda r: r["version"] or 0, reverse=True)
            job["bases"] = [
                {**m, **cached.get(m["model_id"], {"on_disk": False})}
                for kind in job["kinds"]
                for m in self.base_models(kind)
            ]
        return {
            "jobs": [
                j
                for j in jobs.values()
                if j["versions"] or j["failed"] or j["memory_job"]
            ],
            "totals": {
                "versions": sum(len(j["versions"]) for j in jobs.values()),
                "failed": sum(len(j["failed"]) for j in jobs.values()),
                "disk_mb": round(
                    sum(
                        v["model"]["size_mb"] or 0
                        for j in jobs.values()
                        for v in j["versions"]
                    ),
                    1,
                ),
            },
        }

    def _registry_entry(
        self,
        run: Dict[str, Any],
        evaluations: List[Dict[str, Any]],
        datasets: Dict[str, Dataset],
        job: Dict[str, Any],
    ) -> Dict[str, Any]:
        from datetime import datetime

        metric = self._primary_metric(run["kind"], run["task"])
        before, after = run.get("metrics_before") or {}, run.get("metrics_after") or {}
        progress = run.get("progress") or {}
        scores: Dict[tuple, Dict[str, Any]] = {}
        for e in evaluations:
            if e["model"].get("run_id") == run["run_id"]:
                scores[(e["dataset_id"], e["split"])] = e
        shown = []
        for e in scores.values():
            d = datasets.get(e["dataset_id"])
            key = metric if metric in e["metrics"] else "accuracy"
            shown.append(
                {
                    "dataset_id": e["dataset_id"],
                    "dataset_name": e["dataset_name"],
                    "split": e["split"],
                    "count": e["count"],
                    "metric": key,
                    "value": e["metrics"].get(key),
                    "test_set": bool(d and d.test_set),
                    "created_at": e.get("created_at"),
                }
            )
        shown.sort(key=lambda x: (not x["test_set"], -(x["count"] or 0)))
        folder = progress.get("model_dir") or (
            os.path.join(run["output_dir"], "model") if run.get("output_dir") else ""
        )
        size = None
        if folder and os.path.isdir(folder):
            size = sum(
                os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(folder)
                for name in names
            )

        def seconds(a, b):
            try:
                return round(
                    (
                        datetime.fromisoformat(b) - datetime.fromisoformat(a)
                    ).total_seconds()
                )
            except (TypeError, ValueError):
                return None

        replay = [
            {
                **item,
                "dataset_name": datasets[item["dataset_id"]].name
                if item.get("dataset_id") in datasets
                else item.get("dataset_id"),
            }
            for item in (progress.get("replay") or [])
        ]
        in_use = job.get("in_use") or {}
        return {
            "run_id": run["run_id"],
            "label": run["label"],
            "version": run["version"],
            "lineage": run["lineage"],
            "parent_run_id": run.get("parent_run_id") or "",
            "parent_label": run.get("parent_label") or "",
            "status": run["status"],
            "error": run.get("error") or "",
            "kind": run["kind"],
            "task": run["task"],
            "base_model": run["base_model"],
            "short_base": run["short_base"],
            "backbone": run.get("backbone"),
            "backend": run.get("backend"),
            "device": run.get("device"),
            "params": run.get("params") or {},
            "created_at": run["created_at"],
            "duration_s": seconds(run.get("started_at"), run.get("finished_at")),
            "dataset_id": run["dataset_id"],
            "dataset_name": datasets[run["dataset_id"]].name
            if run["dataset_id"] in datasets
            else "(dataset deleted)",
            "counts": progress.get("counts") or {},
            "replay": replay,
            "training": {
                "metric": metric if metric in after else "accuracy",
                "before": before.get(metric if metric in after else "accuracy"),
                "after": after.get(metric if metric in after else "accuracy"),
                "start": run.get("parent_label") or "untrained base",
                "significance": self._significance(after, before),
            },
            "scores": shown,
            "in_use": in_use.get("run_id") == run["run_id"],
            "best": job.get("best_run_id") == run["run_id"],
            "model": {
                "dir": folder,
                "on_disk": size is not None,
                "size_mb": round(size / 1e6, 1) if size is not None else None,
            },
        }

    # -- speed: what a swap costs or saves per call --------------------------

    SPEED_UNITS = {
        "memory_routing": "per memory routed",
        "embedding": "per query embedded",
        "reranking": "per query, 10 memories reranked",
    }

    def speed(
        self, run_id: str, *, progress: Progress = None, samples: int = 20
    ) -> Dict[str, Any]:
        """Time a version doing what it does in use (route one memory, embed
        one query, rerank one query's 10 candidates) next to what it would
        replace: the version in use now or, when this one is in use, what
        Swap back restores. Same inputs (its test examples first), each model
        loaded the way the store loads it, no result cache; a warm-up pass
        over every input, then one call at a time.

        Fine-tuning keeps the base model's size, so a version runs as fast as
        its untrained base: speed changes only with what runs. Kept with the
        run, so its card and the registry can show it."""
        import socket as _socket
        import statistics
        import time as _time

        run = self.run(run_id)
        card = self._card_of(self._use_of(run))
        inputs = self._speed_inputs(run, samples)
        if len(inputs) < 3:
            raise FineTuningError("Not enough accepted examples to time it on")
        relevance = run["kind"] == "decision" and card == "reranking"
        plan = [
            (
                "fine-tuned",
                f"{run['label']} · {run['short_base']}",
                lambda: self._timer(card, run=run),
            )
        ]
        if card and not relevance:
            plan.append(self._speed_rival(run, card))
        rows = []
        for index, (role, label, make) in enumerate(plan):
            if progress:
                progress(index, len(plan), f"Timing {role}…")
            timer, device = make()
            row = {"role": role, "label": label, "n": 0, "device": device}
            if timer is not None:
                # Warm up on every input first: the GPU compiles once per
                # input shape, which no call in use pays twice.
                for item in inputs:
                    timer(item)
                times = []
                for item in inputs:
                    started = _time.perf_counter()
                    timer(item)
                    times.append((_time.perf_counter() - started) * 1000)
                times.sort()
                row.update(
                    median_ms=round(statistics.median(times), 2),
                    p95_ms=round(times[min(len(times) - 1, int(0.95 * len(times)))], 2),
                    n=len(times),
                )
            rows.append(row)
            self._models.clear()
        result = {
            "run_id": run_id,
            "job": run["job"],
            "unit": "per memory judged"
            if relevance
            else self.SPEED_UNITS.get(card or "", "per decision"),
            "host": _socket.gethostname(),
            "measured_at": utcnow(),
            "rows": rows,
        }
        self.store.set_setting(f"speed:{run_id}", result)
        return result

    def _speed_rival(self, run: Dict[str, Any], card: str):
        """``(role, label, make timer)`` for what ``run`` would replace."""
        from . import swap as swapping

        everything = self.runs()
        by_id = {r["run_id"]: r for r in everything}
        in_use = self._in_use(
            card, self.active(), {r["run_id"]: r["label"] for r in everything}
        )
        nothing = lambda: (None, "nothing runs")  # noqa: E731
        if in_use.get("run_id") and in_use["run_id"] != run["run_id"]:
            current = by_id[in_use["run_id"]]
            return "in use now", in_use["label"], lambda: self._timer(card, run=current)
        if in_use.get("run_id") != run["run_id"]:
            # Nothing fine-tuned is in use: the store's own model, if any.
            label = in_use.get("label") or ""
            if card == "embedding":
                settings = swapping.current_settings()
                return "in use now", label, lambda: self._timer(card, settings=settings)
            if card == "reranking":
                from .reranking import configured_reranker

                path = configured_reranker()
                return (
                    "in use now",
                    label,
                    (lambda: self._timer(card, path=path)) if path else nothing,
                )
            return "in use now", label or "No router", nothing
        previous = self.store.get_setting(f"previous:{card}") or {}
        before = by_id.get(previous.get("run_id") or "")
        if before is not None:
            return (
                "before the swap",
                f"{before['label']} · {before['short_base']}",
                lambda: self._timer(card, run=before),
            )
        if card == "embedding" and previous.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"):
            name = (
                str(previous.get("MEMORIZZ_DEFAULT_EMBEDDING_MODEL") or "")
                .rstrip("/")
                .split("/")[-1]
            )
            return "before the swap", name, lambda: self._timer(card, settings=previous)
        if card == "reranking" and previous.get("path"):
            name = Path(previous["path"]).name
            return (
                "before the swap",
                name,
                lambda: self._timer(card, path=previous["path"]),
            )
        return (
            "before the swap",
            {"memory_routing": "No router", "reranking": "No reranker"}.get(
                card, "none"
            ),
            nothing,
        )

    def _speed_inputs(self, run: Dict[str, Any], count: int) -> List[Any]:
        """Inputs to time on: the run's test examples (then any accepted)."""
        rows = self.store.examples(
            run["dataset_id"], status=ExampleStatus.ACCEPTED.value
        )
        rows = sorted(rows, key=lambda e: e.split != "test")[:count]
        if run["kind"] == "decision":
            return [e.input["state"] for e in rows]
        pool = [str(e.input["positive"]) for e in rows if e.input.get("positive")]
        out = []
        for e in rows:
            if not e.input.get("query"):
                continue
            candidates = [
                str(e.input["positive"]),
                *map(str, e.input.get("negatives") or []),
            ]
            candidates += [p for p in pool if p not in candidates][
                : max(0, 10 - len(candidates))
            ]
            out.append((str(e.input["query"]), candidates[:10]))
        return out

    def _timer(
        self,
        card: Optional[str],
        *,
        run: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, str]] = None,
        path: str = "",
    ):
        """``(call one input, device)`` for a model loaded the way the store
        loads it in use: a run's model, embedding ``settings`` or a reranker
        folder at ``path``."""
        from .training.common import pick_device

        device = pick_device(None)
        if run is not None and run["kind"] == "decision":
            model = self._decision_model(run["run_id"])
            return (lambda state: model.decide(state)), device
        if card == "embedding":
            from ..embeddings import (
                EmbeddingManager,
                embed_as_query,
                embedding_settings_from,
            )

            if settings is None:
                settings = {
                    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "huggingface",
                    "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": self.model_dir(run["run_id"]),
                }
            name, config = embedding_settings_from(settings)
            if not name:
                return None, "nothing runs"
            manager = EmbeddingManager(name, config)
            # The provider itself: the manager would answer repeats from its cache.
            return (
                lambda item: embed_as_query(manager._provider.get_embedding, item[0])
            ), (device if name == "huggingface" else name)
        from .reranking import load_scorer

        scorer = load_scorer(path or self.model_dir(run["run_id"]))
        return (lambda item: scorer(item[0], item[1])), device

    def compare(
        self, run_id: str, *, progress: Progress = None
    ) -> List[Dict[str, Any]]:
        """Score a fine-tuned model, its untrained base and (embeddings and
        rerankers) the setup in use now on the run's test examples, scored
        as in training: the comparison its card shows."""
        run = self.run(run_id)
        dataset = self._dataset(run["dataset_id"])
        split = "all" if dataset.test_set else "test"
        if split == "test" and not self.store.counts(dataset.dataset_id)["split"].get(
            "test"
        ):
            split = "validation"
        base = {"base_model": run["base_model"]}
        if (run.get("params") or {}).get("max_length"):
            base["max_length"] = int(run["params"]["max_length"])
        steps = [
            ("the fine-tuned model", {"run_id": run_id}),
            ("the untrained base model", base),
        ]
        if run["kind"] == "embedding":
            # Cut to the store's size, as a swap that keeps it would.
            plan = self._plan_for(run)
            for size in [
                o["dimensions"] for o in plan["options"] if o["kind"] == "cut"
            ]:
                steps += [
                    (
                        f"the fine-tuned model at {size} dimensions",
                        {"run_id": run_id, "dimensions": size},
                    ),
                    (
                        f"the untrained base model at {size} dimensions",
                        {**base, "dimensions": size},
                    ),
                ]
        if run["kind"] != "decision":
            steps.append(("the setup in use now", {"in_use": True}))
        results = []
        for i, (label, which) in enumerate(steps):
            if progress:
                progress(i, len(steps), f"Scoring {label}…")
            results.append(self.evaluate(dataset.dataset_id, split=split, **which))
        return results

    def swap(
        self,
        card: str,
        run_id: str,
        *,
        persist: bool = True,
        progress: Optional[Callable[[str], None]] = None,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Put a fine-tuned model to work: the router for ``memory_type="auto"``
        writes, the embedding model (stored memories are re-embedded beside
        the live ones, at ``dimensions`` from ``embedding_plan``, default the
        plan's), or the reranker for recall and search. ``swap_back`` undoes
        it."""
        if card not in self.CARDS:
            raise FineTuningError(f"Swap one of: {', '.join(self.CARDS)}")
        run = self.store.get_run(run_id)
        if run is None or run.status != RunStatus.SUCCEEDED.value:
            raise FineTuningError(f"Run {run_id} has no finished model")
        use = self._use_of(run.to_dict())
        if use not in self.CARDS[card]:
            raise FineTuningError(
                f"Run {run_id} is a {run.kind} model, not one for {self.CARD_TITLES[card].lower()}"
            )
        say = progress or (lambda message: None)
        active = self.active()
        if card == "memory_routing":
            if "previous:memory_routing" not in self._settings_keys():
                self.store.set_setting(
                    "previous:memory_routing",
                    active.get("memory_routing") or {"run_id": None},
                )
            self.activate(run_id, "memory_routing")
            return {"swapped": card, "run_id": run_id}
        if card == "reranking":
            from .reranking import RERANKER_ENV

            if "previous:reranking" not in self._settings_keys():
                self.store.set_setting(
                    "previous:reranking", {"path": os.environ.get(RERANKER_ENV, "")}
                )
            say("Loading the reranker…")
            from .reranking import load_scorer

            load_scorer(self.model_dir(run_id))  # fail before switching
            self._set_env({RERANKER_ENV: self.model_dir(run_id)}, persist)
            for other in self.CARDS["reranking"]:
                if other != use:
                    self.deactivate(other)
            self._record_active(run_id, use)
            return {"swapped": card, "run_id": run_id}
        return self._swap_embedding(
            run, persist=persist, say=say, dimensions=dimensions
        )

    def swap_back(
        self,
        card: str,
        *,
        persist: bool = True,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Return a job to the model it had before the first swap."""
        if card not in self.CARDS:
            raise FineTuningError(f"Swap one of: {', '.join(self.CARDS)}")
        say = progress or (lambda message: None)
        previous = self.store.get_setting(f"previous:{card}")
        if card == "memory_routing":
            if previous and previous.get("run_id"):
                self.activate(previous["run_id"], "memory_routing")
            else:
                self.deactivate("memory_routing")
        elif card == "reranking":
            from .reranking import RERANKER_ENV

            self._set_env({RERANKER_ENV: (previous or {}).get("path", "")}, persist)
            for use in self.CARDS["reranking"]:
                self.deactivate(use)
        else:
            if not previous:
                raise FineTuningError(
                    "No earlier embedding model is recorded to swap back to"
                )
            result = self._swap_back_embedding(previous, persist=persist, say=say)
            self.deactivate("embedding")
            self.store.set_setting("previous:embedding", None)
            return {"swapped_back": card, **result}
        self.store.set_setting(f"previous:{card}", None)
        return {"swapped_back": card}

    # embedding swaps re-embed the stored memories

    # Families trained to keep their quality when their vectors are cut short.
    MATRYOSHKA = ("embeddinggemma", "qwen3-embedding")
    GEMMA_SIZES = (128, 256, 512, 768)

    def _store_size(self, *, fresh: bool = False) -> Optional[int]:
        """The vector size the store holds now (cached 30 s)."""
        from ..embeddings import get_embedding_manager
        from . import swap as swapping

        cached = getattr(self, "_store_size_cache", None)
        if cached and not fresh and time.monotonic() - cached[0] < 30:
            return cached[1]
        try:
            size = swapping.store_dimensions(self.provider, get_embedding_manager())
        except Exception:
            size = None
        self._store_size_cache = (time.monotonic(), size)
        return size

    def _native_size(self, run_id: str) -> int:
        from . import swap as swapping

        sizes = self.__dict__.setdefault("_native_sizes", {})
        if run_id not in sizes:
            sizes[run_id] = swapping.model_dimensions(self.model_dir(run_id))
        return sizes[run_id]

    def embedding_plan(self, run_id: str) -> Dict[str, Any]:
        """The vector sizes a fine-tuned embedding model can be swapped in at.

        A store whose schema pins one size (Oracle VECTOR columns, a MongoDB
        vector index) keeps it when the model can be cut to it (EmbeddingGemma
        and Qwen3-Embedding keep their quality at smaller sizes): the default.
        A store that can change size (Oracle, the filesystem) can instead
        take the model's full size, re-embedded beside the live vectors.
        Scores on the card are measured at the default size."""
        return self._plan_for(self.run(run_id))

    def _plan_for(self, run: Dict[str, Any]) -> Dict[str, Any]:
        from . import staging

        run_id = run["run_id"]
        if run["kind"] != "embedding":
            raise FineTuningError(f"Run {run_id} is not an embedding model")
        caps = staging.capabilities(self.provider)
        native = self._native_size(run_id)
        family = resolve_base(run["base_model"], "embedding").family
        cuttable = family in self.MATRYOSHKA
        store = self._store_size()
        options: List[Dict[str, Any]] = []
        reason = None
        if not caps["fixed_size"] or store in (None, native):
            # Any size fits: the model's full size, or its vectors cut short.
            change = store not in (None, native)
            options.append(
                {
                    "dimensions": native,
                    "kind": "resize" if change else "keep",
                    "text": f"{native} dimensions, the model's full size",
                }
            )
            if (
                change
                and store < native
                and cuttable
                and (family != "embeddinggemma" or store in self.GEMMA_SIZES)
            ):
                options.append(
                    {
                        "dimensions": store,
                        "kind": "cut",
                        "text": f"Keep {store} dimensions (vectors cut from {native})",
                    }
                )
        else:
            fits = store < native and cuttable
            if family == "embeddinggemma" and store not in self.GEMMA_SIZES:
                fits = False
            if fits:
                options.append(
                    {
                        "dimensions": store,
                        "kind": "cut",
                        "text": f"Keep this store's {store} dimensions (vectors cut from {native})",
                    }
                )
            if caps["resize"]:
                options.append(
                    {
                        "dimensions": native,
                        "kind": "resize",
                        "text": f"Change this store to {native} dimensions, the model's full size",
                    }
                )
            if not options:
                reason = (
                    f"This store holds {store}-dimension vectors and the model makes {native}; "
                    + (
                        "it cannot change size. Train from a base model that makes at least "
                        f"{store}, or start a new store."
                        if store > native or not cuttable
                        else f"EmbeddingGemma vectors come in {', '.join(map(str, self.GEMMA_SIZES))}."
                    )
                )
        return {
            "run_id": run_id,
            "store": store,
            "native": native,
            "cuttable": cuttable,
            **caps,
            "options": options,
            "default": options[0]["dimensions"] if options else None,
            "reason": reason,
        }

    def _scored_size(self, run: Dict[str, Any]) -> Optional[int]:
        """The cut size a run's card is scored at (None: its full size)."""
        if run.get("kind") != "embedding" or run.get("status") != "succeeded":
            return None
        try:
            plan = self._plan_for(run)
        except Exception:
            return None
        size = plan["default"]
        return size if size and size < plan["native"] else None

    def _embedding_manager(self, values: Dict[str, str]):
        """An embedding model for ``values`` (not yet the one in use),
        checked on a probe so a model that does not load changes nothing."""
        from . import swap as swapping

        try:
            manager = swapping.manager_for(values)
            probe = manager.get_embedding("memorizz embedding check")
            if not probe:
                raise RuntimeError("the model returned an empty vector")
        except Exception as exc:
            raise FineTuningError(
                f"The embedding model did not load ({exc}); nothing was changed"
            ) from None
        return manager, len(probe)

    def _swap_embedding(
        self,
        run: TrainingRun,
        *,
        persist: bool,
        say,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        from . import staging
        from . import swap as swapping

        say("Checking the vector size this store holds…")
        self._store_size(fresh=True)
        plan = self.embedding_plan(run.run_id)
        sizes = [o["dimensions"] for o in plan["options"]]
        size = int(dimensions or plan["default"] or 0)
        if size not in sizes:
            raise FineTuningError(
                plan["reason"]
                or f"This store can take this model at {' or '.join(map(str, sizes))} dimensions, not {size}"
            )
        values = {
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "huggingface",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": self.model_dir(run.run_id),
            "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS": str(size),
        }
        before = swapping.current_settings()
        if not self.store.get_setting("previous:embedding"):
            self.store.set_setting("previous:embedding", before)
        say(f"Loading the fine-tuned model ({size} dimensions)…")
        if not plan["staged"]:
            return self._swap_in_place(
                run, values, before, size, persist=persist, say=say
            )
        manager, made = self._embedding_manager(values)
        try:
            report = staging.migrate(
                self.provider, manager, dimensions=size, progress=say
            )
        except Exception as exc:
            raise FineTuningError(
                f"The swap stopped before switching; nothing changed ({exc})"
            ) from None
        self._use_embedding(values, manager, persist=persist)
        # The vectors kept as previous were made by the model in use before.
        self.store.set_setting(
            "vectors:previous", {**before, "dimensions": plan["store"]}
        )
        self._record_active(run.run_id, "embedding")
        self._store_size(fresh=True)
        return {
            "swapped": "embedding",
            "run_id": run.run_id,
            "dimensions": size,
            "staged": True,
            "reembedded": report,
        }

    def _swap_in_place(self, run, values, before, size, *, persist: bool, say):
        """Stores without a staged slot: switch, then re-embed in place."""
        self._switch_embedding(values, persist=persist, say=say)
        try:
            report = self._reembed(say)
            self._record_active(run.run_id, "embedding")
        except Exception as exc:
            # All or nothing: put the earlier model back and its vectors too.
            logger.exception("Embedding swap failed; restoring the earlier model")
            say("The swap failed; restoring the earlier model…")
            self._switch_embedding(before, persist=persist, say=say)
            self._reembed(say)
            raise FineTuningError(f"The swap failed and was undone: {exc}") from None
        return {
            "swapped": "embedding",
            "run_id": run.run_id,
            "dimensions": size,
            "staged": False,
            "reembedded": report,
        }

    def _swap_back_embedding(
        self, previous: Dict[str, str], *, persist: bool, say
    ) -> Dict[str, Any]:
        from . import staging

        values = {k: v for k, v in previous.items() if k.startswith("MEMORIZZ_")}
        if not staging.supported(self.provider):
            self._switch_embedding(values, persist=persist, say=say)
            return {"reembedded": self._reembed(say), "staged": False}
        say("Loading the earlier model…")
        manager, size = self._embedding_manager(values)
        kept = self.store.get_setting("vectors:previous") or {}
        held = self.provider.vector_previous() or {}
        reuse = {
            k: v for k, v in kept.items() if k.startswith("MEMORIZZ_")
        } == values and held.get("dimensions") == size
        try:
            report = staging.migrate(
                self.provider,
                manager,
                dimensions=size,
                reuse_previous=reuse,
                progress=say,
            )
        except Exception as exc:
            raise FineTuningError(
                f"Swapping back stopped before switching; nothing changed ({exc})"
            ) from None
        self._use_embedding(values, manager, persist=persist)
        # The fine-tuned vectors are not kept: nothing swaps back to them.
        self.provider.vector_previous_drop()
        self.store.set_setting("vectors:previous", None)
        self._store_size(fresh=True)
        return {"reembedded": report, "staged": True, "reused": reuse}

    def _use_embedding(self, values: Dict[str, str], manager, *, persist: bool) -> None:
        """Make ``manager`` (already loaded) the embedding model of this
        process and of the stores it writes."""
        from ..embeddings import set_global_embedding_manager
        from . import swap as swapping

        swapping.apply_settings(values, None)
        set_global_embedding_manager(manager)
        if persist:
            self._set_env(values, True)
        swapping.rebind(self.provider, manager)

    def _switch_embedding(self, values: Dict[str, str], *, persist: bool, say) -> None:
        from ..embeddings import get_embedding_manager
        from . import swap as swapping

        before = swapping.current_settings()
        swapping.apply_settings(values, None)
        try:
            manager = get_embedding_manager()
            probe = manager.get_embedding("memorizz embedding check")
            if not probe:
                raise RuntimeError("the model returned an empty vector")
        except Exception as exc:
            swapping.apply_settings(before, None)
            raise FineTuningError(
                f"The embedding model did not load ({exc}); nothing was changed"
            ) from None
        if persist:
            self._set_env(values, True)
        swapping.rebind(self.provider, manager)

    def _reembed(self, say) -> Dict[str, Dict[str, int]]:
        from ..embeddings import get_embedding_manager
        from . import swap as swapping

        say("Re-embedding stored memories…")
        return swapping.reembed(self.provider, get_embedding_manager(), progress=say)

    @staticmethod
    def _set_env(values: Dict[str, str], persist: bool) -> None:
        from . import clients

        for key, value in values.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        clients.refresh()
        if persist:
            from .._env_io import apply_env_updates

            error = apply_env_updates({k: v for k, v in values.items()})
            if error:
                raise FineTuningError(error)
            for key, value in values.items():
                if not value:
                    os.environ.pop(key, None)

    def _settings_keys(self) -> set:
        return {
            key
            for key in (
                "previous:memory_routing",
                "previous:reranking",
                "previous:embedding",
            )
            if self.store.get_setting(key)
        }

    def deactivate(self, use: str) -> None:
        self.store.set_setting(f"active:{use}", None)

    def active(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for use in USES:
            value = self.store.get_setting(f"active:{use}")
            if value:
                out[use] = value
        return out

    def route(self, text: str) -> Dict[str, Any]:
        """Where to store ``text``, from the active memory-routing model."""
        record = self.active().get("memory_routing")
        if not record:
            raise FineTuningError(
                "No memory-routing model is active; train one and activate it"
            )
        answers = self.decide(record["run_id"], text)
        answer = next(iter(answers.values()))
        return {
            "memory_type": answer.get("choice"),
            **answer,
            "run_id": record["run_id"],
        }


__all__ = ["FineTuning", "FineTuningError", "PARTITION", "USES", "environment"]
