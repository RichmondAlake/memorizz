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
import os
import socket
import subprocess
import sys
import threading
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

Progress = Optional[Callable[[int, int, str], None]]
MIN_TRAIN, MIN_HELD_OUT = 8, 2
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
        return [
            {**d.to_dict(), "counts": self.store.counts(d.dataset_id)}
            for d in self.store.list_datasets()
            if owner is None or d.owner == owner
        ]

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
    ) -> Dataset:
        dataset = self._dataset(dataset_id)
        if spec is not None:
            self._validate_spec(dataset.kind, spec)
            dataset.spec = spec
        if name is not None:
            dataset.name = name
        if description is not None:
            dataset.description = description
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
                            "id": str(
                                row.get("memory_id")
                                or row.get("_id")
                                or row.get("id")
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
            data["train"].extend({**row, "replay": source.dataset_id} for row in chosen)
        return data

    def _own_snapshot(self, dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
        accepted = self.store.examples(
            dataset.dataset_id, status=ExampleStatus.ACCEPTED.value
        )
        data: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
        for e in accepted:
            data[e.split if e.split in data else "train"].append(e.to_dict())
        held_out = len(data["validation"]) + len(data["test"])
        if held_out < MIN_HELD_OUT and len(accepted) >= MIN_TRAIN + MIN_HELD_OUT:
            # Everything was marked train: hold some out so the before/after
            # comparison means something. By group first (related examples
            # stay together); if the groups are too few, a share of each label.
            data = {split: [] for split in SPLITS}
            for e in accepted:
                data[split_for(e.group or e.example_id)].append(e.to_dict())
            if len(data["validation"]) + len(data["test"]) < MIN_HELD_OUT:
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
    ) -> TrainingRun:
        """Train on the dataset's accepted examples. ``from_run`` continues
        from that run's model (continual training): same backbone and input
        format, with the lineage recorded."""
        dataset = self._dataset(dataset_id)
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
        if detach:
            log = open(files.log, "ab")
            environ = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                # Apple GPU: run the few operations MPS lacks on the CPU.
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            }
            process = subprocess.Popen(
                [sys.executable, "-m", "memorizz.finetune.worker", str(files.root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=environ,
            )
            log.close()
            run.pid = process.pid
            self.store.save_run(run)
        else:
            from .worker import run as run_worker

            run_worker(files.root)
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
        return [self._sync(run).to_dict() for run in self.store.list_runs(dataset_id)]

    def run(self, run_id: str) -> Dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise FineTuningError(f"No training run {run_id}")
        return self._sync(run).to_dict()

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
        split: str = "test",
        record: bool = True,
    ) -> Dict[str, Any]:
        """Score one model on a decision dataset's accepted examples.

        The model is a trained run, an untrained base model, or Jev
        (``jev_model``, e.g. "jev-latest"). ``split`` is test, validation,
        held_out (both) or all. Results are kept in the memory provider, so
        every version of a model can be compared on the same frozen sets:
        the history continual training is judged by."""
        import time as _time

        from .evaluate import classification_metrics, ranking_metrics, softmax
        from .models import new_id
        from .tasks import gold_key

        dataset = self._dataset(dataset_id)
        if dataset.kind != "decision":
            raise FineTuningError("Cross-version evaluation covers decision datasets")
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
        keys = [key for key, _ in options(question)]
        states = [e.input["state"] for e in rows]
        gold = [keys.index(gold_key(question, e.label[qid])) for e in rows]
        started = _time.monotonic()
        extra: Dict[str, Any] = {}
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
            probabilities = [row if sum(row) > 0 else uniform for row in probabilities]
            extra["usage"] = verifier.usage()
            model = {"type": "jev", "label": f"Jev ({jev_model}), zero-shot"}
        else:
            from .inference import base_decision_logits

            base = resolve_base(
                base_model or default_base("decision").model_id, "decision"
            )
            logits = base_decision_logits(
                base.model_id, base.family, base.backbone, question, states
            )
            probabilities = [softmax(row) for row in logits]
            model = {
                "type": "base",
                "base_model": base.model_id,
                "label": f"{base.label}, untrained",
            }
        metrics = classification_metrics(probabilities, gold, keys)
        if question["type"] == "noul":
            groups: Dict[str, List[tuple]] = {}
            for e, p, g in zip(rows, probabilities, gold):
                state = e.input["state"]
                query = state.get("query") if isinstance(state, dict) else e.group
                groups.setdefault(str(query), []).append((p[1], float(g)))
            metrics.update(ranking_metrics(list(groups.values())))
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

    def evaluations(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.list_evals(dataset_id)

    def activate(self, run_id: str, use: str) -> Dict[str, Any]:
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
