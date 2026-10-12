"""Fine-tuning records: datasets, examples, training runs."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class ModelKind(str, Enum):
    """The three kinds of model MemoRizz fine-tunes."""

    DECISION = "decision"  # System One: state + typed questions -> probabilities
    EMBEDDING = "embedding"  # bi-encoder for retrieval
    RERANKER = "reranker"  # cross-encoder: (query, passage) -> relevance


class ExampleStatus(str, Enum):
    PENDING = "pending"  # awaiting review
    ACCEPTED = "accepted"  # used for training and evaluation
    REJECTED = "rejected"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"  # the worker died without finishing


FINAL_RUN_STATES = {
    RunStatus.SUCCEEDED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELED.value,
    RunStatus.INTERRUPTED.value,
}
SPLITS = ("train", "validation", "test")


@dataclass
class Dataset:
    """Examples for one task. ``spec`` holds the task's definition: for a
    decision model the System One questions (instructions and options), for
    retrieval the query instruction."""

    name: str
    kind: str
    task: str
    spec: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    owner: str = ""  # the MCP principal that created it over HTTP; "" for local use
    dataset_id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Example:
    """One training example in MemoRizz's neutral form.

    decision:  input {"state": str|dict}, label {question_id: {"choice": k} |
               {"noul": bool|float} | {"level": int}}
    embedding: input {"query", "positive", "negatives": [...]}, label {}
    reranker:  input {"query", "passage"}, label {"relevant": 0..1}
    """

    dataset_id: str
    input: Dict[str, Any]
    label: Dict[str, Any] = field(default_factory=dict)
    status: str = ExampleStatus.PENDING.value
    split: str = "train"
    source: str = "manual"  # synthetic | memory | manual | import
    group: str = ""  # examples that must share a split (one generation batch)
    case_type: str = ""  # standard | negation | boundary | hard_negative | ...
    verified: Optional[bool] = None  # a blind second labelling agreed
    teacher: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    example_id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRun:
    dataset_id: str
    kind: str
    task: str
    base_model: str
    backbone: str  # embedding | lm | cross_encoder
    backend: str = "local"  # local (sentence-transformers / PEFT) | unsloth
    params: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    status: str = RunStatus.QUEUED.value
    device: str = ""
    output_dir: str = ""
    progress: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    pid: Optional[int] = None
    parent_run_id: str = (
        ""  # continual training: the run whose model this one starts from
    )
    run_id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "Dataset",
    "Example",
    "ExampleStatus",
    "FINAL_RUN_STATES",
    "ModelKind",
    "RunStatus",
    "SPLITS",
    "TrainingRun",
    "new_id",
    "utcnow",
]
