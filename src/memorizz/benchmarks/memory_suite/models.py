"""Small, dependency-free data contracts used by the memory suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class MemoryDocument:
    """One independently retrievable unit with stable upstream provenance."""

    source_id: str
    content: str
    parent_source_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    linked_source_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("MemoryDocument.source_id is required")
        if not str(self.content).strip():
            raise ValueError("MemoryDocument.content is required")
        object.__setattr__(
            self,
            "linked_source_ids",
            tuple(
                dict.fromkeys(
                    str(item) for item in self.linked_source_ids if str(item).strip()
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "parent_source_id": self.parent_source_id or self.source_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "linked_source_ids": list(self.linked_source_ids),
        }


@dataclass(frozen=True)
class MemoryBenchmarkCase:
    """Normalized query over a corpus shared by one or more benchmark cases."""

    case_id: str
    benchmark_id: str
    corpus_id: str
    category: str
    documents: Tuple[MemoryDocument, ...]
    question: str
    answers: Tuple[str, ...] = ()
    relevant_source_ids: Tuple[str, ...] = ()
    rubric: Tuple[str, ...] = ()
    scorer: str = "answer_f1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ValueError("MemoryBenchmarkCase.case_id is required")
        if not str(self.corpus_id).strip():
            raise ValueError("MemoryBenchmarkCase.corpus_id is required")
        if not str(self.question).strip():
            raise ValueError("MemoryBenchmarkCase.question is required")
        if self.scorer not in {
            "answer_f1",
            "exact_match",
            "substring_exact_match",
            "recall_at_5",
            "llm_judge",
        }:
            raise ValueError(f"Unsupported benchmark scorer: {self.scorer}")

    @classmethod
    def create(
        cls,
        *,
        case_id: Any,
        benchmark_id: str,
        corpus_id: Any,
        category: Any,
        documents: Sequence[MemoryDocument],
        question: Any,
        answers: Sequence[Any] = (),
        relevant_source_ids: Sequence[Any] = (),
        rubric: Sequence[Any] = (),
        scorer: str = "answer_f1",
        metadata: Mapping[str, Any] | None = None,
    ) -> "MemoryBenchmarkCase":
        return cls(
            case_id=str(case_id),
            benchmark_id=str(benchmark_id),
            corpus_id=str(corpus_id),
            category=str(category or "uncategorized"),
            documents=tuple(documents),
            question=str(question),
            answers=tuple(str(item) for item in answers if item is not None),
            relevant_source_ids=tuple(
                str(item) for item in relevant_source_ids if item is not None
            ),
            rubric=tuple(str(item) for item in rubric if item is not None),
            scorer=scorer,
            metadata=dict(metadata or {}),
        )


__all__ = ["MemoryBenchmarkCase", "MemoryDocument"]
