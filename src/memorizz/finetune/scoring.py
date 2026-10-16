"""Scoring a model the same way everywhere: a run's before/after in
training, the comparisons on the Models in use cards, and the scores in
the Across versions table all come from these functions, so one model on
one set gets one number."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

from .evaluate import classification_metrics, ranking_metrics

SPLIT_ORDER = ("train", "validation", "test")
# A reranker reorders each question's memory, its look-alikes and this many
# other memories from the dataset (a fixed sample).
SAMPLED_OTHERS = 9

Score = Callable[[str, List[str]], List[float]]


def memory_pool(rows_by_split: Dict[str, Sequence[Dict[str, Any]]]) -> List[str]:
    """Every memory in a dataset (answers and look-alikes), once each, in
    the order train, validation, test."""
    pool: List[str] = []
    seen = set()
    for split in (*SPLIT_ORDER, *(s for s in rows_by_split if s not in SPLIT_ORDER)):
        for row in rows_by_split.get(split) or []:
            source = row.get("input", row)
            for text in [source["positive"], *(source.get("negatives") or [])]:
                text = str(text)
                if text not in seen:
                    seen.add(text)
                    pool.append(text)
    return pool


def rank_held_out(
    kind: str,
    rows: Sequence[Dict[str, Any]],
    pool: Sequence[str],
    score: Score,
    *,
    seed: int = 7,
) -> Dict[str, Any]:
    """Ranking metrics for held-out rows ({"input": {query, positive,
    negatives}, "example_id"}). An embedding model searches every memory in
    ``pool``; a reranker reorders the row's memory, its look-alikes and
    ``SAMPLED_OTHERS`` other memories. ``per_item`` holds each question's
    nDCG@10, keyed by example id."""
    rng = random.Random(seed)
    groups, ids = [], []
    for index, row in enumerate(rows):
        source = row.get("input", row)
        positive = str(source["positive"])
        if kind == "reranker":
            negatives = [str(n) for n in (source.get("negatives") or [])]
            others = [t for t in pool if t != positive and t not in negatives]
            candidates = [
                positive,
                *negatives,
                *rng.sample(others, min(SAMPLED_OTHERS, len(others))),
            ]
        else:
            candidates = [positive, *(t for t in pool if t != positive)]
        values = score(str(source["query"]), candidates)
        groups.append(
            [(float(v), 1.0 if i == 0 else 0.0) for i, v in enumerate(values)]
        )
        ids.append(str(row.get("example_id") or index))
    metrics = ranking_metrics(groups, ids=ids)
    metrics["per_item"] = metrics.pop("per_query", {})
    metrics["count"] = len(rows)
    return metrics


def decision_metrics(
    rows: Sequence[Dict[str, Any]],
    probabilities: Sequence[Sequence[float]],
    gold: Sequence[int],
    keys: Sequence[str],
    question_type: str,
    *,
    abstained: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Classification metrics for decision rows; for a yes/no relevance
    question also ranking metrics over each query's candidates (by
    P(true)). ``per_item`` holds what the job's main metric is made of:
    right (1) or wrong (0) per example, or nDCG@10 per query."""
    ids = [str(r.get("example_id") or i) for i, r in enumerate(rows)]
    metrics = classification_metrics(
        probabilities, gold, keys, ids=ids, abstained=abstained
    )
    if question_type == "noul":
        groups: Dict[str, List[tuple]] = {}
        for row, probs, g in zip(rows, probabilities, gold):
            state = row["input"]["state"]
            query = state.get("query") if isinstance(state, dict) else row.get("group")
            groups.setdefault(str(query), []).append((probs[1], float(g)))
        query_ids = [hashlib.sha1(q.encode()).hexdigest()[:16] for q in groups]
        metrics.update(ranking_metrics(list(groups.values()), ids=query_ids))
        metrics["per_item"] = metrics.pop("per_query", {})
    return metrics


def embedding_scorer(
    model: Any, prompts: Dict[str, str], dimensions: Optional[int] = None
) -> Score:
    """``(query, passages) -> cosine`` for a Sentence Transformers model;
    each passage is encoded once and reused for every query. ``dimensions``
    scores the vectors cut to that size (as a store of that size keeps them):
    the first ``dimensions`` values, normalized again."""
    vectors: Dict[str, Any] = {}

    def encode(texts: List[str], prompt: Optional[str]):
        import torch

        out = model.encode(
            texts,
            prompt=prompt or None,
            normalize_embeddings=not dimensions,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if dimensions:
            out = torch.nn.functional.normalize(
                out[:, : int(dimensions)].float(), dim=-1
            )
        return out

    def score(query: str, passages: List[str]) -> List[float]:
        import torch

        q = encode([query], prompts.get("query"))
        missing = [p for p in dict.fromkeys(passages) if p not in vectors]
        if missing:
            vectors.update(zip(missing, encode(missing, prompts.get("document"))))
        d = torch.stack([vectors[p] for p in passages])
        return (q @ d.T)[0].float().cpu().tolist()

    return score


def cross_encoder_scorer(model: Any) -> Score:
    def score(query: str, passages: List[str]) -> List[float]:
        values = model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(v) for v in values]

    return score


__all__ = [
    "SAMPLED_OTHERS",
    "cross_encoder_scorer",
    "decision_metrics",
    "embedding_scorer",
    "memory_pool",
    "rank_held_out",
]
