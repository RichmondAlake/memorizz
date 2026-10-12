"""Metrics for decisions and rankings, computed the same way before and
after training so the two can be compared."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def softmax(row: Sequence[float], temperature: float = 1.0) -> List[float]:
    scaled = [value / temperature for value in row]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def classification_metrics(
    probabilities: Sequence[Sequence[float]], gold: Sequence[int], keys: Sequence[str]
) -> Dict[str, object]:
    """Accuracy, macro F1, confusion, and how well calibrated the
    probabilities are (NLL, Brier, ECE over 10 confidence bins)."""
    n = len(gold)
    if not n:
        return {"count": 0}
    k = len(keys)
    predicted = [max(range(k), key=lambda i: row[i]) for row in probabilities]
    confusion = [[0] * k for _ in range(k)]
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1
    f1s = []
    per_class = {}
    for i, key in enumerate(keys):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(k)) - tp
        fn = sum(confusion[i]) - tp
        if tp + fp + fn == 0:
            continue  # class absent from gold and predictions
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1s.append(f1)
        per_class[key] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[i]),
        }
    nll = -sum(math.log(max(row[g], 1e-12)) for row, g in zip(probabilities, gold)) / n
    brier = (
        sum(
            sum((row[i] - (1.0 if i == g else 0.0)) ** 2 for i in range(k))
            for row, g in zip(probabilities, gold)
        )
        / n
    )
    bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, confidence, correct
    for row, g, p in zip(probabilities, gold, predicted):
        confidence = row[p]
        b = bins[min(9, int(confidence * 10))]
        b[0] += 1
        b[1] += confidence
        b[2] += 1.0 if p == g else 0.0
    ece = sum(abs(c - a) for count, c, a in bins if count) / n
    return {
        "count": n,
        "accuracy": round(sum(1 for g, p in zip(gold, predicted) if g == p) / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "confusion": {"labels": list(keys), "matrix": confusion},
        "per_class": per_class,
    }


def fit_temperature(logits: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    """The temperature that makes softmax(logits / T) best match how often
    the model is right (lowest NLL on held-out rows)."""
    if not gold:
        return 1.0

    def nll(t: float) -> float:
        return -sum(
            math.log(max(softmax(row, t)[g], 1e-12)) for row, g in zip(logits, gold)
        )

    grid = [0.05 * 1.15**i for i in range(42)]  # 0.05 .. ~16
    best = min(grid, key=nll)
    return round(best, 4)


def ranking_metrics(
    groups: Sequence[Sequence[Tuple[float, float]]], k: int = 10
) -> Dict[str, float]:
    """MRR@k, nDCG@k and Recall@1/5/k over groups of (score, relevance)."""
    groups = [g for g in groups if any(rel > 0 for _, rel in g)]
    if not groups:
        return {"queries": 0}
    mrr = ndcg = 0.0
    recall = {1: 0.0, 5: 0.0, k: 0.0}
    for group in groups:
        ranked = sorted(group, key=lambda item: -item[0])
        relevances = [rel for _, rel in ranked]
        first = next((i for i, rel in enumerate(relevances[:k]) if rel > 0), None)
        mrr += 1.0 / (first + 1) if first is not None else 0.0
        dcg = sum(
            (2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k])
        )
        ideal = sorted(relevances, reverse=True)
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
        ndcg += dcg / idcg if idcg else 0.0
        positives = sum(1 for rel in relevances if rel > 0)
        for cut in recall:
            recall[cut] += sum(1 for rel in relevances[:cut] if rel > 0) / positives
    n = len(groups)
    out = {"queries": n, f"mrr@{k}": round(mrr / n, 4), f"ndcg@{k}": round(ndcg / n, 4)}
    out.update({f"recall@{cut}": round(value / n, 4) for cut, value in recall.items()})
    return out


__all__ = ["classification_metrics", "fit_temperature", "ranking_metrics", "softmax"]
