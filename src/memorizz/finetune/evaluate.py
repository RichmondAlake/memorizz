"""Metrics for decisions and rankings, computed the same way before and
after training so the two can be compared."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def softmax(row: Sequence[float], temperature: float = 1.0) -> List[float]:
    scaled = [value / temperature for value in row]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def classification_metrics(
    probabilities: Sequence[Sequence[float]],
    gold: Sequence[int],
    keys: Sequence[str],
    *,
    ids: Optional[Sequence[str]] = None,
    abstained: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Accuracy, macro F1, confusion, and how well calibrated the
    probabilities are (NLL, multi-class Brier in [0, 2], ECE over 10
    confidence bins).

    ``abstained`` marks rows with no answer (a verifier call that failed):
    they count as wrong rather than as a vote for the first option. With
    ``ids``, ``per_item`` maps each id to 1.0 (right) or 0.0, for paired
    comparisons between models scored on the same examples."""
    n = len(gold)
    if not n:
        return {"count": 0}
    k = len(keys)
    skipped = list(abstained) if abstained is not None else [False] * n
    predicted = [
        -1 if skip else max(range(k), key=lambda i: row[i])
        for row, skip in zip(probabilities, skipped)
    ]
    confusion = [[0] * k for _ in range(k)]
    for g, p in zip(gold, predicted):
        if p >= 0:
            confusion[g][p] += 1
    f1s = []
    per_class = {}
    for i, key in enumerate(keys):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(k)) - tp
        # Rows with no answer are misses for their gold class.
        fn = sum(1 for g in gold if g == i) - tp
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
            "support": sum(1 for g in gold if g == i),
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
        confidence = row[p] if p >= 0 else max(row)
        b = bins[min(9, int(confidence * 10))]
        b[0] += 1
        b[1] += confidence
        b[2] += 1.0 if p == g else 0.0
    ece = sum(abs(c - a) for count, c, a in bins if count) / n
    correct = [1.0 if g == p else 0.0 for g, p in zip(gold, predicted)]
    out: Dict[str, object] = {
        "count": n,
        "accuracy": round(sum(correct) / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "confusion": {"labels": list(keys), "matrix": confusion},
        "per_class": per_class,
    }
    if any(skipped):
        out["abstained"] = sum(1 for skip in skipped if skip)
    if ids is not None:
        out["per_item"] = {str(i): v for i, v in zip(ids, correct)}
    return out


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


def _group_scores(
    group: Sequence[Tuple[float, float]], k: int, cuts: Sequence[int]
) -> Tuple[float, float, Dict[int, float]]:
    """nDCG@k, MRR@k and Recall@cut for one query, as expected values over a
    random order of tied scores (McSherry & Najork, 2008): a model that
    cannot tell two candidates apart gets the average of both orders, not
    the order they happened to be listed in."""
    ranked = sorted(group, key=lambda item: -item[0])
    blocks: List[Tuple[float, List[float]]] = []
    for score, rel in ranked:
        if blocks and blocks[-1][0] == score:
            blocks[-1][1].append(rel)
        else:
            blocks.append((score, [rel]))
    positives = sum(1 for _, rel in group if rel > 0)
    dcg = mrr = 0.0
    hits = {cut: 0.0 for cut in cuts}
    position = 0
    found = False
    for _, rels in blocks:
        m = len(rels)
        gain = sum(2**rel - 1 for rel in rels) / m  # expected gain per slot
        share = sum(1 for rel in rels if rel > 0) / m  # P(a slot is relevant)
        for i in range(position, position + m):
            if i < k:
                dcg += gain / math.log2(i + 2)
            for cut in cuts:
                if i < cut:
                    hits[cut] += share
        r = sum(1 for rel in rels if rel > 0)
        if not found and r:
            # P(the first relevant item is at offset j in a random order)
            for j in range(m - r + 1):
                if position + j < k:
                    mrr += (
                        math.comb(m - 1 - j, r - 1)
                        / math.comb(m, r)
                        / (position + j + 1)
                    )
            found = True
        position += m
    ideal = sorted((rel for _, rel in group), reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return (
        dcg / idcg if idcg else 0.0,
        mrr,
        {cut: value / positives for cut, value in hits.items()},
    )


def ranking_metrics(
    groups: Sequence[Sequence[Tuple[float, float]]],
    k: int = 10,
    *,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """MRR@k, nDCG@k and Recall@1/5/k over groups of (score, relevance).

    A query counts only when it has a relevant candidate and a candidate
    that is not: with nothing to rank it wrong against, every model scores
    1.0 on it. Tied scores are averaged over their orders. With ``ids``
    (one per group), ``per_query`` maps each counted query to its nDCG@k."""
    cuts = sorted({1, 5, k})
    kept = []
    skipped = 0
    for index, group in enumerate(groups):
        has_relevant = any(rel > 0 for _, rel in group)
        has_other = any(rel <= 0 for _, rel in group)
        if has_relevant and not has_other:
            skipped += 1
        if has_relevant and has_other:
            kept.append((index, group))
    out: Dict[str, object] = {"queries": len(kept)}
    if skipped:
        out["skipped_no_negative"] = skipped
    if not kept:
        return out
    ndcg = mrr = 0.0
    recall = {cut: 0.0 for cut in cuts}
    per_query: Dict[str, float] = {}
    for index, group in kept:
        g_ndcg, g_mrr, g_recall = _group_scores(group, k, cuts)
        ndcg += g_ndcg
        mrr += g_mrr
        for cut in cuts:
            recall[cut] += g_recall[cut]
        if ids is not None:
            per_query[str(ids[index])] = round(g_ndcg, 6)
    n = len(kept)
    out.update({f"mrr@{k}": round(mrr / n, 4), f"ndcg@{k}": round(ndcg / n, 4)})
    out.update({f"recall@{cut}": round(value / n, 4) for cut, value in recall.items()})
    out["candidates_per_query"] = round(sum(len(g) for _, g in kept) / n, 1)
    if ids is not None:
        out["per_query"] = per_query
    return out


def paired_test(
    a: Dict[str, float],
    b: Dict[str, float],
    *,
    resamples: int = 20000,
    seed: int = 7,
) -> Optional[Dict[str, float]]:
    """Is model a better than model b on the items both were scored on, or
    could the difference be chance? A two-sided paired sign-flip test: under
    "no difference" each item's a − b is as likely to be flipped, so p is
    the share of sign assignments whose total is at least as far from zero.

    With right/wrong scores this is exactly McNemar's test (the binomial
    sign test on the items where the two models disagree); with graded
    scores (nDCG per query) it is exact for up to 12 differing items and a
    seeded Monte Carlo estimate above that. Returns the shared count ``n``,
    the mean difference ``diff``, how many items ``differ`` and ``p``; None
    when fewer than 5 items are shared."""
    import random

    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None
    diffs = [float(a[i]) - float(b[i]) for i in shared]
    n = len(diffs)
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    m = len(nonzero)
    mean = sum(diffs) / n
    if not m:
        p = 1.0
    elif all(abs(abs(d) - abs(nonzero[0])) < 1e-12 for d in nonzero):
        # Equal magnitudes (right/wrong): the exact binomial sign test.
        wins = sum(1 for d in nonzero if d > 0)
        tail = max(wins, m - wins)
        p = min(1.0, 2 * sum(math.comb(m, j) for j in range(tail, m + 1)) / 2**m)
    else:
        observed = abs(sum(nonzero)) - 1e-12
        if m <= 12:
            hits = sum(
                1
                for mask in range(2**m)
                if abs(sum(d if mask >> i & 1 else -d for i, d in enumerate(nonzero)))
                >= observed
            )
            p = hits / 2**m
        else:
            rng = random.Random(seed)
            hits = sum(
                1
                for _ in range(resamples)
                if abs(sum(d if rng.random() < 0.5 else -d for d in nonzero))
                >= observed
            )
            p = (hits + 1) / (resamples + 1)
    return {"n": n, "diff": round(mean, 6), "differ": m, "p": round(p, 4)}


__all__ = [
    "classification_metrics",
    "fit_temperature",
    "paired_test",
    "ranking_metrics",
    "softmax",
]
