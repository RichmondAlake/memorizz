#!/usr/bin/env python3
"""Paired comparison of two LongMemEval runs (A/B arms).

Matches samples by question_id (falling back to question text), reports
overall and per-category accuracy for both arms, the question-level flips
(the paired signal — far more informative than comparing two averages),
a McNemar-style exact binomial p-value on the flips, and token/cache
metrics.

Usage:
    python compare_runs.py results/ab_baseline_50.json results/ab_candidate_50.json
"""

import json
import math
import sys
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def key_of(result):
    return result.get("question_id") or result.get("question")


def binomial_two_sided_p(k, n):
    """Exact two-sided binomial test p-value for k successes of n at p=0.5."""
    if n == 0:
        return 1.0

    def pmf(i):
        return math.comb(n, i) * 0.5**n

    p_k = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= p_k + 1e-12))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])
    label_a = a["metadata"].get("config_label") or "A"
    label_b = b["metadata"].get("config_label") or "B"

    results_a = {key_of(r): r for r in a["detailed_results"]}
    results_b = {key_of(r): r for r in b["detailed_results"]}
    common = [k for k in results_a if k in results_b]

    print(f"\n{'=' * 72}")
    print(f"Paired comparison: {label_a}  vs  {label_b}")
    print(f"{'=' * 72}")
    print(
        f"Matched samples: {len(common)} "
        f"(A total {len(results_a)}, B total {len(results_b)})"
    )
    for arm, data in ((label_a, a), (label_b, b)):
        md = data["metadata"]
        print(f"  {arm}: build={md.get('memorizz_path', '?')}")
        print(
            f"      window={md.get('context_window_tokens')} "
            f"ingest={md.get('ingest_mode')} "
            f"prompt_tokens={md.get('total_prompt_tokens', 0):,} "
            f"cached={md.get('total_cached_tokens', 0):,}"
        )

    # Per-category and overall paired accuracy
    per_cat = defaultdict(lambda: {"a": 0, "b": 0, "n": 0})
    flips_a_only = []  # A correct, B wrong
    flips_b_only = []  # B correct, A wrong
    correct_a = correct_b = 0

    for k in common:
        ra, rb = results_a[k], results_b[k]
        ca = bool(ra["evaluation"]["correct"])
        cb = bool(rb["evaluation"]["correct"])
        cat = ra.get("category", "?")
        per_cat[cat]["n"] += 1
        per_cat[cat]["a"] += ca
        per_cat[cat]["b"] += cb
        correct_a += ca
        correct_b += cb
        if ca and not cb:
            flips_a_only.append((k, ra))
        elif cb and not ca:
            flips_b_only.append((k, rb))

    n = len(common) or 1
    print(f"\n{'category':<28}{label_a[:18]:>18}{label_b[:18]:>18}{'Δ':>8}")
    print("-" * 72)
    for cat in sorted(per_cat):
        s = per_cat[cat]
        acc_a = s["a"] / s["n"]
        acc_b = s["b"] / s["n"]
        print(
            f"{cat:<28}{acc_a:>17.1%} {acc_b:>17.1%} {acc_b - acc_a:>+7.1%}"
            f"  (n={s['n']})"
        )
    print("-" * 72)
    print(
        f"{'OVERALL':<28}{correct_a / n:>17.1%} {correct_b / n:>17.1%} "
        f"{(correct_b - correct_a) / n:>+7.1%}"
    )

    # McNemar on the discordant pairs
    d = len(flips_a_only) + len(flips_b_only)
    p = binomial_two_sided_p(len(flips_b_only), d) if d else 1.0
    print(
        f"\nDiscordant pairs: {d} "
        f"({label_b} wins {len(flips_b_only)}, {label_a} wins {len(flips_a_only)})"
    )
    print(
        f"McNemar exact two-sided p = {p:.3f}"
        + (
            "  (differences at this sample size are suggestive, not conclusive)"
            if d and p >= 0.05
            else ""
        )
    )

    if flips_b_only:
        print(f"\nQuestions {label_b} fixed (up to 5):")
        for k, r in flips_b_only[:5]:
            print(f"  [{r['category']}] {str(r['question'])[:70]}")
    if flips_a_only:
        print(f"\nQuestions {label_b} regressed (up to 5):")
        for k, r in flips_a_only[:5]:
            print(f"  [{r['category']}] {str(r['question'])[:70]}")
    print()


if __name__ == "__main__":
    main()
