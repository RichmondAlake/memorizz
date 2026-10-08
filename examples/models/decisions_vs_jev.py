"""Paired, small relevance smoke comparison; never writes credentials.

Configure OPENAI_API_KEY and TYPESAFE_API_KEY in the environment. For a local
interactive run, a missing Jev key is read without echoing it to the terminal.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from memorizz.benchmarks.measurement import MeasurementLedger
from memorizz.benchmarks.rerankers import Reranker
from memorizz.cli.config import load_layered_env

CASES = [
    (
        "Who currently owns Harbor?",
        [
            "Mira is the current Harbor owner, replacing Mina.",
            "Mina was the previous Harbor owner.",
            "Harbor is a release project.",
        ],
        0,
    ),
    (
        "What is Harbor's earliest launch time?",
        [
            "The launch time has been revised to after 19:00 UTC.",
            "The previous brief specified after 18:00 UTC.",
            "Mira owns the release.",
        ],
        0,
    ),
    (
        "What is the remaining budget?",
        [
            "The budget is $1,200 and spending is $940, leaving $260.",
            "The budget was approved last week.",
            "The launch is scheduled for Friday.",
        ],
        0,
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/memorizz-decisions-comparison.json")
    args = parser.parse_args()
    load_layered_env()
    if not os.environ.get("TYPESAFE_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = getpass.getpass("Jev API key (hidden): ")
    results = []
    for method in ("noul", "choice", "score"):
        for provider, model in (
            ("jev", "jev-1.13.0"),
            ("openai_decisions", "gpt-6-luna"),
        ):
            ledger = MeasurementLedger()
            row = {"provider": provider, "model": model, "method": method, "cases": []}
            for i, (query, texts, expected) in enumerate(CASES):
                ledger.case_id = str(i)
                try:
                    candidates = [
                        {"id": str(j), "content": text} for j, text in enumerate(texts)
                    ]
                    ranked = Reranker(
                        {"provider": provider, "model": model, "jev_method": method},
                        ledger,
                    ).rank(query, candidates, 3)
                    row["cases"].append(
                        {
                            "case": i,
                            "status": "passed",
                            "correct_top_1": ranked[0]["id"] == str(expected),
                            "ranking": ranked,
                        }
                    )
                except Exception as exc:
                    # Only error class and sanitized adapter status; never HTTP bodies.
                    row["cases"].append(
                        {
                            "case": i,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)
                            if isinstance(exc, RuntimeError)
                            and "returned HTTP" in str(exc)
                            else "See adapter validation",
                        }
                    )
            row["calls"] = ledger.calls
            row["correct"] = sum(c.get("correct_top_1", False) for c in row["cases"])
            row["seconds"] = sum(c["seconds"] for c in ledger.calls)
            costs = [c["cost_usd"] for c in ledger.calls]
            row["estimated_api_cost_usd"] = (
                sum(costs) if all(c is not None for c in costs) else None
            )
            results.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "provider",
                            "method",
                            "correct",
                            "seconds",
                            "estimated_api_cost_usd",
                        )
                    }
                ),
                flush=True,
            )
    evidence = {
        "scope": "Three synthetic cases per method; integration smoke test, not a quality benchmark.",
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
