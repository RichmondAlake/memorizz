"""Offline bounded-index release gate. No models, network or existing stores.

PYTHONPATH=src python examples/observability/benchmark.py --events 10000
"""

import argparse
import json
import math
import time
from tempfile import TemporaryDirectory

from memorizz.observability.sql_index import SQLiteSpanIndex


def benchmark(root, *, events=10000, repeats=30):
    index = SQLiteSpanIndex(root)
    index.initialize()
    started = time.perf_counter()
    for number in range(math.ceil(events / 500)):
        size = min(500, events - number * 500)
        index.write_bundle(
            {
                "record_id": f"bundle-{number}",
                "type": "trace_bundle",
                "version": 2,
                "agent_id": "agent",
                "thread_id": "thread",
                "root_trace_id": "root",
                "turn_id": str(number),
                "run_id": "run",
                "user_id": "synthetic",
                "events": [
                    {
                        "event_id": f"event-{number}-{i}",
                        "trace_kind": "model_result",
                        "timestamp": "2026-09-04T00:00:00Z",
                        "content": "synthetic preview " * 200,
                    }
                    for i in range(size)
                ],
            }
        )
    write_seconds = time.perf_counter() - started
    latencies = []
    for _ in range(repeats):
        page = index.query(thread_id="thread", user_id="synthetic", limit=250)
        assert len(page["items"]) == min(events, 250) and page["scanned_count"] <= 251
        latencies.append(page["query_duration_ms"])
    rows, cursor, errors = set(), None, 0
    while True:
        page = index.query(limit=250, cursor=cursor)
        ids = {e["event_id"] for e in page["items"]}
        assert not rows & ids
        rows.update(ids)
        errors += page["normalization_errors"]
        cursor = page["next_cursor"]
        if not cursor:
            break
    total = sum(row["event_count"] for row in index.summaries())
    assert len(rows) == total == events and errors == 0
    return {
        "provider": "filesystem-sqlite",
        "events": events,
        "bundle_size": 500,
        "page_size": 250,
        "write_seconds": round(write_seconds, 3),
        "query_p50_ms": sorted(latencies)[len(latencies) // 2],
        "query_p95_ms": sorted(latencies)[
            min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)
        ],
        "duplicates": 0,
        "normalization_errors": errors,
        "parity": True,
        "scope": "synthetic_local_workload_not_production_SLA",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--max-p95-ms", type=float, default=250)
    args = parser.parse_args()
    if not 250 <= args.events <= 1000000:
        parser.error("events must be between 250 and 1000000")
    with TemporaryDirectory(prefix="memorizz-observability-benchmark-") as root:
        result = benchmark(root, events=args.events)
        print(json.dumps(result, indent=2))
        if result["query_p95_ms"] > args.max_p95_ms:
            raise SystemExit("Query latency gate failed")
