# LongMemEval-V2

This adapter registers MemoRizz as a memory backend with the unmodified
official LongMemEval-V2 harness. It performs source-diversified hybrid
semantic/lexical trajectory retrieval,
creates durable episodic records, runs MemoRizz summary/compaction, synthesizes
a bounded memory briefing, preserves both boundaries of oversized accessibility
trees, and records an exact semantic-cache probe.

The default command is a one-question, text-only smoke evaluation. It is not a
leaderboard-comparable result: the public leaderboard requires the complete
small or medium tier and the benchmark's fixed reader configuration.

The runner verifies `checksums.sha256` and writes a hosted-cost forecast before
the first model or embedding request. It refuses to continue when the forecast
plus 50% contingency exceeds `--max-estimated-cost-usd` (default `$2`). Use
`--forecast-only` to perform every local preflight without calling a provider:

```bash
python eval/longmemeval_v2/evaluate_memorizz.py \
  --official-root /path/to/LongMemEval-V2 \
  --data-root /path/to/longmemeval-v2-data \
  --output-dir /tmp/lme-v2-forecast \
  --question-id 01307e07 \
  --forecast-only
```

```bash
python eval/longmemeval_v2/evaluate_memorizz.py \
  --official-root /path/to/LongMemEval-V2 \
  --data-root /path/to/longmemeval-v2-data \
  --output-dir eval/longmemeval_v2/results/smoke-01307e07 \
  --memory-backend filesystem
```

The result separates estimated embedding cost, MemoRizz model cost from
provider-reported usage, official-reader cost, and optional judge cost. The
provider invoice remains authoritative because preflight is not a transactional
billing limit.

The learning control plane is enabled by default and gives the official reader
only a bounded memory briefing rather than replaying every raw retrieval and
summary. Add `--legacy-memory-path` for a matched legacy comparison. Use
`--memory-backend oracle --embedding-dimensions N` for Oracle; the runner
checks `N` against the existing vector schema before a paid model call.

Use repeated `--question-id` flags for a larger same-domain subset. Screenshot
questions are deliberately rejected unless the runner is extended to download
and validate the official screenshot bundles.
