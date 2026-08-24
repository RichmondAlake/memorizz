# MemBench

MemoRizz supports MemBench's four conceptual tracks:

- participation / factual;
- participation / reflective;
- observation / factual;
- observation / reflective.

The official Git repository contains the categorical source data but not its
README-referenced, paper-sampled `data2test` 0–10k/100k bundle. Consequently,
the default command below is an official-raw-data smoke run and is labeled
non-comparable to paper results.

```bash
python eval/membench/evaluate_memorizz.py \
  --official-root /path/to/Membench \
  --output-dir /tmp/memorizz-membench \
  --samples-per-track 1 \
  --memory-backend filesystem
```

The default path uses a token-bounded EvidencePack and records a verified
answer-key outcome. Add `--legacy-memory-path` for the older manual context
path. Oracle runs use `--memory-backend oracle --embedding-dimensions N` and
fail preflight when `N` differs from the existing vector columns.

The report includes answer accuracy, source-step retrieval recall, effective
capacity, ingest/retrieval/answer latency, per-track metrics, summary and
compaction evidence, an exact-repeat semantic-cache probe, and observability.
