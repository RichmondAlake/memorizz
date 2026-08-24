# MemoRizz evaluation suite

The suite separates official benchmark scoring from MemoRizz diagnostics. Each
runner records the upstream repository/dataset revision, exact subset, model,
memory feature evidence, raw official score, and whether the result is valid
for paper or leaderboard comparison.

| Benchmark | What it measures | MemoRizz runner |
|---|---|---|
| LongMemEval | Long-term conversational memory | `longmemeval/` |
| LongMemEval-V2 | Experience from long web/enterprise trajectories | `longmemeval_v2/` |
| SWE-bench Lite | Real repository issue resolution | `swe_bench_lite/` |
| MemBench | Participation/observation × factual/reflective memory | `membench/` |
| AgentMemBench, LongMemEval-V2, LoCoMo-Plus, BEAM, MemoryAgentBench | Paper-aware local-memory regression suite | `memory_suite/` |
| Terminal-Bench 2.1 | General terminal-agent task completion | `memorizz.benchmarks.terminal_bench` |
| MetaHarness adaptive predecessor | Codex, Claude Code, and adaptive MemoRizz panels on Filesystem and Oracle | `metaharness/provider_comparison.py` |
| MetaHarness factorial comparison | Direct MetaHarness, one-harness MemAgent, full Codex+Claude panel, and adaptive panel across Filesystem and Oracle | `metaharness/factorial_comparison.py` |
| MetaHarness Opus 5 small suite | Four independent repair tasks comparing Luna-only, Opus-only, always-on, and preregistered risk-routed MemAgent strategies | `metaharness/opus5_small_suite.py` |

## Installation

Install only the optional integrations you use:

```bash
pip install "memorizz[longmemeval-v2]"
pip install "memorizz[swe-bench]"
pip install "memorizz[membench-eval]"
pip install "memorizz[terminal-bench]"
```

LongMemEval-V2, SWE-bench, and MemBench require their official repositories or
datasets. The runners do not vendor benchmark code and do not silently replace
official graders.

The unified `memory_suite/` runner supports filesystem or Oracle memory while
keeping Ollama embeddings local. Ollama supplies the zero-cost default
reader/judge; an optional OpenAI path records token usage and estimated cost.
Versioned manifests, source/data verification, strict comparability gates,
calibrated retrieval fusion, content-addressed embedding snapshots, grounded
source citations, and a gold-evidence oracle-reader lane keep retrieval quality,
reader quality, and upstream scoring distinct.

```bash
memorizz eval list
memorizz eval dataset sync beam
memorizz eval dataset verify beam --data-path /data/BEAM --variant 128k
memorizz eval run beam --data-path /data/BEAM --profile smoke
memorizz eval terminal-bench forecast --total-budget-usd 1000
```

The LongMemEval-V2 adapter loads only the official memory registry contract, so
unselected benchmark backends do not pull mutually incompatible dependencies.
It also verifies official checksums and emits a fail-closed hosted-cost
forecast before any paid request.

See [`docs/evaluation-suite.md`](../docs/evaluation-suite.md) for pinned local
run results, exact interpretation boundaries, feature evidence, and the
improvement backlog produced by the suite. Sanitized machine-readable records
are the [baseline smoke snapshot](results/2026-08-19-local-smoke.json) and the
[filesystem/Oracle learning-control-plane comparison](results/2026-08-19-learning-control-plane-comparison.json).
The [optimized MetaHarness evaluation report](results/2026-08-22-metaharness-provider-comparison-optimized.md)
documents the adaptive Filesystem/Oracle run, fairness controls, costs, results,
and next platform improvements. Because both panel rows executed one Codex call,
it is evidence for structured early stopping rather than a two-harness panel
comparison. The factorial runner separates wrapper, mandatory coordination,
adaptive-routing, and provider effects; its dry-run mode writes a complete
protocol manifest with zero external calls. The
[earlier baseline](results/2026-08-22-metaharness-provider-comparison.md) is
retained so the engineering delta remains auditable.

The 23 August release-readiness evidence combines an unofficial, no-call
[Terminal-Bench forecast](results/2026-08-23-terminal-bench-2-1-forecast.json),
an official-adapter
[LongMemEval-V2 smoke](results/2026-08-23-longmemeval-v2-official-smoke.json),
and a zero-cost local
[BEAM ten-ability smoke](results/2026-08-23-beam-128k-local-smoke-ready.json).
The [method and interpretation report](results/2026-08-23-terminal-memory-benchmark-readiness.md)
states exactly what was measured and why none is a leaderboard or paper claim.

The later [paid factorial report](results/2026-08-22-metaharness-factorial-paid.md)
and its [sanitized artifact](results/2026-08-22-metaharness-factorial-paid.json)
separate direct MetaHarness calls, single-harness MemAgent wrappers, mandatory
Codex+Claude panels, adaptive panels, and Filesystem/Oracle effects across 24
observations. Required-criterion recall and deterministic validity gates are
primary; the scalar LLM judge is retained as diagnostic-only after its
construct-validity audit failed. The report includes the known-spend lower
bound and every reporting correction.

The later
[Opus 5 task-native report](results/2026-08-23-metaharness-opus5-small-suite.md)
uses GPT-5.6 Luna and Claude Opus 5 at medium effort over four fresh Filesystem
repair tasks. It replaces the scalar judge with 32 held-out checks, records
exact model identities, approvals, grounding, verification, cost, latency, and
tokens, and preserves the one-repeat claim boundary. Its
[sanitized artifact](results/2026-08-23-metaharness-opus5-small-suite.json) is
the source for the referenceable table in MetaHarness Notebook 06.

## Memory-first evaluation profile

The new runners exercise and report:

- task-, user-, and thread-scoped durable filesystem memory;
- semantic retrieval and provenance;
- semantic-cache misses, writes, exact-repeat hits, bypass reasons, and age;
- summary creation and original-message compaction links;
- workflow and tool-log persistence for tool-using agents;
- the host-enforced completion policy for measurable final-answer gates;
- observability summaries and per-stage timing.
- immutable control-plane events, verified benchmark outcomes, deterministic
  compilation, and a bounded `EvidencePack` (`--legacy-memory-path` disables
  this layer for matched comparisons);
- filesystem or Oracle memory (`--memory-backend`), with vector-dimension
  preflight before Oracle benchmark model calls.

Side-effecting command turns are intentionally excluded from cache admission.
This is expected safe behavior, not a missing cache feature.

## Interpretation

A tiny local subset is a smoke test, not a leaderboard score. Never extrapolate
one-instance accuracy to a full benchmark. Compare systems only when dataset
revision, split, model/reader, resources, retry count, timeout, and official
grader are all identical. Keep failures in the denominator.
