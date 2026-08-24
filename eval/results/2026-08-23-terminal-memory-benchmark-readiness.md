# MemoRizz benchmark readiness: Terminal-Bench 2.1, LongMemEval-V2, and BEAM

Date: 23 August 2026

This is an engineering-readiness report. It distinguishes measured results,
forecasts, and official-protocol claims. No Terminal-Bench submission was made,
and neither memory smoke run is presented as a full paper reproduction.

## Results at a glance

| Evaluation | Exact scope | Result | Retrieval / memory signal | Cost | Correct interpretation |
|---|---|---:|---|---:|---|
| Terminal-Bench 2.1 | Forecast only; 89 tasks × 5 trials required | Accuracy and rank withheld | Native Harbor adapter, ATIF v1.7, task-local filesystem memory, host verification | $421.15 external Terra reference; $315.86–$631.72 scenario | Unofficial budget forecast, not a MemoRizz score |
| LongMemEval-V2 | Official pinned harness; one enterprise dynamic question; 100 trajectories | **1/1** | 4,072 chunks, 5 summaries, 4,991/24,802 evidence tokens selected, exact cache hit | **$0.207843 estimated** | Official adapter with diagnostic models/subset; not paper-comparable |
| BEAM 128K | One pinned 128K/100K conversation; one deterministic case from each of 10 abilities | **0.250 mean partial-credit score**; 0 exact accuracy | Recall@8 0.511, MRR 0.542, nDCG@8 0.417; gold-evidence reader 0.389 | **$0 external API cost** | MemoRizz diagnostic runner/scorer; not the official 2,000-question protocol |

## Terminal-Bench 2.1 forecast

The release path is ready for a bounded pilot, not a full paid run. MemoRizz now
prices the selected model instead of applying Terra prices to every OpenAI
model, rejects unknown pricing, writes pricing provenance into the ATIF/run
metadata, and exposes:

```bash
memorizz eval terminal-bench forecast \
  --model openai/gpt-5.6-terra \
  --per-trial-spend-guard-usd 1.75 \
  --total-budget-usd 1000 \
  --output eval/results/terminal-bench-forecast.json
```

The official minimum is 445 trials. A $1.75 per-trial guard produces a nominal
upper total of $778.75 and leaves $221.25 of headroom for in-flight-call
overshoot. The $421.15 point is the dated public Codex/Terra cost, not measured
MemoRizz spend. Its 0.75×–1.5× scenario is $315.86–$631.72. MemoRizz refuses to
forecast accuracy or rank until a pilot contains at least ten distinct tasks.

Recommended next action: run ten stratified tasks once with the exact intended
model, effort, resources, and spend guard; regenerate the forecast; then decide
whether a 445-trial run is justified. Community submissions were closed at the
time of this report, so any result remains local unless the maintainers accept
it.

## LongMemEval-V2

The adapter ran against upstream revision
`2cc8c540bdb87fe6761629b585e727e1c4704520`. The three required text assets
matched the official checksums before any paid request. Preflight forecast
$0.263522, or $0.395283 with 50% contingency; measured token-based spend was
$0.207843.

The downstream reader consumed only 339 tokens and received a 29-token memory
briefing. That compact context did not mean the memory work was free: MemoRizz
used 114,897 model tokens across six summary/synthesis calls, and embedding
8.33 million estimated tokens accounted for most cost. Compaction took 96.05
seconds and the complete memory query took 134.80 seconds. The main optimization
opportunity is durable embedding/summary checkpoints shared across reader A/B
runs—not reducing the already-small downstream evidence pack.

The result is not paper-comparable: it covers one of 451 questions and uses
GPT-5 mini plus `text-embedding-3-small`, not the paper's Qwen3.5-9B,
Qwen3-Embedding-8B, and GPT-5.2 configuration.

## BEAM

The zero-cost local rerun used pinned upstream revision
`3e12035532eb85768f1a7cd779832b650c4b2ef9`, Qwen 2.5 7B for reading/judging,
`nomic-embed-text`, filesystem storage, grouped weighted reciprocal-rank
fusion, query expansion, local reranking, MMR, and parent-source
deduplication.

| Ability | Score | Recall@8 | Interpretation |
|---|---:|---:|---|
| Abstention | 0.0 | n/a | Reader guessed rather than abstaining |
| Contradiction resolution | 0.0 | 0.500 | Evidence was partial; no answer credit |
| Event ordering | 0.0 | 0.333 | Evidence was partial; no answer credit |
| Information extraction | 0.0 | **1.000** | Retrieval succeeded; reader/scorer conversion failed |
| Instruction following | 0.5 | 0.500 | Partial answer |
| Knowledge update | 0.5 | 0.500 | Partial answer |
| Multi-session reasoning | 0.5 | 0.667 | Partial answer |
| Preference following | 0.5 | **1.000** | Retrieval succeeded; answer remained incomplete |
| Summarization | 0.5 | 0.100 | Partial credit despite a weak evidence set |
| Temporal reasoning | 0.0 | 0.000 | Retrieval failure |

The gold-evidence reader score of 0.389 versus the retrieved-evidence score of
0.250 leaves a 0.139 retrieval/context gap, while full-recall zero/partial cases
show a separate reader and scorer gap. Both need work. The current local assets
contain one of 20 official 128K conversations; the official benchmark contains
100 conversations and 2,000 questions across all four scales.

The upstream evaluator was inspected before choosing this label. It expects
answer files produced by BEAM's own generation pipeline, initializes three
sentence-transformer models, and invokes its configured LLM judge once per
rubric item. Feeding the normalized MemoRizz score into that code would not
reproduce the upstream answer-generation lifecycle. A native official lane is
therefore left as explicit follow-up work rather than being simulated here.

## Highest-value next improvements

1. Persist LongMemEval chunk embeddings and summary checkpoints by official
   dataset checksum, chunking policy, embedding digest, and memory-policy hash.
2. Add BEAM's upstream answer-generation and rubric scorer as a native official
   lane; retain the normalized runner only as a diagnostic.
3. Improve BEAM abstention calibration and structured reader output before
   adding a more expensive hosted reader.
4. Add a lightweight label-blind reranker for temporal, ordering, and
   contradiction evidence, then rerun the same fixed ten-case regression.
5. Run a ten-task Terminal-Bench pilot and emit Wilson uncertainty, cost, and
   serial/concurrent latency forecasts. Do not infer a rank from another
   agent's result.

## Reproducibility boundary

The committed records contain metrics, hashes, model names, and feature
evidence only. Raw benchmark corpora, prompts, answers, credentials, machine
paths, and provider logs remain outside the repository. The detailed sanitized
LongMemEval record is `2026-08-23-longmemeval-v2-official-smoke.json`; the BEAM
run is `2026-08-23-beam-128k-local-smoke-ready.json`; the Terminal-Bench
forecast is `2026-08-23-terminal-bench-2-1-forecast.json`.
