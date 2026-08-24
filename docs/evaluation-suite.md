# Memory-first evaluation suite

MemoRizz provides local adapters for complementary agent and memory benchmarks.
Every runner keeps the official score separate from MemoRizz diagnostics,
records upstream revisions and scope, and refuses to describe a bounded smoke
subset as a leaderboard result.

## Paper protocols versus engineering diagnostics

Evalground, the SDK, and `memorizz eval` expose five paper-backed adapters. A
normalized MemoRizz run is an engineering diagnostic; it is not automatically
the protocol used in a paper. Versioned manifests now record the dataset and
source revision, preprocessing boundary, full-split count, model snapshots,
quantization/decoding requirements, retrieval `k`, prompt and scorer hashes,
dependency fingerprint, seed, and hardware.

The comparison gate is fail-closed. Selecting `--profile paper` does not make a
result paper-comparable. `paper_comparable` becomes true only when the official
runner and scorer ran and every required manifest field matched. Otherwise the
result is visibly labelled **Diagnostic** or **Official adapter · diagnostic
configuration**, with exact mismatch reasons.

| Profile | Selection | Intended use |
|---|---|---|
| `smoke` | one deterministic case per available category | dependency and integration checks |
| `regression` | fixed stratified subset (50 by default) | reader/retriever/provider A/B checks |
| `paper` | complete split, no sample limit | strict upstream reproduction only |

Use `--strict-paper` in CI or release evidence when a diagnostic result must be
rejected rather than merely labelled.

## Official dataset workflow

Official data remains outside the wheel. MemoRizz synchronizes only pinned,
allowlisted source repositories and never executes downloaded code. Data that
requires a separate upstream/Hugging Face download remains an explicit operator
step.

```bash
memorizz eval list
memorizz eval protocol show longmemeval-v2
memorizz eval dataset sync longmemeval-v2
memorizz eval dataset verify longmemeval-v2 \
  --data-path /data/longmemeval-v2 --variant small-web
```

Verification records the Git origin and revision, required assets, byte counts,
per-file SHA-256 hashes, a dataset fingerprint, license locations, and missing
next actions. Add `--deep` only when you need content hashes for every file in a
large directory.

## Normalized memory diagnostic runner

The shared runner uses a larger first-stage candidate pool and grouped weighted
reciprocal-rank fusion over semantic and lexical lanes. Label-blind query
expansion connects situations such as overload to prior capacity and boundary
memories. Query-independent semantic records preserve constraints, preferences,
goals, commitments, state changes, causal fields, timestamps, and multi-source
event provenance. Parent-source deduplication prevents an original record and
its derived copy from consuming two evidence slots. Concept, temporal, entity,
local rerank, and MMR signals are applied only in the diagnostic lane. No gold
ID, reference answer, or benchmark category enters retrieval.

### Diagnostic retrieval versus full MemAgent execution

Evalground and the SDK expose two deliberately different execution modes:

| Mode | What actually runs | Use it for |
|---|---|---|
| `retrieval` | MemoRizz's normalized diagnostic fusion and a small grounded reader prompt | isolating retriever, reader, and scorer behavior |
| `memagent` | the selected template through `MemAgent.run()`, automatic memory retrieval, normal context assembly, and its semantic-cache policy | measuring the deployed agent path end to end |

Full MemAgent mode loads the selected template onto a disposable benchmark
memory provider. It disables tools, delegates, MCP, browser, sandbox,
automations, and continual-learning mutation; caps the agent at one model step;
and never writes secrets into the template snapshot. The reader model selected
for the run overrides the template model so A/B comparisons remain controlled.
The report records every override. A selected agent in diagnostic mode is only
metadata and is not described as agent execution.

Static question answering does not exercise summary/compaction. Reports say so
instead of claiming every memory feature was used. Use an incremental benchmark
profile for `observe → consolidate → query → update → forget → query` behavior.

Corpus embeddings are cached by dataset content, chunking configuration,
embedding provider/model, and optional immutable model digest. Reader A/B tests
therefore reuse the same vectors. Supply `--embedding-model-digest` for mutable
local aliases; reports explicitly record an unresolved digest when only a model
name is known.

Every case has separate lanes:

1. retrieval-only Recall@k, MRR, nDCG, ranks, and fusion evidence;
2. a structured retrieved-evidence answer with exact source IDs;
3. a gold-evidence oracle-reader answer that measures the reader ceiling; and
4. the benchmark scorer, citation validity, grounding state, and warnings when
   an answer receives credit despite missing gold evidence.

The generic judge remains a MemoRizz diagnostic. It is never presented as a
replacement for an upstream scorer.

| Adapter | MemoRizz support | Interpretation boundary |
|---|---|---|
| [AgentMemBench](https://arxiv.org/abs/2608.00009) | LoCoMo/MultiDoc2Dial/MSC normalized diagnostic | not the paper's five-strategy Qwen configuration |
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | normalized diagnostic plus `eval/longmemeval_v2/` official-harness adapter | official adapter remains diagnostic when subset/models differ |
| [LoCoMo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) | original and cognitive normalized diagnostic | upstream unified-input/prediction/judge scripts remain authoritative |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | 128K/500K/1M/10M loader and ten-category diagnostic | not a 2,000-question paper run until the official evaluator is used |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | four-competency normalized diagnostic | static exports do not reproduce incremental update/conflict lifecycles |

The provider contract now includes `store_many()`, scoped `search_memory()`,
rank scores/provenance capability metadata, and a compatible fallback. Filesystem
and MongoDB use optimized batch writes; Oracle uses the correct portable
fallback. The same diagnostic can run against filesystem or Oracle, and the
result records the provider's actual capabilities.

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
memorizz eval run locomo-plus \
  --variant cognitive \
  --data-path /path/to/Locomo-Plus/data \
  --profile smoke \
  --candidate-pool-size 256 \
  --output eval/results/locomo-plus-local.json
```

Run a secret-free saved-agent template through the full agent path:

```bash
memorizz eval run locomo-plus \
  --variant cognitive \
  --data-path /path/to/Locomo-Plus/data \
  --profile smoke \
  --evaluation-mode memagent \
  --agent-template /path/to/agent-template.json \
  --top-k 6 \
  --candidate-pool-size 256 \
  --output eval/results/locomo-plus-memagent.json
```

The template is a `MemAgentModel` JSON document without credentials. In the
local UI, choose **Full MemAgent execution** and select a saved agent; Evalground
creates this snapshot automatically.

Use Oracle with the identical dataset, reader, and seed:

```bash
memorizz eval run locomo-plus \
  --variant cognitive \
  --data-path /path/to/Locomo-Plus/data \
  --profile regression --limit 50 \
  --memory-provider oracle \
  --output eval/results/locomo-plus-oracle.json
```

The Oracle command reads `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN` from
the process/layered environment. Credentials are never copied into result JSON.
Oracle preflight runs before the benchmark and rejects an embedder/schema
dimension mismatch. Match `--embedding-model` to the dimensions reported by
`provider.preflight()["vector_dimensions"]`; a 384-dimensional schema can use
Ollama `all-minilm`, while `nomic-embed-text` produces 768 dimensions.

The SDK exposes the same controls:

```python
from memorizz.benchmarks.memory_suite import run_memory_suite

report = run_memory_suite(
    "locomo-plus",
    "/data/Locomo-Plus/data",
    variant="cognitive",
    profile="regression",
    limit=50,
    workspace=".memorizz-eval/locomo-plus",
    memory_backend="filesystem",  # or "oracle"
    candidate_pool_size=256,
    query_expansion=True,
    rerank_weight=0.15,
    reader_repair=True,
    oracle_reader=True,
    embedding_model_digest="sha256:<immutable-ollama-digest>",
)
```

SDK callers can instead pass `evaluation_mode="memagent"` with either
`agent_template=<MemAgentModel>` or an already-created `agent=<MemAgent>`. The
same isolated provider and side-effect boundary applies when a template is
used.

For a controlled hosted-reader comparison, keep the same subset and retrieval
configuration and add `--model-provider openai --model gpt-5.5`. GPT-5.5 uses
the Responses API with low reasoning effort by default; use
`--max-output-tokens` and `--reasoning-effort` to make the bound explicit.

Use `memorizz eval list` to inspect variants and dataset environment variables.
Local bounded reports are deliberately marked `paper_comparable: false`;
official repositories and graders remain the authority for leaderboard
submissions.
The sanitized end-to-end validation record is
[`eval/results/2026-08-22-memory-suite-ollama-smoke.json`](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-22-memory-suite-ollama-smoke.json).
It intentionally retains the retrieval miss and partial-credit result instead
of presenting harness execution as model quality. The same record includes an
AgentMemBench/LoCoMo run that achieved full source Recall@6 but low answer F1
and zero local-judge faithfulness, demonstrating that retrieval and grounded
answer quality remain separate signals.
The fixed-subset Qwen 2.5 7B versus GPT-5.5 rerun is recorded in
[`eval/results/2026-08-22-memory-suite-model-comparison.json`](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-22-memory-suite-model-comparison.json).
It records the hosted token cost and explicitly treats LoCoMo-Plus partial
answer credit as a retrieval failure because neither gold cue entered the
reader context.

## Targeted memory-path regression — 23 August 2026

After the MetaHarness-only security change correctly produced no memory score
movement, the same one-sample raw artifacts were rerun against the changed
memory path with Qwen 2.5 7B and local `nomic-embed-text`:

| Lane | Task score before → after | Recall@6 before → after | What the result establishes |
|---|---:|---:|---|
| AgentMemBench/LoCoMo diagnostic | 0.000 → **1.000** | 1.0 → 1.0 | rank-1 retrieval was already correct; absolute-date normalization and reader instructions fixed a false-zero synthesis/scoring boundary |
| LoCoMo-Plus cognitive diagnostic | 0.000 → 0.000 | 0.0 → **1.0** | both source-linked cues now reach top-6; the local reader still fails despite evidence |
| LoCoMo-Plus full MemAgent | n/a → **0.500** | n/a → **1.0** | the real automatic-retrieval/context-assembly path works and produces a partially correct grounded response |

The LoCoMo-Plus gold-evidence oracle reached only 0.5 with this local reader, so
the remaining task-score gap must not be presented as a retrieval failure. The
run is one sample, zero external API cost, and explicitly not paper-comparable.
Its sanitized evidence is
[`eval/results/2026-08-23-memory-path-fix-local.json`](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-23-memory-path-fix-local.json).

The identical source subsets were then evaluated with GPT-5.5 at low reasoning
effort while retaining local `nomic-embed-text` retrieval:

| Lane | Task score | Recall@6 | Grounded | Estimated API cost |
|---|---:|---:|---:|---:|
| AgentMemBench/LoCoMo diagnostic | 0.462 | 1.0 | 1.0 | $0.011485 |
| LoCoMo-Plus cognitive diagnostic | **1.000** | 1.0 | 1.0 | $0.015820 |
| LoCoMo-Plus full MemAgent | **1.000** | 1.0 | 1.0 | $0.024415 |

The AgentMemBench answer and its gold-evidence oracle were both factually
correct and faithfulness-scored 1.0, but token F1 was 0.462 because the complete
sentence contains more tokens than the terse reference. LoCoMo-Plus achieved
complete answer, retrieval, and citation scores in both execution modes. A
first diagnostic attempt exposed an ambiguous multi-source citation rendering;
it was discarded, the contract was corrected to present a JSON source-ID list,
and the affected lane was rerun. Retained artifacts total $0.051720 estimated;
actual session spend is estimated at $0.068155 including that discarded run.
This remains a one-sample diagnostic, not a leaderboard or paper result. The
sanitized record is
[`eval/results/2026-08-23-memory-path-fix-gpt55.json`](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-23-memory-path-fix-gpt55.json).

## Local evidence from 19 August 2026

These runs were local and were not submitted to any leaderboard.

| Benchmark | Exact local scope | Earlier smoke | Learning control plane / filesystem | Learning control plane / Oracle |
|---|---|---:|---:|---:|
| LongMemEval-V2 | One enterprise/dynamic text question over the same 100 trajectories; upstream `2cc8c540` | 1/1; 26,581 downstream-reader tokens | **1/1; 337 tokens** | **1/1; 275 tokens** |
| SWE-bench Lite | `sympy__sympy-20590`; dataset `b0dde109`; harness `490635b2`; GPT-5.4 mini/high | 1/1; 342,293 tokens; $0.1184; 141.99 s | **1/1; 192,439 tokens; $0.0616; 71.59 s** | 1/1; 553,434 tokens; $0.1420; 188.55 s |
| MemBench | One raw sample from each participation/observation × factual/reflective track; upstream `f66d8d10` | 4/4, recall 1.0; 40,633 tokens | **4/4, recall 1.0; 39,068 tokens** | **4/4, recall 1.0; 38,862 tokens** |

On the fixed SWE-bench case, the final filesystem run reduced total model
tokens by **43.8%**, estimated model cost by **47.9%**, agent duration by
**49.6%**, and executed terminal calls by **24.1%** while retaining the 1/1
official result. It also completed the full memory lifecycle: host acceptance,
one summary, safe side-effect cache bypass, an exact read-only reflection hit,
verified outcome evidence, and a successful incremental compiler checkpoint.
The fix that made this reliable was small: completion retries receive a bounded
reservation beyond the ordinary tool-work budget, so a required verification
call cannot be rejected merely because exploration consumed the work budget.

LongMemEval-V2 retained 1/1 correctness while reducing the official downstream
reader from 26,581 tokens to 337 on filesystem (**98.7%**) and 275 on Oracle
(**99.0%**). These are reader-only counts: embedding, summary, and memory-
synthesizer calls are not included. Lazy compaction still dominated memory
latency (81.41 s filesystem; 85.97 s Oracle), so this result demonstrates a
context reduction, not an end-to-end cost claim.

MemBench retained 4/4 accuracy and complete source recall. Filesystem reduced
tokens 3.9%, total measured runtime 18.2%, and answer latency 23.4%. Its
estimated cost rose 25.1% because this single run received fewer provider
prompt-cache hits. Oracle reduced tokens 4.4% but was slower. The SWE Oracle
trial also used twelve more tool calls than the filesystem trial; because each
configuration has one stochastic repetition, that difference must not be
attributed solely to the memory provider.

The sanitized comparison is
`eval/results/2026-08-19-learning-control-plane-comparison.json`; the earlier
baseline is `eval/results/2026-08-19-local-smoke.json`. Both omit raw benchmark
data, prompts, patches, credentials, and local paths.

## What each adapter measures

### LongMemEval-V2

The adapter registers a native MemoRizz `Memory` implementation with the
[official LongMemEval-V2 harness](https://github.com/xiaowu0162/LongMemEval-V2).
It batch-embeds trajectory states, stores provenance-bearing chunks and
episodic digests, performs scoped and source-diversified hybrid retrieval,
generates durable summaries/compaction links, and returns both raw evidence and
a memory briefing to the unchanged reader and scorer.

Accessibility trees are bounded without retaining only their prefix: both the
start and end remain available because menus and dialogs are commonly appended
late. Ranking strips only the benchmark's output-format instruction; it never
uses the reference answer. Screenshot questions fail preflight until the
official screenshot bundle is present.

```bash
python eval/longmemeval_v2/evaluate_memorizz.py \
  --official-root /path/to/LongMemEval-V2 \
  --data-root /path/to/longmemeval-v2-data \
  --output-dir /tmp/memorizz-lmev2 \
  --question-id 01307e07
```

### SWE-bench Lite

The adapter loads the pinned
[official SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite),
runs the agent only inside the official network-disabled instance image, emits
the standard prediction JSONL, and invokes the unmodified
[SWE-bench Docker grader](https://github.com/SWE-bench/SWE-bench). Neither the
gold patch nor the test patch enters the agent prompt.

`terminal_verify` records the final command, return code, and description. The
host `CompletionPolicy` rejects final text until this evidence exists and is
successful. The mutation turn is never admitted to semantic cache. A separate
read-only reflection turn exercises safe cache reuse.

```bash
python eval/swe_bench_lite/evaluate_memorizz.py \
  --official-root /path/to/SWE-bench \
  --output-dir /tmp/memorizz-swe-lite \
  --instance-id sympy__sympy-20590 \
  --max-cost-usd 2
```

On Apple Silicon, upstream labels Docker execution experimental. Pull the
official `linux/amd64` image before the run and allocate enough Docker Desktop
memory if a local image rebuild is required. The runner does not relax the
official task timeout or resources.

### MemBench

The adapter reads the four conceptual tracks from the
[official MemBench repository](https://github.com/import-myself/Membench),
preserves source-step keys, reports effective token capacity and source recall,
and applies a host JSON-choice validator. The repository's raw categorical
data supports a deterministic smoke matrix; it does not include the
README-referenced paper-sampled `data2test` bundle.

```bash
python eval/membench/evaluate_memorizz.py \
  --official-root /path/to/Membench \
  --output-dir /tmp/memorizz-membench \
  --samples-per-track 1
```

## MemoRizz feature evidence

| Runtime feature | LongMemEval-V2 | SWE-bench Lite | MemBench |
|---|---:|---:|---:|
| Tenant/thread-scoped durable memory | Filesystem + Oracle | Filesystem + Oracle | Filesystem + Oracle |
| Semantic or hybrid retrieval | Yes | N/A for repository commands | Yes |
| Summary creation and compaction | Yes | Yes | Yes |
| Semantic-cache admission/freshness evidence | Exact-repeat hit | Side-effect bypass + read-only probe | Exact-repeat hit |
| Workflow/tool execution evidence | N/A | Yes | N/A |
| Host completion acceptance | Reader is official harness | Passing terminal verification required | Valid A-D JSON required |
| Observability summary | Yes | Yes | Yes |
| Bounded `EvidencePack` | Yes | Yes | Yes |
| Immutable events + verified outcome + compiler | Yes | Yes | Yes |

Semantic similarity alone never proves freshness. Every adapter supplies a data
version, tenant/thread scope, cache admission class, and exact-repeat or bypass
evidence. Cached answers are revalidated by the current completion policy.

## Improvement backlog produced by the runs

### P0: implemented in this change

1. **Bounded recall:** one scoped `EvidencePack` now selects conversation,
   knowledge, summary, skill, and compiled evidence under a hard token budget
   before inference, with provenance and rejection reasons.
2. **Outcome-to-learning boundary:** only host/application-verified outcomes
   are authoritative for learning; immutable events feed a deterministic,
   idempotent compiler and existing workflow-to-skill promotion gates.
3. **Completion/cache boundary:** host completion policies revalidate cache
   hits, action turns bypass admission, and benchmark completion retries have a
   reserved bounded tool budget for mandatory verification.
4. **Exact-label retrieval:** LongMemEval-V2 combines semantic and lexical
   lanes, diversifies trajectory sources, and retains late accessibility-tree
   content. The adapter loads only the official registry contract and
   preflights the fixed Qwen processor before paid ingestion.
5. **Provider parity:** the same adapters run on filesystem or Oracle. Oracle
   retains MemBench source provenance in its existing namespace field, exposes
   shared control-plane records, and fails early on vector-dimension mismatch.
   Filesystem can use exact cosine search without loading FAISS when another
   native runtime would conflict.
6. **Strict protocol evidence:** versioned manifests and a fail-closed gate
   distinguish diagnostic, official-adapter, and fully paper-comparable runs.
7. **Dataset readiness:** pinned source synchronization, resumable Git fetch,
   artifact verification, checksums, revision/license reporting, and explicit
   next actions are available through `memorizz eval dataset`.
8. **Calibrated retrieval:** larger candidate pools, weighted reciprocal-rank
   fusion, generic semantic constraint/state records, temporal/entity signals,
   and MMR replace lexical-first concatenation. Retrieval never reads labels.
9. **Reader/retriever separation:** retrieved-evidence and gold-evidence reader
   lanes, source citations, grounding status, miss warnings, and conditioned
   scores prevent plausible ungrounded answers from hiding retrieval failures.
10. **Reusable corpus state:** content-addressed embedding snapshots and batch
    provider contracts remove repeat ingestion from reader A/B comparisons.
    MongoDB now uses one bulk write for ordinary corpus rows.

### Measured LongMemEval-V2 and BEAM evidence — 23 August 2026

The release-readiness pass added a no-call Terminal-Bench forecast, verified
the complete LongMemEval-V2 small-tier text assets, and ran two bounded memory
evaluations:

| Evaluation | Scope | Score | Memory/retrieval signal | External cost |
|---|---|---:|---|---:|
| LongMemEval-V2 | official pinned harness; 1 dynamic enterprise question over 100 trajectories | **1/1** | 4,072 chunks; 5 summaries; 29 downstream context tokens; exact semantic-cache hit | **$0.207843 estimated** |
| BEAM 128K | 1 pinned conversation; 10 abilities; local normalized runner | **0.250** mean partial credit | Recall@8 0.511; MRR 0.542; gold-evidence reader 0.389 | **$0** |

LongMemEval cost was forecast before execution at $0.263522, or $0.395283
including 50% contingency. The adapter now verifies official checksums, records
the actual scorer hash and dependency fingerprint, pins seed/order, batches
provider writes, and accounts separately for embeddings, MemoRizz synthesis,
the official reader, and any LLM judge. The committed record is sanitized and
contains no benchmark text or credentials.

BEAM source verification now accepts equivalent Git remotes with or without a
`.git` suffix, recognizes a data-in-source-checkout layout, and reports
coverage. This run had one of the 20 official 128K conversations. Its
full-recall information-extraction case was correctly reported as a
reader/scorer failure, while the temporal case remained a retrieval failure.

Neither result is paper-comparable. LongMemEval used one of 451 questions and
diagnostic models; BEAM used 10 of 2,000 questions and not the upstream runner
or rubric scorer. See
`eval/results/2026-08-23-terminal-memory-benchmark-readiness.md` for the exact
method, category table, cost boundary, and next improvements.

### P1: next harness work

1. Move summary generation to incremental/background ingestion. The measured
   LongMemEval runs spent 81–96 seconds compacting lazily.
2. Extend the new content-addressed embedding snapshot to resumable summary and
   compaction checkpoints for the official incremental adapters.
3. Extend the shared runner's lane-level token, cache-discount, cost, p50/p95,
   and footprint accounting into every upstream evaluator and agent loop.
4. Add a complete host-resource preflight for Docker architecture, image
   availability, free disk, and memory; dependency and Oracle schema checks are
   already early failures.
5. Evaluate retrieval and promoted skills across related task sequences. A
   one-off coding task proves harness correctness but cannot measure reuse or
   continual-learning benefit.
6. Repeat stochastic agent trials with fixed resources and confidence
   intervals. The observed Oracle SWE trial used more tool calls; one sample
   cannot separate provider overhead from model-path variance.
7. Add native official-runner/scorer bridges for LoCoMo-Plus, BEAM, and
   MemoryAgentBench. Until then those adapters remain correctly labelled
   diagnostics; do not recreate their scorers inside MemoRizz.
8. Add a versioned claim/state ledger and incremental
   `observe → consolidate → query → update → forget → query` session API after
   provider schemas and upstream conflict semantics are agreed. This is
   intentionally not hidden inside the static runner.

### P2: comparable measurement

Run the complete official benchmark splits with their prescribed readers,
resources, retries, and scoring, then publish confidence intervals and all
failures. Until then, keep these records as engineering evidence only.

## Reproducibility and security

- Pin repository and dataset revisions; retain failed and successful runs.
- Never copy API keys into benchmark containers or result JSON.
- Do not commit official benchmark data whose license or size requires external
  distribution.
- Use a fresh output directory for every run.
- Rotate any credential that has appeared in a terminal transcript before
  sharing raw logs publicly.
