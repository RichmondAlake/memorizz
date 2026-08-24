# Local memory benchmark suite

This runner adds protocol-aware diagnostics for AgentMemBench, LongMemEval-V2,
LoCoMo-Plus, BEAM, and MemoryAgentBench. It can use MemoRizz filesystem or
Oracle memory and local Ollama embeddings. Readers/judges can run through
Ollama (zero external API cost) or OpenAI. Official data stays external.

List adapters and their dataset environment variables:

```bash
memorizz eval list
memorizz eval protocol show locomo-plus
memorizz eval dataset sync locomo-plus
memorizz eval dataset verify locomo-plus --data-path /data/Locomo-Plus/data
```

Run a bounded smoke evaluation:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
memorizz eval run locomo-plus \
  --variant cognitive \
  --data-path /path/to/Locomo-Plus/data \
  --profile smoke \
  --output eval/results/locomo-plus-local.json
```

The JSON report separates retrieval-only metrics, the retrieved-evidence reader,
a gold-evidence oracle reader, citation grounding, and the diagnostic scorer.
It also records confidence intervals, cold/warm ingestion, corpus-cache hits,
retrieval/reranking/generation/judge timings, memory tokens/bytes, provider
capabilities, model usage, cost, and reproducibility metadata.

The diagnostic retrieval path uses a 256-item first-stage pool, grouped weighted
reciprocal-rank fusion, label-blind concept expansion, source- and event-linked
constraint/preference/state records, temporal/entity/concept signals, a local
rerank, parent-source deduplication, and MMR. Persistent corpus vectors are keyed
by data content, chunking, and embedding identity, so reader comparisons do not
re-embed unchanged data.

Use `--evaluation-mode retrieval` to isolate that normalized diagnostic. Use
`--evaluation-mode memagent --agent-template TEMPLATE.json` to run the real
`MemAgent.run()` automatic-retrieval and context-assembly path against isolated
benchmark memory. The latter disables side-effecting integrations and records
the exact overrides; it does not pretend to use diagnostic lexical/fusion
weights that are absent from the agent path.

Run the same bounded subset with GPT-5.5 while keeping retrieval and embeddings
local:

```bash
memorizz eval run locomo-plus \
  --variant cognitive \
  --data-path /path/to/Locomo-Plus/data \
  --limit 5 \
  --model-provider openai \
  --model gpt-5.5 \
  --reasoning-effort low \
  --output eval/results/locomo-plus-gpt-5.5.json
```

The CLI loads `OPENAI_API_KEY` from the process environment or MemoRizz's
layered `.env` files. Reports contain token counts and an estimated API cost,
never the credential.

One malformed JSON-like reader response receives at most one formatting-only
repair. Plain text is not retried. LoCoMo timestamps are normalized to ISO at
ingestion, common relative dates are attached as source-faithful absolute-date
annotations, and the diagnostic token scorer canonicalizes unambiguous ISO and
named-month dates before F1.

For Oracle, set `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN`, then add
`--memory-provider oracle`. The same SDK surface accepts
`memory_backend="oracle"`. The runner calls Oracle preflight before evaluation
and stops when the embedding model dimension differs from the existing VECTOR
columns. Choose a matching model (for example `--embedding-model all-minilm`
for a 384-dimensional schema) or migrate to a fresh, consistently dimensioned
schema; do not mix dimensions in one memory corpus.

## Comparability boundary

Profiles are `smoke` (one case/category), `regression` (fixed stratified
subset), and `paper` (complete split). A paper profile is still labelled
**Diagnostic** unless the official runner/scorer and every versioned manifest
field match. Add `--strict-paper` to fail instead of saving a non-comparable
claim. At present the separate LongMemEval-V2 adapter runs its upstream harness;
the other shared adapters remain explicit diagnostics until equivalent official
bridges exist.
