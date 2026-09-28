# Compare answer models and rerankers

Open **Evalground → New comparison** (`/evalground/compare`) in the Memorizz UI. Connect a memory provider first. Experiments use isolated benchmark storage, rather than changing your agent's memories.

The Evalground home page opens with **Recent evaluations**, a table of saved comparisons and agent evaluations from the current server session. Search by name, model or run ID; filter by type/status; sort by date or API estimate; then open a result. Progress counts evaluated answers across configurations, not unique dataset questions. Each quality value names its scorer; the table is not a leaderboard across different experiments. Saved comparisons survive restarts; the existing agent-run registry is session-scoped.

1. **Dataset:** name the experiment, choose answer models, rerankers or complete pipelines, and select a sample or your labeled server-side dataset. The sample checks integration; held-out application data is needed for model selection. Saved candidate pools and sample configurations are under expandable settings.
2. **Models:** select providers and models from populated menus. The first row is explicitly marked as the baseline. Model discovery uses server credentials; custom deployments remain supported. Answer comparisons start with OpenAI examples; choose any supported provider for your environment. The fixed judge defaults to `gpt-4.1-mini`. Normal first-stage retrieval uses the configured Ollama embedding model; snapshot replays disable those unused settings.
3. **Review & run:** inspect the full workload, baseline and alternatives, set repetitions, evidence/candidate counts, time limit and optional API spend threshold, then start inference. Moving between setup steps does not start a run.

Results have **Overview**, **Evidence**, and **Calls & measurements** views. **View configuration** shows the actual saved settings read-only; **Duplicate configuration** fills a new editable setup without changing the original run. Exports preserve full measurements. Examples, notebooks and appbook links remain under **Examples & developer guides**, outside the primary workflow.

## Providers

Answer models and judges use Memorizz's existing Ollama, OpenAI, Anthropic, Azure, Hugging Face, and MLX adapters. Adapter-specific dependencies and credentials must be available to the UI server. Azure uses a deployment name. Optional packages are needed for local Hugging Face and MLX inference.

Rerankers include:

| Option | Behavior | Setup |
| --- | --- | --- |
| No reranker | Keeps initial candidate order | None |
| Token overlap | Deterministic lexical relevance baseline | None |
| LLM relevance scores | Scores every candidate using a generative model | Select LLM provider/model |
| Cross-encoder | Scores query/document pairs locally | Install `memorizz[rerank]`; specify a compatible model |
| Cohere | Uses `/v2/rerank` and billed search units | `COHERE_API_KEY`; specify model and optional USD/search-unit price |
| Voyage | Uses `/v1/rerank` with reported billed tokens | `VOYAGE_API_KEY`; select a supported reranking model |
| Jev | Noul per candidate, rubric-based Score, or a pool-wide Choice | `TYPESAFE_API_KEY`; specify Jev model and recipe |

Invalid, missing, nonfinite, or out-of-range scores fail the configuration explicitly. They do not silently fall back to another reranker. Jev is a decision model used here for relevance scoring; it is not an answer-generating LLM. Keep its candidate pool and document lengths within the selected model's context limits.

Jev Choice probabilities compare candidates within one pool; they are not absolute relevance probabilities and must not be used as a cross-query deletion threshold. The Choice adapter accepts at most 255 candidates. Score uses a fixed relevance rubric and computes the expected grade from the returned distribution. The UI preserves recipe names in result tables and chart legends.

## Oracle System One course presets

Set `MEMORIZZ_SYSTEM_ONE_HOME` to the sibling course's `system_one_models` directory, then run that course's `export_memorizz.py`. In the Dataset step, open **Start from a sample configuration**. The **7 reranking methods** and **3 answer models** buttons load the exported Oracle candidate snapshot and model settings. The course's `launch_memorizz.py` configures these paths and starts the UI.

The snapshot contains source IDs, text, relevance labels, measured retrieval provenance and a checksum. Memorizz validates it and sends identical candidates to each configuration. Provider calls are new live calls. Original Oracle retrieval is not run again: its latency and embedding costs are excluded from this replication's serving totals. Use the notebook ledger to inspect those ingestion and retrieval measurements.

The reranking preset compares original order, Voyage, an OpenAI reranker, local MiniLM, and Jev Noul/Score/Choice. The reader preset compares Opus 5.5, GPT-6 Sol and GPT-6 Luna on fixed evidence. Six synthetic questions are integration diagnostics; the reader score is explicitly labeled a lexical check, not independent factual accuracy. Replace the course snapshot with representative, held-out evidence before selecting a production model.

## Reading results

Results open with a findings summary and four interactive SVG charts. Reranking experiments lead with nDCG, with a selector for MRR, recall and precision at the configured k. Reader experiments lead with their answer scorer and Wilson intervals. Choose **Reranker only**, **Answer model only** or **Full answer pipeline** to change cost and latency scope; quality remains unchanged. Stage costs use the corresponding call lanes, and percentiles use actual per-question timings. The stage table keeps evaluation charges and one-time model loading separate.

Toggle configurations in the legend and hover or focus chart points for values. **Metric definitions** explains denominators and truncated MRR. The paired comparison view uses matching question IDs and human-readable model names; ranking deltas are point estimates, not significance claims.

Select a configuration and question under **Evidence before → after**. It shows original top-k evidence next to the actual records sent to the reader, source text, relevance grades, rank movements and recorded reranker scores. All original candidates are expandable. Only returned final positions were saved, so the UI does not invent ranks for unselected candidates. Full call records remain available below.

**Examples & developer guides** links the other four decision experiments to their appbook chapters. Set `MEMORIZZ_APPBOOK_URL` if the appbook runs somewhere other than `http://127.0.0.1:8878`. Those chapters run in the educational appbook, while reranking and reader comparisons use native Evalground adapters.

- **Quality:** accuracy under each case's scorer, mean score, recall@k, MRR, nDCG@k, bootstrap score intervals, and Wilson 95% accuracy intervals. A 6/6 score has an approximately 61%–100% Wilson interval even before considering dataset representativeness. Open individual cases to see expected answers, predictions, relevant/retrieved source IDs, and ranking traces. Cases without relevance labels cannot establish retrieval accuracy.
- **Latency:** p50/p95 of shared initial retrieval + reranking + reader calls, including formatting repairs. Percentiles use linear interpolation. TTFT is the first visible text token of reader calls when streaming is enabled; reranker calls are excluded from reader TTFT. Model loading is included; cold starts and tiny datasets can dominate results. Output throughput includes prefill time.
- **Tokens:** provider-reported input, cache reads, cache writes (including 1-hour writes), output, and reasoning. Cached input is a subset of input; reasoning is a subset of output. Missing usage is marked partial. Rerankers billed in search units do not necessarily report tokens.
- **Cost:** answer/reranker calls are serving cost; judge/oracle calls are evaluation overhead. Per-question and per-correct costs use serving costs. All measured attempts, including formatting repairs, remain in the ledger. Unavailable billing stays **Unknown**. Each call includes its price source or explicit override and available response model/service-tier metadata. Anthropic cache writes use their separate 5-minute/1-hour rates. Unsupported service-tier pricing stays unknown unless an explicit price override is provided.
- **Local inference:** external API charge is zero. An optional hourly rate estimates hardware cost from measured local model-call time. It excludes ingestion, idle time, and other machine costs. Local embedding calls have no external API charge and their token counts are outside the model-call ledger.

Price overrides use USD per million tokens for input, cached input, and output; enter all three. OpenAI uses the shared dated pricing registry; supported Anthropic models and Jev have dated rates. Unknown hosted models need explicit pricing. Anthropic cache-write overrides are optional additional fields; when writes occur without a known write price the cost remains unknown. Standard-rate tables do not silently price unsupported service tiers or long-context Claude requests. These are estimates, not reconciled invoices. A spend threshold requires all hosted providers to have prices. It is checked between calls and can be exceeded by an in-flight request. The time threshold is also checked between calls/stream events; **Stop** terminates the worker process group immediately.

The selected cases and candidate evidence are frozen across configurations. Initial retrieval timings are measured once and replayed, so latency compares the replaceable model/reranker stages against the same retrieval baseline. The first successfully completed configuration is the comparison baseline. Later repetitions shuffle configuration order. Paired intervals compare matching case scores; small samples and repeated use of the same questions do not establish general statistical significance.

## Persistence and scope

Experiments store `config.json`, `cases.json`, `candidates.json`, `result.json`, individual memory-suite reports, and a worker log in the comparison directory. Set `MEMORIZZ_COMPARISON_HOME` to change its location. Configuration files contain inference settings, never API key fields. Exports contain benchmark questions, answers, and retrieved evidence text; treat them as dataset artifacts.

Completed calls are saved as they arrive; completed cases are saved after scoring. A terminated request may have been billed without reporting usage. Interrupted results mark that billing incomplete rather than claiming a complete total. Workers do not resume automatically after a UI restart.

This workflow evaluates **memory QA**, with fixed reader, retrieval, and reranker stages. It does not evaluate unrestricted tool use, memory-write/extraction policies, or continual-learning updates. Existing agent evaluations retain their isolated diagnostic behavior. These results are not official benchmark leaderboard submissions.

The same engine can run without the UI:

```sh
python -m memorizz.benchmarks.comparison /path/to/experiment/config.json
```

Use the saved configuration file, or save the `config` object from a JSON export as a new `config.json` in a fresh directory. This runs inference and writes results beside the configuration.

## Verification

Unit tests cover accounting, cached/reasoning tokens, unknown prices, streaming completion, invalid reranker results, fixed evidence, mixed-provider judges, persistence, and read-only routes:

```sh
python -m pytest tests/unit/test_comparisons.py tests/unit/test_memory_suite.py tests/unit/test_ui_smoke.py tests/unit/test_usage_analytics.py
```

`tests/browser/comparisons_app.py` starts an isolated local fixture. With the default Ollama models installed and Playwright available, run `node tests/browser/comparisons.cjs` against it. This live test runs two readers and an LLM reranker, checks exported measurements and candidate fairness, reloads saved history, tests mobile layout, and cancels a real worker. Set `MEMORIZZ_PLAYWRIGHT_MODULE` for a non-default Playwright install and `MEMORIZZ_COMPARISON_EVIDENCE` for screenshots and JSON artifacts. Hosted integrations are covered with mocked contract responses; this test does not incur hosted API charges.

`tests/browser/comparisons_hosted.cjs` is an explicitly opt-in paid acceptance test (`MEMORIZZ_LIVE_HOSTED_EVAL=1`). It selects account-listed GPT-4.1 mini and Claude Haiku 4.5 models through the UI, runs all six strict checks with a $0.25 stop threshold, independently checks exported arithmetic, and captures charts on desktop/mobile. `comparisons_review.cjs` checks saved results and model-menu races without inference.

Discovery uses [OpenAI’s Models API](https://developers.openai.com/api/reference/resources/models/methods/list) and [Anthropic’s Models API](https://platform.claude.com/docs/en/api/models/list). Current supported standard prices were verified against [OpenAI pricing](https://developers.openai.com/api/docs/pricing) and [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) on 2026-09-24. Model-list access does not guarantee that every listed model supports the chosen endpoint or has inference quota.
