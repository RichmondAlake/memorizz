# Changelog

## Unreleased

No changes yet.

## 0.2.0 — 2026-07-18

### Breaking changes

* Removed `MemAgent.switch_thread`, `CWM`, `ConfigBuilder`, and
  `WorkflowManager`. These surfaces were unreferenced by the maintained docs,
  examples, and test suite; applications importing them must migrate before
  upgrading.

### Codebase cleanup & de-bloat

* **Removed ~5,000 lines of verified-dead code.** An abandoned half-refactor
  in `memagent/` (`handlers/`, `utils/{formatters,helpers,validators}.py`,
  `builders/config_builder.py`, the never-called `WorkflowManager`), the
  legacy pre-Toolbox `database/` package, the dead `memory_unit` layer
  (`MemoryUnit` was never constructed; `summary_component.py` was
  byte-identical to the episodic copy), ~650 lines of orphaned provider
  methods, unused `Workflow`/`EntityMemory`/`KnowledgeBase`/semantic-cache
  methods, the unused `CWM` working-memory class, dead embeddings-provider
  info methods, and repo orphans (`scenario/`, superseded eval scripts,
  drifted Homebrew formula, obsolete Oracle thick-client installers).
* **pytest configuration was silently inert and is now active.**
  `pytest.ini` used the `[tool:pytest]` header (only valid in `setup.cfg`),
  so strict markers, timeouts, and warning filters never applied. Fixed the
  header and registered the three markers tests actually use.
* **Streaming and non-streaming tool loops now share one body**
  (`_execute_and_record_tool_call`): ~190 near-verbatim duplicated lines
  (argument parsing, execution, tool-log offload, placeholder persistence,
  workflow capture) existed twice and had already drifted once.
* **MongoDB provider dispatch consolidated**: three identical
  collection-mapping dicts and five if/elif chains replaced by a single
  `_collection()` resolver (−520 lines), pinned by a new mongomock dispatch
  test suite covering every memory type.
* **MongoDB agents are now addressable by their custom `agent_id`.**
  `store_memagent` previously stripped the custom id and every lookup
  resolved only ObjectId `_id`s — agents saved under string ids (the SDK
  default is a uuid4) were unretrievable, repeated `save()` calls inserted
  duplicates, and toolbox syncing was scoped to the wrong id. The id is now
  persisted, all agent operations resolve either identifier, and
  `store_memagent` upserts (matching the filesystem provider's semantics).
* **`ui/app.py` fully decomposed**: 9,052 → 1,910 lines. Eight route groups
  extracted into `ui/routers/` (traces, memory pages, vercel-skills,
  knowledge-base, whatsapp webhook, evalground, agents CRUD, playground)
  with shared helpers in `ui/helpers.py`; all 35 routes byte-identical in
  path/method/response class, gated by a new whole-app smoke suite plus
  functional tests for the agent-CRUD/playground/evalground flows.
* **`memagent/core.py` decomposed**: 6,612 → 5,367 lines. Tool-log
  placeholder helpers → `memagent/utils/tool_log.py`; save/load/refresh and
  memory download/update → `memagent/persistence.py` (thin delegates keep
  every public signature); summary generation → `MemoryManager`.
* **`clear_semantic_cache` now works on every provider** via a generic
  base-class implementation (previously MongoDB-only; filesystem/Oracle
  calls raised and were silently swallowed).
* **Agent persistence bug fixed**: the filesystem and MongoDB
  `retrieve_memagent`/`list_memagents` reconstruction whitelists dropped
  `continual_learning`/`continual_learning_config` on load.
* **Packaging/CI**: the wheel no longer ships an unowned top-level `eval/`
  directory into site-packages (moved under `memorizz/eval/`); removed the
  unused `platformdirs` base dependency; the release gate now runs the same
  test set as CI (performance suite excluded); the Homebrew tap job's
  token gate moved into a step (the job-level `secrets` condition never
  evaluated); `make lint` now fails on real errors (F-class/E9) instead of
  `|| true`-ing everything; fixed the broken `setup_oracle` import in the
  remote-oracle notebook; UI whole-app smoke tests added (44 routes/pages).
* **Release hardening**: UI template rendering now supports both legacy and
  current Starlette signatures without pinning users to an old FastAPI;
  package/UI/npm metadata is synchronized at `0.2.0`; agent edits preserve
  WhatsApp configuration; dashboard automation counts work on every supported
  backend; and docs use the canonical CLI commands and valid links.

### Continual learning: human-in-the-loop UI

* **Trajectory-class view** (`/memory/workflows`): workflow runs are now
  grouped by canonical hash into classes with per-class executions,
  success rate, query diversity, recency, and a pass/fail chip per
  promotion gate; expandable run lists mark skill-suppressed rows.
* **Promotion controls**: a per-agent **Run promotion cycle** button and a
  per-class **Distill now** button (backed by the new
  `PromotionEngine.promote_class` /
  `ContinualLearningManager.promote_class`) — both still enforce the full
  eligibility and validation gates; the resulting promotion report renders
  on the page.
* **Skill lifecycle page** (`/memory/skills`): status badges, versions,
  activation stats, baselines, preconditions, and the distilled SKILL.md,
  with **Activate** (shadow → active + stamp backfill) and **Demote**
  (new `ContinualLearningManager.demote_skill`; releases suppressed
  workflows) actions.

### Continual learning (workflows → skills)

* **Trajectory canonicalization** (`long_term/procedural/workflow/canonicalization.py`):
  every stored workflow now carries a `canonical_hash` — the sha256 of its
  (tool, argument-shape, error-class) unit sequence with retries collapsed —
  so runs that are "the same procedure" count together regardless of
  argument values. Computed inside `Workflow.store_workflow`, shared by both
  the streaming and non-streaming capture paths. Backfill existing rows with
  `scripts/backfill_canonical_hashes.py` (aggregation also hashes legacy
  rows on the fly).
* **Skillbox memory store** (`MemoryType.SKILLBOX`, `long_term/procedural/skillbox/`):
  learned skills are SKILL.md documents in the database with a lifecycle
  (`candidate | shadow | active | deprecated | demoted`), applicability-only
  embeddings (name + description + preconditions + trigger queries — never
  the tool sequence), and per-skill stats. Supported by all three providers
  (MongoDB, filesystem, Oracle — including new Oracle table DDL +
  migrations).
* **Promotion engine** (`PromotionEngine` / `PromotionConfig`): trajectory
  classes clear explicit gates (`min_executions`, `min_success_rate`,
  `min_distinct_queries`, recency) before LLM distillation; distilled
  skills must pass a machine validation gate (tool resolution, no
  hallucinated tools, intent-not-mechanism description, size cap, no
  literal-value leaks, LLM judge) before gaining any context authority.
  `require_shadow=True` stores new skills as non-injecting SHADOW for
  human review.
* **Cache-safe injection**: a static learned-skills contract in the system
  prompt (priors-not-mandates, precondition checking) plus per-turn
  rendering of matching skills at the top of the volatile context block —
  the prompt-cache stable prefix is never touched. Retrieval threshold
  0.70, stricter than any other memory retrieval.
* **Lifecycle monitoring** (`SkillMonitor`): every run records
  `skills_activated`; skills track successes/failures/deviations and a
  rolling success window, are demoted on drift below their promotion-time
  baseline, deprecated when a referenced tool disappears, and release
  their suppressed workflows back to retrieval on any exit from ACTIVE.
  Re-qualification (post-demotion runs only) produces a v2 skill informed
  by the old demotion reason.
* **Retrieval suppression**: `Workflow.retrieve_workflows_by_query` now
  excludes trajectories covered by an ACTIVE skill by default
  (`exclude_promoted=False` opts out); runs are always still *written* —
  they are the drift-evidence stream.
* **Agent surface**: `MemAgent(continual_learning=True,
  continual_learning_config={...})`,
  `MemAgentBuilder.with_continual_learning(...)`, persisted on
  `MemAgentModel`, `MEMORIZZ_CONTINUAL_LEARNING=1` env default; learned
  skills appear in `list_skills` / `read_skill` tagged `source: "learned"`.
  Promotion cycles run off the hot path (daemon thread, every
  `promotion_every_n_runs` stored runs; `0` = manual).
* **Educational notebook + live A/B evaluation**
  (`examples/continual_learning/continual_learning_guide.ipynb`): walks the
  full loop programmatically, then benchmarks two real MemAgents — workflow
  memory only vs. a promoted learned skill — with paired trials, τ-bench-style
  `pass^2`, trajectory-graded accuracy, token accounting across the whole tool
  loop, and a McNemar-style exact binomial test on the paired flips (the same
  convention as `eval/longmemeval/compare_runs.py`). On `gpt-4.1-mini` the
  skill arm scored 100% vs 55% accuracy (p = 0.0039) for a ~33% prompt-token
  tax.
* **Fixes surfaced by the loop**: MongoDB now preserves `workflow_id` and
  `agent_id` on workflow documents (previously stripped at store time,
  which broke per-agent trajectory aggregation); `Workflow.from_dict` and
  `user_id` round-trip stored embeddings/user scope instead of re-embedding
  on every load; Oracle gained a proper `retrieve_by_id` branch for
  workflow rows.

### Context efficiency & prompt caching

* **Prompt-cache-friendly context assembly.** Requests are now ordered
  stable-prefix → volatile-tail: the system prompt is frozen for the session
  (tool-log digest, entity facts, and per-call request context moved out of
  it), conversation history is append-only with chunked eviction (window
  start only moves at 20-message boundaries), and all per-turn content is
  rendered into a `<memorizz:context>` block at the start of the final user
  message. Tool definitions are serialized in deterministic (sorted) order.
* **Anthropic prompt caching.** The `Anthropic` provider attaches
  `cache_control` breakpoints automatically (system prompt + last two
  messages) — in live testing ~99% of the prompt is served from cache from
  turn 2 onward. Opt out with `enable_prompt_caching=False`.
* **OpenAI cache routing.** The `OpenAI` provider pins a per-thread
  `prompt_cache_key` and supports `prompt_cache_retention="24h"`; streaming
  requests now request the final usage chunk (`stream_options`) so cache
  metrics are reported on streams too.
* **Cache observability.** `get_last_usage()` now includes `cached_tokens`
  (plus `cache_read_input_tokens` / `cache_creation_input_tokens` on
  Anthropic). Anthropic `prompt_tokens` now reports the true prompt size
  (uncached + cached), fixing under-counted context-window telemetry.

### Pre-inference deduplication

* **Assembly-time dedup pipeline** (`memagent/utils/context_dedup.py`):
  retrieved memories are exact-hash deduplicated across sources, dropped if
  already present in the in-window conversation history, near-dup filtered
  via stored-embedding cosine similarity (>= 0.95), MMR-selected (relevance
  blended with recency), and rendered in deterministic chronological order.
* **Write-path guards.** Identical consecutive `(query, response)` pairs
  (semantic-cache hits, retries) are no longer double-written to
  conversation memory; entity upserts that add no new information skip the
  re-embed and re-store (NOOP).

### Retrieval fixes & performance

* **Pre-inference retrieval is now actually used.** Previously every turn ran
  3 vector searches + 3 embedding calls whose results were silently
  discarded; retrieval is now gated on the agent's active memory types,
  returns stored embeddings for dedup (`include_embedding`), and the deduped
  selection is injected into the prompt. Toolbox pre-retrieval was dropped
  (tool schemas already ride in the `tools` parameter).
* **Episodic semantic recall works.** Conversation rows (previously stored
  with `embedding=None`, making vector recall structurally empty) are now
  embedded by a background worker after each turn — the hot path never
  blocks on the embedding API. Oracle gained the missing
  conversation-memory string-query vector-search branch. Disable via
  `MEMORIZZ_DISABLE_CONVERSATION_EMBEDDINGS=1`.
* **Shared embedding memo.** `EmbeddingManager` caches text → vector (LRU),
  collapsing the up-to-5 identical query embeddings per turn into one API
  call.
* **Indexed summaries query.** `load_summaries_for_thread` uses a native
  scoped MongoDB query (`list_summaries`) instead of scanning the whole
  summaries collection every turn; other providers keep the shared fallback.
* **Background summarization.** Automatic context summarization now runs on
  a daemon thread with an in-flight guard instead of blocking the turn.
* **Filesystem provider write race fixed.** Document/index writes are now
  serialized per store and use unique tmp names (concurrent writers could
  previously fail with ENOENT on the shared `index.tmp`).
* **Oracle conversation rows are now individually addressable.**
  `store()` returned the conversation's grouping `memory_id` instead of the
  row id, and `update_by_id` used `memory_id` as its WHERE key — so any
  per-row update (embedding backfill, summary marking) silently rewrote
  EVERY row in the conversation (all rows ended up sharing the last
  message's embedding, degrading episodic recall to noise), and
  `retrieve_by_id` fell into a generic fallback that always returned None.
  Store now returns the RAW(16) row id, updates/reads address rows by it,
  and episodic vector recall on Oracle returns correctly-ranked matches
  (verified: planted-fact recall score 0.77 vs 0.02 before).

### Evaluation

* **LongMemEval harness upgrades** (`eval/longmemeval/`): benchmark-correct
  `--ingest_mode direct` (stores the dataset's user+assistant turns
  verbatim with embeddings; the legacy replay mode regenerated assistant
  replies and lost the evidence for single-session-assistant questions),
  `--context_window_tokens` to force genuine long-horizon memory dependence,
  `--config_label`/`--judge_model`, deterministic evenly-spaced sampling for
  paired A/B runs, per-sample token/cache metrics, and a `compare_runs.py`
  paired-comparison report (per-category deltas, question flips, McNemar).
* **Mechanism probes** (`scripts/memory_accuracy_probes.py`): live
  Oracle-backed checks for eviction-boundary recall, dedup false-positives,
  knowledge updates, prompt-cache neutrality, and the repeat-turn write
  guard.

## 0.1.1 — 2026-06-21 (repository-only)

### Features

* **Automations on every backend.** Scheduled `agent.run()` automations now work
  on the default FileSystem store and on MongoDB, not just Oracle
  (`FileSystemAutomationStore` + `MongoDBAutomationStore`).
* **`/automation` REPL command** — list and run the automations attached to the
  current agent (`/automation`, `/automation run <id>`).
* **Automations can search the web.** A scheduled agent restores its internet
  provider (Tavily / Firecrawl) on load and can call `internet_search` /
  `open_web_page` during a run.
* **One-line installers.** `npm i -g memorizz` and
  `curl -fsSL https://raw.githubusercontent.com/RichmondAlake/memorizz/main/install.sh | sh`
  (both bootstrap `uv`), alongside `pip install memorizz` / `uvx memorizz`.
* **Ollama Cloud models.** An explicit `:cloud` model (e.g. `glm-5.2:cloud`) is
  now honored by the CLI instead of being substituted with a local model.
  Authenticate with `/login ollama` (runs `ollama signin`), then
  `/model glm-5.2:cloud`.

### Fixes

* **Default local stack works on a lean install.** Ollama embeddings required
  `langchain_ollama` (an optional extra) and the `ollama` client wasn't a base
  dependency, so a plain `pip install memorizz` + `memorizz` crashed with
  "langchain_ollama is required for Ollama embeddings". Embeddings now use the
  native `ollama` client, and `ollama` is a base dependency (still no langchain,
  no torch).
* Agents executing a scheduled automation no longer wander into managing
  automations — automation-management tools are suppressed for the run (fixes
  off-task / empty outputs from smaller models).
* `FileSystemAutomationStore` is now multi-worker-safe (POSIX file lock around
  job claiming).

## 0.1.0 — 2026-06-18

### Features

* **New interactive CLI — `memorizz` is now a Claude-Code-style local agent.**
  Running `memorizz` with no arguments launches an interactive REPL that streams
  the agent's replies live, with `/` slash commands (`/help`, `/model`,
  `/provider`, `/ollama`, `/code`, `/memory`, `/history`, `/agents`, `/agent`,
  `/new`, `/persona`, `/persona-reset`, `/tools`, `/ingest`, `/forget`, `/ui`,
  `/login`, `/config`, `/docs`, `/clear` (wipe all memory, with confirmation),
  `/cls`, `/exit`).
  Also adds `memorizz chat`, one-shot `memorizz run "<prompt>"`, `memorizz init`,
  `memorizz config`, and `memorizz --version`. `python -m memorizz` now works.
* **Zero-config local stack.** With no API key set and an Ollama daemon running,
  the CLI defaults to Ollama (LLM + `nomic-embed-text` embeddings) + an on-disk
  FileSystem memory store under `~/.memorizz/memory` — a 100%-local agent with no
  keys. Cloud keys (Anthropic / OpenAI / Azure) are auto-detected when present.
* **`/code` coding mode.** `memorizz --code` (or `/code` in the REPL) enables the
  agent's self-aware file read/write + bounded command tools, scoped to the
  working directory (writes on, deletes off).
* **Internet access.** Set `TAVILY_API_KEY` or `FIRECRAWL_API_KEY` (or `/login
  tavily`) and the agent gains web search + page reading (`internet_search` /
  `open_web_page`); toggle at runtime with `/web on|off|tavily|firecrawl`. No
  extra install — the providers call the REST APIs directly. Tavily runs at advanced search
  depth for ~5x richer results.
* **Canonical config at `~/.memorizz/`.** Keys/settings live in
  `~/.memorizz/.env` (override via `MEMORIZZ_HOME` / `MEMORIZZ_ENV_FILE`), shared
  by the CLI and the local web UI. `$CWD/.env` is still honored for back-compat.
* **One persistent agent + memory across launches.** Each `memorizz` session
  loads (or creates once) a single default `MemAgent` and reuses its rolling
  memory id, persisted to `~/.memorizz/state.json`, so facts learned in one
  session are recalled in the next instead of starting fresh every launch.
* **Distribution.** Installable via `uv tool install memorizz` / `pipx install
  memorizz`, a Homebrew tap, and an npm shim (`npm i -g memorizz`, bootstraps uv).

### Bug fixes

* **Local UI wrote `.env` to an unreadable path.** The Settings page computed the
  env file as `<package>/../../../.env`, which under a pip/uv install landed in
  `site-packages` where nothing read it. Env handling is now centralized in
  `memorizz._env_io`, and the UI reads/writes the same `~/.memorizz/.env` as the
  CLI.
* **Ollama reasoning models returned truncated/empty output.** `OllamaLLM` now
  auto-enables `think` for reasoning families (qwen3 / deepseek-r1 / qwq /
  magistral), so their thinking is surfaced (shown dimmed in the REPL) and the
  full answer streams through instead of being cut off.
* **FileSystem semantic recall could crash** with *"only length-1 arrays can be
  converted to Python scalars"* when a stored embedding had an unexpected shape;
  cosine similarity now flattens and shape-checks operands, skipping mismatches
  rather than aborting retrieval.
* **Persona / knowledge-base embeddings failed on a keyless local stack.**
  Components using the module-level `get_embedding()` defaulted to OpenAI; the
  CLI now points the global embedding manager at the active provider (e.g.
  Ollama `nomic-embed-text`), so `/persona` and `/ingest` work fully offline.
* **Small local models looped on tool calls during plain chat**, hitting the
  20-step cap on even simple questions. The default memory-assistant no longer
  exposes the auto-registered lookup/utility tools (`knowledge_base_lookup`,
  entity/summary/tool-log helpers) that small models compulsively call — memory
  is still injected via context, so recall is unaffected. `/code` keeps its tools.

### Breaking changes

* **The heavy local-ML stack is no longer installed by default.** `transformers`,
  `sentence-transformers`, `accelerate`, and `huggingface-hub` moved out of the
  base dependencies into the existing `huggingface` extra, so a default
  `pip install memorizz` (and `uv tool install memorizz`) no longer pulls a
  multi-GB PyTorch stack. If you use local HuggingFace models or HF embeddings,
  install `pip install "memorizz[huggingface]"`.
* **Minimum Python is now 3.10** (the old `>=3.7` never matched the actual
  dependency floors).

## 0.0.52 — 2026-06-16

### Bug fixes

* **Oracle provider: workflows were stored without embeddings, so
  `retrieve_workflows_by_query` never matched.** The `workflow_memory` table has
  an `embedding VECTOR` column and `retrieve_by_query` runs a `VECTOR_DISTANCE`
  search over it, but `_store_workflow_memory` skipped embedding entirely —
  leaving every workflow unsearchable by intent. It now embeds the workflow's
  stable identity (name + description) via `_generate_embedding_if_needed`, so
  `Workflow.retrieve_workflows_by_query(...)` returns semantically relevant
  runbooks. Steps/outcome are still excluded from the embedding (they carry
  volatile, arbitrary tool output).

## 0.0.51 — 2026-06-16

### Bug fixes

* **Oracle provider: `Toolbox.register_tool` failed with `DPY-3002`.** `_store_toolbox`
  bound the tool id straight from the caller, but `Toolbox` hands it a `uuid.UUID`
  object, which python-oracledb cannot bind ("Python value of type UUID is not
  supported"). The id is now coerced to text (`str(tool_id)`) before binding, so
  registering tools into a `Toolbox` (and the procedural-memory notebook flow)
  works against Oracle. Complements the 0.0.50 VECTOR-binding fix.

## 0.0.50 — 2026-06-16

### Bug fixes

* **Oracle provider: `ORA-01484` on VECTOR writes in thin mode.** Several store
  paths bound the embedding as a raw Python `list`, which python-oracledb's thin
  mode rejects ("arrays can only be bound to PL/SQL statements") — breaking
  `agent.save()` (toolbox + persona rows), entity-memory upserts, and the
  multi-agent / shared-memory writes behind `MultiAgentOrchestrator`. Every
  embedding bind now flows through the existing `_prepare_vector_value` helper
  (`list` → `array.array("f", …)`): `_generate_embedding_if_needed` normalizes
  both the provided- and freshly-generated-embedding branches (covering all
  `_store_*` impls), and the agent / persona / toolbox upsert sites use the same
  helper. Verified end-to-end against Oracle Database 23ai Free.

## 0.0.47 — 2026-05-31

### Bug fixes

* **Concurrent provider init no longer crashes with `CollectionInvalid`.**
  `MongoDBProvider._create_memory_store` checks `list_collection_names()`
  and then `create_collection()` for each memory store. Under concurrent
  initialisation — e.g. multiple gunicorn workers each building the
  provider on their first request after a deploy, before the collections
  exist — two workers could both pass the existence check and the loser's
  `create_collection()` then raised
  `pymongo.errors.CollectionInvalid: collection <name> already exists`,
  aborting provider construction (and taking down the agent for that
  worker). Creation is now idempotent: `CollectionInvalid` and
  `OperationFailure` code 48 (`NamespaceExists`) are swallowed, since the
  collection exists either way. Any other `OperationFailure` still
  propagates.

## 0.0.46 — 2026-05-24

### Performance

* **`tool_log` queries are now native-indexed.** `MemoryManager.list_tool_logs`
  previously did a full `list_all(MemoryType.TOOL_LOG)` and filtered + sorted +
  sliced in Python — O(n) in the user's lifetime tool-call count. The MongoDB
  provider now ships a native `list_tool_logs(memory_id, user_id, limit)`
  that pushes filter + sort + limit to the server, backed by two new
  compound btree indexes (`tool_log_memory_timestamp`,
  `tool_log_user_timestamp`) plus a unique `tool_log_id` index. The
  manager uses the native helper via duck-typing (`hasattr`) so other
  providers continue to use the in-memory fallback unchanged.
* **Hot-path btree indexes** added for CONVERSATION_MEMORY
  (`memory_id + thread_id + timestamp`, `user_id + timestamp`),
  ENTITY_MEMORY (`user_id + updated_at`), and SUMMARIES
  (`memory_id + period_end`). Created idempotently at provider init
  via a new `_ensure_btree_indexes` helper. Failures (duplicate spec
  from an earlier migration, etc.) are logged at DEBUG so a single
  provisioning hiccup never breaks agent boot.

## 0.0.45 — 2026-05-24

### Bug fixes

* **`reset_tool_context()` token argument is now optional.** The 0.0.43
  signature change required callers to pass the token returned by
  `set_tool_context`, breaking the 0.0.42-era no-arg pattern downstream
  code still uses. The argument is back to optional with a documented
  fallback ("clear unconditionally") for cleanup contexts that no
  longer have the token in scope.
* **Vector-index creation noise softened.** Atlas free / shared tiers
  cap the number of search indexes per cluster (M0/M2 = 3). Before this
  release the MongoDB provider raised `RuntimeError` from
  `_ensure_semantic_cache_vector_index` and printed a stderr stack
  trace at *every* agent boot. Now we classify quota / unsupported
  errors (`code=20 IllegalOperation`, "Atlas Search not enabled", etc.)
  and:
    * Track the affected collection in a `_vector_indexes_unavailable`
      set on the provider so downstream `find_similar_*` /
      `_knowledge_base_vector_search` helpers short-circuit to `[]`
      instead of issuing aggregations that can't succeed.
    * Skip `store_semantic_cache_entry` writes when the cache index
      is unavailable (silent writes would mask a misconfigured
      deployment).
    * Emit *one* structured WARNING per unique root cause, not one
      per collection — typical M0 boot now logs a single line instead
      of twelve stack traces.
  Filesystem and Oracle providers already had analogous graceful
  degradation paths (`_vector_search_disabled_for`, FAISS fallback);
  the MongoDB fix brings parity.
* **Mixed inclusion/exclusion projection in vector pipelines.** Six
  `$vectorSearch` consumers (`retrieve_persona_by_query`,
  `retrieve_toolbox_item`, `retrieve_entity_memory_records`,
  `retrieve_workflow_by_query`, `retrieve_summaries_by_query`,
  `find_similar_cache_entries`) built a `$project` stage that mixed
  inclusion (`"field": 1`) with exclusion (`"embedding": 0`) — Mongo
  rejects that combination outside of `_id`. Centralized into one
  `_build_vector_search_pipeline` helper that uses the
  `$vectorSearch` → `$project: {embedding: 0}` → `$addFields: {score:
  $meta}` shape across every consumer, so the rule can't be
  re-violated by future copies.
* **`MemAgent.llm_config` / `llm_provider` / `llm_model` are now
  exposed.** The constructor accepts an `llm_config` dict but never
  stored it back as an attribute, making introspection awkward for
  status pages and debug endpoints. Now `agent.llm_config` returns
  the resolved dict and the two `@property` accessors return the
  string fields callers actually want.
* **MongoDB KB dict-query path with `{"embedding": [...]}` now uses
  vector search.** Previously, a dict carrying a pre-computed
  embedding fell through to `.find({"embedding": [...], "limit": 5})`
  which literally filtered for that vector — almost always returning
  zero rows. Routed through the new
  `_knowledge_base_vector_search` helper instead.

## 0.0.44 — 2026-05-24

### Bug fixes

* **MongoDB provider:** `retrieve_by_query` with a string query on
  `CONVERSATION_MEMORY`, `KNOWLEDGE_BASE`, or `SHORT_TERM_MEMORY` no
  longer crashes with `filter must be an instance of dict`. Two new
  helpers — `find_similar_conversation_entries` and
  `find_similar_knowledge_base_entries` — implement proper Atlas
  `$vectorSearch` over per-turn embeddings (mirroring the existing
  `find_similar_cache_entries`). `SHORT_TERM_MEMORY` string queries
  degrade to `[]` for now (no semantic index yet).
* **MemAgent runtime:** `MemoryManager.retrieve_relevant_memories` now
  normalises provider return values — `None` → `[]`, PyMongo cursors
  are materialised — so the `len(results)` debug log can no longer
  raise `'NoneType' has no len()` and mask the real error. Every
  branch in the MongoDB provider's `retrieve_by_query` now returns a
  list rather than `None`.
* **MongoDB TOOL_LOG persistence:** `store()` no longer strips
  `thread_id`, `memory_id`, and `agent_id` when writing tool-log rows,
  so per-thread and per-agent audit views work end-to-end. Existing
  rows that were written without these fields are unchanged; new
  rows carry the full context.
* **MemAgent prompt building:** `_prepare_history_messages` now drops
  `role: "tool"` entries pulled back from `conversation_memory`. The
  OpenAI Chat Completions API rejects any `tool` message that isn't
  immediately preceded by an `assistant` message carrying matching
  `tool_calls` metadata — which stored history doesn't preserve.
  The tool placeholder text is already embedded in the persisted
  assistant turn, so no information is lost.

These changes ship the fixes verified end-to-end against the
Speechlyze MemAgent integration (analyses, docs, content generation,
TalkThrough).
