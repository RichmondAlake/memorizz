# Changelog

## 0.1.0 — 2026-06-18

### Features

* **New interactive CLI — `memorizz` is now a Claude-Code-style local agent.**
  Running `memorizz` with no arguments launches an interactive REPL that streams
  the agent's replies live, with `/` slash commands (`/help`, `/model`,
  `/provider`, `/ollama`, `/code`, `/memory`, `/history`, `/agents`, `/agent`,
  `/new`, `/persona`, `/tools`, `/ingest`, `/ui`, `/login`, `/config`, `/clear`
  (wipe memory, with confirmation), `/cls`, `/exit`).
  Also adds `memorizz chat`, one-shot `memorizz run "<prompt>"`, `memorizz init`,
  `memorizz config`, and `memorizz --version`. `python -m memorizz` now works.
* **Zero-config local stack.** With no API key set and an Ollama daemon running,
  the CLI defaults to Ollama (LLM + `nomic-embed-text` embeddings) + an on-disk
  FileSystem memory store under `~/.memorizz/memory` — a 100%-local agent with no
  keys. Cloud keys (Anthropic / OpenAI / Azure) are auto-detected when present.
* **`/code` coding mode.** `memorizz --code` (or `/code` in the REPL) enables the
  agent's self-aware file read/write + bounded command tools, scoped to the
  working directory (writes on, deletes off).
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
