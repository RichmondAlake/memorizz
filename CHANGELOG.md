# Changelog

## Unreleased

_No unreleased changes._

## 0.8.0 — 2026-09-05

### Added

- Added ordered, bounded `StreamEvent` delivery through `run_stream_events()`
  and `arun_stream_events()`, including answer digests, cancellation, explicit
  terminal status, provider usage, and separate answer/persistence completion.
- Added default incremental answer delivery to the CLI, REPL, Playground and
  its desktop webview, plus MCP progress with an opt-in versioned answer-event
  extension. Added flushed CLI text/JSONL output and `--no-stream` compatibility.
- Added concrete provider streaming adapters, including native OpenAI Responses
  and Azure Chat streaming, with interruption, backpressure and incomplete-stream
  checks across supported providers.
- Added typed host observability events, trace propagation, memory-selection
  ledgers, artifact lineage, output contracts, browser-acknowledgement evidence,
  coverage profiles, and conservative outcome diagnostics.
- Added native SQLite, MongoDB and Oracle observability indexes, explicit
  migrations, scoped pagination, retention, backfill/parity checks, and operator
  tooling. Added an isolated local Oracle setup and live database test fixtures.
- Added Incident Finder, causal waterfall, memory/artifact/contract inspectors,
  selected-event navigation, structural comparison, health views and responsive
  mobile layouts, with synthetic browser and native WebKit regression fixtures.

### Fixed

- Historical bundled traces now retain child events, tool names, timestamps and
  model metadata. Health and insights share authorized selection/count semantics.
- Read completeness no longer implies end-to-end instrumentation or verified
  outcomes. Missing host hooks are shown as unavailable instead of usable actions.
- Search and cross-view links retain trace selection. Legacy JSON-string replay
  source lists no longer become individual-character resource identifiers.
- Streaming failures and cancellation preserve partial text without adding
  exception prose to answers; dumb-terminal REPL output now flushes incrementally.
- Oracle observability initialization now handles its first-install dynamic SQL
  correctly. Audit reports are kept outside the repository and release archives.

### Security and compatibility

- Added tenant-scoped, fail-closed operator access, audited explicit reveals,
  transient account/artifact resolver hooks, immutable event envelopes and inert
  replay drafts. MCP observability queries remain opt-in and read-scoped.
- `MemAgent.run()` still returns a complete string. Whole-answer validators
  continue to buffer until acceptance; streaming does not expose private tool
  drafts or rejected answers. Third-party cancellation remains cooperative.
- Host instrumentation, production migration/backfill and signed desktop
  distribution require separate deployment acceptance. This release does not
  change OpenSpeech or invent missing historical evidence.

## 0.7.0 — 2026-08-29

### Added

- Added provider-neutral structured tool outcomes across the SDK, CLI, MCP
  server, UI, observability, and continual-learning evidence. Tool executions
  now distinguish clean success, empty results, degraded capability, usable
  fallback, provider errors, and other failures without changing the payload
  exposed to the model.
- Added explicit `ToolResult`/`ToolOutcome` SDK types, per-turn
  `MemAgent.last_tool_outcomes`, provider/fallback trace metadata, outcome-aware
  cache admission, and Oracle tool-log persistence with a forward migration.
- Added provider-neutral `PersonalizationContext` SDK/MCP abstractions with
  opt-in, tenant-scoped cross-thread recall, natural-use prompt rules, bounded
  writing/profile inputs, and content-free per-turn memory supply evidence.
- Added memory-first observability across the SDK and UI, separating candidates
  retrieved, context supplied, and sources explicitly referenced while marking
  behavioral style attribution as not measurable.
- Added deterministic canonical entity identities and dry-run-first duplicate
  consolidation with conflict reporting and reversible soft supersession.

### Fixed

- Internet-provider failures and offline placeholders no longer appear as
  successful green tool completions. The CLI and UI now report states such as
  **Completed via fallback** and **Provider error**.
- Source distributions now exclude downloaded benchmark corpora under
  `eval/datasets/`, preventing local evaluation data from inflating or leaking
  into release artifacts.

### Security

- Automatic memory recall now rechecks exact `memory_id` and `user_id` ownership
  after every provider query, preventing an overly broad third-party provider
  from supplying cross-tenant prompt context.
- Canonical entity `identity_key` binding remains trusted host/application
  authority and is deliberately absent from the model-visible entity tool.
- Personalization authority rules are rendered before memory-derived content,
  so a large or adversarial excerpt cannot truncate away the instruction
  boundary. Reference attribution also ignores ambiguous single-token overlap.

## 0.6.3 — 2026-08-25

### Fixed

- The npm postinstall bootstrap and on-demand launcher fallback now explicitly
  request uv-managed Python 3.12. The npm CLI therefore works on hosts whose
  default system Python is older than MemoRizz's supported Python range.
- npm launcher tests now cover both the pinned bootstrap and recovery paths,
  including their managed-Python requirement.

## 0.6.2 — 2026-08-25

### Fixed

- Source distributions now explicitly exclude local-only plans, credentials,
  research output, dependency trees, and generated desktop artifacts even when
  built from a developer checkout. This is the canonical production package
  for the 0.6.1 runtime fixes; 0.6.1 is superseded because its manually
  uploaded source archive contained ignored development files.

## 0.6.1 — 2026-08-25

### Fixed

- Entity-memory tools now publish an explicit nested `name`/`value` schema,
  canonicalize common model-generated attribute aliases such as `attribute`
  and `attribute_name`, and reject unknown model-facing fields. This prevents a
  valid user-profile fact from failing Pydantic validation before persistence.
- Callable tool schemas now inline annotation-local Pydantic definitions, so
  nested input models do not emit broken root-level `$ref` pointers when sent
  to OpenAI-, Anthropic-, or MCP-compatible providers.
- Release artifacts explicitly target Core Metadata 2.4 so current stable
  Twine and PyPI validation remain compatible with newer Hatchling defaults.
- The npm launcher now rejects stale or self-resolving `memorizz` executables,
  prefers the exact uv-managed release, and falls back to the version-pinned
  package specification instead of silently running an older installation.

## 0.6.0 — 2026-08-24

### Evaluation, MetaHarness, and memory-platform additions

- Added an explicitly unofficial Terminal-Bench 2.1 forecast command with the
  canonical 89-task/445-trial protocol, a dated leaderboard context snapshot,
  nominal spend/headroom scenarios, pilot latency extrapolation, Wilson
  uncertainty, and fail-closed rank reporting until at least ten distinct
  pilot tasks exist.
- LongMemEval-V2 official-adapter runs now verify upstream checksums before
  spend, forecast embedding/reader/synthesis/judge cost with a configurable
  preflight limit, batch-persist thousands of chunks, pin question order,
  fingerprint the actual scorer/dependencies, and report MemoRizz-owned model
  usage separately from the downstream reader.
- Added sanitized release evidence for a passing 1/1 LongMemEval-V2 official-
  adapter smoke and a zero-cost, ten-ability BEAM 128K diagnostic, with exact
  paper-comparability exclusions.

- Evalground now distinguishes a normalized memory-retrieval diagnostic from
  **Full MemAgent execution**. The latter loads a secret-free selected-agent
  template onto isolated benchmark memory, calls `MemAgent.run()`, disables
  side-effecting integrations, and reports actual automatic-retrieval/context-
  assembly evidence. The SDK and `memorizz eval run` expose the same mode,
  template, query-expansion, rerank, and reader-repair controls.
- Memory diagnostics now use grouped semantic-query RRF, label-blind
  capacity/boundary expansion, source-linked causal and multi-turn event
  memories, widest-provenance representative selection, parent-source
  deduplication, timestamp preservation, relative-date annotations, and
  unambiguous date normalization. One bounded formatting-only repair handles
  malformed JSON-like local-reader output.
- Multi-source diagnostic evidence now renders an explicit JSON source-ID list
  and requires one citation per array element. This closes an ambiguity where a
  grounded reader could copy two comma-separated IDs as one invalid citation.
  A bounded GPT-5.5 rerun achieved 1.0 answer, Recall@6, and grounding on both
  LoCoMo-Plus diagnostic and full-MemAgent lanes; the sanitized artifact keeps
  the one-sample/non-paper-comparable boundary and full spend accounting.

- Added a four-task, task-native MetaHarness regression pinned to GPT-5.6 Luna
  and Claude Opus 5 at medium effort. It compares Codex-only, Opus-only,
  always-on, and preregistered risk-routed MemAgent strategies with fresh
  Filesystem scopes, durable write approvals, host verification, 32 held-out
  checks, complete cost/usage evidence, and no LLM judge.
- Notebook 06 now opens with a model-aware example result table and reconstructs
  the same table from the sanitized evaluation artifact at the end, exposing
  reusable table rows plus an artifact SHA-256 and protocol fingerprint.
- Claude Code harness tasks can pin claude_effort to low, medium, high, xhigh,
  or max; invalid values fail before launch. Secret-value redaction now requires
  a credential-prefix boundary, avoiding false positives such as task_native
  while continuing to redact real provider keys.
- Added the paid, paired MetaHarness factorial evaluation across direct Codex,
  direct Claude Code, MemAgent wrappers, mandatory panels, adaptive panels,
  Filesystem, and Oracle AI Database, with a sanitized educational artifact and
  Notebook 06 analysis.
- Hardened evaluation reporting with layered dotenv loading, publishable path
  redaction, safe numeric token telemetry, explicit judge scale normalization,
  construct-irrelevant rationale rejection, bounded retries, and durable cost
  evidence for failed judge calls.
- Made deterministic required-criterion recall, grounding, schema validity, and
  host verification the primary fixture-quality gates; LLM scalar scores are
  explicitly secondary diagnostics.

### Fixed

* Terminal-Bench cost accounting now prices the selected registered OpenAI
  model and rejects unknown or variant pricing instead of silently applying
  GPT-5.6 Terra rates to every model.
* Dataset verification now treats GitHub remotes with and without `.git` as
  equivalent, detects source checkouts that also contain benchmark data, and
  reports BEAM conversation coverage without presenting a partial smoke corpus
  as the full variant.

* Fixed the memory benchmark path that left LoCoMo-Plus cognitive cues outside
  top-k while unrelated MetaHarness security work correctly had no score
  effect. On the same one-sample local diagnostic, gold Recall@6 moved from 0.0
  to 1.0; Qwen answer quality remains reported separately rather than being
  misrepresented as retrieval success.
* Exact duplicates returned through multiple automatic-retrieval query variants
  now retain their strongest provider score, and grouped records retain complete
  linked-source provenance. Full MemAgent reports no longer claim diagnostic
  fusion or semantic-cache decisions that did not run.
* Oracle knowledge-base rows now persist source, parent, linked-source, and
  structured metadata provenance through additive migration 006. The same
  evidence shape already survives Filesystem and MongoDB storage.
* Invalid non-list retrieval and tool-log values from third-party providers now
  degrade to empty evidence instead of leaking sentinel attributes into prompt
  provenance or preventing an otherwise healthy model turn.

* Coordinated MetaHarness panels now retrieve one ranked, tenant-scoped evidence
  snapshot for the original request and reuse it across role-specific workers.
  Dependency outputs reach later stages through bounded allowlisted context,
  while synthesis is grounded in root memory and instructed to preserve only
  supported evidence. The root reuses an identical delegate evidence snapshot
  when it fits the coordinator budget, removing a duplicate provider read while
  closing the earlier ungrounded-consolidation failure mode. The optimization is
  provider-neutral and now has explicit filesystem, MongoDB, and live local
  Oracle coverage. MongoDB knowledge-base evidence preserves `memory_id` and
  degrades to a bounded, exactly scoped lexical retrieval when Atlas vector
  search is unavailable; Oracle's legacy `store(data=..., memory_id=...)` path
  now persists that scope instead of silently generating an unrelated one.
  Shared context-cache keys include the effective evidence owner, while
  coordinated panels explicitly use the root owner for procedural retrieval.
* Deterministic, host-verified, read-only delegation can now reuse a semantic
  cache entry only when the caller supplies a data version and the complete
  plan, delegate, model, harness, and policy fingerprint still matches.
  Successful coordinated root responses are recorded in conversation memory
  and their workflow is captured by the learning control plane.
* Multi-agent result consolidation now treats an intentionally model-less root
  as a supported deterministic aggregation mode instead of logging a false
  provider error. Configured synthesis failures remain visible as structured
  partial-workflow evidence, while successful model synthesis records provider,
  model, latency, and available usage. Model-based consolidation now also honors
  the root coordinator's configured instruction.
* MetaHarness action budgets now count one vendor command/tool lifecycle once
  across `in_progress` and `completed` events, and runtime-backed MemAgents
  propagate their configured harness model pin into the executed `HarnessTask`.
* Filesystem lexical fallback now ranks sufficiently overlapping scoped records
  when delegation adds a role-specific query suffix, instead of requiring the
  complete extended query to occur verbatim. Multi-agent orchestration now
  treats a failed runtime-backed harness result as a failed subtask rather than
  successful serialized text.
* Meta-harness authentication readiness now has one secret-free
  `error_code`/message/remediation contract across the SDK, CLI, local UI, and
  first-party MCP server. Claude Code reports its required bare-mode
  credentials before launch, Codex verifies either an environment key or its
  cached CLI login, and runtime 401/login failures are normalized without
  exposing vendor output or credential values.
* Persisted harness-backed agents can be created on developer, CI, or
  deployment hosts before a vendor CLI is installed. Construction still
  validates adapter registration and native recursion, while executable and
  authentication readiness remain fail-closed at execution with the same
  structured remediation across SDK, CLI, UI, and MCP.
* Real Codex + Claude Code notebook execution now preserves numeric token-usage
  telemetry through redaction, falls back to tenant/thread-scoped lexical
  retrieval when filesystem records have no vectors, and uses a realistic
  bounded multi-turn output budget in the two-harness review example.

### Added

* A sixth educational MetaHarness notebook and versioned factorial evaluator
  separate direct-MetaHarness versus MemAgent wrapper overhead, mandatory
  Codex-plus-Claude coordination, adaptive fallback, and Filesystem-versus-
  Oracle effects. The 12-arm protocol uses exact call-policy validation,
  provider-neutral evidence fingerprints, deterministic model-less union,
  independently blinded per-candidate judging, reverse-order balancing, and an
  explicit one-fixture inference boundary. It reinterprets the earlier
  one-call adaptive artifact instead of presenting it as a two-harness result,
  makes no external call by default, and keeps paid execution behind an
  environment opt-in. A dedicated `notebooks` package extra provides a
  consistent Jupyter kernel and execution toolchain.
* Memory evaluation now has fail-closed versioned protocol manifests,
  smoke/regression/paper profiles, pinned official-source synchronization and
  checksum verification, calibrated reciprocal-rank fusion, conservative
  source-linked constraint memory, persistent corpus embeddings, grounded
  citation scoring, an oracle-reader ceiling, confidence intervals, and
  phase/cost accounting. The CLI and Evalground expose the same profile and
  Filesystem/Oracle controls; diagnostic runs cannot claim paper comparability.
* The memory-provider contract now includes portable bulk storage, scoped
  normalized search, and declared capabilities, with matching Filesystem,
  MongoDB, and Oracle behavior.
* Multi-agent orchestration now offers `model`, `deterministic`, and `primary`
  consolidation strategies plus explicit dependency/synthesis context bounds.
  Capability reports expose ranked evidence, workflow snapshot reuse,
  dependency propagation, and verified delegation-cache support.
* A fifth MetaHarness notebook compares Codex-only, Claude-only, and a
  MemAgent-coordinated Codex + Claude panel using a blinded LLM judge, gold
  findings, host verification, memory grounding, latency, tokens, normalized
  actions, reported/estimated cost, and cost per quality point. Provider-aware
  enforceable budgets and a fail-closed validity gate prevent failed baselines
  from being ranked as wins; invalid attempts retain diagnostic JSON evidence
  without a judge ranking or winner. Filesystem remains the default evaluation
  provider; an opt-in Oracle mode runs preflight before spend, records provider
  capabilities, and deletes the synthetic scopes during cleanup.

### MetaHarness and interface foundations

* A memory-first MetaHarness for running Codex, Claude Code, OpenHands, and
  native MemAgent workers behind one durable contract. It includes
  deterministic routing, exact-envelope host approval, workspace policy and
  leases, cancellation, bounded budgets, normalized events and results,
  host-side verification, tenant-scoped memory context, and verified
  continual-learning evidence.
* MetaHarness parity across the Python SDK, persisted MemAgent configuration,
  `memorizz harness` CLI, local operator UI, and six first-party MCP tools plus
  a run resource. OpenHands is fail-closed until an operator-supplied isolated
  wrapper is configured.
* A developer-oriented documentation information architecture with a single
  installation path, core runtime/scoping model, tools and durable approval
  guide, model-provider and scheduled-automation guides,
  configuration/secrets reference, capability/preflight guide, curated Python
  API, troubleshooting playbook, and static documentation quality gates.
* The first-party MCP server now exposes a 23-tool operational surface with
  local agent update/deletion, combined agent/cache/learning/observability
  inspection, scoped learning compilation, conversation compaction, and
  governed harness execution. Tool
  schemas reject undeclared arguments, while credentials, local-path ingestion,
  approval decisions, and outbound MCP configuration remain host-only.
* A stable headless-runtime capability report and subprocess coverage confirm
  that the SDK, CLI, and stdio MCP server run without a display server or an
  import of the optional UI application.

### Fixed

* Developer examples now use the real public entity, episodic, semantic-cache,
  Toolbox, shared-memory, and agent-mode APIs; supported Python versions and
  local UI authentication/read-only guidance now match the package.
* Persisted agent updates retain continual-learning workflow and skill memory
  types when application-mode defaults are recalculated.
* Explicit SDK model overrides no longer initialize the persisted provider as a
  side effect, and `tool_access` now survives construction, save/load, and MCP
  agent updates.
* Harness diagnostics no longer crash when an optional configured memory
  provider is unavailable, source checkouts report the source package version,
  and Codex project configuration cannot re-enable hooks, web search, egress,
  extra writable roots, or non-MemoRizz MCP servers for a governed run.

## 0.5.3 — 2026-08-20

### Added

* Explicit persisted-agent creation parity across the SDK, CLI
  (`memorizz agents create/list/show`), local UI, and first-party MCP server.
  MCP creation is a local-stdio-only write that resumes through durable host
  approval; remote HTTP creation remains disabled until agent ownership is
  durably tenant-scoped.

* Privacy-safe `observability_context` on `MemAgent.run` and `run_stream`, with
  allowlisted page/thread/grounding lineage, request-context fingerprints,
  cache hit/miss/bypass events, timeline metadata, and deterministic Trace
  Insights for wrong-page and missing-grounding failures.
* A provider-neutral memory-first learning control plane with immutable,
  idempotent run/tool/cache/workflow/outcome/skill events; deterministic
  incremental compilation; bounded and explainable `EvidencePack` retrieval;
  host-verified outcome evidence; and reversible, single-use, operator-approved
  forgetting plans. It reuses private shared memory on filesystem, MongoDB, and
  Oracle rather than adding a service or schema.
* SDK (`with_learning_control_plane`, reports, retrieval explanations,
  compiler, durable forgetting), `memorizz learning` CLI commands, and a local
  UI Learning Control Plane page with agent configuration support.
* Memory-first adapters and revision manifests for the official
  LongMemEval-V2 harness, SWE-bench Lite Docker grader, and all four MemBench
  memory tracks, plus optional benchmark dependency groups and local result
  documentation.
* `CompletionPolicy`, `CompletionCandidate`, `CompletionDecision`, and
  `CompletionRejectedError` for host-enforced, auditable final-answer gates
  with bounded same-loop retries and builder/persistence support.
* Batch embedding support in `EmbeddingManager` and the OpenAI embedding
  provider for high-volume benchmark ingestion.
* The LongMemEval-V2 adapter now isolates the official base memory registry from
  unrelated optional backends, avoiding incompatible eager dependencies.
* Two arXiv-compatible companion systems-paper drafts with a shared
  bibliography: one covering the memory-first agent harness, taxonomy, unit
  shapes, context tokenomics, providers, and runtime components; the other
  covering the agent-memory and continual-learning platform. The latter
  explicitly attributes its recency/importance/relevance inspiration to
  Generative Agents. Both include reproducibility statements and the sanitized
  filesystem/Oracle benchmark comparison.

### Changed

* The local control-plane UI now uses a restrained, high-density visual system
  with matte surfaces, metric-first overview cards, compact navigation, and
  semantic observability states for provenance, ownership, grounding, cache,
  and trace identity. Trace Insights group related signals into six operational
  summaries instead of presenting every counter at equal weight.
* Continual-learning skill candidate, promotion, demotion, and deprecation
  transitions now emit control-plane evidence; per-turn context assembly can
  delegate multi-source recall to one tenant-scoped token budget.
* LongMemEval-V2, SWE-bench Lite, MemBench, and Terminal-Bench adapters now
  exercise and report the control plane; benchmark memory can run on filesystem
  or Oracle, with Oracle vector-dimension preflight before paid model calls.
* LongMemEval-V2 retrieval combines tenant-scoped lexical and vector lanes,
  diversifies results across trajectory sources, and preserves both ends of
  bounded accessibility trees so exact late-menu UI labels are not silently
  discarded.
* Terminal-Bench and SWE-bench agents require a successful substantive
  `terminal_verify` call before the host accepts a final response.
* The filesystem provider loads FAISS lazily per provider and supports
  `use_faiss=False` exact cosine search for small stores or incompatible native
  runtime combinations.

### Fixed

* Reloaded agents now retain their secret-free LLM provider/model metadata in
  `llm_config`, keeping SDK, CLI, MCP, and UI introspection consistent with the
  hydrated runtime provider.
* Entity-memory reads are strictly scoped by both `memory_id` and `user_id`;
  authenticated requests no longer inherit anonymous legacy rows. A separate
  operator-only `EntityMemory.migrate_legacy_scope()` API provides deliberate
  provider-managed adoption for filesystem, MongoDB, and Oracle.
* Learning-control-plane `EvidencePack` retrieval now uses entity memory's
  bounded exact fallback and carries retrieval mode/degradation metadata when
  vector search is unavailable or fails.
* MongoDB entity vector failures propagate typed evidence to the fallback
  layer; lazy missing indexes remain eligible for creation, failed filter-index
  reconciliation remains retryable, and stale status entries are invalidated
  after successful creation or update.
* Oracle entity upserts lock and verify the existing memory/user owner before
  updating a globally unique `entity_id`, preventing direct SDK calls from
  moving another tenant's entity row.
* The model-visible `entity_memory_upsert` schema no longer contains
  `memory_id`; active scope is supplied only by the host. Repeated identical
  entity relations are deduplicated and no longer trigger redundant embedding
  and storage writes.

* Oracle `list_all(shared_memory)` now returns its typed relational projection,
  allowing immutable learning events and compiled artifacts to survive and be
  enumerated after agent reload.
* Completion-gated streaming buffers candidates until host acceptance, cached
  completions are revalidated by the current policy, and tool-evidence policies
  cannot be bypassed by a cached string.
* Exact semantic-cache repeats use their deterministic scoped key before vector
  search, avoiding false misses caused by floating-point score rounding at a
  strict `1.0` threshold.
* SWE-bench completion retries now have a bounded reservation beyond the normal
  work-loop tool budget, so the mandatory final verification cannot be blocked
  by an exhausted exploration budget.

## 0.5.2 — 2026-08-16

### Added

* Optional Harbor/Terminal-Bench adapter with native ATIF v1.7 trajectories,
  isolated per-trial memory, bounded terminal execution, token/cost reporting,
  spend guards, and deadline-aware finalization.
* `RetrievalPolicy` separates automatic episodic/knowledge recall from memory
  storage and tool availability, with exact thread and KB-namespace scopes.
* Capability flags for thread-scoped summaries, retrieval policy, shared-agent
  concurrency safety, logical tool trace names, and session-safe cache defaults.
* Durable observability timelines that expand streamed trace bundles and tool
  execution logs by their real memory/thread scope.
* Read-only Trace Insights for agent/thread windows, with JSON export and
  evidence-ranked recommendations for tool, prompt, retrieval, context, and
  continual-learning improvements.
* Canonical v2 trace envelopes with application, agent, run, turn, root trace,
  span, memory, thread, and user identity. Named/application-scoped agents use
  stable IDs and automatically upsert themselves and new memory associations.
* Provider-neutral cursor-paginated observability queries, with a native
  indexed MongoDB implementation and query latency, scan, truncation, and
  freshness metadata.
* Durable recommendation reviews, versioned draft Evalground experiments, and
  trace-linked verified feedback/task outcome records.
* Optional local UI token authentication, signed expiring sessions, read-only
  provider inspection, field-level trace redaction, and trace-view audit logs.
* Oracle migration `005_scoped_retrieval_052.sql` for summary thread markers,
  knowledge-base namespaces, safe legacy backfill, and exact-scope indexes.

### Changed

* A bare `MemAgent()` now uses the filesystem memory provider at
  `~/.memorizz/memory` by default. `MEMORIZZ_MEMORY_ROOT` relocates it and
  `memory_provider=False` remains the explicit stateless opt-out.
* OpenAI's provider supports a Responses API tool loop and current GPT-5.6
  completion limits, reasoning, context-window, usage, and prompt-cache rules.
* Summary registries, expansion, source-message reconstruction, MongoDB vector
  lookup, and Oracle summary storage now preserve exact `thread_id` scope.
* Shared `MemAgent`, progressive router, stream callback, and semantic-cache
  execution state is context-local for concurrent web requests.
* Routed stream traces report both `logical_tool_name` and
  `model_tool_name`; `tool_name` is the application-level logical name.
* Persisted tool-result traces retain success/error status and monotonic
  duration so the observability UI can calculate failure and latency signals.
* Model and tool spans are captured for synchronous and streaming execution;
  model spans include provider/model identity, duration, and available token
  usage, while all spans retain parent/root correlation.
* Trace bundles now live in private observability records instead of hidden
  conversation rows, so telemetry cannot enter chat recall or inflate raw
  conversation history. Legacy conversation bundles remain readable.
* The traces dashboard uses bounded provider queries instead of collection-wide
  reads and exposes a cursor JSON endpoint at `/traces/events.json`.
* Semantic cache is session-scoped by default and fingerprints request context
  to prevent stale reuse when page or grounding context changes.
* MCP client/server dependencies moved to `memorizz[mcp]`; importing and using
  non-MCP Memorizz features no longer requires MCP, Uvicorn, or cryptography.
  The npm bridge installs this extra so its complete CLI surface remains intact.
* The continual-learning guide now makes experimental-arm isolation, outcome
  grading, and developer-versus-user skill authority explicit.

### Fixed

* Automatic recall can no longer silently broaden a requested thread or
  namespace when a third-party provider ignores those query parameters.
* In-memory semantic cache entries cannot cross memory scopes when one agent is
  shared by multiple concurrent requests.
* Filesystem and Oracle retrieval apply thread/namespace boundaries before
  top-k ranking, avoiding false misses caused by out-of-scope candidates.
* Per-call stream callbacks and resolved execution IDs survive the UI worker
  boundary, and UI edits preserve persisted retrieval/governance settings.
* Non-progressive tool routing can invoke every schema it disclosed instead of
  incorrectly requiring the hidden discovery handle.

## 0.5.1 — 2026-08-12

### Added

* Structured `ApprovalResumeResult` evidence with the exact consumed proposal,
  exact tool result, optional assistant continuation, and deterministic
  `continue_model=False` operation.
* `MemAgentBuilder.with_oracle_from_env()` and `.with_e2b_from_env()` presets,
  including local-runtime readiness, Oracle preflight, and secret-free reports.
* Response-free semantic-cache inspection, tenant-scoped
  `agent.observability_summary(...)`, and agent lifecycle/context-manager APIs
  with optional exact-scope cleanup.

### Changed

* Summary generation accepts explicit memory, tenant, and thread scopes and no
  longer inherits the most recently executed user's identity.
* Streaming provider/API failures carry typed terminal event fields and may be
  re-raised with `raise_on_provider_error=True`.
* Deterministic delegation plans normalize `SubTask` instances into JSON-safe
  records and round-trip status/results through `SubTask.from_dict()`.

### Fixed

* Optional provider tenant filters consistently distinguish omitted/unscoped
  reads from explicit `None` anonymous reads.
* Oracle preflight now reports the database Release Update in `version_full`
  instead of exposing only the compatibility version.
* Oracle preflight now fails closed on configured-embedder/schema dimension
  mismatches before a later write can raise `ORA-51803`.

## 0.5.0 — 2026-08-12

### Added

* First-class MCP client connectivity over stdio, Streamable HTTP, and legacy
  SSE, including Notion and Google Calendar OAuth presets, encrypted
  credentials, SSRF controls, mutation approval, and CLI/UI management.
* A first-party MemoRizz MCP server with 11 memory, conversation, and agent
  tools; resources and a prompt; local stdio; authenticated Streamable HTTP;
  per-principal tenant isolation; scoped bearer grants; and explicit remote
  write, execution, and agent-exposure policy.
* Durable, generic human approval proposals with exact argument hashes,
  serialized checkpoints, expiry, approver audit fields, atomic single-use
  consumption, Python/UI decision surfaces, MCP-specific CLI controls, and
  exact resume semantics. Model-visible `approved` and `confirm` arguments have
  been removed.
* First-class progressive tool disclosure through `SemanticToolRouter`, with
  scoped top-k retrieval, stable discovery/invocation handles, allowlisted
  dispatch, strict JSON Schema/signature binding, aliases, deprecated-argument
  migration, and duplicate/retry budgets.
* Size-aware `ToolResultPolicy`: small responses remain inline, large responses
  are persisted exactly once behind digest-bearing pointers, and expansion
  tools are never re-offloaded.
* Semantic-cache governance with tenant-complete keys, admission/bypass rules,
  model/prompt/tool/data fingerprints, freshness and provenance, operational
  counters, and domain/tag/version invalidation APIs.
* A governed semantic layer with versioned entities, measures, dimensions,
  relationships, policies, synonyms, lineage, and validated query plans.
* Oracle local-runtime bootstrap, structured preflight, configurable vector
  index policies, exact-search fallback, sizing diagnostics, and transactional
  `delete_scope(...)` cleanup.
* `memorizz.capabilities()`, `agent.capability_report()`, `memorizz
  capabilities`, and `memorizz oracle preflight` for feature-level deployment
  checks.
* Provider-neutral browser control with an isolated Browser Use provider,
  bounded natural-language tasks, domain policy, direct-IP blocking, private
  fixed-worker execution through the isolated tool interpreter, structured
  results, process-group timeouts, deterministic browser cleanup, builder and
  persistence support, and CLI/UI configuration. Every model-initiated browser
  call uses generic durable HITL.
* Generic `/approvals` and browser-specific `/browser` REPL commands, plus
  `--browser-control/--no-browser-control` on `chat` and `run`.

### Changed

* E2B uses the current `Sandbox.create(...)` lifecycle with an older-SDK
  compatibility path, one bounded stateful session, normalized result shapes,
  fail-fast API-key validation, explicit egress/compute metadata, and reliable
  context-manager termination.
* GraalPy subprocess mode is now explicitly a bounded execution provider—not a
  strong sandbox—with private-path confinement, environment allowlisting,
  resource limits, and fail-closed network policy. The packaged Java wrapper is
  mandatory for `UNTRUSTED` mode. The UI now keeps mode selection independent
  from egress and defaults trusted subprocess execution to network denied.
* Multi-agent orchestration and decomposition use the configured `LLMProvider`
  and model, support deterministic plans, make delegates operational, propagate
  tenant/request/tool/trace context, scope shared memory by workflow, and
  expose dependency-aware partial failures.
* `MemAgentBuilder` now covers sandbox, Toolbox, Skillbox/authored skills,
  skills marketplaces, tool/context policies, approval stores, delegation, the
  semantic layer, validation, and persistence.
* Deterministic Toolbox registration no longer constructs an LLM or embedding
  client. Trusted callables can be rebound after restart only through explicit
  registries or import references.
* Authored Skillbox retrieval is independent of continual learning and applies
  agent/user isolation before vector top-k selection.
* Browser Use is deliberately invoked through the isolated Python interpreter
  of a separately installed tool environment because its current package
  dependency line is incompatible with MemoRizz's MCP 2.x requirement;
  model-provided code is never executed.
* Oracle setup and teardown no longer contain, infer, return, or print default
  database passwords. Non-interactive setup requires explicit credentials and
  interactive setup uses hidden input. UI-created Docker containers pass
  credentials through a private short-lived env file rather than process argv.

### Fixed

* Oracle now persists complete Toolbox JSON Schemas and restores required,
  default, enum, nested-type, alias, policy, and import-reference metadata.
* Oracle summaries now retain canonical source-message IDs, period boundaries,
  unit counts, summary-by-ID lookup, normalized message links, atomic original
  marking, and lossless expansion parity with other providers.
* Oracle knowledge-base and short-term writes return their physical record IDs,
  restoring reliable write/read round trips through the first-party MCP server.
* MCP mutation classification now honors tool annotations and configured
  metadata before conservative name inference.
* Tool-log listing now preserves exact anonymous/user scope and cannot leak a
  tenant's records through an omitted optional argument.

## 0.4.0 — 2026-07-19

### Added

* Opt-in passive SHADOW-traffic evaluation now observes only new,
  post-distillation workflows in a bounded background queue. It performs
  deterministic, tenant-scoped canonical-trajectory and business-outcome
  comparisons without prompt injection, LLM calls, tool re-execution, a
  second action-taking agent, or automatic activation.
* Workflow records persist versioned, idempotent `shadow_evaluations`; skill
  records expose a bounded derived `stats["shadow"]` aggregate and the
  advisory `get_shadow_readiness(skill_id)` API.
* Filesystem, MongoDB, and Oracle apply SHADOW lifecycle and tenant filters
  before final top-k retrieval. MongoDB reconciles Skillbox vector-index
  filter fields, and Oracle migration
  `003_add_shadow_evaluations.sql` adds a JSON-constrained workflow CLOB.
* The Local UI exposes passive evaluation and displays observation count,
  trajectory-match rate, matched-trajectory success rate, last evaluation,
  and readiness reasons while preserving explicit activation.

### Changed

* Active attribution and drift monitoring now accept only ACTIVE skills.
  SHADOW evidence never enters `skills_activated`, activation counters, or
  rolling demotion statistics.
* Parts 5 and 6 of the continual-learning notebook are split into shorter
  narrated stages while retaining case-level and aggregate DataFrames.
* Release recovery verifies wheel bytes exactly and compares normalized sdist
  contents, avoiding false failures caused only by cross-platform gzip/tar
  metadata.

## 0.3.0 — 2026-07-19

### Added

* Learned skills now persist a trust-aware `injection_role` (`user` or
  `developer`) across filesystem, MongoDB, and Oracle. `user` remains the
  backward-compatible default; developer-authority promotion is rejected
  unless shadow review is required, and reviewers can choose the role
  programmatically at activation.
* MemAgent prompt assembly emits reviewed developer skills as a separate
  developer message. Official OpenAI requests preserve that native role;
  Anthropic maps it to the top-level system parameter; Ollama,
  OpenAI-compatible endpoints, Hugging Face, and MLX use their
  system-equivalent compatibility path.
* The agent form exposes learned-skill authority and shadow review, and the
  skill lifecycle page displays each stored role and the activation authority.
* Oracle schemas now include `skillbox.injection_role`, with additive
  migration `002_add_skill_injection_role.sql` for existing deployments.
* The continual-learning notebook now compares raw workflow replay, the same
  reviewed skill at user authority, and that skill at developer authority in
  case-level and aggregate DataFrames covering accuracy, provider tokens,
  latency, role placement, and raw-workflow isolation.

### Changed

* Continual-learning documentation now defines the workflow-to-skill use case,
  instruction hierarchy, provider mappings, Oracle-backed order ingestion,
  role safety boundary, and the invariant that skill agents capture workflows
  as evidence without automatically retrieving them into prompts.

### Fixed

* Oracle lazy vector-index creation now passes the live cursor to
  `_table_has_column(cursor, table_name, column_name)`. Previously skill
  retrieval logged a missing-argument warning and returned no matches even
  though earlier promotion writes could succeed.

## 0.2.2 — 2026-07-19

### Added

* The interactive CLI banner now shows the installed Memorizz version and the
  live continual-learning state. `memorizz config` and `/config` also report
  the selected memory backend, embedding mode, and learning state.
* The Local UI now gives workflow trajectories and learned skills a dedicated
  **Continual Learning** navigation group, and its Oracle settings expose an
  explicit in-database ONNX versus external-provider embedding mode.

### Changed

* The automations worker now uses the configured filesystem, MongoDB, or Oracle
  memory backend instead of constructing an Oracle provider unconditionally,
  and closes the provider cleanly when the worker exits.
* The CLI guide now documents persistent and one-session continual-learning
  activation, the requirement for tool-calling workflows, and external-vector
  Oracle configuration.

### Fixed

* First-party CLI and UI Oracle clients now preserve existing external-vector
  schemas when an external embedding provider is configured, while retaining
  in-database embeddings as the default for new setups.
* Oracle vector-index creation now skips legacy or non-vector tables that do
  not contain an `embedding` column instead of issuing an invalid database
  operation.

## 0.2.1 — 2026-07-18

### Fixed

* Anthropic prompt caching no longer mutates MemAgent's reusable conversation
  history or accumulates stale `cache_control` fields across tool-loop
  iterations. Streaming and non-streaming calls now share one request builder
  that owns deep copies of message/tool data, preserves caller-supplied
  breakpoints, and enforces Anthropic's four-breakpoint request-wide limit
  before any API call.
* Release automation now verifies and accepts an existing byte-identical PyPI
  upload before requesting Trusted Publishing credentials, allowing a
  project-token fallback release to continue to GitHub/npm/Homebrew jobs. It
  also supports resuming an existing immutable tag through manual dispatch and
  skips npm publication cleanly when its repository token is not configured.

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
  backend; release CI installs the UI and optional-provider test dependencies
  instead of silently skipping those suites; and docs use the canonical CLI
  commands and valid links.

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
