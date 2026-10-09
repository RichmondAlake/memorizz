# Changelog

## Unreleased

### Added

- Interactive CLI: a turn on a harness shows what the harness is doing while
  it works, with a running clock (`Codex is working (gpt-6.1-sol) · 4s`,
  `Codex: running a command`, `Codex: calling <tool>`, `Codex: writing`)
  instead of "preparing" until the answer arrives. Streams from a runtime
  harness carry the same progress as `harness_running` and `harness_activity`
  status events (kind of work only, no command text or reasoning).

### Fixed

- Interactive CLI: keys pressed while a turn streams or a command loads (an
  arrow while `/harnesses` checks the harnesses, say) no longer echo as
  `^[[B` over the output; they wait for the next prompt or picker.

## 0.16.3 — 2026-10-09

### Added

- Interactive CLI: the chat's model is carried over to a harness when the
  harness can run it; `/harness <name> <model>` and `/harness model [name]`
  choose the model (with a picker over the harness's catalogue), the switch
  message and the status bar show the model that will answer, and
  `/harnesses` lists each harness's default model.

### Fixed

- Memory history: saving an agent that has a persona logged "Memory history
  preparation failed ... Unable to serialize unknown type: Persona" and
  skipped the journal entry (it also lost the agent's previous version, so
  updates were journalled as creations). The journal now stores the persona
  as the providers do. It surfaced in the chat after `/harnesses`, which
  switches on history recording.
- Interactive CLI: terminal focus reports (sent when you click back into the
  window) were read as the Esc key, so pickers closed with "No change." on
  their own and the prompt could gain a stray "[I". They are now ignored.

## 0.16.2 — 2026-10-08

### Added

- Interactive CLI: `/session <run> [N]` shows one recorded Codex or Claude
  Code session inside the chat (header, last N turns, tool activity counted
  between turns), taking the short run id `/sessions` prints. The sessions
  table fits narrow terminals. `memorizz harness show` and `harness events`
  accept a unique run-id prefix.

- Interactive CLI pickers: `/agents`, `/harnesses`, `/sessions`, `/models`
  and `/menu` open an arrow-key list (type to filter, Enter to choose, Esc to
  cancel) and act on the choice; `list` after the command, no terminal, or
  `MEMORIZZ_NO_PICKER=1` prints the plain list.
- The memory history warning "preparation failed; proceeding with the write"
  now names the exception.

### Changed

- Update notices: the chat re-checks PyPI every ten minutes instead of once
  a day (a same-day release could go unannounced until the next day), shows
  a found update in the banner and in the status bar under the prompt until
  you upgrade, and `/update` checks right away and prints the upgrade
  command for your installer.

## 0.16.1 — 2026-10-08

### Added

- Interactive CLI: `/sessions [codex|claude-code] [N]` lists the Codex and
  Claude Code sessions the MemoRizz plugins recorded (agent, project memory,
  turns, folder, opening prompt, run id), and `/memory project [path]`
  switches the chat to a folder's project memory, the one the plugins write
  to, so questions about recent coding-agent work retrieve their turns.
- Interactive CLI look and feel: a crest animation on startup (skipped off
  a colour terminal or with `MEMORIZZ_NO_ANIMATION=1`), a status bar under the
  prompt (provider and model, active memory, active harness, hotkeys), a
  coloured prompt, arrow-key shortcuts on an empty line (← agents, →
  harnesses, ↓ quick actions) and a `/menu` of the most used commands.

### Fixed

- Filesystem provider: a store written by two embedding models (a plugin's
  MCP server and the chat, for example) no longer aborts vector search with
  "all input arrays must have the same shape"; the FAISS index is kept per
  embedding size and a query searches the rows embedded at its own size.

## 0.16.0 — 2026-10-08

### Added

- Memory evolution and historical context: Playground → Memory → **Memory
  evolution** shows a scrollable timeline of creations, corrections, deletions
  and explicit source relationships with the actor, originating source,
  changed fields, version hashes and run identifiers of each change, filters
  by memory type, actor or change, and paging that keeps loaded history. It is
  backed by a provider-neutral memory history journal (opt-in for SDK
  providers, on for the UI and MCP host) and by context snapshots captured at
  the model boundary, so the Context tab replays the exact request of a past
  turn (`capture_context_snapshots=True` for standalone SDK agents). Forgetting
  decisions appear on the same timeline. See the Memory evolution guide.
- Memory archive v1, export and import: a portable `.memorizz.json` archive
  (`"format": "memorizz.memory"`, version 1) exports and restores a memory
  graph through the UI (**Memory types → Export & import**), the SDK, the CLI,
  the MCP server and the Codex and Claude Code plugins (`/export-memory`,
  `/import-memory` and the memory-transfer skill). Filesystem, MongoDB, Oracle
  and Notion implement archive storage; agent exports carry delegates, their
  private namespaces, owned shared records and evolution/context records. One
  redaction engine now scrubs credentials across MCP traffic, harness reports,
  archives, the UI and the observability validator. See the Memory archive
  format and Export and import memory guides.
- Benchmark comparisons can rerank with the OpenAI Decisions API through a
  typed client that never executes actions or tools, using the paired question
  recipes shared with Jev.
- Sandbox, internet-access and browser-control providers are built from one
  name-keyed provider registry.
- Forgetting mechanism modelled on Generative Agents: retrieval now scores
  memories by recency since last access, stored importance and relevance (each
  min-max normalised, equal weights by default), reinforces memories when they
  are selected into a prompt, and rates importance at store time (heuristic,
  model-rated or off). A governed retention planner lists primary memories
  whose retention decayed, applies an approved plan as reversible suppression
  (never deletion), and can restore any record. Available from the SDK
  (`plan_retention`, `apply_retention`, `unsuppress_memory`), the CLI
  (`memorizz learning retention-plan|retention-apply|suppressed|unsuppress`),
  the Learning control plane page, a new **Forgetting mechanism** section on
  the Settings page and per-agent overrides in the agent form. See the
  Forgetting mechanism guide.
- Provider contract: `touch_many` records recalls, `purge_expired_semantic_cache`
  removes expired cache rows; Oracle persists importance, last access, recall
  counts and retention state.
- Interactive CLI harness routing: `/harnesses` lists the external harnesses
  and their readiness, `/harness <codex|claude-code|...|auto|off>` runs the
  following turns on that harness with MemoRizz memory, tracing and approvals
  (session only; `off` restores the saved agent), the prompt shows the active
  harness, and approvals raised by a harness turn are printed inline.
  `memorizz chat --harness` and `memorizz run --harness` set it at launch.
  `/harness delegate` keeps the agent's model in charge with its harness
  delegates and the `run_harness_task` tool; `/compare <harness> <harness>
  <task>` runs a read-only comparison from the prompt and saves it as a
  workflow.

### Changed

- The interactive CLI now uses `/models` to show the current model, list
  installed Ollama models or switch models. Help, Tab completion and model
  suggestions use the plural command; `/model` remains a compatibility alias.

### Fixed

- Reapplying an empty MCP configuration at CLI startup no longer prints
  "Tool not found for removal" warnings for helpers that were never registered.
  Repeated cleanup also tolerates MCP tools removed individually.
- Slash-command help now displays optional arguments such as `[name]`
  literally instead of interpreting them as console formatting.

- Security: the self-aware command tool could run arbitrary host commands
  through `find -exec`, git configuration flags or a script whose basename was
  allow-listed; it now resolves binaries from PATH, denies those flags and
  requires approval. Cross-origin form posts were accepted on every
  non-trace UI route; all state-changing routes now reject them (bearer-token
  clients and the signature-verified Twilio webhook excepted). Connected MCP
  tools without annotations were treated as read-only unless their name
  contained a known verb; they are now mutations unless a read verb, the
  server annotation or the host allowlist says otherwise. Internet-provider
  API keys are no longer written into agent records.
- Data integrity: filesystem record ids can no longer escape the memory root;
  embedding-only updates keep the original timestamp (conversation order was
  being scrambled by the backfill); several processes on one filesystem root
  no longer overwrite each other's index and read paths pick up other
  processes' writes without a restart; MCP `update_memory` no longer persists
  the redacted view; deleting a knowledge base from one agent no longer deletes
  it for every agent; Oracle semantic-cache clear/invalidate were silent
  no-ops and summaries/entities could not be deleted by their exposed ids;
  MongoDB cascade delete now removes the agent and every memory type and by-id
  reads accept string ids.
- Agent loop: provider errors are no longer stored as assistant replies or
  admitted to the semantic cache; an approval pause inside a multi-tool turn no
  longer drops sibling calls; the embedding backfill is drained on close;
  internet tools recover after a cool-down instead of locking out; compaction
  counts in-session rows; tool-log reads are scoped to the conversation;
  context-local state is created eagerly to avoid a first-use race.
- UI and automations: blocking model pulls, Docker and dataset downloads no
  longer stall the event loop; the Twilio webhook works with UI auth enabled;
  "Run now" keeps paused automations paused and timed-out runs are neither
  retried while still running nor delivered twice; uploads are capped at 50 MiB.
- Semantic cache: expired rows are purged, re-caching upserts by cache key,
  preload loads vectors on MongoDB and Oracle, hit counts increment.
- Entity memory: model-supplied identity keys can no longer poison a tenant
  scope; name-only upserts converge on one record.
- The capability report derives its MCP tool count from the registry.
- WhatsApp: one E.164 normaliser is shared by every address handler.

### Performance

- Learning control-plane reads no longer scan the shared-memory partition on
  per-turn paths and use indexed observability queries where providers offer
  them; fleet pages read one row per conversation and no longer instantiate a
  full agent per row; Oracle conversation history, tool logs and start-up DDL
  use SQL limits and a single dictionary read; agent saves skip re-embedding
  unchanged tools; comparison runs throttle result rewrites.

## 0.15.0 — 2026-10-04

### Added

- Configurable LLM judging for agent harness answers: evaluate saved outputs or
  opt in when launching a run or comparison. The default is local Ollama
  `qwen2.5:3b`; choose another model or OpenAI/Anthropic, edit the evaluation
  prompt, supply reference evidence and set an accuracy target. Judgments keep
  their own cost and timing, survive session imports and become stale when the
  answer changes. Local evaluation never falls back to a paid provider.
- Comparisons show judged accuracy, **Most accurate (judge)** badges and the
  cheapest/fastest answers meeting the target. Badges require comparable current
  judgments; scores estimate correctness from the supplied evidence.

### Fixed

- Codex session cost estimates include current GPT-6 rate cards, cached writes,
  service tiers and long-context pricing, calculated per unique request rather
  than from cumulative session tokens.
- npm-installed harnesses use their installation's Node runtime for availability
  probes and execution. Failed probes show a diagnostic instead of a stack trace
  path as the version.
- Starting a comparison reveals its approval queue or workflow, including the
  first comparison after a reload. Approver identity explains that it records
  the operator's name or email and remembers it in the browser.
- Official GPT-6 models use Responses by default for reasoning with function
  tools. Native MemAgent provider errors mark harness runs as failed and are
  never cached as successful answers.

### Changed

- Consolidated OpenAI Responses request options and harness browser fixtures,
  removed empty type-checking blocks, and added judge and approval navigation
  acceptance checks to the release browser gate.

## 0.14.1 — 2026-10-02

### Changed

- Observability shows Codex and Claude Code plugin sessions as **Codex
  sessions** and **Claude Code sessions** instead of "Unregistered runtime
  traces". Each thread links to the session's run on Agent harnesses
  (`/harnesses?run=<id>` now opens a run). MemAgent-specific findings, such
  as "Trace ownership is not durable" and the advice to save the agent, and
  the usage, lineage and replay panels, no longer appear for them.
- The Dashboard has a **Coding-agent sessions** panel, and Usage & cost a
  matching section by agent and model: sessions, turns, tokens and spend from
  the agents' own logs, reported apart from MemoRizz's agent runs.

## 0.14.0 — 2026-10-02

### Added

- Coding-agent sessions on the Harnesses page: the Codex and Claude Code
  plugins record each session as a run, read from the agent's own session log
  after every turn. The run has its prompts, commands, tool calls (MemoRizz's
  included), file changes, the memory the plugin loaded, tokens and cost, so
  the UI shows its trajectory and Compare puts a Codex and a Claude Code
  session side by side. `memorizz plugin import-session` adds past sessions;
  `MEMORIZZ_PLUGIN_RUNS=false` turns it off. Local stores only.
- `examples/coding_agent_plugins`: Codex fixes a bug and saves why; Claude
  Code, in a new session, explains it from MemoRizz memory. Both sessions
  appear as runs (`--ui` adds them to your usual UI).
- `HarnessRunStore.replace(run, events)` writes a run and all its events in
  one transaction.

## 0.13.0 — 2026-10-02

### Added

- Tool call cache: a MemAgent with `tool_cache=True` (or `.with_tool_cache()`,
  `memorizz agents create --tool-cache`, or **Enable tool call cache** in the
  UI) reuses the result of a repeated tool call while it is fresh.
  - **What it caches:** Python tools that opt in with
    `@governed_tool(cacheable=True, cache_ttl_seconds=...)` (only
    deterministic tools without side effects or approval can), and MCP tools
    their server marks read-only and idempotent. Failures are never stored.
  - **Keys and freshness:** the tool, its arguments, a fingerprint of the tool
    (code and schema, or MCP server), the user and, by default, the agent;
    300 s by default, capped by domain freshness (MCP 300 s).
  - **Visibility:** hits appear on tool trace events, in
    `agent.tool_cache_stats()`, in UI trajectories ("from cache") and in the
    harness Compare view, which also names each MemAgent lane by agent.
  - **Measured:** on eight support tickets with local `qwen2.5:7b`, real API
    calls fell from 8 to 4 and the queue from 36.6 s to 28.6 s, with 8 of 8
    correct answers either way (`examples/tool_cache/benchmark.py`).

- MemoRizz plugins for Codex and Claude Code (`plugins/codex/memorizz`,
  `plugins/claude-code/memorizz`):
  - **What they add:** MemoRizz's MCP server (memory writes allowed) and
    skills for recalling, saving, correcting and curating memory and for
    using saved MemoRizz agents.
  - **Episodic memory:**
    - SessionStart gives the agent the project's memory ID, the last
      session's summary and its newest facts.
    - UserPromptSubmit keeps each prompt and, with `MEMORIZZ_PROMPT_RECALL`,
      adds related memories.
    - Stop saves each turn (request and final answer, secrets removed) to
      conversation memory. The save runs detached, so `claude -p` and
      `codex exec` exiting don't cancel it.
    - PreCompact and SessionEnd summarize the session with MemoRizz's default
      model, after any turn saves still running.
    - Turns go in thread `codex-<session>` / `claude-code-<session>`.
    - Off with `MEMORIZZ_SESSION_CAPTURE=off`; summaries alone with
      `MEMORIZZ_SESSION_SUMMARY=false`.
  - **Claude Code extras:** `/memorizz:remember`, `/memorizz:recall`,
    `/memorizz:forget` and `/memorizz:memory-status`, and a
    `memory-curator` subagent (also a skill in both plugins).
  - **Shared memory:** both agents use the same memory ID for a project
    (`project-<folder>-<hash>`), so what one learns the other recalls.
  - **Hosted server:** `memorizz plugin install <agent> --remote URL
    [--token-env NAME]` points the plugin's tools and hooks at a hosted
    MemoRizz MCP server, with a per-person API key; `--local` switches back.
    `deploy/mcp-server/` has a Dockerfile, a Compose file with Caddy for
    automatic HTTPS, and a guide.
  - **Repository marketplaces:** the repository is a marketplace for both
    agents (`.agents/plugins/marketplace.json`,
    `.claude-plugin/marketplace.json`).
  - **Install commands:** `memorizz plugin install|uninstall codex|claude-code`
    with `--user`, `--no-capture`, `--no-summaries`, `--prompt-recall`,
    `--allow-agents`, `--allow-harness --harness-root`, `--allow-traces`,
    `--remote`/`--local` (saved as settings `memorizz config` also sets);
    `memorizz plugin memory-id`; and `memorizz plugin hook
    session-start|prompt|stop|summarize`.
  - **Packaging:** the wheel ships both plugins.
- `memorizz.episodic_capture`: `record_turn()` and `summarize_thread()`, used
  by the plugin hooks and the MCP server.
- MCP server: `memorizz_ingest` chunks local files and folders into the
  knowledge base.
  - **Where it reads:** only inside `MEMORIZZ_MCP_SERVER_INGEST_ROOTS` /
    `--ingest-root` (default: home for stdio, none for HTTP).
  - **What it skips:** hidden, dependency and VCS folders, and files that look
    like secrets.
  - **Cap:** 500 files / 20 MB per call.
  - **Ingesting again:** unchanged files are skipped, and a changed file's old
    chunks are superseded.
- MCP server: `memorizz_lookup_entities` and `memorizz_upsert_entity` for entity memory.
- MCP server: `memorizz_update_memory` replaces a memory without deleting it.
  - **Records:** the new record gets `supersedes` and `supersede_reason`; the
    old one gets `status="superseded"`, `superseded_by` and `superseded_at`.
  - **Duplicates:** `duplicate_of` retires a duplicate in favour of an
    existing memory.
  - **Defaults:** list and search hide superseded records unless
    `include_superseded=true`.
- MCP server: `memorizz_memory_status` reports backend, policy, embedding
  readiness (one short embed call) and per-type record counts.
- MCP server: `memorizz_record_turn` and `memorizz_summarize_session` save a
  client's turns and summarize a session (with the server's model) for the
  calling principal.

- Harness delegates, finished: **Cancel** now stops a memagent run between its
  model and tool calls and cancels every delegate harness run it started (one
  still being prepared starts already canceled; `MetaHarness.cancel(...,
  before_start=True)`). Delegates of a run approved for edits may edit too,
  taking turns in the workspace under the run's lease, unless their own
  harness permissions are narrower; the service waives a delegate's approval
  only when its parent run is still running and was approved for that access
  in the same folder (`SQLiteHarnessRunStore.workspace_holder()`). A
  coordinator's ledger row, details and workflow node show its own model
  calls (planning and combining included, priced at list rates;
  `MemAgent.count_model_call()`) plus its delegates' runs.
- Harness delegates in the playground: a coordinator whose delegates run on
  harnesses shows a **Harness delegates** card in the inspector and a chip by
  the message box. Each conversation's delegates get their own fresh folder
  instead of the folder the UI runs in. **Allow** gives them a folder of yours,
  web access or edits. Web access and edits need an approver's name, and the
  grant is recorded as an approved `metaharness.delegate_access` approval
  that lasts eight hours, so their harness runs start without asking again.
  An expired grant is reported in the reply.
  `MetaHarness.grant_delegate_access()`, `delegate_access()`;
  `POST /api/agents/{id}/harness-access`,
  `GET /api/agents/{id}/harness-access/{grant_id}`,
  `POST /api/agents/{id}/harness-workspace`; the playground stream takes
  `harness_grant` and `harness_workspace`.
- Deleting removes the scratch folder MemoRizz made for a blank workspace once
  no remaining run, workflow or delegate grant uses it (`keep_scratch` /
  `remove_scratch=False` keeps it). A conversation page has **Delete
  conversation** (`MetaHarness.delete_conversation()`,
  `DELETE /api/harness-conversations/{id}`).
- CLI parity with the Agent Harnesses page:
  - `memorizz harness`: `models`, `continue`, `conversation`,
    `delete-conversation`, `delete`, `delete-workflow`, `rerun-workflow`,
    `delegate options|create`, `events --follow/--limit`, and
    `--allow-subagents`.
  - `compare --harness-model HARNESS=MODEL`, `plan --stage-model` and
    `--stage-agent`, plus the user, thread, tool, cost, token and
    execution-backend options on `plan` and `compare`.
  - `--harness memagent` without `--agent-id` runs the agent last used with a
    harness, as the UI does. Unknown IDs end with a message and exit code 1
    instead of a traceback. Every command has help text.
  - `memorizz agents`: `update` and `delete`; `create --harness-model`,
    `--delegate`, `--delegation/--no-delegation`, `--delegation-max-workers`,
    `--delegation-consolidation` and `--root-fallback`. `show` includes the
    harness model, delegates and delegation settings.
  - `memorizz eval terminal-bench tasks|status|run` (Terminal-Bench 4.0 from
    the CLI), `memorizz eval compare CONFIG.json` (Evalground comparisons), and
    `eval run --agent-id` (evaluate a saved agent) and `--ollama-host`.
- MCP server: `memorizz_rerun_harness_workflow`,
  `memorizz_delete_harness_runs`, `memorizz_delete_harness_workflow`,
  `memorizz_get_harness_conversation`,
  `memorizz_continue_harness_conversation`; `allow_subagents` on runs, plans
  and comparisons, and `harness_models` on comparisons. Deleting needs the
  write scope and touches only the caller's runs.
- Shared homes for what the UI, CLI and MCP server all need:
  `memorizz.metaharness.catalog` (model choices, saved agents and their
  harness delegates, the default memagent agent, conversation setup, creating
  a harness delegate) and `memorizz.memagent.delegation_settings` (validated
  delegate choices, refusing loops).
- Rate cards for Claude Sonnet 5.5 and Claude Mythos 5 / 5.1.

- Delete harness runs and workflows from the Agent Harnesses page: **Delete**
  in a run's details, **Delete** on a finished workflow card (with all its
  runs), and **Delete selected** for the runs ticked in the ledger, whose
  header box ticks every run the filter shows. A run goes with its events and
  the runs its MemAgent delegates made. Runs still working, and workflow steps
  (delete the workflow instead), are kept, and the page says why. Files and
  saved memory answers stay. `MetaHarness.delete_run()`, `delete_runs()`,
  `delete_orchestration()`; `SQLiteHarnessRunStore.delete()` and
  `delete_orchestration()`; `DELETE /api/harness-runs/{id}`,
  `POST /api/harness-runs/delete`, `DELETE /api/harness-orchestrations/{id}`.
  Compare still takes two to four ticked runs.
- Harness workflows run again from the page: **Run again** repeats a finished
  plan or comparison with its settings (a fresh scratch folder if it had one,
  and a saved agent filled in for memagent), and **Edit and run** loads them
  into the launch form first. `MetaHarness.rerun_orchestration()`,
  `MetaHarness.is_scratch_workspace()`,
  `POST /api/harness-orchestrations/{id}/rerun`.
- Live trace in each workflow node: its reasoning, model calls, tool calls and
  subagents as they happen, with a subagent's steps nested under the call that
  started it. New `HarnessEventType.REASONING`. Codex reports reasoning items
  and `collab_tool_call` subagents; Claude Code reports thinking blocks (marked
  hidden when it withholds the text) and which subagent an event came from;
  a MemAgent run reports model calls, tool calls, reasoning and delegate tasks
  (`MemAgent.set_delegation_event_callback()`).
- Each workflow node, ledger row and run's details name the model the harness
  ran on, from its events while it runs (`SQLiteHarnessRunStore.event_models()`).
- Workflow cards fold: the chevron or header folds a card to one line with a
  status dot per harness, and **Collapse all** folds them all. Choices are
  remembered per workflow.
- **Subagents** on the harness launch form (`HarnessPermissions.allow_subagents`,
  `harness_task(allow_subagents=True)`): Claude Code gets its `Task` tool
  alongside its usual tools, and Codex may start sub-agents. It is not a tool
  list, so it works in comparisons too. Codex sub-agents are followed from
  their session files under `$CODEX_HOME/sessions` while the run goes, so each
  appears in the node's trace with its own searches, reasoning and answer.
- Compare runs side by side: **Compare traces** on a plan or comparison, or
  tick two to four runs in the ledger and press **Compare selected**. A table
  sets out status, model, time, cost, tokens, actions, reasoning and
  verification (highlighting the fastest, cheapest and leanest run that
  succeeded, never judging answers); a timeline puts each run in a lane from
  its own start, with model calls, tool calls, commands and subagents as bars;
  and each run's answer and trajectory sit in columns. It updates while runs
  are active. `window.MemorizzCompare.open(runIds)`.
- Comparisons take a model per harness (`start_compare(..., models=...)`,
  `harness_models` on `POST /api/harness-orchestrations`); the launch form
  shows a picker per compared harness with that harness's own models and
  default. **Run again** and **Edit and run** keep them.
- pi works with no setup beyond an API key: with no provider or model
  configured, MemoRizz runs it on the first provider you have a key for
  (`anthropic/claude-sonnet-5-5`, then `openai/gpt-5.5`), and the form suggests
  pi's own model list.
- Model pickers on the harness page are dropdowns: each harness lists up to
  ten of the newest models from each provider it runs (read from the
  providers' own model lists, newest first, cached for six hours; local
  Ollama models too), its default, and **Other model…** to type any name. It
  applies to the quick bar, single runs, plan stages and comparisons.
  `memorizz.llms.model_lists`.
- memagent can run on another model from the harness page. The saved agent
  loads with that model for the run only (`provider/model` switches provider
  too); a save during the run keeps the agent's own model.
- OpenHands and Hermes work as harnesses. OpenHands reads the CLI 1.x event
  format, takes the run's model through `LLM_MODEL`, runs only with Network
  Full behind its isolation wrapper (which mounts the workspace read-only
  unless edits were approved, via `MEMORIZZ_WORKSPACE_WRITABLE`), and the form
  sets the Docker wrapper backend when OpenHands is chosen. Hermes run homes
  live under `~/.hermes/profiles`, which Hermes 0.21.5+ needs to keep its
  install intact.
- Harnesses as delegates: a MemAgent's delegates can be agents that run their
  turns on a harness (meta-harness runtime mode), so a coordinator splits a
  request and Codex, Claude Code, pi, Hermes or OpenHands work on the parts in
  parallel. In a MemAgent harness run each such delegate works in the run's
  workspace and may use what the run was approved for (network, MemoRizz MCP,
  subagents, edits) without another approval; its harness run is tagged
  **Delegate** in the ledger and linked from the subagent step
  (**Open its codex run**). `MemAgent.run_on_harness(parent_run=...,
  on_start=...)`; delegation events include `task_progress` with
  `harness_run_id`. The playground lists each delegate as it starts and
  finishes (**Delegate · Codex reviewer (codex)**). On the agent form,
  **Add a harness as a delegate** creates a harness-backed delegate with its
  model (`POST /api/harness-delegates`), harness-backed delegates are marked
  "runs on …", and the harness section has a **Harness model** picker.
  On the harness launch form, choosing such a coordinator for a memagent run
  lists the delegates it hands work to and the harness each runs on.
  The coordinator's final answer can say which delegate found what: results
  reach it labelled with each delegate's name and harness, not only its ID.
  Two worked examples, a code review crew and a research desk, are in
  `examples/metaharness/multi_harness_delegates/`.
- A **Delegates** section on the agent form: pick other saved agents as
  delegates, turn delegation on or off, and choose how results are combined
  and how many delegates work at once. A memagent harness run of such an agent
  shows each delegate as a subagent, by name. Choices that would loop are
  refused, and loading a saved loop skips the repeated agent.

- Terminal-Bench 4.0 in Evalground: **Test a harness** runs Codex or Claude
  Code (installed in each task container by Harbor), a MemAgent (driving the
  container from the host) or the free reference solutions on chosen tasks,
  with an optional MemoRizz memory in front of each task and a lesson saved
  per finished trial (never shown to the same task). Results show pass rate,
  categories, per-trial time, tokens and cost, and join the run library.
  New: `MemorizzCodexAgent`, `MemorizzClaudeCodeAgent`,
  `python -m memorizz.benchmarks.terminal_bench_runner`, and
  `GET /evalground/terminal-bench/status`,
  `POST /evalground/terminal-bench/runs`. The `terminal-bench` extra now
  needs Harbor 0.23. `HarnessContextBuilder.build()` takes `exclude`.
- Harness conversations: **Continue** on any run opens `/harnesses/chat`, where
  each message is a new run with the run's setup (harness, model, folder,
  permissions, agent, memory, limits) and earlier turns as prompt context, on
  every harness. SDK: `MetaHarness.continue_conversation()`,
  `MetaHarness.conversation()`; API: `GET/POST /api/harness-conversations/…`.
- Harness trajectories and execution graph: a selected run shows its steps
  live, a **Steps** column counts actions per run, and plans and comparisons
  are drawn as connected nodes. `SQLiteHarnessRunStore.event_counts()`.
- Model suggestions per harness on the quick bar and launch form.
- Playground conversations can be renamed and deleted from the left pane
  (`POST`/`DELETE /api/agents/{id}/threads/{memory_id}`). Names are saved on
  the agent as `thread_titles`; deleting removes the conversation's messages,
  summaries and tool logs for that agent only.
- Playground **Settings → Context window** for Ollama agents, listing sizes up
  to the model's own maximum (`GET /api/ollama/context-length`). Choosing one
  replaces any agent-level cap, so it is the window the agent runs with. The
  context panel names the model and its longest window, and reply footnotes
  name the model that answered (`run.started` capabilities include `model`).
- Auto-compaction that actually runs: before a request would pass
  `context_policy.compact_at` percent of the window (default 80, 30–80, 0 for
  off), older messages are summarized into summary memory, the newest
  `keep_recent_messages` stay word for word, and the summary text goes into
  the prompt. The playground's **Auto-compact at** selector changes the
  threshold per agent, a note in the chat reports each compaction, and
  `context.compacted` stream events carry the counts.
- The playground offers to enable what an agent can't do yet. When a request
  needs web search, email, calendar, notes, code execution or a browser that
  the agent lacks, a card under the reply lists the ways to enable it, built
  from the agent's live configuration: Tavily or Firecrawl (with an inline key
  field when needed), a Gmail, Calendar or Notion connection, sign-in for a
  connected server, or a registry search. The agent lists missing capabilities
  in its instructions and calls the new `request_capability` tool
  (`capability.requested` event); a keyword check also emits
  `capability.suggested` so small models still surface the card. Sign-in
  started from the card returns to the playground.
- Harnesses reach an agent's own MCP servers through MemoRizz:
  `memorizz_list_connected_tools`, `memorizz_read_connected_tool`,
  `memorizz_call_connected_tool` and `memorizz_resume_connected_tool_call`.
  Credentials stay in MemoRizz, calls are audited under the agent, reads are
  marked read-only so hosts that never prompt (Codex runs) allow them, and
  changes need write access and a person's approval.
- Each MCP tool is exposed to the agent as its own tool (`server__tool`) with
  the server's schema, from a per-agent cache of the last tool listing. The
  tool router lets clear keyword matches among them share the per-turn slots.
  `mcp_list_tools` registers a server's tools at once.
- MCP connections shows **Add OAuth client** and setup steps for Gmail and
  Google Calendar before sign-in, and opens the client fields when a provider
  needs one.
- Find and attach MCP servers from the UI: **MCP Connections → Find a server**
  and the playground's settings list the official connectors and search the
  official MCP Registry, then attach a hosted or local (npm, PyPI, Docker)
  server after asking only for the values it needs. Hosted servers are checked
  for OAuth, token or no sign-in. `memorizz mcp search` does the same from the
  CLI, and `memorizz.mcp.catalog` from code.
- Gmail preset (`--preset gmail`, UI and catalog): Google's Gmail MCP server
  with `gmail.readonly` and `gmail.compose` scopes only.
- Attach marketplace skills to an agent from the Skills page or the
  playground. The `SKILL.md` is saved under `~/.memorizz/skills` with its source
  and hash, and added to the agent's `skill_paths`.
- Personal productivity assistant template (`memorizz.memagent.templates`,
  **Create agent → Start from a template**): instructions, persona and entity
  memory, with Gmail, Google Calendar and Notion attached and changes gated by
  approval.

- Orchestrate harnesses from the UI, the JSON API or the SDK:
  `MetaHarness.start_plan()` runs stages in order, each on its own harness
  (for example plan, implement, review), and `start_compare()` runs one
  read-only task on several harnesses at once. Workflows are durable,
  cancellable and wait for host approval of their edit stage. The harnesses
  page gains a launch mode switch, a stage editor with presets, a Workflows
  panel, workflow tags in the run ledger and live refresh.
- Stage handoffs: each finished stage is reduced to its answer, changed files,
  commands and verification result. The next stage receives all earlier
  handoffs fitted to its context budget, instead of raw text truncated at
  20,000 characters. With a `memory_id`, handoffs are saved to conversation
  memory, where later runs retrieve them and the Memory explorer lists them.
- `deepseek` harness: Claude Code's agent loop on DeepSeek models
  (`deepseek-flash`, `deepseek-v4-pro`) through DeepSeek's Anthropic-compatible
  API. Only `DEEPSEEK_API_KEY` reaches the process, and cost is estimated from
  DeepSeek's peak and off-peak prices.
- Quick launch on the harnesses page: a task, an optional harness and project
  folder, Web access and Allow edits, and Run, with approval asked inline.
  Tasks without a project folder run in a fresh scratch folder
  (`MetaHarness.scratch_workspace()`).
- Harness parity across surfaces. CLI: `memorizz harness plan`, `compare`,
  `workflows`, `show-workflow` and `cancel-workflow`, and `run --scratch`,
  `--allow-tool`, `--deny-tool` and `--output-schema`. MCP: tools to start
  plans and comparisons, list, read and cancel them, and retry runs; a start
  without a workspace uses a scratch folder. UI: Retry, and thread, delegated
  environment, tool and output-schema fields. SDK: `MetaHarness.retry_start()`.
- A standalone run with a `memory_id` saves its answer to conversation memory,
  like plan and comparison steps.
- `hermes` harness for Nous Research's Hermes Agent (0.21.4 or newer), using
  `hermes chat --format stream-json` with a generated per-run home. Only the
  `file` toolset (plus `web` with web access, and the MemoRizz MCP server) is
  enabled; read-only runs cannot write, and `hermes -z` is never used.
- `pi` harness for the pi coding agent (`@earendil-works/pi-coding-agent`),
  with any provider it supports. MemoRizz enables only its file tools, never
  `bash`; edit runs require an attested isolation wrapper.

### Deprecated

- `MemAgent.run_stream()` called directly. Use `run_stream_events()`; the
  string API will be removed in 0.14.

### Removed

- Redundant code, about 900 lines net:
  - **Dead code:** shadowed and unused CSS, an unused LongMemEval loader, the dead
    `MemAgent._compress_memories_with_llm` and `_tool_result_failed`,
    `Toolbox._load_import_reference`, unused CLI REPL labels, write-only attributes,
    `estimate_terra_cost`, the unused Oracle `requirements.txt` and the unused
    `supports_output_schema` adapter flag (probes report schema support).
  - **Now shared, UI:** the automation create/edit forms, playground
    thread-memory loading, field reading, request-body and approval parsing,
    and the JS money and HTML-escape helpers.
  - **Now shared, models and providers:** skills-marketplace config,
    local-model prompt building (`llms/local_chat.py`), provider response
    metadata (`ResponseMetadataMixin`), MongoDB/Oracle embedding fallbacks and
    skill search, and toolbox syncing (`MemoryProvider`).
  - **Now shared, utilities:** benchmark JSON and output bounding
    (`benchmarks/measurement.py`), stored-JSON reading (`memorizz/_json.py`),
    ISO/UTC timestamp parsing (`memorizz/_time.py`), environment flags
    (`env_bool`, `env_text`), and the optional-provider slot of the sandbox and
    internet managers.
  - **Moved out of the UI routers** for the CLI and MCP server to share:
    `memorizz.metaharness.catalog`, `memorizz.memagent.delegation_settings`,
    `memorizz.benchmarks.agent_template.secret_free_agent_template`, and
    `terminal_bench_runner.environment_status`.
  - **Behaviour changes:** MCP approve/reject answer 400 to malformed JSON,
    and the HuggingFace prompt fallback accepts a message without a role.
- `SETUP.md`: its local and hosted setup, `MEMORIZZ_ORACLE_CONTAINER` and the
  production checklist are now in the Oracle provider guide, which also lists
  migration 008.
Unused code was pruned: nothing in the package, its tests, docs, examples,
eval scripts or the OpenSpeech backend called it. Public API that went:

- `create_assistant`, `create_chatbot` and `create_task_agent` (the last two
  set application modes that do not exist); use `MemAgentBuilder`.
- `SummaryComponent` and `StandaloneSemanticCache`.
- `MemAgent.has_meta_harness()`, `MetaHarness.arun()` / `astart()`,
  `MemAgentBuilder.with_auto_registration()`.
- `Toolbox.bind_callable()`, `delete_tool_by_id()`, `delete_tool_by_name()`,
  `update_tool_by_id()`; `EntityMemory.search_entities()`;
  `EntityMemoryManager.lookup_entities()` / `build_context()` (use the
  `*_with_diagnostics` methods); `PersonaManager.load_persona()` /
  `export_persona()`; `MemoryManager.update_memory_ids()` /
  `get_unsummarized_messages()`; `MongoDBProvider.get_summaries_by_memory_id()`
  / `get_summaries_by_time_range()`.
- UI endpoints nothing called: `POST /traces/feedback`, `POST /traces/outcomes`,
  `GET /api/capabilities`, `GET /api/agents/{id}/capabilities` and
  `POST /agents/{id}/delete` (use `memorizz agents delete`).
- About 22 KB of CSS rules that matched no page, an unused template, dead
  page scripts, the shadowed `memorizz/memagent.py`, the stub `memorizz.ui.api`
  package, the unused desktop app shell, stale example notebooks, one-off
  scripts, and duplicate root `install_oracle.sh`, `deploy.sh` and
  `setup_dev.sh`.

### Changed

- Internal cleanup, no behaviour change: the Tavily and Firecrawl providers
  share their setting and truncation helpers through `InternetAccessProvider`;
  entity attributes and relations use one validation helper; local model
  providers are listed once (`memorizz.llms.llm_factory.LOCAL_LLM_PROVIDERS`,
  used by the harness cost checks and the benchmarks); the semantic cache and
  the tool cache share their domain freshness defaults; the playground uses the
  UI's shared `escapeHtml`.

- `memorizz_search_memories` returns `search_mode` (`semantic`, `keyword` or
  `hybrid`) and falls back to keyword matching, with a setup note, when no
  embedding model works.
- `memorizz_store_memory` now embeds what it stores when an embedding model is
  available, so memories saved over MCP are found by semantic search.
- `KnowledgeBase.ingest_knowledge`/`ingest_file` accept `metadata=` and
  `embeddings="required"|"optional"|"off"`; entity upserts no longer fail when
  embedding fails.

- Harness confirmations and error notices are in the page
  (`static/js/page-dialogs.js`), not the browser's blocking `confirm()` and
  `alert()` dialogs.
- Codex sub-agent tracking tolerates other session-file layouts and record
  casing, and when Codex reported starting sub-agents whose session files it
  couldn't read, the trace says so instead of showing them as if they did
  nothing.
- `memagent` harness runs report cancellation as `cooperative` and their cost
  (`cost_reporting`).
- Network **Full** now gives Codex live web search (and network access for
  commands in editing runs), and a memagent run gets web access only then: its
  agent's own internet provider, or MemoRizz's Tavily/Firecrawl key for the
  run. With Network off, a memagent's web tools are hidden for the run.
- The harness launch form greys out harnesses that can't honour the chosen
  Network or tool lists, says why, and explains what each setting gives each
  harness. The stage editor's fields use the app's field style.
- Codex runs keep sub-agents off unless **Subagents** is on
  (`features.multi_agent`). With it on, the run is not `--ephemeral`, because
  Codex sub-agents need a saved parent session; the session lands in the
  usual `~/.codex/sessions`.
- The launch form warns when **Model override** names a model that only
  another harness lists (for example a Codex model on Claude Code).
- The harness launch form shows only the options that apply to the chosen
  harness(es): the saved agent only for memagent, tool lists only for
  harnesses that take them, cost and token limits only where they are
  reported, the isolation backend only when a harness needs a wrapper, and so
  on. Hidden options are left out of the launch.
- Each run's model is the one its harness reported running; memagent shows its
  agent's model instead of a requested one it ignores.

- Ollama's default context window is the model's full length when its
  attention cache, estimated from the daemon's model metadata, fits in a
  quarter of this machine's RAM; otherwise the largest standard size that
  fits, at least 16,384 tokens (8,192 when the length is unknown; 16,384 for a
  daemon on another machine). On a 16 GB Mac, gemma4 gets 131,072 and
  qwen2.5:7b 32,768, so switching models changes the window. Only windows you set are saved with an agent or
  its model settings, and a window in `llm_config` takes precedence over a cap
  saved by earlier releases. `context_window_exceeded` now logs the tokens
  needed and the budget.
- Skill search uses the skills.sh directory, ranked by installs, and needs no
  `GITHUB_TOKEN`; GitHub code search is the fallback. The page is now called
  **Skills**.
- The playground no longer writes MCP servers: its settings list the agent's
  servers and attach new ones through MCP Connections, which keeps credentials
  encrypted. The unused inline server editor is gone.
- Duplicated code now lives in one place: agents are rebuilt from storage by
  `MemAgentModel.from_document()`, the LLM classes share tool-metadata helpers
  (`memorizz.llms.tool_metadata`), the worker and the UI's Run now share one
  automation job runner, trace metadata fields are defined once
  (`TRACE_METADATA_FIELDS`), and UI pages share `escapeHtml` and their icons.
- The LongMemEval script and the accuracy probes require an Oracle password
  instead of defaulting to a published one.

### Fixed

- A cost or token limit made every saved-MemAgent harness run (alone, in a
  plan or in a comparison) fail as "not ready:
  cost_budget_telemetry_unsupported", with the wrong fix ("Create and save a
  MemAgent"). The harness reports whole-run tokens, and cost whenever the model
  can be priced, so limits now apply:
  - a local model (Ollama, MLX, Hugging Face) has no API charges, so a cost
    limit always holds;
  - a cloud model MemoRizz has no price for is refused when the run starts
    (`cost_budget_unpriced_model`), with what to change.
- "Not ready" errors say why in words, with the code in brackets, and give the
  fix for that reason. A harness's own setup advice appears only when setup is
  what's missing.
- Agent harnesses: searching runs by a MemAgent's name found nothing. Runs are
  now searchable by it, the Harness column shows it, and the run details and
  Compare view show it only on MemAgent runs (a comparison copies `agent_id`
  onto every lane).

- Ollama embeddings ignored `OLLAMA_HOST` and always used
  `http://localhost:11434`; they now read it, like the Ollama LLM provider
  (an explicit `base_url` still wins).

- The playground rendered model output as raw HTML, so a reply could run
  script in the page. The playground and harness chat now share one safe
  Markdown renderer: raw HTML shows as text, only web, mail and in-app links
  stay links, and remote images become links instead of loading.
- The MemoRizz MCP server's `memorizz_list_agents` failed with an internal
  error when an agent had a persona; `memorizz_list_connected_tools` with no
  agent attached now returns an empty list and says how to pick one; a harness
  run's MCP server opens the memory store the UI is connected to.
- Workflow graph arrowheads take their edge's colour instead of white.
- A comparison's single model override went to every harness, so a Codex
  model failed on Claude Code; models are now chosen per harness.
- Claude Code runs record whole-run token usage across the models it used
  (`usage.models`), not only its last model call, so token budgets and
  comparisons see the real totals.
- A MemAgent harness run recorded each long tool result dozens of times (once
  per streamed piece); it records it once, whole.
- A subagent's trace step lasts until its last nested step, not until its
  launching call returned (Claude Code's showed about 15 ms).
- Token counts such as `thinking_tokens` and `cache_write_tokens` were
  redacted as if they were secrets.
- The Oracle provider failed with ORA-00600 [unable to load XDB library] when
  starting on an existing schema on the Oracle AI Database Free lite image: it
  read VECTOR dimensions through `DBMS_METADATA`, which needs XDB. It now reads
  them from the driver's column metadata, then `VECTOR_INFO`, and uses
  `DBMS_METADATA` only as a last resort.
- A Hermes run from the UI could break the Hermes install: its per-run home
  outside `~/.hermes` made Hermes re-point its launcher there before the home
  was deleted.

- The harnesses page reloaded itself every few seconds while work ran, which
  flickered and reset scrolling. It now updates in place; running items show a
  spinner and a live elapsed time.
- Codex runs had no cost and showed "Adapter default" as the model: Codex was
  run on an unnamed default model. MemoRizz now passes the first model in
  Codex's local catalog and estimates cost at OpenAI list rates (≈).
- Comparisons and plans including `memagent` failed with `agent_id_required`
  when no agent was chosen. The agent last used with a harness (else the newest)
  is filled in, and with none saved the request is refused before it starts.
- Prompt caching held less than the layout promised. Measured over six turns
  it rose from 73% to 93% of input tokens read from cache on Claude Sonnet 5
  (writes 24.9k → 9.9k tokens) and from 71% to 84% on `gpt-4.1-mini`:
  - the streamed answer call dropped the tool list, which starts the prompt,
    so it could not reuse the turn's cache and re-wrote the whole history
    every turn. It now sends the same tools; tool calls made while answering
    are never run;
  - on Anthropic, notes added mid-turn (host notes and completion-policy
    retries) were merged into the system prompt, changing it and
    invalidating the cached conversation. They are now appended to the user
    turn they follow; reviewed skills placed before the turn keep their
    system block;
  - each newly relevant tool changed the tool list and re-wrote the whole
    prompt. Agents whose tool schemas fit in
    `ContextPolicy.stable_tool_list_tokens` (default 6,000) now send every
    tool every turn.

  These apply to every SDK entry point (`run()`, `run_stream_events()`,
  `arun_stream_events()`, the deprecated `run_stream()`) and the CLI and UI;
  `agent.run()` measured 88% (Claude) and 85% (OpenAI) over the same six turns.
- An agent asked to create a Notion page never saw `notion-create-pages`: tool
  matching counted substrings, so filler like "it", "me" and "can" matched
  inside long descriptions and outranked the tool the request named. Matching
  now uses whole words (plurals folded), ignores chat filler, and weighs words
  in a tool's name three times those in its description.
- Calling an undisclosed or misnamed tool now returns something the model can
  act on: an exact name is disclosed with its schema, and a near miss
  (`notion-create-page`) gets the real name (`unknown_tool`).
- MCP tools called by name (`server__tool`) did not ask for approval before
  changing data; only `mcp_call_tool` did. The MCP layer then filed a proposal
  the playground never showed. They now pause the run for approval like the
  facade, and read-only annotations are read from the cached tool list.
- MCP arguments are checked against the tool's input schema before a call or
  an approval request, with two certain repairs (a documented field moved into
  its sibling object, wrapped text unwrapped) and otherwise a precise error.
- The playground never drew the approval card for streamed replies (it was
  handed the proposal instead of the event), so a paused write had no way to
  be approved from the chat. The card now appears, loads and shows the exact
  arguments that will be sent, and the paused tool reads "Waiting for
  approval" rather than "Failed".
- After an approval, the resumed reply no longer receives the streamed run's
  instruction to call `memorizz_finalize_answer`, a tool the continuation does
  not offer; models wrote about the missing tool instead of answering.
- A turn failed with "Missing credentials" when no OpenAI key was set,
  because recording the turn's workflow embedded it with the default OpenAI
  provider. The global embedding default now honours
  `MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER`, `_MODEL` and `_DIMENSIONS` (as the
  Oracle provider already did, e.g. `ollama` with `nomic-embed-text`), and a
  workflow is stored without an embedding when none is available.
- The playground's stop button is red with a stop icon, and the Settings
  tab's Save button stays in view with an unsaved-changes note instead of
  sitting below the whole form. The Memory tab's cards inset their contents
  and empty-state text to line up with the card titles.
- `FileSystemProvider.list_tool_logs()` and `OracleProvider.list_tool_logs()`
  returned nothing when no `user_id` was given: their own "not supplied"
  marker reached the shared filter, which treated it as a user to match.
- The context panel read a stored agent's legacy, unmarked window cap even
  where loading the agent would use the window in its LLM configuration.
- A playground reply could vanish while streaming: a conversation load still
  in flight (for example right after **New chat**) redrew the chat over it.
- Streamed replies no longer fail when a model calls a tool after tools are
  disabled for the answer (seen with gemma4 on Ollama). The call is never run;
  text already streamed stands, otherwise the model is asked again (up to
  twice). A reply with no text and no tool call is re-requested once before
  failing with `empty_response`.
- Playground Toolbox, Workflow, Entity and Summary cards used fixed dark
  colours that turned grey in the light theme; they use the theme's surfaces.
- A conversation's message count changed with the conversation open, because
  the open view counted per-turn trace rows and the list did not.
- "Auto-compacts at 80%" never triggered: the background summarizer watched
  the measured request size, which history trimming keeps under 80%, and when
  it ran it summarized every message including the latest. It is replaced by
  auto-compaction (see Added).
- The playground's context panel counted per-turn trace records as
  conversation history, counted the tools twice (as "Toolbox memory"), and
  used word counts. It now estimates the next request the way the agent
  builds it and shows the last request's measured size.
- A summary was lost when no embedding provider was available (for example no
  OpenAI key with a local model); it is now stored without an embedding.
- The playground's context panel showed a 128,000-token window for every
  agent; it now shows the budget the agent really uses. Run stages read in
  plain words ("Thinking…") instead of codes such as "provider first delta".
- `entity_memory_upsert` failed with `unexpected keyword argument
  'identity_key'` when a host added a canonical identity (OpenSpeech's
  "remember this about me"). The manager now forwards it, and a stale
  model-supplied `entity_id` defers to the canonical record.
- `retrieve_tool_log_entry` said "not found" for logs from earlier turns: the
  MongoDB and filesystem providers matched only their record ID, not the
  `tool_log_id` the model is shown. New filesystem logs use one ID for both.
- **Agents → Last created** did nothing on the filesystem backend: agents had
  no creation time. They now record one and keep it across saves; older agents
  use their file's creation time.
- A streaming test could fail under parallel runs by checking for a progress
  update before the client had read it.
- **Authorize** on MCP Connections and `memorizz mcp login` could never start
  OAuth sign-in: the 401 check ran before the OAuth provider saw the response.
  Notion sign-in now starts. Servers that list tools anonymously and ask for
  sign-in only on a call (Gmail, Google Calendar) start it with a read-only
  call instead of reporting "already authenticated".
- OAuth sign-in requests exactly the scopes configured on a connection. The MCP
  SDK used to widen them to every scope the server advertises, which for Gmail
  included full mailbox access.
- Google's OAuth issuer (`https://accounts.google.com` against the
  `https://accounts.google.com/` in its resource metadata) no longer fails the
  SDK's exact-match check; only this bare-origin slash difference is accepted.
- A provider that cannot register clients automatically now says to add an
  OAuth client ID and secret, instead of returning its raw 404 page.
- Creating an agent with a persona on the filesystem backend failed when the
  persona could not be stored separately (`'dict' object has no attribute
  'to_dict'`).
- `SKILL.md` files take their name and description from YAML frontmatter;
  before, the description was the frontmatter's `---` line.
- Loading an agent from MongoDB dropped its tool-result, context, completion,
  delegation, skill-retrieval and semantic-layer settings; the filesystem
  backend dropped its completion policy. Every stored field now round-trips.
- Starring an agent in the UI erased its meta-harness mode, default harness,
  harness configuration, completion policy and application ID; saving agent
  settings from the edit form or the playground dropped fields too. The UI now
  updates only the fields a form changes.
- An automation attempt's time limit never took effect: the worker waited for
  the hung call anyway. The worker also now records the run's answer and
  disables a job whose schedule can no longer be computed, as Run now did.
- The playground's HTML escaping left quotes unescaped.
- Claude Code runs could never search the web or use Glob and Grep: `--bare`
  loads only Bash, Edit and Read. Builds with `--restricted` now load exactly
  the policy's tools (including WebSearch and WebFetch with web access), with a
  private per-run config directory; older builds keep `--bare`.
- A missing or disallowed workspace reports the path problem and the allowed
  workspace roots instead of a raw `[Errno 2]`.
- `MetaHarness.close()` no longer closes the run store under a run that has
  started but not yet written its final status, or that registered after
  shutdown began.
- Read-only stages of `run_plan()` no longer inherit an approval requirement
  from an edit-mode base task.
- Redaction keeps camelCase token counts such as `totalTokens` and numeric
  `tokens` blocks.

## 0.12.0 — 2026-09-28

### Added

- Rebuilt the local UI as a monitoring console. The overview shows runs, success
  rate, run time, tokens, prompt-cache reuse and spend over 24 hours, 7 or 30
  days, with a runs-over-time chart, recent runs and agent, tool and model
  breakdowns. "Needs attention" names causes in plain language (failed runs,
  failing tools, provider stream errors, unpriced models, duplicate agent
  records) from content-free codes, so it also works in metadata-only mode.
- Added a fleet monitor for agents (health, runs, success, p95, tokens, cache
  share, spend, trend, detail pane, quick chat) and monitor pages for memory,
  workflow trajectories, learned skills, the learning control plane, persona
  evolution, harnesses, automations, MCP connections, Vercel skills, settings,
  observability, usage and Evalground: shared filters, sortable grids,
  keyboard navigation (`/`, `j`/`k`), full-width layouts and phone layouts.
- Added a chat-style agent playground with an inspector for the context window,
  prompt cache, memory and settings.
- Added Evalground comparisons of answer models and rerankers on shared
  evidence, with quality, latency and API cost, and the optional `rerank` extra.
- Added prompt-cache reuse across agents and providers: Anthropic cache
  breakpoints for stable prefixes (optionally for single-shot prompts),
  explicit GPT-5.6/GPT-6 cache boundaries and a default routing key for
  OpenAI, per-run usage via `MemAgent.get_last_run_usage()` and in `run.done`,
  and prompt-cache hit rates in the UI and Evalground.
- Tools discovered for an agent and user stay disclosed in later turns
  (`ContextPolicy.sticky_tool_limit`), keeping the cached prompt prefix stable.
- A provider stream that drops before any public answer text or tool call is
  re-sent up to twice; each attempt is traced as its own model call.
- Added `MongoDBProvider.estimate_count()`, `list_recent()` and
  `observability_row_cursor()`; UI counts and memory pages read bounded pages.

### Changed

- Rejected tool calls report `tool_not_disclosed`, `invalid_arguments` or
  `tool_not_callable` instead of `invalid_tool_invocation`; argument errors
  name the expected parameters.
- Anthropic stream error events record the provider's error type.
- `tool.completed` stream events carry a safe `reason_code` for failures.

### Fixed

- MongoDB observability pages stopped at the boundary between ObjectId and
  string ids, so usage and event queries skipped immutable trace bundles.
  Event reads are now batched with exact per-record resume positions.
- Concurrent `MemAgent.save()` calls (for example one per web worker) could
  insert duplicate agent records. Saves are an atomic upsert and MongoDB gets a
  unique `agent_id` index (a warning names duplicates that block it).
- Tools discovered mid-turn were forgotten before the next turn.
- Streamed playground turns started a new thread each turn, losing history.
- Added prices for current Claude models and dated model snapshots; unpriced
  calls report why they could not be priced.
- The settings sandbox readiness check did not appear, and blank learning
  control plane scope filters matched nothing.

## 0.11.0 — 2026-09-18

### Added

- Added provider prompt-cache health panels with measured reads/writes, unknown
  usage, configuration warnings and scoped cache-drop diagnostics. Prefixes
  and routing keys are hashed; prompt text is not stored in these fields.
- Added requested-versus-returned model IDs to model-call traces, inspectors
  and comparisons, including metadata received before a stream finishes.
- Added filesystem and Oracle context-engineering notebooks demonstrating
  persistent note revisions, searchable source history, and context rollover
  with real GPT-6 Astra calls and OpenAI text-embedding-3-small vectors. Both
  notebooks are standalone, use the published PyPI package, and keep each
  code cell to at most 25 lines.
- Added opt-in `ToolResultPolicy.persist_all_results` and scoped
  `MemoryManager.search_tool_logs()` with bounded source excerpts.
- Added guided `memorizz memory configure`, `memorizz notion connect` and
  `/memory-provider` setup with hidden credentials, database URL resolution,
  read-only schema checks, explicit provisioning confirmation and recovery IDs.
- Added standalone and REPL `config set/get/path/keys`, explicit project/custom
  save targets, override warnings, active/saved provider labels and restart
  guidance. UI Settings exposes Notion defaults and the real shared save path.
- Added optional `NotionProvider(config, semantic_provider=...)`: Notion owns
  memory documents; filesystem, MongoDB or Oracle stores vectors and scoped
  references without a second copy of memory text. Existing providers and
  streaming defaults remain unchanged.
- Added explicit Notion workspace provisioning, per-memory-type and linked
  agent/conversation/trace/tool views, CLI init/status/sync/repair commands,
  local UI selection and shared CLI/MCP environment configuration.
- Added durable write-intent/index repair, bounded native observability queries,
  verified webhook synchronization and live hydration of semantic results.
- Added provider-managed embedding and live-read capability contracts, including
  Notion-aware knowledge/entity/summary ingestion and semantic-cache revocation.
- Avoided redundant global-model conversation backfills for provider-managed
  embeddings; vectorless workflow rehydration also no longer re-embeds on read.
- Added deterministic transport, vector, recovery, CLI/UI/MCP and streaming
  MemAgent tests, plus opt-in live Notion and local Oracle tests.
- Release smoke checks now exercise Notion and configuration flows from the
  built wheel in a clean environment with UI/MCP extras.

### Fixed

- Guided setup and REPL configuration use Typer's public prompts and exceptions,
  so clean CLI installs work without an undeclared external Click dependency.
- JSON conversion reads text/binary LOBs once and stops following non-text reader
  results, preventing recursive reader chains from stalling cache fingerprints.
- OpenAI GPT-5.6/GPT-6 Responses calls now preserve a reusable instruction cache
  boundary alongside implicit conversation caching. Anthropic text helpers use
  the same cache policy as chat/streaming; zero read/write counters remain visible.
- Homebrew release updates remove bottles belonging to the previous version
  and tolerate a repeated update without publishing a redundant commit.
- OpenAI GPT-6 Astra context-window detection and legacy output-token translation
  now match the model configuration used by the context-engineering examples.
- Filesystem tool-log text queries now include tool arguments and results.
- Oracle knowledge-base reads by ID preserve namespace and chunk identity;
  tool-log reads also accept the physical row IDs returned by list/search.
- `/login notion` now writes `NOTION_TOKEN`. Credential/configuration slash
  commands are excluded from REPL history; hidden input fails closed without
  a secure prompt, and credentials are not exposed by config reads or UI fields.
- Shared env writes are atomic, serialized and owner-only on POSIX, preserving
  unrelated multiline bindings/comments and refusing malformed/symlink targets.
  Failed saves no longer produce misleading unconditional success messages.
- Explicit invalid/misconfigured CLI memory backends no longer silently fall
  back to filesystem. MongoDB/Oracle CLI connections honor configured external
  embedding defaults; saved provider changes never hot-switch the running agent.
- Fixed a live Notion date-precision mismatch that rejected freshly saved
  memories. Precise timestamps now drive ordering and exact time-window checks;
  minute-level date buckets no longer omit subminute observations. Added live
  regression coverage and a persistent, explicitly synthetic Notion walkthrough.
- Ollama now sends its effective context window to the daemon, defaults to
  8,192 tokens, and preserves the setting on reload. CLI resume and model
  switches refresh the history budget instead of retaining a stale 128k limit.
- Streaming accepts a model's direct tool-phase answer as a finalization
  proposal under the existing host evidence checks. Drafts stay private and
  public generation uses a separate request with tools disabled, preventing
  repeated finalizer retries on local models.
- Complete model requests now budget context, tool schemas and accumulated
  tool evidence on every iteration. Old turns are evicted together; oversized
  required input stops with `context_window_exceeded` before a provider call.
- Provider context limits and supported generation options survive reload.
  SDK model overrides now replace the actual runtime model and its budget,
  while normal reloads preserve agent-level context caps.
- Empty and token-truncated model responses cannot complete or enter the
  answer cache in either streaming or synchronous runs. Errors retain partial
  streamed text, expose stable codes and explain recovery in the CLI.

### Compatibility

- Requires Python 3.10+ and Typer 0.27.2+. Python and npm versions are both 0.11.0;
  the npm package bootstraps the matching Python CLI. No new base dependency is
  required; package installers update Typer automatically when needed.
- Explicit invalid or incomplete CLI backend settings now fail with setup
  guidance instead of silently selecting filesystem memory. Configuration
  saves take effect on the next launch and do not migrate existing memory.
- Empty/truncated model output and requests that exceed the context budget now
  raise explicit errors. Applications using synchronous runs should handle
  `ProviderStreamError`; streamed runs expose a terminal error event.
- Provider-cache telemetry is additive. Missing counters and returned model
  IDs remain unknown in older traces; cache diagnostics do not imply a price.
- Notion requires explicit provisioning, access configuration and reconciliation
  after human edits. Local journal status is not workspace-wide completeness.
  Live service tests require credentials and remain opt-in.
- Notion is not an atomic coordination provider: concurrent shared-memory
  delegation, leases and coordination-dependent automation must use an existing
  atomic primary provider. The semantic backend remains vector-only.

## 0.10.0 — 2026-09-08

### Added

- Added usage dashboards, accessible charts and structured tables in
  Observability, selected traces and Evalground: daily tokens and known charges,
  agent/model/interaction breakdowns, memory prompt share and retrieval latency.
- Added public `aggregate_usage`, `query_usage`, `RateCard`, `PricingRegistry`
  and `MemAgent.last_usage_analytics()` APIs. Bounded, tenant-scoped reads and
  offline decimal pricing avoid network pricing lookups on the agent path.
- Added versioned OpenAI text rate cards, cache-write and service-tier telemetry,
  long-context pricing, and durable cost provenance. Other providers and custom
  billing agreements can supply explicit rate cards.
- Added request-local `PersonaManager.use_snapshot()` contexts for account-owned
  personas, safe persistence of shared agent configuration, versioned persona
  trace references and a distinct approximate persona-token category.
- Added a host-adapter Persona Evolution UI for profile inspection, evidence,
  reviewed proposals, version history and optional reflection/approval/undo/
  scheduling controls. Private profiles require authentication and audited
  unrestricted administrator access; actions are disabled unless enabled by
  the host.
- Interactive CLI sessions now announce newer stable PyPI releases with an
  installer-specific upgrade command. Checks run in the background, cache
  successful results for 24 hours, tolerate offline use, and can be disabled
  with `MEMORIZZ_NO_UPDATE_CHECK=1`.

### Changed

- Persona goals are explicitly treated as style preferences, not authority over
  application instructions, grounding, permissions or the user's current request.
- Semantic-cache identity now incorporates rendered persona text, preventing
  reuse across different goals that happen to share a name and version.
- Condensed the README around memory management, the agent framework,
  MetaHarness, and continual learning, positioning Memorizz as the intelligence
  plane for memory-first agents with links to detailed setup and feature guides.

### Fixed

- Preserved persona identity/version through persisted trace compaction, not
  only live callbacks, and kept persona style supply separate from claimed
  behavioral adoption.
- Preserved MongoDB conversation `agent_id` ownership metadata, including bulk
  writes. Existing rows are not silently backfilled or broadened across tenants.
- Scoped trace views now distinguish uninspected agent registration from a
  genuinely missing registration.
- Rejected oversized and malformed persona actions before invoking host hooks,
  including non-boolean settings and invalid revision/proposal identifiers.
- Added the IANA timezone dependency on Windows for timezone-aware analytics.

### Compatibility and release checks

- Token charges are calculated from dated rates, not reconciled invoices.
  Missing usage/prices remain unknown; memory token attribution is approximate.
  Bounded recorded windows are not complete historical account billing.
- Persona persistence, scheduling, approval policy and account authorization
  remain host responsibilities. This release does not bundle an application
  scheduler or automatically apply persona changes.
- Streaming remains enabled by default. No provider data migration is required
  for the new scalar telemetry; an uninitialized native trace index still needs
  the existing explicit migration before index-mode reads are enabled.
- Added synthetic browser release gates for usage analytics, Persona Evolution
  and existing trace inspection. Audit reports and local data stay excluded from
  release archives.

## 0.9.0 — 2026-09-07

### Fixed

- Delegation now validates complete pending plans before work, propagates runtime
  failures and approval waits, and never reruns the root after child effects.
  Empty automatic plans require explicit root-fallback opt-in.
- Child conversation threads are isolated by default, with copied request and
  cancellation contexts, bounded worker supervision, runtime receipt/lifecycle
  callbacks and host completion-policy gates. Shared cached agents retain their
  memory-ID configuration unless participant persistence is explicitly enabled.
- Shared coordination writes use conditional atomic updates on filesystem,
  MongoDB and Oracle providers. Reports and sessions agree on outcome/counts;
  recording errors remain distinct from successful execution.
- Added public-API artifact/tenant/cancellation regressions, cross-process and
  live database concurrency tests, and a developer delegation guide. Local audit
  inputs under `docs/reports/` are excluded from source packages and documentation.

### Compatibility

- Delegation now defaults to task-scoped child threads and disables automatic
  root fallback. Set `allow_root_fallback=True` explicitly when that behavior is
  appropriate; participant configuration persistence is also opt-in.
- Hosts must supply a completion policy to verify their own artifact receipts
  and business outcomes. Cancellation remains cooperative for third-party
  blocking calls, which stay supervised until they settle.
- Streaming remains enabled by default. This release changes Memorizz only;
  consuming applications still need their own integration acceptance tests.

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
