# Memorizz observability roadmap

## What the OpenSpeech production audit found

OpenSpeech had durable conversation and tool records, but older processes did
not consistently register one logical agent. That left an empty or fragmented
agent index, runtime IDs that changed across restarts, and conversations that
could not be reached reliably from an agent-first trace UI. The dashboard also
loaded entire collections, streaming was the only complete trace path, and
heuristic recommendations had no durable review or evaluation lifecycle.

The source build still recovers legacy rows under **Unregistered runtime
traces**, but new executions no longer depend on that fallback.

## Highest-priority upstream work completed

1. **Canonical ownership and trace identity**
   - `MemAgent` accepts `application_id`, derives a stable UUID5 when an
     application/name is supplied, and preserves an explicit `agent_id`.
   - First execution automatically upserts the agent and its current memory
     associations. Registration is fail-soft so chat continues during a
     database outage or against a deliberately read-only provider.
   - Version 2 trace envelopes carry `application_id`, `agent_id`, `run_id`,
     `turn_id`, `root_trace_id`, `span_id`, `parent_span_id`, `memory_id`,
     `thread_id`, and `user_id`.

2. **Query-native bounded trace access**
   - The provider contract now exposes `query_observability_records(...)` with
     agent, memory, thread, tenant, time, tool, and success filters.
   - MongoDB executes native indexed queries with opaque cursors. The UI never
     needs a collection-wide trace snapshot and reports query duration, scan
     count/lower-bound status, truncation, and freshness.
   - Private trace bundles are stored outside `conversation_memory`, preventing
     telemetry from entering chat recall while retaining legacy-bundle replay.
   - `/traces/events.json` exposes bounded pages for local analysis tooling.

3. **Safe production inspection**
   - `MongoDBConfig(read_only=True)` skips collection and index creation.
   - The UI can enforce a mutation-blocking provider proxy and HTTP policy with
     `MEMORIZZ_UI_READ_ONLY=true`.
   - `MEMORIZZ_UI_AUTH_TOKEN` enables token login and a signed, expiring,
     HttpOnly session. Trace content supports `full`, `redacted`, and
     `metadata` modes; read-only mode defaults to `redacted`.
   - Every trace dashboard, timeline, JSON page, export, save, and review action
     emits a content-free local audit event. Connection credentials remain
     masked and are not rendered into browser-readable values.

4. **Synchronous and streaming model/tool spans**
   - Every normal model invocation and tool invocation now produces child span
     events for both `run()` and `run_stream()`.
   - Model results record monotonic provider duration, provider/model identity,
     terminal status, and available input/output/cache/total token usage.
   - Tool results record the logical and model-facing names, tool-call ID,
     monotonic duration, success/status, and typed error code.

5. **Reviewable recommendations and Evalground handoff**
   - Deterministic Trace Insights can be persisted without mutating the agent.
   - Recommendation revisions and append-only accepted/rejected/deferred
     decisions are stored privately through the configured memory provider.
   - Accepting a revision creates exactly one versioned draft Evalground
     experiment with a baseline fingerprint, proposed change, evidence refs,
     metrics, and an explicit operator promotion gate.

6. **Verified feedback and outcome joins**
   - Applications can call `agent.record_feedback(...)` and
     `agent.record_task_outcome(...)` using `agent.get_trace_context()`.
   - Workflow outcomes are linked automatically; they are marked verified only
     when an application-supplied business outcome evaluator produced them.
   - Trace Insights prioritize verified negative feedback and task failures.
     Comments are hashed by default and copied only with explicit opt-in.

## Follow-on work

- Backfill canonical identity onto legacy records after an operator approves a
  migration plan; immutable legacy data is not rewritten automatically.
- Add context/retrieval/cache/write spans, TTFT, provider-specific estimated
  cost, payload byte counts, and retry taxonomy.
- Add server-side aggregate p50/p95/p99 views, waterfall rendering, trace
  comparison, version diffs, sandbox replay, saved queries, and permalinks.
- Run draft experiments directly against outcome-linked datasets, persist
  baseline/candidate scores, and add an audited promote/rollback workflow.
- Add alerting, sampling/retention controls, dropped-event counters, schema
  coverage, worker health, and an observability self-test.

## Release gate

Before exposing production data, create a database user with only the `read`
role for the selected database, enable UI authentication and read-only mode,
keep the service bound to localhost or a trusted private network, and validate
conversation/model/tool/error traces plus cursor pagination against production-
scale data. See `docs/observability-ui.md`.
