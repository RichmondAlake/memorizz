# Observability and Trace Inspection

Memorizz records scoped operational evidence so developers can explain a turn
without scanning and deserializing every conversation row. Traces cover the
model/tool loop, context provenance, semantic-cache decisions, verified host
outcomes, and learning signals.

## What is captured

| Evidence | Examples |
|---|---|
| Turn identity | agent, memory, user, thread, run, turn, and root trace IDs |
| Runtime spans | model calls, structured tool outcomes, duration, and usage |
| Context provenance | privacy-safe page/thread binding, ownership, grounding status, source IDs, and content fingerprints |
| Cache routing | hit, miss, bypass, rejection, fingerprint, and bypass reason |
| Outcomes | verified feedback and application task result |
| Learning | candidates, lifecycle events, and draft recommendations |
| Memory supply | retrieved candidates, context actually supplied, conservative explicit references, source counts, and fallback/degradation state |

Host provenance is allowlisted and content-safe: arbitrary host fields, page
bodies, prompts, excerpts, URLs, and credentials are not copied into the
provenance event. IDs and fingerprints can still be sensitive metadata, so
apply normal access and retention controls.

## Attach host provenance

```python
response = agent.run(
    "Summarize the current analysis.",
    memory_id="workspace-7",
    user_id="user-42",
    thread_id="analysis-123",
    context={"current_page": {"type": "analysis", "id": "123"}},
    observability_context={
        "request_id": "turn-456",
        "client_page_type": "analysis",
        "client_page_id": "123",
        "canonical_page_type": "analysis",
        "canonical_page_id": "123",
        "thread_binding_status": "matched",
        "expected_thread_id": "analysis-123",
        "ownership_verified": True,
        "grounding_status": "ready",
        "grounding_source": "stored_text",
        "grounding_source_ids": ["chunk-1", "chunk-2"],
        "grounding_excerpt_count": 2,
    },
)
```

`context` is model-visible for this turn. `observability_context` is not sent
to the model and is sanitized before persistence.

## Join verified outcomes to a turn

```python
trace = agent.get_trace_context()

agent.record_feedback(
    rating=-1,
    trace_context=trace,
    verified=True,
    label="thumbs_down",
    comment="Answer used the wrong workspace",
)

agent.record_task_outcome(
    "failure",
    trace_context=trace,
    verified=True,
    source="application_job_status",
    metrics={"completed": False},
)
```

Comments are hashed unless `include_comment=True` is explicit. “Verified”
should mean that a trusted application or operator observed the outcome—not
that the model described its own response as successful.

## Get a content-free summary

```python
summary = agent.observability_summary(
    memory_id="workspace-7",
    user_id="user-42",
    thread_id="analysis-123",
)
```

The result includes bounded counts and freshness information without returning
conversation bodies. Use provider-native observability queries for large
deployments; compatibility fallbacks scan a bounded provider window.

Every non-cached model turn also records a content-free **Memory supplied**
event. The memory-first view separates three stages that are easy to conflate:

1. **Retrieved** — candidates considered by memory retrieval.
2. **Supplied** — bounded history, profile attributes, preferences, relevant
   cross-thread excerpts, summaries, and writing samples actually placed in the
   model context.
3. **Referenced** — exact source overlap detected in the final text.

Tone preferences and writing samples influence behavior rather than producing
reliably attributable phrases. The UI labels those sources as “not measured”
instead of incorrectly reporting that they were unused. Persisted traces keep
counts, scores, attribute/preference names, and hashed references; raw values
remain ephemeral.

## Inspect traces in the local UI

Install `memorizz[ui]`, connect the same memory provider, and open **Traces**.
Filter by agent and thread to view the timeline, memory supply pipeline,
context/cache tags, tool health, verified outcomes, and Trace Insights. Insights currently identify
issues such as page/thread mismatch, unverified ownership, missing current-page
grounding, content attached to an unscoped thread, repeated provider fallback,
and repeated degraded tool execution.

Tool result badges distinguish **Completed**, **Completed · no results**,
**Completed with limitations**, **Completed via fallback**, **Provider error**,
and **Failed**. Fallback and degraded results remain usable but are counted
separately from clean primary-provider success. Provider errors count as tool
failures. This prevents a green completion state from hiding an offline or
unavailable primary provider.

Accepted recommendations create draft experiments. They do not edit a live
prompt, policy, tool, or learned skill automatically.

The bounded JSON endpoint
`/traces/events.json?store=trace&limit=250` returns trace pages;
`store=conversation` and `store=tool_log` select the other indexed sources.
Cursors are opaque.

### Normalized events and coverage

`view=events` is the default, with `store=trace` selecting private trace bundles.
`view=bundles` returns the original provider records for forensic inspection.
Use `store=conversation` or `store=tool_log` to export those additional sources;
the HTML timeline combines all three through the same normalizer.

```text
GET /traces/events.json?view=events&agent_id=chat-assistant&thread_id=thread-123&limit=250
GET /traces/events.json?view=bundles&agent_id=chat-assistant&thread_id=thread-123
```

The event page includes `source_rows`, `bundle_count`, `normalized_events`,
`normalization_errors`, `duplicates_removed`, `coverage`, and `next_cursor`.
Limits count child events. A cursor retains a position inside a bundle, is
bound to its query filters, and rejects a changed partially consumed bundle.
Page counters describe records read for that page: a bundle spanning multiple
pages contributes to each page's source count. Do not sum those source counts
to estimate unique bundles. Follow `next_cursor` until it is null.

Legacy v1/v2 bundles and typed v3 children normalize together. An event matching
the selected agent remains visible even when its memory ID is absent from the
saved agent registration. Child timestamps retain UTC and fractional seconds.
Malformed records produce data-quality findings rather than conversation rows.
The UI and analyzer label incomplete evidence as partial, unknown, or untrusted.
`metric_states.tool_calls` carries an explicit state and nullable value; legacy
numeric summary fields remain observed counts for API compatibility.

Coverage without a declared profile means only that the loaded events normalized.
It does not establish artifact or delivery success. Declare `chat_response`,
`content_ingestion`, `content_derived_artifact`, or `interactive_response` through
the recorder's `coverage_profile` intent attribute. Required stages are checked
separately for each tenant, turn and task; a model returning prose is insufficient evidence
for a required interactive contract.

### Health, causal inspection, and comparison

The trace timeline includes a **Causal waterfall** that pairs start/result events,
shows recorded parent links, links spans to their event evidence, and flags
missing parents, missing causal events and cycles. Missing durations are shown
as unknown. An observed parent link does not prove source ownership or quality.

`/traces/health` and `/traces/health.json` expose normalized-event and schema
counts, normalization/query failures, missing identities and stages, orphan
spans, query timing and compatibility fallback. Optional `agent_id`, `thread_id`
and `limit` (up to 1,000 rows per store) narrow the inspected window. Write
failures and dropped events are explicitly unknown: absent storage records
cannot establish whether telemetry was lost. Inspect the host recorder's health
counters separately. This is not a deployment-wide health probe.

`/traces/compare` and `/traces/compare.json` compare two captured root traces for
one agent. Supply `agent_id`, `baseline_root`, and `candidate_root`; optionally
use `thread_id`, `baseline_turn`, and `candidate_turn`. The comparison reports
changes in operations, source refs, artifact state, parser results, delivery,
and diagnostic codes. It never runs models or tools or replays side effects.
It uses metadata-only output regardless of the timeline content mode. A finding
absent from a partial candidate window is **not** labelled resolved. Unknown
traces return a bounded-window not-found error rather than an empty success.

The normalized event export also accepts `root_trace_id`, `run_id`, `turn_id`,
`resource_ref`, `event_kind`, `status`, `start_time`, and `end_time`. Time filters
apply to child-event timestamps, not bundle write time. These filters require
`view=events`; raw bundles do not silently ignore them. Queries remain bounded
compatibility reads, not an indexed global incident directory.

All new views inherit UI authentication, audit logging and no-store responses.
Structural inspection helpers are also available from `memorizz.observability`:
`build_trace_health`, `build_causal_waterfall`, and `compare_trace_windows`.

## Host actions, workers, and artifacts

Memorizz cannot observe a host database write or browser render without an SDK
call. Instrument application fallbacks, queue boundaries, persistence, parsing,
emission, and acknowledgements at the point where the host knows the result.

```python
from memorizz.observability import ObservabilityRecorder, ResourceRef, TraceContext

trace = TraceContext.from_carrier(agent.get_trace_context())
recorder = ObservabilityRecorder(agent.memory_provider, trace)
source = ResourceRef(
    resource_type="analysis",
    ref="analysis-123",
    version="2",
    role="authoritative_current_source",
    ownership_verified=True,
    provenance_status="verified",
)
recorder.record_intent(
    "slides", expected_artifact_types=["slide_deck"], input_refs=[source]
)

with recorder.start_span(
    "slides.safety_net",
    input_refs=[source],
    attributes={"fallback_reason": "agent_task_outstanding", "side_effect": True},
) as span:
    deck = create_deck(source_id=source.ref)
    recorder.record_artifact(
        ResourceRef(resource_type="slide_deck", ref=str(deck.id)),
        input_refs=[source],
        producing_span_id=span.span_id,
        persistence_verified=deck.persisted,
        task_id="slides",
        external_id=f"deck:{deck.id}",
    )
    span.succeed()
```

`agent.observe(...)` is a shortcut for a span using the agent's last trace and
provider. Use one `ObservabilityRecorder` for nested spans so their parent
relationships are tracked. Both spans and contracts support `with` and
`async with`; context-local parent IDs keep concurrent async tasks separate.
Provider writes remain synchronous, so async applications should account for
their provider's I/O latency.

Serialize `span.to_carrier()` into a job and restore it with
`TraceContext.from_carrier(job_carrier)` in the worker. This preserves root,
turn, run, thread, memory, and parent identity. An incomplete or unknown-field
carrier raises `ValueError`; a worker must not invent success or silently
create an unrelated root. A carrier is not an authentication credential:
validate tenant/job ownership and bind the provider before trusting it.

Stable `external_id` values upsert the same event rather than appending duplicate
worker delivery records. Use a distinct external ID for a genuinely new attempt.
Events are stored in private v2 envelopes with v3 children, so existing provider
queries work without a schema migration. Runtime bundles remain v2 and separate
turns sharing a root no longer overwrite one another.

Artifact recording describes an already observed result; it does not create or
repair the artifact. `input_refs` and `producing_span_id` are required arguments.
An empty source list records invalid provenance and triggers a P0 diagnostic.
Only set `source_independent=True` for an intentionally source-free artifact.

### Output contracts and delivery

```python
with recorder.contract("learning_map", required=True, output_limit=2200, task_id="slides") as contract:
    parsed = parse_learning_map(raw_response)
    contract.record(
        opening_marker_found=parsed.opening_found,
        closing_marker_found=parsed.closing_found,
        parser_status=parsed.status,  # success, failed, missing, or partial
        recovery_used=parsed.recovery_used,
        item_counts={"nodes": len(parsed.nodes), "edges": len(parsed.edges)},
    )

recorder.record_delivery("learning_map", emitted=True, acknowledged=None, task_id="slides")
```

Record a separate delivery update when the client acknowledges the payload.
For artifact delivery, pass the artifact refs in `input_refs` so persistence
can be reconciled with the exact resources delivered.
`acknowledged=None` means unknown. Recovery is partial even if the application
can render a recovered structure. Missing markers, failed parsing, failed
emission, and missing acknowledgements have distinct findings. The library
does not parse application markup or supply the browser acknowledgement.

`reconcile_outcome(events)` accepts the complete authorized normalized window.
It records success only when an intent, valid artifact/contract evidence, and
acknowledged delivery exist without diagnostic gaps or query truncation. A
partial window produces a partial outcome. This is deliberately stricter than
the model's turn status.

Use matching `task_id` values on intent, artifact, contract, action attributes
and delivery events when multiple tasks share a turn. Legacy unlabelled evidence
is assigned only when there is one unambiguous intent. `reconcile_outcome(...,
task_id="slides")` verifies that task; without a task ID every intent in the
recorder's tenant/turn scope must be satisfied. Other tenants, threads or turns
cannot supply missing evidence. Stable external IDs are isolated by tenant and
turn as well as root trace.

Reconciliation uses the latest artifact state and contract/delivery result.
Deleted artifacts and failed updates do not satisfy requested artifacts.
Delivery must identify the persisted resource version and follow its persistence
and parsing evidence; an old acknowledgement cannot verify a new revision.
Producers must finish successfully, every declared authoritative source must
match, and explicitly failed ownership or missing-provenance checks block success.
A successful replacement parse can supersede an earlier failure but
requires a fresh delivery. Diagnostic evidence links point to the implicated
events, with lower confidence for absent evidence in incomplete windows.

Keep normalization metadata with the events. A plain list defaults to unverified;
pass its actual query metadata via `normalization_metadata` when necessary.
Event pages expose `window_complete`: even the final continuation page is not
the complete window. After collecting all pages, the host must establish complete
coverage without errors before reconciling; do not mark an arbitrary page complete.

### Metadata and operational limits

Resource refs are opaque IDs or keyed fingerprints; do not pass raw URLs,
emails, prompts, transcripts, or credentials. Attribute names are allowlisted,
with at most 64 fields, 32 scalar list entries, and 240 characters per string.
Nested arbitrary dictionaries, unknown fields, nonfinite numbers and recognizable
credentials/emails/URLs are rejected before host-event persistence. This cannot
detect every form of sensitive free text: callers must follow the content-free
contract. Existing runtime content previews still follow the configured UI
content mode and must be handled according to the host's storage policy.

Validation failures raise. Provider write failures normally leave the host
operation running, log a content-free error code, and increment
`recorder.health["write_failures"]`. Use `strict=True` in development or when
telemetry persistence is required. Forward the health counter to application
monitoring; it is local to the recorder, not a durable health service.

Provider adapters expose optional `get_last_response_metadata()` with stop
reason, request ID, token counts, response length and configured output limit
where available. MemAgent adds stream TTFT and provider time. Providers that
do not expose a field leave it unknown; retry counts and stop reasons are not
inferred. This optional method does not change the existing LLMProvider protocol.

Set `MEMORIZZ_UI_PSEUDONYM_KEY` to a secret and `MEMORIZZ_UI_AUDIT_SCOPE` to an
audit-domain identifier for stable HMAC display and memory-reference pseudonyms.
Without a key they are stable only within the current process. Production hosts
must share the configured secret and scope across their workers.

Native SQLite, MongoDB and Oracle span indexes, explicit migrations, backfill,
parity checks, retention, account-resolution hooks, role-gated reveals and inert
Evalground replay drafts are implemented. The compatibility path remains the
default rollback path and can scan stored rows. See the
[rollout and operator guide](observability-rollout.md) for provisioning, cutover,
privacy policy, permissions, endpoint contracts and deployment validation.

The Incident Finder searches opaque user/thread/root/job/source/artifact/error
identifiers with time filters and pagination. Email resolution is a separate,
permission-gated POST to a host-supplied resolver, not a persisted account directory.
The agent-table search still filters its bounded overview.

### Reproduce the validation locally

```bash
PYTHONPATH=src python examples/observability/host_workflow.py
python -m pytest tests/unit/test_observability_normalization.py tests/unit/test_observability_recorder.py tests/unit/test_observability_response_metadata.py -q
python -m pytest tests/unit/test_observability_evidence.py tests/unit/test_observability_inspection.py -q
python -m pytest tests/unit/test_observability_index.py tests/unit/test_observability_maintenance.py tests/unit/test_observability_operator_security.py tests/unit/test_observability_roadmap.py tests/unit/test_observability_scale_oracle.py -q
PYTHONPATH=src python examples/observability/benchmark.py --events 10000
PYTHONPATH=src python -m pytest tests/unit tests/integration -q
```

The example uses temporary storage, synthetic source/artifact IDs, and no model
or production services. Regression tests cover the 3-bundle/74-event incident,
14 tool calls, mismatched memory associations, malformed bundles, child cursors,
filesystem and mongomock queries, host retry/async behavior, artifact/contract
failures, task/tenant isolation, revision-aware outcomes, health/causal/comparison
views, authentication, metadata privacy, successful delivery, immutable retries,
native-index parity, retention and role/tenant boundaries. Optional live database
and browser gates are documented in the rollout guide. Synthetic local checks
do not replace production host integration or database deployment validation.

## Secure operator setup

Use a read-only database identity, localhost or a private network, UI token
authentication, redaction, and audit logging:

```bash
export MEMORIZZ_BACKEND="mongodb"
export MONGODB_URI="mongodb://read-only-user:password@db.internal/memorizz"
export MEMORIZZ_UI_READ_ONLY="true"
export MEMORIZZ_UI_AUTH_TOKEN="a-long-random-operator-token"
export MEMORIZZ_UI_SESSION_SECRET="a-separate-stable-session-secret"
export MEMORIZZ_UI_COOKIE_SECURE="true"
export MEMORIZZ_UI_TRACE_CONTENT_MODE="redacted"
export MEMORIZZ_UI_AUDIT_LOG="$HOME/.memorizz/audit/trace_views.jsonl"
memorizz ui --host 127.0.0.1 --port 8765
```

Content modes:

- `metadata` hides content, arguments, and results;
- `redacted` masks common secret, credential, email, and user-ID patterns;
- `full` is an explicit opt-in for controlled test data.

Read-only mode rejects provider mutations, including nested index operations.
Audited account resolution and content reveal remain read operations; replay
creation is blocked, while an authorized replay-plan export is allowed.
The audit JSONL records access metadata, role and keyed operator/client
pseudonyms, not query strings, account emails, credentials, or trace content. Redaction is a
defense in depth, not a substitute for preventing secrets from entering agent
inputs or tool results.

See [Configuration and Secrets](reference/configuration.md) and
[Production Governance](guides/production-governance.md) before exposing the
operator UI outside a developer machine.
