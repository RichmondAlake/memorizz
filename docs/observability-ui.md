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

Read-only mode rejects provider mutations and unsafe routes outside the
login/connect flow. The audit JSONL records access metadata and a salted client
address hash, not query strings, credentials, or trace content. Redaction is a
defense in depth, not a substitute for preventing secrets from entering agent
inputs or tool results.

See [Configuration and Secrets](reference/configuration.md) and
[Production Governance](guides/production-governance.md) before exposing the
operator UI outside a developer machine.
