# Memory-first learning control plane

MemoRizz uses one deliberately small control-plane layer to connect retrieval,
observability, memory compilation, workflow-to-skill learning, and forgetting.
It does not introduce another database or service. The layer stores its logical
records in the existing private `shared_memory` partition, so the same contract
works with filesystem, MongoDB, and Oracle.

> **Agent memory is how we make intelligent systems retain, reuse, recall and
> refine information.** Retention alone is not enough: a production agent also
> needs bounded recall, evidence about outcomes, a path from experience to
> reusable procedure, and a governed way to stop retrieving obsolete
> projections.

## The five records and decisions

| Concept | Purpose | Durability |
|---|---|---|
| `LearningEvent` | Immutable run, cache, tool, workflow, outcome, skill-lifecycle, compilation, and forgetting fact | Stored once with a content hash and idempotency key |
| `EvidencePack` | Per-turn, tenant-scoped retrieval decision with provenance, trust, freshness, conflicts, and a hard token budget | Selection metadata is emitted as an event; raw content is not copied into that event |
| Compiled artifact | Small deterministic projection of related run/workflow events | Rebuildable from events; incremental checkpoint prevents duplicate work |
| `OutcomeEvidence` | Host/application result with source, score, metrics, and a `verified` boundary | Joined to the exact trace and observability record |
| Forgetting tombstone | Reversible suppression of an obsolete compiled projection | Requires a durable plan and named approver; immutable source events remain |

The data path is intentionally linear:

```text
run/tool/cache/workflow events
          │
          ├── observability trace + host/application outcome
          │
          ▼
deterministic incremental compiler ──► compact retrieval artifacts
          │                                  │
          └── workflow promotion events      ▼
                                      bounded EvidencePack
                                               │
                                               ▼
                                      next model inference
```

## Enable it

Use the builder when composing a complete agent:

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("support-agent")
    .with_model(model)
    .with_memory_provider(provider)
    .with_semantic_cache(enabled=True, threshold=0.92)
    .with_continual_learning(
        True,
        config={"require_shadow": True, "promotion_every_n_runs": 25},
    )
    .with_learning_control_plane(
        evidence_token_budget=1600,
        evidence_max_items=6,
        freshness_limits_seconds={"knowledge_base": 3600},
        compile_every_n_events=12,
    )
    .build(validate=True, persist=True)
)
```

`continual_learning=True` enables the control plane by default. Pass
`learning_control_plane=False` explicitly when compatibility requires the old
multi-source context path. The filesystem provider remains the default when no
provider is supplied.

The direct constructor accepts the same configuration:

```python
agent = MemAgent(
    model=model,
    memory_provider=provider,
    learning_control_plane={
        "enabled": True,
        "evidence_sources": [
            "knowledge_base",
            "conversation_memory",
            "summaries",
            "entity_memory",
            "skillbox",
            "learning_artifacts",
        ],
        "evidence_token_budget": 1200,
        "compile_async": True,
    },
)
```

## Bounded recall before inference

When enabled, `EvidencePack` owns automatic multi-source retrieval for each
turn. It applies memory, user, thread, and agent scope before ranking; excludes
promoted raw workflows when their active skill is available; removes exact
history duplicates; caps items per source; and enforces the token budget before
the model request. The model receives evidence as untrusted retrieved data,
not instructions.

Entity memory uses the same retrieval contract inside and outside the control
plane. Semantic search is attempted when its provider reports a usable or
lazy-provisionable vector index. A missing, unsupported, or failed vector path
falls back to a bounded exact candidate window in the same `memory_id` and
`user_id`; retrieval mode and degradation reason are retained in evidence
metadata and warnings. Anonymous `user_id=None` entity rows are never included
for an authenticated user.

```python
answer = agent.run(
    "What did we learn from the failed deployment?",
    memory_id="release-42",
    user_id="tenant-a",
    thread_id="incident-7",
)

decision = agent.explain_memory_decision(include_content=False)
print(decision["tokens_used"], decision["tokens_saved_vs_candidates"])
print(decision["items"])  # source, hash, score, trust, age, stale/conflict flags
```

`freshness_limits_seconds` applies a ranking penalty and exposes the stale flag;
it does not magically validate changing business data. Use semantic-cache
domain/version invalidation and a current tool call for operational facts.

## Events, outcomes, and incremental compilation

Run, cache, tool, and workflow hooks are automatic. Application outcomes must
be attached by the host so a model cannot certify itself:

```python
result = agent.record_task_outcome(
    "success",
    verified=True,
    source="payment_processor_webhook",
    score=1.0,
    metrics={"settlement_id": "safe-reference-only"},
    trace_context=agent.get_trace_context(),
    external_id="settlement-event-9182",
)
assert result["outcome_evidence"]["learning_authoritative"] is True
```

An `external_id` makes retries idempotent. Reusing an idempotency key with
different content raises a conflict instead of silently rewriting history.
Do not place credentials or unnecessary personal data in outcome metrics.

The fast compiler is deterministic and does not call an LLM. It groups related
events into small run/outcome artifacts and checkpoints event hashes:

```python
report = agent.compile_memory(
    memory_id="release-42",
    user_id="tenant-a",
    thread_id="incident-7",
)
print(report["compiled_events"], report["artifacts_written"])
```

Compiling the same unchanged stream again writes no new work. The original
events remain the source of truth; artifacts may be rebuilt.

## Continual learning and instruction hierarchy

The control plane records workflow and skill candidate/promotion/demotion/
deprecation transitions. The existing promotion engine still owns trajectory
canonicalization, frequency/success/recency/query-diversity gates,
distillation, shadow evaluation, review, and drift monitoring.

This separation matters because a learned procedure eventually enters an
LLM's instruction hierarchy. Raw trajectories are observations, not
instructions. MemoRizz compiles repeated successful trajectories into a
validated `SKILL.md`; new skills default to user-context authority. A skill can
enter the developer/application instruction channel only after shadow review
and explicit activation, and it remains subordinate to system policy and
current tool evidence. Demotion releases its source workflows back to normal
retrieval.

Tool execution success is not automatically business success. Configure
`workflow_outcome_evaluator` when promotion must depend on a domain result. The
control-plane event marks execution-only outcomes `verified=False`; verified
application evidence can be added later with `record_task_outcome()`.

## Reversible forgetting

Forgetting is a retrieval-governance mechanism, not destruction of the audit
trail. The planner identifies duplicate projections and old, low-utility
artifacts. It preserves verified outcome artifacts by default and produces a
dry-run plan with a stable content-derived ID.

```python
plan = agent.plan_forgetting(
    memory_id="release-42",
    user_id="tenant-a",
    thread_id="incident-7",
)
print(plan.to_dict())

# The plan can be reloaded after a process restart.
plan = agent.get_forgetting_plan(
    plan.plan_id,
    memory_id="release-42",
    user_id="tenant-a",
    thread_id="incident-7",
)

applied = agent.apply_forgetting(
    plan,
    approved_by="operator@example.com",
    reason="newer projection verified",
    memory_id="release-42",
    user_id="tenant-a",
    thread_id="incident-7",
)
```

Application is single-use. It rechecks each target, writes an auditable
tombstone, and never hard-deletes immutable learning events. Provider-level
`delete_scope(...)` remains the separate lifecycle operation for exact-scope
data removal.

## CLI and local UI

The CLI uses the same provider selected by `--backend`:

```bash
memorizz learning status --agent-id AGENT --memory-id MEMORY --user-id USER --json
memorizz learning events --agent-id AGENT --memory-id MEMORY --user-id USER
memorizz learning compile --agent-id AGENT --memory-id MEMORY --user-id USER
memorizz learning forget-plan --agent-id AGENT --memory-id MEMORY --user-id USER
memorizz learning forget-apply PLAN_ID --agent-id AGENT \
  --memory-id MEMORY --user-id USER --approved-by operator@example.com
```

Valid backends are `filesystem`, `mongodb`, and `oracle`; filesystem is the
default. In the local UI, enable the feature on the agent form and open
**Learning Control Plane** to inspect counts and identifiers, compile events,
create a dry-run plan, and apply approved tombstones. Raw event payloads and
retrieved content are intentionally not rendered on that operator page.

## Oracle

No new Oracle table is required: logical control-plane records use the existing
private `shared_memory` table, whose full list projection is supported by the
provider. Agent configuration persists in the existing JSON configuration
column.

Run preflight before production or evaluation:

```bash
memorizz oracle preflight --index-policy lazy --json
```

The configured embedding dimension must match every existing Oracle `VECTOR`
column. For the package local-runtime default this is commonly 384; an older
external-embedding schema may differ. Treat
`embedding_dimension_compatible=false` as a hard pre-inference failure. The
benchmark adapters accept `--embedding-dimensions` and fail before a paid model
call when the supplied value does not match Oracle.

## Operational boundaries

- Keep `memory_id`, `user_id`, and `thread_id` explicit in multi-tenant jobs.
- Treat model output as a hypothesis; only host/application evidence is
  verified.
- Keep the evidence token budget materially below the model context window.
- Use cache invalidation for changing data and forgetting for obsolete
  compiled projections.
- Review plans before applying tombstones; use `delete_scope` only for a real
  lifecycle deletion request.
- Monitor `agent.learning_report(...)`, `agent.observability_summary(...)`,
  semantic-cache stats, compiler errors, and skill drift together.
