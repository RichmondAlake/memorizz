# Forgetting mechanism

MemoRizz forgets the way *Generative Agents* (Park et al., 2023) do, with one
addition. Memories are never deleted by the mechanism: they sink in the
retrieval ranking as they age unused, and a governed, reversible
**suppression** tier can hide records that would otherwise distort retrieval.
Audit evidence, verified outcomes and pinned records are never touched.

## Tier 0: ranking decay (always on)

Every candidate retrieved for a turn is scored before it enters the prompt:

```text
recency    = decay ** hours_since_last_access      (decay 0.995 by default)
importance = stored 0..1 rating (default 0.5 when a record has none)
relevance  = cosine to the query, or the provider score
score      = (a_recency * recency + a_importance * importance + a_relevance * relevance) / (sum of weights)
```

Each term is min-max normalised over the candidate set, exactly as in the
paper, and the weights default to 1, 1, 1. Recency is anchored on the memory's
**last access** (`last_accessed_at`) and falls back to its creation time, so a
memory that keeps being recalled decays slowly and one that is never used
sinks. Diversity selection (maximal marginal relevance) runs afterwards.
Selected memories carry a `scoring` block with the normalised and raw
components, which retrieval diagnostics expose.

Global defaults come from environment settings (the Settings page writes
them): `MEMORIZZ_RECENCY_DECAY_PER_HOUR`, `MEMORIZZ_RECENCY_ANCHOR`
(`last_accessed` or `created`), `MEMORIZZ_ALPHA_RECENCY`,
`MEMORIZZ_ALPHA_IMPORTANCE`, `MEMORIZZ_ALPHA_RELEVANCE`,
`MEMORIZZ_DEFAULT_IMPORTANCE` and `MEMORIZZ_SCORE_NORMALIZE`. Per-agent
overrides live on the agent's retrieval policy under `scoring` and are edited
in the agent form's **Forgetting (per-agent overrides)** group; blank fields
inherit the global value.

```python
from memorizz.memagent.utils.context_dedup import RetrievalScoring, dedupe_and_select

selected = dedupe_and_select(
    candidates,
    query_embedding=query_vector,
    scoring=RetrievalScoring(alpha_importance=2.0, recency_anchor="last_accessed"),
)
```

## Importance ratings

The paper rates each memory's "poignancy" from 1 to 10 at creation. MemoRizz
stores the rating as `importance` in `[0, 1]` and offers three rater modes
(`MEMORIZZ_IMPORTANCE_RATER`):

| Mode | Behaviour |
|---|---|
| `off` | Nothing is stored; retrieval uses the default importance. |
| `heuristic` | Deterministic and free: host-asserted facts and verified outcomes high, tool output low, decisions and corrections boosted. |
| `llm` | The paper's prompt, answered by a small or local model (`MEMORIZZ_IMPORTANCE_MODEL`), batched and run off the user path, with the heuristic as fallback. |

Ratings are immutable once stored; only a verified outcome raises one to 1.0.

```python
from memorizz.memagent.utils.importance import ImportanceConfig, ImportanceRater

rater = ImportanceRater(ImportanceConfig(mode="llm", purpose="tracking a product release"), generate=model.generate_text)
importance, source = rater.rate("The release owner changed to Mira", role="user")
```

## Reinforcement and reflection

Whenever a memory is selected into a prompt, the agent records the recall off
the user path (`touch_many`): `last_accessed_at` moves to now and
`access_count` increments, so recency is measured since last use as in the
paper. New conversation rows are rated at store time and their importance
accumulates per conversation; when `MEMORIZZ_REFLECTION_ENABLED` is on and the
sum passes `MEMORIZZ_REFLECTION_THRESHOLD` (default 15.0 on the 0 to 1 scale,
the paper's 150 on 1 to 10), the agent runs one summarisation pass over that
conversation, the paper's reflection step, and records it in the turn trace.

## Tier 1: governed suppression (plan, approve, restore)

A **retention planner** scores primary memory records (conversation,
knowledge base, entities, summaries and workflows) with

```text
retention = mean(importance, usage, recency)
usage     = log1p(access_count) / log1p(max access_count in the set)
```

and lists the records whose retention stayed below `min_retention` for at
least `grace_days` since creation, unless they are pinned, verified, already
suppressed, or cited as a source by another record. The plan is a dry run with
a content-hashed id; a named approver applies it; each record receives
`retention_state = "suppressed"` with the plan id, reason and approver. Every
retrieval path honours that state, the memory history journal records the
change with the approver as writer and `forgetting:<plan-id>` as source, and
any suppression can be reversed.

Settings: `MEMORIZZ_RETENTION_ENABLED`, `MEMORIZZ_RETENTION_MIN_SCORE`,
`MEMORIZZ_RETENTION_GRACE_DAYS`, `MEMORIZZ_RETENTION_PROTECT_VERIFIED`,
`MEMORIZZ_RETENTION_PROTECT_PINNED`, `MEMORIZZ_RETENTION_MAX_CANDIDATES` and
`MEMORIZZ_RETENTION_MEMORY_TYPES`. Per-agent overrides sit on the retrieval
policy under `retention`.

```python
plan = agent.plan_memory_retention(memory_id=memory_id)      # dry run
applied = agent.apply_memory_retention(plan, approved_by="operator-7", reason="quarterly tidy")
agent.suppressed_memories(memory_id=memory_id)               # review
agent.unsuppress_memory(record_id, "knowledge_base", approved_by="operator-7")
```

With the learning control plane enabled these calls go through
`plane.plan_retention` / `plane.apply_retention` and leave durable
`forgetting_planned` / `memory_forgotten` events; without it the agent runs the
planner directly against its provider. `agent.retrieval_scoring` and
`agent.retention_config` expose the resolved settings (global defaults overlaid
with the agent's `retrieval_policy` overrides).

From the command line:

```sh
memorizz learning retention-plan --agent-id AGENT --memory-id MEMORY --json
memorizz learning retention-apply PLAN_ID --agent-id AGENT --approved-by operator-7
memorizz learning suppressed --agent-id AGENT
memorizz learning unsuppress RECORD_ID --memory-type knowledge_base --agent-id AGENT --approved-by operator-7
```

In the UI, the **Learning control plane** page has *Plan memory retention
(dry run)*, an approval panel for the resulting plan, a *Suppressed memories*
table and a *Restore* action per record. Hard deletion is unchanged: it stays
an exact-scope, approval-gated lifecycle operation.

## Semantic cache

The semantic cache keeps its own time-to-live (`ttl_hours`, default 24) and
per-domain freshness windows. Expired entries are never served, expired rows
are purged from the provider on preload and periodically on writes, and
re-caching the same query updates the existing row instead of duplicating it.

## Provider parity

All fields (`importance`, `last_accessed_at`, `access_count`,
`retention_state`) are plain record fields, so the scoring and the planner
behave the same on the filesystem, MongoDB, Oracle and Notion providers. The
learning-plane store uses indexed observability queries where a provider
offers them and never scans the shared-memory partition on per-turn paths.
