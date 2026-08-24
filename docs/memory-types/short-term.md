# Short-Term Memory and Semantic Cache

Short-term memory is runtime-managed working state for the current task. It is
not a public mutable message buffer. Memorizz assembles a bounded model context
from the current request, recent history, retrieved long-term memory, tools,
and summaries.

## Inspect context use

```python
agent.run(
    "Give me the highlights from yesterday's sync.",
    memory_id="team-assistant",
    user_id="user-42",
    thread_id="daily-sync",
)

stats = agent.get_context_window_stats()
usage = agent.model.get_last_usage()
print(stats)
print(usage)
```

Use a `ContextPolicy` to set retrieval/tool budgets and the context-efficiency
guide to tune stable prefixes, deduplication, summaries, and compaction.

## Semantic cache

The semantic cache can avoid a model call when a sufficiently similar,
in-scope, fresh response is available. It records real hit, miss, bypass,
write, eviction, and size counters.

```python
stats = agent.semantic_cache_stats()

inspection = agent.inspect_semantic_cache(
    "What is our return policy?",
    user_id="user-42",
    thread_id="returns",
    context={"cache_domains": ["policy"], "data_version": "2026-08-21"},
)

removed = agent.invalidate_semantic_cache(
    domains=["policy"],
    data_version="2026-08-20",
)
```

Inspection reports match provenance, similarity, age, TTL, hit count,
fingerprints, invalidation domains, and any bypass reason without exposing the
cached answer. Side-effecting and non-deterministic tool candidates bypass
admission by default.

!!! warning "Similarity is not freshness"
    A semantically similar answer can still be operationally stale. Define
    domain TTLs and data versions, invalidate entries when source data changes,
    and keep mutation/tool responses out of the cache.

See [Context Efficiency and Prompt Caching](../guides/context-efficiency.md)
for configuration and measurement guidance.
