# MongoDB Provider

The MongoDB provider offers a lightweight starting point for experimentation or hosted Atlas deployments. It is implemented in `src/memorizz/memory_provider/mongodb/`.

## Installation

```bash
pip install -e ".[mongodb]"
```

## Configuration

```python
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig

provider = MongoDBProvider(MongoDBConfig(
    uri=os.environ["MONGODB_URI"],
    db_name="memorizz",
    lazy_vector_indexes=True,
))
```

Collections are created lazily (e.g., `agents_personas`, `agents_knowledge_base`). Each document stores:

- Serialized payload (`data`)
- Embedding vectors (array fields you can index with MongoDB Atlas Vector Search)
- Agent + namespace metadata

## Atlas Vector Search

1. Enable Atlas Vector Search on your cluster.
2. Configure the provider with your embedding model dimensions.
3. Give the provider search-index management permission, or provision the
   indexes separately.

MemoRizz reconciles the filter paths each vector index needs beyond
`memory_id`: `user_id` for entity, workflow and summary retrieval;
`knowledge_base_id`, `namespace` and `user_id` for knowledge-base retrieval;
and `status`, `agent_id` and `user_id` for Skillbox retrieval. Filters are
applied inside `$vectorSearch` before top-k selection. Adding a path updates an
existing index in place, and Atlas rebuilds it.
If Atlas Search is missing or unavailable, entity memory uses a strict bounded
exact fallback. Query failures are reported as degraded retrieval rather than
healthy zero-match results.

Knowledge-base retrieval follows the same production-safe principle for
MetaHarness and learning-control-plane evidence. MemoRizz first applies exact
`memory_id`, `user_id`, and optional namespace filters. If `$vectorSearch` is
not available (for example MongoDB Community, local Docker, or an Atlas tier
without Search), it examines at most 200 rows inside that scope, ranks them
lexically, and labels every result `scoped_lexical_fallback` with a degraded
reason. It never turns a vector failure into an unscoped collection scan.

Both `provider.store(data, ..., memory_id="...")` and a `memory_id` embedded in
the data now persist the same knowledge-base scope. This parity matters for
shared MetaHarness evidence snapshots: later panel stages can reuse the first
bounded snapshot without another MongoDB query.

## Delete a scope

`provider.delete_scope(memory_id=..., user_id=..., agent_ids=[...])` deletes one
scope with a `delete_many` per collection. It is not a transaction. Pass
`keep_traces=True` to keep the scope's trace records. See
[Delete a scope](scoped-cleanup.md).

## When to Choose MongoDB

- Prototype agents without running Oracle locally
- Serverless / hosted deployments where MongoDB Atlas is already approved
- Horizontal scaling scenarios using MongoDB's built-in sharding

Use MongoDB for agility and switch to Oracle when you need stronger relational guarantees or AI Vector Search optimizations.
