# MemoRizz Zero to Hero

These notebooks teach agent memory with the current MemoRizz SDK. They are
standalone and default to a deterministic local teaching model. The filesystem
lessons write only to isolated temporary directories; the Oracle edition uses
the configured local Oracle AI Database and deletes its tutorial scope after a
successful run. Set `MEMORIZZ_TUTORIAL_LLM=openai` to replace the teaching
model with a hosted OpenAI model.

| Notebook | Purpose |
|---|---|
| `agent_memory.ipynb` | Build one memory-first engineering copilot layer by layer: episodic, semantic, procedural, cache, compaction, shared memory, and observability. |
| `agent_memory_oracle.ipynb` | Run the same memory-first lifecycle on a real local Oracle AI Database, including runtime readiness, preflight, vector validation, and transactional scoped cleanup. |
| `memory_types.ipynb` | Inspect every `MemoryType`, its unit shape, ownership model, and supported create/read/update/delete or runtime operations. |

## Recommended learning path

1. Start with `agent_memory.ipynb`. It develops the mental model from a
   stateless model to a complete memory-first agent using a zero-cost,
   deterministic filesystem environment.
2. Use `memory_types.ipynb` as the detailed field guide. Each memory type is
   explained through its purpose, unit shape, authoritative writer, retrieval
   path, lifecycle, failure modes, output evidence, and selection rules.
3. Run `agent_memory_oracle.ipynb` to move the same contracts onto a real
   Oracle AI Database. It adds Docker/runtime readiness, database preflight,
   vector-dimension validation, transactional behavior, operational evidence,
   and scoped cleanup.

The notebooks distinguish a runnable integration demonstration from an
evaluation claim. Deterministic assertions prove persistence and lifecycle
mechanics; retrieval quality, grounded answer accuracy, latency, tokens, and
cost must still be measured on a representative evaluation set.

## Run locally

From the repository root:

```bash
source .venv/bin/activate
jupyter lab examples/zero_to_hero
```

Select the repository virtual environment as the kernel and run the cells from
top to bottom. The filesystem notebooks make no network requests and incur no
model cost. For a hosted reasoning run:

```bash
export OPENAI_API_KEY="..."
export MEMORIZZ_TUTORIAL_LLM=openai
export MEMORIZZ_TUTORIAL_MODEL=gpt-5-mini
```

Credentials are read from the environment and are never written into notebook
cells or persisted agent configuration.

The Oracle notebook additionally requires a running local Oracle AI Database
and `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN`. Its default external
embedding lane uses `text-embedding-3-small` at 384 dimensions, so it also
requires `OPENAI_API_KEY`; set `MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING=1` to use
the configured Oracle ONNX embedding model instead. If your persisted VECTOR
columns use a different dimension, set
`MEMORIZZ_TUTORIAL_EMBEDDING_DIMENSIONS` to that preflight value. Set
`MEMORIZZ_TUTORIAL_KEEP_DATA=1` only when you deliberately want to inspect the
created Oracle rows after the notebook finishes.

## Why there are two agent-memory notebooks

The filesystem edition is the fastest path for learning and CI. The Oracle
edition keeps the same application-level scope and memory APIs while adding
runtime readiness, schema/vector validation, transactional persistence, and
scoped cleanup:

```python
from memorizz import OracleProvider

provider = OracleProvider.from_env(index_policy="lazy")
print(provider.preflight())
```

There is no silent fallback between them. An Oracle configuration failure is a
real failure in the Oracle notebook, which keeps integration results honest.
