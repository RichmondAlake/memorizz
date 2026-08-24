# Filesystem Provider

The filesystem provider persists every MemoRizz memory type as JSON files on disk and uses FAISS for vector similarity search. It is ideal for local development, CI runs, or lightweight deployments where running MongoDB/Oracle would be overkill.

## Highlights

- No external database required—everything lives under the configured root directory.
- Works with the exact same `MemoryProvider` API as Oracle/MongoDB, so agents can swap providers without code changes.
- Optional FAISS acceleration for semantic queries with automatic fallbacks to cosine or keyword search when embeddings are missing.

## Installation

```bash
pip install memorizz[filesystem]
```

This installs `faiss-cpu`. If you skip the extra, the provider still works but falls back to keyword search until FAISS (and an embedding provider) are available.

## Configuration

Filesystem memory is the SDK default. A bare agent stores data beneath
`~/.memorizz/memory`:

```python
from memorizz import MemAgent

agent = MemAgent(name="durable-agent")
```

Set `MEMORIZZ_HOME` to relocate all MemoRizz state, or set
`MEMORIZZ_MEMORY_ROOT` to relocate only memory data. For an intentionally
stateless agent, pass `memory_provider=False`; `memory_types=[]` disables active
memory retrieval but retains the provider for agent/config persistence.

Pass an explicit provider when you need a different root or embedding setup:

```python
from pathlib import Path
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

config = FileSystemConfig(
    root_path=Path("~/.memorizz").expanduser(),  # Each MemoryType gets its own folder
    lazy_vector_indexes=True,                    # Build FAISS indexes on demand
    use_faiss=True,                              # False uses exact cosine search
    embedding_provider="openai",                 # Optional, enables semantic search
    embedding_config={"model": "text-embedding-3-small"},
)

provider = FileSystemProvider(config)
```

- `root_path` is the only required field. The provider creates subdirectories named after each `MemoryType`.
- Set `lazy_vector_indexes=True` to skip vector index builds until a semantic query hits a store.
- Set `use_faiss=False` for exact cosine search. This is useful for small
  stores and processes where FAISS would conflict with another native OpenMP
  runtime; retrieval remains semantic but trades index speed for a linear scan.
- You can also pass a fully constructed `EmbeddingManager` instance via `embedding_provider` for complete control.

## Storage Layout

```
~/.memorizz/memory/
├── conversation_memory/
│   ├── index.json                # Lightweight metadata for quick lookups
│   ├── 4c1d9a2f.json             # Individual memory documents
│   └── vector.index (optional)   # Saved FAISS index when embeddings are enabled
├── long_term/
│   └── …
└── agents/                       # Stored MemAgent configurations
```

Each JSON file contains the raw document plus MemoRizz metadata (`_id`, `memory_id`, timestamps, embeddings, etc.). When FAISS is installed, the provider builds an in-memory index and snapshots it to `vector.index` for fast restarts.

## Usage Tips

- **Embeddings optional**: If you only need deterministic lookups (ID/name filters), skip embedding configuration and the provider will stick to metadata filtering/keyword search.
- **Backups**: Because everything is plain JSON, standard tools (`tar`, `rsync`, cloud sync) can back up or relocate memory stores easily.
- **Cleanup**: Call `delete_memagent(..., cascade=True)` to remove all memories tied to an agent (the provider deletes the related JSON files).

For in-depth details, see `src/memorizz/memory_provider/filesystem/provider.py`.
