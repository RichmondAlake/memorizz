# Bring Your Own Provider

MemoRizz decouples the high-level memory interfaces from the backing database via the `MemoryProvider` base class (`src/memorizz/memory_provider/base.py`). Implementing your own provider lets you plug in any datastore that can persist JSON blobs plus embeddings.

## Steps

1. **Subclass `MemoryProvider`** and implement CRUD helpers for each memory bucket you care about (personas, knowledge base, etc.).
2. **Handle embeddings** – either pre-compute embeddings before storing documents or call the shared embedding registry inside your provider methods.
3. **Respect schemas** – store the `id`, `agent_id`, `memory_type`, `data`, `embedding`, and timestamps so higher layers can filter and audit records consistently.
4. **Register the provider** – pass an instance to `MemAgentBuilder().with_memory_provider(...)`.

```python
from memorizz.memory_provider.base import MemoryProvider

class PostgresProvider(MemoryProvider):
    def save_persona(self, persona):
        ...
```

## Testing Checklist

- Run the provider's unit tests under `pytest tests/memory_provider/test_<name>.py`.
- Use `mkdocs serve` to confirm your new provider docs appear under **Memory Providers**.
- Update `pyproject.toml` with a matching optional extra if you ship new dependencies.

Custom providers make it easy to align MemoRizz with corporate infra while keeping the rest of the SDK untouched.
