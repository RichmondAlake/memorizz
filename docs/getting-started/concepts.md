# Concepts

Memorizz composes agent behavior from memory types, storage providers, and application modes.

## Memory Types

`MemoryType` is defined in `src/memorizz/enums/memory_type.py`.

| Enum | Purpose | Main Implementation |
|---|---|---|
| `KNOWLEDGE_BASE` | Semantic facts and reusable knowledge | `src/memorizz/long_term/semantic/` |
| `ENTITY_MEMORY` | Structured entity profiles and attributes | `src/memorizz/long_term/semantic/entity_memory/` |
| `TOOLBOX` | Callable tools and tool metadata | `src/memorizz/long_term/procedural/toolbox/` |
| `WORKFLOW_MEMORY` | Process and task execution traces | `src/memorizz/long_term/procedural/workflow/` |
| `CONVERSATION_MEMORY` | User/assistant interaction history | `src/memorizz/long_term/episodic/` |
| `SUMMARIES` | Compressed conversation summaries | `src/memorizz/long_term/episodic/summary_component.py` |
| `SHORT_TERM_MEMORY` | Working session context | `src/memorizz/short_term_memory/working_memory/` |
| `SEMANTIC_CACHE` | Similar-query response caching | `src/memorizz/short_term_memory/semantic_cache.py` |
| `SHARED_MEMORY` | Multi-agent coordination state | `src/memorizz/coordination/shared_memory/` |
| `MEMAGENT` | Persisted agent configuration | `src/memorizz/memagent/models.py` |

## Providers vs Memory Types

- **Memory types** define what data is stored.
- **Providers** define where data is stored (filesystem, Oracle, MongoDB, custom).
- **Application modes** choose a default combination of memory types.

## Application Modes

Mode defaults come from `src/memorizz/enums/application_mode.py`.

- `assistant`: conversation, long-term, personas, entity memory, short-term, summaries
- `workflow`: workflow memory, toolbox, long-term, short-term, summaries
- `deep_research`: toolbox, shared memory, long-term, short-term, summaries

You can still override with explicit `memory_types` if your use case needs a custom stack.

## Typical Runtime Lifecycle

1. Agent receives a query.
2. Relevant memory is retrieved from active memory types via the configured provider.
3. LLM produces a response (and may call registered tools).
4. Interaction is written back to memory stores.
5. Optional semantic cache and summary logic optimize future turns.
