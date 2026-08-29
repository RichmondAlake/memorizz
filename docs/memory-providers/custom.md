# Bring Your Own Memory Provider

Implement a custom provider when memory must live in an existing data platform.
`MemoryProvider` is a behavioral contract, not only a CRUD interface: tenant
filtering, ordering, stable identifiers, JSON fidelity, and lifecycle behavior
must match the first-party providers.

## Required surface

A concrete subclass implements these abstract methods:

| Area | Methods |
|---|---|
| Records | `store`, `retrieve_by_query`, `retrieve_by_id`, `retrieve_by_name` |
| Mutation | `update_by_id`, `delete_by_id`, `delete_by_name`, `delete_all` |
| Listing/history | `list_all`, `retrieve_conversation_history_ordered_by_timestamp` |
| Agent definitions | `store_memagent`, `delete_memagent`, `update_memagent_memory_ids`, `delete_memagent_memory_ids`, `list_memagents` |
| Lifecycle | `close` |

Read the live signatures in the [Python API reference](../reference/python-api.md).
Optional base implementations cover bounded observability queries and semantic
cache invalidation; database providers should override them with indexed,
provider-native operations.

## Minimal shape

```python
from typing import Any

from memorizz import MemoryProvider


_USER_FILTER_UNSET = object()


class PostgresProvider(MemoryProvider):
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.connection = open_application_connection(self.config)

    def list_all(
        self,
        memory_store_type: str,
        user_id: Any = _USER_FILTER_UNSET,
    ) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"memory_type": str(memory_store_type)}
        if user_id is not _USER_FILTER_UNSET:
            filters["user_id"] = user_id
        return self._select(filters=filters)

    def retrieve_by_query(
        self,
        query: dict[str, Any] | str,
        memory_store_type: str | None = None,
        limit: int = 1,
        memory_id: str | None = None,
        memory_type: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        resolved_type = memory_type or memory_store_type
        user_supplied = "user_id" in kwargs
        user_id = kwargs.get("user_id")
        return self._scoped_search(
            query=query,
            memory_type=resolved_type,
            memory_id=memory_id,
            user_supplied=user_supplied,
            user_id=user_id,
            limit=max(1, int(limit)),
        )
```

This excerpt demonstrates scope handling; it is intentionally not a complete
provider. Do not copy it without implementing every abstract method, resource
cleanup, transactions, indexes, serialization, and error mapping.

## Three-state tenant filters

Provider reads use three distinct states:

| Call | Meaning |
|---|---|
| omit `user_id` | Administrative/unscoped read |
| pass `user_id=None` | Only anonymous/legacy records |
| pass a string | Only that exact tenant |

Use your own private sentinel; do not import Memorizz's private `_UNSET`
object. For semantic retrieval, apply tenant, agent, toolbox, status, and other
authorization filters **before** vector top-k selection. Post-filtering a
global top-k can return too few valid results and can expose cross-tenant
metadata.

## Record fidelity

- Preserve `_id`/`id` as a stable string and do not silently regenerate it on
  update.
- Round-trip `memory_id`, `user_id`, `agent_id`, `thread_id`, `memory_type`,
  timestamps, metadata, and embeddings when present.
- Include `memory_id` and `user_id` on every semantic-retrieval result. MemAgent
  rechecks both fields after the provider call and discards rows that omit or
  mismatch the authenticated scope before prompt assembly.
- Store complete toolbox JSON Schemas, including `required`, defaults, enums,
  nested types, and `additionalProperties`.
- Preserve summary provenance and source-message markers so expansion is
  lossless.
- Return JSON-serializable dictionaries and normalize database-native IDs,
  LOBs, timestamps, and vectors.
- Keep ordered history deterministic and enforce `limit` after exact scope
  filtering.

## Production extensions

Consider provider-native implementations for:

- atomic summary creation plus source-message marking;
- scoped observability pages and cursors;
- domain/version semantic-cache invalidation;
- pre-filtered toolbox and Skillbox retrieval;
- transactional scope cleanup with per-store counts;
- `preflight()` reporting schema, permissions, vector dimensions, indexes, and
  exact-search fallback state.

## Conformance checklist

1. Run the same CRUD, history, cache, summary, agent-persistence, and
   observability tests used for the filesystem provider.
2. Test omitted, `None`, and concrete `user_id` values for every public read.
3. Test two tenants with identical content and embeddings.
4. Test restart/rebind behavior for saved agents and trusted tools.
5. Test concurrent writers, transaction rollback, connection loss, and
   idempotent close.
6. Compare `capability_report()` and any provider `preflight()` result in the
   target deployment.

Attach the finished provider with
`MemAgentBuilder().with_memory_provider(provider)` and persist one agent before
checking SDK, CLI, UI, and MCP discovery against the same backend.
