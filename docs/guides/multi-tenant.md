# Multi-Tenant Applications

Memorizz supports multi-tenant use cases — applications that serve a user
base where every end user must have an isolated memory surface. The scope
model is layered:

```
user_id  →  agent_id  →  memory_id  →  thread_id
```

Each level below `user_id` is agent-level infrastructure. `user_id` is the
tenant boundary that the framework enforces on every memory read and write.

## When to use `user_id`

Pass `user_id` whenever one deployed agent serves many end-users. Typical
examples:

- A chat product where each signed-in user has their own conversation history
- A B2B SaaS where memory must never cross customer boundaries
- A classroom assistant with per-student context

For single-operator local development (the default `memorizz ui` experience),
you can leave `user_id` out entirely — the framework falls back to the
anonymous/legacy scope.

## API

`user_id` is an **invocation** parameter, not a builder parameter. One
`MemAgent` instance can safely serve every user in your application:

```python
from memorizz.memagent import MemAgent

agent = MemAgent.load(agent_id, memory_provider=provider)

# Alice's turn
agent.run("Remember that my favorite color is purple.", user_id="alice")

# Bob's turn — same agent, isolated memory
agent.run("What's my favorite color?", user_id="bob")
# → Bob's response will NOT reference Alice's purple
```

The same parameter is accepted by `run_stream()`:

```python
for chunk in agent.run_stream(query, user_id=current_user_id):
    yield chunk
```

Pass `memory_id` and `thread_id` on every request in a shared deployment. The
agent's per-turn IDs, routed-tool budget, stream callback, and cache scope are
context-local, so one singleton can safely serve concurrent threads:

```python
for chunk in agent.run_stream(
    query,
    user_id=current_user_id,
    memory_id=f"primary_{current_user_id}",
    thread_id=conversation_id,
    event_callback=emit_event,
):
    yield chunk
```

Prefer the per-call `event_callback` over changing a singleton callback just
before a request.

## Isolation semantics

The framework guarantees the following:

1. **Writes carry `user_id`.** Every memory unit created during a turn —
   conversation history, summaries, entity memory, workflow steps, tool
   logs, semantic cache entries — is tagged with the `user_id` passed into
   `run()`.

2. **Reads are strictly scoped.** When a read is issued for `user_id=X`, the
   provider only returns rows whose stored `user_id == X`. There is **no
   "match everything" fallback** on the public API.

3. **`None` is a separate agent scope.** Passing `user_id=None` (or omitting
   it) to `run()`, `run_stream()`, or other agent APIs means
   "anonymous/legacy scope" — reads return only rows with no `user_id`.
   Legacy data written before you adopted multi-tenant support does **not**
   silently leak into authenticated sessions.

Low-level provider read APIs that accept `user_id` use a three-state
administrative contract: omitting the keyword means *unscoped*, explicitly
passing `None` selects anonymous rows, and passing a string selects that
tenant. Never omit the provider filter in a request-serving path; the unscoped
form exists for migration, audit, and maintenance jobs. Agent internals always
pass their scope explicitly.

## What is NOT scoped by `user_id`

`user_id` scopes *per-user memory*. It does **not** scope:

- `MemAgentModel` — agent definitions are shared app configuration.
- Personas, toolbox entries, MCP server configs, skills.
- `SHARED_MEMORY` — intentionally cross-agent/cross-tenant.
- Automation jobs — these are operator-level schedules.

If your use case needs per-user personas or per-user automations, that is a
separate feature; open an issue before relying on either.

## Provider behavior

| Provider   | Tenant scoping |
|------------|----------------|
| Oracle     | Server-side filter pushed into every read path, including vector search. Requires running the `migrations/001_add_user_id.sql` migration on existing schemas. |
| MongoDB    | Server-side filter on every `find()` and on the Atlas `$vectorSearch.filter` block. Add `user_id` to your Atlas vector index definitions for the conversation, summaries, semantic cache, workflow, and entity collections. |
| Filesystem | Strict client-side filter in every read method. The local `index.json` files now include `user_id` so enumeration stays cheap. |

### Oracle migration (required for existing installs)

Run once, as the Memorizz schema owner:

```bash
sqlplus MEMORIZZ/<password>@<service> \
    @src/memorizz/memory_provider/oracle/migrations/001_add_user_id.sql
```

The migration is additive: all new columns are nullable and existing rows
default to `user_id = NULL` (the legacy/anonymous scope). Those rows remain
available only to callers that explicitly request `user_id=None`; authenticated
reads never adopt them implicitly.

After verifying ownership, an operator can adopt entity-memory rows from one
legacy memory scope explicitly:

```python
from memorizz import EntityMemory

migrated = EntityMemory(provider).migrate_legacy_scope(
    memory_id="primary-user-a",
    user_id="user-a",
)
print(f"migrated {migrated} entity rows")
```

Run this as an administrative migration with an exact `memory_id`, never from
model-generated arguments or a request-serving code path. Filesystem performs
the change under its provider write lock; MongoDB and Oracle use one database
update.

### MongoDB Atlas vector index update

Writable MongoDB providers reconcile `user_id` as a filterable field on the
indexes below. The database principal must have search-index update permission;
otherwise MemoRizz records one diagnostic, keeps the reconciliation retryable,
and uses strict bounded exact fallback.

- `conversation_memory`
- `summaries`
- `semantic_cache`
- `workflow_memory`
- `entity_memory`

Without this change, `$vectorSearch` queries that include a `user_id` filter
will error or silently return empty result sets.

## Local UI and `user_id`

The bundled local UI (`memorizz ui`) is a single-operator developer tool.
If your production app embeds the UI and proxies requests server-side, the
`POST /agents/{agent_id}/run` and `POST /agents/{agent_id}/playground/stream`
endpoints accept an optional `user_id` form field that is forwarded to
`MemAgent.run` / `run_stream`. The UI itself does not surface a per-request
`user_id` input — exposing one on a localhost dev tool would be misleading
theater rather than real tenancy.

## Checklist for multi-tenant deployments

- [ ] Run the Oracle migration (if using Oracle).
- [ ] Update Atlas vector indexes to mark `user_id` filterable (if using MongoDB).
- [ ] Always pass `user_id` through `run()` / `run_stream()` at every request.
- [ ] Explicitly migrate any intended anonymous entity rows before authenticated
      traffic; never depend on an implicit legacy fallback.
- [ ] Never mix `user_id=None` traffic with authenticated traffic on the same
      agent instance — the two scopes are correctly isolated, but mixing
      makes auditing harder.
- [ ] When deleting a user, query all user-scoped memory types for that
      `user_id` and delete explicitly. A framework-level `delete_user_data()`
      cascade is intentionally not provided yet (cascade semantics for shared
      agents, personas, and automations are deployment-specific).
