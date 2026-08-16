# Oracle provider internals

This package contains MemoRizz's relational Oracle AI Database provider,
schema, migrations, and local-runtime helper. The user guide is
[`docs/memory-providers/oracle.md`](../../../../docs/memory-providers/oracle.md)
and the complete local setup is [`SETUP.md`](../../../../SETUP.md).

## Contents

- `provider.py` — pooled provider, in-database/external embeddings, vector
  search/index policy, preflight, migrations-on-startup, and scoped cleanup;
- `runtime.py` — bounded Docker lifecycle through the same injection-safe
  implementation used by the local UI;
- `schema_relational.sql` — current install schema;
- `migrations/` — ordered upgrade artifacts, including 0.5 production
  governance and summary-compaction parity;
- `setup.py` — explicit admin/user-only schema bootstrap with no password
  defaults or credential output.

## Minimal use

```python
from memorizz.memory_provider.oracle import OracleProvider

provider = OracleProvider.from_env(index_policy="lazy")
report = provider.preflight()
assert report["ok"], report["diagnostics"]
```

Required environment variables are `ORACLE_USER`, `ORACLE_PASSWORD`, and
`ORACLE_DSN`. `ORACLE_ADMIN_PASSWORD` is never required by the normal provider.

## Index policy

- `none`: exact vector distance only;
- `lazy`: create on first use;
- `selected`: create for `selected_vector_indexes`;
- `eager`: attempt every enabled vector index on initialization.

Every policy retains exact-search fallback. CPU/vector-memory sizing and index
availability are reported by `preflight()`.

## Schema contract in 0.5

Migration `004_production_governance_050.sql` adds complete Toolbox JSON
schemas, canonical summary source IDs/period/count fields, conversation
summary markers, the normalized ordered `summary_message_links` table, and
semantic-cache metadata. Provider startup checks are additive and idempotent,
but operators should still apply migration files under change control.

Migration `005_scoped_retrieval_052.sql` adds and backfills the summary
`thread_id`, restores knowledge-base chunk/namespace columns on older schemas,
and creates the indexes used by exact thread and namespace retrieval in 0.5.2.

## Cleanup

`delete_scope(memory_id=..., user_id=..., agent_ids=[...])` requires at least
one exact scope, deletes related package-owned records transactionally, and
returns per-table counts. It is intended for test/course lifecycle cleanup,
tenant erasure, and controlled agent teardown—not broad unscoped deletion.
