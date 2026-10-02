# Oracle AI Database Provider

The Oracle provider stores every MemoRizz memory partition in relational
tables, supports Oracle AI Vector Search, and can generate embeddings inside
the database with an ONNX model. MemoRizz 0.5 adds structured bootstrap and
preflight, explicit index policies, exact-search fallback, summary-compaction
parity, governed cache metadata, and transactional scoped cleanup.

## Install and configure

```bash
pip install "memorizz[oracle]"
cp .env.example .env
```

Set unique secrets; MemoRizz has no database-password defaults:

```dotenv
MEMORIZZ_BACKEND=oracle
ORACLE_USER=memorizz_user
ORACLE_PASSWORD=<application-password>
ORACLE_DSN=localhost:1521/FREEPDB1
```

`ORACLE_ADMIN_PASSWORD=<admin-password>` is needed only when MemoRizz must
create or grant the application schema. Normal application processes should
receive only `ORACLE_USER`, `ORACLE_PASSWORD` and `ORACLE_DSN`.

### Start Oracle locally

With Docker (or OrbStack) running, set the admin password in the environment
or enter it at the hidden prompt:

```bash
memorizz oracle install --image lite
memorizz oracle setup
```

Neither command echoes a password. `oracle setup` works in two modes:

- **admin mode** creates the configured application user, grants the required
  privileges and initializes the schema;
- **user-only mode** connects to an already-provisioned schema and initializes
  objects without administrator access.

For an existing schema, use the non-destructive update instead:

```bash
memorizz oracle setup-schema
```

Applications and notebooks can use the same bootstrap from the SDK:

```python
from memorizz.memory_provider.oracle import LocalOracleRuntime, OracleProvider

runtime = LocalOracleRuntime.from_env(provision_if_missing=True)
runtime.ensure_ready()
provider = OracleProvider.from_env(provision_if_missing=False, index_policy="lazy")
assert provider.preflight()["ok"]
```

`LocalOracleRuntime` uses explicit Docker argument arrays, the requested host
port and container name, and a bounded readiness wait. Set
`MEMORIZZ_ORACLE_CONTAINER` when reusing a non-default container.

### Hosted Oracle

Set the application credentials and DSN, leave out `ORACLE_ADMIN_PASSWORD`, and
run `memorizz oracle setup-schema`. Ask the DBA for the privileges reported by
`preflight()`, and for permission to install an ONNX model if in-database
embedding is required.

## Recommended construction

```python
from memorizz.memory_provider.oracle import OracleProvider

provider = OracleProvider.from_env(
    provision_if_missing=False,
    index_policy="lazy",
)

report = provider.preflight()
if not report["ok"]:
    raise RuntimeError(report["diagnostics"])
```

For a complete agent preset that readies the local runtime and retains the
preflight report:

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_oracle_from_env(
        ensure_ready=True,
        provision_if_missing=True,
        index_policy="lazy",
    )
    .build()
)
report = agent.environment_reports["oracle"]["preflight"]
```

`OracleProvider.from_env(provision_if_missing=True)` can create/start the local
package-owned Docker runtime before connecting. For explicit control:

```python
from memorizz.memory_provider.oracle import LocalOracleRuntime

runtime = LocalOracleRuntime.from_env(provision_if_missing=True)
runtime.ensure_ready()
```

## Configuration

```python
import os

from memorizz.enums import MemoryType
from memorizz.memory_provider.oracle import OracleConfig, OracleProvider

provider = OracleProvider(
    OracleConfig(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        schema=os.environ.get("ORACLE_SCHEMA"),
        in_database_embedding=True,
        index_policy="selected",
        selected_vector_indexes=[
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.KNOWLEDGE_BASE,
        ],
        pool_min=1,
        pool_max=8,
        pool_increment=1,
    )
)
```

| Setting | Default | Meaning |
|---|---:|---|
| `schema` | application user | Owner of MemoRizz tables |
| `in_database_embedding` | `True` | Use Oracle ONNX embedding when no external provider is supplied |
| `embedding_provider` | `None` | Optional external provider or injected embedding manager |
| `embedding_config` | `{}` | Model, dimensions, and provider options |
| `index_policy` | `lazy` | `none`, `lazy`, `selected`, or `eager` |
| `selected_vector_indexes` | `[]` | Memory types indexed under `selected` policy |
| `pool_min` / `pool_max` | `1` / `5` | Connection-pool bounds |

`lazy_vector_indexes` remains a compatibility argument; use `index_policy` in
new code.

## Embeddings and vector search

In-database embedding is the default. The standard local setup uses the
`ALL_MINILM_L12_V2` ONNX model with 384 dimensions. Override model and
dimension only when the installed model and every vector column match.

For an external provider:

```python
provider = OracleProvider(
    OracleConfig(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        in_database_embedding=False,
        embedding_provider="openai",
        embedding_config={
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
        index_policy="lazy",
    )
)
```

Index behavior:

- `none` never creates HNSW indexes and uses exact vector distance;
- `lazy` creates an index only when the associated memory type is used;
- `selected` limits indexes to configured memory types;
- `eager` attempts all enabled indexes at initialization.

An unavailable vector pool or index does not disable semantic retrieval:
MemoRizz falls back to exact search and emits one concise diagnostic.

## Structured preflight

```bash
memorizz oracle preflight --index-policy lazy --json
```

```python
report = provider.preflight()
```

The report covers the DSN/service, database product, compatibility
`database_version`, full Release Update `version_full`, PDB/open state, schema
privileges, embedding model/dimensions, vector-column
dimensions, `VECTOR_MEMORY_SIZE`, index status, exact-search fallback, and a
recommended vector-memory size.

`embedding_dimension_compatible` is `False` and `ok` is set to `False` when
the configured embedder differs from an existing `VECTOR` column. The report's
`embedding_dimension_mismatches` map identifies each incompatible column so a
builder preset fails before any partial write or `ORA-51803`.

An authorized administrator can request a persistent vector-memory change:

```python
provider.set_vector_memory_size(
    "1G",
    admin_user=os.environ["ORACLE_ADMIN_USER"],
    admin_password=os.environ["ORACLE_ADMIN_PASSWORD"],
)
# Restart Oracle after the SPFILE change.
```

Do not pass administrator credentials to the normal application process.

## 0.5 data parity

Oracle persists the same production metadata as the document providers:

- complete Toolbox JSON Schema, including `required`, defaults, enums, nested
  types, aliases, deprecated arguments, policy, and trusted import reference;
- summary `source_message_ids`, period bounds, unit count, conversation
  `summary_id`, and normalized ordered summary/message links;
- semantic-cache tenant/session/memory scope, fingerprints, freshness,
  provenance, tags, and invalidation metadata;
- workflow, skill, shared-memory, tool-log, and first-party MCP data required by
  the 0.5 runtime.
- immutable learning events, compiler checkpoints, compiled artifacts, and
  reversible forgetting tombstones as logical records in private shared
  memory; `list_all(shared_memory)` returns the complete typed projection.

Summary creation and original-message marking are transactional. Retrieval by
`summary_id` and `expand_summary()` reconstruct the linked source messages.

## Upgrade an existing schema

Back up first, then apply every migration that has not yet been applied to the
schema, in numeric order. The packaged migrations are:

```text
src/memorizz/memory_provider/oracle/migrations/001_add_user_id.sql
src/memorizz/memory_provider/oracle/migrations/002_add_skill_injection_role.sql
src/memorizz/memory_provider/oracle/migrations/003_add_shadow_evaluations.sql
src/memorizz/memory_provider/oracle/migrations/004_production_governance_050.sql
src/memorizz/memory_provider/oracle/migrations/005_scoped_retrieval_052.sql
src/memorizz/memory_provider/oracle/migrations/006_knowledge_base_provenance.sql
src/memorizz/memory_provider/oracle/migrations/007_structured_tool_outcomes.sql
src/memorizz/memory_provider/oracle/migrations/008_observability.sql
```

Migration 005 adds/backfills exact summary thread scope and restores indexed
knowledge-base namespace metadata on older schemas. Migration 006 preserves
source provenance for grounded knowledge records. Migration 007 adds structured
`outcome` and `outcome_details` fields to tool logs and backfills legacy rows;
it is idempotent and can be rerun safely. Migration 008 adds the private
observability index behind traces and usage; `OracleSpanIndex.initialize()`
applies it too. The provider performs additive startup checks for
availability, but the SQL files are the recommended review and change-control
artifacts.

## Scoped cleanup

```python
result = provider.delete_scope(
    memory_id="course-run-17",
    user_id="student-42",
    agent_ids=["planner", "executor"],
)
print(result["counts"], result["total_deleted"])
```

At least one exact scope is required. Conversations, caches, workflows,
tool logs, skills, summaries/links, shared memory, automations, and selected
agent records are deleted in one transaction, with per-table counts. A failure
rolls the transaction back.

## Operations

- pass `user_id` on every multi-tenant call;
- keep the provider pool process-local;
- monitor preflight diagnostics and cache/tool-log metrics;
- use `index_policy="none"` for small data sets or constrained vector memory;
- use `selected` for the memory partitions that actually require approximate
  search;
- close the provider during application shutdown, or use `with agent:` /
  `agent.lifecycle(...)` to close the complete runtime.

```python
with agent:
    agent.run("Remember this", user_id="tenant-a")
```

## Production checklist

- use a secret manager and separate admin and application credentials;
- keep `MEMORIZZ_HOME` and database storage durable;
- apply migrations under change control and back up before upgrades;
- choose an explicit index policy and monitor vector memory;
- pass `user_id` and enforce tenant identity at the application edge;
- call `agent.capability_report(preflight=True)` in deployment health checks;
- run `scripts/verify_production_050.py` against a staging database before
  promotion.

## Troubleshooting

**Connection fails:** confirm the mapped Docker port, service name, PDB state,
and application credentials. MemoRizz does not guess a default password.

**Dimension mismatch:** compare `preflight()["vector_dimensions"]` with the
embedding report. Align the model and schema before writing more vectors.

**ORA-00600 [unable to load XDB library] on the Free lite image:** the lite
image registers XDB but doesn't ship its library, so `DBMS_METADATA` fails and
ends the session. MemoRizz reads each `EMBEDDING` column's declared dimension
from the driver's column metadata (python-oracledb reports it on a zero-row
query), then `ALL_TAB_COLS.VECTOR_INFO`, and uses `DBMS_METADATA` only when
neither is available, so restarting on an existing schema works on the lite
image. The lite image also has no `USERS` tablespace: give the schema an
automatic-segment-space tablespace, because VECTOR columns can't live in
`SYSTEM` (ORA-43853).

**ORA-51962:** inspect `VECTOR_MEMORY_SIZE`, switch to `none`/`lazy`, or have an
authorized DBA increase vector memory and restart the database.

**Missing summary/cache fields:** apply migration 004, then restart the
provider and rerun preflight/tests.

**Missing knowledge provenance fields:** apply migration 006. It adds
`source_id`, `parent_source_id`, `linked_source_ids`, and `metadata` to
`knowledge_base` so the same grounded evidence contract survives Filesystem,
MongoDB, and Oracle round trips.

**Missing structured tool outcomes:** apply migration 007, then restart the
provider. It adds `outcome` and `outcome_details` to `tool_log` and maps legacy
success/failure rows to `success`/`error` without changing their payloads.
