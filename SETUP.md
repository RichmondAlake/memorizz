# MemoRizz 0.5 Setup Guide

This guide sets up MemoRizz with a local Oracle AI Database, verifies the
database capabilities used by 0.5, and shows the production configuration
boundary. MemoRizz never supplies or prints a default database password.

## Prerequisites

- Python 3.10 or newer;
- Docker Desktop, OrbStack, or another Docker-compatible runtime;
- enough local memory for Oracle Free and the configured vector pool;
- provider credentials for the LLM, internet, sandbox, or browser features you
  explicitly enable.

Install the package and Oracle/UI extras:

```bash
python -m pip install "memorizz[oracle,ui]"
```

For development from this repository:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[oracle,ui,dev,sandbox-e2b]"
```

## Configure secrets

Copy `.env.example` to `.env` and use unique passwords. Keep `.env` outside
source control and inject production values from your secret manager.

```dotenv
MEMORIZZ_BACKEND=oracle
ORACLE_ADMIN_USER=system
ORACLE_ADMIN_PASSWORD=<strong-admin-password>
ORACLE_USER=memorizz_user
ORACLE_PASSWORD=<different-strong-application-password>
ORACLE_DSN=localhost:1521/FREEPDB1
OPENAI_API_KEY=<provider-key>
```

`ORACLE_ADMIN_PASSWORD` is needed only when MemoRizz must create or grant the
application schema. Normal application processes should receive only
`ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN`.

## Start Oracle locally

### Package CLI

Set the admin password in the environment or enter it at the hidden prompt:

```bash
memorizz oracle install --image lite
memorizz oracle setup
```

The install command never echoes either password. `oracle setup` detects two
modes:

- admin mode creates the configured application user, grants required
  privileges, and initializes the schema;
- user-only mode connects to an already-provisioned schema and initializes
  objects without requiring administrator access.

For an existing schema, prefer the non-destructive update path:

```bash
memorizz oracle setup-schema
```

### SDK-owned local runtime

Applications and notebooks can use one package-owned bootstrap contract:

```python
from memorizz.memory_provider.oracle import LocalOracleRuntime, OracleProvider

runtime = LocalOracleRuntime.from_env(provision_if_missing=True)
runtime_report = runtime.ensure_ready()

provider = OracleProvider.from_env(
    provision_if_missing=False,
    index_policy="lazy",
)
report = provider.preflight()
assert report["ok"], report["diagnostics"]
```

`LocalOracleRuntime` uses explicit Docker argument arrays, the requested host
port and container name, and a bounded readiness wait. Set
`MEMORIZZ_ORACLE_CONTAINER` when reusing a non-default container.

## Upgrade an existing schema

Back up the schema, then apply the additive migrations in numeric order as the
application schema owner under your normal change-control process:

```text
src/memorizz/memory_provider/oracle/migrations/004_production_governance_050.sql
src/memorizz/memory_provider/oracle/migrations/005_scoped_retrieval_052.sql
```

These add complete Toolbox schemas, canonical summary-source metadata,
conversation summary markers, normalized summary/message links, semantic-cache
metadata, and exact summary-thread/knowledge-namespace retrieval scopes. The
provider also performs idempotent additive checks at startup, but the SQL
migrations are the reviewable deployment artifacts.

Earlier installations must also apply migrations `001`, `002`, and `003` in
numeric order if they have not already done so.

## Configure the Oracle provider

Oracle in-database ONNX embeddings are the default:

```python
import os

from memorizz.memory_provider.oracle import OracleConfig, OracleProvider

provider = OracleProvider(
    OracleConfig(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        in_database_embedding=True,
        index_policy="lazy",
    )
)
```

Index policies are:

- `none`: exact vector search only;
- `lazy`: create an index when that memory type is first used;
- `selected`: create only `selected_vector_indexes`;
- `eager`: attempt all enabled vector indexes at provider startup.

Exact vector search remains available when vector memory or an HNSW index is
unavailable. `preflight()` emits one structured diagnostic and a recommended
`VECTOR_MEMORY_SIZE` instead of one warning per table.

To use an external embedding service instead:

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
    )
)
```

The vector-column dimension must match the embedding output dimension.

## Verify readiness

Run both feature and provider reports:

```bash
memorizz capabilities --json
memorizz oracle preflight --json
```

The Oracle report includes:

- database product/version, registered service, PDB and open state;
- schema privileges;
- in-database model availability and embedding dimensions;
- vector-column dimensions and `VECTOR_MEMORY_SIZE`;
- vector-index status, selected index policy, and exact-search fallback.

The Python equivalent is:

```python
report = provider.preflight()
agent_report = agent.capability_report(preflight=True)
```

## Build an agent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_instruction("You are a tenant-scoped memory assistant.")
    .with_memory_provider(provider)
    .with_llm_config({"provider": "openai", "model": "gpt-4.1-mini"})
    .with_tool_result_policy({"offload_above_chars": 12000})
    .with_context_policy({"progressive_tool_disclosure": True, "tool_top_k": 6})
    .build(validate=True, persist=True)
)

agent.run("Remember that the launch is Thursday.", user_id="tenant-a")
```

Always pass `user_id` in multi-user applications. A globally scoped call is
not a substitute for tenant identity.

## Optional Browser Use capability

Keep Browser Use in an isolated tool environment so its dependency graph does
not replace MemoRizz's MCP runtime:

```bash
uv tool install --python 3.12 browser-use
browser-use install
browser-use doctor
```

Then set the matching LLM credential and explicitly opt in:

```dotenv
MEMORIZZ_BROWSER_CONTROL_PROVIDER=browseruse
MEMORIZZ_BROWSER_USE_LLM_PROVIDER=openai
MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS=example.com
```

```python
agent.with_browser_control(
    {
        "provider": "browseruse",
        "llm_provider": "openai",
        "allowed_domains": ["example.com"],
        "max_steps": 10,
        "task_timeout": 300,
    }
)
```

Every model-initiated task pauses for a durable host decision. See
[`docs/browser-control/index.md`](docs/browser-control/index.md) for CLI/UI,
approval, domain, CDP, cloud, result, and custom-provider configuration.

## Scoped cleanup

Test/course data can be removed transactionally without dropping the schema:

```python
report = provider.delete_scope(
    memory_id="verification-run-17",
    user_id="tenant-a",
    agent_ids=["planner", "executor"],
)
print(report["counts"])
```

At least one exact scope is mandatory. The operation reports per-table counts
and rolls back on failure.

## Troubleshooting

### Connection refused

```bash
docker ps -a
docker logs <oracle-container-name>
```

Confirm that the host port in `ORACLE_DSN` matches the Docker port mapping and
that `FREEPDB1` is open read/write.

### Invalid credentials

MemoRizz does not know or guess the container password. Confirm the values in
your secret manager or recreate a development container with a newly generated
password. Do not place secrets in command-line arguments or screenshots.

### ORA-51962 or unavailable HNSW indexes

Use `index_policy="none"` or `"lazy"` while exact search is sufficient, then
inspect `provider.preflight()`. Increase `VECTOR_MEMORY_SIZE` through an
authorized database administrator and restart Oracle when the report recommends
it.

### Dimension mismatch

Compare `preflight()["vector_dimensions"]` with
`preflight()["embedding"]["dimensions"]`. Rebuild empty vector columns with
the intended dimension or configure the matching embedding model; do not mix
dimensions in one schema.

### Hosted Oracle

Set the application credentials and DSN, omit `ORACLE_ADMIN_PASSWORD`, and run
`memorizz oracle setup-schema`. Ask the DBA for the privileges reported by
`preflight()` and for permission to install an ONNX model if in-database
embedding is required.

## Production checklist

- use a secret manager and separate admin/application credentials;
- keep `MEMORIZZ_HOME` and database storage durable;
- apply migrations under change control and back up before upgrades;
- choose an explicit index policy and monitor vector memory;
- pass `user_id` and enforce tenant identity at the application edge;
- call `capability_report(preflight=True)` during deployment health checks;
- run `scripts/verify_production_050.py` against a staging database before
  promotion.
