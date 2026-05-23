# Oracle Provider

The Oracle AI Database provider offers fully managed JSON + vector storage for every MemoRizz memory type. It targets Oracle 23ai/26ai and lives in `src/memorizz/memory_provider/oracle/`.

## Highlights

- Native VECTOR datatype with automatic HNSW indexes
- Connection pooling + lazy schema creation
- Works with JSON Relational Duality Views for structured + vector queries

## Installation

```bash
pip install -e ".[oracle]"
```

## Configuration

```python
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig

provider = OracleProvider(OracleConfig(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1",
    schema="MEMORIZZ",
    embedding_provider="openai",
    embedding_config={"model": "text-embedding-3-small"},
    lazy_vector_indexes=False,
))
```

Set `lazy_vector_indexes=True` if you want faster cold starts and are ok with indexes being created on demand.

### Shared Embedding Defaults (UI + Notebook Alignment)

If you connect from multiple clients (UI, notebooks, scripts), set common embedding defaults so Oracle VECTOR dimensions stay consistent:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=1536
```

When `OracleConfig.embedding_provider` is omitted, the Oracle provider will use these env defaults automatically.

## Database Prep

1. Create a dedicated user with `CREATE SESSION`, `CREATE TABLE`, `CREATE INDEX`, `UNLIMITED TABLESPACE`.
2. Grant `EXECUTE ON DBMS_VECTOR` for vector search.
3. Run `memorizz setup-oracle` or the scripts in `src/memorizz/memory_provider/oracle/` to create the tables.

## Tables

Every memory bucket gets its own table plus a VECTOR index:

- `personas`
- `toolbox`
- `knowledge_base`
- `entity_memory`
- `short_term_memory`
- `conversation_memory`
- `workflow_memory`
- `shared_memory`
- `summaries`
- `semantic_cache`

## Troubleshooting

- **Vector datatype missing** – Ensure you're running 23ai+ and have `DBMS_VECTOR` privileges.
- **Connection refused** – Use Easy Connect Plus (`host:port/service`) or TNS alias strings.
- **Slow cold start** – Enable `lazy_vector_indexes` or pre-create indexes manually using the SQL files in the provider folder.
- **Embedding dimension mismatch** – Align provider model/output dimensions with existing table VECTOR dimensions, or use a separate schema per embedding profile.

For the full reference, open `src/memorizz/memory_provider/oracle/README.md`.
