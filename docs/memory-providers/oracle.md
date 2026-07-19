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
    in_database_embedding=True,
    lazy_vector_indexes=False,
))
```

Set `lazy_vector_indexes=True` if you want faster cold starts and are ok with indexes being created on demand.

### In-Database Embeddings (Default)

Oracle connections default to `in_database_embedding=True`. MemoRizz checks for
the augmented `ALL_MINILM_L12_V2` ONNX model, installs it from Oracle's public
model bucket when it is missing, and uses `VECTOR_EMBEDDING` for writes and
queries. The default model produces 384-dimensional vectors.

The database user needs `CREATE MINING MODEL` and `EXECUTE ON DBMS_VECTOR`.
`memorizz oracle setup` grants both in admin mode.

```python
provider = OracleProvider(OracleConfig(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1",
    in_database_embedding=True,
    embedding_config={
        "model": "ALL_MINILM_L12_V2",
        "dimensions": 384,
        # Optional: use a local augmented ONNX file instead of the public URL.
        # "onnx_path": "/models/all_MiniLM_L12_v2.onnx",
        "install_if_missing": True,
    },
))
```

Set `in_database_embedding=False` to disable this behavior. An explicitly
supplied `embedding_provider` still takes precedence, which preserves existing
OpenAI, Ollama, Voyage AI, Azure, and Hugging Face configurations:

```python
provider = OracleProvider(OracleConfig(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1",
    in_database_embedding=False,
    embedding_provider="openai",
    embedding_config={"model": "text-embedding-3-small"},
))
```

Oracle documents both
[`DBMS_VECTOR.LOAD_ONNX_MODEL`](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/load_onnx_model-procedure.html)
and
[`VECTOR_EMBEDDING`](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/vector_embedding.html)
in the AI Vector Search guide.

### External Embedding Defaults

When in-database embeddings are explicitly disabled, clients can share an
external embedding configuration through environment variables:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=1536
```

These defaults are read when `in_database_embedding=False` and
`OracleConfig.embedding_provider` is omitted.

## Database Prep

1. Create a dedicated user with `CREATE SESSION`, `CREATE TABLE`, `CREATE INDEX`, `CREATE MINING MODEL`, `UNLIMITED TABLESPACE`.
2. Grant `EXECUTE ON DBMS_VECTOR` for model loading and vector search.
3. Run `memorizz oracle setup` to create the user and tables, or
   `memorizz oracle setup-schema` to apply schema updates without dropping
   an existing user.

## Tables

Every memory bucket gets its own table plus a VECTOR index:

- `personas`
- `toolbox`
- `knowledge_base`
- `entity_memory`
- `short_term_memory`
- `conversation_memory`
- `workflow_memory`
- `skillbox` (including persisted `injection_role` for learned skills)
- `shared_memory`
- `summaries`
- `semantic_cache`

## Troubleshooting

- **Vector datatype missing** – Ensure you're running 23ai+ and have `DBMS_VECTOR` privileges.
- **ONNX model installation fails** – Grant `CREATE MINING MODEL` and `EXECUTE ON DBMS_VECTOR`, or set `embedding_config["onnx_path"]` to an augmented ONNX file readable by the notebook process.
- **Connection refused** – Use Easy Connect Plus (`host:port/service`) or TNS alias strings.
- **Slow cold start** – Enable `lazy_vector_indexes` or pre-create indexes manually using the SQL files in the provider folder.
- **Embedding dimension mismatch** – Align provider model/output dimensions with existing table VECTOR dimensions, or use a separate schema per embedding profile.
- **`OracleProvider._table_has_column()` missing `column_name`** – Upgrade
  MemoRizz. Older lazy-index code called the helper without its cursor, then
  skill retrieval failed safely and returned no matches even though promotion
  writes had succeeded.

## Existing-schema migration for skill authority

Fresh setup and `memorizz oracle setup-schema` include
`skillbox.injection_role`. For an existing schema managed outside those
commands, run:

```sql
@src/memorizz/memory_provider/oracle/migrations/002_add_skill_injection_role.sql
```

Existing rows receive `user`, preserving the previous behavior. The column
accepts only `user` or `developer`.

For the full reference, open `src/memorizz/memory_provider/oracle/README.md`.
