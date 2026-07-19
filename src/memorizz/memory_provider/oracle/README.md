# Oracle AI Database Memory Provider

This module provides Oracle Database 23ai/26ai support for Memorizz, enabling vector similarity search and memory persistence using Oracle's native VECTOR datatype and AI capabilities.

## Features

- **Native Vector Support**: Leverages Oracle 23ai+ VECTOR datatype for efficient similarity search
- **Connection Pooling**: Built-in connection pooling for optimal performance
- **Vector Indexes**: Automatic creation of HNSW vector indexes with configurable accuracy
- **In-Database Embeddings**: Default ONNX inference with Oracle `VECTOR_EMBEDDING`
- **Full CRUD Operations**: Complete support for all memory operations
- **JSON Storage**: Utilizes Oracle's JSON capabilities for flexible document storage
- **Lazy Index Creation**: Optional deferred index creation for faster startup

## Requirements

### Database Requirements

- Oracle Database 23ai or later (26ai recommended)
- Oracle AI Vector Search feature enabled
- User with appropriate privileges (`CREATE TABLE`, `CREATE INDEX`, `CREATE MINING MODEL`, and `EXECUTE ON DBMS_VECTOR`)

### Python Requirements

```bash
pip install oracledb
```

## Quick Start

### 1. Set Up Oracle Schema with Embedding Dimensions

The default in-database model produces 384-dimensional vectors:

```bash
# Optional; 384 is already the setup default.
export ORACLE_EMBEDDING_DIM=384

# Run Oracle setup
python -m memorizz.memory_provider.oracle.setup
```

### 2. Initialize Oracle Provider

```python
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig

config = OracleConfig(
    user="memorizz_user",
    password="secure_password",
    dsn="localhost:1521/FREEPDB1",
    schema="memorizz",
    lazy_vector_indexes=False,
    in_database_embedding=True,
)

provider = OracleProvider(config)
```

On first use, MemoRizz installs Oracle's augmented `ALL_MINILM_L12_V2`
ONNX model when it is absent. Set `embedding_config["onnx_path"]` to install
from a local file instead of Oracle's public model URL.

### 3. Use an External Embedding Provider (Optional)

```python
from memorizz.embeddings import configure_embeddings

configure_embeddings("openai", {
    "model": "text-embedding-3-small",
    "dimensions": 256,
})
```

For this opt-out path, create `OracleConfig` with
`in_database_embedding=False`, set `ORACLE_EMBEDDING_DIM=256` before schema
setup, and supply the external provider explicitly.

### 4. Create Agent with Oracle Backend

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (MemAgentBuilder()
    .with_instruction("You are a helpful assistant")
    .with_memory_provider(provider)
    .with_llm_config({'provider': 'openai', 'model': 'gpt-4'})
    .build())

response = agent.run("Hello, how can you help me?")
```

## Configuration Options

### OracleConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | str | Required | Oracle database username |
| `password` | str | Required | Oracle database password |
| `dsn` | str | Required | Data Source Name or connection string |
| `schema` | str | `user` | Schema name for all tables |
| `lazy_vector_indexes` | bool | `False` | Defer vector index creation until first use |
| `in_database_embedding` | bool | `True` | Install/use an Oracle ONNX embedding model when no external provider is supplied |
| `embedding_provider` | str/object | `None` | Explicit external embedding provider; takes precedence over the in-database default |
| `embedding_config` | dict | `{}` | Model, dimensions, ONNX source, or external-provider options |
| `pool_min` | int | `1` | Minimum connections in pool |
| `pool_max` | int | `5` | Maximum connections in pool |
| `pool_increment` | int | `1` | Connections to add when pool exhausted |

### Connection String Formats

#### Basic Format
```python
dsn="localhost:1521/FREEPDB1"
```

#### Easy Connect Plus
```python
dsn="myhost.example.com:1521/xepdb1"
```

#### TNS Format
```python
dsn="""(DESCRIPTION=
    (ADDRESS=(PROTOCOL=TCP)(HOST=myhost)(PORT=1521))
    (CONNECT_DATA=(SERVICE_NAME=FREEPDB1)))"""
```

#### TNS Alias (requires tnsnames.ora)
```python
dsn="mydb_alias"
```

## Database Setup

### 1. Create Database User

```sql
-- Connect as SYSTEM or DBA
CREATE USER memorizz_user IDENTIFIED BY secure_password;

GRANT CREATE SESSION TO memorizz_user;
GRANT CREATE TABLE TO memorizz_user;
GRANT CREATE INDEX TO memorizz_user;
GRANT CREATE MINING MODEL TO memorizz_user;
GRANT UNLIMITED TABLESPACE TO memorizz_user;

-- For vector operations (Oracle 23ai+)
GRANT EXECUTE ON DBMS_VECTOR TO memorizz_user;
```

### 2. Verify Vector Support

```sql
-- Check if AI Vector Search is available
SELECT * FROM V$VERSION WHERE BANNER LIKE '%23ai%' OR BANNER LIKE '%26ai%';

-- Test VECTOR datatype
CREATE TABLE test_vectors (
    id NUMBER,
    vec VECTOR(1536, FLOAT32)
);

DROP TABLE test_vectors;
```

### 3. Initialize Schema

The provider automatically creates all required tables on first initialization:

- `personas` - Persona definitions with embeddings
- `toolbox` - Tool definitions
- `short_term_memory` - Working memory
- `knowledge_base` - Knowledge base
- `conversation_memory` - Chat history
- `workflow_memory` - Process workflows
- `skillbox` - Learned SKILL.md documents, lifecycle state, and
  `injection_role` (`user` or `developer`)
- `agents` - Agent configurations
- `shared_memory` - Multi-agent shared state
- `summaries` - Conversation summaries
- `semantic_cache` - Response caching

Each table includes:
- `id` (RAW(16)) - UUID primary key
- `data` (CLOB) - JSON document storage
- `embedding` (VECTOR) - Vector embeddings
- `name`, `memory_id`, `agent_id` - Indexed query fields
- `created_at`, `updated_at` - Timestamps

### 4. Upgrade an Existing Continual-Learning Schema

Fresh setup includes learned-skill authority. If the schema predates that
field and is managed manually, execute:

```sql
@migrations/002_add_skill_injection_role.sql
@migrations/003_add_shadow_evaluations.sql
```

The additive migration assigns existing skills `user` authority and adds a
constraint for `user | developer`. Filesystem and MongoDB need no schema
migration because the role is stored in each skill document.

Migration 003 adds the JSON-constrained
`workflow_memory.shadow_evaluations` CLOB used by opt-in passive SHADOW
evaluation. Workflow records remain the audit source; the worker derives
bounded skill-level readiness statistics from them without injecting shadow
content or rerunning tools.

If an older package logs
`OracleProvider._table_has_column() missing 1 required positional argument:
'column_name'`, upgrade before retrying retrieval. Promotion writes could
appear successful because learned-skill retrieval is fail-safe and converted
the lazy-index exception into an empty match list. The corrected provider
passes the active cursor to the column check.

## Vector Search

Oracle's vector search uses the `VECTOR_DISTANCE` function with COSINE similarity by default.

### Vector Index Types

The provider creates HNSW (Hierarchical Navigable Small World) indexes:

```sql
CREATE VECTOR INDEX idx_tablename_vec
ON tablename (embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;
```

### Similarity Metrics

- **COSINE** (default) - Cosine similarity (best for normalized vectors)
- **EUCLIDEAN** - L2 distance
- **DOT** - Dot product similarity

To change the metric, modify the `_ensure_vector_index` method in `provider.py`.

## Performance Tuning

### Connection Pool Sizing

```python
config = OracleConfig(
    user="memorizz_user",
    password="password",
    dsn="localhost:1521/FREEPDB1",
    pool_min=2,      # Minimum idle connections
    pool_max=10,     # Maximum total connections
    pool_increment=2 # Add 2 at a time when needed
)
```

### Vector Index Tuning

For better performance with large datasets:

```sql
-- Increase target accuracy (slower indexing, better recall)
ALTER INDEX idx_tablename_vec REBUILD
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 99;

-- Prefer faster indexing / lower recall
ALTER INDEX idx_tablename_vec REBUILD
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 90;
```

### Query Optimization

```sql
-- Add hints for vector search queries
SELECT /*+ INDEX(t idx_tablename_vec) */
    id, data,
    VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
FROM tablename t
WHERE memory_id = :memory_id
ORDER BY distance
FETCH FIRST 10 ROWS ONLY;
```

## Migration from MongoDB

If you're migrating from MongoDB to Oracle:

### 1. Export MongoDB Data

```python
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig

mongo_config = MongoDBConfig(uri="mongodb://localhost:27017")
mongo_provider = MongoDBProvider(mongo_config)

# Export all memory types
for memory_type in MemoryType:
    documents = mongo_provider.list_all(memory_type, include_embedding=True)
    # Store documents...
```

### 2. Import to Oracle

```python
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig

oracle_config = OracleConfig(
    user="memorizz_user",
    password="password",
    dsn="localhost:1521/FREEPDB1"
)
oracle_provider = OracleProvider(oracle_config)

# Import documents
for doc in documents:
    oracle_provider.store(doc, memory_type)
```

## Troubleshooting

### Connection Issues

```python
# Test connection
import oracledb

try:
    conn = oracledb.connect(
        user="memorizz_user",
        password="password",
        dsn="localhost:1521/FREEPDB1"
    )
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
```

### Vector Index Creation Fails

If vector index creation fails during initialization:

1. Use `lazy_vector_indexes=True` to defer creation
2. Verify Oracle version supports VECTOR datatype
3. Check user has CREATE INDEX privilege
4. Verify sufficient tablespace

```python
config = OracleConfig(
    user="memorizz_user",
    password="password",
    dsn="localhost:1521/FREEPDB1",
    lazy_vector_indexes=True  # Create indexes on first use
)
```

### ORA-02236: invalid file name

Oracle raises `ORA-02236` when it needs a concrete datafile path but Oracle Managed Files is disabled. Before running the setup script again, point MemoRizz at an existing datafile directory (inside the Oracle server/container) or provide an explicit path:

```bash
export ORACLE_DATAFILE_DIR="/opt/oracle/oradata/FREEPDB1"
# or
export ORACLE_TABLESPACE_DATAFILE="/opt/oracle/oradata/FREEPDB1/memorizz_ts01.dbf"
```

Optional overrides:

```bash
export ORACLE_TABLESPACE_NAME="MEMORIZZ_TS"
export ORACLE_TABLESPACE_SIZE_MB="200"
export ORACLE_TABLESPACE_AUTOEXTEND_MB="25"
```

Rerun `memorizz oracle setup` (or
`python -m memorizz.memory_provider.oracle.setup`) after setting these
variables.

### Configuring Embedding Dimensions

**IMPORTANT**: The embedding dimensions in your Oracle schema must match the dimensions produced by your embedding model.

#### Setting Dimensions During Setup

Use the `ORACLE_EMBEDDING_DIM` environment variable to configure embedding dimensions when running the setup script:

```bash
# For 256-dimensional embeddings (e.g., OpenAI text-embedding-3-small with dimensions=256)
export ORACLE_EMBEDDING_DIM=256

# For 512-dimensional embeddings (e.g., Voyage AI voyage-3-lite)
export ORACLE_EMBEDDING_DIM=512

# For 1536-dimensional embeddings (e.g., OpenAI text-embedding-3-small default)
export ORACLE_EMBEDDING_DIM=1536

# Run setup with the configured dimension
python -m memorizz.memory_provider.oracle.setup
```

**Default**: If `ORACLE_EMBEDDING_DIM` is not set, the schema defaults to
**384 dimensions**, matching `ALL_MINILM_L12_V2`.

#### Supported Embedding Models by Dimension

**384 dimensions (default):**
```python
config = OracleConfig(
    user="memorizz_user",
    password="password",
    dsn="localhost:1521/FREEPDB1",
    in_database_embedding=True,
)
```

**256 dimensions:**
```python
configure_embeddings('openai', {
    'model': 'text-embedding-3-small',
    'dimensions': 256  # Must match ORACLE_EMBEDDING_DIM
})
```

**512 dimensions:**
```python
configure_embeddings('voyageai', {
    'model': 'voyage-3-lite',  # Native 512 dimensions
})
```

**1536 dimensions (OpenAI default):**
```python
configure_embeddings('openai', {
    'model': 'text-embedding-3-small',
    'dimensions': 1536  # OpenAI's default
})
```

**3072 dimensions (OpenAI max):**
```python
configure_embeddings('openai', {
    'model': 'text-embedding-3-large',
    'dimensions': 3072
})
```

#### Checking Current Dimensions

```python
# Check your current embedding configuration
from memorizz.embeddings import get_embedding_dimensions

dims = get_embedding_dimensions()
print(f"Current embedding dimensions: {dims}")
```

```sql
-- Check Oracle schema VECTOR column dimensions
SELECT table_name, column_name, data_type
FROM user_tab_columns
WHERE data_type LIKE 'VECTOR%'
ORDER BY table_name;
```

#### Dimension Mismatch Errors

If you see errors like:
```
ORA-51814: Vector of dimension [X] cannot be inserted into a vector column of dimension [Y]
```

This means your embedding dimension doesn't match the Oracle schema. To fix:

1. **Option 1: Recreate schema with correct dimensions**
   ```bash
   export ORACLE_EMBEDDING_DIM=256  # or your desired dimension
   python -m memorizz.memory_provider.oracle.setup
   ```
   ⚠️ This will drop existing data

2. **Option 2: Change your embedding configuration to match existing schema**
   ```python
   # If schema uses 256, configure embeddings to produce 256
   configure_embeddings('openai', {
       'model': 'text-embedding-3-small',
       'dimensions': 256
   })
   ```

### Performance Issues

1. **Enable connection pooling** (default, but verify pool size)
2. **Create additional indexes** on frequently queried fields
3. **Gather statistics** on tables and indexes:

```sql
-- Gather statistics for better query plans
EXEC DBMS_STATS.GATHER_TABLE_STATS('MEMORIZZ_USER', 'CONVERSATION_MEMORY');
```

## Advanced Usage

### Custom Embedding Provider

```python
from memorizz.embeddings import EmbeddingManager
from memorizz.memagent.builders import MemAgentBuilder

embedding_mgr = EmbeddingManager('voyageai', {
    'model': 'voyage-3-large',
    'output_dimension': 1024
})

config = OracleConfig(
    user="memorizz_user",
    password="password",
    dsn="localhost:1521/FREEPDB1",
    embedding_provider=embedding_mgr
)

provider = OracleProvider(config)

# Build agent with custom configuration
agent = (MemAgentBuilder()
    .with_instruction("You are an AI assistant with long-term memory")
    .with_memory_provider(provider)
    .with_llm_config({'provider': 'openai', 'model': 'gpt-4'})
    .with_semantic_cache(enabled=True, threshold=0.9)
    .with_verbose(True)
    .build())
```

### Multi-Schema Deployment

```python
# Separate schemas for different environments
dev_config = OracleConfig(
    user="memorizz_dev",
    password="dev_password",
    dsn="localhost:1521/FREEPDB1",
    schema="memorizz_dev"
)

prod_config = OracleConfig(
    user="memorizz_prod",
    password="prod_password",
    dsn="prod-host:1521/PRODDB",
    schema="memorizz_prod"
)
```

### Monitoring

```sql
-- Monitor connection pool usage
SELECT * FROM V$SESSION WHERE USERNAME = 'MEMORIZZ_USER';

-- Check vector index status
SELECT index_name, status, tablespace_name
FROM user_indexes
WHERE index_name LIKE 'IDX%VEC';

-- Query performance
SELECT sql_text, executions, elapsed_time, cpu_time
FROM v$sql
WHERE sql_text LIKE '%VECTOR_DISTANCE%'
ORDER BY elapsed_time DESC;
```

## Best Practices

1. **Use connection pooling** - Don't create new providers for each operation
2. **Keep one embedding space** - Use the Oracle default or configure one external provider consistently
3. **Use appropriate pool sizes** - Match your concurrency needs
4. **Monitor index health** - Rebuild indexes periodically for large datasets
5. **Use lazy indexes for development** - Faster startup, create indexes as needed
6. **Secure credentials** - Use environment variables or secure vaults
7. **Regular backups** - Oracle RMAN or Data Pump exports
8. **Test failover** - Use Oracle RAC or Data Guard for high availability

## Examples

See the `/examples` directory for complete examples:

- `oracle_basic.py` - Basic setup and usage
- `oracle_migration.py` - Migrating from MongoDB
- `oracle_multi_agent.py` - Multi-agent with Oracle backend
- `oracle_performance.py` - Performance optimization techniques

## Support

For issues specific to:
- **Oracle Database**: Check Oracle documentation for 23ai/26ai
- **Vector Search**: Refer to Oracle AI Vector Search documentation
- **Python Driver**: See python-oracledb documentation
- **Memorizz Integration**: Open an issue on GitHub

## License

Same as Memorizz - see main LICENSE file.
