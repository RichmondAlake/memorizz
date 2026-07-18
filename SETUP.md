# MemoRizz Setup Guide for Oracle AI Database

This guide provides step-by-step instructions for setting up MemoRizz with Oracle AI Database on your local machine.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

- **Python 3.7+** - Check with `python --version` or `python3 --version`
- **Docker** - Required for running Oracle Database locally
  - Check with `docker --version`
  - Install from [docker.com](https://www.docker.com/get-started)
  - **Apple Silicon Note**: Oracle Free may require emulation. Use `PLATFORM_FLAG="--platform linux/amd64"` if you encounter issues
- **Git** - For cloning the repository (if installing from source)
- **python-dotenv** (Optional but recommended) - For `.env` file support
  - Install with: `pip install python-dotenv`

### Required Python Packages

Install MemoRizz and Oracle driver:

```bash
# Install MemoRizz
pip install memorizz

# Install Oracle driver (or use: pip install memorizz[oracle])
pip install oracledb

# Install OpenAI SDK (for LLM and embeddings)
pip install openai
```

> **Hugging Face support:** Install `memorizz[huggingface]` before selecting
> `embedding_provider="huggingface"` or the Hugging Face LLM provider. The
> heavyweight Transformers/PyTorch stack is intentionally excluded from the
> base install.

### API Keys

- **OpenAI API Key** - Required for embeddings and LLM functionality
  - Get one at [platform.openai.com](https://platform.openai.com/api-keys)

---

## Quick Start

For a quick setup with default settings, follow these steps:

### Step 1: Start Oracle Database

```bash
# Start Oracle Database with the recommended lite image
memorizz oracle install --image lite
```

**For Apple Silicon (M1/M2/M3) users:**
```bash
# Oracle Free may require emulation on ARM64
export PLATFORM_FLAG="--platform linux/amd64"
memorizz oracle install --image lite
```

**Oracle Image Version Selection:**
```bash
# Use lite version (1.78GB - recommended for development)
memorizz oracle install --image lite

# Use full version (9.93GB - includes all features)
memorizz oracle install --image full

# Use the community image (faster startup)
memorizz oracle install --image community
```

This will:
- Pull Oracle Database 23ai Free Lite Docker image (1.78GB, default) if not already present
- Create a persistent Docker volume (`oracle-memorizz-data`) for data storage
- Create and start a container named `oracle-memorizz`
- Wait for the database to be ready (~2-3 minutes)
- **Data persists** between container restarts thanks to the Docker volume

**Oracle Image Versions:**
- **Lite version (default)**: `latest-lite` - 1.78GB, faster download, recommended for development
- **Full version**: `latest` - 9.93GB, includes all features and tools
- **Community version**: `gvenzl/oracle-free:latest` - faster startup for development

To use the full version:
```bash
memorizz oracle install --image full
```

**Default Connection Details:**
- Host: `localhost`
- Port: `1521`
- Service Name: `FREEPDB1`
- Admin User: `system`
- Admin Password: `MyPassword123!`

**Note:** The Docker volume ensures your data persists even if you stop/remove the container. To remove all data, delete the volume: `docker volume rm oracle-memorizz-data`

### Step 2: Set Up Database Schema

**Option A: Using CLI command (Recommended - works for all installation methods)**

```bash
# After installing memorizz[oracle]
memorizz oracle setup

# Or using Python module
python -m memorizz oracle setup
```

**Option B: Using the examples script (Repo-cloned users only)**

```bash
# Only works if you cloned the repository
# Note: CLI command (Option A) is recommended for most users
python examples/setup_oracle_user.py
```

**Option C: Using Python directly**

```python
from memorizz.memory_provider.oracle.setup import setup_oracle_user

setup_oracle_user()
```

#### Automatic Mode Detection

The setup function **automatically detects** your database configuration and adapts:

**Admin Mode** (Local/Self-Hosted Databases):
- Detects when admin credentials are available
- Creates new user with all required privileges
- Grants permissions automatically
- Full setup with user creation

**User-Only Mode** (Hosted Databases like FreeSQL.com):
- Detects when admin access is not available
- Uses your existing schema/user
- Checks available privileges
- Creates tables and views in your schema
- Warns about missing privileges

**What the setup does:**
- ✅ Automatically detects setup mode (admin vs user-only)
- ✅ Creates relational schema (12 tables + indexes)
- ✅ Creates JSON Duality Views (10 views)
- ✅ Verifies the setup
- ✅ Shows detected mode and available privileges

**Default Credentials (Local Docker):**
- User: `memorizz_user`
- Password: `SecurePass123!`
- DSN: `localhost:1521/FREEPDB1`

**For Hosted Databases (FreeSQL.com, etc.):**
- Set `ORACLE_USER` to your existing schema username
- Set `ORACLE_PASSWORD` to your schema password
- Set `ORACLE_DSN` to your database connection string
- The setup will automatically use user-only mode

### Step 3: Configure Your Application

**Option A: Using .env file (Recommended)**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and fill in your credentials
# Then load it in your Python code:
```

```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORACLE_USER = os.getenv("ORACLE_USER")
# ... etc
```

**Option B: Environment Variables**

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="SecurePass123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"
```

See `examples/example_with_env.py` for a complete example using `.env` files.

### Step 4: Create Your First Agent

```python
import os
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig
from memorizz.memagent.builders import MemAgentBuilder

# Set up your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Configure Oracle memory provider
oracle_config = OracleConfig(
    user=os.environ.get("ORACLE_USER", "memorizz_user"),
    password=os.environ.get("ORACLE_PASSWORD", "SecurePass123!"),
    dsn=os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1"),
    embedding_provider="openai",
    embedding_config={
        "model": "text-embedding-3-small",
        "api_key": os.environ["OPENAI_API_KEY"]
    }
)
oracle_provider = OracleProvider(oracle_config)

# Create a MemAgent using the builder pattern
agent = (MemAgentBuilder()
    .with_instruction("You are a helpful assistant with persistent memory.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Save the agent to Oracle
agent.save()

# Start conversing - the agent will remember across sessions
response = agent.run("Hello! My name is John and I'm a software engineer.")
print(response)

# Later in another session...
response = agent.run("What did I tell you about myself?")
print(response)  # Agent remembers John is a software engineer
```

---

## Detailed Setup

### Customizing Credentials

You can customize all credentials using environment variables:

#### For Oracle Database Container

```bash
# Set custom admin password before starting Oracle
export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
memorizz oracle install --image lite

# Or use full version instead of lite (default)
export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
memorizz oracle install --image full
```

#### For Database Schema Setup

```bash
# Set custom credentials before running setup
export ORACLE_ADMIN_USER="system"
export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="YourMemorizzPassword123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"

python examples/setup_oracle_user.py
```

**Environment Variables Reference:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_ADMIN_USER` | `system` | Oracle admin username |
| `ORACLE_ADMIN_PASSWORD` | `MyPassword123!` | Oracle admin password |
| `ORACLE_IMAGE_TAG` | `latest-lite` | Oracle Docker image tag (options: `latest-lite` 1.78GB, `latest` 9.93GB, or custom) |
| `ORACLE_USER` | `memorizz_user` | MemoRizz database user |
| `ORACLE_PASSWORD` | `SecurePass123!` | MemoRizz database password |
| `ORACLE_DSN` | `localhost:1521/FREEPDB1` | Oracle connection string |
| `ORACLE_TABLESPACE_NAME` | auto (`MEMORIZZ_USER_TS`) | Target tablespace for VECTOR columns; created if missing |
| `ORACLE_DATAFILE_DIR` | auto-detected | Directory for new tablespace datafile (useful when Oracle Managed Files is disabled) |
| `ORACLE_TABLESPACE_DATAFILE` | auto-detected | Full path override for new tablespace datafile |
| `ORACLE_TABLESPACE_SIZE_MB` | `100` | Initial datafile size for the created tablespace |
| `ORACLE_TABLESPACE_AUTOEXTEND_MB` | `10` | Autoextend increment for the created tablespace |
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings/LLM |
| `PLATFORM_FLAG` | (empty) | Docker platform flag (use `--platform linux/amd64` for Apple Silicon) |

### Manual Oracle Setup (Alternative)

If you prefer to set up Oracle manually or use an existing Oracle instance:

#### 1. Start Oracle Database

**Using Docker:**
```bash
# Lite version (1.78GB, recommended)
docker pull container-registry.oracle.com/database/free:latest-lite

docker run -d \
  --name oracle-memorizz \
  -p 1521:1521 \
  -e ORACLE_PWD=MyPassword123! \
  container-registry.oracle.com/database/free:latest-lite

# Or full version (9.93GB)
# docker pull container-registry.oracle.com/database/free:latest
# docker run -d --name oracle-memorizz -p 1521:1521 \
#   -e ORACLE_PWD=MyPassword123! \
#   container-registry.oracle.com/database/free:latest

# Wait for database to be ready
docker logs -f oracle-memorizz
# Wait until you see: "DATABASE IS READY TO USE!"
```

**Using Existing Oracle Instance:**
- Ensure Oracle Database 23ai or later is running
- Note your connection details (host, port, service name)

**Using Hosted Databases (FreeSQL.com, Oracle Cloud, etc.):**
- Sign up for a hosted Oracle database service
- Get your connection credentials (user, password, DSN)
- Set environment variables and run setup (see below)
- The setup will automatically detect user-only mode

#### 2. Create Database User (Manual SQL)

**Note:** For hosted databases, skip this step - you'll use your existing schema. The automated setup will detect this automatically.

Connect as SYSTEM or DBA:

```sql
-- Create user
CREATE USER memorizz_user IDENTIFIED BY "SecurePass123!";

-- Grant basic privileges
GRANT CREATE SESSION TO memorizz_user;
GRANT CREATE TABLE TO memorizz_user;
GRANT CREATE INDEX TO memorizz_user;
GRANT CREATE VIEW TO memorizz_user;
GRANT UNLIMITED TABLESPACE TO memorizz_user;

-- Grant AI Vector Search privileges (Oracle 23ai+)
GRANT EXECUTE ON DBMS_VECTOR TO memorizz_user;
GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO memorizz_user;

-- Grant JSON Duality View privileges
GRANT SODA_APP TO memorizz_user;
GRANT SELECT ANY TABLE TO memorizz_user;
```

#### 3. Create Schema

The automated setup command (`memorizz oracle setup`) handles this automatically. It will:
- Detect if you have admin access (admin mode) or need to use existing schema (user-only mode)
- Create all required tables and views
- Verify the setup

You can also run the SQL files manually:

```bash
# SQL files are located at:
# src/memorizz/memory_provider/oracle/schema_relational.sql
# src/memorizz/memory_provider/oracle/duality_views.sql
```

### Using Hosted Databases (FreeSQL.com, Oracle Cloud, etc.)

MemoRizz works with hosted Oracle databases! The setup automatically adapts to your environment.

#### Setup for Hosted Databases

1. **Get your credentials** from your database provider:
   - Username (your schema name)
   - Password
   - Connection string (DSN)

2. **Set environment variables:**
   ```bash
   export ORACLE_USER="your_schema_name"
   export ORACLE_PASSWORD="your_password"
   export ORACLE_DSN="host:port/service_name"
   # Example for FreeSQL.com:
   # export ORACLE_DSN="db.freesql.com:1521/23ai_34ui2"
   ```

3. **Run the setup:**
   ```bash
   memorizz oracle setup
   ```

   The setup will:
   - ✅ Automatically detect user-only mode
   - ✅ Use your existing schema
   - ✅ Check available privileges
   - ✅ Create tables and views
   - ✅ Warn about any missing privileges

#### What to Expect

**User-Only Mode Output:**
```
ℹ User-only mode detected: Using existing schema
  Admin connection failed (this is OK for hosted databases)
  Will use existing user: your_schema_name
  ✓ Connected as your_schema_name

  User privileges:
    ✓ CREATE_TABLE
    ✓ CREATE_VIEW
    ✗ DBMS_VECTOR (may need admin to grant)
    ✗ SODA_APP (may need admin to grant)
```

**Note:** Some privileges (like `DBMS_VECTOR` and `SODA_APP`) may require admin access. If these are missing:
- Contact your database administrator to grant them
- Or use features that don't require these privileges
- The setup will still work for basic functionality

---

## Configuration

### Oracle Provider Configuration

The `OracleConfig` class supports various configuration options:

```python
from memorizz.memory_provider.oracle import OracleConfig

config = OracleConfig(
    user="memorizz_user",                    # Database username
    password="SecurePass123!",               # Database password
    dsn="localhost:1521/FREEPDB1",          # Connection string
    schema="memorizz",                       # Schema name (optional)
    lazy_vector_indexes=False,               # Create indexes immediately
    embedding_provider="openai",             # Embedding provider
    embedding_config={                       # Embedding configuration
        "model": "text-embedding-3-small",
        "api_key": "your-key"
    },
    pool_min=1,                              # Min connection pool size
    pool_max=5,                              # Max connection pool size
    pool_increment=1                         # Pool increment size
)
```

### Connection String Formats

**Basic Format:**
```python
dsn="localhost:1521/FREEPDB1"
```

**Easy Connect Plus:**
```python
dsn="myhost.example.com:1521/xepdb1"
```

**TNS Format:**
```python
dsn="""(DESCRIPTION=
    (ADDRESS=(PROTOCOL=TCP)(HOST=myhost)(PORT=1521))
    (CONNECT_DATA=(SERVICE_NAME=FREEPDB1)))"""
```

**TNS Alias (requires tnsnames.ora):**
```python
dsn="mydb_alias"
```

### Embedding Provider Configuration

MemoRizz supports multiple embedding providers:

**OpenAI:**
```python
embedding_provider="openai"
embedding_config={
    "model": "text-embedding-3-small",
    "api_key": "your-openai-key"
}
```

**Voyage AI:**
```python
embedding_provider="voyageai"
embedding_config={
    "model": "voyage-3-large",
    "api_key": "your-voyage-key"
}
```

**Ollama (Local):**
```python
embedding_provider="ollama"
embedding_config={
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434"
}
```

---

## Verification

### Verify Oracle Database is Running

```bash
# Check Docker container status
docker ps | grep oracle-memorizz

# Check container logs
docker logs oracle-memorizz

# Test connection (requires oracledb)
python -c "import oracledb; conn = oracledb.connect(user='system', password='MyPassword123!', dsn='localhost:1521/FREEPDB1'); print('✓ Connected successfully!'); conn.close()"
```

### Verify Database Schema

```python
import oracledb

conn = oracledb.connect(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1"
)
cursor = conn.cursor()

# Check tables
cursor.execute("""
    SELECT table_name FROM user_tables
    WHERE table_name IN ('AGENTS', 'AGENT_LLM_CONFIGS', 'AGENT_MEMORIES',
                         'PERSONAS', 'TOOLBOX', 'CONVERSATION_MEMORY',
                         'KNOWLEDGE_BASE', 'SHORT_TERM_MEMORY',
                         'WORKFLOW_MEMORY', 'SHARED_MEMORY', 'SUMMARIES',
                         'SEMANTIC_CACHE')
    ORDER BY table_name
""")
tables = cursor.fetchall()
print(f"Found {len(tables)} tables: {[t[0] for t in tables]}")

# Check views
cursor.execute("""
    SELECT view_name FROM user_views
    WHERE view_name LIKE '%_DV'
    ORDER BY view_name
""")
views = cursor.fetchall()
print(f"Found {len(views)} duality views: {[v[0] for v in views]}")

cursor.close()
conn.close()
```

### Verify Vector Support

```python
import oracledb

conn = oracledb.connect(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1"
)
cursor = conn.cursor()

# Check Oracle version
cursor.execute("SELECT * FROM V$VERSION WHERE BANNER LIKE '%23ai%' OR BANNER LIKE '%26ai%'")
version = cursor.fetchone()
if version:
    print(f"✓ Oracle version: {version[0]}")
else:
    print("⚠ Oracle 23ai+ not detected")

# Test VECTOR datatype
try:
    cursor.execute("CREATE TABLE test_vectors (id NUMBER, vec VECTOR(1536, FLOAT32))")
    cursor.execute("DROP TABLE test_vectors")
    print("✓ VECTOR datatype supported")
except Exception as e:
    print(f"✗ VECTOR datatype not available: {e}")

cursor.close()
conn.close()
```

---

## Troubleshooting

### Oracle Connection Issues

**Problem:** Cannot connect to Oracle database

**Solutions:**
1. Verify Oracle container is running:
   ```bash
   docker ps | grep oracle-memorizz
   ```

2. Check container logs for errors:
   ```bash
   docker logs oracle-memorizz
   ```

3. Verify connection details:
   - Host: `localhost`
   - Port: `1521`
   - Service Name: `FREEPDB1`
   - Password matches what you set

4. Test connection manually:
   ```python
   import oracledb
   try:
       conn = oracledb.connect(
           user="system",
           password="MyPassword123!",
           dsn="localhost:1521/FREEPDB1"
       )
       print("✓ Connection successful!")
       conn.close()
   except Exception as e:
       print(f"✗ Connection failed: {e}")
   ```

### Apple Silicon (M1/M2/M3) Issues

**Problem:** Oracle container fails to start or runs very slowly

**Solutions:**
1. Use platform flag for emulation:
   ```bash
   export PLATFORM_FLAG="--platform linux/amd64"
   memorizz oracle install --image lite
   ```

2. Verify Docker Desktop is using Rosetta 2 (if available):
   - Docker Desktop → Settings → General → Use Rosetta for x86/amd64 emulation

3. Performance note: Emulation may be slower than native. Consider using a cloud Oracle instance for better performance on Apple Silicon.

### Vector Index Creation Fails

**Problem:** Vector indexes fail to create during initialization

**Solutions:**
1. Verify Oracle version supports VECTOR datatype (23ai+ required)
2. Use lazy index creation:
   ```python
   config = OracleConfig(
       user="memorizz_user",
       password="SecurePass123!",
       dsn="localhost:1521/FREEPDB1",
       lazy_vector_indexes=True  # Create indexes on first use
   )
   ```
3. Check user has CREATE INDEX privilege
4. Verify sufficient tablespace available

### ORA-02236: invalid file name

**Problem:** Setup script reports `ORA-02236: invalid file name` while creating the default tablespace. This happens when Oracle Managed Files isn't configured, so Oracle needs an explicit datafile location.

**Solutions:**
1. Tell the setup script where datafiles live inside the Oracle server/container:
   ```bash
   # Directory that already contains other datafiles (inside the container/DB host)
   export ORACLE_DATAFILE_DIR="/opt/oracle/oradata/FREEPDB1"

   # Or supply the full path you want to use
   export ORACLE_TABLESPACE_DATAFILE="/opt/oracle/oradata/FREEPDB1/memorizz_ts01.dbf"
   ```
2. (Optional) Override the tablespace name or size:
   ```bash
   export ORACLE_TABLESPACE_NAME="MEMORIZZ_TS"
   export ORACLE_TABLESPACE_SIZE_MB="200"
   export ORACLE_TABLESPACE_AUTOEXTEND_MB="25"
   ```
3. Re-run `python -m memorizz.memory_provider.oracle.setup` (or `memorizz oracle setup`).

If you run Oracle outside Docker, update the paths to match the server's filesystem. The script will now reuse that path when creating the tablespace.

### Schema Creation Issues

**Problem:** Tables or views fail to create

**Solutions:**
1. Verify user has required privileges:
   ```sql
   SELECT * FROM USER_SYS_PRIVS WHERE USERNAME = 'MEMORIZZ_USER';
   ```

2. Check for existing objects:
   ```sql
   SELECT table_name FROM user_tables;
   SELECT view_name FROM user_views;
   ```

3. Drop and recreate user (if needed):
   ```bash
   python examples/setup_oracle_user.py
   # Script will drop existing user and recreate
   ```

### Embedding Dimension Mismatch

**Problem:** Dimension mismatch errors when storing embeddings

**Solutions:**
1. Check current embedding dimensions:
   ```python
   from memorizz.embeddings import get_embedding_dimensions
   dims = get_embedding_dimensions()
   print(f"Current dimensions: {dims}")
   ```

2. Ensure consistent dimensions across configuration:
   ```python
   embedding_config={
       "model": "text-embedding-3-small",
       "dimensions": 1536  # Explicitly set dimensions
   }
   ```

### Performance Issues

**Problem:** Slow queries or high memory usage

**Solutions:**
1. Increase connection pool size:
   ```python
   config = OracleConfig(
       pool_min=2,
       pool_max=10,
       pool_increment=2
   )
   ```

2. Rebuild vector indexes for better accuracy:
   ```sql
   ALTER INDEX idx_tablename_vec REBUILD
   ORGANIZATION NEIGHBOR PARTITIONS
   DISTANCE COSINE
   WITH TARGET ACCURACY 99;
   ```

3. Gather table statistics:
   ```sql
   EXEC DBMS_STATS.GATHER_TABLE_STATS('MEMORIZZ_USER', 'CONVERSATION_MEMORY');
   ```

---

## Quick Restart

If you've already set up MemoRizz and just need to restart Oracle:

```bash
# Start existing Oracle container (data persists via Docker volume)
docker start oracle-memorizz

# Or use the helper command
memorizz oracle install --image lite  # Detects and starts the existing container

# Verify it's running
docker ps | grep oracle-memorizz
```

**Your data persists** between restarts thanks to the Docker volume (`oracle-memorizz-data`).

To completely reset (⚠️ **WARNING**: This deletes all data):
```bash
# Stop and remove container
docker stop oracle-memorizz
docker rm oracle-memorizz

# Remove volume (deletes all data)
docker volume rm oracle-memorizz-data

# Start fresh
memorizz oracle install --image lite
memorizz oracle setup
```

---

## Next Steps

After completing the setup:

1. **Explore Examples:**
   - [Local Oracle Agent](examples/single_agent/memagent_local_oracle.ipynb)
   - [Remote Oracle Agent](examples/single_agent/memagent_remote_oracle.ipynb)
   - [Deep Research](examples/deep_research/deep_research_memagent.ipynb)
   - [Continual Learning](examples/continual_learning/continual_learning_guide.ipynb)

2. **Read Documentation:**
   - [Main README](README.md)
   - [Oracle Provider README](src/memorizz/memory_provider/oracle/README.md)
   - [Memory Architecture](src/memorizz/MEMORY_ARCHITECTURE.md)

3. **Build Your First Agent:**
   ```python
   from memorizz.memagent.builders import MemAgentBuilder
   from memorizz.memory_provider.oracle import OracleProvider, OracleConfig

   # Your agent code here...
   ```

---

## Additional Resources

- **Oracle Database 23ai Documentation:** [oracle.com](https://docs.oracle.com/en/database/oracle/oracle-database/23ai/)
- **Oracle AI Vector Search:** [Vector Search Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23ai/vect-search.html)
- **Python Oracle Driver:** [python-oracledb](https://python-oracledb.readthedocs.io/)
- **MemoRizz GitHub:** [github.com/RichmondAlake/memorizz](https://github.com/RichmondAlake/memorizz)

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the [Oracle Provider README](src/memorizz/memory_provider/oracle/README.md)
3. Check example notebooks in the `examples/` directory
4. Open an issue on [GitHub](https://github.com/RichmondAlake/memorizz/issues)

---

**Happy coding with MemoRizz! 🧠**
