# MongoDB Provider

The MongoDB provider offers a lightweight starting point for experimentation or hosted Atlas deployments. It is implemented in `src/memorizz/memory_provider/mongodb/`.

## Installation

```bash
pip install -e ".[mongodb]"
```

## Configuration

```python
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig

provider = MongoDBProvider(MongoDBConfig(
    uri=os.environ["MONGODB_URI"],
    database="memorizz",
    collection_prefix="agents",
))
```

Collections are created lazily (e.g., `agents_personas`, `agents_knowledge_base`). Each document stores:

- Serialized payload (`data`)
- Embedding vectors (array fields you can index with MongoDB Atlas Vector Search)
- Agent + namespace metadata

## Atlas Vector Search

1. Enable the [Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/) preview on your cluster.
2. Create an index per collection referencing the embedding field.
3. Configure the provider with your embedding model dimensions.

## When to Choose MongoDB

- Prototype agents without running Oracle locally
- Serverless / hosted deployments where MongoDB Atlas is already approved
- Horizontal scaling scenarios using MongoDB's built-in sharding

Use MongoDB for agility and switch to Oracle when you need stronger relational guarantees or AI Vector Search optimizations.
