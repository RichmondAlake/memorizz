# Entity Memory Module

Entity memory provides structured long-term storage for facts about specific people,
organizations, products, or other named entities. Each entity is stored as a record of
attribute–value pairs plus optional relations to other entities so agents can recall and
update stable facts over time.

## Features

- Store entities with typed attributes, confidence scores, provenance, and timestamps
- Link entities together via labeled relations (e.g., *coworker*, *purchased*)
- Vector-searchable using the combined attribute text for natural-language lookup
- Memory-ID aware so facts can be scoped to a specific user, tenant, or agent
- Convenience helpers for recording single attributes, retrieving profiles, and
  attaching relations

## Usage

```python
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig

provider = MongoDBProvider(MongoDBConfig("mongodb://localhost:27017"))
entity_store = EntityMemory(provider)

# Create or update an entity
entity_id = entity_store.upsert_entity(
    name="Avery Stone",
    entity_type="customer",
    memory_id="tenant-123",
    attributes=[{"name": "preferred_language", "value": "Japanese", "confidence": 0.95}],
)

# Record a new fact without building the full payload
entity_store.record_attribute(
    entity_id=entity_id,
    attribute_name="favorite_product",
    attribute_value="Nebula Pro Drone",
    source="support_chat",
)

# Look up relevant entities for a query
matches = entity_store.search_entities("user who likes the drone", memory_id="tenant-123")
```

The module intentionally mirrors the layout of other long-term memory components (such
as the knowledge base and persona modules) so it can be attached to `MemAgent`
instances or used standalone.
