# Semantic Memory

Semantic memory stores facts, concepts, and structured entity knowledge that should persist beyond a single conversation.

Key components:

- `KnowledgeBase`
- `Persona`
- `EntityMemory`

## Knowledge Base Example

```python
from pathlib import Path

from memorizz.long_term_memory.semantic import KnowledgeBase
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
kb = KnowledgeBase(provider)

ltm_id = kb.ingest_knowledge(
    corpus="Python supports multiple paradigms including object-oriented and functional programming.",
    namespace="python_basics",
)

results = kb.retrieve_knowledge_by_query("What paradigms does Python support?", namespace="python_basics", limit=3)
print(results)
```

## Entity Memory Example

```python
from memorizz.long_term_memory.semantic.entity_memory import EntityMemory

entities = EntityMemory(provider)

entity_id = entities.upsert_entity(
    name="Leah",
    entity_type="user",
    memory_id="tenant-1",
    attributes=[{"name": "preferred_language", "value": "Python", "confidence": 0.95}],
)

profile = entities.get_entity_profile(entity_id)
```

## With MemAgent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_instruction("Use stored knowledge and entity facts in responses.")
    .with_memory_provider(provider)
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

agent.with_entity_memory(True)
response = agent.run("What do you know about Leah's preferred language?")
```

Notes:

- Semantic entries are stored under `MemoryType.LONG_TERM_MEMORY`.
- Entity profiles are stored under `MemoryType.ENTITY_MEMORY`.
