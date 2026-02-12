# Persona Module

`Persona` defines an agent's stable identity: name, role, goals, and background.

Canonical imports:

```python
from memorizz.long_term_memory.semantic.persona import Persona, RoleType
```

## Quick Start

```python
from pathlib import Path

from memorizz.long_term_memory.semantic.persona import Persona, RoleType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))

persona = Persona(
    name="TechExpert",
    role=RoleType.TECHNICAL_EXPERT,
    goals="Help developers debug production incidents quickly.",
    background="10+ years across backend systems and incident response.",
)

persona_id = persona.store_persona(provider)

same_persona = Persona.retrieve_persona(persona_id, provider)
similar = Persona.get_most_similar_persona("senior backend mentor", provider, limit=1)
all_personas = Persona.list_personas(provider)
```

## Use with MemAgent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_persona(persona)
    .with_instruction("You are concise, practical, and technical.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

# Persona can also be changed at runtime
agent.set_persona(persona)
```

## Notes

- Personas are persisted under `MemoryType.PERSONAS`.
- The role can be a `RoleType` enum or a compatible role string.
