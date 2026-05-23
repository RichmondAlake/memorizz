# Assistant Mode

Assistant mode is the default conversational setup for MemoRizz. It prioritizes continuity, personalization, and a rich memory stack so users feel like they're chatting with the same agent every time.

## Memory Stack

- `MemoryType.CONVERSATION_MEMORY`
- `MemoryType.KNOWLEDGE_BASE` + `MemoryType.ENTITY_MEMORY`
- `MemoryType.PERSONAS`
- `MemoryType.SHORT_TERM_MEMORY`
- `MemoryType.SUMMARIES`

## Configuration

```python
from memorizz.enums import ApplicationMode
from memorizz.memagent.builders import MemAgentBuilder

agent = (MemAgentBuilder()
    .with_application_mode(ApplicationMode.ASSISTANT)
    .with_memory_provider(provider)
    ...
    .build())
```

## Tips

- Seed personas with voice/tone guidelines and safety rails.
- Use entity memory to store user preferences (e.g., "prefers dark mode UI").
- Enable semantic cache for repeated Q&A answers to cut LLM costs.

Assistant mode is ideal for customer support, onboarding companions, or internal help desks.
