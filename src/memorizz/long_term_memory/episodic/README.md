# Episodic Memory

Episodic memory captures what happened in interactions over time.

Primary structures:

- `ConversationMemoryUnit` for individual turns
- `SummaryComponent` for compressed history snapshots

## Typical MemAgent Flow

```python
from pathlib import Path

from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_instruction("Remember conversation details across turns.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

memory_id = "user-42"
agent.run("My preferred deployment region is us-east-1.", memory_id=memory_id)
agent.run("Also note that I work mostly in Python.", memory_id=memory_id)

history = agent.load_conversation_history(memory_id)
print(len(history))
```

## Generating Summaries

```python
summary_ids = agent.generate_summaries(days_back=7, max_memories_per_summary=50)
print(summary_ids)
```

## Direct Models

```python
from memorizz.long_term_memory.episodic import ConversationMemoryUnit, SummaryComponent

turn = ConversationMemoryUnit(
    role="user",
    content="Please remember my timezone is UTC-5",
    timestamp="2026-02-12T00:00:00",
    memory_id="user-42",
    conversation_id="conv-1",
)

summary = SummaryComponent(
    memory_id="user-42",
    agent_id="agent-1",
    summary_content="User shared timezone preference.",
    period_start=1739328000.0,
    period_end=1739331600.0,
    memory_units_count=4,
)
```

`MemAgent` handles episodic persistence automatically during `run()`.
