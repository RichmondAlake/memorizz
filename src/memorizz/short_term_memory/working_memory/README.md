# Working Memory

Working memory controls what is actively in-context during a run.

In Memorizz, this is represented by:

- `CWM` prompt guidance (`CWM.get_prompt_from_memory_types(...)`)
- runtime context assembly inside `MemAgent`
- token usage tracking via `get_context_window_stats()`

## CWM Prompt Guidance

```python
from memorizz.enums import MemoryType
from memorizz.short_term_memory.working_memory.cwm import CWM

prompt = CWM.get_prompt_from_memory_types(
    [
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.LONG_TERM_MEMORY,
        MemoryType.SUMMARIES,
    ]
)

print(prompt)
```

## Runtime Context + Stats

```python
from pathlib import Path

from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_instruction("Use available memory context efficiently.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

agent.run("Summarize my project updates from earlier this week.", memory_id="user-42")
stats = agent.get_context_window_stats()
print(stats)
```

## Practical Guidance

- Keep `memory_id` stable per user/thread for coherent context reuse.
- Use summaries (`generate_summaries`) to reduce long-history prompt pressure.
- Monitor token usage in production with `get_context_window_stats()` telemetry.
