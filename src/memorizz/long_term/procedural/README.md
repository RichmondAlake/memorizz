# Procedural Memory

Procedural memory captures how an agent acts and executes tasks.

It is implemented through:

- `Toolbox` (`MemoryType.TOOLBOX`) for callable tools
- workflow trace persistence (`MemoryType.WORKFLOW_MEMORY`) generated during agent runs

## Typical Usage

```python
from pathlib import Path

from memorizz.enums import ApplicationMode, MemoryType
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))

def lookup_ticket(ticket_id: str) -> str:
    """Return a mock ticket status."""
    return f"Ticket {ticket_id}: in_progress"

agent = (
    MemAgentBuilder()
    .with_application_mode(ApplicationMode.WORKFLOW)
    .with_memory_provider(provider)
    .with_instruction("Automate ticket workflows.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .with_tool(lookup_ticket)
    .build()
)

result = agent.run("Check ticket A-42 and summarize next actions")
```

## Inspect Stored Workflow Memory

```python
workflows = provider.list_all(MemoryType.WORKFLOW_MEMORY)
print(f"Stored workflows: {len(workflows)}")
```

## Related Modules

- `src/memorizz/long_term/procedural/toolbox/`
- `src/memorizz/long_term/procedural/workflow/`
