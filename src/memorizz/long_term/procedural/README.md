# Procedural Memory

Procedural memory captures how an agent acts and executes tasks.

It is implemented through:

- `Toolbox` (`MemoryType.TOOLBOX`) for callable tools
- workflow trace persistence (`MemoryType.WORKFLOW_MEMORY`) generated during agent runs
- `Skillbox` (`MemoryType.SKILLBOX`) for validated, versioned procedures
  distilled from repeated successful trajectories

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

## Continual Learning

```python
agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_continual_learning(
        True,
        config={
            "require_shadow": True,
            "skill_injection_role": "developer",  # "user" is the default
        },
    )
    .build()
)
```

Developer-authority skills require shadow review and explicit activation.
OpenAI receives a developer message; Anthropic and providers without a native
developer role receive the reviewed procedure through their system-equivalent
path. Filesystem, MongoDB, and Oracle all persist the skill authority. Raw
workflow traces remain evidence/audit records and are not automatically added
to prompts.

## Related Modules

- `src/memorizz/long_term/procedural/toolbox/`
- `src/memorizz/long_term/procedural/workflow/`
- `src/memorizz/long_term/procedural/skillbox/`
- `docs/guides/continual-learning.md`
