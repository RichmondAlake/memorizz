# Workflow Mode

Workflow mode targets deterministic task execution (think onboarding checklists, ticket triage, or knowledge-base upkeep). It favors procedural memory and tools over conversational depth.

## Memory Stack

- `MemoryType.WORKFLOW_MEMORY`
- `MemoryType.TOOLBOX`
- `MemoryType.KNOWLEDGE_BASE`
- `MemoryType.SHORT_TERM_MEMORY`

## Sample Flow

```python
from memorizz.enums import ApplicationMode
from memorizz.memagent.builders import MemAgentBuilder

def process_ticket(ticket_id: str) -> str:
    """Example workflow tool that processes a ticket id."""
    return f"Processed ticket {ticket_id}"

agent = (MemAgentBuilder()
    .with_application_mode(ApplicationMode.WORKFLOW)
    .with_memory_provider(provider)
    .with_tool(process_ticket)
    .build())

agent.run("Process ticket 12491 and summarize the outcome")
```

Workflow mode keeps episodic memory minimal so the agent can stay focused on the currently executing process. Pair it with shared memory if you need a supervisor agent to inspect progress.
