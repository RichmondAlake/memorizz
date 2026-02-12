# Coordination Memory

Coordination memory enables multiple agents to share state through `SharedMemory`.

It is primarily used by `MultiAgentOrchestrator` and `DeepResearchOrchestrator`.

## Shared Session Basics

```python
from pathlib import Path

from memorizz.coordination import SharedMemory
from memorizz.coordination.shared_memory import SharedMemoryMessageType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
shared = SharedMemory(provider)

session_id = shared.create_shared_session(
    root_agent_id="root-agent",
    delegate_agent_ids=["research-agent", "analysis-agent"],
)

shared.post_command(
    memory_id=session_id,
    agent_id="root-agent",
    command_id="cmd-1",
    target_agent_id="research-agent",
    instructions="Collect three sources on vector databases.",
)

shared.post_status(
    memory_id=session_id,
    agent_id="research-agent",
    command_id="cmd-1",
    status="in_progress",
    progress=50,
)

shared.post_report(
    memory_id=session_id,
    agent_id="research-agent",
    command_id="cmd-1",
    findings="Collected 3 sources with benchmark notes.",
)

reports = shared.list_messages(session_id, SharedMemoryMessageType.REPORT)
```

## Orchestrated Multi-Agent Workflow

```python
from memorizz.memagent.orchestrators import DeepResearchWorkflow

workflow = DeepResearchWorkflow.from_config(
    memory_provider=provider,
    delegate_instructions=[
        "Research benchmarks",
        "Analyze tradeoffs",
    ],
)

final_report = workflow.run("Compare vector database indexing strategies.")
```

## Notes

- Shared session data is stored under `MemoryType.SHARED_MEMORY`.
- Message helpers (`COMMAND`, `STATUS`, `REPORT`) provide a structured coordination protocol.
