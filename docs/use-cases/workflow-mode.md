# Workflow Mode

Workflow mode emphasizes typed tools, workflow traces, knowledge, short-term
state, and summaries for repeatable operational tasks.

## Build a read-only workflow agent

```python
from memorizz import ApplicationMode, MemAgentBuilder, governed_tool


@governed_tool(deterministic=True, domains=("tickets",))
def ticket_status(ticket_id: str) -> dict:
    """Return the current status of one ticket."""
    return {"ticket_id": ticket_id, "status": "open"}


agent = (
    MemAgentBuilder()
    .with_name("Ticket workflow")
    .with_application_mode(ApplicationMode.WORKFLOW)
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_ids("ticket-operations")
    .with_tools([ticket_status])
    .build_and_save()
)

result = agent.run(
    "Check ticket 12491 and summarize its status.",
    memory_id="ticket-operations",
    user_id="operator-7",
    thread_id="ticket-12491",
)
```

## Design guidance

- Mark mutating or non-deterministic tools explicitly and require durable host
  approval where policy demands it.
- Supply idempotency keys in host/tool context for retryable mutations.
- Record verified application outcomes; the model's claim that a workflow
  succeeded is not sufficient learning evidence.
- Use a deterministic delegation plan for known workflows and inspect partial
  failures/dependency states.
- Enable continual learning only after defining promotion, shadow evaluation,
  demotion, and forgetting policy.

Use shared memory when delegates need a workflow- and user-scoped blackboard.
See [Tools, Safety, and Human Approval](../guides/tools-and-approvals.md) and
[Continual Learning](../guides/continual-learning.md).
