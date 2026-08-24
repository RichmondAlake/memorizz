# Episodic Memory

Episodic memory records what happened: user and assistant messages, thread
identity, timestamps, tool-related context, and summaries that link back to
their source messages. It is represented by `CONVERSATION_MEMORY` and
`SUMMARIES`.

## Write through the agent runtime

Normal applications should let `MemAgent.run()` write conversation records so
the same tenant, thread, trace, cache, and compaction policy is applied:

```python
scope = {
    "memory_id": "support",
    "user_id": "user-42",
    "thread_id": "ticket-918",
}

agent.run("I am configuring Oracle connection pooling.", **scope)
agent.run("Remember that this deployment uses a private endpoint.", **scope)
```

Use the provider for administrative inspection, with the same exact scope:

```python
from memorizz import MemoryType

history = provider.retrieve_conversation_history_ordered_by_timestamp(
    memory_id="support",
    memory_type=MemoryType.CONVERSATION_MEMORY,
    user_id="user-42",
    thread_id="ticket-918",
)
```

Do not use a provider-wide, unscoped history read in a user-facing request.

## Summaries and compaction

Summary generation accepts public scope arguments and preserves the source
message links required for lossless expansion:

```python
summary_ids = agent.generate_summaries(
    days_back=7,
    max_memories_per_summary=50,
    memory_id="support",
    user_id="user-42",
    thread_id="ticket-918",
)
```

Stored summaries include their period, source-message IDs, and memory-unit
count. Original conversation rows receive a summary marker atomically where
the provider supports the operation. Summarization reduces prompt context; it
does not erase the original history. Use the provider's scoped lifecycle API
when records must actually be deleted.

## Use episodic memory for

- conversation continuity across sessions;
- auditable interaction history;
- source-linked context compaction;
- outcome and workflow evidence that depends on what happened in a thread.

Pair it with semantic memory for canonical facts and with
[context efficiency](../guides/context-efficiency.md) to control what enters
each prompt.
