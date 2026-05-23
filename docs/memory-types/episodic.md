# Episodic Memory

Episodic memory chronicles every interaction an agent has with users, teammates, or tools. It lives under `src/memorizz/long_term/episodic/` and fulfills both `MemoryType.CONVERSATION_MEMORY` and `MemoryType.SUMMARIES`.

## Structure

- **Conversation Memory Units** – Raw transcripts with timestamps, speaker metadata, and embeddings for semantic retrieval.
- **Summaries** – Periodic rollups that compress older chunks to keep prompts small while retaining context.

## Example

```python
agent.memory.conversation_memory.add_message(
    role="user",
    content="Can you remind me of the Oracle setup steps?",
)

agent.memory.summaries.create_or_update(
    topic="setup",
    content="User configured Oracle last week and is stuck on connection pooling.",
)
```

## Use Cases

- Long-running assistants that must reference previous sessions
- Relationship and preference tracking for customer success bots
- Auditable records of how multi-agent systems reached a decision

Pair episodic memory with semantic cache or working memory to prioritize the most relevant snippets for a given prompt window.
