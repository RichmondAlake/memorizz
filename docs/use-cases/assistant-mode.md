# Assistant Mode

Assistant mode prioritizes conversation continuity, personalization, and
source-linked compaction. It enables conversation, knowledge, persona, entity,
short-term, and summary memory by default.

## Build an assistant

```python
from memorizz import ApplicationMode, MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Onboarding assistant")
    .with_instruction("Help users onboard and distinguish facts from guesses.")
    .with_application_mode(ApplicationMode.ASSISTANT)
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_ids("onboarding")
    .with_semantic_cache(enabled=True, threshold=0.88)
    .build_and_save()
)

answer = agent.run(
    "Remember that I prefer dark mode.",
    memory_id="onboarding",
    user_id="user-42",
    thread_id="first-run",
)
```

## Design guidance

- Put stable user or organization facts in entity memory; keep one-off turn
  details in conversation memory.
- Keep persona changes versioned and auditable. A transient user instruction
  should not rewrite the agent's durable identity.
- Set semantic-cache freshness by domain, and invalidate cached policy or
  catalog answers when the source changes.
- Generate summaries with explicit memory, user, and thread scope.
- Add external tools only with typed schemas and appropriate approval policy.

Assistant mode is a good starting point for support, onboarding, and internal
help desks. Use workflow mode when repeatable tool execution matters more than
conversational history.
