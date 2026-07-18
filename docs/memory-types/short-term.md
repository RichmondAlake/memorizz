# Short-Term Memory

Short-term memory is the agent's active workspace. MemoRizz separates it into a semantic cache and a working-memory controller located under `src/memorizz/short_term_memory/`.

## Semantic Cache (`MemoryType.SEMANTIC_CACHE`)

- Stores short-lived key/value pairs with embeddings for fast similarity matches.
- Ideal for caching expensive LLM responses, transient API payloads, or session-only facts.
- Automatically expires or can be explicitly cleared when you rotate sessions.

```python
agent.memory.semantic_cache.save(
    key="oracle_setup_docs",
    value={"answer": "Install the provider, then run memorizz oracle setup"},
)
```

## Working Memory (`MemoryType.SHORT_TERM_MEMORY`)

- Tracks the active conversation window across all memory sources.
- Manages token budgets by summarizing or truncating inputs before they reach the LLM.
- Responsible for stitching retrieved semantic, episodic, and procedural memories into a cohesive prompt.

```python
window = agent.memory.short_term.window_for(agent_id=agent.id)
window.push_user_message("Give me the highlights from yesterday's sync.")
```

Short-term memory keeps the agent grounded in the current turn while semantic + episodic stores provide long-term continuity.
