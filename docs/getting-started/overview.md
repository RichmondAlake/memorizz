# Overview

Memorizz helps you build AI agents that can remember, retrieve, and coordinate over time. It combines a configurable memory architecture with provider-backed persistence so agents can run beyond a single prompt window.

## Core Building Blocks

```text
src/memorizz/
├── memagent/                # Agent runtime + builder APIs
├── long_term_memory/        # semantic, procedural, episodic memory systems
├── short_term_memory/       # working memory + semantic cache
├── coordination/            # shared memory for multi-agent workflows
├── memory_provider/         # Oracle, MongoDB, filesystem, custom providers
├── internet_access/         # Tavily / Firecrawl / offline web providers
├── sandbox/                 # E2B / Daytona / GraalPy execution providers
└── ui/                      # Local FastAPI-based web UI
```

## Key Capabilities

| Capability | What You Get |
|---|---|
| Persistent agent state | Conversations, summaries, tools, and agent config persisted in a provider |
| Memory-mode presets | `assistant`, `workflow`, and `deep_research` mode defaults |
| Semantic retrieval | Embedding-based similarity search for relevant memory recall |
| Operational tooling | Semantic cache, context-window stats, and auto-summarization support |
| Extensibility | Custom memory providers and custom internet/sandbox providers |

## Requirements

- Python 3.7+
- At least one LLM provider (for example OpenAI)
- A persistence backend (filesystem, Oracle, MongoDB, or custom `MemoryProvider`)

## Next Steps

1. Follow [Python SDK Quickstart](python-sdk-quickstart.md) to run your first agent.
2. Use the [Local UI Guide](local-ui.md) if you prefer setting up and operating agents from the browser.
3. Review [Concepts](concepts.md) for memory and mode mappings.
4. Pick a backend under [Memory Providers](../memory-providers/filesystem.md).
