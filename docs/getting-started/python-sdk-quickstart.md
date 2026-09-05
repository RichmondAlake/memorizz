# Python SDK Quickstart

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](../guides/streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

This quickstart builds a tenant-scoped agent whose definition and conversation
survive process restarts.

## 1. Install and configure a model

```bash
python -m pip install memorizz
export OPENAI_API_KEY="your-openai-api-key"
```

For a fully local setup, install `memorizz[local]` and use Ollama. See
[Installation](installation.md) for every provider extra.

## 2. Build and persist the agent

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Developer assistant")
    .with_instruction(
        "Help with software questions. Reuse relevant preferences and explain uncertainty."
    )
    .with_llm_config(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
        }
    )
    .with_memory_ids("developer-demo")
    .with_semantic_cache(enabled=True, threshold=0.85)
    .build_and_save()
)
```

No memory provider was passed, so Memorizz uses filesystem storage beneath
`~/.memorizz/memory`. `build()` creates a runnable agent; `build_and_save()`
also persists its definition so the CLI, UI, MCP server, or a later process can
load it. Credentials remain in the environment and are not part of
`llm_config`.

## 3. Run a scoped conversation

```python
scope = {
    "memory_id": "developer-demo",
    "user_id": "user-42",
    "thread_id": "onboarding",
}

agent.run("I prefer concise Python examples.", **scope)
answer = agent.run("How should you explain an API client to me?", **scope)
print(answer)
```

Reuse the same `user_id` and `thread_id` for continuity. Use a different scope
for another tenant or conversation. Do not derive security-sensitive tenant
scope from model-generated tool arguments.

## 4. Inspect and close

```python
print(agent.capability_report())
print(agent.semantic_cache_stats())
print(agent.get_context_window_stats())

agent.close()
```

For long-lived services, make lifecycle ownership explicit:

```python
with agent.lifecycle(close_memory_provider=True):
    print(agent.run("Summarize my preferences.", **scope))
```

Do not use the agent again after closing its provider.

## 5. Restore it later

```python
from memorizz import MemAgent

restored = MemAgent.load(agent.agent_id)
try:
    print(restored.run("What style of examples do I prefer?", **scope))
finally:
    restored.close()
```

If you construct an explicit provider, pass the same provider to
`MemAgent.load(...)`. A saved `agent_id` is discoverable only in the provider
where it was stored.

## Use an explicit filesystem provider

Configure a provider directly when you need a different path, embedding model,
or FAISS policy:

```python
from pathlib import Path

from memorizz import FileSystemConfig, FileSystemProvider, MemAgentBuilder

provider = FileSystemProvider(
    FileSystemConfig(
        root_path=Path("./var/memorizz"),
        embedding_provider="openai",
        embedding_config={"model": "text-embedding-3-small"},
    )
)

agent = (
    MemAgentBuilder()
    .with_name("Explicit provider example")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_provider(provider)
    .with_memory_ids("developer-demo")
    .build_and_save()
)
```

Without an embedding provider, filesystem persistence still works and semantic
operations use their documented exact or lexical fallback where available.
Test retrieval quality before relying on that fallback in an application.

## Next steps

- Add [typed tools and durable approval](../guides/tools-and-approvals.md).
- Review [multi-tenant isolation](../guides/multi-tenant.md).
- Connect an [MCP server](../guides/mcp-connectivity.md).
- Run workspace tasks through the [memory-first meta-harness](../guides/meta-harness.md).
- Configure [observability and trace inspection](../observability-ui.md).
- Choose [MongoDB or Oracle](../memory-providers/filesystem.md) for a deployed backend.
