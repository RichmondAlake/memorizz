# Python SDK Quickstart

This quickstart shows the fastest path to a persistent Memorizz agent.

## 1. Install

For local development against this repo:

```bash
pip install -e ".[filesystem]"
```

For PyPI usage:

```bash
pip install "memorizz[filesystem]"
```

If you prefer Oracle or MongoDB, install `"memorizz[oracle]"` or `"memorizz[mongodb]"` instead.

## 2. Configure Credentials

Set your LLM credentials (example with OpenAI):

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 3. Build a Stateful Agent

```python
from pathlib import Path

from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(
    FileSystemConfig(
        root_path=Path("~/.memorizz").expanduser(),
        embedding_provider="openai",
        embedding_config={"model": "text-embedding-3-small"},
    )
)

agent = (
    MemAgentBuilder()
    .with_instruction("You are a helpful assistant with memory.")
    .with_memory_provider(provider)
    .with_llm_config(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "your-openai-api-key",
        }
    )
    .with_semantic_cache(enabled=True, threshold=0.85)
    .build()
)
```

## 4. Run a Conversation

```python
memory_id = "demo-user-001"

print(agent.run("Hi, I'm Sam and I work in developer relations.", memory_id=memory_id))
print(agent.run("What do I do for work?", memory_id=memory_id))

stats = agent.get_context_window_stats()
print(stats)
```

## 5. Optional: Oracle Backend

```bash
./install_oracle.sh
memorizz setup-oracle
```

Then swap the provider to `OracleProvider(OracleConfig(...))`. See `docs/memory-providers/oracle.md`.

## Next Steps

- Explore memory behavior in [Memory Types](../memory-types/semantic.md).
- Add internet tooling for research with [Internet Access Providers](../internet-access/providers.md).
- Enable code execution with [Sandbox Providers](../sandbox/index.md).
