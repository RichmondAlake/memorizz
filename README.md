<div align="center">

# Memorizz

[![PyPI version](https://badge.fury.io/py/memorizz.svg)](https://badge.fury.io/py/memorizz)
[![PyPI Downloads](https://static.pepy.tech/badge/memorizz)](https://pepy.tech/projects/memorizz)

</div>

> **Experimental software**
>
> Memorizz is an educational/experimental framework. APIs may change and the project has not undergone security hardening for production workloads.

Memorizz is a Python framework for building memory-augmented AI agents.
It provides:

- multiple memory systems (episodic, semantic, procedural, short-term, shared)
- pluggable storage providers (Oracle, MongoDB, filesystem)
- agent builders and application modes (`assistant`, `workflow`, `deep_research`)
- scheduled automations (cron, interval, one-shot) with optional WhatsApp delivery
- optional internet access, sandbox code execution, and local web UI

## Key Capabilities

- **Persistent memory** across sessions and conversations
- **Semantic retrieval** with embeddings + vector search
- **Entity memory** tools for profile-style facts (`entity_memory_lookup` / `entity_memory_upsert`)
- **Tool calling** with automatic function registration
- **Semantic cache** to reduce repeat LLM calls
- **Multi-agent orchestration** with shared blackboard memory
- **Context-window telemetry** via `get_context_window_stats()`
- **Scheduled automations** via SDK, web UI, or agent conversation (see `src/memorizz/automation/README.md`)

## Installation

Base install:

```bash
pip install memorizz
```

Common extras:

```bash
pip install "memorizz[oracle]"          # Oracle provider
pip install "memorizz[mongodb]"         # MongoDB provider
pip install "memorizz[filesystem]"      # Local filesystem + FAISS
pip install "memorizz[sandbox-e2b]"     # E2B sandbox
pip install "memorizz[sandbox-daytona]" # Daytona sandbox
pip install "memorizz[ui]"              # Local web UI
pip install "memorizz[all]"             # Everything
```

## Quick Start (Filesystem Provider)

```python
import os
from pathlib import Path

from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

provider = FileSystemProvider(
    FileSystemConfig(
        root_path=Path("~/.memorizz").expanduser(),
        embedding_provider="openai",
        embedding_config={"model": "text-embedding-3-small"},
    )
)

agent = (
    MemAgentBuilder()
    .with_instruction("You are a helpful assistant with persistent memory.")
    .with_memory_provider(provider)
    .with_llm_config(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
        }
    )
    .with_semantic_cache(enabled=True, threshold=0.85)
    .build()
)

print(agent.run("Hi, my name is Leah and I work on payments systems."))
print(agent.run("What did I tell you about my work?"))

stats = agent.get_context_window_stats()
print(stats)
```

## Oracle Setup (Optional)

If you want Oracle AI Database as the backing store:

```bash
./install_oracle.sh
memorizz setup-oracle
```

Then configure `ORACLE_USER`, `ORACLE_PASSWORD`, `ORACLE_DSN`, and your LLM credentials. Full setup details are in `SETUP.md`.

For multi-client consistency (UI + notebooks), you can set shared embedding defaults:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=1536
```

## Application Modes

`ApplicationMode` presets automatically enable different memory stacks:

- `assistant`: conversation, long-term, personas, entity memory, short-term, summaries
- `workflow`: workflow memory, toolbox, long-term, short-term, summaries
- `deep_research`: toolbox, shared memory, long-term, short-term, summaries

Example:

```python
import os

from memorizz.enums import ApplicationMode
from memorizz.memagent.builders import MemAgentBuilder

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"],
}

agent = (
    MemAgentBuilder()
    .with_application_mode(ApplicationMode.DEEP_RESEARCH)
    .with_memory_provider(provider)
    .with_llm_config(llm_config)
    .build()
)
```

## Internet Access (Deep Research)

Deep Research agents can attach internet providers and expose `internet_search` / `open_web_page` tools.

```python
import os

from memorizz.internet_access import TavilyProvider
from memorizz.memagent.builders import create_deep_research_agent

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"],
}

internet_provider = TavilyProvider(api_key=os.environ["TAVILY_API_KEY"])

agent = (
    create_deep_research_agent(internet_provider=internet_provider)
    .with_memory_provider(provider)
    .with_llm_config(llm_config)
    .build()
)

results = agent.search_internet("latest vector database benchmark")
```

## Sandbox Code Execution

Attach a sandbox provider to enable `execute_code`, `sandbox_write_file`, and `sandbox_read_file` tools.

```python
import os

from memorizz.memagent import MemAgent

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"],
}

agent = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    sandbox_provider="e2b",  # or "daytona" / "graalpy"
)

print(agent.execute_code("print(2 ** 16)"))
```

## Multi-Agent Deep Research Workflow

```python
from memorizz.memagent.orchestrators import DeepResearchWorkflow

workflow = DeepResearchWorkflow.from_config(
    memory_provider=provider,
    delegate_instructions=[
        "Financial researcher: collect metrics and citations.",
        "Risk analyst: identify key downside scenarios.",
    ],
)

report = workflow.run("Analyze the last 3 years of cloud infrastructure trends.")
print(report)
```

## CLI

After installation, the `memorizz` command exposes:

```bash
memorizz run local                  # start local web UI (requires [ui])
memorizz install-oracle             # start Oracle container helper
memorizz setup-oracle               # initialize Oracle schema/user
```

## Examples

- `examples/single_agent/memagent_local_oracle.ipynb`
- `examples/single_agent/memagent_remote_oracle.ipynb`
- `examples/deep_research/deep_research_memagent.ipynb`
- `examples/sandbox/memagent_e2b_sandbox.ipynb`
- `examples/sandbox/memagent_daytona_sandbox.ipynb`
- `examples/sandbox/memagent_graalpy_sandbox.ipynb`
- `examples/automations/automations_guide.ipynb`
- `examples/model_providers/openai_provider.ipynb`
- `examples/model_providers/anthropic_provider.ipynb`
- `examples/model_providers/ollama_provider.ipynb`
- `examples/model_providers/compare_providers.ipynb`

## Documentation

- Docs source: `docs/`
- Local preview: `make docs-serve` (or `mkdocs serve`)
- Architecture notes: `src/memorizz/MEMORY_ARCHITECTURE.md`

## License

PolyForm Noncommercial 1.0.0.
See `LICENSE` and `NOTICE`.
