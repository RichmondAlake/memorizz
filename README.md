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
- optional internet access, sandbox code execution, skills marketplace, and local web UI

## Key Capabilities

- **Persistent memory** across sessions and conversations
- **Semantic retrieval** with embeddings + vector search
- **Knowledge base** with file/folder ingestion (`.pdf`, `.md`, `.txt`, `.csv`, `.json`, …) and configurable chunking (`fixed` / `sentence` / `paragraph` / `semantic` / custom). Same extractor registry powers the SDK and the local UI's drag-and-drop uploader; see [`long_term/semantic/README.md`](src/memorizz/long_term/semantic/README.md).
- **Entity memory** tools for profile-style facts (`entity_memory_lookup` / `entity_memory_upsert`)
- **Tool calling** with automatic function registration
- **Semantic cache** to reduce repeat LLM calls
- **Multi-agent orchestration** with shared blackboard memory
- **Context-window telemetry** via `get_context_window_stats()`
- **Skills marketplace** with Vercel Agent Skills and SkillsMP providers
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
pip install "memorizz[huggingface]"     # transformers + sentence-transformers
pip install "memorizz[mlx]"             # Apple-Silicon MLX backend (native arm64 only)
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

Building a multi-user application? Pass `user_id` to isolate memory per
end-user — one agent can serve every tenant in your app. See the
[Multi-Tenant Guide](docs/guides/multi-tenant.md) for the full contract.

```python
agent.run("Remember my favorite color is purple.", user_id="alice")
agent.run("What's my favorite color?", user_id="bob")  # won't see alice's data
```

## Local LLMs (Gemma 4, Llama, Qwen, …)

Memorizz speaks several local-LLM backends so you can run an entire agent
loop without sending tokens to a third-party API. The local UI exposes
all of these in the agent form's **Provider** dropdown.

| Provider value | Backend | Best for | Apple Silicon? |
|---|---|---|---|
| `huggingface` | `transformers` + PyTorch (MPS/CUDA/CPU) | the most-supported path; widest model selection | ✓ via MPS |
| `mlx` | Apple [`mlx-lm`](https://github.com/ml-explore/mlx-lm) | fastest on Macs, lowest memory | ✓ native (required) |
| `local-openai` | any OpenAI-compatible HTTP server (llama.cpp, LM Studio, vLLM) | reusing existing servers; CPU/GGUF; tool-calling on llama.cpp | ✓ |
| `ollama` | Ollama daemon | one-command pulls, integrated model store | ✓ |

> **Gemma 4 is gated.** Accept the license once at
> [huggingface.co/google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
> (or the variant you want) and set `HF_TOKEN` in Settings before
> pulling. The agent form surfaces this hint inline whenever a gated
> repo is selected.

### Path A — Hugging Face Transformers (works everywhere)

```bash
pip install "memorizz[huggingface]"
export HF_TOKEN=hf_...

# In the UI: Agents → New → Provider = HuggingFace,
#           Model = google/gemma-4-E2B-it
# Or via the SDK:
```

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_llm_config({
        "provider": "huggingface",
        "model": "google/gemma-4-E2B-it",
        "max_new_tokens": 512,
        "temperature": 0.7,
    })
    .build()
)
```

The HF provider auto-detects offline mode (sets `local_files_only=True`
when `HF_HUB_OFFLINE=1` is set or `huggingface.co` is unreachable) and
streams tokens via `TextIteratorStreamer`.

### Path B — MLX (Google's recommendation for Apple Silicon)

> Requires a **native arm64 Python** *for the memorizz process itself*.
> `pip install memorizz[mlx]` will fail on Rosetta x86_64 environments.
> If your memorizz env is x86_64, skip to **Path B-sidecar** below — it
> runs MLX in a separate arm64 process and works regardless.

In-process MLX (best when memorizz's own Python is arm64):

```bash
pip install "memorizz[mlx]"

# In the UI: Provider = MLX (Apple Silicon),
#           Model = mlx-community/gemma-4-E2B-it-4bit
```

```python
agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_llm_config({
        "provider": "mlx",
        "model": "mlx-community/gemma-4-E2B-it-4bit",
        "max_new_tokens": 512,
    })
    .build()
)
```

Pre-quantized weights live under
[`mlx-community/*`](https://huggingface.co/mlx-community) — they reuse
the standard `~/.cache/huggingface/hub` cache, so the playground's
"Available offline" indicator covers them too.

#### Path B-sidecar — MLX through `mlx_lm.server` (works from x86_64 too)

If your primary memorizz env is x86_64 (Rosetta-emulated conda envs are
common on Macs with an Intel-era Anaconda install), run MLX in its own
small arm64 venv and let memorizz talk to it via OpenAI-compatible HTTP
(this reuses Path C plumbing — same `OpenAI` provider with `base_url`):

```bash
# One-time, in a native arm64 Python (system /usr/bin/python3 works):
/usr/bin/python3 -m venv ~/.mlx_serve
~/.mlx_serve/bin/pip install mlx-lm

# Each session — pick the model and port:
~/.mlx_serve/bin/python -m mlx_lm.server \
    --model mlx-community/gemma-4-E2B-it-4bit \
    --port 8080
```

In the UI: **Provider = Local OpenAI-compatible**, pick any
`mlx-community/*` entry (the dropdown groups them under "MLX —
mlx_lm.server"), leave the base URL as `http://127.0.0.1:8080/v1`. The
hint in the agent form auto-detects the `mlx-community/` prefix and
shows the correct startup command.

### Path C — llama.cpp / LM Studio (OpenAI-compatible)

Run an OpenAI-compatible server externally, then point memorizz at it.
The `OpenAI` provider accepts a `base_url`, so the agent talks to your
local server through the same code path as the real OpenAI API.

```bash
brew install llama.cpp                                    # or build from source
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF \
             --port 8080 --jinja
```

```bash
# In the UI: Provider = Local OpenAI-compatible (llama.cpp / LM Studio)
#           Model = whatever the server exposes at /v1/models
#           Base URL = http://127.0.0.1:8080/v1
```

```python
agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gemma-4-e2b",                # whatever your server reports
        "base_url": "http://127.0.0.1:8080/v1", # llama.cpp default
    })
    .build()
)
```

LM Studio defaults to `http://127.0.0.1:1234/v1`. vLLM and any other
`/v1/chat/completions`-compatible server work the same way.

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

## Skills Marketplace

MemAgents can search and use agent skills from external marketplaces at runtime. Two providers are available:

- **Vercel Agent Skills** (`vercel`) – searches the open [skills.sh](https://skills.sh) ecosystem and fetches `SKILL.md` instruction files from any GitHub repository. No API key required (set `GITHUB_TOKEN` for better rate limits).
- **SkillsMP** (`skillsmp`) – searches [skillsmp.com](https://skillsmp.com). Requires `SKILLSMP_API_KEY`.

### Vercel Agent Skills

When enabled, the agent receives two tools:

- `vercel_skills_search(q)` – search the skills ecosystem by keyword
- `vercel_skill_fetch(repo)` – fetch a skill's instructions from a GitHub repo (`owner/repo` or full URL)

The agent reads the fetched `SKILL.md` instructions and follows them to complete the task.

```python
from memorizz.memagent import MemAgent

agent = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    skills_marketplace_provider="vercel",
)

# The agent can now search for and apply Vercel Agent Skills
print(agent.run("Build a Next.js app with best practices"))
```

Users can also pass a specific repo directly. The agent fetches the `SKILL.md` and applies the instructions:

```python
agent = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    skills_marketplace_provider="vercel",
)

print(agent.run("Use the skill from vercel/ai-chatbot to set up a chatbot"))
```

The local web UI includes a dedicated **Vercel Skills** page for browsing, searching, and previewing skill instructions. Enable the Vercel provider on any agent via the Skills Marketplace dropdown in the agent creation/edit form.

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
