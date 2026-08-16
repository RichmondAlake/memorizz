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
- optional internet access, governed browser control, sandbox code execution, skills marketplace, and local web UI
- first-class MCP connectivity over stdio, Streamable HTTP, and SSE, including OAuth, encrypted credentials, resources, prompts, and tool approval policy
- an interactive, Claude-Code-style terminal CLI (`memorizz`) with persistent memory — see [CLI](#cli)

## Key Capabilities

- **Persistent memory** across sessions and conversations
- **Semantic retrieval** with embeddings + vector search
- **Knowledge base** with file/folder ingestion (`.pdf`, `.md`, `.txt`, `.csv`, `.json`, …) and configurable chunking (`fixed` / `sentence` / `paragraph` / `semantic` / custom). Same extractor registry powers the SDK and the local UI's drag-and-drop uploader; see [`long_term/semantic/README.md`](src/memorizz/long_term/semantic/README.md).
- **Entity memory** tools for profile-style facts (`entity_memory_lookup` / `entity_memory_upsert`)
- **Tool calling** with automatic function registration
- **Semantic cache** to reduce repeat LLM calls
- **Prompt-cache-friendly context assembly** — stable prefix (frozen system prompt, append-only chunk-evicted history) with all per-turn content at the tail; automatic Anthropic `cache_control` breakpoints and OpenAI `prompt_cache_key` routing serve most of each turn's prompt at cached-input rates (see [docs/guides/context-efficiency.md](docs/guides/context-efficiency.md))
- **Pre-inference deduplication** — retrieved memories are exact-hash, similarity (cosine ≥ 0.95), and vs-history deduplicated, then MMR-selected before entering the context window
- **Continual learning** — repeated successful tool workflows are promoted into reusable learned skills (gated by frequency × success × recency × query diversity, LLM-distilled into validated SKILL.md documents, monitored for drift, and demoted when they stop working). Skills use user-context authority by default; reviewed skills can opt into developer/application authority. Enable with `continual_learning=True` (see [docs/guides/continual-learning.md](docs/guides/continual-learning.md))
- **Multi-agent orchestration** with shared blackboard memory
- **Context-window telemetry** via `get_context_window_stats()` and per-turn cache metrics (`cached_tokens`) from `get_last_usage()`
- **Skills marketplace** with Vercel Agent Skills and SkillsMP providers
- **Scheduled automations** via SDK, web UI, or agent conversation (see `src/memorizz/automation/README.md`)
- **Production-oriented MCP connectivity** for Notion, Google Calendar, and custom servers, with encrypted OAuth/bearer credentials, SSRF controls, bounded retries/timeouts, mutation approval, and UI/CLI management (see [MCP Connectivity](docs/guides/mcp-connectivity.md))
- **First-party MCP server** over local stdio or authenticated Streamable HTTP, exposing tenant-scoped memory, conversations, agents, resources, and prompts (see [Expose MemoRizz as an MCP Server](docs/guides/mcp-server.md))
- **0.5 production governance** with durable host approvals, progressive tool disclosure, size-aware tool results, cache freshness controls, Oracle compaction parity, bounded sandboxes, and provider-neutral orchestration (see [Production Governance](docs/guides/production-governance.md))
- **Operational runtime APIs** for JSON-safe deterministic delegation, explicitly scoped compaction, structured approval-resume evidence, provider-error propagation, Oracle/E2B environment presets, cache inspection, scoped observability, and context-managed cleanup.
- **Governed browser control** through a provider-neutral capability and an isolated Browser Use provider. Every model-initiated browser task requires a durable, single-use host approval (see [Browser Control](docs/browser-control/index.md))

Expose local MemoRizz functionality to a desktop MCP host:

```json
{
  "mcpServers": {
    "memorizz": {
      "command": "memorizz",
      "args": ["mcp", "serve"]
    }
  }
}
```

## Installation

Base install:

```bash
pip install memorizz
```

The base install also gives you the interactive **`memorizz` CLI** (see
[CLI](#cli)). Install the `mcp` extra when the CLI or an agent must connect to,
or serve, MCP.

Common extras:

```bash
pip install "memorizz[oracle]"          # Oracle provider
pip install "memorizz[mongodb]"         # MongoDB provider
pip install "memorizz[filesystem]"      # Local filesystem + FAISS
pip install "memorizz[mcp]"             # MCP client/server + encrypted credentials
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

## Continual Learning and Reviewed Skill Authority

Continual learning is provider-independent across filesystem, MongoDB, and
Oracle. Workflow memory remains the audit and outcome-evidence store; automatic
prompt retrieval does not replay raw workflows. Matching active skills are
retrieved from Skillbox instead.

```python
from memorizz import MemAgent

agent = MemAgent(
    model=model,
    memory_provider=provider,
    tools=tools,
    continual_learning=True,
    continual_learning_config={
        "require_shadow": True,
        "skill_injection_role": "developer",  # or "user" (default)
        "shadow_evaluation_enabled": True,    # optional passive evidence
    },
)
```

`developer` is deliberately rejected unless `require_shadow=True`: generated
instructions must be validated, reviewed, and explicitly activated before
gaining application-level authority. The official OpenAI API receives a
developer message; Anthropic receives the reviewed instructions through its
[top-level system parameter](https://platform.claude.com/docs/en/api/messages/create).
System policy and current tool/database facts still win. Legacy skills remain
at `user` authority.

Passive evaluation observes only new, post-distillation workflows in a bounded
background queue. It never injects a shadow skill, calls an LLM judge, reruns
a tool, or activates a skill. Semantic retrieval may still use the configured
embedding provider. The local UI exposes the opt-in and shows
observation, trajectory-match, matched-success, and advisory readiness metrics.
See the [Continual Learning guide](docs/guides/continual-learning.md) for
promotion, passive evaluation, review, provider mapping, Oracle migrations, and
the three-arm token/latency/accuracy notebook evaluation.

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
# Set unique admin/application passwords in your environment or .env first.
memorizz oracle install
memorizz oracle setup
memorizz oracle preflight --json
```

Configure `ORACLE_USER`, `ORACLE_PASSWORD`, `ORACLE_DSN`, and your LLM
credentials. MemoRizz has no database-password defaults and never prints the
configured value. Full bootstrap, migration, index-policy, preflight, and
cleanup details are in [`SETUP.md`](SETUP.md) and the [Oracle provider
guide](docs/memory-providers/oracle.md).

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

## Browser Control

Install [Browser Use](https://github.com/browser-use/browser-use) as a separate
Python 3.11+ CLI tool. This keeps its dependency environment isolated from
MemoRizz's MCP 2.x runtime:

```bash
uv tool install --python 3.12 browser-use
browser-use install
browser-use doctor
```

Attach the provider with the builder, a direct `MemAgent` argument, the agent
editor UI, or `memorizz chat --browser-control`:

```python
agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_llm_config({"provider": "openai", "model": "gpt-4.1-mini"})
    .with_browser_control(
        {
            "provider": "browseruse",
            "llm_provider": "openai",
            "allowed_domains": ["example.com", "*.notion.so"],
            "max_steps": 25,
            "task_timeout": 600,
        }
    )
    .build(validate=True)
)
```

The model receives one `browser_control(task, max_steps)` tool. It is always
nondeterministic, side-effecting, and approval-required. MemoRizz stores the
exact task and argument hash, then the UI/host/CLI can approve and resume the
original checkpoint exactly once. Model-visible `approved`/`confirm` switches
are not accepted. Each execution uses a private fixed worker in the isolated
Browser Use Python environment and closes its browser on success, failure, or
timeout. See the
[Browser Control guide](docs/browser-control/index.md) for installation,
policy, result shape, and security guidance.

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

Memorizz ships an interactive, Claude-Code-style terminal agent with **persistent
memory**.

**Install** (pick one):

```bash
pip install memorizz                                   # if you have Python 3.10+
uv tool install --python 3.12 memorizz                 # isolated tool, no system Python needed
npm install -g memorizz                                # bootstraps uv under the hood
curl -fsSL https://raw.githubusercontent.com/RichmondAlake/memorizz/main/install.sh | sh
```

Then just run `memorizz`:

```bash
memorizz                       # interactive REPL (memory persists across launches)
memorizz chat --code           # enable coding tools (read/write files + commands)
memorizz run "your prompt"     # one-shot, prints the reply
memorizz chat --browser-control # attach governed Browser Use
memorizz run --browser-control "read example.com"
memorizz ui                    # start the local web UI (requires [ui])
```

With no API key and a running [Ollama](https://ollama.com) daemon it runs a
**100% local stack** (Ollama LLM + embeddings + on-disk memory). Inside the REPL,
`/help` lists 20+ slash commands (`/model`, `/code`, `/browser`, `/approvals`, `/persona`, `/memory`,
`/conversations`, `/forget`, `/clear`, `/ingest`, `/ui`, …).

See the **[CLI Guide](docs/getting-started/cli.md)** for the full reference.

Database/admin helpers:

```bash
memorizz oracle install             # start Oracle container helper
memorizz oracle setup               # initialize Oracle schema/user
```

## Examples

- `examples/single_agent/memagent_local_oracle.ipynb`
- `examples/single_agent/memagent_remote_oracle.ipynb`
- `examples/deep_research/deep_research_memagent.ipynb`
- `examples/sandbox/memagent_e2b_sandbox.ipynb`
- `examples/sandbox/memagent_daytona_sandbox.ipynb`
- `examples/sandbox/memagent_graalpy_sandbox.ipynb`
- `examples/automations/automations_guide.ipynb`
- `examples/continual_learning/continual_learning_guide.ipynb`
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
