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
- a memory-first meta-harness for running Codex, Claude Code, OpenHands, or a native MemAgent behind shared memory, policy, approval, verification, and learning controls
- an interactive, Claude-Code-style terminal CLI (`memorizz`) with persistent memory — see [CLI](#cli)

## Key Capabilities

- **Persistent memory** across sessions and conversations
- **Semantic retrieval** with embeddings + vector search
- **Knowledge base** with file/folder ingestion (`.pdf`, `.md`, `.txt`, `.csv`, `.json`, …) and configurable chunking (`fixed` / `sentence` / `paragraph` / `semantic` / custom). Same extractor registry powers the SDK and the local UI's drag-and-drop uploader; see [`long_term/semantic/README.md`](src/memorizz/long_term/semantic/README.md).
- **Entity memory** tools for profile-style facts (`entity_memory_lookup` / `entity_memory_upsert`), with host-owned tenant scope and bounded exact fallback when vector search is unavailable
- **Provider-neutral personalization context** combining canonical profile facts, explicit product preferences, opt-in relevance-gated conversation recall, and host-authorized writing samples under one bounded SDK/MCP contract. Memory-first traces separate what was retrieved, supplied, and explicitly referenced without persisting raw memory values.
- **Tool calling** with automatic function registration
- **Semantic cache** to reduce repeat LLM calls
- **Prompt-cache-friendly context assembly** — stable prefix (frozen system prompt, append-only chunk-evicted history) with all per-turn content at the tail; automatic Anthropic `cache_control` breakpoints and OpenAI `prompt_cache_key` routing serve most of each turn's prompt at cached-input rates (see [docs/guides/context-efficiency.md](docs/guides/context-efficiency.md))
- **Pre-inference deduplication** — retrieved memories are exact-hash, similarity (cosine ≥ 0.95), and vs-history deduplicated, then MMR-selected before entering the context window
- **Continual learning** — repeated successful tool workflows are promoted into reusable learned skills (gated by frequency × success × recency × query diversity, LLM-distilled into validated SKILL.md documents, monitored for drift, and demoted when they stop working). Skills use user-context authority by default; reviewed skills can opt into developer/application authority. Enable with `continual_learning=True` (see [docs/guides/continual-learning.md](docs/guides/continual-learning.md))
- **Memory-first learning control plane** — immutable learning events, host-verified outcomes, deterministic incremental compilation, per-turn token-bounded `EvidencePack` retrieval, skill lifecycle evidence, and reversible operator-approved forgetting across filesystem, MongoDB, and Oracle (see [Learning Control Plane](docs/guides/learning-control-plane.md))
- **Memory-first MetaHarness** — run Codex, Claude Code, OpenHands, and native MemAgent workers through one durable, tenant-scoped control plane with deterministic routing, exact host approvals, workspace isolation, budgets, cancellation, normalized evidence, and host verification (see [Memory-First Meta-Harness](docs/guides/meta-harness.md))
- **Multi-agent orchestration** with shared blackboard memory
- **Context-window telemetry** via `agent.get_context_window_stats()` and
  per-turn cache metrics (`cached_tokens`) from `agent.model.get_last_usage()`
- **Skills marketplace** with Vercel Agent Skills and SkillsMP providers
- **Scheduled automations** via SDK, web UI, or agent conversation (see `src/memorizz/automation/README.md`)
- **Production-oriented MCP connectivity** for Notion, Google Calendar, and custom servers, with encrypted OAuth/bearer credentials, SSRF controls, bounded retries/timeouts, mutation approval, and UI/CLI management (see [MCP Connectivity](docs/guides/mcp-connectivity.md))
- **First-party MCP server** over headless local stdio or authenticated Streamable HTTP, with 24 strict-schema tools spanning safe agent lifecycle, bounded personalization previews, tenant-scoped memory/conversations, observability, cache, learning, compaction, and harness execution. Credential and approval authority remains host-only (see [Expose MemoRizz as an MCP Server](docs/guides/mcp-server.md))
- **Production governance** with durable host approvals, progressive tool disclosure, size-aware tool results, cache freshness controls, Oracle compaction parity, bounded sandboxes, and provider-neutral orchestration (see [Production Governance](docs/guides/production-governance.md))
- **Operational runtime APIs** for JSON-safe deterministic delegation, explicitly scoped compaction, structured approval-resume evidence, provider-error propagation, Oracle/E2B environment presets, cache inspection, scoped observability, and context-managed cleanup.
- **Structured tool outcomes** across SDK, CLI, MCP, UI, traces, and learning, distinguishing clean success, empty results, degraded capability, successful fallback, provider errors, and other execution failures without changing model-visible tool payloads.
- **Host-enforced completion gates** with bounded same-loop retries, buffered streaming, auditable decision evidence, cache-safe revalidation, and fail-closed trusted validator rebinding.
- **Governed browser control** through a provider-neutral capability and an isolated Browser Use provider. Every model-initiated browser task requires a durable, single-use host approval (see [Browser Control](docs/browser-control/index.md))
- **Memory-first evaluation adapters** for AgentMemBench, LongMemEval-V2, LoCoMo-Plus, BEAM, MemoryAgentBench, Terminal-Bench, SWE-bench Lite, and MemBench, including a zero-cost Ollama suite (see [Evaluation Suite](docs/evaluation-suite.md)).
- **Two local systems-paper drafts** cover the memory-first agent harness and
  the agent memory/continual-learning platform. They are intentionally ignored
  by Git; see [Research Papers](docs/research-paper.md) for local build and
  review instructions.

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
export OPENAI_API_KEY="your-key"
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
pip install "memorizz[terminal-bench]"   # Harbor benchmark adapter
pip install "memorizz[longmemeval-v2]"  # LongMemEval-V2 official harness adapter
pip install "memorizz[swe-bench]"       # SWE-bench Lite + Docker grader
pip install "memorizz[membench-eval]"   # MemBench capacity tokenization
pip install "memorizz[ui]"              # Local web UI
pip install "memorizz[huggingface]"     # transformers + sentence-transformers
pip install "memorizz[mlx]"             # Apple-Silicon MLX backend (native arm64 only)
pip install "memorizz[all]"             # Everything
```

## Quick Start (Filesystem Provider)

Filesystem memory is now the zero-configuration SDK default: `MemAgent()` and
`MemAgentBuilder().build()` persist beneath `~/.memorizz/memory`. Set
`MEMORIZZ_MEMORY_ROOT` to relocate it, pass an explicit provider for custom
embeddings/storage, or use `memory_provider=False` for an intentionally
stateless agent.

```python
from pathlib import Path

from memorizz import FileSystemConfig, FileSystemProvider, MemAgentBuilder

provider = FileSystemProvider(
    FileSystemConfig(
        root_path=Path("~/.memorizz/memory").expanduser(),
        lazy_vector_indexes=True,
        embedding_provider="openai",
        embedding_config={"model": "text-embedding-3-small"},
    )
)

llm_config = {"provider": "openai", "model": "gpt-4o-mini"}

agent = (
    MemAgentBuilder()
    .with_instruction("You are a helpful assistant with persistent memory.")
    .with_memory_provider(provider)
    .with_llm_config(llm_config)
    .with_memory_ids("payments-assistant")
    .with_semantic_cache(enabled=True, threshold=0.85)
    .build_and_save()
)

scope = {
    "memory_id": "payments-assistant",
    "user_id": "leah",
    "thread_id": "onboarding",
}
print(agent.run("Hi, I work on payments systems.", **scope))
print(agent.run("What did I tell you about my work?", **scope))

stats = agent.get_context_window_stats()
print(stats)
```

Add the memory-first learning layer without changing providers:

```python
agent = (
    MemAgentBuilder()
    .with_llm_config(llm_config)
    .with_memory_provider(provider)
    .with_memory_ids("release-assistant")
    .with_learning_control_plane(
        evidence_token_budget=1600,
        compile_every_n_events=12,
    )
    .build()
)

agent.run(
    "Use what we learned from the last release.",
    memory_id="release-assistant",
    user_id="alice",
    thread_id="release-42",
)
print(agent.explain_memory_decision())
```

Agent memory is how we make intelligent systems retain, reuse, recall and
refine information. The control plane keeps that loop bounded and auditable;
it reuses existing memory-provider storage rather than adding another service.

Run a bounded repository task through an installed coding harness while
MemoRizz owns memory scope, policy, evidence, and verification:

```python
from pathlib import Path

from memorizz import HarnessPermissions, HarnessTask, MetaHarness

repo = Path.cwd().resolve()
with MetaHarness.from_env(allowed_workspace_roots=[str(repo)]) as harness:
    result = harness.run(
        HarnessTask(
            task="Inspect the test failures and report the likely cause.",
            workspace=str(repo),
            harness="auto",
            permissions=HarnessPermissions(workspace_mode="read_only"),
        )
    )

print(result.status.value, result.final_response)
```

Use `memorizz harness doctor` to inspect locally installed adapters. Direct
edits, expanded tools, secrets, and network access pause for an exact,
single-use host approval.

Building a multi-user application? Pass `user_id` to isolate memory per
end-user — one agent can serve every tenant in your app. See the
[Multi-Tenant Guide](docs/guides/multi-tenant.md) for the full contract.

```python
scope = {"memory_id": "preferences", "thread_id": "profile"}
agent.run("Remember my favorite color is purple.", user_id="alice", **scope)
agent.run("What's my favorite color?", user_id="bob", **scope)  # isolated
```

## Continual Learning and Reviewed Skill Authority

Continual learning is provider-independent across filesystem, MongoDB, and
Oracle. Workflow memory remains the audit and outcome-evidence store; automatic
prompt retrieval does not replay raw workflows. Matching active skills are
retrieved from Skillbox instead.

```python
from memorizz import MemAgentBuilder


def release_status(release_id: str) -> dict:
    """Return the current status of one release."""
    return {"release_id": release_id, "status": "healthy"}

agent = (
    MemAgentBuilder()
    .with_llm_config(llm_config)
    .with_memory_provider(provider)
    .with_tools([release_status])
    .with_continual_learning(
        enabled=True,
        config={
            "require_shadow": True,
            "skill_injection_role": "developer",  # or "user" (default)
            "shadow_evaluation_enabled": True,
        },
    )
    .build()
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
from memorizz import ApplicationMode, MemAgentBuilder

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
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
from memorizz.internet_access import TavilyProvider
from memorizz.memagent.builders import create_deep_research_agent

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
}

internet_provider = TavilyProvider()  # reads TAVILY_API_KEY

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
from memorizz import MemAgent

llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
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

## Evaluation Suite

MemoRizz includes local, revision-recording adapters for Terminal-Bench 2.1,
LongMemEval-V2, SWE-bench Lite, and MemBench. The runners keep official scoring
separate from MemoRizz diagnostics and exercise scoped filesystem memory,
hybrid/semantic retrieval, summarization and compaction, semantic-cache safety,
workflow persistence, observability, and host-enforced completion.

The memory suite adds versioned paper-protocol manifests, official source/data
verification, `smoke`/`regression`/`paper` profiles, fail-closed comparability,
calibrated rank fusion, reusable corpus embeddings, grounded citations, and a
gold-evidence reader lane. Filesystem and Oracle are selectable from the same
SDK/CLI/UI workflow; MongoDB implements the same batch/search capability
contract for application integrations.

```bash
memorizz eval list
memorizz eval protocol show longmemeval-v2
memorizz eval dataset verify longmemeval-v2 --data-path /data/lme-v2
memorizz eval terminal-bench forecast --total-budget-usd 1000
memorizz eval run longmemeval-v2 \
  --data-path /data/lme-v2 --variant small-web --profile smoke
```

Small local subsets are reported as smoke tests, never extrapolated into
leaderboard scores. See the [evaluation methodology, commands, measured local
results, and improvement backlog](docs/evaluation-suite.md).

Database/admin helpers:

```bash
memorizz oracle install             # start Oracle container helper
memorizz oracle setup               # initialize Oracle schema/user
```

## Examples

- `examples/zero_to_hero/README.md` — current memory-first tutorials: a
  zero-configuration filesystem agent, a real Oracle AI Database companion,
  and a field guide to every MemoRizz memory type
- `examples/single_agent/memagent_local_oracle.ipynb`
- `examples/single_agent/memagent_remote_oracle.ipynb`
- `examples/deep_research/deep_research_memagent.ipynb`
- `examples/sandbox/memagent_e2b_sandbox.ipynb`
- `examples/sandbox/memagent_daytona_sandbox.ipynb`
- `examples/sandbox/memagent_graalpy_sandbox.ipynb`
- `examples/automations/automations_guide.ipynb`
- `examples/continual_learning/continual_learning_guide.ipynb`
- `examples/metaharness/README.md` — six-notebook path from an offline adapter
  contract to a Codex + Claude Code review team and the measured fair comparison
- `examples/model_providers/openai_provider.ipynb`
- `examples/model_providers/anthropic_provider.ipynb`
- `examples/model_providers/ollama_provider.ipynb`
- `examples/model_providers/compare_providers.ipynb`

## Documentation

- [Installation and interface selection](docs/getting-started/installation.md)
- [Core concepts and scope model](docs/getting-started/concepts.md)
- [Model providers](docs/getting-started/model-providers.md)
- [Python SDK quickstart](docs/getting-started/python-sdk-quickstart.md)
- [Scheduled automations](docs/guides/automations.md)
- [Configuration and secrets](docs/reference/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Python API reference](docs/reference/python-api.md)

For a local preview, run `make docs-serve` (or `mkdocs serve`). Contributor
rules are in [`docs/README.md`](docs/README.md); implementation architecture is
in [`src/memorizz/MEMORY_ARCHITECTURE.md`](src/memorizz/MEMORY_ARCHITECTURE.md).

## License

PolyForm Noncommercial 1.0.0.
See `LICENSE` and `NOTICE`.
