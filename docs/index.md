# Memorizz

Memorizz is a memory-first Python framework for agents that must retain,
retrieve, reuse, and refine information across turns and processes. It combines
an agent runtime, typed tools, provider-backed memory, context controls,
observability, and governed continual learning behind one SDK.

!!! warning "Project status"
    Memorizz is experimental software licensed under PolyForm Noncommercial
    1.0.0. APIs may change, and deployments must supply their own security,
    privacy, availability, and model-risk controls.

## Start in five minutes

```bash
python -m pip install memorizz
export OPENAI_API_KEY="your-key"
```

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Support memory")
    .with_instruction("Answer concisely and use prior context when relevant.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_ids("support-demo")
    .build()
)

scope = {
    "memory_id": "support-demo",
    "user_id": "user-42",
    "thread_id": "thread-1",
}
agent.run("I prefer Python examples.", **scope)
print(agent.run("How should you show me code?", **scope))
agent.close()
```

With no provider supplied, the SDK persists to
`~/.memorizz/memory`. Use `memory_provider=False` only for an intentionally
stateless agent.

## Choose your interface

| You want to… | Start here |
|---|---|
| Embed an agent in Python | [Python SDK quickstart](getting-started/python-sdk-quickstart.md) |
| Chat with and administer agents in a terminal | [CLI guide](getting-started/cli.md) |
| Configure and inspect agents in a browser | [Local UI guide](getting-started/local-ui.md) |
| Let an MCP host operate Memorizz | [Memorizz MCP server](guides/mcp-server.md) |
| Give an agent access to Notion, Calendar, or another MCP server | [MCP connectivity](guides/mcp-connectivity.md) |
| Run Codex, Claude Code, OpenHands, or MemAgent behind one control plane | [Memory-first MetaHarness](guides/meta-harness.md) |

Not sure which path fits? Read [Choose Your Path](getting-started/overview.md)
and [Installation](getting-started/installation.md).

## The system at a glance

```text
request + tenant scope
        │
        ▼
  MemAgent runtime ── policies ── tools / MCP / browser / sandbox
        │
        ├── retrieve and assemble bounded context
        ├── call the configured model and governed capabilities
        └── persist outcomes, traces, summaries, and learning evidence
        │
        ▼
filesystem (default) │ MongoDB │ Oracle AI Database │ custom provider
```

For repository work, the same control plane can route a bounded task to a
Codex, Claude Code, OpenHands, or native MemAgent worker. The worker owns its
agent loop; MemoRizz owns memory scope, policy, approvals, normalized evidence,
verification, and learning.

The important distinction is that a **memory type** defines what a record
means, while a **memory provider** defines where it is persisted and queried.
The runtime decides what enters the model context on each turn. See
[Core Concepts](getting-started/concepts.md) for the scopes, lifecycle, and
trust boundaries.

## Build, operate, and evaluate

- Add application functions with [typed tools and durable approvals](guides/tools-and-approvals.md).
- Choose a [memory provider](memory-providers/filesystem.md) and follow the
  [multi-tenant contract](guides/multi-tenant.md).
- Control token use with [context efficiency, semantic caching, and compaction](guides/context-efficiency.md).
- Inspect runs with [observability and trace analysis](observability-ui.md).
- Verify optional integrations with [capability reports and preflight](reference/capabilities.md).
- Use the [evaluation suite](evaluation-suite.md) for reproducible comparisons;
  benchmark results are not leaderboard submissions unless submitted through
  the benchmark's official process.

For production-oriented review, start with
[Configuration and Secrets](reference/configuration.md),
[Production Governance](guides/production-governance.md), and
[Troubleshooting](troubleshooting.md).
