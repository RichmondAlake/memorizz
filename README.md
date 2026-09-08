# Memorizz

**The intelligence plane for memory-first agents.**

Memorizz is a Python framework that connects persistent memory, models, prompts,
and reasoning so agents can recall what matters, reuse experience, and make
better decisions across sessions.

[Documentation](https://richmondalake.github.io/memorizz/) ·
[PyPI](https://pypi.org/project/memorizz/) ·
[Examples](examples/zero_to_hero/README.md) ·
[Changelog](CHANGELOG.md)

## Memory management

Memory is the foundation: retain information, retrieve relevant context, and
refine what an agent knows over time.

| Memory | What it gives your agents |
|---|---|
| [Episodic](docs/memory-types/episodic.md) | Conversation history, past interactions, and summaries. |
| [Semantic](docs/memory-types/semantic.md) | Knowledge bases, entity facts, preferences, and personas. |
| [Procedural](docs/memory-types/procedural.md) | Registered tools, recorded workflows, and reusable skills. |
| [Short-term](docs/memory-types/short-term.md) | Working context and semantic caching. |
| [Shared](docs/memory-types/shared.md) | A shared workspace for agent coordination and handoffs. |

- **Persist and isolate:** scope memory by agent, user, and conversation. Start
  with local filesystem storage, or use MongoDB or Oracle through the same
  provider contract.
- **Ingest and retrieve:** turn files and folders into a knowledge base with
  configurable chunking, embeddings, and semantic or hybrid search.
- **Manage context:** deduplicate retrieved memories, summarize and compact
  history, inspect recall decisions, and control retention and forgetting.

Explore [memory concepts and scope](docs/getting-started/concepts.md),
[storage providers](docs/memory-providers/filesystem.md), and
[context efficiency](docs/guides/context-efficiency.md).

## Agent framework

Build persistent agents with `MemAgent` and `MemAgentBuilder`. Choose
`assistant`, `workflow`, or `deep_research` modes and connect cloud or local
[model providers](docs/getting-started/model-providers.md).

- **Tools and integrations:** Python functions, MCP servers, internet search,
  governed browser control, sandboxes, and external skills.
- **Coordination:** delegate work to specialist agents, share memory, and run
  scheduled automations.
- **Interfaces:** use the Python SDK, interactive CLI, local web UI, or expose
  Memorizz as an MCP server.

Start with the [SDK quickstart](docs/getting-started/python-sdk-quickstart.md)
or the [CLI guide](docs/getting-started/cli.md).

## Harness

`MetaHarness` runs tasks through Codex, Claude Code, OpenHands, or native
MemAgent workers. It carries memory scope and learning evidence across
harnesses while managing permissions, approvals, workspace isolation, budgets,
cancellation, and host-side verification.

Use an external harness for an entire task or let a MemAgent call one as a
specialist tool. Inspect installed adapters with `memorizz harness doctor`.

See the [MetaHarness guide](docs/guides/meta-harness.md) and
[hands-on examples](examples/metaharness/README.md).

## Continual learning and the intelligence plane

The **intelligence plane** is the model layer and everything that helps it
think: the LLM, prompts, reasoning strategies such as planning, reflection, and
tool selection, plus the memory and retrieval that supply context. Its job is
**decision quality**. A bad plan or a hallucinated tool call is an intelligence
plane problem.

Memorizz makes memory the foundation of this plane. Its opt-in continual
learning loop captures tool workflows and their outcomes, distills repeated
successful procedures into reusable skills, retrieves relevant active skills,
and monitors their performance. Shadow evaluation and review support skill
activation; skills that stop working can be demoted.

The learning control plane adds traceable outcome evidence, bounded retrieval,
and reversible forgetting. Together, these capabilities help agents retain,
recall, reuse, and refine experience.

Read about [continual learning](docs/guides/continual-learning.md), the
[learning control plane](docs/guides/learning-control-plane.md), and
[evaluation methods](docs/evaluation-suite.md).

## Quick start

Requires Python 3.10+. Install and launch the CLI with a cloud model:

```bash
python -m pip install memorizz
export OPENAI_API_KEY="your-key"
memorizz
```

For a fully local setup, use [Ollama](docs/getting-started/cli.md#option-a-local-no-api-key-ollama).
The [installation guide](docs/getting-started/installation.md) covers `uv`,
`pipx`, npm, Homebrew, and optional integrations.

Build a memory-first agent in Python using the same environment:

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_instruction("Remember useful context and use it to help the user.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_ids("my-assistant")
    .build_and_save()
)

scope = {"memory_id": "my-assistant", "user_id": "alice", "thread_id": "intro"}
with agent.lifecycle(close_memory_provider=True):
    agent.run("I work on payments systems.", **scope)
    print(agent.run("What did I tell you about my work?", **scope))
```

Memory persists under `~/.memorizz/memory` by default. Save `agent.agent_id` to
[restore the agent later](docs/getting-started/python-sdk-quickstart.md#5-restore-it-later).

## Explore

- [Memory-first tutorials](examples/zero_to_hero/README.md)
- [MCP connectivity](docs/guides/mcp-connectivity.md) and [MCP server](docs/guides/mcp-server.md)
- [Automations](docs/guides/automations.md) and [multi-user applications](docs/guides/multi-tenant.md)
- [Observability](docs/observability-ui.md) and [evaluation suite](docs/evaluation-suite.md)
- [Usage, cost and memory analytics](docs/usage-analytics.md)
- [Account-owned persona evolution](docs/persona-evolution-hosts.md)
- [API reference](docs/reference/python-api.md), [configuration](docs/reference/configuration.md), and [troubleshooting](docs/troubleshooting.md)

## License

Source available under [PolyForm Noncommercial 1.0.0](LICENSE). See [NOTICE](NOTICE).
Memorizz is experimental: APIs may change and it has not undergone security
hardening for production workloads.
