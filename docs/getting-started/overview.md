# Choose Your Path

Memorizz exposes one agent and memory model through four form factors. Choose
the interface that owns your deployment; saved agents remain available to the
other interfaces when they use the same provider and scope.

## Choose an interface

| Interface | Best for | Process model | Guide |
|---|---|---|---|
| Python SDK | Product integration and custom orchestration | In your application | [SDK quickstart](python-sdk-quickstart.md) |
| CLI | Local development, scripting, and operations | Interactive or one-shot terminal | [CLI guide](cli.md) |
| Local UI | Configuration, playground use, and trace inspection | Local FastAPI server | [UI guide](local-ui.md) |
| MCP server | Claude Desktop, IDEs, and other MCP hosts | Headless stdio or HTTP server | [MCP server](../guides/mcp-server.md) |

All four can run headlessly. Only the browser UI requires an HTTP client; the
server itself does not require a desktop or display.

## Choose a memory provider

| Provider | Choose it when | Operational tradeoff |
|---|---|---|
| Filesystem | Developing locally, evaluating, or running one process | Zero configuration; coordinate writes if several processes share a directory |
| MongoDB | You already operate MongoDB or need document-native scale | Requires indexes, backups, and tenant-aware query review |
| Oracle AI Database | You need relational governance plus native vector operations | Requires schema provisioning, preflight, and an explicit vector-index policy |
| Custom | Storage must live in an existing platform | You own conformance, tenant isolation, atomicity, and lifecycle behavior |

The filesystem provider is the default. Explicitly select a backend before
production and test it with the same scopes and retrieval settings used by the
application.

## Choose an agent mode

| Mode | Default emphasis | Typical use |
|---|---|---|
| `assistant` | Conversation, entities, knowledge, summaries | Stateful support or personal assistants |
| `workflow` | Tools, workflow traces, learned procedures | Repeatable operational tasks |
| `deep_research` | Tools, knowledge, shared coordination | Multi-step investigation and synthesis |

Modes select sensible memory defaults; they do not grant authority. Add tools,
MCP, browser control, internet access, or code execution explicitly and apply
the corresponding host policy.

For repository and terminal work, a MemAgent can either delegate to an external
harness or run its complete turn on Codex, Claude Code, or an isolated
OpenHands worker. See the [memory-first meta-harness guide](../guides/meta-harness.md).

## Recommended build sequence

1. [Install](installation.md) the base package and only the extras you need.
2. Build one agent and make `memory_id`, `user_id`, and `thread_id` explicit.
3. Add external authority through [governed tools](../guides/tools-and-approvals.md).
4. Enable caching, summaries, or continual learning only after defining
   freshness, evaluation, and rollback requirements.
5. Add [observability](../observability-ui.md) and a capability/preflight check
   to deployment startup.
6. Exercise both happy paths and isolation/failure paths against the real
   provider before accepting traffic.

## Requirements

- CPython 3.10, 3.11, or 3.12.
- An LLM provider for model-backed runs. OpenAI and Ollama clients ship in the
  base install; other providers use optional extras.
- An embedding provider only when semantic retrieval or semantic caching is
  required. Exact persistence and retrieval remain available without one.

Continue with [Core Concepts](concepts.md) for the runtime mental model, or go
directly to the [Python SDK quickstart](python-sdk-quickstart.md).
