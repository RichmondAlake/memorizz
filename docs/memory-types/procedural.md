# Procedural Memory

Procedural memory captures *how* an agent should act. It bundles tool registration, workflows, and scripted behaviors so that the agent can plan or execute actions consistently. Source code lives in `src/memorizz/long_term/procedural/`.

## Components

- **Toolbox (`MemoryType.TOOLBOX`)** – Python callables wrapped with metadata so LLMs can discover and execute them safely.
- **Workflow Memory (`MemoryType.WORKFLOW_MEMORY`)** – Per-run tool-calling trajectories captured automatically, each carrying a `canonical_hash` (trajectory identity) so repeated procedures can be counted and learned from. Agents connected to MCP servers, including [desktop control](../desktop-control/index.md), keep every run this way.
- **Skillbox (`MemoryType.SKILLBOX`)** – Learned skills: SKILL.md documents distilled from repeated successful workflow trajectories, with a full promotion/monitoring/demotion lifecycle. See the [Continual Learning guide](../guides/continual-learning.md).
- **Personas** – While technically part of semantic memory, personas often work hand-in-hand with procedural steps to enforce tone and guardrails.

## Registering Tools

```python
from memorizz import MemAgentBuilder, Toolbox


def system_status() -> dict:
    """Return current system status."""
    return {"status": "ok"}

toolbox = Toolbox.from_functions(
    [system_status],
    memory_provider=memory_provider,
    agent_id="operations-agent",
    augment=False,
)

agent = (
    MemAgentBuilder()
    .with_name("Operations")
    .with_memory_provider(memory_provider)
    .with_tools([system_status])
    .with_toolbox(toolbox)
    .build()
)
```

`Toolbox.from_functions(...)` preserves the trusted Python callables and can
persist their strict schemas for progressive discovery. Deterministic
registration does not construct an LLM. Avoid using `@toolbox.register_tool` as
a normal decorator: the low-level registration method returns a tool ID rather
than the original function.

Tool schemas are capability descriptions, not authority. Apply a
[`ToolPolicy`](../guides/tools-and-approvals.md) to side effects and use host
approval for mutations.

## MCP tools in toolbox memory

Tools listed by a connected [MCP server](../guides/mcp-connectivity.md) are
stored in the agent's toolbox too: one record per tool, with an embedding of
its name, description and argument names, so a request finds the right tool
by meaning. A catalog record per server keeps a fingerprint of each tool, so
only tools whose description changed are embedded again; tools the server no
longer lists are removed, and disconnecting the server removes its tools.
Saving the agent keeps these records.

## Runs kept as workflow memory

Tool-calling runs are captured only while `workflow_memory` is one of the
agent's memory types. The `workflow` mode includes it, and so does a
`MemAgent(...)` constructed with neither a mode nor memory types. The
`assistant` mode, `MemAgentBuilder`'s default, does not: a builder-made
assistant captures nothing until you turn on continual learning
(`with_continual_learning(True)`), which adds it, or list it in the agent's
memory types.

An agent with an MCP connection records each tool-calling run as a workflow,
even when its mode does not list workflow memory, unless it was configured
with no memory types at all. Each record holds the request, every tool call
with its arguments and result, and the outcome; a run that pauses for an
approval stays one record. A [desktop control](../desktop-control/index.md)
step's result also names the screenshot of the app's window, when one was
kept. Browse runs in `memorizz ui` under **Memory Types → Workflows**.

## When to Reach for Procedural Memory

- Automations that call APIs, databases, or internal services
- Agents that must follow compliance-friendly workflows
- Research or analyst bots that gather, synthesize, then report findings based on a repeatable checklist

## Learning From Repeated Workflows

With `continual_learning=True`, a MemAgent promotes trajectory classes that
keep succeeding (gated by execution count × success rate × recency × query
diversity) into learned skills and demotes them when they drift. Skills are
injected as user-context priors by default. Application-owned skills can use
reviewed developer authority with
`{"require_shadow": True, "skill_injection_role": "developer"}`. System
policy, skill preconditions, and current tool results remain authoritative.
Raw workflow memory is captured for evidence but is not part of automatic
pre-inference prompt retrieval. See the
[Continual Learning guide](../guides/continual-learning.md) and the
runnable walkthrough in `examples/continual_learning/continual_learning_guide.ipynb`.
