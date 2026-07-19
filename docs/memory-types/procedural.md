# Procedural Memory

Procedural memory captures *how* an agent should act. It bundles tool registration, workflows, and scripted behaviors so that the agent can plan or execute actions consistently. Source code lives in `src/memorizz/long_term/procedural/`.

## Components

- **Toolbox (`MemoryType.TOOLBOX`)** – Python callables wrapped with metadata so LLMs can discover and execute them safely.
- **Workflow Memory (`MemoryType.WORKFLOW_MEMORY`)** – Per-run tool-calling trajectories captured automatically, each carrying a `canonical_hash` (trajectory identity) so repeated procedures can be counted and learned from.
- **Skillbox (`MemoryType.SKILLBOX`)** – Learned skills: SKILL.md documents distilled from repeated successful workflow trajectories, with a full promotion/monitoring/demotion lifecycle. See the [Continual Learning guide](../guides/continual-learning.md).
- **Personas** – While technically part of semantic memory, personas often work hand-in-hand with procedural steps to enforce tone and guardrails.

## Registering Tools

```python
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.long_term.procedural.toolbox import Toolbox

toolbox = Toolbox(memory_provider)

@toolbox.register_tool
def system_status():
    """Return current system status."""
    return {"status": "ok"}

agent = MemAgentBuilder().with_memory_provider(memory_provider).with_tool(system_status).build()
```

Each tool is stored inside your configured provider with embedding metadata so agents can retrieve the right action based on the natural language plan they produce.

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
