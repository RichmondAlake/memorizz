# Procedural Memory

Procedural memory captures *how* an agent should act. It bundles tool registration, workflows, and scripted behaviors so that the agent can plan or execute actions consistently. Source code lives in `src/memorizz/long_term_memory/procedural/`.

## Components

- **Toolbox (`MemoryType.TOOLBOX`)** – Python callables wrapped with metadata so LLMs can discover and execute them safely.
- **Workflow Memory (`MemoryType.WORKFLOW_MEMORY`)** – Declarative or code-defined processes that map multi-step plans.
- **Personas** – While technically part of semantic memory, personas often work hand-in-hand with procedural steps to enforce tone and guardrails.

## Registering Tools

```python
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.long_term_memory.procedural.toolbox import Toolbox

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
