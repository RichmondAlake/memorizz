# Memorizz Memory Architecture

This document explains how Memorizz organizes memory for single-agent and multi-agent systems.

## Design Goals

- Separate memory concerns by cognitive role (semantic, episodic, procedural, short-term, shared)
- Keep storage backend-agnostic through a `MemoryProvider` interface
- Support mode-driven defaults (`assistant`, `workflow`, `deep_research`)
- Allow optional runtime capabilities (internet access, sandbox execution, skills, MCP)

## Module Layout

```text
src/memorizz/
├── long_term/
│   ├── semantic/
│   │   ├── knowledge_base.py
│   │   ├── persona/
│   │   └── entity_memory/
│   ├── procedural/
│   │   ├── toolbox/
│   │   └── workflow/
│   └── episodic/
│       ├── conversational_memory_unit.py
│       └── summary_component.py
├── short_term_memory/
│   ├── working_memory/
│   └── semantic_cache.py
├── coordination/
│   └── shared_memory/
├── memory_provider/
│   ├── oracle/
│   ├── mongodb/
│   └── filesystem/
└── memagent/
    ├── core.py
    └── managers/
```

## Memory Types

`MemoryType` is defined in `src/memorizz/enums/memory_type.py`.

- `KNOWLEDGE_BASE`: semantic facts/knowledge
- `ENTITY_MEMORY`: structured entity attributes and updates
- `TOOLBOX`: executable tools and metadata
- `WORKFLOW_MEMORY`: process/task state
- `CONVERSATION_MEMORY`: chat timeline
- `SUMMARIES`: compressed conversation summaries
- `SHORT_TERM_MEMORY`: active context scratchpad
- `SEMANTIC_CACHE`: similar-query response cache
- `SHARED_MEMORY`: multi-agent blackboard/session coordination
- `MEMAGENT`: persisted agent configuration/state

## Application Mode Defaults

`ApplicationModeConfig` in `src/memorizz/enums/application_mode.py` maps modes to default memory stacks:

- `assistant`
  - `CONVERSATION_MEMORY`, `KNOWLEDGE_BASE`, `PERSONAS`, `ENTITY_MEMORY`, `SHORT_TERM_MEMORY`, `SUMMARIES`
- `workflow`
  - `WORKFLOW_MEMORY`, `TOOLBOX`, `KNOWLEDGE_BASE`, `SHORT_TERM_MEMORY`, `SUMMARIES`
- `deep_research`
  - `TOOLBOX`, `SHARED_MEMORY`, `KNOWLEDGE_BASE`, `SHORT_TERM_MEMORY`, `SUMMARIES`

These defaults can be overridden by passing explicit `memory_types` to `MemAgent`.

## Runtime Orchestration (MemAgent)

`MemAgent` (`src/memorizz/memagent/core.py`) coordinates memory and tools through manager components:

- `MemoryManager`
- `EntityMemoryManager`
- `ToolManager`
- `CacheManager`
- `PersonaManager` — exposes a versioned `Persona` (stable identity with
  `evolution_history`) and registers the `update_persona` / `read_persona`
  tools on every agent that has a persona attached. Changes are traceable
  via a `change_trigger` (reason + source memory/conversation id).
- `WorkflowManager`
- `InternetAccessManager`
- `SandboxManager` (when configured)

Request lifecycle (high level):

1. Resolve `memory_id` and `thread_id`
2. Try semantic cache (if enabled)
3. Build context from active memory types
4. Execute LLM/tool loop
5. Persist user + assistant interaction
6. Update context-window stats and summary registry as needed

## Short-Term Context Control

Memorizz includes built-in context-window observability and compression helpers:

- `get_context_window_stats()` for latest token usage snapshot
- summary generation (`generate_summaries`) for history compression
- summary registry helpers for retrieving recent summary metadata

## Multi-Agent Coordination

`SharedMemory` (`src/memorizz/coordination/shared_memory/shared_memory.py`) stores a shared session payload and blackboard entries for orchestrated workflows.

Structured helpers include:

- `post_command(...)`
- `post_status(...)`
- `post_report(...)`

`MultiAgentOrchestrator` and `DeepResearchOrchestrator` use these messages to coordinate delegates and synthesis agents.

## Provider Abstraction

All persistence backends implement the `MemoryProvider` interface.

Built-in providers:

- Oracle (`src/memorizz/memory_provider/oracle/`)
- MongoDB (`src/memorizz/memory_provider/mongodb/`)
- Filesystem (`src/memorizz/memory_provider/filesystem/`)

Because memory logic is provider-agnostic, the same agent code can switch backends with minimal changes.

## Optional Capability Layers

Memorizz can attach runtime extensions on top of the memory architecture:

- Internet access providers (`internet_search`, `open_web_page` tools)
- Sandbox providers (`execute_code`, `sandbox_write_file`, `sandbox_read_file` tools)
- Skill path loading (`list_skills`, `read_skill`, `run_skill_code`, `run_skill_script`)
- MCP server tooling (`list_mcp_servers`, `mcp_list_tools`, `mcp_call_tool`)

These layers are additive and do not replace the core memory model.

## Practical Guidance

- Use `assistant` mode for persistent conversational agents.
- Use `workflow` mode for deterministic task/tool pipelines.
- Use `deep_research` mode for coordinated multi-agent exploration with shared memory.
- Start with the filesystem provider for local development; move to Oracle/MongoDB for heavier workloads.
