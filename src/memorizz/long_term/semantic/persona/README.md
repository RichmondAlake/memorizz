# Persona Module

`Persona` defines an agent's stable identity: name, role, goals, and
background. Personas are **versioned**: every call to `Persona.update`
appends a traceable entry to the persona's `evolution_history`, linking
the change to the conversation or memory unit that triggered it. This
gives an agent a continuous, auditable record of how its persona has
evolved — and why.

Canonical imports:

```python
from memorizz.long_term.semantic.persona import Persona, RoleType
```

## Quick start

```python
from pathlib import Path

from memorizz.long_term.semantic.persona import Persona, RoleType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))

persona = Persona(
    name="TechExpert",
    role=RoleType.TECHNICAL_EXPERT,
    goals="Help developers debug production incidents quickly.",
    background="10+ years across backend systems and incident response.",
)

# Store in the PERSONAS collection — returns the provider-assigned id
storage_id = persona.store_persona(provider)

# Rehydrate a stored persona (no default-merging, preserves embedding,
# version, and evolution_history)
same_persona = Persona.retrieve_persona(storage_id, provider)
persona_restored = Persona.from_dict(same_persona)

# Vector search over stored personas
similar = Persona.get_most_similar_persona("senior backend mentor", provider, limit=1)

# List all personas (used by the UI picker)
all_personas = Persona.list_personas(provider)
```

## Use with MemAgent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_persona(persona)
    .with_instruction("You are concise, practical, and technical.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

# Persona can also be changed at runtime
agent.set_persona(persona)
```

## Persona evolution

A persona exposes an `update` method that applies a partial set of changes
and appends a history entry with **change_trigger** metadata. The agent has
a built-in `update_persona` tool wired to this method — so LLM-driven
evolution is the default runtime path — but you can also call it directly
from the SDK.

```python
result = persona.update(
    updates={"goals": "Help developers debug AND design resilient systems."},
    change_trigger={
        "reason": "User repeatedly asked for architecture feedback across "
                  "three sessions — expand scope beyond debugging.",
        "source_type": "conversation_memory",
        "source_id": "conv_abc123",
        "conversation_id": "thread_7",
        "agent_id": agent.agent_id,
    },
    provider=provider,
)

print(result["version"])             # incremented (e.g. 2)
print(result["history_entry"])       # appended to persona.evolution_history
```

### `change_trigger` schema

| Field             | Required | Notes                                                                           |
|-------------------|----------|---------------------------------------------------------------------------------|
| `reason`          | yes      | Non-empty string. Saved verbatim to history for future audit.                   |
| `source_type`     | no       | A `MemoryType` value (e.g. `conversation_memory`, `summaries`, `entity_memory`) or one of `manual`, `ui_form`, `user_feedback`, `llm_inference`, `tool_call`. |
| `source_id`       | no       | Storage id of the triggering memory unit. Enables reverse lookup.               |
| `conversation_id` | no       | Thread id, when the change originated in a conversation.                        |
| `agent_id`        | no       | Agent that initiated the change.                                                |
| `triggered_at`    | no       | ISO-8601 timestamp. Auto-populated when absent.                                 |

Only the listed fields are considered updatable: `name`, `role`, `goals`,
`background`. Unknown keys are ignored with a warning. Changes that match
the current value are dropped (no-op). Each applied change is stored
with `{"old": ..., "new": ...}` so the full trail can be reconstructed.

### Evolution history layout

```json
{
  "version": 3,
  "timestamp": "2026-04-19T14:02:55.112893",
  "changes": {
    "goals": {"old": "Help developers debug.", "new": "Help developers debug AND design."}
  },
  "change_trigger": {
    "reason": "User repeatedly asked for architecture feedback.",
    "source_type": "conversation_memory",
    "source_id": "conv_abc123",
    "conversation_id": "thread_7",
    "agent_id": "agent_xyz",
    "triggered_at": "2026-04-19T14:02:55.112880"
  }
}
```

## The `update_persona` tool (agent runtime)

Every `MemAgent` with an attached persona gets two tools registered
automatically:

- **`update_persona(updates, reason, source_type?, source_id?, conversation_id?)`**
  — delegates to `Persona.update`. The agent's document snapshot is
  refreshed (`agent.save()`) after a successful update so the PERSONAS
  collection and the agent's embedded persona stay consistent.
- **`read_persona()`** — returns the current persona state plus the full
  `evolution_history`, useful when the recent-history summary in the
  system prompt is not enough context.

The agent system prompt includes:

1. The current persona block (name, role, goals, background).
2. The persona version (e.g. `Persona version: 3.`).
3. A summary of the 5 most recent evolution entries.
4. Explicit guidance on when to call `update_persona`, which
   `change_trigger` fields to populate, and how to maintain continuity
   with prior versions.

**When agents should call `update_persona`:**

- The user explicitly asks for a different persona/role.
- A sustained pattern across conversations warrants a durable shift in
  goals or background (one-off user facts belong in `entity_memory`, not
  the persona).
- New information fundamentally changes the long-term framing of the
  agent.

**When agents should NOT call it:**

- To record a one-off user preference → use `entity_memory_upsert`.
- To remember a specific conversation fact → rely on conversation/summary
  memory; persona is for *identity*, not facts.

## Storage and consistency

- Personas live in the `MemoryType.PERSONAS` collection. A persona's
  provider-assigned id (`_id` for Mongo, `id` for filesystem) is captured
  into `Persona._storage_id` and surfaced in `to_dict()` as `storage_id`
  so it round-trips on agent documents that embed the persona.
- Updates target the existing record via `provider.update_by_id`. If that
  fails for any reason, the persona is re-stored and a `persistence_note`
  is appended to the history entry so the inconsistency is visible.
- `Persona.from_dict` is the canonical rehydration entry point; unlike the
  constructor, it does *not* re-merge `PREDEFINED_INFO` defaults (which
  would double-apply each reload) or regenerate the embedding.

## Local UI integration

The agent configuration page offers two dropdowns above the persona
fields:

- **Load built-in preset** — pulls from
  `RoleType` + `PREDEFINED_INFO` via `GET /api/persona-presets`. Selecting
  a preset fills the form and clears any saved-persona link so a new
  persona record is created on save.
- **Load saved persona** — lists existing PERSONAS documents via
  `GET /api/personas`. Selecting a saved persona populates the form and
  preserves the link via a hidden `persona_id` field. Editing the fields
  before saving triggers a `ui_form`-sourced `Persona.update` on the
  linked record, producing a new version.

## Production notes

- Embedding regeneration is automatic when any semantic field (`name`,
  `role`, `goals`, `background`) changes.
- Evolution history grows monotonically. For very long-lived personas
  consider archiving older entries outside the collection; the first
  version of the module keeps history unbounded and inline.
- The API is concurrency-aware at the provider layer (single-document
  updates); concurrent `update_persona` calls on the same persona may
  interleave and produce version gaps. If strict ordering matters for
  your application, serialize updates at the agent level.
