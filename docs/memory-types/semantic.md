# Semantic Memory

Semantic memory stores canonical facts, personas, and entity attributes that rarely change. In MemoRizz, this maps to `src/memorizz/long_term/semantic/` and is backed by the `MemoryType.KNOWLEDGE_BASE` and `MemoryType.ENTITY_MEMORY` enums.

## Components

- **Knowledge Base** – Vectorized documents segmented by namespace or topic.
- **Personas** – Versioned agent identities (name, role, goals, background) with an auditable `evolution_history` and a runtime `update_persona` tool for LLM-driven evolution. Stored under `MemoryType.PERSONAS`.
- **Entity Memory** – Structured attributes for people, organizations, or devices. The `entity_memory` module exposes helper methods to upsert and query profile fields.

## Typical Operations

```python
from memorizz import EntityMemory, KnowledgeBase, MemoryType

kb = KnowledgeBase(memory_provider=provider)

# Ingest a document. The corpus is chunked (default: "fixed" 1000-char
# windows with 100-char overlap). Each chunk is embedded separately.
kb_id = kb.ingest_knowledge(
    corpus="The premium plan includes unlimited vector storage.",
    namespace="support",
    user_id="user-42",
)

# Attach to an agent so its retrievals scope to this knowledge
kb.attach_to_agent(agent, kb_id)

# For a user-facing lookup, call the provider with the exact tenant scope.
hits = provider.retrieve_by_query(
    query="What does premium include?",
    memory_store_type=MemoryType.KNOWLEDGE_BASE,
    namespace="support",
    user_id="user-42",
    limit=3,
)

entities = EntityMemory(provider)
entities.upsert_entity(
    entity_id="company_acme",
    name="Acme",
    entity_type="organization",
    attributes=[
        {
            "name": "plan",
            "value": "premium",
            "confidence": 0.95,
            "source": "billing-system",
        }
    ],
    memory_id="support",
    user_id="user-42",
)
```

The `EntityMemory` helper owns structured entity operations; there is no
mutable `agent.memory.entity_memory` façade. Always supply the same user and
memory scope for entity writes and reads.

### Chunking strategies

`ingest_knowledge` accepts a `chunking_strategy` argument:

- `"fixed"` *(default)* — fixed-size character windows with `chunk_overlap`
- `"sentence"` — packs sentences up to `chunk_size` characters
- `"paragraph"` — splits on blank lines
- `"semantic"` — embeds each sentence and breaks at the largest topic shifts, tuned by `breakpoint_percentile` (default 95; lower = more chunks)
- `"none"` — stores the corpus as a single document
- A callable `(corpus, **kwargs) -> list[str]` for custom logic

The provider automatically embeds each chunk, stores metadata, and tags the
record with the owning agent or namespace. All chunks from one ingest share
one `knowledge_base_id`, so `retrieve_knowledge(kb_id)` returns every chunk
sorted by `chunk_index`.

### Ingesting files and folders

Most workflows don't want to extract text by hand. `ingest_file` and
`ingest_directory` handle it for you — they route through the central
extractor registry in `memorizz.long_term.semantic.extractors`, which the
UI uploader also uses. Same behavior everywhere.

```python
from memorizz import KnowledgeBase

kb = KnowledgeBase(memory_provider=provider)

# Single file — path, bytes, or file-like
kb.ingest_file("docs/handbook.pdf", namespace="handbook")
kb.ingest_file(pdf_bytes, filename="handbook.pdf", namespace="handbook")

# Whole folder, optionally restricted by extension
summary = kb.ingest_directory(
    "docs/",
    recursive=True,
    extensions=[".pdf", ".md"],         # None = every registered type
    chunking_strategy="semantic",        # forwarded to every file
)
print(f"{summary['ingested']} / {summary['total']} succeeded")
for row in summary["results"]:
    if not row["ok"]:
        print("failed:", row["path"], "-", row["error"])
```

**Supported formats out of the box**: every common text type (`.txt`,
`.md`, `.json`, `.csv`, `.py`, `.html`, `.yaml`, `.sql`, and more) plus
**PDF** when `pypdf` is installed:

```bash
pip install 'memorizz[ingest-pdf]'
```

**Add your own format** with one function:

```python
from memorizz.long_term.semantic import register_extractor

def extract_docx(raw: bytes) -> str:
    import docx, io
    return "\n\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)

register_extractor(".docx", extract_docx)
```

Both the SDK and the UI pick up the new extractor immediately — there is
only one registry.

### Ingesting from the Local UI

The playground composer exposes a **📎 Attach** button next to **Send**, and
you can also drag-and-drop files onto the chat area. The UI calls
`KnowledgeBase.ingest_file` per upload, so format support, chunking, and
error messages match the SDK exactly.

## Personas and evolution

A persona captures an agent's stable identity — name, role, goals, and
background. Personas are versioned: every change is recorded in the
persona's `evolution_history` along with a `change_trigger` that links
the change back to the memory unit or conversation that prompted it.

```python
from memorizz.long_term.semantic.persona import Persona, RoleType

persona = Persona(
    name="TechExpert",
    role=RoleType.TECHNICAL_EXPERT,
    goals="Help developers debug production incidents quickly.",
    background="10+ years across backend systems and incident response.",
)
persona.store_persona(provider)

# LLM-driven evolution from within a tool or SDK call
persona.update(
    updates={"goals": "Help developers debug AND design resilient systems."},
    change_trigger={
        "reason": "User asked for architecture feedback across 3 sessions.",
        "source_type": "conversation_memory",
        "source_id": "conv_abc123",
        "agent_id": agent.agent_id,
    },
    provider=provider,
)
```

Every `MemAgent` with a persona attached automatically gets two tools:

- `update_persona(updates, reason, source_type?, source_id?, conversation_id?)`
  — appends a traceable entry to `evolution_history` and persists to the
  PERSONAS collection. After a successful update the agent's embedded
  persona snapshot is refreshed so the two stores stay consistent.
- `read_persona()` — returns the current state plus the full evolution
  history, for when the recent-history summary in the system prompt is
  not enough context.

The agent system prompt always surfaces the current persona, its version,
the five most recent evolution entries, and explicit guidance on when to
call `update_persona` (durable identity shifts only — one-off facts go to
`entity_memory`). A `change_trigger` records the reason, source type,
optional source id/conversation id, and agent id so every evolution remains
auditable.

The Local UI's agent configuration page exposes the evolution model
through two dropdowns: **Load built-in preset** (from `RoleType` +
`PREDEFINED_INFO` via `GET /api/persona-presets`) and **Load saved
persona** (from `GET /api/personas`). Selecting a saved persona preserves
its id via a hidden form field; editing the persona fields before saving
produces a new version with `source_type='ui_form'`.

## When to Use

- Product catalogs and policy manuals
- Persona systems for specialized assistants (support, researcher, interviewer)
- Entity profiles that must persist across sessions and devices

Semantic memory powers long-lived recall. Pair it with episodic memory when you also care about interaction history.
