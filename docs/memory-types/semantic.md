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

### Large entities and their vectors

Each entity is embedded as one text: its name, type, attributes and
relations. Embedding models have an input limit (8,191 tokens for OpenAI's).
An entity past it still gets a vector, made from its name, type and newest
attributes, up to about 90% of the limit. Tokens are estimated at three
characters each. When the provider still rejects the text as too long, it is
halved and sent again.

The record says when its vector does not cover all of it:

```python
report = entities.upsert_entity_with_report(name="Ines", attributes=facts,
                                            memory_id="m1", user_id="u1")
report["embedding"]
# {'status': 'partial', 'reason': 'over_token_limit',
#  'attributes_covered': 41, 'attributes_total': 160}

record = entities.get_entity(report["entity_id"], memory_id="m1", user_id="u1")
record["metadata"]["embedding_status"]   # "partial"
EntityMemory.embedding_report(record)    # the same summary, from the record
```

- `partial`: the vector covers the name, type and newest facts. The reason is
  `over_token_limit`, `provider_length_error`, or `previous_vector_kept` (a
  re-embed failed and the old vector stays).
- `missing`: there is no vector (`embedding_failed`).

The keys go away when a later write embeds the whole record. Every fact is
still stored: lookups by name or id, and the exact fallback search, find it.
The agent's `entity_memory_upsert` tool result carries `embedding` when the
vector is partial or missing, and the memory pages show *partial vector* or
*no vector* next to the entity. Set the limit with
`EntityMemory(provider, max_embedding_tokens=...)` or
`MEMORIZZ_ENTITY_EMBEDDING_MAX_TOKENS`.

### Entity memory in an agent

- With a background model set, the agent files the entity facts of each user
  message after the turn, without waiting for the chat model to call
  `entity_memory_upsert`. See
  [entity capture](../guides/memory-routing.md#entity-capture-without-routing).
- `entity_memory_lookup` is offered only when the user's memory and user scope
  holds an entity. Until then the model gets `entity_memory_upsert` alone, and
  its prompt says there is nothing to look up yet. The check is one small query
  per scope, cached for 30 seconds while the scope is empty and five minutes
  once it is not. A write through `EntityMemory` in the same process makes the
  lookup available on the next turn.

### One record for the user

Without a policy, entity writes find their record by name, so "user", "me"
and the user's real name can become three records of one person. An
`EntityPolicy` names the user's record by an identity key the host owns:

```python
from memorizz import EntityPolicy, MemAgent

agent = MemAgent(
    ...,
    entity_policy=EntityPolicy(self_identity_key="authenticated_user"),
)
# or agent.with_entity_memory(True, policy={"self_identity_key": "authenticated_user"})
```

Every write path follows it: the `entity_memory_upsert` tool, entity capture
after a turn, memory routing, `remember` and `capture_entity_facts`. A write
about the user (named "user", "me", the record's own name, or a name the record
holds such as `first_name`) lands on that one record, whatever name or entity id
the writer gave. Fact extraction reads the record under the name "user", so
facts it already holds are not filed again.

`subjects="self"` keeps only facts about the user. Capture and routing drop
facts about anyone else, and an `entity_memory_upsert` call about anyone else is
refused: the model is told to store facts about the user as "user", so "my
colleague Ben teaches physics" never becomes the user's subject. Use it for an
application whose entity memory is a user profile. Each turn then reads the
user's record by its key (`EntityMemory.get_by_identity_key`) rather than
searching by meaning: no query embedding and no vector search. The policy is
saved with the agent, and `agent.describe_memory()["entity_policy"]` shows it.

### Where each fact came from

Each attribute keeps the conversation that taught it (`thread_id`; the tool and
capture fill it in) and, when its value is replaced, the earlier values in
`history`, newest first, at most five. Prompts and vectors never read the
history. The same value said again from the same place changes nothing, and a
change the vector does not read (provenance, confidence, pinning) keeps the
stored vector, so it costs no embedding call.

Forget what one conversation taught, and keep what others did:

```python
EntityMemory(provider).forget_thread("thread-42", memory_id="...", user_id="...")
# {'entities_changed': 1, 'entities_deleted': 0,
#  'attributes_removed': 1, 'attributes_restored': 1}
```

An attribute from that conversation goes back to its newest earlier value from
somewhere else, or goes away. The conversation's entries also leave every
history and the retired facts. An entity left with nothing is deleted.
`sources=[...]` also matches attributes by `source`, for values written before
attributes carried a thread. `forget_sources(["doc_comment:c1"], ...)` does the
same for particular sources alone, for example when that comment is deleted.

### Pinned facts

An attribute written with `pinned=True` is the user's own: something they set
or confirmed. A write that is not pinned (the chat model, capture, routing)
never replaces it; `upsert_entity_with_report` lists it under `kept_pinned`.
Consolidation does not retire it. A pinned write replaces it, and
`retire_attributes(..., include_pinned=True)` takes it out of prompts when the
user asks, keeping a copy that can be restored; `forget_attributes(entity_id,
names, ...)` deletes facts for good, with their history and any retired copy.

### Facts from outside the chat

`agent.capture_entity_facts(text, ...)` files the facts in words the user wrote
somewhere else, such as a comment on a document or an edit, the way capture
does after a turn. It runs at once on the background model (the agent's model
when there is none) and follows the policy.

```python
agent.capture_entity_facts(
    "Make every chart match: our brand green is #0E4F2F",
    user_id="u1",
    thread_id="deck:42",            # forget_thread("deck:42") takes it out
    source="slide_comment:c1",      # stored on each attribute
    description="a comment the user left on their slide deck",
)
```

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
