# Semantic Memory

Semantic memory stores facts, concepts, and structured entity knowledge that should persist beyond a single conversation.

Key components:

- `KnowledgeBase`
- `Persona`
- `EntityMemory`

## Knowledge Base Example

```python
from pathlib import Path

from memorizz.long_term.semantic import KnowledgeBase
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
kb = KnowledgeBase(provider)

kb_id = kb.ingest_knowledge(
    corpus="Python supports multiple paradigms including object-oriented and functional programming.",
    namespace="python_basics",
)

results = kb.retrieve_knowledge_by_query("What paradigms does Python support?", namespace="python_basics", limit=3)
print(results)
```

### Chunking

`ingest_knowledge` splits the corpus into chunks so that semantic retrieval
lands on the most relevant passage rather than the full source. Each chunk
becomes its own embedded document; all chunks for one ingest share one
`knowledge_base_id`.

```python
# Fixed-window chunking is the default (1000 chars, 100-char overlap)
kb.ingest_knowledge(corpus=long_text, namespace="handbook")

# Override the strategy and knobs
kb.ingest_knowledge(
    corpus=long_text,
    namespace="handbook",
    chunking_strategy="sentence",  # or "fixed" | "paragraph" | "none"
    chunk_size=500,
    chunk_overlap=50,               # only used by "fixed"
)

# Or supply a custom callable: (corpus, **kwargs) -> list[str]
kb.ingest_knowledge(
    corpus=long_text,
    namespace="handbook",
    chunking_strategy=lambda text, **_: text.split("---"),
)
```

Built-in strategies:

| Strategy    | Behavior                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------ |
| `fixed`     | Fixed-size character windows with optional overlap. **Default.**                           |
| `sentence`  | Packs sentences up to `chunk_size` characters.                                             |
| `paragraph` | Splits on blank lines.                                                                     |
| `semantic`  | Embeds each sentence, breaks at the largest topic shifts (tuned by `breakpoint_percentile`). |
| `none`      | Stores the full corpus as a single document (original behavior).                           |

**Semantic chunking** embeds every sentence up front, so it costs one extra
embedding call per sentence and is slower than the syntactic strategies —
prefer it for heterogeneous documents where topic boundaries matter more
than chunk-size uniformity. `breakpoint_percentile` (default `95`) controls
how aggressive the splits are: lower values produce more chunks, higher
values produce fewer.

```python
kb.ingest_knowledge(
    corpus=long_text,
    namespace="handbook",
    chunking_strategy="semantic",
    breakpoint_percentile=90,
)
```

Each stored entry includes `chunk_index`, `chunk_count`, and
`chunking_strategy` for inspection. `retrieve_knowledge(kb_id)` returns every
chunk for a source ingest, sorted by `chunk_index`. `update_knowledge` does a
delete-then-reinsert so the new chunk count reflects the new corpus.

### Ingesting files and folders

For files on disk or in memory, use `ingest_file` / `ingest_directory` — they
go through the centralized extractor registry in
[`extractors.py`](extractors.py) and then call `ingest_knowledge` under the
hood. The local UI uploader goes through the same code path, so behavior is
identical whether you ingest from Python or from the web UI.

```python
# Single file from disk
kb_id = kb.ingest_file("docs/handbook.pdf", namespace="handbook")

# Raw bytes (filename is required so the extension can be detected)
kb_id = kb.ingest_file(pdf_bytes, filename="handbook.pdf", namespace="handbook")

# Open file-like object
with open("notes.md", "rb") as fh:
    kb_id = kb.ingest_file(fh, namespace="notes")

# Whole folder — one knowledge_base_id per file, per-file errors collected
summary = kb.ingest_directory(
    "docs/",
    recursive=True,
    chunking_strategy="semantic",     # forwarded to every file
)
print(summary["ingested"], "/", summary["total"])
for row in summary["results"]:
    if not row["ok"]:
        print("failed:", row["path"], "-", row["error"])
```

### Supported formats & adding your own

Out of the box the registry handles every common text format
(`.txt`, `.md`, `.json`, `.csv`, `.py`, `.html`, `.yaml`, `.sql`, …) plus
**PDF** when `pypdf` is installed:

```bash
pip install 'memorizz[ingest-pdf]'
```

Register your own extractor for any format in one call:

```python
from memorizz.long_term.semantic import register_extractor

def extract_docx(raw: bytes) -> str:
    import docx, io
    doc = docx.Document(io.BytesIO(raw))
    return "\n\n".join(p.text for p in doc.paragraphs)

register_extractor(".docx", extract_docx)

# Both the SDK and the UI pick it up immediately
kb.ingest_file("contract.docx", namespace="legal")
```

Extractor errors are structured so callers can handle them cleanly:

| Exception                     | Meaning                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `UnsupportedFileType`         | No extractor registered for this extension.                  |
| `MissingExtractorDependency`  | Extractor exists but its optional package isn't installed.   |
| `EmptyDocumentError`          | Extraction ran but returned no text (likely scanned PDF).    |
| `ExtractionError`             | Catch-all for extractor-specific failures.                   |

All inherit from `ExtractorError`, which is also exported from the package.

### Ingesting from the Local UI

The [Local UI](../../../ui/README.md) playground accepts files two ways:
click 📎 **Attach** next to **Send**, or drag-and-drop onto the chat area.
Internally it calls `KnowledgeBase.ingest_file` for each uploaded file, so
format support, chunking, and error messages match the SDK exactly.

## Entity Memory Example

```python
from memorizz.long_term.semantic.entity_memory import EntityMemory

entities = EntityMemory(provider)

entity_id = entities.upsert_entity(
    name="Leah",
    entity_type="user",
    memory_id="tenant-1",
    attributes=[{"name": "preferred_language", "value": "Python", "confidence": 0.95}],
)

profile = entities.get_entity_profile(entity_id)
```

## With MemAgent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_instruction("Use stored knowledge and entity facts in responses.")
    .with_memory_provider(provider)
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .build()
)

agent.with_entity_memory(True)
response = agent.run("What do you know about Leah's preferred language?")
```

Notes:

- Semantic entries are stored under `MemoryType.KNOWLEDGE_BASE`.
- Entity profiles are stored under `MemoryType.ENTITY_MEMORY`.
