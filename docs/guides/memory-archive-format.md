# MemoRizz memory archive format

**MemoRizz Memory Archive v1** is a portable JSON format for exporting and
importing a memory graph. Files end in **`.memorizz.json`** and identify themselves
with `"format": "memorizz.memory"` and the integer `"version": 1`.

The same format is used by the UI, Python SDK, CLI, MCP server, and Codex and
Claude Code plugins. Filesystem, MongoDB, Oracle AI Database and Notion providers
implement archive storage. See [Export and import memory](memory-transfer.md)
for the UI walkthrough and commands.

## Envelope

An archive has exactly six top-level fields. Version 1 requires all 13 store
keys and their taxonomy mappings, even when an export selects only one type.
Unselected or empty stores contain `[]`.

| Field | Type | Meaning |
|---|---|---|
| `format` | String | Always `memorizz.memory` |
| `version` | Integer | Always `1`; unsupported versions are rejected |
| `taxonomy` | Object | Exact canonical store-to-category mapping below |
| `manifest` | Object | Export provenance, scope, counts, included features and omissions |
| `stores` | Object | All 13 store keys, each containing an array of records |
| `integrity` | Object | `algorithm: "sha256"` and a 64-character lowercase hexadecimal `digest` |

The archive version describes the file contract. `manifest.memorizz_version`
records the package version that produced it; these are separate version numbers.

## Taxonomy

The keys below are the canonical names used in both `taxonomy` and `stores`.
For example, the agent configuration store is `agents`, rather than `memagent`.

| Store key | Taxonomy category | Records |
|---|---|---|
| `personas` | `long_term.semantic` | Personas and behavioral configuration |
| `entity_memory` | `long_term.semantic` | Structured entities, facts and relationships |
| `knowledge_base` | `long_term.semantic` | Knowledge, documents and preferences |
| `toolbox` | `long_term.procedural` | Tool definitions and schemas |
| `workflow_memory` | `long_term.procedural` | Recorded workflows |
| `skillbox` | `long_term.procedural` | Reusable skills |
| `conversation_memory` | `long_term.episodic` | Conversation messages |
| `summaries` | `long_term.episodic` | Summaries and source-message links |
| `tool_log` | `long_term.episodic` | Tool calls and their recorded results |
| `short_term_memory` | `short_term` | Working memories |
| `semantic_cache` | `short_term` | Cached queries and responses |
| `shared_memory` | `coordination` | Shared coordination and available history/context records |
| `agents` | `agent_configuration` | Saved agents, configured delegates and namespace bindings |

## Records and relationships

Every entry in a store is an object with exactly `id` and `data`:

```json
{
  "id": "dfbde207-859c-4cde-9921-37e959d31ade",
  "data": {
    "memory_id": "harbor-demo",
    "user_id": null,
    "content": "The release owner is Mira. Launch is after 19:00 UTC."
  }
}
```

`id` is a nonempty string, unique within its store, with at most 512 characters.
`data` holds the portable record fields for that memory type. When present,
`data.id` and `data._id` must match the enclosing `id`. Natural identity fields
such as `agent_id` for `agents`, `persona_id` for `personas`, and `summary_id`
for `summaries` must also agree with it.

Scope and relationships remain explicit fields in `data`: for example,
`memory_id`, `user_id`, `agent_id`, `owner_agent_id`, `delegates`, `source_ids`,
`supersedes`, and `source_message_ids`. Nested coordination and observability
payloads retain their structured links. A link does not imply that its target
is included: deleted or unselected targets are reported as external references.

Restoring with preserved IDs keeps those identities. Cloning with new IDs
rewrites documented identity and reference fields, while preserving memory
prose and keeping delegate-private namespaces separate. A namespace identifies
the memory's scope; it is distinct from the record's own `id`.

## Manifest

The required fields are `created_at`, `memorizz_version`, `source_provider`,
`scope`, `counts`, `record_count`, `includes`, and `warnings`.

| Field | Meaning |
|---|---|
| `created_at` | Export timestamp in UTC |
| `memorizz_version`, `source_provider` | Producing package version and provider |
| `scope` | Selected agent/namespace, included agent IDs, user scope and application scope |
| `counts` | Record count for every store, including zeros |
| `record_count` | Sum of the per-store counts |
| `includes` | Whether delegates, history, captured context and embeddings were requested |
| `omitted_fields` | Fields removed during export, when present |
| `external_references` | Explicit references outside the selected graph, when present |
| `warnings` | Export omissions and limitations |

`scope.user_scoped` distinguishes an exact user selection from an administrative
export. `user_scoped: true` with `user_id: null` means anonymous records;
`user_scoped: false` means the export was not filtered by user. The importer
still enforces its caller's scope and destination policy.

## Complete example

[Download a complete v1 example](../assets/examples/memory-archive-v1.memorizz.json).
It contains one fictional knowledge-base record in `harbor-demo`, all 13 store
keys, matching manifest counts and a valid checksum. The other stores are empty.
You can preview this file through the UI or CLI before importing it.

```bash
memorizz memory import ./memory-archive-v1.memorizz.json
```

The file is ordinary JSON: inspect `taxonomy` to see the categories, `stores`
to see records, and `manifest` to understand the export's scope. The record
example above is an individual entry, not a complete importable archive.

## Integrity and validation

The packaged JSON Schema is
`memorizz/schemas/memory-archive-v1.schema.json`. Load it from an installed
package if you are building another importer:

```python
import json
from importlib.resources import files

schema = json.loads(
    files("memorizz").joinpath("schemas/memory-archive-v1.schema.json").read_text()
)
```

Use the SDK's validator to check the complete archive, including the checksum
and cross-field rules that JSON Schema alone does not verify:

```python
import json
from pathlib import Path
from memorizz.memory_archive import validate_archive

archive = json.loads(Path("memory-archive-v1.memorizz.json").read_text())
validated = validate_archive(archive)
print(validated["manifest"]["record_count"])
```

Validation rejects unsupported versions, unknown envelope fields, incorrect
taxonomy, missing store keys, duplicate or inconsistent IDs, incorrect counts,
and checksum mismatches. The default archive limits are 50 MiB and 100,000
records. Transport limits can be smaller; inline MCP requests default to 4 MiB.

The checksum covers canonical UTF-8 JSON of every field except `integrity`:
object keys sorted, compact separators, literal Unicode, and no NaN or Infinity.
Numbers use a shared Python/browser representation: integral floats omit `.0`;
magnitudes from `1e-6` through less than `1e21` use decimal notation; other floats
use an exponent without zero padding, with `+` for positive exponents; negative
zero is `0`. Use strings for identifiers and exact integers beyond JavaScript's
safe integer range.

Whitespace and object-key order in the file do not affect the checksum. An
intentional SDK transformation can recompute it with
`memorizz.memory_archive.seal_archive()`, then run validation again. A checksum
detects corruption; it does not authenticate the source or encrypt memory values.

## What is portable

An archive can preserve current values, ownership, configured agent/delegate
relationships, private and shared namespaces, explicit derivation links, and
available evolution and context records. Recorded history retains timestamps,
writers, sources, hashes and changed field names; it cannot reconstruct older
or deleted values. Backend-local history heads are excluded.

Credentials and executable runtime objects are omitted. Embeddings are excluded
by default and discarded on standard import; explicit destination re-embedding
is available through the SDK and CLI. Tool schemas and import references remain
data, and imported agent automations start disabled. Attachments, model weights,
external services and the harness SQLite run database are outside this format.

Exports can overlap concurrent writes. Imports preview conflicts before writing,
but restore is not atomic across stores: a provider failure can leave a partial
import, with committed progress reported. See the
[export and import guide](memory-transfer.md) for conflict policies, permissions
and provider-specific behavior.
