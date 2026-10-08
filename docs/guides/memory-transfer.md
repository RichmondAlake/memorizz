# Export and import memory

MemoRizz exports a memory graph to a portable **`.memorizz.json`** archive and
restores it through the SDK, CLI, portal, MCP server, and coding-agent plugins.
Agent exports include configured delegates, their separate private namespaces,
owned shared records, and available evolution/context records. A project export
selects one namespace; a trusted local export without filters selects the store.

## Portal

Open **Memory types → Export & import**, or select **Export & import memory**
at the top of the Playground's **Memory** tab. The Playground link preselects
the current agent. Memory explorers and Memory evolution also link to this page.
Its direct route is `/memory-transfer` on your running MemoRizz UI. Connect a
memory provider first.

### Download memories

1. Select an **Agent**, or leave **All agents and memories** selected. Optionally
   enter a **Memory namespace** to narrow the export.
2. Select the memory taxonomy types. Choose whether to include delegates and
   their private memories, recorded memory evolution, and captured context.
3. Click **Download archive** to save a `.memorizz.json` file.

### Restore or copy memories

1. Connect the UI to the **destination** memory provider. Imports write to the
   currently connected store.
2. Under **Import memory**, select the archive file.
3. Choose **Preserve IDs** to restore the original graph or **New IDs** to create
   an independent copy with remapped internal links and separate namespaces.
4. Choose how to handle existing records: stop, skip, or replace matching records.
5. Click **Preview import** and review the planned counts, conflicts and warnings.
6. Click **Import reviewed archive**, then inspect the result and any errors.

Changing the file or options invalidates the preview. Unrestricted administrator
access is required; trace-only accounts cannot export memory values. Read-only
portals allow export and block import. Standard UI imports omit embeddings; use
the CLI or SDK's explicit re-embedding option before relying on semantic search.

## CLI

Commands use the currently configured memory provider. Configure the destination
provider before importing into another backend.

```bash
memorizz memory export ./harbor.memorizz.json --agent-id AGENT_ID
memorizz memory export ./project.memorizz.json --memory-id PROJECT_ID --no-context

memorizz memory import ./harbor.memorizz.json                 # preview only
memorizz memory import ./harbor.memorizz.json --apply          # preserve IDs
memorizz memory import ./harbor.memorizz.json --new-ids --apply # clone graph
memorizz memory import ./harbor.memorizz.json --conflict skip --apply
memorizz memory import ./harbor.memorizz.json --conflict replace --apply
```

Export accepts `--types knowledge_base,conversation_memory,summaries`,
`--no-delegates`, `--no-history`, `--no-context`, `--user-id NAME`, and
`--anonymous`. File exports use mode `0600`; existing files require
`--overwrite`. Browser downloads use their browser/OS file permissions.

Import defaults to a preview and refuses existing records. `--conflict skip`
keeps existing records; `replace` overwrites matching ones. `--new-ids` creates
new record, agent, namespace and execution IDs, rewriting structured internal
references while preserving memory prose. Delegate namespaces remain separate.
`--target-memory-id NEW_NAMESPACE` remaps a single-namespace archive; it refuses
to collapse several delegate namespaces into one.

User scope omitted means trusted local administrator. `--user-id NAME` or
`--anonymous` validates incoming records and destination conflicts against the
exact tenant. MCP always binds to its caller's tenant.

Embeddings are omitted by default and discarded on standard import, avoiding
model/dimension mismatches. `--reembed` explicitly calls the **destination**
embedding model and may incur costs. Without it, restored values remain readable
but semantic retrieval may not find them. Notion projections stay pending until
`provider.repair_index()` or `memorizz notion sync` reindexes them.

## SDK

```python
from memorizz import MemoryArchive, FileSystemConfig, FileSystemProvider

source = FileSystemProvider(FileSystemConfig(root_path="./source-memory"))
target = FileSystemProvider(FileSystemConfig(root_path="./restored-memory"))

archive = MemoryArchive(source).export(agent_id="AGENT_ID")
preview = MemoryArchive(target).import_archive(archive)  # dry_run=True
assert preview["ok"]
result = MemoryArchive(target).import_archive(archive, dry_run=False)
assert result["ok"], result["errors"]

MemoryArchive(source).export_file("./backup.memorizz.json", memory_id="PROJECT_ID")
result = MemoryArchive(target).import_file(
    "./backup.memorizz.json", dry_run=False, id_strategy="new"
)
```

Export options: `memory_types`, `include_delegates`, `include_history`,
`include_context`, `include_embeddings`, exact `user_id`, `application_id`,
`max_bytes` and `max_records`. `include_embeddings=True` retains vectors for
inspection; import regenerates vectors only with `reembed=True`.

Import options: `dry_run`, `conflict="error" | "skip" | "replace"`,
`id_strategy="preserve" | "new"`, `target_memory_id`, exact user/application
scope, `allowed_types`, and `reembed`. Results contain `planned`, `imported`,
`skipped`, per-type `counts`, `conflicts`, `id_map`, `warnings`, `errors`, and
`atomic=False`. Preview clone IDs are illustrative: apply generates a fresh map
and checks current conflicts again.

Filesystem, MongoDB, Oracle and Notion implement archive storage. Oracle restores
native columns and preserves other portable fields in an additive JSON
`archive_extras` column, created on first import into each affected table. This
requires schema alteration privileges; preview is read-only. Non-UUID identities
for Oracle RAW-backed stores map deterministically to UUIDs, with references
updated in `id_map`. Agent/delegate relationships restore in two phases.
Third-party providers must preserve the planned record IDs; a provider that
changes an ID produces a reported restore error.

## MCP and plugins

```text
memorizz_export_memories(memory_id="PROJECT_ID")
memorizz_import_memories(archive=ARCHIVE, dry_run=true)
memorizz_import_memories(archive=ARCHIVE, dry_run=false, id_strategy="new")
```

Export and preview require read scope. Apply requires write scope and the server's
write policy. MCP validates the caller's exact tenant and agent allowlist. Remote
transports expose tenant-scoped stores; global agents/personas/tools/skills and
shared coordination are excluded. Inline archives must fit the configured request
limit (4 MiB by default). Limits fail explicitly instead of truncating records;
use the local CLI file interface for larger archives.

Both Codex and Claude Code plugins include the `memory-transfer` skill. Claude
Code also has `/memorizz:export-memory` and `/memorizz:import-memory` commands.

```bash
memorizz plugin export-memory ./project.memorizz.json
memorizz plugin import-memory ./project.memorizz.json          # preview
memorizz plugin import-memory ./project.memorizz.json --apply
```

These commands select the project's namespace and configured user. Import targets
the current project for single-namespace archives; `--preserve-namespace` keeps
the original. Use the general memory CLI for multi-namespace agent graphs. A
configured hosted MCP server is supported without opening the local store.
Upgrade MemoRizz and update/reinstall plugins to expose the new tools and skills
in already installed plugin bundles.

## Format v1

The format is **MemoRizz Memory Archive v1**, identified by
`"format": "memorizz.memory"` and `"version": 1`. See the
[memory archive format reference](memory-archive-format.md) for the full
structure, all 13 taxonomy stores, identity rules, checksum validation, and a
complete downloadable example.

The packaged machine-readable schema is
`memorizz/schemas/memory-archive-v1.schema.json`. The envelope contains:

| Field | Meaning |
|---|---|
| `format` | Always `memorizz.memory` |
| `version` | Integer `1`; unknown versions are rejected |
| `taxonomy` | Canonical store-to-category mapping below |
| `manifest` | Export time/version/provider, scope, counts, included features, omissions, warnings and external references |
| `stores` | All 13 canonical keys, each an array of `{ "id": "RECORD_ID", "data": { ... } }`; empty stores remain present |
| `integrity` | `{ "algorithm": "sha256", "digest": "64 lowercase hex characters" }` |

| Canonical store | Taxonomy category |
|---|---|
| `personas`, `entity_memory`, `knowledge_base` | `long_term.semantic` |
| `toolbox`, `workflow_memory`, `skillbox` | `long_term.procedural` |
| `conversation_memory`, `summaries`, `tool_log` | `long_term.episodic` |
| `short_term_memory`, `semantic_cache` | `short_term` |
| `shared_memory` | `coordination` |
| `agents` | `agent_configuration` |

Integrity covers canonical UTF-8 JSON of all fields except `integrity`: sorted
object keys, compact separators, literal Unicode, and no NaN/Infinity. It detects
corruption, not authenticity. Validation rejects duplicate IDs, identity/count
mismatches, wrong taxonomy/checksum, and over-limit files (50 MiB / 100,000
records by default). SDK transformations can intentionally recompute integrity
with `memorizz.memory_archive.seal_archive()`.

The format reference defines the shared
[canonical JSON and numeric representation](memory-archive-format.md#integrity-and-validation)
used by Python and browser imports.

## Boundaries and provenance

Credential-shaped fields and runtime objects are omitted. Ordinary memory text
and captured prompts are included; archives are not encrypted. JSON tool schemas
and import references remain data; restore runs no agent or tool. Imported agent
automations start disabled. Reconnect credentials/providers before using them.

History preserves recorded dates, writers, sources, links, hashes and field names,
not past/deleted values. Backend-local history heads are excluded, so later writes
start a fresh boundary. Projections, redaction and remapped identities can change
version hashes; the timeline must not claim restored current content matches an
older version when it does not. Deleted/unselected references remain explicit
and appear in the manifest. Missing delegate configurations prevent restore
unless they already exist at the destination.

External attachments, model weights, credentials and harness SQLite run records
are outside the memory archive. Export does not lock a live store, so concurrent
writes can produce a mixed-time view. Restore preflights before writing but is
not a transaction across stores: failures stop and report committed progress.
Inspect errors and select a conflict policy before retrying.

## Runnable example

`examples/memory_transfer/demo.py` creates a coordinator and delegate using every
memory type, records a corrected fact, exports/restores the graph and verifies
relationships and history. It can also restore to a real Oracle AI Database
without an LLM call:

```bash
python examples/memory_transfer/demo.py --root ./memory-transfer-demo
python examples/memory_transfer/demo.py --root ./memory-transfer-demo \
  --oracle-config /path/to/private-oracle-connection.json
```

Its `evidence.json` identifies source/restored agents, namespaces, archive path,
counts and ID maps. Connect the portal to that filesystem store or Oracle schema,
then open the restored agent's Playground → Memory.
