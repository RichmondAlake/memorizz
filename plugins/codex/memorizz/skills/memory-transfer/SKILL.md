---
name: memory-transfer
description: Export or import a MemoRizz project or agent memory archive, back up memory, move memory between projects/providers, or clone a memory graph with its taxonomy and relationships.
---

# Export and import memory

MemoRizz archives use `format: memorizz.memory`, `version: 1`, and the suffix
`.memorizz.json`. They contain the 13 canonical taxonomy stores and a SHA-256
integrity check. Records retain their identity, scope, timestamps and explicit
relationships. Available evolution/context records can travel with them.

## Project archives

Use the project's memory ID from the session context (or `memorizz plugin
memory-id`). For local file operations use:

```sh
memorizz plugin export-memory ./project-memory.memorizz.json
memorizz plugin import-memory ./project-memory.memorizz.json
memorizz plugin import-memory ./project-memory.memorizz.json --apply
```

The import command previews by default. It maps a single project namespace to
the current project. `--preserve-namespace` retains the source project instead.
`--new-ids` makes a separate copy and rewrites internal links. Existing records
stop the restore; `--conflict skip` retains them, and `--conflict replace`
replaces them when that behavior is authorized by the user. Do not add
`--overwrite` to export without authorization to replace that file.

If the user asks to import, validate/preview the file and apply the authorized
restore when the preview succeeds. Do not request another approval for a
restore they already requested. Explain conflicts and only apply the selected
conflict policy. Report actual committed/skipped counts and any partial errors.

## MCP / hosted memory

`memorizz_export_memories(memory_id=PROJECT_ID)` returns an inline archive.
`memorizz_import_memories(archive=ARCHIVE, dry_run=true)` previews it; set
`dry_run=false` to apply. Tools enforce the caller's tenant and server write
policy. Remote transports exclude global agent/tool/shared configuration.
The plugin CLI also supports its configured hosted server. Inline archives
must fit the server request-size limit (4 MiB by default); use the local CLI
file interface for larger archives. Do not print the full archive into chat.

## Agents and whole stores

```sh
memorizz memory export ./agent.memorizz.json --agent-id AGENT_ID
memorizz memory import ./agent.memorizz.json --new-ids
memorizz memory import ./agent.memorizz.json --new-ids --apply
```

Agent exports include configured delegates and their separate namespaces by
default. `--no-delegates`, `--no-history`, `--no-context`, and `--types` narrow
an export. The UI has **Memory types → Export & import**, also linked from
Playground and Memory evolution. The SDK uses `MemoryArchive(provider)`.

## Limits

Archives omit credential-shaped fields, runtime objects and embeddings;
ordinary memory text and captured prompts are included. Imported agent
automations start disabled; no agent/tool is executed during restore. Semantic
search requires regenerating embeddings (`memorizz memory import --reembed`,
which may call the configured embedding service). Notion projections remain
pending until explicitly reindexed. Historical/deleted values cannot be
reconstructed from metadata-only history. External attachments and harness
SQLite runs are not included. Writes can overlap export; restore is not atomic.
See https://richmondalake.github.io/memorizz/guides/memory-transfer/.
