# Notion: document memory with a separate vector provider

`NotionProvider` makes Notion the source of truth for memory documents. Pass
`semantic_provider=` to select where embeddings and scoped references live.
The secondary provider is **vector-only**: it does not receive a second copy
of conversation text, agent instructions, or knowledge documents.

```text
MemAgent / SDK / CLI / Memorizz MCP
                  |
            NotionProvider
             /           \
Notion data source     semantic_provider
documents + views      vectors + IDs + scope + hashes
                  |
          local SQLite repair journal
          IDs, hashes and pending operations only
```

This is intended for a human-editable, modest-volume memory library. It is not
a low-latency replacement for an operational database or a distributed agent
coordination store. Existing providers and streaming defaults are unchanged.

## Create the Notion area

Install `pip install "memorizz[notion]"`. The adapter uses the existing Requests
dependency directly; neither an LLM nor Notion MCP is in the storage path.
Add `[mongodb]`, `[oracle]`, `[ui]`, or `[mcp]` for those optional delivery paths.

Create a Notion connection with read, insert and update content capabilities.
Share a dedicated parent page with it. Put `NOTION_TOKEN` in your environment
or your uncommitted `.env`; never pass credentials in source code or chat.
The recommended interactive setup handles credentials, IDs, vector settings
and saving without manually editing `.env`:

```bash
memorizz notion connect
# or: memorizz memory configure notion
# or inside the REPL: /memory-provider notion
```

Select **existing** for a Memorizz library and paste its database URL or
data-source ID. Setup resolves database IDs to their data source, asks which
source to use when there are several, and checks managed column names/types
without changing the schema. For an existing library whose managed columns
were renamed, restore the managed names before using this setup flow.
Select **create** and paste a shared parent-page URL/ID to create the layout
below after explicit confirmation. A successful create saves a non-secret
`notion-setup-*.json` manifest beside the selected env file; partial failures
print known resource IDs for recovery. Unknown create outcomes require manual
inspection before retrying. Use `--project` to save to the current directory's
`.env`; otherwise setup shows the shared Memorizz save target.

Setup makes no embedding calls/downloads and does not validate the external
vector database. Read access/schema checks do not prove write permissions for
an existing library. Restart before `memorizz notion status` / `sync`; sync can
incur embedding charges. Defaults do not reconnect a running agent or migrate
memory from another provider. [Configuration precedence and editing](../getting-started/cli.md#configuration)
also apply to npm, pipx, uv, pip and Homebrew installations.

For scripts that already provide credentials, the explicit provisioning command
remains available (it prints a manifest but does **not** save environment settings):

```bash
memorizz notion init --parent-page-id YOUR_SHARED_PAGE_ID
```

The returned manifest contains the IDs to retain. Each invocation creates a
**new** area; it does not migrate or adopt unrelated existing databases:

```text
Your shared page
└── Memorizz
    ├── Memory
    │   └── Memorizz memories: one data source, a table view per memory type
    └── Agent workspace
        ├── Agents
        ├── Conversations
        ├── Traces
        └── Tool activity
```

These are separate pages in the same Notion workspace. The four interface
views reference the same data, rather than copying it. `--no-views` creates
the database without the additional views. On partial provisioning failure,
the exception's `notion_workspace` attribute preserves known resource IDs;
inspect those resources before retrying. An uncertain creation can require
manual investigation in Notion. No partial area is automatically deleted.

All 13 `MemoryType` values round-trip: agents, personas, toolbox, entities,
short-term memory, knowledge, conversations, workflows, skills, shared memory,
summaries, semantic cache and tool logs. Trace bundles remain structured JSON
in their Content property. The linked views are inspection/editing surfaces,
not a newly implemented Notion chat client or token-by-token Notion renderer.

## Python constructor

For this local example, run Ollama and pull `nomic-embed-text` first. The agent's
generation model is configured independently from this embedding model.

```python
import os

from memorizz import MemAgent
from memorizz.enums import MemoryType
from memorizz.memory_provider import (
    FileSystemConfig, FileSystemProvider, NotionConfig, NotionProvider,
)

vectors = FileSystemProvider(FileSystemConfig(
    root_path="./notion-vectors",  # Use a dedicated directory.
    use_faiss=False,
    embedding_provider="ollama",
    embedding_config={"model": "nomic-embed-text"},
))
memory = NotionProvider(
    NotionConfig(
        data_source_id=os.environ["MEMORIZZ_NOTION_DATA_SOURCE_ID"],
        # token defaults to NOTION_TOKEN.
        state_path="./notion-repair.sqlite3",
    ),
    semantic_provider=vectors,
)

record_id = memory.store(
    {"content": "Alice prefers espresso.", "user_id": "alice"},
    MemoryType.KNOWLEDGE_BASE,
)
matches = memory.retrieve_by_query(
    "coffee preference", MemoryType.KNOWLEDGE_BASE, user_id="alice", limit=5,
)

agent = MemAgent(memory_provider=memory, automations_enabled=False)
# Configure the LLM and attach knowledge/memory scopes as in the SDK guide.
# agent.run_stream(...) retains normal streaming behavior.

memory.close()
vectors.close()  # Injected providers remain caller-owned.
```

The constructor validates an already provisioned data source; it does not
create Notion pages/databases. The journal defaults to
`$MEMORIZZ_HOME/notion/<data-source-id>.sqlite3` (`~/.memorizz` by default).
Keep it on durable local storage and back it up. It is not a second document
database. Losing it loses pending-write evidence, even though documents remain
in Notion. Reconcile uncertain writes before replacing a lost journal, then
rebuild known content with `sync(force=True)`.

Use **one active writer process per data source**, reusing its provider/client.
Threads on that instance serialize writes; stable IDs and durable create
intents prevent blind retries. Separate processes/journals do not provide
Notion transactions, unique constraints or compare-and-swap. Do not run sync
maintenance concurrently with a separate writing application process.

## Select a different semantic provider

| `semantic_provider` | Vector execution | Requirements |
| --- | --- | --- |
| `FileSystemProvider(...)` | Scoped exact cosine; warmed in-process vector cache | Dedicated directory, configured embedding model; single writer |
| `MongoDBProvider(...)` | Native Atlas `$vectorSearch` | Vector-capable deployment, queryable `memorizz_semantic_index`, appropriate index privileges |
| `OracleProvider(...)` | Native `VECTOR_DISTANCE(..., COSINE)` | Compatible Oracle vector schema/dimensions and credentials |
| `None` | No semantic search | Document CRUD and views still work; semantic queries raise explicitly |

Pass the constructed instance directly, for example:

```python
from memorizz.memory_provider import MongoDBConfig, MongoDBProvider

vectors = MongoDBProvider(MongoDBConfig(
    uri=os.environ["MONGODB_URI"], db_name="memorizz_notion_vectors",
    lazy_vector_indexes=True,
    embedding_provider="ollama",
    embedding_config={"model": "nomic-embed-text"},
))
memory = NotionProvider(config, semantic_provider=vectors)  # config: NotionConfig
```

Use dedicated vector storage. MongoDB Community without vector search is not
a supported semantic substitute; the adapter does not silently return recent
documents. The composed vector API reports cosine scores consistently across
these three backends. Oracle can use exact vector search without a vector
index, but larger datasets need database-specific indexing and tuning.
An Oracle provider's constructor may initialize its own schema independently
of the Notion constructor; use an already prepared database and limited grants
when deploying a read-only application.

The selected vector provider owns embedding generation. Incoming legacy
`embedding` fields are discarded on document writes so unrelated global models
cannot contaminate this index. Raw vector queries are an advanced API: the
caller must use the same model/dimensions. Knowledge ingestion, entity writes,
summary persistence and semantic caching defer generation to the provider;
conversation writes do not schedule a second global-model embedding backfill.
Legacy DTOs that eagerly embed during their own construction remain independent
of storage. Use plain document dictionaries to avoid that eager work.

Changing the backend/model, enabling semantics after document-only operation,
or changing `semantic_memory_types` marks tracked records for repair. Stop the
old writer, construct the new configuration, then run `repair_index()` followed
by `sync(force=True)` to discover any untracked pages. Dimension changes may
also require a new vector directory/database/index. Old external indexes are
not deleted automatically. Read-only construction does not activate a different
index configuration.

Custom providers opt in with `MemoryProviderCapabilities(vector_store=True)`
and implement `embed_text`, `semantic_identity`, `upsert_vector`, `query_vectors`
and `delete_vector`. The identity must change with backend location or embedding
model, even when dimensions stay the same. Queries must filter scopes **before**
top-k, return cosine scores plus source references, and raise on backend failure.

## CLI, local UI and MCP

```dotenv
MEMORIZZ_BACKEND=notion
NOTION_TOKEN=<set-through-your-secret-manager>
MEMORIZZ_NOTION_DATA_SOURCE_ID=<data_source_id-from-init>
MEMORIZZ_NOTION_SEMANTIC_BACKEND=filesystem
MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=ollama
MEMORIZZ_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
```

The shared factory also accepts `mongodb`, `oracle` and `none`.
Optional settings are `MEMORIZZ_NOTION_VECTOR_PATH`,
`MEMORIZZ_NOTION_STATE_PATH`, `MEMORIZZ_NOTION_MONGODB_URI` (otherwise
`MONGODB_URI`) and `MEMORIZZ_NOTION_VECTOR_DB` (default
`memorizz_notion_vectors`). Oracle uses `ORACLE_USER`, `ORACLE_PASSWORD`,
`ORACLE_DSN` and the normal Oracle configuration. Environment-created secondary
providers are owned and closed by the Notion provider.

The local UI connection screen offers **Notion + separate semantic provider**.
Tokens are not prefilled into HTML or saved into browser connection settings.
Its dashboard explicitly labels local journal counts; it does not scan the
entire Notion library for counts. Memory previews are bounded to 100 records.
Observability and usage reads use bounded native filters and continuation
cursors; a bounded window is not an account-wide billing total.

The existing [Memorizz MCP server](../guides/mcp-server.md) uses the same backend
selection. It can expose Notion-backed memory tools to an MCP host. Connecting
Notion Agent to that server is a separate host-side setup; this adapter neither
enables Notion AI nor deploys a public server. Plan/admin requirements and
credential-sharing behavior must be checked in
[Notion's MCP connection documentation](https://www.notion.com/help/connect-mcp-servers-to-your-notion-agent).

## Human edits, synchronization and repair

Edit **Content** in Notion (valid JSON when the original value was structured),
or **Name** when the original record had a string name. Matching text/query
aliases follow those edits. Page-body blocks are not the document store.
Do not manually change Memorizz ID, Memory type, scope columns, Timestamp, Event time, or
Memorizz metadata. Property names may be renamed because stable property IDs
are retained in the journal; changing their types or deleting them is invalid.

Notion's date column can discard seconds and subseconds. `Event time` is a
minute-level display/query bucket; `Timestamp` is canonical UTC text at full
precision, while returned records preserve their original timestamp value.
Observability uses the coarse date filter to bound reads, then applies exact
time bounds to the protected record timestamp. Ordering uses the precise text
column, and continuation cursors are bound to the exact requested time window.

```bash
memorizz notion status
memorizz notion sync --memory-type knowledge_base
memorizz notion repair
memorizz notion sync --force
```

ID/filter reads fetch current Notion content. Semantic search fetches current
pages for candidate vectors, rechecks tenant scope and compares fingerprints.
Changed, moved, archived or inaccessible records are never served from cached
vector metadata. Human edits temporarily reduce semantic recall until sync;
they are not silently treated as up-to-date indexed content. Semantic-cache
exact hits also honor this live-read policy. Unchanged sync rows do not trigger
another embedding call.

`sync_page(page_id)` refreshes one edited page. `process_webhook(raw_body,
signature=..., verification_token=...)` verifies Notion's HMAC signature,
fetches the latest page and deduplicates completed events. Your host owns the
HTTPS endpoint, initial verification handshake and durable retry queue. Webhook
delivery is asynchronous; keep periodic reconciliation for missed events.

`NotionIndexingError` means the document was saved but its vector update needs
repair; the exception carries `record_id` and `page_id`. `NotionWriteUncertain`
means a remote write may have committed: preserve/reconcile the stable ID, do
not retry a create with a new random ID. Durable intents survive crashes during
create/update/archive. `repair_index()` never blindly recreates an absent,
uncertain page. After manual investigation, an unresolved create without a
known page can be released with
`abandon_pending_create(id, memory_type, confirm=True)`; this does not cancel a
remote request or delete content. Use a new ID afterward and reconcile any
delayed page via sync.

Deletion archives the exact Notion page and removes only its owned vector.
Archived pages are recoverable through Notion; stores will not silently
resurrect an archived ID. `delete_all` resolves the bounded target set before
starting. Bulk writes/deletes are not transactions and can partially succeed.
`index_status()` reports the **local repair journal**, not proof that every
Notion page has been discovered or is currently accessible.

## Limits and security

- `user_id=None` means the anonymous bucket; omitting it means unscoped access.
  Hosts must supply and authorize user/agent/thread/application scopes. The
  integration token's access is not the individual caller's Notion identity.
- A filtered Notion view is **not** row-level access control. Workspace editors
  with access to the underlying database can inspect its other rows and managed
  metadata. Separate data sources/workspaces where users must not share access.
- Agent connection secrets are stripped before persistence, but arbitrary
  memory text and trace payloads are not a general-purpose secret scrubber.
  Redact sensitive content before storage. Embeddings/references also require
  appropriate access controls even without copied text.
- Notion CRUD uses API version `2026-03-11`, bounded requests, no redirects and
  a per-client default 2.5 requests/second. Rate limiting is not global across
  processes. Retries respect bounded `Retry-After`; long waits are surfaced for
  host retry rather than blocking indefinitely.
- Text properties are split into 2,000-character elements. The adapter rejects
  oversized records (180,000 characters per text field; roughly 440 KB encoded
  properties), rather than silently truncating traces or content. Chunk large
  documents; reduce trace verbosity for this provider.
- Explicit scans stop at configured budgets and reject the 10,000-row service
  boundary as incomplete. Semantic hydration defaults to at most 100 candidates
  (`max_semantic_candidates`, maximum 1,000). Partition large libraries and
  observability time windows. Every semantic candidate can require a Notion GET;
  expect higher latency than a single-database provider.
- Atomic shared-memory delegation, leases, workflow claiming, automatic agent
  cascade deletion and coordination-dependent automation are not implemented
  by this provider. The semantic backend is not secretly used for coordination.
  Use an existing atomic provider as primary for those workloads.

See the official [request limits](https://developers.notion.com/reference/request-limits),
[data-source query limits](https://developers.notion.com/reference/query-a-data-source),
[views API](https://developers.notion.com/guides/data-apis/working-with-views),
[webhooks](https://developers.notion.com/reference/webhooks) and
[workspace block limits](https://developers.notion.com/reference/workspace-block-limits)
before production deployment.

## Verification

Offline tests exercise all memory types, scoped vector contracts, crashes and
uncertain writes, API pagination, model changes, webhooks, CLI/UI/MCP delivery,
semantic-cache revocation and a real MemAgent streaming conversation with usage
and persisted traces. The HTTP end-to-end fixture uses loopback, not Notion:

```bash
pytest tests/unit/test_notion_provider.py tests/unit/test_notion_delivery.py \
  tests/unit/test_vector_provider_composition.py \
  tests/integration/test_notion_provider_e2e.py tests/integration/test_notion_demo.py
```

To verify release artifacts without the test suite's source-path injection,
install the built wheel into a clean target directory and run
`python tests/integration/run_notion_installed.py --installed-root /path/to/target`.
This checks import provenance, CLI/template packaging and both end-to-end flows.

Live tests are **opt-in** and require separate credentials and permission:

```bash
# Creates a unique test area and archives ONLY that created area in cleanup.
MEMORIZZ_TEST_NOTION_LIVE=1 pytest tests/integration/test_notion_live.py -k live_notion
# Tests the vector adapter against a local Oracle instance using synthetic IDs.
MEMORIZZ_TEST_NOTION_ORACLE=1 pytest tests/integration/test_notion_live.py -k oracle
```

The Notion test requires `NOTION_TOKEN` and
`MEMORIZZ_NOTION_TEST_PARENT_PAGE_ID` for a dedicated shared test page.
Mock/loopback success does not certify live Notion permissions, Atlas index
readiness or Oracle schema compatibility. Run these deployment checks against
your configured services before an operational rollout.

### Keep a visible, verified demo

From a source checkout with the package dependencies installed, run:

```bash
PYTHONPATH=src python examples/notion/live_demo.py \
  --parent-page-id YOUR_SHARED_PAGE_ID \
  --output-dir .local/notion-demo
```

Unlike the disposable integration test, this leaves a **new** child demo area
in Notion. It includes all 13 memory types, a readable verification overview,
two streaming MemAgent turns, persisted traces and a direct Notion edit followed
by synchronization. It verifies scoped CRUD, stale-read protection, restart
persistence and the content-free filesystem vector store. Only synthetic test
records are used. The deterministic model, embedding vectors and token counts
are explicitly demonstrations, not a production model or provider bill; no paid
LLM service is called. The Notion views are inspection/editing surfaces, not an
interactive Notion chat application.

Keep `workspace.json`, `repair.sqlite3` and `vectors/` in the chosen local output
directory. The manifest has resource IDs/results, not the connection token.
`--resume` reuses that area's saved manifest instead of provisioning another
root. Interrupted creation with an unknown outcome requires inspecting the
recorded resource IDs before retrying. Switching to a production embedding model
requires configuring that backend and rebuilding its semantic index.
