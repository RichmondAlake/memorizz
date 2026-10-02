# Expose MemoRizz as an MCP server

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

MemoRizz includes a first-party MCP server built on the official Python SDK.
Any standards-compliant MCP host can use MemoRizz memory, agents, and
conversations through local `stdio` or remote Streamable HTTP.

Install the server runtime with `pip install "memorizz[mcp]"`. The base package
keeps these protocol dependencies optional for applications that do not use
MCP.

## Local setup (recommended for desktop MCP hosts)

Install MemoRizz in the environment that owns the memory store, then point the
MCP host at the CLI:

```json
{
  "mcpServers": {
    "memorizz": {
      "command": "memorizz",
      "args": ["mcp", "serve"]
    }
  }
}
```

The local server uses `~/.memorizz` and the configured memory provider. `stdio`
is single-user, does not require bearer authentication, and permits memory
writes and agent execution by default. An MCP host should still request user
approval before it calls a mutating tool.

Local MCP hosts can manage the safe agent lifecycle with create, read, update,
delete, and execute tools. MemoRizz's MCP client turns mutating calls into exact,
durable host proposals before dispatch; other MCP hosts must enforce their own
human approval policy. Agent deletion creates an additional server-owned,
single-use proposal for the operator commands shown below. There is no
model-visible `approved` or `confirm` argument. Newly created or updated agents
are immediately visible in the same server process.

The creation schema accepts secret-free `llm_provider` and `llm_model` names.
Credentials remain server-side. If an explicitly selected provider cannot be
initialized, creation fails without persisting a partially configured agent.

Run the same server directly when troubleshooting:

```bash
memorizz mcp serve --transport stdio
# Equivalent:
python -m memorizz.mcp_server
```

Protocol messages use stdout. Startup diagnostics and logs use stderr, so MCP
framing is not corrupted.

## Remote Streamable HTTP

Remote mode is secure-by-default:

- bearer authentication is mandatory unless `--allow-anonymous` is explicit;
- direct writes and agent execution are disabled by default;
- remote agents must be explicitly allowlisted;
- a non-loopback public URL must use HTTPS; and
- each bearer principal gets an isolated `user_id` memory and conversation
  scope.

Generate a long random bearer token with your secret manager or, for a local
test:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Configure principal-to-token grants as JSON. Tokens never appear in the public
server metadata:

```bash
export MEMORIZZ_MCP_SERVER_API_KEYS='{
  "alice": {
    "token": "replace-with-a-long-random-token",
    "scopes": ["memorizz:read", "memorizz:write"]
  },
  "automation-reader": {
    "token": "replace-with-another-long-token",
    "scopes": ["memorizz:read"]
  }
}'

memorizz mcp serve \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8766 \
  --public-url http://127.0.0.1:8766 \
  --allow-writes
```

The MCP endpoint is `http://127.0.0.1:8766/mcp`. Configure the remote MCP
client with `Authorization: Bearer <token>`. MemoRizz's own client can connect
to it as follows:

```bash
memorizz mcp add company-memory \
  --transport streamable_http \
  --url https://memory.example.com/mcp \
  --auth bearer \
  --token "$MEMORIZZ_REMOTE_TOKEN"
```

### Exposing agent execution

Agent turns always persist conversation memory. Consequently, remote execution
requires all three controls:

1. the token has `memorizz:execute` and `memorizz:write` scopes;
2. the operator enables both writes and execution; and
3. every runnable agent is explicitly named with `--agent-id`.

```bash
memorizz mcp serve \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8766 \
  --public-url https://memory.example.com \
  --allow-writes \
  --allow-agent-execution \
  --agent-id 6e643c4b-... \
  --agent-id 9f72500c-...
```

Find agent IDs in the local UI or with `/agents` in the MemoRizz REPL. Put a
TLS reverse proxy or ingress in front of the bound port; `--public-url` is the
external origin, not the internal bind address.

!!! warning "Agent tools are operator authority"
    An exposed agent may have filesystem, internet, database, or outbound MCP
    tools configured with deployment-wide credentials. Tenant isolation covers
    MemoRizz memory and conversations; it cannot make an arbitrary agent tool
    tenant-safe. Enable remote execution only for agents whose tool policy and
    credentials are appropriate for every granted principal.

## Available MCP capabilities

| Capability | MCP name | Required scope |
|---|---|---|
| Inspect server policy | `memorizz_server_info` | `memorizz:read` |
| List/read exposed agents | `memorizz_list_agents`, `memorizz_get_agent` | `memorizz:read` |
| Inspect capabilities, cache, learning, and scoped observability | `memorizz_inspect_agent` | `memorizz:read` |
| Preview bounded personalization and content-safe memory evidence | `memorizz_preview_personalization` | `memorizz:read` |
| Create a persisted local agent | `memorizz_create_agent` | `memorizz:write`; local `stdio` only; durable approval required |
| Update safe local agent configuration | `memorizz_update_agent` | `memorizz:write`; local `stdio` only; durable approval required |
| Propose local agent deletion | `memorizz_delete_agent` | `memorizz:write`; local `stdio` only; durable operator approval required |
| Execute an exposed agent | `memorizz_execute_agent` | `memorizz:execute` + `memorizz:write` |
| Compile scoped continual-learning memory | `memorizz_compile_memory` | `memorizz:write` |
| Compact a scoped conversation | `memorizz_compact_conversation` | `memorizz:execute` + `memorizz:write` |
| Inspect installed harnesses | `memorizz_list_harnesses` | `memorizz:read` |
| Start a bounded harness run | `memorizz_start_harness_run` | `memorizz:execute`; writes also require `memorizz:write` |
| Read/list harness runs | `memorizz_get_harness_run`, `memorizz_list_harness_runs` | `memorizz:read` |
| Read normalized harness events | `memorizz_get_harness_events` | `memorizz:read` |
| Cancel or retry an owned harness run | `memorizz_cancel_harness_run`, `memorizz_retry_harness_run` | `memorizz:execute`; retrying an edit run also requires `memorizz:write` |
| Start a staged plan or a comparison | `memorizz_start_harness_plan`, `memorizz_start_harness_comparison` | `memorizz:execute`; an edit stage also requires `memorizz:write` |
| Read/list/cancel owned plans and comparisons | `memorizz_get_harness_workflow`, `memorizz_list_harness_workflows`, `memorizz_cancel_harness_workflow` | `memorizz:read`; cancel requires `memorizz:execute` |
| Run an owned plan or comparison again | `memorizz_rerun_harness_workflow` | `memorizz:execute`; one with an edit stage also requires `memorizz:write` |
| Read or continue an owned harness conversation | `memorizz_get_harness_conversation`, `memorizz_continue_harness_conversation` | `memorizz:read`; continuing requires `memorizz:execute` (and `memorizz:write` for edits) |
| Delete owned finished runs, plans and comparisons | `memorizz_delete_harness_runs`, `memorizz_delete_harness_workflow` | `memorizz:execute` + `memorizz:write` |
| List an exposed agent's MCP servers and tools | `memorizz_list_connected_tools` | `memorizz:read` |
| Call a read-only tool on the agent's MCP server | `memorizz_read_connected_tool` | `memorizz:read` |
| Call any tool on the agent's MCP server | `memorizz_call_connected_tool` | `memorizz:read`; tools that change data also need `memorizz:write` and return a durable proposal |
| Run an approved connected-tool call | `memorizz_resume_connected_tool_call` | `memorizz:write` |
| List/search/read memories | `memorizz_list_memories`, `memorizz_search_memories`, `memorizz_get_memory` | `memorizz:read` |
| Report backend, policy, embedding readiness and record counts | `memorizz_memory_status` | `memorizz:read` |
| Store memory | `memorizz_store_memory` | `memorizz:write` |
| Replace a memory, keeping the old one as superseded | `memorizz_update_memory` | `memorizz:write` |
| Ingest local files and folders into the knowledge base | `memorizz_ingest` | `memorizz:write`; only inside the ingest folders |
| Find entities by name or query | `memorizz_lookup_entities` | `memorizz:read` |
| Create an entity or merge facts into it | `memorizz_upsert_entity` | `memorizz:write` |
| Propose deletion of owned memory | `memorizz_forget_memory` | `memorizz:write`; returns a durable proposal |
| Save one turn of a session (secrets removed) | `memorizz_record_turn` | `memorizz:write` |
| Summarize a session's turns with the server's model | `memorizz_summarize_session` | `memorizz:write` |
| List/read conversations | `memorizz_list_conversations`, `memorizz_get_conversation` | `memorizz:read` |

`memorizz_preview_personalization` never performs cross-thread recall unless
`include_conversation_recall=true`. The authenticated principal is always used
as `user_id`, one `memory_id` must be explicit (or the agent must expose exactly
one), and thresholds/source limits are bounded server-side. Set
`include_content=false` to return only counts, scores, attribute/preference
names, and hashed source references for an observability-only client.

Harness capability and startup responses use the same secret-free
`error_code`, message, and remediation contract as the SDK, CLI, and UI. When
authentication is absent or expired, `memorizz_start_harness_run` returns
`ok=false`, `status="failed"`, the durable failed run, and
`error.code="authentication_required"`; no API-key value is returned.

`memorizz_execute_agent` also returns a `tool_outcomes` array for the completed
turn. Each item includes the tool name, terminal status, success flag, duration,
and any content-free provider/fallback reason fields. MCP hosts can therefore
distinguish clean completion from `empty`, `degraded`, `fallback`,
`provider_error`, and `error` without parsing the assistant response.

The server also exposes:

- `memorizz://server`;
- `memorizz://agents/{agent_id}`;
- `memorizz://agents/{agent_id}/conversations`;
- `memorizz://harness-runs/{run_id}`; and
- the `memorizz_memory_assistant` prompt.

Direct memory writes are limited to `knowledge_base` and `short_term_memory`
(plus entities through `memorizz_upsert_entity`).

### Searching, updating and ingesting memory

**Search with or without embeddings.** `memorizz_search_memories` returns
`search_mode`. It is `semantic` when an embedding model works. It is `keyword`
when none does, or when semantic search fails: records then rank by how many
query words they share, case-insensitively, with the same `memory_id`, tenant
and superseded filters. Keyword results carry a `note` saying how to turn
semantic search on, for example
`memorizz config set MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER ollama` and
`memorizz config set MEMORIZZ_DEFAULT_EMBEDDING_MODEL nomic-embed-text`. It is
`hybrid` when keyword matches fill out a short semantic list, which happens
when some records were stored before embeddings were on. `memorizz_store_memory`
embeds what it stores whenever an embedding model works.

**Update instead of delete.** `memorizz_update_memory(record_id, content,
memory_type="knowledge_base", reason=None, duplicate_of=None)` stores a new
record and keeps the old one:

| Record | Fields set |
|---|---|
| New | `supersedes` (the old record ID), `supersede_reason` |
| Old | `status: "superseded"`, `superseded_by` (the new record ID), `superseded_at` |

`memorizz_list_memories` and `memorizz_search_memories` hide superseded records
unless you pass `include_superseded=true`. Nothing is deleted, so you can always
see the history. Updating a superseded record fails with `memory_superseded` and
names the newer record. A backend that can't store these fields (Oracle's
fixed-column knowledge-base table) fails with `supersede_unsupported` and changes
nothing. Entity consolidation uses the same idea, as `metadata.superseded_by`,
and the filters treat both forms the same.

To retire a duplicate, pass `duplicate_of` (the memory to keep, in the same
`memory_id`) instead of `content`: the duplicate is marked superseded by that
memory and nothing new is stored.

**Session memory.** `memorizz_record_turn(memory_id, thread_id, user_message,
assistant_message, agent_id=None)` saves one turn as two conversation-memory
records for the calling principal, with secrets removed and embeddings when a
model works. `memorizz_summarize_session(memory_id, thread_id, agent_id=None)`
summarizes that principal's turns in the thread with the server's default
model (`MEMORIZZ_DEFAULT_LLM_PROVIDER` / `_MODEL`) into the summaries store;
without a model it returns `ok=false` and `reason="no_model"`. The Codex and
Claude Code plugins call these in remote mode
([MemoRizz in Codex and Claude Code](coding-agent-plugins.md)).

**Ingest files.** `memorizz_ingest(paths, memory_id, recursive=True,
include=None, chunk_size=None, chunk_overlap=None)` chunks text, Markdown, code, config,
HTML, JSON, CSV and PDF (with `pypdf` installed) into the knowledge base under
`memory_id`. Each chunk records `source_path`, `source_name`, `source_sha256`,
`chunk_index` and `chunk_count`. The result counts the files ingested, the
chunks stored and embedded, and the files skipped, each with a reason.

- **Where it reads.** Absolute paths inside the ingest folders only:
  `MEMORIZZ_MCP_SERVER_INGEST_ROOTS` or `--ingest-root`. By default that is
  your home folder for a local stdio server, and nothing for a remote one, so a
  remote server can't ingest until the operator names folders. A path outside
  them is refused (`path_not_allowed`), and so is a symlink that leads outside
  them.
- **What it skips.**
  - Hidden files and folders.
  - `.git`, `node_modules`, `.venv` and similar folders.
  - MemoRizz's own data folder.
  - Unsupported file types.
  - Anything that looks like a secret: `.env*`, `*.pem`, `*.key`, `id_rsa*` and
    other SSH keys, `*.p12`, `*.pfx`, keystores, `credentials*` and `secrets*`.
- **Limits.** At most 500 files and 20 MB per call. A larger batch is refused
  (`ingest_too_large`) before anything is stored; narrow it with `include`
  patterns such as `["*.md"]`.
- **Ingesting again.** Unchanged files are skipped. When a file has changed, its
  new chunks are stored and the old ones are marked superseded.
- **Without embeddings.** Chunks are stored without embeddings, and keyword
  search still finds them.

**Entities.** `memorizz_upsert_entity(name, entity_type=None, attributes=None,
relations=None, memory_id=None)` creates an entity or merges facts into the one
with that name. `attributes` maps fact names to values, such as
`{"role": "tech lead"}`. Each relation is `{"target", "relation_type"}`, where
the target is another entity's name or ID; a target that doesn't exist yet is
created by name. `memorizz_lookup_entities(query=None, name=None,
memory_id=None, limit=10)` finds entities by case-insensitive name, by query
(semantic, or keyword without embeddings), or by both.

**Status.** `memorizz_memory_status(memory_id=None)` reports:

- the version and transport;
- the policy flags (writes, agent execution, harness execution, trace queries);
- the backend and where it stores data (a database's connection string is never
  shown);
- the writable memory types;
- the embedding provider and model, and whether a short embedding call works
  (checked with a timeout; it never fails the tool);
- active and superseded record counts per memory type; and
- the ingest folders and limits.
Remote callers cannot access global configuration stores such as
personas, toolboxes, agents, shared memory, or skill definitions through the
generic memory tools.

Remote HTTP agent creation, update, and deletion are deliberately unavailable.
Persisted agents do not yet carry a durable tenant-owner field, and a
process-wide exposed-agent list is not a safe substitute for ownership. Manage
remote agents through the trusted host SDK, CLI, or authenticated local UI,
then explicitly expose their IDs. Remote callers can inspect or execute only
allowlisted agents, and all memory, conversation, observability, learning, and
compaction calls use the authenticated principal as `user_id` server-side.

`memorizz_store_memory` returns the physical `record_id` assigned by the
provider. Pass that exact value to `memorizz_get_memory`; `memory_id` remains
the logical conversation/application scope and is not interchangeable with the
record ID. Oracle RAW identifiers are normalized to canonical UUID strings, so
a write → read round trip has the same contract as filesystem and MongoDB.
The authenticated principal supplies tenant identity server-side—callers
cannot escape their scope by placing another `user_id` in tool arguments.

Deletion and harness execution never accept a model-visible `confirm` or
`approved` argument. The server operator must decide and consume exact
memory-deletion, agent-deletion, or harness-envelope proposals out of band:

```bash
memorizz mcp server-approvals --status pending
memorizz mcp server-approve PROPOSAL_ID --approver operator@example.com
memorizz mcp server-resume PROPOSAL_ID
# Or: server-reject / server-cancel with --approver
```

## Form-factor parity and trust boundary

The 44-tool MCP surface covers the common operational capabilities available
through the SDK, CLI, and local UI: agent lifecycle and execution, scoped memory
and conversations, capability/cache/learning/observability inspection,
continual-learning compilation, conversation compaction, and governed harness
execution. Every tool schema rejects undeclared arguments
(`additionalProperties: false`). The server also publishes four resources and
one stable assistant prompt.

Some host-administration functions intentionally remain outside model-visible
MCP tools:

- credentials and API-key management;
- ingestion outside the operator's ingest folders;
- approval, rejection, and resumption decisions; and
- configuration of outbound MCP connections.

Those operations remain in trusted SDK, CLI, or authenticated local-UI code.
An MCP caller may cancel its own tenant-scoped harness run, but cannot approve
or resume the proposed execution envelope. This is the parity boundary: an MCP
client can use a configured MemoRizz agent and its memory-first runtime without
receiving the host's secrets or approval authority.

## Headless operation

The SDK, CLI, and stdio/HTTP MCP server do not require a display server and do
not import the optional UI application. A minimal headless installation is:

```bash
pip install "memorizz[mcp]"
unset DISPLAY WAYLAND_DISPLAY
memorizz agents create --name "Headless Agent" --no-llm --json
memorizz mcp serve --transport stdio
```

Add an LLM provider configuration before executing the no-LLM agent. The local
UI remains an optional, separately launched form factor.

## Configuration reference

Every CLI option has a deployment-friendly environment equivalent:

| Environment variable | Purpose |
|---|---|
| `MEMORIZZ_MCP_SERVER_TRANSPORT` | `stdio` or `streamable-http` |
| `MEMORIZZ_MCP_SERVER_HOST` / `MEMORIZZ_MCP_SERVER_PORT` | HTTP bind address |
| `MEMORIZZ_MCP_SERVER_PATH` | Endpoint path; default `/mcp` |
| `MEMORIZZ_MCP_SERVER_PUBLIC_URL` | External HTTP loopback or HTTPS origin |
| `MEMORIZZ_MCP_SERVER_API_KEYS` | Principal-to-token grant JSON |
| `MEMORIZZ_MCP_SERVER_ALLOW_WRITES` | Enable direct writes (`true`/`false`) |
| `MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION` | Enable agent turns |
| `MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION` | Enable governed external harness runs |
| `MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS` | Comma-separated allowed workspace roots |
| `MEMORIZZ_MCP_SERVER_INGEST_ROOTS` | Comma-separated folders `memorizz_ingest` may read (`--ingest-root`); default: home folder for stdio, none for HTTP |
| `MEMORIZZ_MCP_SERVER_AGENT_IDS` | Comma-separated remote agent allowlist |
| `MEMORIZZ_MCP_SERVER_ALLOW_ANONYMOUS` | Explicitly disable HTTP auth |
| `MEMORIZZ_MCP_SERVER_STATELESS_HTTP` | Stateless Streamable HTTP sessions |

Static bearer grants are suitable for a single service or a controlled
deployment. Rotate them through the platform secret manager and use separate
principals per client. Anonymous HTTP callers all share the legacy `user_id`
scope and should only be used for local development.

For production, keep `MEMORIZZ_HOME` and the provider data on durable storage,
run a single process per filesystem store, use MongoDB or Oracle for horizontally
scaled deployments, terminate TLS at a trusted proxy, apply network controls,
and monitor server/provider logs. Do not place bearer tokens in command-line
arguments or source control.
