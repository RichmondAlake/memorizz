# Expose MemoRizz as an MCP server

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
| Execute an exposed agent | `memorizz_execute_agent` | `memorizz:execute` + `memorizz:write` |
| List/search/read memories | `memorizz_list_memories`, `memorizz_search_memories`, `memorizz_get_memory` | `memorizz:read` |
| Store memory | `memorizz_store_memory` | `memorizz:write` |
| Propose deletion of owned memory | `memorizz_forget_memory` | `memorizz:write`; returns a durable proposal |
| List/read conversations | `memorizz_list_conversations`, `memorizz_get_conversation` | `memorizz:read` |

The server also exposes:

- `memorizz://server`;
- `memorizz://agents/{agent_id}`;
- `memorizz://agents/{agent_id}/conversations`; and
- the `memorizz_memory_assistant` prompt.

Direct memory writes are limited to `knowledge_base` and `short_term_memory`.
Remote callers cannot access global configuration stores such as
personas, toolboxes, agents, shared memory, or skill definitions through the
generic memory tools.

`memorizz_store_memory` returns the physical `record_id` assigned by the
provider. Pass that exact value to `memorizz_get_memory`; `memory_id` remains
the logical conversation/application scope and is not interchangeable with the
record ID. Oracle RAW identifiers are normalized to canonical UUID strings, so
a write → read round trip has the same contract as filesystem and MongoDB.
The authenticated principal supplies tenant identity server-side—callers
cannot escape their scope by placing another `user_id` in tool arguments.

Deletion never accepts a model-visible `confirm` argument. The server operator
must decide and consume the exact proposal out of band:

```bash
memorizz mcp server-approvals --status pending
memorizz mcp server-approve PROPOSAL_ID --approver operator@example.com
memorizz mcp server-resume PROPOSAL_ID
# Or: server-reject / server-cancel with --approver
```

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
