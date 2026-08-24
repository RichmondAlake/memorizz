# Configuration and Secrets

MemoRizz accepts explicit SDK objects and environment-backed defaults. Keep
credentials outside persisted agent JSON.

## Precedence

For CLI and UI processes, values resolve in this order:

1. explicit SDK/CLI arguments;
2. existing process environment;
3. project-local `./.env`;
4. `$MEMORIZZ_ENV_FILE`, or `$MEMORIZZ_HOME/.env` when no override is set.

Environment files load with `override=False`, so a deployed process variable
cannot be silently replaced by a developer `.env` file. The CLI and UI write
only the canonical MemoRizz environment file; they do not rewrite a project
`.env` unless `MEMORIZZ_ENV_FILE` points there.

## Paths

| Variable | Default | Purpose |
|---|---|---|
| `MEMORIZZ_HOME` | `~/.memorizz` | Config, credentials, approvals, CLI state, and default memory |
| `MEMORIZZ_ENV_FILE` | `$MEMORIZZ_HOME/.env` | Canonical environment file |
| `MEMORIZZ_MEMORY_ROOT` | `$MEMORIZZ_HOME/memory` | Filesystem provider root |

`$MEMORIZZ_HOME/harnesses.json` contains the secret-free external-harness
allowlist, preference order, executable names, workspace roots, and OpenHands
isolation declaration. Create it with `memorizz harness init`; see the
[meta-harness guide](../guides/meta-harness.md).

## Provider selection

| Variable | Example |
|---|---|
| `MEMORIZZ_DEFAULT_LLM_PROVIDER` | `openai`, `anthropic`, `ollama`, `azure` |
| `MEMORIZZ_DEFAULT_LLM_MODEL` | Provider model or deployment name |
| `MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER` | `openai`, `ollama`, or another registered provider |
| `MEMORIZZ_DEFAULT_EMBEDDING_MODEL` | Embedding model name |
| `MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS` | Dimension matching the persisted vector schema |
| `MEMORIZZ_BACKEND` | `filesystem`, `mongodb`, or `oracle` |

## Meta-harness

| Variable | Purpose |
|---|---|
| `MEMORIZZ_CODEX_COMMAND` / `MEMORIZZ_CODEX_MODEL` | Codex CLI and model override |
| `MEMORIZZ_CLAUDE_CODE_COMMAND` / `MEMORIZZ_CLAUDE_CODE_MODEL` | Claude Code CLI and model override |
| `MEMORIZZ_OPENHANDS_COMMAND` / `MEMORIZZ_OPENHANDS_MODEL` | Isolated OpenHands wrapper and model override |
| `MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION` | Confirms the OpenHands command enters an operator-owned isolation boundary |
| `MEMORIZZ_HARNESS_ALLOWLIST` | Comma-separated adapter allowlist |
| `MEMORIZZ_HARNESS_CONTEXT_MAX_CHARS` | Bounded memory-context size |
| `MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION` | Enables first-party MCP harness tools |
| `MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS` | Comma-separated MCP workspace roots |

If `MEMORIZZ_BACKEND` is unset, CLI/UI agent creation uses filesystem memory.
SDK construction also defaults to filesystem unless
`memory_provider=False` is explicit.

## Credentials

Set only the providers you use:

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
FIRECRAWL_API_KEY=
E2B_API_KEY=
MONGODB_URI=
ORACLE_USER=
ORACLE_PASSWORD=
ORACLE_DSN=
```

Use a deployment secret manager for production. Do not put secrets in
`llm_config`, `mcp_servers`, browser-control configuration, agent metadata,
benchmark results, or committed `.env` files.

MCP client tokens and OAuth credentials use the encrypted credential store.
Set `MEMORIZZ_MCP_ENCRYPTION_KEY` to a stable Fernet key in containers and
multi-instance deployments. Losing that key makes stored tokens unrecoverable.

## Backend-specific configuration

=== "Filesystem"

    No variables are required. See the [filesystem provider](../memory-providers/filesystem.md).

=== "MongoDB"

    ```dotenv
    MEMORIZZ_BACKEND=mongodb
    MONGODB_URI=mongodb://localhost:27017
    MONGODB_DB_NAME=memorizz
    ```

=== "Oracle"

    ```dotenv
    MEMORIZZ_BACKEND=oracle
    ORACLE_USER=memorizz_user
    ORACLE_PASSWORD=
    ORACLE_DSN=localhost:1521/FREEPDB1
    MEMORIZZ_ORACLE_INDEX_POLICY=lazy
    ```

    Run `memorizz oracle preflight --json` before traffic. See the
    [Oracle guide](../memory-providers/oracle.md).

## UI security controls

| Variable | Purpose |
|---|---|
| `MEMORIZZ_UI_AUTH_TOKEN` | Enables token login and signed sessions |
| `MEMORIZZ_UI_SESSION_SECRET` | Stable session-signing secret |
| `MEMORIZZ_UI_SESSION_SECONDS` | Session lifetime, bounded to 5 minutes–24 hours |
| `MEMORIZZ_UI_COOKIE_SECURE` | Marks the session cookie secure behind HTTPS |
| `MEMORIZZ_UI_READ_ONLY` | Blocks provider mutations and unsafe HTTP methods |
| `MEMORIZZ_UI_TRACE_CONTENT_MODE` | `full`, `redacted`, or `metadata` |
| `MEMORIZZ_UI_AUDIT_LOG` | Trace-view audit JSONL path |

The default UI is a localhost development tool. Set authentication, HTTPS,
read-only mode, redaction, and network controls before inspecting production
data.

## Inspect resolved state

```bash
memorizz config
memorizz capabilities --json
```

The first command shows resolved paths and selected defaults. The capability
report shows installed dependencies and effective readiness without returning
credential values.
