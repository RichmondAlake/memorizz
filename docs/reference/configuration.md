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
the canonical MemoRizz environment file by default. CLI editors also accept
`--project` or `--env-file PATH` as explicit write targets; a custom file is
only loaded when selected by `MEMORIZZ_ENV_FILE` at launch.

Run `memorizz config path`, `memorizz config keys`, `memorizz config get KEY`, or
`memorizz config set KEY [VALUE]`. Omit VALUE for a prompt; credentials and
unknown keys always require hidden input. `memorizz memory configure` guides
memory setup, and `memorizz notion connect` handles Notion specifically.
The [CLI configuration guide](../getting-started/cli.md#configuration) covers
REPL equivalents, override warnings, atomic/private file writes and history safety.
Saved defaults require restart; existing clients/agents are not reconnected
or migrated. UI Settings and `/login` additionally update the current process's
environment, without replacing already constructed clients.

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
| `MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER` | `openai`, `ollama`, `huggingface` (a local Sentence Transformers model or folder), or another registered provider |
| `MEMORIZZ_DEFAULT_EMBEDDING_MODEL` | Embedding model name |
| `MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS` | Dimension matching the persisted vector schema |
| `MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL` | With `openai`: an OpenAI-compatible server, such as `http://localhost:11434/v1` (Ollama), vLLM or LM Studio |
| `MEMORIZZ_DEFAULT_EMBEDDING_API_KEY` | That server's key, if it takes one; the OpenAI key is never sent to it |
| `MEMORIZZ_BACKEND` | `filesystem`, `mongodb`, `oracle`, or `notion` |
| `OLLAMA_HOST` | Ollama server for models and embeddings; default `http://localhost:11434` |

If `MEMORIZZ_BACKEND` is unset, CLI/UI agent creation uses filesystem memory.
SDK construction also defaults to filesystem unless
`memory_provider=False` is explicit.

### Embeddings from another server, and shorter vectors

The `openai` embedding provider also talks to OpenAI-compatible servers. With
a base URL that is not OpenAI's, it takes any model the server serves:

```dotenv
# Ollama through its OpenAI-compatible API, at 256 dimensions
MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
MEMORIZZ_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL=http://localhost:11434/v1
MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=256
```

In code, pass the same settings as the provider's config:

```python
from memorizz.embeddings import EmbeddingManager

embeddings = EmbeddingManager(
    "openai",
    {"model": "nomic-embed-text", "base_url": "http://localhost:11434/v1", "dimensions": 256},
)
```

- Without `dimensions`, the server's full size is used, learned from its
  first answer.
- With `dimensions`, the server is asked for that size. A server that refuses
  or ignores it gets its vectors cut here: the first values, scaled back to
  unit length, which is what servers that take the setting do for
  Matryoshka-trained models (nomic-embed-text, EmbeddingGemma,
  Qwen3-Embedding).
- The OpenAI key is not sent to another server; set
  `MEMORIZZ_DEFAULT_EMBEDDING_API_KEY` (or `api_key` in the config) when the
  server takes a key.

The `ollama` provider takes `dimensions` too (`{"model": "nomic-embed-text",
"dimensions": 256}`, or `MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS`): Ollama cuts
the vectors, and an older Ollama's full vectors are cut the same way here. A
size larger than the model's is an error. The vector index must be built for
the size you choose, so changing it means re-embedding the store.

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

## Coding-agent plugins

Settings the Codex and Claude Code plugins read (`memorizz config set NAME
VALUE`, or the options of `memorizz plugin install`); see
[MemoRizz in Codex and Claude Code](../guides/coding-agent-plugins.md).

| Variable | Purpose |
|---|---|
| `MEMORIZZ_SESSION_CAPTURE` | `turns` (default) saves each turn; `off` saves no turns and makes no summaries |
| `MEMORIZZ_SESSION_SUMMARY` | `false` keeps turns but skips session summaries (default `true`) |
| `MEMORIZZ_PLUGIN_RUNS` | `false` stops recording each session as a run on the UI's Harnesses page (default `true`; local store only) |
| `MEMORIZZ_PROMPT_RECALL` | `true` adds related memories to every prompt (default `false`) |
| `MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL` | The user whose memories a local (stdio) server reads and writes, for a shared store |
| `MEMORIZZ_MCP_SERVER_INGEST_ROOTS` | Comma-separated folders `memorizz_ingest` may read (default: home for stdio, none for HTTP) |
| `MEMORIZZ_PLUGIN_REMOTE_URL` | A hosted MemoRizz MCP server (`https://…/mcp`) used instead of the local store |
| `MEMORIZZ_PLUGIN_REMOTE_TOKEN_ENV` | The environment variable holding that server's API key (default `MEMORIZZ_MCP_TOKEN`) |
| `MEMORIZZ_BIN` | The `memorizz` executable the plugins run when the one recorded at install is gone |

## Fine-tuning

See [Fine-tuning decision models, embeddings and rerankers](../guides/fine-tuning.md).

| Variable | Purpose |
|---|---|
| `MEMORIZZ_FINETUNE_TEACHER` | The model that writes synthetic examples (`provider/model`); default: your default LLM, then a local Ollama model |
| `MEMORIZZ_RERANKER_MODEL` | A fine-tuned reranker folder (a cross-encoder or a relevance decision model). Agent recall and `memorizz_search_memories` reorder what vector search found with it. Empty: no reranking |
| `MEMORIZZ_ROUTING_MIN_CONFIDENCE` | Below this router confidence, a `memory_type="auto"` write goes to the knowledge base (default `0.5`) |
| `MEMORIZZ_MCP_SERVER_ALLOW_TRAINING` | Lets `memorizz_start_finetune_run` train (default: on for local `stdio` with agent execution, off over HTTP) |

**Swap in** on the Fine-tuning page, or `memorizz finetune swap`, writes
`MEMORIZZ_RERANKER_MODEL` or the `MEMORIZZ_DEFAULT_EMBEDDING_*`
settings to MemoRizz's `.env` (`MEMORIZZ_HOME/.env`); the active router is
kept in the memory provider with the datasets. An exported variable of the
same name overrides the saved value when a process starts (see
[Precedence](#precedence)), so unset it for a swap to hold after a restart.

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
NOTION_TOKEN=
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

=== "Notion"

    Run `memorizz notion connect` for guided setup, or configure:

    ```dotenv
    MEMORIZZ_BACKEND=notion
    NOTION_TOKEN=
    MEMORIZZ_NOTION_DATA_SOURCE_ID=
    MEMORIZZ_NOTION_SEMANTIC_BACKEND=filesystem
    MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=ollama
    MEMORIZZ_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
    ```

    The LLM remains independently configured. See the
    [Notion guide](../memory-providers/notion.md) for secondary storage options,
    explicit provisioning, permissions, synchronization and operational limits.

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
