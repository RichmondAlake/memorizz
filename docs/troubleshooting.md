# Troubleshooting

## A saved CLI setting is not taking effect

Run `memorizz config path` to see the save target and `memorizz config get KEY`
to compare the effective source with the saved value (credentials remain
hidden). Process exports take precedence over project `.env`, which takes
precedence over the shared Memorizz file. Use `memorizz config set KEY --project`
to edit an intentional project override, or update/unset its shell export.
An empty assignment in a project `.env` also masks a shared value.

Restart after `config set` or `memory configure`; they do not reconnect a
running agent or migrate its memory. `/config` shows the active provider.
`/provider` selects the LLM, `/memory` selects a memory ID, and
`/memory-provider` configures memory storage for the next launch.

For Notion, run `memorizz notion connect` to check the exact shared library and
select the vector backend. `/login notion` saves `NOTION_TOKEN`, not `NOTION`.
A 404 can mean the connection cannot access the page/database: share it with
the connection, even if you can open it in your browser. Setup's read-only
check is not a write-permission or vector-service readiness test.

If a save fails, inspect the shown target's permissions, invalid dotenv syntax
or symlinks. Nothing is overwritten on a failed atomic config save. Hidden
input requires an interactive terminal; use a secret manager/process export
in headless environments. Never pass a credential as a command argument.


Start with the two structured diagnostics:

```bash
memorizz config
memorizz capabilities --json
```

For Oracle, also run `memorizz oracle preflight --json`.

## Agent has no model

Symptom: creation succeeds with `--no-llm`, but execution reports that no LLM
is configured.

Configure a provider in the process environment or rebuild/update the agent
with an `llm_config`. A no-LLM agent is useful for configuration and discovery,
but it cannot execute a turn.

## Ollama repeats its persona or ignores a new question

Check `ollama ps` for the active context size and the Ollama server log for
`truncating input prompt`. Earlier Memorizz versions reported a 128,000-token
window without sending `num_ctx` to Ollama. The daemon could truncate the agent
instructions, tool schemas, or conversation using a much smaller window.

The Ollama provider now explicitly requests 8,192 tokens by default and uses
that same value for history budgeting. Set `context_window_tokens` in the model
configuration, or `OLLAMA_CONTEXT_LENGTH` before launching the CLI, to choose
another window. Resuming the CLI or switching models refreshes the history
budget. See [model configuration](getting-started/model-providers.md).

If a local model answers directly during the private tool phase, the streaming
loop now checks completion requirements and proceeds to public generation with
tools disabled. It does not repeatedly demand the internal finalization tool.
See the [streaming contract](guides/streaming.md).

## Context budget or incomplete-answer errors

- `context_window_exceeded`: shorten supplied context or tool descriptions, or
  increase `llm_config["context_window_tokens"]` within your model's capacity.
  Older conversation turns are removed first; the current question and its
  tool evidence are preserved.
- `empty_response`: the model returned no answer. Retry or select another model.
- `provider_length`: increase the provider's output-token limit (`num_predict`
  for Ollama, `max_tokens` or `max_completion_tokens` as supported by the
  provider), or ask for a shorter answer.

These errors do not count as completed answers and are not cached. Streaming
retains any partial text already displayed. Synchronous `run()` raises
`ProviderStreamError` with the same `code`.

## Semantic recall or cache is degraded

Filesystem persistence works without an embedder. Semantic retrieval and
semantic cache require a compatible embedding provider.

For a local setup:

```bash
ollama pull nomic-embed-text
```

For OpenAI embeddings, set `OPENAI_API_KEY` and configure the model/dimensions.
Never change dimensions against existing vector columns without a migration.

## A saved agent is not visible

The SDK, CLI, UI, and MCP server must use the same memory provider and
`MEMORIZZ_HOME`. Use `build_and_save()` or `build(persist=True)`; `build()` alone
creates only a runtime object.

```bash
memorizz config
memorizz agents list --json
```

## Memory appears to cross or disappear between users

Pass `user_id`, `memory_id`, and `thread_id` explicitly on every shared-service
request. `user_id=None` selects anonymous/legacy rows; it is not a wildcard.
See [multi-tenant applications](guides/multi-tenant.md).

## Semantic cache returns stale business data

Similarity does not establish freshness. Mark side-effecting tools correctly,
set data-version/domain metadata, use domain freshness limits, and invalidate a
domain when its source changes. See
[production governance](guides/production-governance.md#semantic-cache-correctness).

## MCP reports `authorization_required`

The endpoint is reachable but the remote grant is incomplete. Finish OAuth or
configure the expected bearer credential. `approval_required` is different: it
means MemoRizz paused a mutating call for a host decision.

For local services, private-network access must be an explicit operator choice.
For remote services, use HTTPS and review host/stdio allowlists. See
[MCP connectivity](guides/mcp-connectivity.md).

## MCP mutation never executes

List, approve, and resume the exact durable proposal:

```bash
memorizz mcp approvals --status pending
memorizz mcp approve PROPOSAL_ID --approver operator@example.com
memorizz mcp resume PROPOSAL_ID
```

Expired, rejected, consumed, owner-mismatched, or argument-mismatched proposals
fail closed.

## UI returns 401 or blocks writes

When `MEMORIZZ_UI_AUTH_TOKEN` is set, sign in with that token or supply it as a
Bearer credential. `MEMORIZZ_UI_READ_ONLY=true` intentionally blocks agent
creation, approvals, and provider mutations. See
[observability and trace inspection](observability-ui.md).

## Oracle connection, vector, or index errors

Use preflight to check the DSN/service, PDB state, schema privileges, ONNX
model, vector dimensions, index state, and `VECTOR_MEMORY_SIZE`.

- `ORA-51803`: configured embedding dimensions do not match the vector column.
- `ORA-51962`: vector memory is insufficient; use `none`/`lazy` indexing or ask
  a DBA to resize it.

Follow the [Oracle troubleshooting guide](memory-providers/oracle.md#troubleshooting).

## E2B or browser control is unavailable

- E2B validates `E2B_API_KEY` at construction. Confirm the sandbox extra and
  key are present.
- Browser Use should be installed in an isolated tool environment. Run
  `browser-use doctor` and inspect `agent.capability_report()`.

Browser Use has its own model configuration. An Ollama chat session does not
supply credentials for its default OpenAI browser model. If enabling it reports
`OPENAI_API_KEY is required`, configure that key or select another supported
browser model provider with `MEMORIZZ_BROWSER_USE_LLM_PROVIDER` and its matching
credentials. This configuration error does not disable ordinary chat.

See [E2B](sandbox/e2b.md) and [browser control](browser-control/index.md).

## An agent harness reports `authentication_required`

Run `memorizz harness doctor HARNESS --json` and follow its `remediation` field.
Claude Code requires `ANTHROPIC_API_KEY` or an enabled Bedrock, Vertex, or
Foundry mode because MemoRizz uses `--bare`. Codex can use an authenticated
`CODEX_HOME` session, and OpenHands can use provider/CLI configuration, so a
missing environment key alone does not incorrectly mark those adapters failed.

SDK, CLI, UI, and MCP return the same typed, secret-free error. Rotate or revoke
credentials in the provider, never paste them into a task, notebook, trace, or
support report.

## Provider errors during streaming

`run_stream()` emits a typed terminal error event. Set
`raise_on_provider_error=True` when the host also needs an exception:

```python
for event in agent.run_stream(query, raise_on_provider_error=True):
    handle(event)
```

Authentication failures propagate after the terminal event.

## Preparing a useful bug report

Include:

- MemoRizz version and `capabilities --json` with identifiers reviewed;
- Python and operating-system versions;
- selected provider and secret-free preflight output;
- the smallest reproducing code sample;
- the typed error code and traceback; and
- whether the problem reproduces with filesystem memory.

Never attach `.env`, OAuth tokens, database passwords, raw production traces,
or benchmark datasets containing restricted data.
