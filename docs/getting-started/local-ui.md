# Local UI Guide

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](../guides/streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

The Memorizz local UI gives you a browser-based workflow for connecting to your memory provider, creating/editing agents, running conversations, and inspecting memory state without writing extra code.

## What You Can Do

- Connect to Oracle, MongoDB, or filesystem providers.
- Create, edit, favorite, and delete agents.
- Run agents in Playground with streaming responses.
- Launch and govern Codex, Claude Code, OpenHands, and native MemAgent tasks
  through the Agent Harnesses operator console.
- Configure Browser Use per agent and approve/reject exact browser tasks in
  Playground before they execute.
- Create, pause, resume, trigger, inspect, and delete durable scheduled
  automations.
- Connect agents to Notion, Google Calendar, or custom MCP servers, including
  OAuth, encrypted credentials, allowlists, and mutation approvals.
- Inspect memory types (personas, toolbox, conversations, workflows, long-term, short-term, entity, summaries, shared, cache).
- Configure continual learning, choose user or reviewed developer authority
  for newly promoted skills, inspect trajectory gates, and activate/demote
  learned skills.
- Enable the memory-first learning control plane, inspect immutable event and
  artifact counts, compile pending events, and approve reversible forgetting.
- Review run traces by agent and thread.
- Run AgentMemBench, LongMemEval-V2, LoCoMo-Plus, BEAM, MemoryAgentBench, and
  legacy LongMemEval in Evalground.
- Manage runtime keys and defaults in Settings.

## 1. Install UI Dependencies

For local development against this repo:

```bash
pip install -e ".[ui]"
```

For package usage:

```bash
pip install "memorizz[ui]"
```

If you also want all optional integrations (Oracle, MongoDB, sandbox, docs, etc.):

```bash
pip install "memorizz[all]"
```

## 2. Start The UI

```bash
memorizz ui
```

Default URL: `http://127.0.0.1:8765`

Optional host/port overrides:

```bash
memorizz ui --host 0.0.0.0 --port 9000
```

## 3. Connect To A Memory Provider

On first load, the UI opens the `/connect` page.

### Oracle

- Required: `Username`, `Password`, `DSN`
- Optional: `Schema` (defaults to `Username`)
- Example DSN: `localhost:1521/FREEPDB1`

### MongoDB

- Required: `Connection URI`
- Optional: `Database Name` (defaults to `memorizz`)
- Example URI: `mongodb://localhost:27017`

### Filesystem

- Required: `Storage Path`
- Example path: `~/.memorizz/memory`

For Oracle embedding consistency across UI, SDK, and evaluations, first run
`memorizz oracle preflight --json`, then set the dimension reported by the
existing schema:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=384  # example; use preflight value
```

## 4. Navigation Map

Once connected, the sidebar is your main navigation.

| Section | Route | Purpose |
|---|---|---|
| Dashboard | `/dashboard` | High-level provider connection details and memory counts. |
| Agents | `/agents` | Browse agents, sort, quick-chat, and jump into edit/playground. |
| Create Agent | `/agents/new` | Build a new agent with mode, persona, tool/memory options, and provider config. |
| Playground | `/playground` and `/agents/{id}/playground` | Interactive chat, thread switching, token/context stats, and per-agent runtime config. |
| MCP Connections | `/mcp` | Configure/test MCP servers, authorize OAuth, inspect capabilities, and invoke governed tools. |
| Automations | `/automations` | Create schedules, inspect runs, and manage worker-backed agent jobs. |
| Memory Types | `/memory/{type}` | Browse stored memory entries by type (`personas`, `toolbox`, `conversations`, etc.). |
| Continual Learning | `/memory/workflows` and `/memory/skills` | Review canonical trajectory classes, run gated distillation, inspect persisted skill authority, and activate/demote skills. |
| Learning Control Plane | `/learning-control-plane` | Inspect scoped event/artifact counts, compile pending events, and plan/apply reversible forgetting. |
| Agent Harnesses | `/harnesses` | Probe adapters, launch bounded runs, decide exact approvals, cancel work, and inspect verification evidence. |
| Traces | `/traces` | Filter/search agents and inspect thread-level trace timelines. |
| Evalground | `/evalground` | Run local open-model memory suites, monitor logs, and review run history/results. |
| Settings | `/settings` | Save API keys and runtime defaults into the UI session and `.env`. |

## 5. Suggested First Run Workflow

1. Open `Settings` and add at least `OPENAI_API_KEY`.
2. Create an agent in `Agents -> Create Agent`.
   To evaluate learned procedures, enable **Continual learning**, choose
   **User context** or **Developer instructions**, and keep **Require shadow
   review** enabled. Developer authority cannot be saved without review.
   Enable **Memory-first learning control plane** to use bounded EvidencePack
   retrieval and the operator page.
   Select **Browser Use** under Browser Control only after installing the
   isolated `browser-use` CLI and configuring its matching LLM key.
3. Open that agent in `Playground`.
4. Send a message and confirm streaming response.
5. Switch to `Traces` to inspect events for that run.
6. Review memory entries under `Memory Types` (especially conversations/summaries/cache).
7. Open `Learning Control Plane` to inspect the same scope and its compiler or
   forgetting state.
8. Open `Agent Harnesses` to inspect local adapter readiness or launch a
   bounded repository task. Direct edits pause until a host operator approves
   the exact envelope.

## Agent Harnesses in the UI

The `/harnesses` page is an operator surface, not a vendor terminal emulator.
It shows adapter capability probes, bounded launch controls, durable runs and
events, token/cost data when the adapter reports it, workspace changes, host
verification, and pending exact-envelope proposals. `MEMORIZZ_UI_READ_ONLY=true`
blocks launches, cancellation, approval, rejection, and resume endpoints.
Select a saved agent when choosing the `memagent` adapter; native execution is
explicit-only and never wins automatic routing.

Agent create/edit forms can persist `runtime` mode, where an external harness
owns the full turn, or `delegate` mode, where MemAgent receives governed
specialist tools. See the
[Memory-First Meta-Harness guide](../guides/meta-harness.md).

## Browser control in the UI

Install Browser Use outside the MemoRizz environment, then validate it:

```bash
uv tool install --python 3.12 browser-use
browser-use install
browser-use doctor
```

In **Settings -> Browser Control**, choose the Browser Use LLM provider/model,
set allowed/prohibited domains, and configure step and wall-clock limits. API
keys are stored in the shared MemoRizz environment file; agent records retain
only secret-free policy. Browser control remains disabled for an agent until it
is selected in the create/edit form or Playground.

When the model requests `browser_control`, Playground renders the exact task,
arguments, policy reason, and proposal identifier. **Approve & resume** consumes
that single-use proposal; **Reject** prevents execution. A model-visible
`approved` or `confirm` field does not exist.

See the [Browser Control guide](../browser-control/index.md).

## MCP in the UI

Open **MCP Connections** and select an agent. Presets are available for Notion,
Google Calendar, and local stdio servers. Bearer tokens, OAuth client secrets,
custom header values, and stdio environment secrets move to the encrypted
credential store rather than agent JSON. Use **Test**, **Tools**, **Resources**,
and **Prompts** to inspect a connection; mutating tool calls enter the same
durable approval lifecycle.

The page also shows commands for exposing MemoRizz itself as a local stdio or
authenticated Streamable HTTP MCP server. See the
[MCP Connectivity](../guides/mcp-connectivity.md) and
[MemoRizz MCP Server](../guides/mcp-server.md) guides.

For developer authority, an activated skill is sent as a native developer
message to OpenAI and through Anthropic's top-level system parameter. Only
application-owned, reviewed skills should use it; system policy and live tool
results remain higher-trust inputs. See the
[Continual Learning guide](../guides/continual-learning.md).

## Evalground Requirements

The five memory-suite adapters require a running Ollama daemon for local
`nomic-embed-text` embeddings and an official dataset path. The default Ollama
reader (for example, `qwen2.5:3b`) uses no external API. Selecting an OpenAI
reader keeps memory and embeddings local but requires `OPENAI_API_KEY` and
records an estimated token cost. Neither route requires Oracle. The legacy
LongMemEval option still requires an Oracle connection, `OPENAI_API_KEY`, an
agent, and its downloaded dataset files.

## Security Notes

- The local UI is intended for development/local usage.
- Default bind is localhost (`127.0.0.1`).
- Authentication is disabled by default, but built-in token login and signed
  sessions are enabled when `MEMORIZZ_UI_AUTH_TOKEN` is set.
- Avoid exposing the UI directly on public networks.
- Browser actions can change external systems. Keep approval enabled and use a
  narrow domain allowlist for production agents.

For an operator deployment, use a read-only database identity and configure
authentication, a separate session secret, secure cookies behind HTTPS,
redacted or metadata-only trace content, and audit logging. See
[Observability and Trace Inspection](../observability-ui.md) and
[Configuration and Secrets](../reference/configuration.md).

## Troubleshooting

### UI Fails To Start

- Confirm UI extras are installed: `pip install "memorizz[ui]"`
- If using editable install, reinstall dependencies: `pip install -e ".[ui]"`

### Connection Errors

- Oracle: verify DSN/user/password and Oracle client requirements.
- MongoDB: verify URI and DB permissions.
- Filesystem: verify the path is writable.

### Port In Use

Start on a different port:

```bash
memorizz ui --port 9000
```
