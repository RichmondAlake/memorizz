# Local UI Guide

The Memorizz local UI gives you a browser-based workflow for connecting to your memory provider, creating/editing agents, running conversations, and inspecting memory state without writing extra code.

## What You Can Do

- Connect to Oracle, MongoDB, or filesystem providers.
- Create, edit, favorite, and delete agents.
- Run agents in Playground with streaming responses.
- Inspect memory types (personas, toolbox, conversations, workflows, long-term, short-term, entity, summaries, shared, cache).
- Review run traces by agent and thread.
- Run LongMemEval benchmarks in Evalground.
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
memorizz run local
```

Default URL: `http://127.0.0.1:8765`

Optional host/port overrides:

```bash
memorizz run local --host 0.0.0.0 --port 9000
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
- Example path: `~/.memorizz/data`

Tip for Oracle embedding consistency across UI + notebooks:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=1536
```

## 4. Navigation Map

Once connected, the sidebar is your main navigation.

| Section | Route | Purpose |
|---|---|---|
| Dashboard | `/dashboard` | High-level provider connection details and memory counts. |
| Agents | `/agents` | Browse agents, sort, quick-chat, and jump into edit/playground. |
| Create Agent | `/agents/new` | Build a new agent with mode, persona, tool/memory options, and provider config. |
| Playground | `/playground` and `/agents/{id}/playground` | Interactive chat, thread switching, token/context stats, and per-agent runtime config. |
| Memory Types | `/memory/{type}` | Browse stored memory entries by type (`personas`, `toolbox`, `conversations`, etc.). |
| Traces | `/traces` | Filter/search agents and inspect thread-level trace timelines. |
| Evalground | `/evalground` | Run LongMemEval, monitor logs, and review run history/results. |
| Settings | `/settings` | Save API keys and runtime defaults into the UI session and `.env`. |

## 5. Suggested First Run Workflow

1. Open `Settings` and add at least `OPENAI_API_KEY`.
2. Create an agent in `Agents -> Create Agent`.
3. Open that agent in `Playground`.
4. Send a message and confirm streaming response.
5. Switch to `Traces` to inspect events for that run.
6. Review memory entries under `Memory Types` (especially conversations/summaries/cache).

## Evalground Requirements

Evalground currently requires:

- Oracle as the connected provider.
- `OPENAI_API_KEY` set.
- LongMemEval dataset files available (UI can download missing variants).

## Security Notes

- The local UI is intended for development/local usage.
- Default bind is localhost (`127.0.0.1`).
- No built-in authentication is enabled.
- Avoid exposing the UI directly on public networks.

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
memorizz run local --port 9000
```
