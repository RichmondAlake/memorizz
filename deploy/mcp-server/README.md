# Host the MemoRizz MCP server

Run MemoRizz's MCP server as a web service so a team (or a plugin directory
that requires a public HTTPS endpoint) uses one shared memory. Each person
connects with their own API key, and each key's memories are kept apart.

## 1. Choose keys and settings

Create `deploy/mcp-server/.env` (never commit it):

```bash
MEMORIZZ_MCP_DOMAIN=mcp.example.com
# One entry per person: their name and a long random token (openssl rand -hex 32).
MEMORIZZ_MCP_SERVER_API_KEYS={"ada": "…token…", "grace": "…token…"}
# Semantic search and session summaries need models:
OPENAI_API_KEY=sk-…
# Optional: a shared database instead of files on the server's volume.
# MEMORIZZ_BACKEND=mongodb
# MONGODB_URI=mongodb+srv://…
```

## 2. Start it

Point the DNS name at the host, open ports 80 and 443, then:

```bash
docker compose -f deploy/mcp-server/compose.yaml --env-file deploy/mcp-server/.env up -d --build
```

Caddy gets a certificate from Let's Encrypt and serves the server at
`https://mcp.example.com/mcp`. Memories live in the `memorizz-data` volume
(or your database).

Without Docker: `memorizz mcp serve --transport streamable-http --host 0.0.0.0
--port 8766 --allow-writes --public-url https://mcp.example.com` behind any HTTPS
reverse proxy, with `MEMORIZZ_MCP_SERVER_API_KEYS` set.

## 3. Connect Codex and Claude Code

Each person puts their token in an environment variable and installs the
plugin in remote mode:

```bash
export MEMORIZZ_MCP_TOKEN=…their token…
memorizz plugin install codex --remote https://mcp.example.com/mcp
memorizz plugin install claude-code --remote https://mcp.example.com/mcp
```

The plugin's tools then call the hosted server, and its hooks (session
context, turn capture, summaries) go there too. `memorizz plugin install
<agent> --local` switches back to the local store.

## Check it

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://mcp.example.com/mcp   # 401 without a key
```

In a session, ask the agent for `memorizz_memory_status`.
