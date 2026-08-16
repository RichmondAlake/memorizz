# memorizz (npm bootstrapper)

This npm package is a **convenience bridge** that installs the real
[`memorizz`](https://pypi.org/project/memorizz/) Python CLI for Node-first
developers. On `npm install -g memorizz` it:

1. finds or installs [`uv`](https://docs.astral.sh/uv/) (a standalone binary that
   manages its own Python — you do **not** need Python pre-installed), then
2. runs `uv tool install "memorizz[mcp]"` so the full CLI, including MCP
   client/server commands, is available.

The `memorizz` command then execs the uv-installed CLI.

## First-class install paths

If you already use Python tooling, prefer these:

```bash
uv tool install "memorizz[mcp]"     # recommended for the complete CLI
pipx install "memorizz[mcp]"
```

## Uninstall

```bash
npm uninstall -g memorizz
uv tool uninstall memorizz   # the npm shim does not remove the uv-managed tool
```

## Env

- `MEMORIZZ_SKIP_POSTINSTALL=1` — skip the bootstrap during `npm install`.
