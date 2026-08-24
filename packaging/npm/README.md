# memorizz (npm bootstrapper)

This npm package is a **convenience bridge** that installs the real
[`memorizz`](https://pypi.org/project/memorizz/) Python CLI for Node-first
developers. On `npm install -g memorizz` it:

1. finds or installs [`uv`](https://docs.astral.sh/uv/) (a standalone binary that
   manages its own Python — you do **not** need Python pre-installed), then
2. runs `uv tool install --python 3.12 "memorizz[mcp]"` so the full CLI,
   including MCP client/server commands, is available even when the host's
   default Python is older.

The `memorizz` command then execs the uv-installed CLI.

This includes `memorizz harness` for governing installed Codex, Claude Code,
OpenHands, and native MemAgent workers. The bootstrapper does not install those
vendor CLIs; inspect local readiness with `memorizz harness doctor`.

## First-class install paths

If you already use Python tooling, prefer these:

```bash
uv tool install --python 3.12 "memorizz[mcp]"  # recommended complete CLI
pipx install "memorizz[mcp]"
```

## Uninstall

```bash
npm uninstall -g memorizz
uv tool uninstall memorizz   # the npm shim does not remove the uv-managed tool
```

## Env

- `MEMORIZZ_SKIP_POSTINSTALL=1` — skip the bootstrap during `npm install`.
