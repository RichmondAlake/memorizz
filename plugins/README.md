# MemoRizz plugins for coding agents

| Folder | Agent | Install |
|---|---|---|
| [`codex/memorizz`](codex/memorizz) | Codex | `memorizz plugin install codex`, or `codex plugin marketplace add RichmondAlake/memorizz` then `codex plugin add memorizz@memorizz` |
| [`claude-code/memorizz`](claude-code/memorizz) | Claude Code | `memorizz plugin install claude-code`, or `claude plugin marketplace add RichmondAlake/memorizz` then `claude plugin install memorizz@memorizz` |

Both give the agent long-term project memory through MemoRizz's MCP server:
hooks that load the project's memories and last session summary, save each
turn and summarize each session; and skills for recalling, saving, correcting
and curating memory. The Claude Code plugin adds slash commands and a
`memory-curator` subagent. They share one memory per project, so what Codex
learns Claude Code can recall, and the other way round, locally or through a
hosted server ([deploy/mcp-server](../deploy/mcp-server/README.md)).

The `scripts/` and `skills/` folders are the same files in both plugins; edit
them in one and copy to the other (`tests/unit/test_agent_plugins.py` checks
they match). The marketplace catalogs are `.agents/plugins/marketplace.json`
(Codex) and `.claude-plugin/marketplace.json` (Claude Code) at the repository
root.

See [MemoRizz in Codex and Claude Code](../docs/guides/coding-agent-plugins.md).
