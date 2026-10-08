# MemoRizz for Claude Code

A Claude Code plugin that gives Claude durable, searchable project memory
through MemoRizz's MCP server, shared with the MemoRizz plugin for Codex and
with your MemoRizz agents.

- **MCP server** (`memorizz mcp serve --transport stdio --allow-writes`, or a
  hosted server with `--remote`): search memories and past sessions, save and
  correct facts (a correction supersedes the old fact; nothing is deleted),
  record entities, load project documents (`memorizz_ingest`) and check status.
  Forgetting asks for approval in MemoRizz. Saved agents, other harnesses and
  trace queries are opt-in.
- **Hooks:**
  - SessionStart: the project's memory ID (`project-<folder>-<hash>`, the
    same one the Codex plugin uses), the last session's summary and the
    newest facts;
  - UserPromptSubmit: keeps the prompt for turn capture; with
    `MEMORIZZ_PROMPT_RECALL=true`, also adds memories related to it;
  - Stop: saves the turn (request and final answer, secrets removed) to
    conversation memory, and records the session as a run on the MemoRizz UI's
    Harnesses page, with its trajectory, in a detached process;
  - PreCompact and SessionEnd: summarize the session with MemoRizz's default
    model, in a detached process.
- **Skills:** `/memorizz:memorizz-memory`, `/memorizz:memory-curator` and
  `/memorizz:memorizz-agents`.
- **Commands:** `/memorizz:remember <fact>`, `/memorizz:recall [topic]`,
  `/memorizz:forget <what>`, `/memorizz:memory-status`.
- **Subagent:** `memory-curator` finds duplicates, contradictions and stale
  facts, corrects them and proposes deletions.

## Install

With MemoRizz installed (`pip install 'memorizz[mcp]'`):

```bash
memorizz plugin install claude-code
```

Options: `--no-capture`, `--no-summaries`, `--prompt-recall`, `--user NAME`,
`--allow-agents`, `--allow-harness --harness-root PATH`, `--allow-traces`,
`--remote URL [--token-env NAME]` / `--local`.

From GitHub or a checkout of the MemoRizz repository:

```bash
claude plugin marketplace add RichmondAlake/memorizz   # or the checkout's path
claude plugin install memorizz@memorizz
```

Or try it for one session without installing:
`claude --plugin-dir plugins/claude-code/memorizz`.

Start a new session and allow the `memorizz` tools when Claude first asks.
Without a `memorizz` on Claude Code's PATH the plugin runs it from PyPI with
`uvx`.

## Portable memory archives

The `memory-transfer` skill exports/restores project or agent memory graphs
through the MCP archive tools and `memorizz plugin export-memory` /
`import-memory`. Import previews by default; `--apply` commits it. See the
[archive guide](../../../docs/guides/memory-transfer.md) for format, scope and
conflict options.
