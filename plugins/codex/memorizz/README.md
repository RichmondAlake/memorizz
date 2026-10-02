# MemoRizz for Codex

A Codex plugin that gives Codex durable, searchable project memory through
MemoRizz's MCP server, and access to your saved MemoRizz agents.

- **MCP server** (`memorizz mcp serve --transport stdio --allow-writes`, or a
  hosted server with `--remote`): search memories and past sessions, save and
  correct facts (a correction supersedes the old fact; nothing is deleted),
  record entities, load project documents (`memorizz_ingest`) and check status.
  Forgetting asks for approval in MemoRizz. Saved agents, other harnesses and
  trace queries are opt-in.
- **Hooks:**
  - SessionStart: the project's memory ID (`project-<folder>-<hash>`),
    the last session's summary and the newest facts;
  - UserPromptSubmit: keeps the prompt for turn capture; with
    `MEMORIZZ_PROMPT_RECALL=true`, also adds memories related to it;
  - Stop: saves the turn (request and final answer, secrets removed) to
    conversation memory, in a detached process;
  - PreCompact and SessionEnd: summarize the session with MemoRizz's default
    model, in a detached process.
- **Skills:** `memorizz-memory` (recall, save, correct, forget, load
  documents), `memory-curator` (tidy memory) and `memorizz-agents` (saved
  agents and other harnesses).

## Install

With MemoRizz installed (`pip install 'memorizz[mcp]'`):

```bash
memorizz plugin install codex
```

Options: `--no-capture`, `--no-summaries`, `--prompt-recall`, `--user NAME`,
`--allow-agents`, `--allow-harness --harness-root PATH`, `--allow-traces`,
`--remote URL [--token-env NAME]` / `--local`.

From a checkout of the MemoRizz repository, or from GitHub:

```bash
codex plugin marketplace add RichmondAlake/memorizz   # or the checkout's path
codex plugin add memorizz@memorizz
```

Start a new Codex thread. The first time (and after an update that changes
them), review and trust the MemoRizz hooks with `/hooks`; until then Codex
doesn't run them. Reading and saving memories, corrections and entity updates
don't ask for approval; other MemoRizz tools that change something do.
Without a `memorizz` on Codex's PATH the plugin runs it from PyPI with `uvx`.

Memories go to your MemoRizz store (`~/.memorizz/memory` by default), so the
MemoRizz UI and your other agents see them too.
