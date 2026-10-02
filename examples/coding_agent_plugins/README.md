# One memory, two coding agents

Codex works on a project; Claude Code opens the same project later and already
knows what Codex did and why. Both run the MemoRizz plugin, and the plugin
gives every project one memory ID (`project-<folder>-<hash>`), whichever agent
is open.

## Run it

```bash
python examples/coding_agent_plugins/cross_agent_demo.py
python examples/coding_agent_plugins/cross_agent_demo.py --workdir ~/memorizz-demo --ui
```

`--workdir` keeps the files somewhere you choose. `--ui` also adds both
sessions to the Harnesses page of the MemoRizz UI you normally run.

You need `memorizz[mcp]`, the `codex` and `claude` CLIs (signed in) and,
for search and session summaries, Ollama with `nomic-embed-text` and
`qwen2.5:7b`. The script doesn't touch your own setup. Codex gets a throwaway
`CODEX_HOME` (your login is linked in, nothing else), Claude Code loads the
plugin for that run only (`--plugin-dir`), and MemoRizz writes to a fresh
store in the work folder.

What happens:

1. It creates a small project, `fernvale-checkout`, with one failing test.
   Reward points use `round()`, so a $49.99 order earns 50 points instead of 49.
2. It runs `memorizz plugin install codex`.
3. **Session 1 is Codex.** It fixes the test and saves a fact saying what it
   changed and why. The plugin saves the turn, and when the session ends it
   writes a summary.
4. The script prints what MemoRizz now holds for the project.
5. **Session 2 is Claude Code**, in the same folder. It's asked *"Why do reward
   points use int() rather than round(), and what was done in the last
   session?"* It starts with Codex's session summary and searches what Codex
   saved.
6. The plugin has recorded each session as a run. The script lists each run's
   commands, tool calls and file changes; the UI shows them as trajectories.

From real runs (Codex on a ChatGPT plan; Claude Code on Haiku 4.5 costs $0.05–0.06 a run):

```text
3. Session 1 — Codex fixes the bug and records why
  Codex (53s):
    Fixed `test_points_are_whole_dollars_rounded_down` with a one-line change in
    `src/checkout/rewards.py`: replaced `round(subtotal)` with `int(subtotal)` …
    Saved one fact to MemoRizz project memory describing the change and why.

4. What MemoRizz now holds for the project
  fact: Changed src/checkout/rewards.py points_for from round(subtotal) to int(subtotal)
  because rewards award one point per whole dollar spent: a $49.99 subtotal must earn
  49 base points, not 50. …
  summary [codex]: - Fixed a failing test … - Changed `round(subtotal)` to `int(subtotal)` …

5. Session 2 — Claude Code, a different agent, opens the same project
  Claude Code ($0.060):
    The rewards system awards one point per whole dollar spent … A $49.99 subtotal
    should earn 49 base points, not 50 …
    The previous session (Oct 2, 2026) fixed the failing test
    `test_points_are_whole_dollars_rounded_down` by making this one-line change …

6. One memory, two agents
  Conversation threads in project-fernvale-checkout-5aef21: claude-code, codex
  Facts: 1 · turns: 4 · session summaries: 2

7. Each session as a run, with its trajectory
  codex run 20933cb3: 6 commands, 3 tool calls (3 to MemoRizz: memorizz_search_memories,
  memorizz_server_info, memorizz_store_memory), 1 file change, 53s
  claude-code run 866a564c: 0 commands, 2 tool calls (1 to MemoRizz:
  memorizz_search_memories), 0 file changes, 13s, $0.050
```

## See the trajectories

![Compare: the Codex session that fixed the bug and the Claude Code session that explained it from memory.](../../docs/assets/screenshots/plugin-compare-dark.png)

Open the UI on the demo's store (or your usual UI after `--ui`), go to
**Harnesses**, and pick a run marked *plugin session*. Its trajectory shows
what the plugin loaded at session start, every command and tool call (the
MemoRizz ones as `memorizz › memorizz_search_memories` and so on) and the file
change with its diff. Tick both runs and press **Compare** to see the Codex
and Claude Code sessions side by side.

```bash
MEMORIZZ_HOME=<workdir>/memorizz-home memorizz ui --port 8806
```

The connect page opens with FileSystem and the demo's store filled in; press
**Connect**.

The same UI's **Memory → Conversation** has one thread per agent, and
**Summaries** has each session's summary.

## Show it live

Two terminals, one project:

```bash
memorizz plugin install codex
memorizz plugin install claude-code

# Terminal 1
cd fernvale-checkout && codex
#   (first run: open /hooks and trust the MemoRizz hooks)
#   "Fix the failing test, then remember what you changed and why."
#   Exit Codex when it's done; the plugin writes the session summary.

# Terminal 2
cd fernvale-checkout && claude
#   "Why do reward points use int() rather than round()?
#    What happened in the last session?"
```

Claude Code's answer comes from what Codex saved. Then do it the other way
round: tell Claude Code something new ("remember we only ship to the UK and
Ireland"), exit, and ask Codex about it. With `memorizz ui` open on the
Harnesses page, each session appears as a run after its first turn and
updates as you go.
