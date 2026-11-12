---
name: memorizz-agents
description: Consult the user's saved MemoRizz agents (each with its own instructions, memory and tools) through the memorizz MCP tools, or hand a bounded task to another agent harness (Codex, Claude Code, pi, Hermes) through MemoRizz. Use when the user mentions a MemoRizz agent by name, asks to "ask my <name> agent", or wants another harness's opinion or help.
---

# MemoRizz agents and harnesses

## Saved agents

- `memorizz_list_agents` lists the agents this server exposes; `memorizz_get_agent`
  and `memorizz_inspect_agent` describe one (instructions, memory, tools,
  delegates).
- `memorizz_execute_agent` runs one turn of an agent with a message and returns
  its answer. It is off unless the user enabled it
  (`memorizz config set MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION true`); if
  the tool says it isn't allowed, tell the user that command.
- An agent's turn uses its own model and may cost money; say which agent you
  are asking and why. Quote or summarise its answer and say it came from that
  agent.

## Other harnesses

With harness execution enabled
(`memorizz config set MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION true` and
`MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS` set to the allowed folders),
`memorizz_start_harness_run` runs a bounded task on Claude Code, pi, Hermes,
OpenHands, Codex or a saved MemAgent, in a project folder or a fresh scratch folder.
Edits, web access and model-chosen verification commands wait for the user's
approval in MemoRizz. Poll `memorizz_get_harness_run` for the result, and
`memorizz_cancel_harness_run` to stop it. Never start the agent you are running in (it would run inside itself).

`memorizz_start_harness_comparison` gives one read-only question to several
harnesses at once (each on its own model with `harness_models`), which is
useful for a second opinion on a design or a bug.

## A saved agent's own memory

- `memorizz_list_conversations` (agent_id) and `memorizz_get_conversation`
  read a saved agent's conversations.
- `memorizz_compact_conversation` summarizes and compacts one of them, and
  `memorizz_compile_memory` turns its continual-learning events into recall
  material. Both change the agent's memory, so run them only when the user asks.
- `memorizz_refine` runs one refine cycle: the agent's memory upkeep once, in a
  fixed order (entities, stated facts, skill staleness, promotion, skill
  review, persona learning, retention). Pass `dry_run=true` to see what it
  would do without changing anything. Run a real cycle only when the user
  asks: it calls the agent's background model. Changes that need a person's
  review come back under `pending`; tell the user they are waiting.
- `memorizz_update_agent` takes `holdout_rate` (0 to 1, default 0.1): the
  share of matching threads that run without an active learned skill, so
  each skill has a with-and-without figure. 0 turns it off. Change it only
  when the user asks.
  `memorizz_list_refine_runs` lists past cycles with what each job changed.
- `memorizz_get_agent` shows a persona's `guardrails`. They are rules only the
  host or an operator can change; do not try to change them through
  `memorizz_update_agent`.
- `memorizz_query_traces` (when the user enabled it with
  `memorizz config set MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES true`) shows what
  an agent did on a turn: model calls, tools and memory reads.
