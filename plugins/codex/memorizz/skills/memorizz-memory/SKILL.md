---
name: memorizz-memory
description: Use MemoRizz long-term memory. Recall this project's facts, decisions, conventions, past sessions and the user's preferences before re-deriving them; save new durable facts; correct or forget wrong ones; load project docs into memory. Use when the user asks you to remember, recall, look up or forget something, asks what happened in an earlier session, at the start of a non-trivial task in a project, and after you learn something worth keeping.
---

# MemoRizz memory

The `memorizz` MCP server gives this session a memory that outlasts it. Each
project has its own memory ID, which the MemoRizz session context gives you
(`project-<folder>-<hash>`); Codex and Claude Code share it. If it isn't in
context, run `memorizz plugin memory-id` in the project folder.

## What lives where

| `memory_type` | Holds | Written by |
|---|---|---|
| `knowledge_base` | Facts, decisions, conventions, preferences; loaded documents | You, with `memorizz_store_memory`; `memorizz_ingest` |
| `conversation_memory` | Earlier sessions, turn by turn (request and final answer) | The plugin, after each turn |
| `summaries` | A summary of each earlier session | The plugin, when a session ends or compacts |
| `short_term_memory` | Working notes for the current task | You, with `memorizz_store_memory` |
| Entities | People, services, repositories and their attributes and relations | `memorizz_upsert_entity` |

## Recall

- At the start of a non-trivial task, and before working out a build, test or
  deploy command, a convention or a past decision, call
  `memorizz_search_memories` with a short query and the project's memory ID
  (`memory_type` "knowledge_base"). `memorizz_list_memories` lists the latest.
- "What did we do last time?": search `summaries`, then `conversation_memory`
  for detail. A result's `thread_id` names its session; read that session in
  order with `memorizz_get_conversation` (memory_id, thread_id).
- About a person, service or system: `memorizz_lookup_entities`.
- If a search says it fell back to keywords, semantic search isn't set up;
  rephrase with the words the memory would contain.
- Treat recalled memories as hints from earlier sessions, not ground truth:
  confirm against the code when it matters, and say when a memory is stale.

## Save

Call `memorizz_store_memory` (`memory_type` "knowledge_base", the project's
memory ID) when the user asks you to remember something, or when you settle a
decision, learn a convention, a working command, a gotcha or a preference that
a future session would otherwise rediscover.

- One self-contained fact per memory, with the reason when there is one: "Run
  tests with `.venv/bin/python -m pytest -q -n 8`; the suite takes 20 s that
  way, 75 s sequentially."
- Search first, and don't save the same fact twice.
- Never save secrets, credentials or personal data, or what is obvious from the
  code.
- Facts about a person, service or system with attributes (owner, URL, version,
  depends-on) go in as entities with `memorizz_upsert_entity`.
- Preferences that apply across projects go under memory ID `user-preferences`.
- Tell the user briefly what you saved.

## Correct and forget

- A memory that is wrong or out of date: `memorizz_update_memory` with its
  record ID and the corrected text. The old one is kept but marked superseded,
  so searches stop returning it and the change can be undone. A memory that
  repeats another: `memorizz_update_memory` with its record ID and
  `duplicate_of` the one to keep (no new text).
- `memorizz_forget_memory` asks the user to approve the deletion in MemoRizz
  (`memorizz mcp approvals`, or the MemoRizz UI); it doesn't delete at once. Use
  it when the user asks, or for something that must not be kept (a secret saved
  by mistake).

## Load documents

`memorizz_ingest` loads files or folders (design docs, ADRs, runbooks, a
README) into the project's knowledge base, so later searches cite them. Use it
when the user asks, or offer it when the same documents keep coming up. It
skips hidden files, secrets and dependency folders.

## Check

`memorizz_memory_status` shows where memories are stored, whether semantic
search works, what the server allows, and how many memories this project has.
Use it when something seems missing.

## Where memories live

In the user's MemoRizz store (files under `~/.memorizz/memory` by default, or
MongoDB/Oracle as configured). The MemoRizz UI (`memorizz ui`) and CLI show
them, and the user's MemoRizz agents can use them too.
