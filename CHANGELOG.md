# Changelog

## 0.0.44 — 2026-05-24

### Bug fixes

* **MongoDB provider:** `retrieve_by_query` with a string query on
  `CONVERSATION_MEMORY`, `KNOWLEDGE_BASE`, or `SHORT_TERM_MEMORY` no
  longer crashes with `filter must be an instance of dict`. Two new
  helpers — `find_similar_conversation_entries` and
  `find_similar_knowledge_base_entries` — implement proper Atlas
  `$vectorSearch` over per-turn embeddings (mirroring the existing
  `find_similar_cache_entries`). `SHORT_TERM_MEMORY` string queries
  degrade to `[]` for now (no semantic index yet).
* **MemAgent runtime:** `MemoryManager.retrieve_relevant_memories` now
  normalises provider return values — `None` → `[]`, PyMongo cursors
  are materialised — so the `len(results)` debug log can no longer
  raise `'NoneType' has no len()` and mask the real error. Every
  branch in the MongoDB provider's `retrieve_by_query` now returns a
  list rather than `None`.
* **MongoDB TOOL_LOG persistence:** `store()` no longer strips
  `thread_id`, `memory_id`, and `agent_id` when writing tool-log rows,
  so per-thread and per-agent audit views work end-to-end. Existing
  rows that were written without these fields are unchanged; new
  rows carry the full context.
* **MemAgent prompt building:** `_prepare_history_messages` now drops
  `role: "tool"` entries pulled back from `conversation_memory`. The
  OpenAI Chat Completions API rejects any `tool` message that isn't
  immediately preceded by an `assistant` message carrying matching
  `tool_calls` metadata — which stored history doesn't preserve.
  The tool placeholder text is already embedded in the persisted
  assistant turn, so no information is lost.

These changes ship the fixes verified end-to-end against the
Speechlyze MemAgent integration (analyses, docs, content generation,
TalkThrough).
