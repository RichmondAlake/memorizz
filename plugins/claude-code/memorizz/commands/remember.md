---
description: Save a fact to this project's MemoRizz memory
argument-hint: <fact to remember>
---

Save this to the project's MemoRizz memory: $ARGUMENTS

Use the project's memory ID from the MemoRizz session context. Search first
(`memorizz_search_memories`); if a memory already says this, say so and don't
save it again; if one says something different, correct it with
`memorizz_update_memory`. Otherwise save it with `memorizz_store_memory`
(memory_type "knowledge_base") as one self-contained fact, with its reason if
given. If it describes a person, service or system with attributes, use
`memorizz_upsert_entity` instead. Never save secrets. Reply with what you saved
and its record ID.
