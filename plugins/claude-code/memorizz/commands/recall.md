---
description: Recall what MemoRizz remembers about a topic, or about this project
argument-hint: [topic]
---

Recall from the project's MemoRizz memory (the memory ID is in the session
context): $ARGUMENTS

If no topic was given, summarize what memory holds for this project: the
latest facts (`memorizz_list_memories`), the last session summaries
(`memory_type` "summaries") and the known entities. Otherwise search the
knowledge base, then summaries and conversation memory for earlier sessions,
and look up matching entities. Quote what you found with its date, say which
memories may be stale, and don't add facts memory doesn't contain.
