---
description: Forget or correct something in this project's MemoRizz memory
argument-hint: <what to forget or correct>
---

The user wants this forgotten or corrected in the project's MemoRizz memory:
$ARGUMENTS

Find the matching memories (`memorizz_search_memories`, also in
"conversation_memory" if it's about an earlier session). Show them with their
record IDs. For a fact that is wrong or outdated, correct it with
`memorizz_update_memory` (the old one is kept, marked superseded). For one that
must not be kept at all, call `memorizz_forget_memory`; it asks the user to
approve the deletion in MemoRizz (`memorizz mcp approvals`), so tell them that.
Never forget a memory the user didn't point at.
