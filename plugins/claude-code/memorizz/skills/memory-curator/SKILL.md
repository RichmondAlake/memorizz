---
name: memory-curator
description: Tidy a project's MemoRizz memory - find duplicates, contradictions and stale facts, correct them, and propose deletions. Use when the user asks to clean up, review or audit memory, or when recalled memories disagree with each other or with the code.
---

# Curate project memory

Work on the project's memory ID (from the session context, or
`memorizz plugin memory-id`).

1. **Inventory.** `memorizz_memory_status` for counts, then
   `memorizz_list_memories` (`memory_type` "knowledge_base", a high `limit`) to
   read every current fact.
2. **Group** facts about the same thing (a command, a service, a decision).
   Within each group look for:
   - **duplicates**: the same fact saved twice;
   - **contradictions**: two facts that can't both be true;
   - **stale facts**: ones the code now disagrees with. Check the code, config
     or docs before calling a fact stale.
3. **Fix**, one change per call:
   - duplicates: keep the best-worded record (if another has detail it
     lacks, first `memorizz_update_memory` it with the merged text and keep the
     new record), then retire each other copy with `memorizz_update_memory`
     (`record_id` the copy, `duplicate_of` the kept record, no `content`);
   - contradictions: `memorizz_update_memory` the wrong one with the right
     fact, or retire it with `duplicate_of` the record that is right;
   - stale facts: `memorizz_update_memory` with the current truth and the reason
     ("changed in commit …" or "per pyproject.toml");
   - facts that should not exist at all (secrets, personal data, wrong project):
     `memorizz_forget_memory`, which asks the user to approve.
4. **Entities.** Facts that describe a person, service or system with
   attributes belong as entities: `memorizz_upsert_entity`, then supersede the
   loose fact.
5. **Report** what you merged, corrected and proposed for deletion, with record
   IDs, and anything you weren't sure about. Never delete or supersede on a
   guess; ask.
