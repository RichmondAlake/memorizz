# Memory consolidation and the background model

An agent writes memory one turn at a time. Over many conversations its store
collects entities recorded twice, the same fact under two names, values the
user has since changed, and things the user said that the agent never wrote
down. MemoRizz keeps the prompt clean at read time (see
[pre-inference deduplication](context-efficiency.md#pre-inference-deduplication)),
but the store itself keeps growing. Memory consolidation tidies the store.

Letta calls this kind of work *sleep-time compute*: a second agent edits
memory between conversations. Anthropic's Managed Agents call it *dreaming*.
MemoRizz does it as a consolidation pass, which usually runs on a separate
background model and never on the user's turn.

## The background model

Summaries, context compaction, reflection, LLM importance ratings, skill
distillation and [persona learning](persona-learning.md) are all memory upkeep.
They run on the agent's own model unless
you give it a background model, which is usually smaller and cheaper. The
user's turns keep the agent's model.

```python
from memorizz import MemAgent

agent = MemAgent(
    llm_config={"provider": "openai", "model": "gpt-5"},
    background_llm_config={"provider": "openai", "model": "gpt-5-mini"},
    memory_provider=provider,
)

# or later, or with a model instance:
agent.with_background_model({"provider": "ollama", "model": "qwen2.5:3b"})
agent.with_background_model(None)  # back to the agent's own model
```

`agent.background_model` is the model in use. A config dict is saved with the
agent, without its API key. A model instance is used for this process only.
If the background model can't be built, upkeep falls back to the agent's model
and a warning is logged. When you swap the agent's model, skill distillation
follows it, unless a background model is set.

## Upkeep that runs off the turn

Two upkeep tasks used to block a turn. Both now run on a background worker,
using the background model:

- **Context compaction.** When a thread is about to pass `compact_at`, it is
  compacted after the turn that took it there, not before the first token of
  the next one. The next turn of that thread waits for it to finish (at most
  `background_compaction_wait_seconds`, 120 by default) before loading
  history. If a turn passes `compact_at` anyway, it still compacts before its
  first token, as before.
- **Reflection.** When `MEMORIZZ_REFLECTION_ENABLED` is on, the turn that
  crosses the importance threshold schedules the summarisation pass and
  returns. The trace shows `Reflection scheduled`.

`agent.wait_for_memory_upkeep(timeout)` waits for upkeep to finish. `close()`
waits up to `memory_upkeep_close_timeout` (60 s) before it closes the provider.

## What a consolidation pass does

A pass covers one memory and one user. It first makes a dry-run plan:

| Step | Model | What changes |
|---|---|---|
| Duplicate entities | no | Records for the same entity are merged into one. |
| Duplicate rows | no | A summary with the same text as an older one in its thread, or near-identical to it (cosine ≥ 0.97), is hidden. With `knowledge_base=True`, so is a knowledge-base chunk that repeats another chunk in the same knowledge base and namespace. |
| Attribute review | yes | Within an entity, of two attributes about the same thing (one fact under two names, or an answer that changed), the older is retired. Two entities that are one person or thing under two names merge when they also share at least two facts. |
| New facts | yes | Durable facts the user stated in recent conversations, and the agent never stored, are added to entity memory. |
| Confirmations | yes | A stored value the user stated again in a later conversation is confirmed (`reinforce_fact`): its count goes up and nothing else changes. See [Confirmations](#confirmations). |
| Note relations | a decision model | With `relations=True`, the agent's own knowledge-base notes are compared with the closest older ones: the same fact said again is hidden, a note that a newer one refines or contradicts is hidden (superseded by it), and another fact about the same subject is linked. See [Decision Models](decision-models.md#consolidation-relations). |

Guards on the model's suggestions:

- **New facts must quote the user.** Each new fact must quote the user's own
  words, and the quote has to appear in a user message of that conversation.
  A fact that rests on something the assistant said, or on nothing, is
  dropped.
- **The newer fact stays.** The model only names the attributes that are
  about the same thing; which one goes is decided by when each was written.
  When two of them disagree and neither is newer, both stay.
- **Merges need evidence.** Two entities merge only when they share at least
  two facts word for word. A pass doesn't read the conversations of a sealed
  thread, or rows a sealed thread wrote.
- **Pinned and verified rows are never hidden.**

Nothing is deleted:

- Hidden rows are suppressed: `retention_state` is set, as in
  [governed forgetting](forgetting-mechanism.md#tier-1-governed-suppression-plan-approve-restore).
- Retired attributes move to the entity's `metadata.retired_attributes`.
- Merged duplicates get `superseded_by`.
- Added facts carry `source: consolidation:<plan id>`.
- Confirmations carry the same source in the attribute's `confirmed_by`.

Restoring a plan undoes all five.

```python
plan = agent.plan_memory_consolidation(memory_id, user_id="user-42", thread_ids=["t1", "t2"])
print(plan.counts)            # {'merge_entities': 1, 'suppress_duplicate': 0, ...}
for op in plan.ops:
    print(op.kind, op.reason, op.details)

result = agent.apply_memory_consolidation(plan, approved_by="operator-7")
agent.restore_memory_consolidation(plan.plan_id, approved_by="operator-7",
                                   memory_id=memory_id, user_id="user-42")
```

`user_id=None` is the anonymous scope, as everywhere else. Without
`thread_ids`, a pass reads the threads this agent served since its last pass.
It never reads a conversation it wasn't given, and never one that ran sealed
(see [sealed threads](multi-tenant.md#sealed-threads)). An agent whose
retrieval policy keeps entity facts per thread (`entity_scope="thread"`) gets
only the duplicate-row step.

## Source levels

A write to an entity attribute replaces the current value only when its
source is trusted at least as much. There are three levels.

| Level | Sources | Example |
|---|---|---|
| 3 | Pinned, or written by the host through the SDK | The host writes a limit from the policy system |
| 2 | Stated by a user in conversation, or through an MCP client | A handler says the limit is now 3,000 |
| 1 | Inferred by the agent, taken from tool output, or added by consolidation | A tool result mentions 2,500 |

At the same level the newer write wins, as before. A weaker write that
disagrees does not replace the value: it is kept in the attribute's history
with `applied: False`, and `upsert_entity_with_report` names the attribute in
`kept_stronger`. There is no queue and no review step; the entries that were
not applied are visible on the entity's page.

A direct call to `EntityMemory.upsert_entity` is host code, so it writes at
level 3 unless you pass `source_level`. The agent's own write paths pass
theirs: the model's entity tool and turn capture write at 2, the `remember`
tool and consolidation at 1. A value stored before levels existed is ranked
from what it says: pinned is 3, a source naming the conversation is 2, and
anything else is 1, so old values stay as replaceable as they were.

Attribute review follows the same rule: of two attributes about the same
thing, the one from the stronger source stays, whichever is newer.

!!! note "Changed behaviour"
    A value your application wrote through the SDK is no longer replaced by
    what a user says later, or by what the agent infers. To let conversation
    update a seeded value, seed it with `source_level=2`.

## Dated history

A replaced value moves to the attribute's `history` with `valid_from` (when
it became the current value) and `valid_to` (when it was replaced). The
history keeps 20 values per attribute (5 before); set
`EntityPolicy(attribute_history_limit=...)` to change that. Only the current
value is rendered into a prompt. `attribute_timeline(attribute)` returns the
values in order with their dates; for a value stored before dates were kept,
`valid_from` falls back to `updated_at`, so nothing is migrated.

### Two clocks

Every value is dated twice.

| Clock | Fields | Answers |
|---|---|---|
| Valid time | `valid_from`, `valid_to` | When was this true in the world? |
| Recorded time | `recorded_at`, `superseded_at` | When did the system learn it, and stop believing it? |

By default the two agree: a value is taken to be true from the moment it was
stored. A host that knows better says so on the write, and the two clocks
then tell a change from a correction:

```python
# A change: £2,500 was true until January, £3,000 from then on.
entities.upsert_entity(name="Total Payout Authority", memory_id=memory_id, attributes=[
    {"name": "grade_1", "value": "£3,000", "valid_from": "2027-01-01"}])

# A correction: £2,500 was never right. The new value covers the same period.
entities.upsert_entity(name="Total Payout Authority", memory_id=memory_id, attributes=[
    {"name": "grade_1", "value": "£2,750", "correction": True}])
```

- **A change** ends the old value's `valid_to` at the new value's
  `valid_from` (or at the time of the write when the host gave none).
- **A correction** keeps the old value in the history with
  `corrected: True` and no `valid_to`: it was believed from `recorded_at` to
  `superseded_at`, and was never true. A write whose `valid_from` equals the
  current value's explicit `valid_from` is treated as a correction too.
- **A backfill.** A write whose `valid_from` is earlier than the current
  value's explicit `valid_from` describes the past: it goes into the history
  for its own period and the current value stays.

`attribute_timeline` rows carry both clocks. To ask a point-in-time question:

```python
entities.value_as_of("grade_1", "2026-09-30", name="Total Payout Authority",
                     memory_id=memory_id)             # what was true then
entities.value_as_of("grade_1", "2026-09-30", known_at="2026-12-01",
                     name="Total Payout Authority",
                     memory_id=memory_id)             # what we believed in December about September
entities.value_as_of("grade_1", known_at="2026-12-01", name="Total Payout Authority",
                     memory_id=memory_id)             # the value we believed in December
```

Each returns a timeline row, or None. A corrected value is never returned by
valid time alone; with `known_at` before the correction, it is. The read also
covers values a consolidation pass retired (`entity_attribute_timeline(record,
"grade_1")` is the combined timeline). Records written before the second
clock read as before: `recorded_at` falls back to `valid_from` and
`superseded_at` to `valid_to`.

## Confirmations

A value stated again is recorded, not ignored. The attribute keeps
`confirmations` (how often it was stated again after it was first stored),
`last_seen` and the newest five in `confirmed_by` (`source`, `thread_id`,
`level`, `at`). A confirmation never changes the value, its source level or
its valid-from, and never replaces the stored vector.

| What happened | What is recorded |
|---|---|
| A weaker source states the value the record holds (a user repeats what the host wrote) | A confirmation. Before, nothing. |
| The same source in the same conversation repeats it | Nothing within an hour (`REINFORCE_QUIET_SECONDS`), so an agent that restates a fact every turn causes no write; after that, a confirmation |
| Another conversation at the same or a higher level states it | As before, the value takes the new provenance and the earlier one joins the history. It now also counts as a confirmation |
| Consolidation finds the user stating it again in a later conversation | A `reinforce_fact` operation in the plan, applied with the plan and undone with it |

```python
entities.reinforce_attribute("office", name="Counter Fraud Unit", source="audit:2027-01",
                             memory_id=memory_id)       # record one yourself
entities.withdraw_confirmations("audit:2027-01", memory_id=memory_id, user_id=None)
```

Forgetting a conversation removes the confirmations it gave. When a value is
replaced, the old value keeps its count in the history and the new one starts
at none.

## Stale references: when a fact changes

Superseding a value fixes the record. A skill distilled last month, or a
note the agent wrote, can still state the old value in its text. When an
entity value is replaced (by a turn, a host write or a consolidation pass),
the agent's skills and the notes in the same memory that still mention the
old value are flagged. Nothing is rewritten.

```python
for flag in agent.stale_references():
    print(flag["target_type"], flag["target_name"], flag["entity"], flag["attribute"],
          flag["old_value"], "->", flag["new_value"], flag["confidence"], flag["excerpt"])

agent.resolve_stale_reference(flag["flag_id"], by="dev")     # the text was fixed
agent.dismiss_stale_reference(flag["flag_id"], by="dev")     # not about this fact
agent.check_stale_references(memory_id)                      # sweep what is stored now
```

Matching is narrow on purpose:

- The old value has to appear as a whole phrase, not inside a longer word or
  number, and values under four characters are never looked for.
- A name-like value ("Leeds", "Nadia Hussain") is enough on its own and is
  flagged at `confidence: "low"`. A number, an amount or ordinary words also
  need the entity (its name or an alias) or the attribute named in the same
  text; those flags are `"high"`.
- A value that only grew ("Leeds" to "Leeds office") flags nothing.
- Skills are the agent's own, except deprecated ones. Notes are
  knowledge-base and summary rows with the entity's `memory_id`.

A flag is stored once per record and old value; a dismissed or resolved flag
is not reopened. Flagging happens as the value is replaced when an agent
that serves that memory is loaded in the process; `check_stale_references`
covers everything else. Set `agent.propagate_supersessions = False` to turn
the live flagging off. Each supersession reads the agent's skills and up to
2,000 rows of each note store.

## Aliases

When two records are merged, the names of the record merged away become the
survivor's aliases (`entity_aliases(record)`). A later write or lookup by one
of those names resolves to the survivor, after the identity key and the exact
name. Undoing the merge removes the aliases it added, and
`EntityMemory.set_aliases` lets an operator edit the list.

## Run it in the background

```python
agent.with_memory_consolidation(every_n_turns=20, every_hours=24)
```

After each turn the agent counts the turns of that memory and user. When
`every_n_turns` turns have passed, or `every_hours` hours with at least one
turn, it plans a pass on the background model, off the turn. `auto_apply`
(on by default) applies the plan as `memorizz:consolidation`. Turn it off to
keep the plan for review instead:

```python
agent.with_memory_consolidation(every_n_turns=20, auto_apply=False)
...
for plan in agent.pending_memory_consolidations():
    agent.apply_memory_consolidation(plan.plan_id, approved_by="operator-7")
```

Pending plans live in the agent process. The schedule is saved with the agent
(`memory_consolidation`), and the builder has `with_memory_consolidation()`
and `with_background_model()`. Other settings: `review_attributes` and
`extract_facts` turn the two model steps on and off, and `max_entities` (40),
`max_threads` (5) and `max_messages_per_thread` (40) bound how much a pass
reads.

## CLI and local UI

In the CLI:

- `/background-model` shows the background model. `/background-model qwen2.5:3b`
  or `/background-model openai:gpt-5-mini` sets it, and `off` turns it off.
- `/consolidate` reviews the memory in use and lists every change. It applies
  them only after you answer yes. `/consolidate kb` also checks the knowledge
  base, and `/consolidate undo <plan id>` restores a plan.

For operators:

```sh
memorizz learning consolidate --agent-id AGENT --memory-id MEMORY --user-id USER \
  --thread-id t1 --thread-id t2 --json                    # dry run
memorizz learning consolidate ... --apply --approved-by operator-7
memorizz learning consolidation-restore PLAN_ID --agent-id AGENT --memory-id MEMORY \
  --approved-by operator-7
```

`--model provider:model` picks the model. Without it, the command uses the
agent's background model, else its chat model. `--no-model` runs only the
steps that need no model.

In the local UI:

- **Agent form.** *LLM configuration* has the **Background model** picker
  (a provider and a model, or "Same as the chat model"). *Conversation privacy
  and memory upkeep* has **Seal each conversation**, where **Entity facts**
  live, and **Consolidate memory in the background** with its turn count and
  whether changes apply automatically.
- **Continual learning → Control plane → Memory consolidation.** Plan a pass
  for one memory and user (blank conversations means its most recent ones),
  review every change with the quote it rests on, apply it with an approver,
  and undo any applied plan from *Applied plans*. Plans the background
  schedule left for review are under *Waiting for review*. Hidden duplicate
  rows also appear in *Suppressed memories*, where each can be restored on its
  own.
- **Playground → Memory → How this agent remembers.** The agent's memory
  types, where its entity facts live, whether conversations are sealed, its
  memory tools, the background model and the consolidation schedule.
- **Memory evolution.** Every change a plan made is a card labelled with what
  it did ("Merged duplicates of Ada Lovelace", "Retired user.favorite_drink",
  "Added user.city", "Hid a duplicate summary"), with the approver as writer
  and *Memory consolidation · plan …* as source. Undoing a plan adds
  "Undid memory consolidation" cards. *Change → Memory consolidation* shows
  only these.

The schedule is kept per agent and memory provider in the process. A server
that loads a fresh agent for every request, as the local UI does, keeps one
count of turns, one set of plans waiting for review and one compaction per
thread.

## Cost

A pass makes one model call per eight entities reviewed and one per
conversation read, all bounded by the limits above. The duplicate checks need
no model. A small local model handles both prompts; the prompts and their
JSON replies are short.
