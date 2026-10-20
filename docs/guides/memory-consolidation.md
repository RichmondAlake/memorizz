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

Summaries, context compaction, reflection, LLM importance ratings and skill
distillation are all memory upkeep. They run on the agent's own model unless
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

Restoring a plan undoes all four.

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
