# Persona learning

A persona changes when someone asks for a change: the agent calls
`update_persona`, or you edit it on the agent form. Persona learning adds the
slower kind of change. On a schedule, the agent's background model reads its
recent conversations and memories and looks for a lasting pattern, such as a
user who keeps bringing it the same kind of work. When it finds one, it adds a
sentence to the persona's goals or background.

It's off by default. Turn it on for an agent that serves one person.

```python
from memorizz import MemAgent
from memorizz.long_term.semantic.persona import Persona

agent = MemAgent(
    llm_config={"provider": "openai", "model": "gpt-5"},
    background_llm_config={"provider": "openai", "model": "gpt-5-mini"},
    memory_provider=provider,
    persona=Persona(name="Zed", role="Harbour pilot"),
    persona_learning=True,
)
agent.save()
```

The agent has to be saved, because what learning does is kept on the saved
agent. After a few conversations about a bakery, its goals could gain:

> Advise small bakery owners on orders, staffing, waste reduction and pricing.

The change takes effect on the next turn, and new conversations pick it up. The
agent's name and role stay as they were.

## Where to set it

| Where | How |
| --- | --- |
| Python | `MemAgent(persona_learning=...)`, `agent.with_persona_learning()`, `MemAgentBuilder().with_persona_learning()` |
| Local UI | Agent form, **Persona**, **Learn from conversations** |
| Local UI | **Learning control plane**, **Persona learning**: review now, approve, turn down, undo |
| CLI | `/persona-learning on`, `/persona-learning on review`, `/persona-learning off` |

The playground's **How this agent remembers** panel shows the schedule and any
change waiting for review.

## Settings

Settings are those of `PersonaLearningConfig`:

| Setting | Default | What it does |
| --- | --- | --- |
| `every_n_turns` | 20 | Review after this many turns. |
| `every_hours` | 24 | Or after this many hours, whichever comes first. `0` turns this off. |
| `auto_apply` | `True` | Apply a change straight away. `False` makes it wait for review. |
| `min_conversations` | 3 | How many conversations a proposal must quote. The lowest allowed is 2. |
| `max_threads` | 8 | How many recent conversations a review reads. |
| `max_messages_per_thread` | 40 | How many messages it reads from each. |

Reviews run on the [background model](memory-consolidation.md#the-background-model),
or the chat model if the agent has none. The turn count restarts with the
process, and the hours don't, so short sessions still reach a review.

## What a review reads

- The user's own messages in this agent's recent conversations.
- Summaries of those conversations.
- Entity facts, as context. They can't count as evidence.

Each review costs one model call. It makes no call when there's nothing to do:
too few conversations, nothing new since the last review, or a change already
waiting.

## What can change

The persona goes into the system prompt of every turn, so the model only
proposes and the code decides what goes in.

- Only `goals` and `background` grow, by one sentence a review. Nothing already
  written is rewritten, and each field stops growing at 1,500 characters.
- A proposal has to quote `min_conversations` different conversations word for
  word. The code finds each quote itself, in a conversation that shares a word
  with the sentence.
- The sentence has to describe a kind of work. It's refused if it holds a rule
  ("always", "never"), a recommendation, contact details, a person's name, a
  company's name, or anything aimed at the model.
- A message telling the agent who to be isn't read at all, for example "your
  goal is to..." or "add this to your persona". It might be text pasted from
  somewhere else. To change the persona on purpose, ask the agent to, which
  uses `update_persona`, or edit it on the form.

## Who it learns from

A persona is shared by everyone who talks to the agent, so one person's words
shouldn't reach it for someone else to see:

- An agent with `multi_tenant=True` never learns.
- A memory holding another person's conversations with this agent is skipped.
- An agent learns from the first person it reviews. Others are skipped.
- A request-local persona (`use_snapshot`) is left alone. Hosts use the
  [account-owned flow](../persona-evolution-hosts.md) for per-account personas.
- Sealed conversations are never read, and other agents' conversations in a
  shared memory don't count.

## Reviewing and undoing

With `auto_apply=False`, a proposal waits on the saved agent, so it survives a
restart:

```python
review = agent.review_persona(memory_id)   # never changes the persona
if review.proposal:
    print(review.proposal.addition, review.proposal.evidence)
    agent.apply_persona_proposal(approved_by="richmond")
    # or: agent.dismiss_persona_proposal(approved_by="richmond")

agent.undo_persona_change(approved_by="richmond")
agent.persona_learning_status()
```

Every change is a new persona version in `evolution_history`, made through
`Persona.update`, the same call `update_persona` makes. Its `change_trigger`
holds the quotes, who approved it, and `source_type="persona_learning"`.

- The quotes stay in the history. The system prompt shows a reason with no
  user words in it, and `read_persona` leaves them out.
- A sentence you undo or turn down isn't proposed again.
- Undo moves history forward: it writes a new version.
- If someone has edited a field since, undo refuses rather than losing the edit.
- If the persona changes while a proposal waits, the proposal is dropped.

## Limits

- The patterns it picks up are about what the user does, and that can be
  private even without a name. In testing, three conversations about a child's
  asthma sometimes led to a sentence about asthma care. Use `auto_apply=False`
  if that matters for your agent.
- The checks on messages and sentences are word lists. They stopped every
  injection tried in testing, but they can't catch every wording. Treat the
  persona as text the agent's user can influence, which is why the system prompt
  tells the model a persona grants no new authority.
- It only adds. To drop a sentence that has gone stale, undo it or edit the
  persona.
- A change applied from the UI reaches a CLI session already running when it
  next loads the agent.
