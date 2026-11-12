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

## Where to see it

In the local UI, **Continual learning → Persona evolution** lists every saved
agent that has a persona under "Agent personas": the version, whether
learning is on, any change waiting for review, and each change with the
user's words it rests on and who applied it. Reviewing, applying and undoing
a change is on the control plane's Persona learning section. A
[refine cycle](refine-cycle.md) runs persona learning as its sixth job and
lists the addition under "What changed".

A small background model can propose a change and still have it refused,
because every quote must be found word for word in a different conversation.
If reviews keep ending in "rejected", give the agent a stronger
`background_model`.

## Guardrails

A persona's goals and background are free text, and learning adds to them.
Guardrails are the rules that must not drift: a list kept apart from the
rest, which no learning path can write.

```python
persona = Persona(
    name="Zed",
    role="Harbour pilot",
    guardrails=[
        "Never disclose another vessel's manifest.",
        "Quote the tide table before giving a berthing time.",
    ],
)
agent.set_persona_guardrails([...], reason="Policy update")   # later, from host code
```

- They are rendered first in the system prompt, as a numbered list, before
  the role and goals. Earlier instructions are followed more reliably.
- Only host code (`Persona(guardrails=...)`, `agent.set_persona_guardrails`)
  or an operator (the local UI, `memorizz learning guardrails`) can change
  them. The model's `update_persona` tool, persona learning and the
  [refine cycle](refine-cycle.md) are refused: `Persona.update` raises
  `GuardrailWriteRefused` for a `guardrails` key, and
  `Persona.set_guardrails` accepts only a `change_trigger.source_type` of
  `host`, `manual` or `ui_form`.
- A persona holds at most 12, each at most 240 characters, so the list stays
  short enough to be followed.
- A change writes a new persona version, like any other persona edit.

In the local UI, *Memory → Personas* shows the list above the goals, marked
"Not writable by learning", with an editor. Over MCP, `memorizz_get_agent`
returns them and no tool can change them.

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
| `across_users` | `False` | Read every user's conversations with the agent, and require a proposal to quote several people. See [Learning from a team](#learning-from-a-team). |
| `min_users` | 2 | With `across_users`, how many different people a proposal must quote. The lowest allowed is 2. |
| `standing_rules` | `False` | Let a review propose a standing rule from corrections people keep making. See [Standing rules](#standing-rules). |
| `min_corrections` | 2 | How many conversations a rule's corrections must come from. The lowest allowed is 2. |
| `rule_owners` | empty | The people who may approve a standing rule. Empty means any named person. |

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

## Learning from a team

By default a persona learns from one person. A desk that shares one assistant
wants the opposite: what several handlers keep asking for, and not one
person's taste. `across_users=True` turns the review round:

```python
agent = MemAgent(
    ...,
    persona_learning={"across_users": True, "min_users": 2},
)
```

- The review reads the conversations of every user of this agent in the
  memory it is given, taking each person's most recent in turn so one busy
  user can't fill the list.
- A proposal has to quote `min_conversations` conversations **and** at least
  `min_users` different people. Three conversations from one person are not a
  pattern of the desk, and cost no model call.
- Each quote in the evidence keeps the user id and the conversation it came
  from.
- One person's summaries and entity facts are left out of the review. Their
  names are still read, only to keep them out of the persona.
- The rules under [Who it learns from](#who-it-learns-from) about a shared
  memory and a first owner don't apply, because sharing is the point. An agent
  with `multi_tenant=True` is reviewed too when `across_users` is on, except
  by the refine cycle, which still leaves such an agent alone.

The review reads one memory. Users whose conversations with the agent are in
different memories are not brought together.

## Standing rules

A persona has three tiers, and they differ in who may write them:

| Tier | Example | Who writes it |
| --- | --- | --- |
| **Guardrails** | "Never give a customer a coverage decision." | Host code or an operator. Learning never can. |
| **Standing rules** | "In customer updates, say a referred claim is under review." | Learning proposes; a named owner approves. Or the owner adds one by hand. |
| **Goals and background** | "Draft plain-English customer updates." | Learning, one sentence at a time. |

A standing rule is a house rule about how the work is done. Reviews refuse a
rule-shaped sentence for `goals` ("a rule or recommendation, not a kind of
work"), and they still do. With `standing_rules=True` the same review also
looks for a **correction** people keep making to what the assistant produced,
and proposes it as a rule:

```python
agent = MemAgent(
    ...,
    persona_learning={
        "standing_rules": True,
        "across_users": True,          # optional: several people must make it
        "rule_owners": ["Dev", "Callum"],
    },
)

review = agent.review_persona("claims-desk")
for proposal in agent.persona_rule_proposals():
    print(proposal["rule"], proposal["evidence"])

agent.apply_persona_rule("Dev")                      # the oldest one waiting
agent.apply_persona_rule("Dev", proposal_id, text="…")   # in the owner's words
agent.dismiss_persona_rule("Dev", proposal_id)       # never proposed again
agent.add_persona_standing_rule("Give dates in full.", "Callum")
agent.remove_persona_standing_rule("Give dates in full.", "Callum")
agent.undo_persona_rule("Dev")                       # the latest rule change
```

What the code checks before a rule is proposed:

- Each quote is a line a user wrote **after the assistant had replied** in
  that conversation, and shares a word with the rule. A first request is not
  a correction, however it is worded.
- The corrections come from at least `min_corrections` conversations, and
  from `min_users` people when `across_users` is on.
- A message telling the agent what to be ("add to your persona: …") is never
  read, as before, so it can't be quoted.
- The rule is one sentence of at most 200 characters. It is refused if it
  recommends or sells something, sends or reveals information (unless it
  forbids that), holds contact details or wording aimed at the model, names a
  person memory knows, or repeats a rule or guardrail the persona has.
- It is refused if it goes against a guardrail: it shares most of a
  guardrail's words and says the opposite, or carves an exception out of it
  ("unless", "except", "even if"). A contradiction written in different words
  is not detected, which is one reason a person approves every rule.

What happens to a proposed rule:

- **It is never applied on its own.** `auto_apply` covers goals and
  background only. A rule waits on the stored agent until a named person
  approves it; an empty name, an automatic approver and anyone outside
  `rule_owners` are refused, and the proposal keeps waiting.
- Approving writes a new persona version with the approver and the quotes.
  `undo_persona_rule` writes another version that takes it back, and the rule
  is not proposed again.
- A persona holds at most 8 standing rules. The model's `update_persona` tool
  cannot write them.

In the system prompt, standing rules come after the guardrails and before the
role and goals, under a heading that says a guardrail outranks them.

A rule review is one more model call per review, made only when at least
`min_corrections` conversations hold a line written after a reply. Up to five
proposals wait at once. `persona_learning_status()` lists the rules in force,
those waiting, and every rule change.

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
- "A correction" is judged by position and wording: a line after a reply that
  the model says corrects the output. A follow-up request can pass for one.
  The repeat across conversations, the word checks and the owner's approval
  are what stand between that and the prompt.
- Standing rules are an SDK feature for now: the local UI, the CLI and the MCP
  server show a persona's guardrails and goals but not its standing rules or
  the proposals waiting. In a refine cycle the persona job proposes rules
  with the rest of the review, and the cycle's report lists only the goals
  sentence.
- Standing rules have been run on the filesystem provider. The Oracle
  provider stores a persona in named columns and has none for them yet, so a
  rule would not survive there; MongoDB has not been run.
- A change applied from the UI reaches a CLI session already running when it
  next loads the agent.
