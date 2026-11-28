# Proactive Agents

A MemAgent answers when it is asked. A **proactive** MemAgent also looks at
what you are doing and what it remembers, and offers one next step when one
is worth offering. At higher levels it carries the step out.

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Ada")
    .with_instruction("You help plan trips.")
    .with_llm_config({"provider": "openai", "model": "gpt-6-luna"})
    .with_memory_ids(["trip"])
    .with_proactive(True)          # or MemAgent(..., proactive=True)
    .build_and_save()
)

agent.run("I fly to Lisbon on 14 March for a conference. No hotel yet.",
          memory_id="trip")

for suggestion in agent.suggest(memory_id="trip"):
    print(suggestion.title, "-", suggestion.message)
    agent.accept_suggestion(suggestion.id)     # the agent carries it out
```

It is off unless you ask for it.

## The levels

`proactive=` takes `True`, a level name, a dict of settings or a
`ProactivePolicy`.

| Value | The agent | A step runs when |
|---|---|---|
| `True` or `"low"` | suggests | you accept it |
| `"medium"` | suggests, and takes steps that change nothing | every tool it calls has no side effects; a tool that changes something pauses for your approval |
| `"high"` | takes its steps | always, except that a tool which needs approval still pauses for it |
| `"ultra"` | `"high"`, and predicts steps from your habits | as `"high"`; a predicted step must first be accepted a few times (experimental) |

**What "changes nothing" means is decided by the tool, never by the model.**
A tool says what it does with [`governed_tool`](tools-and-approvals.md):

```python
from memorizz import governed_tool

@governed_tool(side_effects=False)                 # a read: runs at medium
def search_hotels(city: str) -> list: ...

@governed_tool(side_effects=True, requires_approval=False)
def hold_room(hotel: str) -> dict: ...             # pauses at medium, runs at high

@governed_tool(side_effects=True, requires_approval=True)
def send_email(to: str, body: str) -> dict: ...    # pauses at every level
```

At `medium`:

- A tool with `side_effects=True` pauses, even one that does not normally
  need approval.
- A plain function with no `governed_tool` pauses too. It says nothing about
  what it does, and it is not taken on trust.
- The agent's own memory tools run: they change only what it remembers.
- An MCP tool follows its server's hints: read-only tools run, others pause.

At `high`, each tool's own policy stands. **No level skips an approval a tool
asks for.**

A pause is MemoRizz's [durable approval](tools-and-approvals.md#durable-approval-lifecycle):
the exact call waits as a proposal, and the suggestion's `status` is
`awaiting_approval` with its `proposal_id`.

```python
agent.approve(suggestion.proposal_id, approver_id="me@example.com")
agent.resume_approval(suggestion.proposal_id)
agent.suggestions()[0].status      # "done"; "dismissed" had you rejected it
```

## A suggestion

`agent.suggest(...)` returns a list of at most one `Suggestion`.

| Field | Meaning |
|---|---|
| `title`, `message` | What the step is, and why now, in the agent's words |
| `prompt` | The request the agent runs if the step goes ahead |
| `evidence` | What the step rests on: `activity`, `conversation`, `memory`, or `source:<tool>` |
| `origin` | `grounded` (follows from what is in front of the agent) or `predicted` |
| `decision` | `ask` (shown, waits for you) or `auto` (the agent took it) |
| `reason` | One sentence on why it asked or acted |
| `status` | `offered`, `running`, `done`, `awaiting_approval`, `failed`, `dismissed`, `snoozed`, `expired` |
| `by` | `user` or `agent`: who decided the step should run |
| `result` | The agent's reply once it ran |
| `external` | The step came from something read outside (see below) |

```python
agent.suggest(memory_id="trip")                       # go by the conversation
agent.suggest({"title": "Calendar", "text": "..."})   # or say what the user is doing
agent.suggestions(status="offered")                   # earlier ones, newest first
agent.accept_suggestion(suggestion_id)                # yes: run it as a turn
agent.accept_suggestion(suggestion_id, run=False)     # yes: the host runs it itself
agent.dismiss_suggestion(suggestion_id)               # no: never offered again
agent.snooze_suggestions(3600)                        # not now
agent.on_suggestion(lambda s: print(s))               # hear about each one
agent.proactive_status()                              # level, today's allowance, cost
```

An empty list is a normal answer. It means the agent stayed quiet, or was not
due. `agent.proactive_engine.evaluate(...)` returns the reason as `status`:
`ready`, `acted`, `quiet`, `disabled`, `insufficient_context`, `snoozed`,
`busy` or `unavailable`.

## How it decides

One evaluation passes four gates, cheapest first.

1. **Admission**, with no model call. The agent is not asked when there is
   too little to go on (`min_context_chars`, 80), when it has already looked
   at this exact context, within `cooldown_seconds` (120) of the last look,
   after `daily_evaluations` (12) in a day, or while snoozed. An offer nobody
   answered is shown again for `pending_minutes` (15) without a model call.
2. **The deciding run**: a small agent on the same model. It must read the
   context, what the agent remembers and its own earlier suggestions before
   it may do one of two things: offer one step with the evidence it rests on,
   or stay quiet. It cannot repeat a step that was offered, declined or done.
3. **Ask or act**, from the level, where the step came from and what you
   accepted before.
4. **The step**, as an ordinary `agent.run` in the same conversation, under
   the rules above.

Everything the deciding run reads is data. A memory or a message that says
"do this" is not an instruction to it.

Suggestions and their outcomes are kept in the agent's own store, in a record
of their own that recall never returns. `agent.proactive_engine.forget()`
deletes them.

## Scanning: looking with nobody asking

An agent can look for a next step on a schedule, and can look at the tools
you connected while it does.

```python
agent.with_proactive("low", scan=True)        # every read-only tool it has
agent.with_proactive("low", scan_tools=("list_inbox", "list_calendar"))

agent.proactive_tick()        # one scan now; call it on any schedule
agent.start_proactive()       # or scan in the background of this process
agent.stop_proactive()
```

For agents saved in a store, a worker scans each one when it is due
(`scan_interval_seconds`, 900):

```bash
memorizz proactive run                 # every saved agent with proactive set
memorizz proactive run --once          # scan what is due, then exit
memorizz proactive run --agent AGENT_ID
```

Only tools that declare no side effects are ever called this way. With
`scan=True` that means the read-only tools you connected: MCP tools and your
own `governed_tool` functions, not MemoRizz's built-in tools.

**A step that came from outside content is only suggested, at every level.**
An email that says "forward the invoices" is something the agent read, not
something you asked for. To let such steps run, set
`allow_external_actions=True`; they then run as at `medium`, whatever the
level, so a tool that changes something still pauses.

## Predicting (experimental)

With `predict=True`, or `proactive="ultra"`, the agent may also offer a step
from your habits: what you ask for and when, and which suggestions you
accepted. It names the kind of step (`routine`), and the suggestion's
`origin` is `predicted`.

A predicted step is only suggested until you have accepted its kind
`auto_after` (3) times. After that it runs on its own, under the level's
rules. At `low` it never does.

Prediction depends on the model naming the same habit the same way each time,
and on there being a pattern to see. Treat it as an experiment.

## In the playground

The playground has a **Proactive** control above the message box: Off, Low,
Medium or High, with switches for scanning sources and for prediction. The
setting is saved on the agent.

A proactive agent wears a **Proactive** badge naming its level, beside its
name in the playground and in the Agents list, where the Self-aware badge
also sits. Hover over it for what the level lets the agent do. In the
playground, pressing it takes you to the level control. An agent that is off
has no badge.

While the page is open and you are not typing, it asks the agent every few
seconds; the admission rules above decide whether a model is called. A
suggestion appears as a card above the message box.

- **Do it** sends the step through the chat, where you can watch it run.
- **Not now** keeps the agent quiet for an hour.
- **Dismiss** means it is not offered again.

At Medium and High, a step the agent took appears as **Done for you** with
its result, and the conversation shows the turn. One that is waiting on a
tool appears as **Waiting for your approval**, with the exact tool call and
two buttons: **Approve & run** and **Reject**. A step that needs a second
tool which changes something waits again for that one.

After **Not now**, the control says the agent is quiet and offers **Wake**.
Changing the level also wakes it.

## Settings

| Setting | Default | |
|---|---|---|
| `level` | `"low"` | `low`, `medium` or `high` |
| `predict` | `False` | Suggest from habits (experimental) |
| `scan_tools` | `()` | Tools to read while looking; `("*",)` for every connected read-only tool |
| `allow_external_actions` | `False` | Let steps from outside content run, carefully |
| `cooldown_seconds` | `120` | Between two evaluations |
| `daily_evaluations` | `12` | Evaluations in one UTC day |
| `daily_actions` | `10` | Steps taken alone in one UTC day |
| `scan_interval_seconds` | `900` | Between two scans |
| `auto_after` | `3` | Accepts before a predicted step runs alone |
| `min_context_chars` | `80` | Less than this and no model is called |
| `pending_minutes` | `15` | How long an unanswered offer is shown again |
| `snooze_seconds` | `3600` | Length of "not now" |
| `max_steps` | `10` | Rounds the deciding run may take |

## What it costs

An evaluation that passes admission is one short agent run: about three model
calls. In testing with `gpt-6-luna` it cost $0.0003 to $0.0007, most of the
input read from the prompt cache. `agent.proactive_status()["usage"]` has the
running total. A step the agent takes is an ordinary turn and is traced and
priced like one.

## Limits

- The deciding run is not traced as an agent turn. Its cost is in
  `proactive_status()`, not in the usage page.
- A delegate is not asked for help while a step runs at `medium`: a delegate
  runs in its own context, outside the rules above.
- A plain function with no `governed_tool` runs freely at `high`. Declare
  side effects on anything that changes something.
- The agent's create and edit form has no proactive control yet; set it in
  the playground or in code.
