# Memory Routing and Reranking

Two models sit behind an agent's memory besides the one that answers:

- **The memory router** decides where something the agent should keep
  belongs: the knowledge base, the facts about a person or thing, working
  notes, or nowhere.
- **The reranker** reorders what the agent recalls before the model sees it.

Both are set per agent, in code or on the agent form (*Model → Memory
models*), and saved with the agent.

## Memory routing

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-6.1-sol"})
    .with_memory_provider(provider)
    .with_memory_routing("openai_decisions")
    .build()
)
```

Or `MemAgent(..., memory_routing="openai_decisions")`, or
`agent.with_memory_routing("auto")` on a live agent.

With routing on:

- **Each user message is routed after its turn**, on the memory upkeep
  worker, so replies are not slower. A message that states a lasting fact
  (even inside a request, like *"My sister Ines is allergic to peanuts. Any
  dessert ideas?"*) is filed for the fact; small talk files nothing.
- **The chat model stops spending tool calls on bookkeeping.** Its prompt
  says facts are saved for it; it calls `entity_memory_upsert` only to
  correct a stored fact.
- **`agent.remember(text)`** files one memory on demand, and the model gets
  a `remember` tool for facts it finds itself (in a document, a tool result,
  a calculation).

Where a routed memory goes:

| The router says | Filed as |
|---|---|
| `entity_memory` | Facts extracted by the background model, each resting on a phrase copied from the text, upserted on the person or thing (the user is `user`). Values already stored are skipped. |
| `knowledge_base` | A knowledge-base entry in the namespace `remembered`, keyed to the agent's memory, so recall finds it. |
| `short_term_memory` | A working note in short-term memory. |
| `do_not_store` | Nothing. |
| `personas` and other stores | Nothing during capture; `remember` keeps it in the knowledge base and its receipt says what would file it properly (`update_persona`, for instance). |

The conversation itself is always recorded, so the router never chooses
conversation memory, summaries or tool logs. Below `min_confidence` (0.5)
capture files nothing, and `remember` keeps the text in the knowledge base.

A sealed conversation (`retrieval_policy="sealed"`) never writes to shared
stores: its facts go to the conversation's own entity scope, and nothing
reaches the knowledge base.

`remember` returns a receipt:

```python
agent.remember("Company policy: expense reports are due by the 5th.")
# {'stored': True, 'memory_type': 'knowledge_base', 'routed_to': 'knowledge_base',
#  'confidence': 0.92, 'router': 'openai_decisions/gpt-6-luna',
#  'reason': 'Routed to knowledge_base.', 'knowledge_base_id': '...', ...}
```

Every routed write is recorded in the memory history journal with the source
`memory_routing:turn`, `memory_routing:tool` or `memory_routing:api`.
Entity capture (below) records `entity_capture:turn`.

### Routers

| Router | What answers | Needs |
|---|---|---|
| `"auto"` | A router fine-tuned in MemoRizz if there is one (the agent's own choice on the fine-tuning page, else the store's), else OpenAI Decisions with an OpenAI key, else the background model | nothing extra |
| `"openai_decisions"` | OpenAI's `/v1/decisions` with `gpt-6-luna` (or `model=`): a typed choice over the stores, no training. About 460 input tokens, roughly $0.00005, per memory | `OPENAI_API_KEY` |
| `"llm"` | The agent's background model, asked the same question | a background or chat model |
| `"finetuned"` | A router trained on the fine-tuning page (`run_id=`), else the agent's or store's | a trained run |

Settings, as keyword arguments or a dict:

```python
agent.with_memory_routing(
    "openai_decisions",
    model="gpt-6-luna",
    capture_turns=True,      # route each user message after its turn
    min_confidence=0.5,      # below this, capture files nothing
)
agent.with_memory_routing(False)   # off
```

Entity facts are extracted by the background model; set one with
`with_background_model()` to keep that off the chat model.

### Entity capture without routing

An agent with entity memory and its own background model captures entity
facts even with routing off. After each turn, the background model reads the
user's message and files the facts it states about the user and named people
or things. No router is asked. The same guards apply: each fact rests on a
phrase copied from the message, and values already stored are skipped.

So entity writes no longer depend on the chat model choosing to call
`entity_memory_upsert`. A bare list such as *"Numbered facts: 1. … 2. …"*
is captured like any other message. The chat model's prompt says facts are
saved for it, as it does with routing.

A long message is read in parts of up to 4,000 characters, cut at a line or
sentence end, one background-model call each, up to 8 parts (32,000
characters). When two parts give the same entity and attribute, the later one
wins, as a later line of the message would. Anything past 8 parts is left
unread, and the capture result's `unread_chars` says how much.

When the chat model does call `entity_memory_upsert` in the same turn, its
write stands: capture skips the same entity and attribute, and values the
chat model already stored. Routing's turn capture does the same.

This is a change from earlier releases, where facts were only kept when the
chat model called the tool. It costs one background-model call per user
message. Turn it off with:

```python
agent.with_entity_memory(True, capture=False)
# or MemAgent(..., memory_routing={"entity_capture": False})
```

### Off the turn, and only when worth it

Capture, routing, compaction, reflection and consolidation never run on the
user's turn: the reply returns, and the work runs on the agent's memory
upkeep threads (`agent.wait_for_memory_upkeep()` waits for it, `close()`
drains it). One agent usually serves every user of a server process, so there
are up to 4 such threads (`MEMORIZZ_UPKEEP_WORKERS`, 1 to 16). Each user's
messages are still filed in order, and each thread compacts one pass at a
time.

A message with nothing to file costs nothing: an acknowledgement or greeting
("thanks", "ok, go on"), a single word or a bare link is skipped before any
model call. For text the user did not write themselves, such as another
agent's request, make the turn teach memory nothing (capture and routing are
skipped, and the model's own `entity_memory_upsert` and `remember` calls are
refused):

```python
agent.run(request_from_another_agent, user_id="u1", memory_capture=False)
```

`capture=True` captures on the chat model when there is no background
model. With routing on and `capture_turns=True`, routing files the facts, so
entity capture does not run as well. With `capture_turns=False`, entity capture
stays off unless you set `capture=True`. The setting is saved with the agent.
`agent.describe_memory()["entity_capture"]` says whether it runs.

### How much code it saves

Filing five notes by hand means choosing each store and structuring each fact
yourself: an `EntityMemory.upsert_entity` call per person with its attributes
spelled out, a `KnowledgeBase.ingest_knowledge` call for the policy, a
`provider.store` call for the note, and a decision to drop the small talk.
That took 25 lines in our test. With routing it is two:

```python
for note in notes:
    agent.remember(note)
```

Both stored the same four facts, the policy and the note.

## Reranking

```python
agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-6.1-sol"})
    .with_memory_provider(provider)
    .with_reranker("voyage")  # or "voyage:rerank-2.5-lite", "openai_decisions", "off"
    .build()
)
agent.with_reranker("cohere")  # change it on a live agent
```

| Setting | Reranker | Needs |
|---|---|---|
| `None` (default) | Follow the store: the agent's choice on the fine-tuning page, else `MEMORIZZ_RERANKER_MODEL`, else none | |
| `"voyage"` | Voyage `rerank-2.5` (or `rerank-2.5-lite`) | `VOYAGE_API_KEY` |
| `"cohere"` | Cohere `rerank-v3.5` | `COHERE_API_KEY` |
| `"openai_decisions"` | `gpt-6-luna` answers "is this relevant?" for every candidate, in one request | `OPENAI_API_KEY` |
| `"finetuned:<run id>"` | A reranker or relevance model trained in MemoRizz | a trained run |
| `"off"` | None | |

The reranker scores the candidates recall found, and its score replaces the
relevance part of the ranking; recency and importance still apply. A hosted
reranker adds one request (about 0.2 to 0.5 s) to each turn that recalls
memories; when it fails, recall keeps its own order. Each reranked candidate's
score is in the turn's memory selection trace (`rerank_score`).

MCP searches in a memory an agent owns use that agent's reranker, and MCP
writes with `memory_type="auto"` into its memory use its router.

## On the agent form

*Model → Memory models* holds the background model, the reranker and memory
routing. The reranker list shows the store's current choice, the hosted
rerankers (marked when their key is missing) and the rerankers fine-tuned in
this store; the router list shows the same for routers. *Route what the user
says after each turn* is `capture_turns`. The Playground's *How this agent
remembers* shows both.
