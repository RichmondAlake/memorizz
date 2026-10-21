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
