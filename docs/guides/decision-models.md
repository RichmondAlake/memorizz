# Decision Models

Several memory decisions are usually made by a similarity score and a fixed
threshold: whether a cached answer fits a new question, which tool a request
needs, whether two memories say the same thing. Near misses score as high as
real matches, so no threshold gets them all right. A **decision model**
answers the question directly instead, with a probability for every answer:
"can this cached answer serve this question?", "which of these tools does
this request need, if any?", "does the new memory contradict the stored
one?".

MemoRizz can ask one in seven places. Each is off until you turn it on, and
each falls back to what MemoRizz does without one when the model fails.

| Where | Turn it on with | The question |
|---|---|---|
| Before fact extraction | `MemoryRoutingConfig(extraction_gate=True)` | Is anything in this message worth keeping, and what kind? |
| Before recall | `RetrievalPolicy(read_routing=True)` | Does this request need episodic, semantic or procedural memory? |
| The semantic cache | `SemanticCacheConfig(verifier=True)` | Can the cached answer serve this question as asked? |
| Writing a note | `MemoryRoutingConfig(dedupe_writes=True)` | How does it relate to the closest stored note? |
| Consolidation | `ConsolidationConfig(relations=True)` | The same for pairs of stored notes |
| Tool routing | `ContextPolicy(tool_selection=True)` | Which tool does this request need, if any? |
| Chunking | `chunking_strategy="decision"` | Does a new topic start at this sentence? |

Two older settings take one too: a reranker (`reranker="d1"`, `"jev"` or
`"liquid"`, asking "does this memory help answer the request?") and
importance ratings (`scoring={"importance": {"mode": "decision"}}`, asking
"is this worth keeping?").

## Choose the model

`True` on a feature uses the agent's decision model:

```python
from memorizz import MemAgent
from memorizz.retrieval import RetrievalPolicy
from memorizz.tooling import ContextPolicy

agent = MemAgent(
    model=llm,
    memory_provider=provider,
    decision_model="d1",
    context_policy=ContextPolicy(tool_selection=True, stable_tool_list_tokens=0),
    retrieval_policy=RetrievalPolicy(read_routing=True),
    memory_routing={"extraction_gate": True, "dedupe_writes": True},
    semantic_cache=True,
    semantic_cache_config={"verifier": True},
)
```

Or `agent.with_decision_model("jev")`, or `.with_decision_model(...)` on the
builder. The setting is saved with the agent. On the agent form, *Model →
Memory models → Decision model* picks it, with a switch for each feature that
asks it; models with a missing key, or d1 without torch, are marked. A feature can name its own
model instead of `True`: `RetrievalPolicy(read_routing="d1")`.

| Decision model | Runs | Needs |
|---|---|---|
| `"d1"` | `LiquidAI/d1-2B` on this machine, about 4 GB, around 100 ms a question on a laptop GPU | torch and transformers (`memorizz[huggingface]`) |
| `"liquid"` | Liquid's hosted `d1.1` | `LIQUID_API_KEY` |
| `"jev"` | TypeSafe's Jev, `jev-1.14.0` | `TYPESAFE_API_KEY` |
| `"openai_decisions"` | OpenAI's Decisions API, `gpt-6-luna` | `OPENAI_API_KEY` |
| `"finetuned:<run id>"` | A decision model fine-tuned in MemoRizz | the run, in this store |
| `"llm"` | The agent's background model, asked for the probabilities as JSON | a background or chat model |
| `"auto"` | OpenAI Decisions with an OpenAI key, else the background model | |
| a function | Your own `(state, questions) -> answers` | |

Add a model after the colon to pick another: `"d1:<model id>"`,
`"jev:jev-latest"`, `"openai_decisions:gpt-6-luna"`. The d1 model runs its
own code (`trust_remote_code`); the default model is pinned to the revision
MemoRizz was tested with. Check its licence (LFM Open License v1.0) before
using it at work.

Each probability is on its own model's scale. A 0.5 line that suits d1 can be
too strict or too loose for another, so every feature has a threshold.
Measure on your own examples before you rely on one.

### Your own model

A function gets the state and the questions in the System One shape, and
returns an answer for each question:

```python
def my_model(state, questions):
    answers = {}
    for name, question in questions.items():
        if question["type"] == "noul":          # yes or no
            answers[name] = {"type": "noul", "noul": 0.8}   # P(yes)
        else:                                   # one of question["criteria"]
            options = list(question["criteria"])
            answers[name] = {"type": "choice", "probabilities": {o: 1 / len(options) for o in options}}
    return answers

agent.with_decision_model(my_model)
```

A function is saved by name only, so pass it again when the agent is loaded
in another process.

`memorizz.decision_models` has the questions MemoRizz asks (`DURABLE`,
`READ_ROUTES`, `CACHE_SERVES`, `RELATION`, `tool_question`, `CHUNK_BOUNDARY`,
`RELEVANT`, `KEEP`) and a `Decider` to ask your own:

```python
from memorizz import decision_models as dm

d1 = dm.decider("d1")
answers = d1.decide("Send the Q3 report to Sarah", {
    "tool": dm.tool_question({"generate_report": "Create a report", "send_email": "Send an email"}),
})
answers["tool"]["choice"]   # "send_email"
```

## Before fact extraction: the gate

Entity capture and memory routing read every user message with the
background model. The gate asks first whether the message holds anything
worth keeping after the conversation (a lasting preference, a fact, a
procedure, a dated event), and what kind. Below `gate_threshold` (0.5)
nothing more is done with it: no extraction call, no router call.

```python
agent.with_memory_routing({"extraction_gate": True, "gate_threshold": 0.5})
```

Given alone, the gate leaves routing off and works with entity capture. In
the decision-model notebook it skipped 23 of 60 extraction calls, about a
fifth of the cost, and stored nothing from filler. A live run skipped
"thanks, that's great" and "lol ok" (P(keep) 0.03) and kept "To deploy, run
the tests, tag the release, then post in #releases" (0.95, a procedure).

### Procedures

Extraction also lists procedures: how the user wants a kind of task done.
Each must rest on words copied from the message. A procedure is kept where
the agent reads procedures:

- **as a skill** of that user (active, `stated_by_user`), when the agent
  recalls skills (`skill_retrieval=True` or continual learning);
- otherwise as a note in the knowledge base, in the namespace
  `procedures`: *Procedure: deploy (when the user asks to deploy). 1. run
  the tests. 2. tag the release. 3. post in #releases.*

Said again under the same name, the new version replaces the old one, which
is kept and hidden from recall. Nothing is kept from a sealed conversation.
With the gate on, a message it calls a procedure goes straight to procedure
filing, without the router. Facts also carry their `kind` (preference, fact
or event) on the stored attribute.

## Before recall: read routing

Recall searches every memory store the agent has, every turn. Read routing
asks one yes-or-no question per kind of memory, in one request:

- **episodic**: does the request refer to something said or done before?
  Past conversations are searched.
- **semantic**: does it depend on a lasting fact or preference? The
  knowledge base and entity facts are searched.
- **procedural**: does it ask for a routine the user set up? Workflows,
  skills and stated procedures are searched.

```python
RetrievalPolicy(read_routing=True, read_threshold=0.5)
```

A kind below `read_threshold` is not searched that turn. The conversation
itself is always loaded. When the model cannot answer, every store is
searched. `agent._last_retrieval_stats["read_routes"]` has the turn's
probabilities. In the notebook, d1 chose the right stores 0.67 of the time
against 0.45 for the nearest store description, and found 8 of 12 requests
that need two stores (none before). Jev and OpenAI Decisions scored lower
there (0.38 and 0.43): the criteria were written with d1. In a live run, "What's 2 + 2?"
searched nothing and "Deploy the new version the usual way" searched only
the stated procedures.

The learning control plane's evidence pack does its own multi-source
retrieval and is not routed.

## The semantic cache

At the old default of 0.78, cosine similarity served today's weather for
"tomorrow", Q2 revenue for Q3, and this year's figure for last year's. No
threshold separates paraphrases from near misses. The verifier reads the
new question against the cached **answer**:

```python
MemAgent(..., semantic_cache=True, semantic_cache_config={"verifier": True})
```

The `verify_top` (3) closest entries at or above `similarity_threshold` are
checked, best first; the first with P(yes) of at least `verify_threshold`
(0.5) is served. A question that repeats a cached one, apart from case,
spacing and closing punctuation, is served without a check. When the verifier
cannot answer, nothing is served.

`similarity_threshold` now defaults to 0.9 without a verifier and 0.7 with
one. Saved agents keep the threshold they were saved with. In the notebook
the verifier got 47 of 48 lookups right, against 28 for cosine at 0.9.

## Writing a note: dedupe

```python
agent.with_memory_routing({"dedupe_writes": True})
```

Before the agent writes a note to its knowledge base (`remember`, a routed
turn, a stated procedure), the decision model compares it with the three
closest notes of this memory and user (cosine 0.5 or more):

| Relation | What happens |
|---|---|
| same | Not written again; the receipt names the note that already says it. |
| refines (more detail) | Written; the old note is hidden from recall, kept, `superseded_by` the new one. |
| contradicts (no longer true) | The same. |
| complements (another fact about it) | Written with `related_ids` to the old note. |
| unrelated | Written. |

A relation acts only at 0.5 confidence or more, and a pinned or verified note
is never hidden. In a live run, "our team office is in Leeds now"
(contradicts, 0.98) hid "the team office is in London".

## Consolidation relations

```python
agent.with_memory_consolidation(True, relations=True)
```

Every consolidation pass, scheduled ones included, then also compares the
agent's own knowledge-base notes of that memory and user (namespaces
`remembered` and `procedures`) with the three closest older notes each. "same" hides the newer note; "refines" and "contradicts" hide the
older one (superseded by the newer); "complements" links them. Like the rest
of a plan, nothing changes until it is applied, and `restore` undoes it. In
the notebook, cosine above 0.95 got 0.20 of these right and caught no
contradiction; d1 got 0.88 and caught 10 of 10.
`relation_min_similarity` (0.5), `relation_threshold` (0.5) and
`max_relation_pairs` (60) bound the pass.

## Tool routing

Tool routing ranks tools by meaning: for "Send the Q3 report to Sarah" the
report tool comes first, and every request is offered some tool. With tool
selection, the decision model picks from the closest five, or none:

```python
ContextPolicy(tool_selection=True, tool_selection_margin=0.15)
```

The runner-up is offered too when it is within `tool_selection_margin`. When
the model says none, no tool is offered. A selector turns off the whole-list
shortcut (`stable_tool_list_tokens`), which would send every tool anyway.
Tools disclosed on earlier turns stay visible up to `sticky_tool_limit`; set
it to 0 to offer only this turn's pick. In the notebook the right tool came
first 0.96 of the time against 0.71, and wrong tool calls fell from 4 to 0.

## Chunking

```python
KnowledgeBase(provider).ingest_knowledge(text, "handbook", chunking_strategy="decision", decision_model="d1")
```

Each gap between sentences is shown with three sentences on each side, and
the model is asked whether a new topic, request or event starts there. A
cut goes where P(yes) reaches `boundary_threshold` (0.5); a chunk is also cut
before it passes twice `chunk_size`. Unlike `"semantic"`, which cuts at the
largest similarity drops, sections that share vocabulary still separate.
There is one question per gap: batched with d1, in parallel against a hosted
model. On the notebook's 1,500-word transcript, d1 found 4 of 7 topic changes
(F1 about 0.5) and percentile chunking found 3, with the same retrieval hit
rate. Hosted models answer yes at most gaps at 0.5 and cut nearly every
sentence: use d1 for chunking, or raise the threshold.

## What it costs

Each feature adds requests: one per turn for the gate and for read routing,
one per lookup for the verifier (up to `verify_top`), one per note compared,
one per gap when chunking. With d1 on a laptop GPU a question takes about
70 to 150 ms after a first load of about 9 s; Jev about 0.2 s; OpenAI
Decisions about 0.7 s; a background model about a second.
