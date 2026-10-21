# Fine-Tuning Decision Models, Embeddings and Rerankers

MemoRizz can train three kinds of small model on your own memory, and use
them back in the memory layer:

| Model | What it does in MemoRizz | Trained on |
|---|---|---|
| **Decision model** (System One) | Answers a typed question about a piece of state with a calibrated probability per option: which memory store an item belongs in, whether a candidate memory is relevant | an embedding model (EmbeddingGemma 2, Qwen3-Embedding) or a small LM (Qwen3-0.6B) |
| **Embedding model** | Puts each query near the memory that answers it | EmbeddingGemma 2, Qwen3-Embedding |
| **Reranker** | Scores (query, memory) pairs so the best evidence ranks first | cross-encoders (MiniLM, GTE ModernBERT, BGE v2 m3) |

Every dataset lives in the memory provider MemoRizz is connected to
(filesystem, MongoDB or Oracle AI Database), next to the memories it came
from. Training runs on your machine (Apple GPU, NVIDIA GPU or CPU), and
every run is scored on held-out examples before and after, against the base
model.

## Decision models and the System One contract

A decision model follows the contract TypeSafe's Jev introduced and
Cloudflare's Clef, Perplexity's pplx-decider and Microsoft-Decision-1 share:
a **state** plus **typed questions**, each answered with a probability per
option, never generated text.

```json
{
  "state": "Remember that Jordan from finance always wants reports as PDF.",
  "questions": {
    "memory_type": {
      "type": "choice",
      "instructions": "Which MemoRizz memory store should this item be written to? ...",
      "criteria": {
        "entity_memory": "A lasting attribute, preference ... of one specific named person ...",
        "knowledge_base": "A durable fact, definition, policy ...",
        "do_not_store": "Nothing of lasting or task value ..."
      }
    }
  }
}
```

The options are part of the input, so a dataset's question (its
instructions and option descriptions) is editable, and a trained model can
still be asked questions it never saw. `choice` picks one of named options,
`noul` answers true or false, `score` picks an ordered level.

Tasks (the **Task** dropdown in the UI):

| Task | Question | Use |
|---|---|---|
| `memory_routing` | choice over MemoRizz memory stores, plus `do_not_store` | file a piece of state in the right store |
| `relevance` | noul: does this candidate help answer the query? | rerank memories by P(true) |
| `custom` | your own choice, noul or score question | anything else |

Two backbones:

- **Embedding (JevEmbed style)**: the state, written with the question's
  instructions, is embedded as a query and each option as a document;
  P(option) = softmax(cos / 0.1). Trained with cross-entropy over the
  question's own options, LoRA on the encoder. Light, fast, and the
  embedding model you already know.
- **Small LM (pplx-decider / Unsloth style)**: the LM reads the state,
  question and lettered options in one prompt; the option letters'
  next-token logits are the scores. Options are shuffled during training,
  and evaluation reports how often reversing their order flips the answer.

Each family is given the input it was trained with: Qwen3-Embedding queries
are `Instruct: {instructions}\nQuery:{state}` (no space after `Query:`);
EmbeddingGemma 2 keeps its `task: classification | query: …` prompt with
`title: {option} | text: {description}` options.

## Data

### Synthetic data, checked blind

`generate` follows the recipe of pplx-decider and JevEmbed:

1. a balanced plan: the same number of examples for every option, spread
   over case types (standard, negation, boundary, hard negative), styles
   (chat message, note, tool output, document excerpt, email) and the
   task's hard phenomena (a preference stated in passing, a quoted versus an
   asserted fact, a temporary versus a lasting detail, a correction...);
2. a **teacher** LLM writes states for each target option;
3. a **blind check** labels every state without seeing the target: the
   teacher again, another LLM, or **Jev** (`typesafe/jev-latest`, using
   `TYPESAFE_API_KEY`), which returns calibrated probabilities that are kept
   on the example;
4. only examples the check agrees with (and calls unambiguous) are kept,
   duplicates are dropped, options that came up short are topped up in
   further rounds, and every option gets held-out test and validation
   examples.

Retrieval data (embeddings, rerankers and the relevance decision) follows
E5 and Gecko: for each memory, the teacher writes a question the memory
answers and a hard negative that looks related but does not answer it; the
blind check keeps only pairs it confirms. Memories come from your stores
(`from_memory`) or are written on a topic.

Synthetic examples are left **pending**. Review them, then accept them for
training: in the UI, with "Accept all checked", or one by one with the
label and split you choose.

The teacher defaults to `MEMORIZZ_FINETUNE_TEACHER` (`provider/model`), then
your default LLM, then a local Ollama model. Jev checks cost about $0.04 per
million input tokens; MemoRizz sends up to eight items per request.

### Your memories as examples

`import_memories` turns stored memories into routing examples labelled with
the store they live in. They are a strong source of real data, but a memory
can be filed in the wrong store, so they are left pending for review.

### By hand

Add examples one at a time (UI, `memorizz finetune add`, or
`memorizz_add_finetune_examples`), relabel any example, move it between
splits, or import exported records.

## Training

```python
from memorizz.finetune import FineTuning

ft = FineTuning(memory_provider)
dataset = ft.create_dataset("decision", "memory_routing", name="Routing")
ft.generate(dataset.dataset_id, count=120,
            teacher={"provider": "ollama", "model": "qwen2.5:7b"},
            verifier={"provider": "typesafe", "model": "jev-latest"})
ft.accept_verified(dataset.dataset_id)
run = ft.start_run(dataset.dataset_id, base_model="Qwen/Qwen3-Embedding-0.6B",
                   params={"epochs": 3, "batch_size": 8})
ft.run(run.run_id)            # progress, then metrics before and after
```

A run snapshots the accepted examples into its folder
(`MEMORIZZ_HOME/finetune/runs/<run>`), trains in its own process, and
records its progress and outcome in the memory provider. It scores the
base model on the held-out examples, trains with LoRA, checks the
validation examples each epoch, fits a temperature so probabilities match
how often the model is right, and scores the held-out examples again.

| Kind | Default settings | Metrics |
|---|---|---|
| Decision | LoRA r16, lr 2e-4, 2-3 epochs, batch 8, τ 0.1 | accuracy, macro-F1, NLL, Brier, ECE, confusion; MRR/nDCG for relevance; order flip rate (small LM) |
| Embedding | LoRA r16, MultipleNegativesRankingLoss (+ Matryoshka for EmbeddingGemma) | Recall@1/5/10, MRR@10, nDCG@10 |
| Reranker | full fine-tune, BinaryCrossEntropyLoss, positives against hard negatives | MAP, MRR@10, nDCG@10 |

On a Mac, weights train in bf16 on the Apple GPU with gradient
checkpointing; never fp16 (EmbeddingGemma returns NaN in fp16). Install
with `pip install "memorizz[finetune]"` (torch, sentence-transformers,
transformers ≥ 5.19, PEFT, datasets; EmbeddingGemma 2 also needs pillow and
torchvision).

**Unsloth** trains on NVIDIA, AMD and Intel GPUs. When it is installed with
a GPU, MemoRizz trains small-LM decision models with Unsloth's
`FastDecisionModel` (Cloudflare Clef's option head). Elsewhere, download an
**Unsloth bundle** (your accepted examples in the typed-decisions format
plus a training script) and run it on Colab or any CUDA machine.

## Continual training

A model improves in rounds. Each round continues from the previous model,
adds data where it is weak, replays some earlier data so it keeps what it
learned, and is judged on test sets nobody trains on. Mark a dataset as a
test set and MemoRizz refuses to train on it or replay it:

```python
ft.update_dataset(frozen, test_set=True)               # only scored on from now on
v1 = ft.start_run(round1, base_model="Qwen/Qwen3-Embedding-0.6B")
ft.evaluate(frozen, run_id=v1.run_id)                  # kept in the provider
ft.generate(round2, labels=["entity_memory", "summaries"], count=40)   # v1's weak options
v2 = ft.start_run(round2, from_run=v1.run_id,
                  replay=[{"dataset_id": round1, "fraction": 0.5}])
ft.evaluate(frozen, run_id=v2.run_id)
ft.evaluate(frozen, jev_model="jev-latest")           # a reference
ft.evaluations(frozen)                                 # every version on this set
ft.rename_run(v2.run_id, "round 2")                    # shown instead of "v2"
```

Runs are numbered per job in the order they were started (v1, v2, …; failed
runs get no number), and each records where it came from ("from v1 +
replay").

In a three-round run on a real agent's memories (Oracle AI Database, a
Qwen3-Embedding-0.6B router), synthetic data alone reached 90% on synthetic
examples but 14% on the agent's own memories; adding half of the agent's
memories with replay lifted that to 37%; replay kept round-1 skill (80% vs
70% without it); and fixing how tools were rendered, found from the
confusion matrix, took the third version to 83% on the frozen agent set,
ahead of Jev zero-shot (77%).

## Putting a model to work

The **Models in use** cards at the top of the Fine-tuning page answer one
question per job: is the fine-tuned model better than what MemoRizz uses
now? Each card shows the best fine-tuned version's score next to what is in
use and its untrained base, on the same test examples, says in plain words
whether fine-tuning helped, and has a **Swap in** button (and **Swap back**).
A version's Results show the same comparison.

Comparisons are scored by the same code that scores a run in training
(`memorizz.finetune.scoring`), so a card and the run's own before/after
agree: an embedding model searches every memory in the dataset for each
test question; a reranker reorders each question's memory, its look-alikes
and 9 other memories; an untrained base reads as many tokens as training let
it. When another fine-tuned version is in use (a routing model, say), a
version is compared with that version, so Swap in never looks like an
upgrade when it is a step back.

A comparison pools every fixed set both models were scored on, example by
example: a gain on one test set and a loss on another both count, and the
caption names the sets. Only examples nothing trained on count (test sets,
or a dataset's test and validation examples, never its "all" split).

A card only says **fine-tuning helped** when the gain is unlikely to be
chance. Every score keeps each example's result, and the two models are
compared example by example with a paired sign-flip test (McNemar's exact
test for right/wrong answers; a randomisation test for nDCG per question):
below p = 0.05 the verdict is "helped" or "worse"; otherwise it says the
difference could be chance and that more test examples would settle it.
4 more right answers out of 20, with none lost, is p = 0.125: not enough.

How the numbers are computed:

- Accuracy, macro-F1 (classes absent from both answers and predictions are
  left out, as scikit-learn does), log loss, a multi-class Brier score
  (0 to 2) and ECE over 10 confidence bins. A base model's "before" is
  calibrated on the validation examples exactly as the trained model is,
  so a change in log loss, Brier or ECE comes from training.
- nDCG@10, MRR@10 and Recall@1/5/10 per question, averaged. Tied scores
  count at their expected value over every order of the tie, so a model
  that cannot tell the right memory from a wrong one gets chance, not
  credit. A question with no wrong candidate to rank against is not
  counted (it would score 1.0 for any model).
- A blind check or Jev call that returned nothing counts as a wrong answer.

| Job | What a swap changes | Who uses it |
|---|---|---|
| Memory routing | the router behind `memory_type="auto"` writes | `memorizz_store_memory`; `ft.route(text)` in Python |
| Memory embeddings | the embedding model, and every stored memory is re-embedded with it | every store and search |
| Memory reranking | `MEMORIZZ_RERANKER_MODEL`: a cross-encoder or a relevance decision model | an agent's recall and `memorizz_search_memories` |

```python
ft.overview()                          # the cards, as data
ft.compare(run)                        # score it, its base and what is in use on its test examples
ft.verdict(run)                        # the comparison its card shows
ft.evaluate(dataset, in_use=True)      # the embedding model in use now
ft.swap("memory_routing", run)         # or "embedding", "reranking"
ft.swap_back("memory_routing")
```

### Swapping embeddings

Every stored memory is re-embedded **beside** the live vectors (Oracle,
filesystem and MongoDB stores): search keeps using the current vectors and
model while the new ones are made, then one switch makes them live. Each
staged vector carries a signature of its text, so memories written or
edited during the re-embed are caught up before the switch. If anything
fails before the switch, the staged vectors are discarded and nothing has
changed. The vectors the switch replaces are kept, so **Swap back** reuses
them and re-embeds only memories written or edited since the swap.

| Store | Staged vectors | Previous vectors | Vector size |
|---|---|---|---|
| Oracle | `EMBEDDING_NEXT` column (+ `_SIG`), renamed to `EMBEDDING` at the switch | `EMBEDDING_PREV` | can change |
| Filesystem | `.vector-staging/next/` sidecar files | `.vector-staging/prev/` | any size |
| MongoDB | `memorizz_vector_staging` collection | same collection | fixed by the Atlas index |
| Others | re-embedded in place | none (swap back re-embeds) | fixed |

**Vector size.** A store whose schema pins one size keeps it by default:
the model's vectors are cut to it (EmbeddingGemma and Qwen3-Embedding are
trained to keep their quality at smaller sizes; EmbeddingGemma comes in 128,
256, 512 or 768). An Oracle store can instead change to the model's full
size: the staged column is created at the new size and renamed into place,
so every vector column changes at once. Processes still writing the old
size then fail loudly (ORA-51803) until they restart with the new model,
rather than storing mismatched vectors. A filesystem store takes the model's
full size by default.

```python
ft.embedding_plan(run_id)
# {"store": 768, "native": 1024, "options": [
#    {"dimensions": 768, "kind": "cut", ...}, {"dimensions": 1024, "kind": "resize", ...}],
#  "default": 768, "staged": True, ...}
ft.swap("embedding", run_id, dimensions=1024)   # default: plan["default"]
```

The card and Compare score an embedding model at the size the swap would
store (cut vectors are normalized again), so the numbers you see are the
numbers you get.

**Vector indexes.** If the live Oracle column has a vector index, an index
of the same organization (HNSW or IVF, cosine distance) is built on the
staged column before the switch, so search is never unindexed; the old
index is then dropped. Vector indexes cannot be renamed, so the new one
alternates between `<name>` and `<name>_N`. Custom index parameters are not
visible to the schema and are not carried over.

Rows edited through a path that does not re-embed them keep a vector of
their earlier text; a swap re-embeds them from their current text, and a
swap back restores exactly the vectors that were live before.

### What a swap affects

A swap is not per agent: it changes the model for everything that shares
the memory store. **Swap in** first shows what it touches: the store, its
agents (and those that keep their own choice), where the setting is saved,
and the other MemoRizz processes on this machine using the same store.

- **Embeddings** are one model per store: every memory is a vector from it,
  and agents sharing a store search the same vectors. Every other process
  on the store must use the new model too: until it restarts it embeds
  queries with the old one and searches the re-embedded memories with it,
  which fails silently (same vector size, wrong results). A restart is not
  enough when that process's MemoRizz home names another model in its
  `.env`.
- **The router** is read from the store on every write: other processes
  pick it up at once.
- **The reranker** is read when a process starts: restart the others.

Each MemoRizz process records which store and models it uses (a small file
in the temp folder, dropped when the process exits). Older versions do not;
the dialog lists them, and flags those connected to the same database
server.

### Per-agent routers and rerankers

The router and the reranker do not touch stored data, so an agent can have
its own: a fine-tuned version, or none. Every other agent follows the
store's choice. Set it in the Model registry (Per-agent choices under each
job) or in Python:

```python
ft.set_agent_model(agent_id, "reranking", run_id)       # or "off", or "default"
ft.set_agent_model(agent_id, "memory_routing", "off")   # its "auto" writes skip the router
ft.agent_models()                                       # every agent and its choices
ft.swap_impact("embedding")                             # what a swap would touch
```

An agent's recall uses its reranker; MCP writes and searches in a memory an
agent owns use that agent's choices. Choices are read with a 30-second cache.

A reranker or router set on the agent itself (`reranker=`, `memory_routing=`,
or *Model → Memory models* on its form; see
[Memory routing and reranking](memory-routing.md)) comes first; an agent that
follows the store uses the choice made here.

- **Routing.** With a router swapped in, a write that asks for
  `memory_type="auto"` is filed where the router says. The MCP tool writes
  the knowledge base and short-term memory; when the router says a memory is
  an entity, a conversation turn or a summary, it is kept in the knowledge
  base and the answer names the tool that files it properly
  (`memorizz_upsert_entity`, `memorizz_record_turn`...). Below
  `MEMORIZZ_ROUTING_MIN_CONFIDENCE` (default 0.5) the write goes to the
  default store; `do_not_store` writes nothing. Every routed memory records
  the router's answer.
- **Embeddings.** A swap sets `MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=huggingface`
  and `MEMORIZZ_DEFAULT_EMBEDDING_MODEL=<model folder>` and re-embeds each
  store with the text that store embeds (a knowledge chunk's content, an
  entity's name, type and attributes, a persona's name, role, goals and
  background, a tool's name, description, signature and queries...).
  Searches embed the query with the model's query prompt and memories with
  its document prompt. See [Swapping embeddings](#swapping-embeddings).
- **Reranking.** The reranker rescores the candidates vector search found;
  recency, importance and diversity still apply. A broken or missing model
  never breaks recall: the vector order is kept.
- Settings are saved in MemoRizz's `.env` (`MEMORIZZ_HOME/.env`), so a swap
  holds after a restart.
- Export any dataset as typed-decisions JSONL (`systemone`), JevEmbed,
  chat, Sentence Transformers or cross-encoder rows.

## The model registry

**Continual learning → Model registry** lists every model behind each job:
what runs now (a fine-tuned version, or an external model such as Ollama's
nomic-embed-text), each fine-tuned version and the base models to start
from. Versions are listed as a family tree, each under the version it
continued from ("↳ from v2 + replay"), with the data it trained on, when
and how long, its training result (and whether that could be chance), its
best score on a fixed test set and its size on disk. Open a version for its
card (data, replay, settings, trainer, every fixed-set score, model folder)
and to swap it in, rename it or delete it. A version in use cannot be
deleted: swap back first. Failed runs keep their error.

The registry is a view over what MemoRizz already records in the memory
provider (run records and evaluations), not a separate store:

```python
ft.registry()   # {"jobs": [{"title", "in_use", "versions", "failed", "bases"}], "totals": {...}}
```

Versions are numbered per job: memory routing, embeddings and reranking
(cross-encoders and relevance decision models together) each count v1, v2,
…; a custom decision is numbered per question.

### How it trained, and on what

A run's results (and its registry card) show **how it trained**: the loss
at every step, raw and smoothed on a log scale with the epoch boundaries
marked; the validation score after each epoch; the test score before and
after training; and notes in plain words: whether the loss was still
falling at the end (more epochs may help) or had levelled off, and whether
validation peaked early and then fell (later epochs may be overfitting).
Embedding and reranker runs before 13 Oct 2026 scored only before and after,
so they show no validation curve. **Compare versions' training** on the
registry overlays several versions of one job: the loss by % of training, so
runs of different lengths line up, and validation by epoch.

**The data it trained on** is the run's own snapshot, taken when it
started: examples per split and per option, what was replayed from which
earlier dataset, how the dataset has changed since, and every example,
filterable by split and source.

```python
ft.curves(run_id)    # {"steps", "validation", "before", "after", "convergence", ...}
ft.run_data(run_id, split="test", source="replay")   # {"summary", "since", "rows", ...}
```

### Speed

**Measure speed** times a version doing what it does in use (route one
memory, embed one query, or rerank one query's 10 memories) next to what it
would replace: the version in use now or, when it is in use itself, what
Swap back restores. Each model is loaded the way the store loads it, timed
on the same inputs (its test examples) one call at a time after a warm-up
pass, without the embedding cache; the median and p95 are kept with the run.

```python
ft.speed(run_id)   # {"unit": "per memory routed", "rows": [{"role", "label", "median_ms", "p95_ms", "n", "device"}]}
```

A fine-tuned version is the same size as its base model, so fine-tuning
alone never changes speed: a routing v6 runs as fast as v4. Speed changes
with what runs: a local fine-tuned EmbeddingGemma in place of Ollama's
nomic-embed-text, or a reranker where none ran (it adds its time to every
search). Timings depend on the machine and on what else it is doing;
measure again on the machine that serves the store.

## CLI, MCP and UI

```text
memorizz finetune doctor | tasks | datasets
memorizz finetune create decision memory_routing --name Routing
memorizz finetune generate DATASET --count 120 --teacher ollama/qwen2.5:7b --verifier typesafe/jev-latest
memorizz finetune import-memories DATASET
memorizz finetune review DATASET --accept-verified
memorizz finetune preview DATASET --base google/embeddinggemma-2
memorizz finetune export DATASET --format systemone [--unsloth-bundle DIR]
memorizz finetune train DATASET --base Qwen/Qwen3-Embedding-0.6B --epochs 3
memorizz finetune train DATASET --from-run RUN --replay OLD_DATASET:0.5
memorizz finetune evaluate DATASET --run RUN | --base MODEL | --jev jev-latest
memorizz finetune history [--dataset DATASET]
memorizz finetune runs | status RUN --follow | cancel RUN
memorizz finetune decide RUN "state" | rank RUN --query Q --passage P ...
memorizz finetune models                    # is each fine-tuned model better? what is in use?
memorizz finetune compare RUN               # score it, its base and the model in use on its test examples
memorizz finetune swap memory_routing|embedding|reranking RUN
memorizz finetune swap-back memory_routing|embedding|reranking
memorizz finetune route "text"
```

MCP tools: `memorizz_list_finetune_datasets`, `memorizz_get_finetune_dataset`,
`memorizz_create_finetune_dataset`, `memorizz_add_finetune_examples`,
`memorizz_review_finetune_examples`, `memorizz_generate_finetune_data`,
`memorizz_start_finetune_run` (needs `MEMORIZZ_MCP_SERVER_ALLOW_TRAINING`,
on by default only for local stdio), `memorizz_get_finetune_run` and
`memorizz_finetune_decide`. Over HTTP each principal sees only its own
datasets.

`memorizz_store_memory` accepts `memory_type: "auto"` to file a memory with
the active router.

The UI's **Continual learning → Fine-tuning** page opens on the **Models in
use** cards (compare, swap in, swap back). Datasets are grouped by job, with
test sets tagged. A dataset walks through three steps: **Data** (examples
per option, generation with live progress, review), **Train** (start a new
version or continue one, replay earlier data, live loss and validation
curves) and **Results** (the card's comparison with Swap in, the training
report in plain words, where the model goes wrong, with a button to generate
more examples for those options, and every version scored on the same
sets). The question and options and each base model's input format are
under **Advanced**. A test set shows **Data** and **Scores** only.

## What to keep in mind

- Small held-out sets swing a lot. The verdict says when a difference could
  be chance; generate or import more test examples to settle it.
- Examples written together (one generation call's paraphrases) always land
  in the same split, so a test example never has a near-copy in training.
- A local 7B teacher writes varied data but labels hard routing cases
  inconsistently; a stronger verifier (Jev, or a frontier LLM) keeps more
  examples, and the report says which options came up short.
- Fine-tuning one task can cost general ability. Keep the base model's
  scores, compare across runs, and keep the datasets: retraining with the
  old examples alongside new ones is how a model keeps what it learned.
