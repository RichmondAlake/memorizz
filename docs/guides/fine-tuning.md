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
learned, and is judged on frozen sets nobody trains on:

```python
v1 = ft.start_run(round1, base_model="Qwen/Qwen3-Embedding-0.6B")
ft.evaluate(frozen, run_id=v1.run_id)                  # kept in the provider
ft.generate(round2, labels=["entity_memory", "summaries"], count=40)   # v1's weak options
v2 = ft.start_run(round2, from_run=v1.run_id,
                  replay=[{"dataset_id": round1, "fraction": 0.5}])
ft.evaluate(frozen, run_id=v2.run_id)
ft.evaluate(frozen, jev_model="jev-latest")           # a reference
ft.evaluations(frozen)                                 # every version on this set
```

In a three-round run on a real agent's memories (Oracle AI Database, a
Qwen3-Embedding-0.6B router), synthetic data alone reached 90% on synthetic
examples but 14% on the agent's own memories; adding half of the agent's
memories with replay lifted that to 37%; replay kept round-1 skill (80% vs
70% without it); and fixing how tools were rendered, found from the
confusion matrix, took the third version to 83% on the frozen agent set,
ahead of Jev zero-shot (77%).

## Using a trained model

```python
ft.activate(run.run_id, "memory_routing")
ft.route("Remember that Jordan from finance always wants reports as PDF.")
# {"memory_type": "entity_memory", "confidence": 0.93, "probabilities": {...}}
ft.decide(run.run_id, state, questions)      # any System One question
ft.rank(run.run_id, query, passages)          # embeddings, rerankers, relevance
```

- An embedding model is a SentenceTransformer folder with its query and
  document prompts: set `MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=huggingface`
  and `MEMORIZZ_DEFAULT_EMBEDDING_MODEL=<model folder>`, then re-embed the
  stored memories.
- A reranker is a CrossEncoder folder: compare it in
  [Evalground](evalground-comparisons.md) as a `cross_encoder` reranker.
- Export any dataset as typed-decisions JSONL (`systemone`), JevEmbed,
  chat, Sentence Transformers or cross-encoder rows.

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
memorizz finetune activate RUN memory_routing | route "text"
```

MCP tools: `memorizz_list_finetune_datasets`, `memorizz_get_finetune_dataset`,
`memorizz_create_finetune_dataset`, `memorizz_add_finetune_examples`,
`memorizz_review_finetune_examples`, `memorizz_generate_finetune_data`,
`memorizz_start_finetune_run` (needs `MEMORIZZ_MCP_SERVER_ALLOW_TRAINING`,
on by default only for local stdio), `memorizz_get_finetune_run` and
`memorizz_finetune_decide`. Over HTTP each principal sees only its own
datasets.

The UI's **Continual learning → Fine-tuning** page covers the whole loop:
the label balance, generation with live progress, review, the input each
base model sees, training with live loss and validation curves, and the
before/after comparison with a confusion matrix, a "try it" box and
activation.

## What to keep in mind

- Small held-out sets swing a lot. Treat a before/after on a few dozen
  examples as a smoke test; generate or import more before trusting it.
- A local 7B teacher writes varied data but labels hard routing cases
  inconsistently; a stronger verifier (Jev, or a frontier LLM) keeps more
  examples, and the report says which options came up short.
- Fine-tuning one task can cost general ability. Keep the base model's
  scores, compare across runs, and keep the datasets: retraining with the
  old examples alongside new ones is how a model keeps what it learned.
