"""Examples written in the input each model, trainer or service expects.

* ``systemone``: typed-decisions rows (``state``, ``questions``, ``gold`` as
  JSON), the body of a ``POST /v1/systemone`` request plus its answers;
  Unsloth's ``FastDecisionModel.build_dataset`` and Jev-style evaluators
  read this.
* ``jevembed``: the decision as an embedding model sees it, the state as a
  query and each option as a document, with the gold option's index.
* ``chat``: the small-LM decision prompt and its answer letter.
* ``sentence-transformers``: (anchor, positive, negative) for an embedding.
* ``cross-encoder``: (query, passage, label) pairs for a reranker.
* ``memorizz``: MemoRizz's own example records, for backup and re-import.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from .bases import (
    BaseModel,
    decision_options,
    decision_query,
    embed_document,
    embed_query,
    lm_messages,
)
from .tasks import gold_key, only_question, options

FORMATS = (
    "systemone",
    "jevembed",
    "chat",
    "sentence-transformers",
    "cross-encoder",
    "memorizz",
)


def _gold(question: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """A gold answer with its (one-hot) distribution, typed-decisions style."""
    keys = [key for key, _ in options(question)]
    chosen = gold_key(question, answer)
    probabilities = {key: (1.0 if key == chosen else 0.0) for key in keys}
    if question["type"] == "choice":
        return {"choice": chosen, "probabilities": probabilities, "confidence": 1.0}
    if question["type"] == "noul":
        value = answer["noul"]
        p_true = (
            float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
        )
        return {"noul": p_true, "probability_true": p_true}
    level = int(chosen)
    return {"level": level, "score": float(level), "probabilities": probabilities}


def systemone_row(spec: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    qid, question = only_question(spec)
    return {
        "id": row.get("example_id"),
        "state": json.dumps(row["input"]["state"], ensure_ascii=False),
        "questions": json.dumps({qid: question}, ensure_ascii=False),
        "gold": json.dumps(
            {qid: _gold(question, row["label"][qid])}, ensure_ascii=False
        ),
        "split": row.get("split"),
    }


def render(
    fmt: str, kind: str, spec: Dict[str, Any], base: BaseModel, row: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """One example in ``fmt`` (some formats expand an example into several rows)."""
    if fmt == "memorizz":
        return [row]
    if kind == "decision":
        qid, question = only_question(spec)
        state = row["input"]["state"]
        if fmt == "systemone":
            return [systemone_row(spec, row)]
        keys = [key for key, _ in options(question)]
        gold = gold_key(question, row["label"][qid])
        if fmt == "jevembed":
            return [
                {
                    "query": decision_query(base.family, question, state),
                    "options": decision_options(base.family, question),
                    "label": keys.index(gold),
                    "answer": gold,
                }
            ]
        if fmt == "chat":
            messages, shown = lm_messages(question, state)
            letter = chr(ord("A") + shown.index(gold))
            return [
                {
                    "messages": [*messages, {"role": "assistant", "content": letter}],
                    "answer": gold,
                }
            ]
        raise ValueError(
            f"A decision dataset exports as systemone, jevembed, chat or memorizz, not {fmt}"
        )
    query, positive = row["input"]["query"], row["input"]["positive"]
    negatives = list(row["input"].get("negatives") or [])
    if fmt == "sentence-transformers":
        instruction = str(spec.get("instruction") or "")
        out = {
            "anchor": embed_query(base.family, instruction, query)
            if base.family == "qwen3-embedding"
            else (
                f"task: search result | query: {query}"
                if base.family == "embeddinggemma"
                else query
            ),
            "positive": embed_document(base.family, positive),
        }
        if negatives:
            out["negative"] = embed_document(base.family, negatives[0])
        return [out]
    if fmt == "cross-encoder":
        return [
            {"query": query, "passage": positive, "label": 1.0},
            *({"query": query, "passage": n, "label": 0.0} for n in negatives),
        ]
    raise ValueError(
        f"A {kind} dataset exports as sentence-transformers, cross-encoder or memorizz, not {fmt}"
    )


def default_format(kind: str, base: BaseModel) -> str:
    if kind == "decision":
        return "chat" if base.backbone == "lm" else "jevembed"
    return "cross-encoder" if kind == "reranker" else "sentence-transformers"


def to_jsonl(rows: Iterable[Dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


UNSLOTH_SCRIPT = '''"""Train a MemoRizz decision dataset with Unsloth on an NVIDIA GPU.

    pip install --upgrade unsloth
    python train_unsloth.py

Data: data.jsonl (typed-decisions rows: state, questions, gold, split).
"""
import json
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

BASE_MODEL = "{base_model}"
rows = [json.loads(line) for line in open("data.jsonl")]
by_split = {{s: [r for r in rows if r.get("split") == s] for s in ("train", "validation", "test")}}

model, tokenizer = FastDecisionModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True)
model = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,
                                         use_gradient_checkpointing="unsloth", random_state=3407)

def items(split):
    if not by_split[split]:
        return []
    built, report = FastDecisionModel.build_dataset(Dataset.from_list(by_split[split]), tokenizer, model)
    print(split, report)
    return built

train_items, eval_items, test_items = items("train"), items("validation"), items("test")
if test_items:
    print("before", FastDecisionModel.evaluate(model, tokenizer, test_items))
trainer = DecisionTrainer(
    model=model, processing_class=tokenizer, train_dataset=train_items, eval_dataset=eval_items or None,
    args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_ratio=0.1,
                           num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine",
                           weight_decay=0.01, bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
                           logging_steps=1, output_dir="outputs", report_to="none", seed=3407),
)
trainer.train()
if eval_items:
    print("calibration", FastDecisionModel.calibrate(model, tokenizer, eval_items))
if test_items:
    print("after", FastDecisionModel.evaluate(model, tokenizer, test_items))
model.save_pretrained_merged("decision-model-merged")
'''


def unsloth_bundle(base_model: str) -> Dict[str, str]:
    """Files to train a decision dataset with Unsloth elsewhere."""
    return {
        "train_unsloth.py": UNSLOTH_SCRIPT.format(base_model=base_model),
        "README.md": (
            "# MemoRizz decision dataset for Unsloth\n\n"
            "1. Copy this folder to a machine with an NVIDIA GPU (or Colab).\n"
            "2. `pip install --upgrade unsloth`\n"
            "3. `python train_unsloth.py`\n\n"
            "`data.jsonl` holds typed-decisions rows (`state`, `questions`, `gold`, `split`),\n"
            "the body of a `/v1/systemone` request plus its gold answers.\n"
        ),
    }


__all__ = [
    "FORMATS",
    "default_format",
    "render",
    "systemone_row",
    "to_jsonl",
    "unsloth_bundle",
]
