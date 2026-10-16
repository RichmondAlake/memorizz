"""Embedding and reranker fine-tuning for memory retrieval.

Both train on (query, positive memory, hard negatives) triples:

* an embedding model learns to put each query near its memory with
  MultipleNegativesRankingLoss (InfoNCE with in-batch and hard negatives),
  using the base model's own query and document prompts (EmbeddingGemma also
  learns Matryoshka sizes 768/512/256/128);
* a reranker (cross-encoder) learns one relevance logit per (query, passage)
  pair with binary cross-entropy, positives against hard negatives and other
  queries' memories.

Each is scored on the held-out queries before and after with
``memorizz.finetune.scoring`` (the code the comparisons in the UI use): an
embedding model searches every memory in the dataset; a reranker reorders
each query's memory, its look-alikes and 9 other memories.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .common import Canceled, Reporter, pick_device
from .hf_callback import reporter_callback

EMBEDDING_DEFAULTS = {
    "epochs": 2,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_length": 256,
    "warmup_ratio": 0.1,
    "seed": 7,
    "precision": "auto",
    "gradient_checkpointing": True,
}
RERANKER_DEFAULTS = {
    "epochs": 2,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_length": 384,
    "warmup_ratio": 0.1,
    "seed": 7,
    "extra_negatives": 3,
}


def prompts_for(family: str, instruction: str) -> Dict[str, str]:
    """The query and document prompts each embedding family was trained with."""
    if family == "embeddinggemma":
        return {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }
    if family == "qwen3-embedding":
        return {"query": f"Instruct: {instruction}\nQuery:", "document": ""}
    return {"query": "", "document": ""}


def corpus_of(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for rows in data.values():
        for row in rows:
            for text in [row["input"]["positive"], *row["input"].get("negatives", [])]:
                texts.setdefault(text, f"doc{len(texts)}")
    return {doc_id: text for text, doc_id in texts.items()}


def train_embedding(
    config: Dict[str, Any],
    data: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Reporter,
) -> Dict[str, Any]:
    from datasets import Dataset as HFDataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    from ..scoring import embedding_scorer, memory_pool, rank_held_out
    from .decision_embed import add_lora, load_encoder, merge_lora

    params = {**EMBEDDING_DEFAULTS, **(config.get("params") or {})}
    family = config["family"]
    device = pick_device(params.get("device"))
    prompts = prompts_for(family, str(config["spec"].get("instruction") or ""))
    report.status(f"Loading {config['base_model']} on {device}", device=device)
    model = load_encoder(config["base_model"], family, device, str(params["precision"]))
    model.max_seq_length = int(params["max_length"])

    pool = memory_pool(data)
    held_out = data.get("test") or data.get("validation") or []

    def measure() -> Dict[str, Any]:
        if not held_out:
            return {}
        return rank_held_out(
            "embedding", held_out, pool, embedding_scorer(model, prompts)
        )

    report.status("Scoring the base model on held-out queries")
    before = measure()
    report.evaluation("before", before)

    if params["lora"]:
        add_lora(model, params)
    rows = data["train"]
    columns = {
        "anchor": [r["input"]["query"] for r in rows],
        "positive": [r["input"]["positive"] for r in rows],
    }
    if rows and all(r["input"].get("negatives") for r in rows):
        columns["negative"] = [r["input"]["negatives"][0] for r in rows]
    train_dataset = HFDataset.from_dict(columns)
    loss = losses.MultipleNegativesRankingLoss(model)
    if family == "embeddinggemma":
        loss = losses.MatryoshkaLoss(model, loss, matryoshka_dims=[768, 512, 256, 128])
    validation = data.get("validation") or []
    callback = reporter_callback(
        report,
        on_epoch=(
            (
                lambda: rank_held_out(
                    "embedding", validation, pool, embedding_scorer(model, prompts)
                )
            )
            if validation
            else None
        ),
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(params["epochs"]),
        per_device_train_batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        warmup_ratio=float(params["warmup_ratio"]),
        bf16=False,  # weights are already loaded in the training dtype
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={
            "anchor": prompts["query"],
            "positive": prompts["document"],
            "negative": prompts["document"],
        },
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=int(params["seed"]),
        use_cpu=device == "cpu",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[callback],
    )
    report.status(f"Training on {len(rows)} queries")
    trainer.train()
    if callback.canceled:
        raise Canceled("Training was canceled")
    # Merge first, so "after" scores the model as saved and used.
    merge_lora(model)
    after = measure()
    report.evaluation("after", after)

    report.status("Saving the model")
    model.prompts = {"query": prompts["query"], "document": prompts["document"]}
    model_dir = out_dir / "model"
    model.save(str(model_dir))
    meta = {
        "kind": "embedding",
        "family": family,
        "base_model": config["base_model"],
        "prompts": model.prompts,
        "dimensions": int(
            model.get_embedding_dimension()
            if hasattr(model, "get_embedding_dimension")
            else model.get_sentence_embedding_dimension()
        ),
    }
    (model_dir / "memorizz_model.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {
        "metrics_before": before,
        "metrics_after": after,
        "model_dir": str(model_dir),
        "device": device,
    }


def reranking_samples(
    rows: List[Dict[str, Any]], pool: List[str], extra: int, seed: int
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    for row in rows:
        positive = row["input"]["positive"]
        negatives = list(row["input"].get("negatives") or [])
        others = [text for text in pool if text != positive and text not in negatives]
        negatives += rng.sample(others, min(extra, len(others)))
        samples.append(
            {
                "query": row["input"]["query"],
                "positive": [positive],
                "negative": negatives,
            }
        )
    return samples


def train_reranker(
    config: Dict[str, Any],
    data: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Reporter,
) -> Dict[str, Any]:
    import torch
    from datasets import Dataset as HFDataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from ..scoring import cross_encoder_scorer, memory_pool, rank_held_out

    params = {**RERANKER_DEFAULTS, **(config.get("params") or {})}
    device = pick_device(params.get("device"))
    seed = int(params["seed"])
    report.status(f"Loading {config['base_model']} on {device}", device=device)
    model = CrossEncoder(
        config["base_model"],
        num_labels=1,
        device=device,
        max_length=int(params["max_length"]),
    )
    pool = list(corpus_of(data).values())
    held_out = data.get("test") or data.get("validation") or []

    def measure() -> Dict[str, Any]:
        if not held_out:
            return {}
        return rank_held_out(
            "reranker", held_out, memory_pool(data), cross_encoder_scorer(model)
        )

    report.status("Scoring the base model on held-out queries")
    before = measure()
    report.evaluation("before", before)

    queries, passages, labels = [], [], []
    for sample in reranking_samples(
        data["train"], pool, int(params["extra_negatives"]), seed
    ):
        for text in sample["positive"]:
            queries.append(sample["query"]), passages.append(text), labels.append(1.0)
        for text in sample["negative"]:
            queries.append(sample["query"]), passages.append(text), labels.append(0.0)
    positives = max(1.0, sum(labels))
    loss = BinaryCrossEntropyLoss(
        model, pos_weight=torch.tensor((len(labels) - positives) / positives)
    )
    validation = data.get("validation") or []
    callback = reporter_callback(
        report,
        on_epoch=(
            (
                lambda: rank_held_out(
                    "reranker",
                    validation,
                    memory_pool(data),
                    cross_encoder_scorer(model),
                )
            )
            if validation
            else None
        ),
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(params["epochs"]),
        per_device_train_batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        warmup_ratio=float(params["warmup_ratio"]),
        fp16=False,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=seed,
        use_cpu=device == "cpu",
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=HFDataset.from_dict(
            {"query": queries, "passage": passages, "label": labels}
        ),
        loss=loss,
        callbacks=[callback],
    )
    report.status(f"Training on {len(labels)} pairs")
    trainer.train()
    if callback.canceled:
        raise Canceled("Training was canceled")
    after = measure()
    report.evaluation("after", after)
    report.status("Saving the model")
    model_dir = out_dir / "model"
    model.save(str(model_dir))
    meta = {"kind": "reranker", "base_model": config["base_model"]}
    (model_dir / "memorizz_model.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {
        "metrics_before": before,
        "metrics_after": after,
        "model_dir": str(model_dir),
        "device": device,
    }


__all__ = [
    "corpus_of",
    "prompts_for",
    "reranking_samples",
    "train_embedding",
    "train_reranker",
]
