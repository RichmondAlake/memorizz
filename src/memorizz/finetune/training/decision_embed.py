"""A decision model on an embedding backbone (JevEmbed style).

The state, written with the question's instructions, is embedded as a
query; each option is embedded as its own document (``name: description``);
P(option) = softmax(cos(query, option) / tau). Training is cross-entropy
over the question's own options (never another question's), with LoRA on
the encoder. Because options are inputs, the trained model can still be
asked questions with options it never saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..bases import decision_options, decision_query
from ..evaluate import classification_metrics, fit_temperature, ranking_metrics, softmax
from ..tasks import gold_key, only_question, options
from .common import (
    Reporter,
    batches,
    pick_device,
    schedule,
    steps_for,
    synchronize,
    train_dtype,
)

DEFAULTS = {
    "epochs": 2,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "tau": 0.1,
    "max_length": 256,
    "warmup_ratio": 0.1,
    "seed": 7,
    "precision": "auto",
    "gradient_checkpointing": True,
}


def load_encoder(base_model: str, family: str, device: str, precision: str = "auto"):
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {
        "model_kwargs": {"torch_dtype": train_dtype(device, precision)}
    }
    if family == "embeddinggemma":
        # Text only: MemoRizz never sends images or audio.
        kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
    return SentenceTransformer(base_model, device=device, **kwargs)


def transformer_module(model):
    from sentence_transformers.base.modules.transformer import Transformer

    for module in model.modules():
        if isinstance(module, Transformer):
            return module
    raise ValueError("No transformer module to fine-tune in this model")


def add_lora(model, params: Dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    module = transformer_module(model)
    config = LoraConfig(
        r=int(params["lora_r"]),
        lora_alpha=int(params["lora_alpha"]),
        lora_dropout=float(params["lora_dropout"]),
        target_modules="all-linear",
    )
    if params.get("gradient_checkpointing"):
        # Recompute activations in the backward pass: much less memory.
        module.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        module.model.enable_input_require_grads()
    module.model = get_peft_model(module.model, config)
    return module


def merge_lora(model) -> None:
    module = transformer_module(model)
    if hasattr(module.model, "merge_and_unload"):
        module.model = module.model.merge_and_unload()


class DecisionScorer:
    """Scores states against one question's options with an encoder."""

    def __init__(self, model, family: str, question: Dict[str, Any], tau: float):
        self.model, self.family, self.question, self.tau = model, family, question, tau
        self.keys = [key for key, _ in options(question)]

    def cosines(self, states: List[Any], batch_size: int = 16) -> List[List[float]]:
        import torch

        queries = [decision_query(self.family, self.question, s) for s in states]
        with torch.no_grad():
            q = self.model.encode(
                queries,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            o = self.model.encode(
                decision_options(self.family, self.question),
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return (q @ o.T).float().cpu().tolist()


def evaluate_rows(
    scorer: DecisionScorer,
    qid: str,
    rows: List[Dict[str, Any]],
    *,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    cosines = scorer.cosines([r["input"]["state"] for r in rows])
    logits = [[c / scorer.tau for c in row] for row in cosines]
    gold = [scorer.keys.index(gold_key(scorer.question, r["label"][qid])) for r in rows]
    probabilities = [softmax(row, temperature) for row in logits]
    metrics = classification_metrics(probabilities, gold, scorer.keys)
    if scorer.question["type"] == "noul":
        metrics.update(ranking_metrics(_relevance_groups(rows, probabilities, gold)))
    return metrics


def _relevance_groups(rows, probabilities, gold) -> List[List[tuple]]:
    """Group pointwise relevance decisions by query to rank candidates."""
    groups: Dict[str, List[tuple]] = {}
    for row, probs, g in zip(rows, probabilities, gold):
        state = row["input"]["state"]
        query = state.get("query") if isinstance(state, dict) else row.get("group")
        groups.setdefault(str(query), []).append((probs[1], float(g)))
    return list(groups.values())


def train(
    config: Dict[str, Any],
    data: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Reporter,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    params = {**DEFAULTS, **(config.get("params") or {})}
    qid, question = only_question(config["spec"])
    family = config["family"]
    device = pick_device(params.get("device"))
    torch.manual_seed(int(params["seed"]))
    report.status(f"Loading {config['base_model']} on {device}", device=device)
    model = load_encoder(config["base_model"], family, device, str(params["precision"]))
    model.max_seq_length = int(params["max_length"])
    scorer = DecisionScorer(model, family, question, float(params["tau"]))
    evaluation_rows = data.get("test") or data.get("validation") or []

    report.status("Scoring the base model on held-out examples")
    before = evaluate_rows(scorer, qid, evaluation_rows)
    report.evaluation("before", before)

    if params["lora"]:
        add_lora(model, params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    report.status(
        f"Training {sum(p.numel() for p in trainable) / 1e6:.1f}M of "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters",
    )
    train_rows = data["train"]
    rows = [
        (
            decision_query(family, question, r["input"]["state"]),
            scorer.keys.index(gold_key(question, r["label"][qid])),
        )
        for r in train_rows
    ]
    option_texts = decision_options(family, question)
    batch_size = int(params["batch_size"])
    epochs = float(params["epochs"])
    total = steps_for(len(rows), batch_size, epochs)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(params["learning_rate"]), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, schedule(total, float(params["warmup_ratio"]))
    )
    tau = float(params["tau"])

    def embed(texts: List[str]):
        features = model.tokenize(texts)
        features = {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()
        }
        return F.normalize(model(features)["sentence_embedding"].float(), dim=-1)

    step = 0
    history = []
    for epoch in range(int(-(-epochs // 1))):
        model.train()
        for batch in batches(rows, batch_size, seed=int(params["seed"]) + epoch):
            if step >= total:
                break
            report.check()
            queries = embed([text for text, _ in batch])
            opts = embed(option_texts)
            logits = queries @ opts.T / tau
            labels = torch.tensor([label for _, label in batch], device=device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
            report.step(
                step,
                total,
                epoch=step / max(1, total) * epochs,
                loss=loss.item(),
                learning_rate=scheduler.get_last_lr()[0],
                batch_accuracy=round(accuracy, 4),
            )
        synchronize(device)
        model.eval()
        if data.get("validation"):
            metrics = evaluate_rows(scorer, qid, data["validation"])
            history.append(
                {
                    "epoch": epoch + 1,
                    **{k: metrics.get(k) for k in ("accuracy", "macro_f1", "nll")},
                }
            )
            report.evaluation("validation", metrics, epoch=epoch + 1, step=step)

    model.eval()
    temperature = 1.0
    if data.get("validation"):
        cos = scorer.cosines([r["input"]["state"] for r in data["validation"]])
        gold = [
            scorer.keys.index(gold_key(question, r["label"][qid]))
            for r in data["validation"]
        ]
        temperature = fit_temperature([[c / tau for c in row] for row in cos], gold)
        report.status(f"Calibrated: temperature {temperature}")
    after = evaluate_rows(scorer, qid, evaluation_rows, temperature=temperature)
    report.evaluation("after", after)

    report.status("Saving the model")
    merge_lora(model)
    model_dir = out_dir / "model"
    model.save(str(model_dir))
    decision = {
        "kind": "decision",
        "backbone": "embedding",
        "family": family,
        "base_model": config["base_model"],
        "task": config.get("task"),
        "question_id": qid,
        "question": question,
        "tau": tau,
        "temperature": temperature,
    }
    (model_dir / "memorizz_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return {
        "metrics_before": before,
        "metrics_after": after,
        "model_dir": str(model_dir),
        "history": history,
        "device": device,
        "temperature": temperature,
    }


__all__ = [
    "DEFAULTS",
    "DecisionScorer",
    "add_lora",
    "evaluate_rows",
    "load_encoder",
    "merge_lora",
    "train",
]
