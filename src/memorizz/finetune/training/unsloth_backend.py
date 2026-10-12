"""Unsloth training, for machines with an NVIDIA GPU.

Unsloth trains decision models on a small LM with Cloudflare Clef's option
head (``FastDecisionModel``) and embedding models with LoRA
(``FastSentenceTransformer``), faster and in less memory than plain
transformers. It needs CUDA; on a Mac MemoRizz trains locally instead, or
exports a bundle (data in Unsloth's format plus a script) to run on a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..formats import systemone_row
from .common import Canceled, Reporter
from .hf_callback import reporter_callback


def available() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "PyTorch is not installed"}
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "Unsloth trains on NVIDIA, AMD or Intel GPUs and this machine has none "
            "(its Apple port is not released yet), so MemoRizz trains here with "
            "sentence-transformers and PEFT",
        }
    try:
        import unsloth  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "reason": "Unsloth is not installed (pip install unsloth)",
        }
    return {"available": True, "reason": ""}


def train(
    config: Dict[str, Any],
    data: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Reporter,
) -> Dict[str, Any]:
    status = available()
    if not status["available"]:
        raise RuntimeError(
            status["reason"]
            + "; train with the local backend or export an Unsloth bundle"
        )
    if config["kind"] != "decision":
        raise RuntimeError(
            "MemoRizz uses Unsloth for decision models; train embeddings and rerankers locally"
        )
    from datasets import Dataset as HFDataset
    from transformers import TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

    params = {
        "epochs": 2,
        "batch_size": 2,
        "grad_accum": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "max_length": 2048,
        "load_in_4bit": True,
        "seed": 3407,
        **(config.get("params") or {}),
    }
    report.status(f"Loading {config['base_model']} with Unsloth")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=int(params["max_length"]),
        load_in_4bit=bool(params["load_in_4bit"]),
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=int(params["lora_r"]),
        lora_alpha=int(params["lora_r"]),
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=int(params["seed"]),
    )

    def items(split: str):
        rows = [systemone_row(config["spec"], row) for row in data.get(split) or []]
        if not rows:
            return []
        built, _ = FastDecisionModel.build_dataset(
            HFDataset.from_list(rows), tokenizer, model
        )
        return built

    train_items, eval_items, test_items = (
        items("train"),
        items("validation"),
        items("test"),
    )
    before = (
        FastDecisionModel.evaluate(model, tokenizer, test_items) if test_items else {}
    )
    report.evaluation("before", before)
    callback = reporter_callback(report)
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items or None,
        args=TrainingArguments(
            per_device_train_batch_size=int(params["batch_size"]),
            gradient_accumulation_steps=int(params["grad_accum"]),
            warmup_ratio=0.1,
            num_train_epochs=float(params["epochs"]),
            learning_rate=float(params["learning_rate"]),
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=1,
            output_dir=str(out_dir / "checkpoints"),
            report_to="none",
            seed=int(params["seed"]),
        ),
        callbacks=[callback],
    )
    trainer.train()
    if callback.canceled:
        raise Canceled("Training was canceled")
    if eval_items:
        report.evaluation(
            "calibration", FastDecisionModel.calibrate(model, tokenizer, eval_items)
        )
    after = (
        FastDecisionModel.evaluate(model, tokenizer, test_items) if test_items else {}
    )
    report.evaluation("after", after)
    model_dir = out_dir / "model"
    model.save_pretrained_merged(str(model_dir))
    (model_dir / "memorizz_decision.json").write_text(
        json.dumps(
            {
                "kind": "decision",
                "backbone": "unsloth",
                "base_model": config["base_model"],
                "question": config["spec"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "metrics_before": before,
        "metrics_after": after,
        "model_dir": str(model_dir),
        "device": "cuda",
    }


__all__ = ["available", "train"]
