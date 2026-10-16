"""A decision model on a small LM, reading option letters (pplx-decider /
Unsloth decision style, without a new head).

The LM reads one chat prompt: the state, the question and the options
lettered A, B, C…; the next-token logits of those letters at the answer
position are the option scores. Training is cross-entropy over the letters
with LoRA. Options are shuffled per example so the model learns the
options, not their positions; evaluation reports how often reversing the
order flips the answer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from ..bases import LETTERS, lm_messages, render_state
from ..evaluate import fit_temperature, softmax
from ..scoring import decision_metrics
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
    "batch_size": 4,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_length": 1024,
    "warmup_ratio": 0.1,
    "seed": 7,
    "shuffle_options": True,
    "precision": "auto",
    "gradient_checkpointing": True,
}


def load_lm(base_model: str, device: str, precision: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"  # the answer slot is the last position
    # Never cut the end of a prompt: that is where the answer slot is. Long
    # states are shortened before this (LMScorer.prompt); this is a backstop.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=train_dtype(device, precision)
    )
    return tokenizer, model.to(device)


def prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:  # Qwen3: answer directly, with an empty thinking block
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def letter_ids(tokenizer, count: int) -> List[int]:
    ids = []
    for letter in LETTERS[:count]:
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"{letter!r} is not a single token for this tokenizer")
        ids.append(tokens[0])
    return ids


class LMScorer:
    """Option logits for one question, in the question's own option order."""

    def __init__(
        self, tokenizer, model, question: Dict[str, Any], device: str, max_length: int
    ):
        self.tokenizer, self.model, self.question = tokenizer, model, question
        self.device, self.max_length = device, max_length
        self.keys = [key for key, _ in options(question)]
        self.ids = letter_ids(tokenizer, len(self.keys))

    def _length(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def prompt(self, state: Any, order: List[int]) -> tuple[str, List[str]]:
        """The prompt for one state, its state shortened (head kept) when the
        whole prompt would not fit in ``max_length`` tokens: the question,
        the options and the answer slot always stay."""
        messages, keys = lm_messages(self.question, state, order)
        text = prompt_text(self.tokenizer, messages)
        if self._length(text) <= self.max_length:
            return text, keys
        empty, _ = lm_messages(self.question, "", order)
        budget = self.max_length - self._length(prompt_text(self.tokenizer, empty)) - 4
        ids = self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"]
        short = self.tokenizer.decode(ids[: max(0, budget)]) + " …"
        messages, keys = lm_messages(self.question, short, order)
        return prompt_text(self.tokenizer, messages), keys

    def forward(self, states: List[Any], orders: List[List[int]]):
        """Letter logits [B, K] (in shown order) and each row's shown keys."""
        texts, shown = [], []
        for state, order in zip(states, orders):
            text, keys = self.prompt(state, order)
            texts.append(text)
            shown.append(keys)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = self.model(**encoded, logits_to_keep=1).logits[:, -1, :]
        return logits[:, self.ids[: len(self.keys)]].float(), shown

    def logits(
        self, states: List[Any], *, reverse: bool = False, batch_size: int = 8
    ) -> List[List[float]]:
        import torch

        out: List[List[float]] = []
        k = len(self.keys)
        order = list(range(k))[::-1] if reverse else list(range(k))
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                chunk = states[start : start + batch_size]
                letter_logits, shown = self.forward(chunk, [order] * len(chunk))
                for row, keys in zip(letter_logits.cpu().tolist(), shown):
                    by_key = dict(zip(keys, row))
                    out.append([by_key[key] for key in self.keys])
        return out


def evaluate_rows(
    scorer: LMScorer,
    qid: str,
    rows: List[Dict[str, Any]],
    *,
    temperature: float = 1.0,
    flips: bool = True,
) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    states = [r["input"]["state"] for r in rows]
    logits = scorer.logits(states)
    gold = [scorer.keys.index(gold_key(scorer.question, r["label"][qid])) for r in rows]
    probabilities = [softmax(row, temperature) for row in logits]
    metrics = decision_metrics(
        rows, probabilities, gold, scorer.keys, scorer.question["type"]
    )
    metrics["temperature"] = temperature
    if flips:
        reversed_logits = scorer.logits(states, reverse=True)
        changed = sum(
            1
            for a, b in zip(logits, reversed_logits)
            if max(range(len(a)), key=a.__getitem__)
            != max(range(len(b)), key=b.__getitem__)
        )
        metrics["order_flip_rate"] = round(changed / len(rows), 4)
    return metrics


def validation_temperature(scorer: LMScorer, qid: str, rows) -> float:
    """The temperature that calibrates ``scorer`` on the validation rows."""
    if not rows:
        return 1.0
    logits = scorer.logits([r["input"]["state"] for r in rows])
    gold = [scorer.keys.index(gold_key(scorer.question, r["label"][qid])) for r in rows]
    return fit_temperature(logits, gold)


def train(
    config: Dict[str, Any],
    data: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Reporter,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    params = {**DEFAULTS, **(config.get("params") or {})}
    qid, question = only_question(config["spec"])
    device = pick_device(params.get("device"))
    torch.manual_seed(int(params["seed"]))
    report.status(f"Loading {config['base_model']} on {device}", device=device)
    tokenizer, model = load_lm(config["base_model"], device, str(params["precision"]))
    model.eval()
    scorer = LMScorer(tokenizer, model, question, device, int(params["max_length"]))
    evaluation_rows = data.get("test") or data.get("validation") or []

    # Calibrated on validation like the trained model (see decision_embed).
    report.status("Scoring the base model on held-out examples")
    before = evaluate_rows(
        scorer,
        qid,
        evaluation_rows,
        temperature=validation_temperature(scorer, qid, data.get("validation")),
    )
    report.evaluation("before", before)

    if params["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(params["lora_r"]),
            lora_alpha=int(params["lora_alpha"]),
            lora_dropout=float(params["lora_dropout"]),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    scorer.model = model
    trainable = [p for p in model.parameters() if p.requires_grad]
    report.status(
        f"Training {sum(p.numel() for p in trainable) / 1e6:.1f}M of "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters (LoRA)"
    )
    rng = random.Random(int(params["seed"]))
    k = len(scorer.keys)
    rows = [
        (r["input"]["state"], scorer.keys.index(gold_key(question, r["label"][qid])))
        for r in data["train"]
    ]
    batch_size = int(params["batch_size"])
    epochs = float(params["epochs"])
    total = steps_for(len(rows), batch_size, epochs)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(params["learning_rate"]), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, schedule(total, float(params["warmup_ratio"]))
    )
    step = 0
    history = []
    for epoch in range(int(-(-epochs // 1))):
        model.train()
        for batch in batches(rows, batch_size, seed=int(params["seed"]) + epoch):
            if step >= total:
                break
            report.check()
            orders = []
            for _ in batch:
                order = list(range(k))
                if params["shuffle_options"]:
                    rng.shuffle(order)
                orders.append(order)
            letter_logits, shown = scorer.forward([state for state, _ in batch], orders)
            targets = torch.tensor(
                [
                    keys.index(scorer.keys[gold])
                    for keys, (_, gold) in zip(shown, batch)
                ],
                device=device,
            )
            loss = F.cross_entropy(letter_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accuracy = (letter_logits.argmax(dim=-1) == targets).float().mean().item()
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
            metrics = evaluate_rows(scorer, qid, data["validation"], flips=False)
            history.append(
                {
                    "epoch": epoch + 1,
                    **{k_: metrics.get(k_) for k_ in ("accuracy", "macro_f1", "nll")},
                }
            )
            report.evaluation("validation", metrics, epoch=epoch + 1, step=step)

    model.eval()
    # Merge first: calibration and the "after" score are then those of the
    # model as saved and used (see decision_embed).
    merged = model.merge_and_unload()
    merged.eval()
    scorer.model = merged
    temperature = validation_temperature(scorer, qid, data.get("validation"))
    if data.get("validation"):
        report.status(f"Calibrated: temperature {temperature}")
    after = evaluate_rows(scorer, qid, evaluation_rows, temperature=temperature)
    report.evaluation("after", after)

    report.status("Saving the model")
    model_dir = out_dir / "model"
    merged.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    decision = {
        "kind": "decision",
        "backbone": "lm",
        "family": config["family"],
        "base_model": config["base_model"],
        "task": config.get("task"),
        "question_id": qid,
        "question": question,
        "temperature": temperature,
        "max_length": int(params["max_length"]),
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
    "LMScorer",
    "evaluate_rows",
    "letter_ids",
    "load_lm",
    "prompt_text",
    "train",
]
