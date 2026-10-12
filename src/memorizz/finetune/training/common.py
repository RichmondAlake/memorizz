"""What every trainer shares: the device, live progress, cancellation."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Iterable, List, Optional

from ..store import RunFiles


class Canceled(Exception):
    """The run was asked to stop."""


def pick_device(prefer: Optional[str] = None) -> str:
    import torch

    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_dtype(device: str, precision: str = "auto"):
    """Weights in bf16 on a GPU (CUDA with bf16 support, or Apple's), which
    halves their memory; fp32 on CPU or when asked. Never fp16:
    EmbeddingGemma returns NaN in fp16. LoRA adapters and the loss stay in
    fp32 either way."""
    import torch

    if precision == "fp32" or device == "cpu":
        return torch.float32
    if precision == "bf16" or device == "mps":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class Reporter:
    """Writes a run's progress to its folder for the UI, CLI and MCP."""

    def __init__(self, files: RunFiles, *, every: int = 1):
        self.files = files
        self.every = max(1, int(every))
        self.started = time.monotonic()
        self.state: Dict[str, Any] = {}

    def status(self, message: str, **extra: Any) -> None:
        self.files.append_event("status", {"message": message, **extra})
        self._state(message=message, **extra)

    def step(
        self, step: int, total: int, *, epoch: float, loss: float, **extra: Any
    ) -> None:
        elapsed = time.monotonic() - self.started
        eta = (elapsed / step) * (total - step) if step else None
        if step == 1 or step % self.every == 0 or step == total:
            self.files.append_event(
                "step",
                {
                    "step": step,
                    "total": total,
                    "epoch": round(epoch, 3),
                    "loss": round(float(loss), 5),
                    **{
                        k: (round(v, 8) if isinstance(v, float) else v)
                        for k, v in extra.items()
                    },
                },
            )
        self._state(
            step=step,
            total=total,
            epoch=round(epoch, 3),
            loss=round(float(loss), 5),
            eta_seconds=round(eta, 1) if eta else None,
        )

    def evaluation(self, phase: str, metrics: Dict[str, Any], **extra: Any) -> None:
        self.files.append_event("eval", {"phase": phase, "metrics": metrics, **extra})

    def check(self) -> None:
        if self.files.cancel_requested:
            raise Canceled("Training was canceled")

    def _state(self, **values: Any) -> None:
        self.state.update(values)
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 1)
        self.files.write_state({"phase": "training", **self.state})


def batches(
    items: List[Any], size: int, *, seed: int, shuffle: bool = True
) -> Iterable[List[Any]]:
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield [items[i] for i in order[start : start + size]]


def schedule(total_steps: int, warmup_ratio: float):
    """Linear warmup then linear decay, as a LambdaLR factor."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return factor


def steps_for(rows: int, batch_size: int, epochs: float) -> int:
    return max(1, math.ceil(rows / batch_size) * int(math.ceil(epochs)))


__all__ = [
    "Canceled",
    "Reporter",
    "batches",
    "pick_device",
    "schedule",
    "steps_for",
    "synchronize",
    "train_dtype",
]
