"""A Trainer callback that streams a Sentence Transformers / CrossEncoder
run's loss and evaluations into the run's events, and stops on cancel."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .common import Reporter


def reporter_callback(
    report: Reporter, on_epoch: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
):
    """``on_epoch`` scores the model on validation examples at the end of
    each epoch (the trainers here do no evaluation of their own), so a run's
    convergence can be seen afterwards."""
    from transformers import TrainerCallback

    class _Callback(TrainerCallback):
        canceled = False

        def on_epoch_end(self, args, state, control, **kwargs):
            if on_epoch is None or self.canceled:
                return
            metrics = on_epoch()
            if metrics:
                report.evaluation(
                    "validation",
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                    epoch=int(round(state.epoch or 0)),
                    step=int(state.global_step),
                )

        def on_log(self, args, state, control, logs: Dict[str, Any] = None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                report.step(
                    int(state.global_step),
                    int(state.max_steps or 0),
                    epoch=float(state.epoch or 0),
                    loss=float(logs["loss"]),
                    learning_rate=float(logs.get("learning_rate") or 0),
                )

        def on_evaluate(
            self, args, state, control, metrics: Dict[str, Any] = None, **kwargs
        ):
            report.evaluation(
                "validation",
                {
                    k: v
                    for k, v in (metrics or {}).items()
                    if isinstance(v, (int, float))
                },
                epoch=float(state.epoch or 0),
                step=int(state.global_step),
            )

        def on_step_end(self, args, state, control, **kwargs):
            if report.files.cancel_requested:
                self.canceled = True
                control.should_training_stop = True

    return _Callback()


__all__ = ["reporter_callback"]
