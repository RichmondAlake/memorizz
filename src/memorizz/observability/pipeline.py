"""Process-local telemetry counters; never confused with deployment totals."""

import threading
import time
import weakref

_lock = threading.Lock()
_providers = weakref.WeakKeyDictionary()


def record_pipeline_metric(provider, name, *, duration_ms=None):
    if provider is None:
        return
    with _lock:
        try:
            metrics = _providers.setdefault(
                provider, {"since": time.time(), "counters": {}, "latencies_ms": {}}
            )
        except TypeError:
            return
        metrics["counters"][name] = metrics["counters"].get(name, 0) + 1
        if duration_ms is not None:
            samples = metrics["latencies_ms"].setdefault(name, [])
            samples.append(round(duration_ms, 3))
            del samples[:-100]


def pipeline_health(provider):
    with _lock:
        metrics = _providers.get(provider)
        if metrics is None:
            return {"scope": "ui_process_only", "reported": False, "counters": {}}
        return {
            "scope": "ui_process_only",
            "reported": True,
            "since": metrics["since"],
            "counters": dict(metrics["counters"]),
            "latencies_ms": {
                name: {
                    "samples": len(samples),
                    "average": round(sum(samples) / len(samples), 3),
                    "max": max(samples),
                }
                for name, samples in metrics["latencies_ms"].items()
                if samples
            },
        }
