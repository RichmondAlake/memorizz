"""Provider-neutral measurements used by Evalground comparisons."""

from __future__ import annotations

import math
import re
import time
from decimal import Decimal
from typing import Any, Callable

from ..llms.llm_factory import LOCAL_LLM_PROVIDERS
from .pricing import resolve_openai_text_pricing

# Local models plus scoring methods that make no API call.
LOCAL_PROVIDERS = set(LOCAL_LLM_PROVIDERS) | {"cross_encoder", "none", "heuristic"}
READER_PROVIDERS = {"openai", "anthropic", "ollama", "mlx", "huggingface", "azure"}


def json_safe(value: Any) -> Any:
    """Convert provider objects into deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return json_safe(model_dump(exclude_none=True))
        except Exception:
            pass
    return str(value)


def bounded_text(value: Any, limit: int) -> tuple[str, bool]:
    """Keep both ends of long command output while bounding model context
    growth; also says whether anything was cut."""
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} characters omitted]...\n{text[-tail:]}", True


def accepts_temperature(provider: str, model: str) -> bool:
    """Whether a greedy ``temperature=0`` default is valid for this model.

    Reasoning models reject sampling parameters with a 400: OpenAI GPT-5/6 and
    o-series, and Claude Sonnet 5, Opus 4.7+ and the Fable/Mythos families.
    Azure deployment names do not identify the model, so they keep the default.
    """
    name = str(model or "").lower()
    if provider == "openai":
        return not re.match(r"^(gpt-[56]|o[1-9])", name)
    if provider == "anthropic":
        return not re.match(r"^claude-(fable|mythos|sonnet-5|opus-5|opus-4-[78])", name)
    return True


def accepts_reasoning_effort(provider: str, model: str) -> bool:
    """Whether the model takes a reasoning-effort setting (OpenAI reasoning models)."""
    return provider == "openai" and bool(
        re.match(r"^(gpt-[56]|o[1-9])", str(model or "").lower())
    )


def token_price(provider: str, model: str, override: dict | None = None) -> dict | None:
    if override:
        price = {
            key: float(override[key]) for key in ("input", "cached_input", "output")
        }
        for key in ("cache_write", "cache_write_1h"):
            if key in override:
                price[key] = float(override[key])
        if any(not math.isfinite(v) or v < 0 for v in price.values()):
            raise ValueError("Token prices must be finite and non-negative")
        return {**price, "source": "experiment override"}
    if provider in LOCAL_PROVIDERS:
        return {
            "input": 0.0,
            "cached_input": 0.0,
            "output": 0.0,
            "source": "no external API charge",
        }
    if provider == "openai":
        try:
            item = resolve_openai_text_pricing(model)
        except ValueError:
            return None
        return {
            "input": item.input_per_million,
            "cached_input": item.cached_input_per_million,
            "output": item.output_per_million,
            "source": item.source_url,
            "as_of": item.as_of,
            "long_context_threshold": item.long_context_threshold,
            "long_input_multiplier": item.long_input_multiplier,
            "long_output_multiplier": item.long_output_multiplier,
            "cache_write": item.input_per_million * 1.25
            if re.match(r"^gpt-(?:5\.6|6)(?:-|$)", model)
            else None,
        }
    if provider == "jev":
        return {
            "input": 0.042,
            "cached_input": 0.042,
            "output": 0.0,
            "source": "https://docs.typesafe.ai/models",
            "as_of": "2026-09-23",
        }
    if provider == "voyage" and model in {"rerank-2.5", "rerank-2.5-lite", "rerank-3"}:
        rate = 0.02 if model == "rerank-2.5-lite" else 0.05
        return {
            "input": rate,
            "cached_input": rate,
            "output": 0.0,
            "source": "https://docs.voyageai.com/docs/pricing",
            "as_of": "2026-09-25",
        }
    if provider == "anthropic":
        rates = {
            "claude-haiku-4-5": (1, 0.1, 5),
            "claude-sonnet-4-5": (3, 0.3, 15),
            "claude-sonnet-4-6": (3, 0.3, 15),
            "claude-sonnet-5": (2, 0.2, 10),
            "claude-opus-4-5": (5, 0.5, 25),
            "claude-opus-4-6": (5, 0.5, 25),
            "claude-opus-4-7": (5, 0.5, 25),
            "claude-opus-4-8": (5, 0.5, 25),
            "claude-opus-5": (5, 0.5, 25),
            "claude-opus-5-5": (4, 0.2, 20),
            "claude-fable-5": (10, 1, 50),
            "claude-fable-5-1": (10, 0.25, 50),
        }
        for base in sorted(rates, key=len, reverse=True):
            if model == base or re.fullmatch(re.escape(base) + r"-\d{8}", model):
                i, c, o = rates[base]
                return {
                    "input": i,
                    "cached_input": c,
                    "output": o,
                    "cache_write": i * 1.25,
                    "cache_write_1h": i * 2,
                    "source": "https://platform.claude.com/docs/en/about-claude/pricing",
                    "as_of": "2026-09-24",
                    "scope": "standard global direct API; short context",
                }
    return None


def usage_cost(provider: str, usage: dict, price: dict | None) -> float | None:
    if provider in LOCAL_PROVIDERS:
        return 0.0
    if (
        not price
        or usage.get("prompt_tokens") is None
        or usage.get("completion_tokens") is None
    ):
        return None
    inputs = max(0, int(usage["prompt_tokens"]))
    cached = min(inputs, max(0, int(usage.get("cached_tokens") or 0)))
    outputs = max(0, int(usage["completion_tokens"]))
    long = bool(
        price.get("long_context_threshold") and inputs > price["long_context_threshold"]
    )
    im = price.get("long_input_multiplier", 1) if long else 1
    om = price.get("long_output_multiplier", 1) if long else 1
    write = (
        int(usage.get("cache_write_tokens") or 0)
        if provider in {"anthropic", "openai"}
        else 0
    )
    one_hour = (
        int(usage.get("cache_write_1h_tokens") or 0) if provider == "anthropic" else 0
    )
    if write < one_hour or inputs < cached + write:
        return None
    if (write and price.get("cache_write") is None) or (
        one_hour and price.get("cache_write_1h") is None
    ):
        return None
    # Legacy Claude long-context premiums require a separate rate rather than a guess.
    if (
        provider == "anthropic"
        and inputs > 200_000
        and "short context" in price.get("scope", "")
    ):
        return None

    def d(n):
        return Decimal(str(n))

    return float(
        (
            d(inputs - cached - write) * d(price["input"]) * d(im)
            + d(cached) * d(price["cached_input"]) * d(im)
            + d(write - one_hour) * d(price.get("cache_write") or 0) * d(im)
            + d(one_hour) * d(price.get("cache_write_1h", 0))
            + d(outputs) * d(price["output"]) * d(om)
        )
        / d(1_000_000)
    )


class MeasurementLedger:
    def __init__(
        self,
        *,
        check: Callable[[], None] | None = None,
        on_append: Callable[[], None] | None = None,
    ):
        self.calls: list[dict] = []
        self.check = check or (lambda: None)
        self.lane = "reader"
        self.case_id = ""
        self.on_append = on_append or (lambda: None)

    def append(
        self,
        *,
        provider: str,
        model: str,
        seconds: float,
        usage: dict,
        price: dict | None,
        status: str = "completed",
        ttft: float | None = None,
        cost: float | None = None,
        **extra: Any,
    ) -> None:
        self.calls.append(
            {
                "case_id": self.case_id,
                "lane": self.lane,
                "provider": provider,
                "model": model,
                "seconds": seconds,
                "ttft_seconds": ttft,
                "status": status,
                "usage": usage,
                "pricing": price,
                "cost_usd": cost
                if cost is not None
                else (
                    usage_cost(provider, usage, price)
                    if status == "completed"
                    or (
                        usage.get("prompt_tokens") is not None
                        and usage.get("completion_tokens") is not None
                    )
                    else None
                ),
                **extra,
            }
        )
        self.on_append()


class MeasuredModel:
    """Measure each call (including repair, failed calls, and judge overhead)."""

    def __init__(self, spec: dict, ledger: MeasurementLedger, *, model: Any = None):
        from ..llms.llm_factory import create_llm_provider

        self.spec = spec
        self.ledger = ledger
        self.price = token_price(spec["provider"], spec["model"], spec.get("pricing"))
        if model is None:
            config = {"provider": spec["provider"], "model": spec["model"]}
            config.update(spec.get("options") or {})
            if accepts_temperature(spec["provider"], spec["model"]):
                config.setdefault("temperature", 0)
            if spec["provider"] == "ollama":
                config.setdefault("num_predict", 512)
                config.setdefault("think", False)
                config.setdefault("seed", 0)
                config.setdefault("timeout", 120)
            elif spec["provider"] == "openai":
                config.setdefault("api_mode", "responses")
                if accepts_reasoning_effort("openai", spec["model"]):
                    config.setdefault("reasoning_effort", "low")
                config.setdefault("max_completion_tokens", 512)
            elif spec["provider"] == "anthropic":
                config.setdefault("max_tokens", 512)
            elif spec["provider"] == "azure":
                config["deployment_name"] = config.pop("model")
            model = create_llm_provider(config)
        self.wrapped = model
        self.last_usage: dict = {}

    def generate_text(self, prompt: str, instructions: str | None = None) -> str:
        self.ledger.check()
        started = time.perf_counter()
        ttft = None
        usage = {}
        status = "failed"
        metadata = {}
        try:
            if self.spec.get("stream", False):
                messages = (
                    [{"role": "system", "content": instructions}]
                    if instructions
                    else []
                )
                messages.append({"role": "user", "content": prompt})
                chunks = []
                complete = False
                stream = self.wrapped.generate_stream(messages)
                try:
                    for event in stream:
                        self.ledger.check()
                        if event.get("type") == "content":
                            if event.get("content") and ttft is None:
                                ttft = time.perf_counter() - started
                            chunks.append(event.get("content", ""))
                        elif event.get("type") == "usage":
                            usage = dict(event.get("usage") or {})
                        elif event.get("type") == "done":
                            complete = True
                            output = event.get("content", "".join(chunks))
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if not complete:
                    raise RuntimeError("Model stream ended without a completed answer")
            else:
                output = self.wrapped.generate_text(prompt, instructions=instructions)
            getter = getattr(self.wrapped, "get_last_usage", None)
            if not usage and callable(getter):
                usage = dict(getter() or {})
            self.last_usage = usage
            status = "completed"
            return output
        finally:
            getter = getattr(self.wrapped, "get_last_response_metadata", None)
            if callable(getter):
                raw_metadata = getter() or {}
                metadata = {
                    k: raw_metadata[k]
                    for k in (
                        "response_model",
                        "service_tier",
                        "finish_reason",
                        "request_id",
                    )
                    if k in raw_metadata
                }
            if not usage and status == "failed":
                getter = getattr(self.wrapped, "get_last_usage", None)
                if callable(getter):
                    usage = dict(getter() or {})
            price = self.price
            if metadata.get("service_tier") not in {
                None,
                "default",
                "standard",
            } and not self.spec.get("pricing"):
                price = None
            self.ledger.append(
                provider=self.spec["provider"],
                model=self.spec["model"],
                seconds=time.perf_counter() - started,
                usage=usage,
                price=price,
                status=status,
                ttft=ttft,
                response_metadata=metadata,
            )

    def get_last_usage(self) -> dict:
        return self.last_usage

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def summarize_calls(calls: list[dict], *, local_hourly_usd: float = 0) -> dict:
    from .memory_suite.runner import _percentile

    def total_cost(rows):
        return (
            None
            if any(x["cost_usd"] is None for x in rows)
            else sum(x["cost_usd"] for x in rows)
        )

    serving = [c for c in calls if c["lane"] in {"reader", "reranker"}]
    overhead = [c for c in calls if c["lane"] not in {"reader", "reranker"}]
    token_keys = (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cache_write_tokens",
        "cache_write_1h_tokens",
    )
    tokens = {k: sum(int(c["usage"].get(k) or 0) for c in calls) for k in token_keys}

    def cache_calls(rows):
        # Hit rate is measured only over calls whose provider reported cache use.
        return {
            "cache_reported_calls": sum(
                c["usage"].get("cached_tokens") is not None for c in rows
            ),
            "cache_hit_calls": sum(
                int(c["usage"].get("cached_tokens") or 0) > 0 for c in rows
            ),
        }

    local_seconds = sum(c["seconds"] for c in calls if c["provider"] in LOCAL_PROVIDERS)
    ttft = [
        c["ttft_seconds"]
        for c in serving
        if c["lane"] == "reader" and c.get("ttft_seconds") is not None
    ]
    generation = [
        c for c in serving if c["lane"] == "reader" and c["status"] == "completed"
    ]
    output = sum(int(c["usage"].get("completion_tokens") or 0) for c in generation)
    duration = sum(c["seconds"] for c in generation)
    return {
        **tokens,
        **cache_calls(calls),
        "calls": len(calls),
        "failed_calls": sum(c["status"] != "completed" for c in calls),
        "total_cost_usd": total_cost(calls),
        "serving_cost_usd": total_cost(serving),
        "evaluation_cost_usd": total_cost(overhead),
        "known_cost_usd": sum(
            c["cost_usd"] for c in calls if c["cost_usd"] is not None
        ),
        "unpriced_calls": sum(c["cost_usd"] is None for c in calls),
        "usage_complete": all(
            c["usage"].get("prompt_tokens") is not None
            and c["usage"].get("completion_tokens") is not None
            for c in calls
        ),
        "local_compute_estimate_usd": local_seconds / 3600 * local_hourly_usd
        if local_hourly_usd
        else None,
        "ttft_p50_seconds": _percentile(ttft, 0.5),
        "ttft_p95_seconds": _percentile(ttft, 0.95),
        "output_tokens_per_second": output / duration
        if duration
        and output
        and all(c["usage"].get("completion_tokens") is not None for c in generation)
        else None,
        "lanes": {
            lane: {
                "calls": len(rows),
                "cost_usd": total_cost(rows),
                **cache_calls(rows),
                **{
                    k: sum(int(c["usage"].get(k) or 0) for c in rows)
                    for k in token_keys
                },
            }
            for lane in dict.fromkeys(c["lane"] for c in calls)
            for rows in [[c for c in calls if c["lane"] == lane]]
        },
    }
