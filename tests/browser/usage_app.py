"""Synthetic usage/Evalground browser fixture; never reads user agent data."""

import os
from tempfile import TemporaryDirectory

import uvicorn

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers import evalground


def seed_usage(provider):
    for day in range(3):
        for agent in ("researcher", "writer"):
            identity = {
                "agent_id": agent,
                "thread_id": "thread",
                "run_id": "run",
                "root_trace_id": f"{agent}-day-{day}",
                "turn_id": "turn",
                "application_id": "fixture-app",
                "user_id": "fixture-user",
            }
            timestamp = f"2026-09-0{5 + day}T12:00:00Z"
            events = [
                {
                    "trace_kind": "model_result",
                    "event_id": "call",
                    "span_id": "call",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 1000,
                    "cached_tokens": 200,
                    "output_tokens": 100,
                    "duration_ms": 300,
                    "timestamp": timestamp,
                    "content": "PRIVATE SYNTHETIC PROMPT",
                }
            ]
            for category, chars, latency in (
                ("knowledge_base", 1200, 45),
                ("history", 400, 8),
                ("entity", 200, 12),
            ):
                events.extend(
                    [
                        {
                            "trace_kind": "memory_supply",
                            "event_id": f"supply-{category}",
                            "memory_type": category,
                            "memory_chars": chars,
                            "memory_tokens_estimate": chars // 4,
                            "timestamp": timestamp,
                        },
                        {
                            "trace_kind": "memory_retrieval",
                            "event_id": f"read-{category}",
                            "memory_type": category,
                            "duration_ms": latency,
                            "timestamp": timestamp,
                        },
                    ]
                )
            ObservabilityStore(provider).record_trace_bundle(
                trace_context=identity, events=events
            )
    evalground._eval_runs["usage-eval"] = {
        "run_id": "usage-eval",
        "status": "completed",
        "agent_id": "researcher",
        "created_at": "2026-09-07T12:00:00Z",
        "num_samples": 10,
        "eval_results": {
            "overall_accuracy": 0.8,
            "overall_score": 0.8,
            "metadata": {
                "num_samples": 10,
                "total_processing_time": 3.2,
                "external_api_cost_usd": 0.002,
                "dataset_variant": "fixture",
                "model": "fixture-model",
                "timestamp": "2026-09-07",
            },
            "category_results": {
                "recall": {"accuracy": 0.9, "average_score": 0.9, "num_samples": 5},
                "reasoning": {"accuracy": 0.7, "average_score": 0.7, "num_samples": 5},
            },
            "efficiency": {
                "average_retrieval_seconds": 0.025,
                "average_generation_seconds": 0.25,
            },
        },
    }


def run():
    with TemporaryDirectory(prefix="memorizz-usage-browser-") as root:
        os.environ.update(
            MEMORIZZ_UI_AUTH_TOKEN="memorizz-browser-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="true",
            MEMORIZZ_UI_TRACE_CONTENT_MODE="metadata",
            MEMORIZZ_UI_AUDIT_LOG=root + "/audit.jsonl",
            MEMORIZZ_OBSERVABILITY_DUAL_WRITE="false",
            MEMORIZZ_OBSERVABILITY_READ_PATH="bundles",
        )
        provider = FileSystemProvider(
            FileSystemConfig(root_path=root + "/memory", lazy_vector_indexes=True)
        )
        seed_usage(provider)
        state._state.update(
            provider=provider, provider_type="filesystem", connection_info={}
        )
        try:
            uvicorn.run(
                create_app(),
                host="127.0.0.1",
                port=int(os.getenv("MEMORIZZ_BROWSER_TEST_PORT", "8781")),
                lifespan="off",
            )
        finally:
            provider.close()


if __name__ == "__main__":
    run()
