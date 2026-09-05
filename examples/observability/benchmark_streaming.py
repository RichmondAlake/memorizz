"""Synthetic SDK timing matrix. No model/network/database performance claims."""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory

from memorizz import MemAgent
from memorizz.completion import CompletionPolicy
from memorizz.llms.streaming import tool_response
from memorizz.streaming import check_cancelled


class Provider:
    def generate_stream(self, messages, tools=None):
        if tools:
            called = any(m.get("role") == "tool" for m in messages)
            name = "memorizz_finalize_answer" if called else "lookup"
            yield {
                "type": "tool_calls",
                "response": tool_response(
                    [{"id": name, "name": name, "arguments": "{}"}],
                    "",
                ),
            }
            return
        for fragment in ("A ", "synthetic ", "answer."):
            check_cancelled()
            time.sleep(0.002)
            yield {"type": "content", "content": fragment}
        yield {"type": "done", "content": "A synthetic answer."}

    def get_last_usage(self):
        return None

    def get_context_window_tokens(self):
        return 8192


def build(case):
    agent = MemAgent(model=Provider(), auto_register=False)
    agent.memory_provider = None
    agent._build_context = lambda *a, **k: {}
    agent._build_system_prompt = lambda: "Synthetic timing fixture"
    agent._build_llm_tools = lambda *a, **k: []
    agent._record_interaction = lambda *a, **k: time.sleep(0.003)
    agent._init_workflow_capture = lambda *a, **k: None
    if case == "buffered":
        agent.completion_policy = CompletionPolicy(
            enabled=True, validator=lambda candidate: True
        )
    if case == "tools":
        agent._build_llm_tools = lambda *a, **k: [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        agent._execute_and_record_tool_call = (
            lambda call, messages, *a, **k: messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Synthetic evidence",
                }
            )
        )
    if case == "cache":
        agent.cache_manager.enabled = True
        agent.cache_manager.get_cached_response = lambda *a, **k: "A synthetic answer."
    return agent


def measure(case, agent=None):
    started = time.monotonic()
    agent = agent or build(case)
    setup = (time.monotonic() - started) * 1000
    events = []
    with agent.run_stream_events("Synthetic benchmark") as stream:
        for event in stream:
            events.append(event)
            if case == "cancelled" and event["type"] == "answer.delta":
                stream.cancellation.cancel()

    def at(kind, stage=None):
        return next(
            (
                e["elapsed_ms"] + setup
                for e in events
                if e["type"] == kind and (stage is None or e.get("stage") == stage)
            ),
            None,
        )

    answer, end = at("answer.done"), at("run.done")
    return {
        "agent_ready_ms": at("status", "agent_ready"),
        "context_ready_ms": at("status", "context_ready"),
        "provider_first_ms": at("status", "provider_first_delta"),
        "public_first_ms": at("answer.delta"),
        "answer_done_ms": answer,
        "run_done_ms": end,
        "finalization_ms": end - answer if answer is not None else None,
        "status": events[-1]["status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.samples <= 200:
        parser.error("samples must be 1–200")
    output = {
        "scope": "synthetic SDK only; 2ms fragment and 3ms persistence fixtures; no production SLA",
        "samples_per_case": args.samples,
        "cases": {},
    }
    for case in (
        "cold",
        "warm",
        "tools",
        "buffered",
        "cache",
        "concurrent",
        "cancelled",
    ):
        if case == "concurrent":
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(measure, [case] * args.samples))
        else:
            warm = build(case) if case == "warm" else None
            rows = [measure(case, warm) for _ in range(args.samples)]
        metrics = {}
        for key in rows[0]:
            if key == "status":
                continue
            values = sorted(row[key] for row in rows if row[key] is not None)
            metrics[key] = (
                {
                    label: round(
                        values[max(0, math.ceil(len(values) * quantile) - 1)], 3
                    )
                    for label, quantile in (("p50", 0.5), ("p95", 0.95))
                }
                if values
                else None
            )
        output["cases"][case] = {
            **metrics,
            "errors": sum(r["status"] == "error" for r in rows),
            "cancellations": sum(r["status"] == "cancelled" for r in rows),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with TemporaryDirectory(prefix="memorizz-stream-benchmark-") as root:
        os.environ["MEMORIZZ_HOME"] = root
        main()
