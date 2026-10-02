"""Measure what the MemAgent tool cache saves, and check it changes no answer.

A support agent works through a queue of tickets, each in its own
conversation, against slow (simulated) back-end APIs. Several tickets need the
same order lookup or shipping quote, as real queues do. The same queue runs
twice with the same model: once without a tool cache and once with one.

    python examples/tool_cache/benchmark.py                 # local Ollama, qwen2.5:7b
    MODEL=gemma4:latest python examples/tool_cache/benchmark.py
    PROVIDER=openai MODEL=gpt-4.1-mini python examples/tool_cache/benchmark.py

It prints the time and tool calls each run took, the cache's hits and the tool
time it saved, and whether every answer still has the right facts. Results
also go to ``tool_cache_benchmark.json`` beside this file.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

from memorizz.memagent import MemAgent
from memorizz.tooling import governed_tool

PROVIDER = os.environ.get("PROVIDER", "ollama")
MODEL = os.environ.get("MODEL", "qwen2.5:7b")
ORDER_API_SECONDS = float(os.environ.get("ORDER_API_SECONDS", "1.5"))
QUOTE_API_SECONDS = float(os.environ.get("QUOTE_API_SECONDS", "1.0"))
# Arms run in A-B-B-A order after a warm-up, so model load time and drift
# don't favour either one.
ROUNDS = int(os.environ.get("ROUNDS", "2"))

ORDERS = {
    "1042": {
        "status": "shipped",
        "carrier": "UPS",
        "eta": "October 6",
        "items": "Ridgeline hiking boots",
    },
    "2077": {
        "status": "delivered",
        "carrier": "FedEx",
        "delivered_on": "October 1",
        "items": "Two-person tent",
    },
}

# Each ticket, and the facts a correct answer must contain.
TICKETS = [
    ("Where is my order 1042?", ["shipped", "ups"]),
    (
        "How much does standard shipping cost for a 2.5 kg parcel to ZIP 98101?",
        ["9.50"],
    ),
    ("Has order 1042 shipped yet, and when will it arrive?", ["shipped", "october 6"]),
    ("Quote me shipping for 2.5 kg to 98101, please.", ["9.50"]),
    ("What's the status of order 2077?", ["delivered"]),
    ("Did my order 2077 arrive? When?", ["delivered", "october 1"]),
    ("Can you check order 1042 for me again?", ["shipped"]),
    ("Please create a return label for order 2077.", ["rl-2077"]),
]

INSTRUCTION = (
    "You are the Fernvale Outfitters support assistant. Always use the tools to "
    "look up orders, quote shipping and create return labels; never guess. "
    "Answer in one or two sentences with the facts the tool returned."
)


class Backend:
    """The store's APIs, slow on purpose, counting every real call."""

    def __init__(self):
        self.calls = {"lookup_order": 0, "shipping_quote": 0, "create_return_label": 0}

    def tools(self):
        backend = self

        @governed_tool(cacheable=True, domains=["orders"])
        def lookup_order(order_id: str) -> dict:
            """Look up an order's status, carrier and dates by its order number."""
            backend.calls["lookup_order"] += 1
            time.sleep(ORDER_API_SECONDS)
            order = ORDERS.get(str(order_id).strip().lstrip("#"))
            return (
                {"order_id": order_id, **order}
                if order
                else {"error": "order not found"}
            )

        @governed_tool(cacheable=True, domains=["shipping"])
        def shipping_quote(destination_zip: str, weight_kg: float) -> dict:
            """Quote standard shipping (USD) for a parcel of the given weight to a US ZIP code."""
            backend.calls["shipping_quote"] += 1
            time.sleep(QUOTE_API_SECONDS)
            price = round(6.50 + 1.20 * float(weight_kg), 2)
            return {
                "destination_zip": destination_zip,
                "weight_kg": weight_kg,
                "price_usd": f"{price:.2f}",
            }

        @governed_tool(side_effects=True, requires_approval=False)
        def create_return_label(order_id: str) -> dict:
            """Create a prepaid return label for an order (each call makes a new label)."""
            backend.calls["create_return_label"] += 1
            return {
                "order_id": order_id,
                "label": f"RL-{order_id}-{uuid.uuid4().hex[:6].upper()}",
            }

        return [lookup_order, shipping_quote, create_return_label]


def run_queue(tool_cache: bool) -> dict:
    backend = Backend()
    agent = MemAgent(
        instruction=INSTRUCTION,
        llm_config={"provider": PROVIDER, "model": MODEL, "temperature": 0},
        tools=backend.tools(),
        memory_provider=False,
        tool_cache=tool_cache,
    )
    if agent.tool_cache is not None:
        agent.invalidate_tool_cache()  # start cold
    tickets = []
    started = time.perf_counter()
    for index, (question, facts) in enumerate(TICKETS, 1):
        t0 = time.perf_counter()
        answer = str(
            agent.run(question, thread_id=f"ticket-{index}", user_id="support-desk")
        )
        seconds = time.perf_counter() - t0
        outcomes = list(getattr(agent, "_last_tool_outcomes", None) or [])
        tickets.append(
            {
                "ticket": index,
                "question": question,
                "seconds": round(seconds, 2),
                "tools": [o.get("tool_name") for o in outcomes],
                "cache": [(o.get("cache") or {}).get("status") for o in outcomes],
                "correct": all(
                    fact in answer.lower().replace(",", "") for fact in facts
                ),
                "answer": answer,
            }
        )
        print(
            f"  [{'cache' if tool_cache else 'plain'}] ticket {index}: {seconds:5.1f}s  {tickets[-1]['tools']} {tickets[-1]['cache']}",
            flush=True,
        )
    total = time.perf_counter() - started
    stats = agent.tool_cache_stats()
    agent.close()
    return {
        "tool_cache": tool_cache,
        "seconds": round(total, 2),
        "backend_calls": backend.calls,
        "cache": {
            k: stats.get(k)
            for k in ("hits", "misses", "stored", "saved_ms", "hit_rate")
        },
        "correct": sum(t["correct"] for t in tickets),
        "tickets": tickets,
    }


def warm_up() -> None:
    """Load the model before timing anything."""
    agent = MemAgent(
        llm_config={"provider": PROVIDER, "model": MODEL, "temperature": 0},
        memory_provider=False,
    )
    agent.run("Reply with OK.")
    agent.close()


def average(runs: list) -> dict:
    """One arm's runs, averaged (times) and summed per run (calls, hits)."""
    first = runs[0]
    n = len(runs)
    return {
        **first,
        "seconds": round(sum(r["seconds"] for r in runs) / n, 2),
        "runs": [r["seconds"] for r in runs],
        "backend_calls": {
            k: sum(r["backend_calls"][k] for r in runs) / n
            for k in first["backend_calls"]
        },
        "cache": {
            "hits": sum(r["cache"]["hits"] or 0 for r in runs) / n,
            "saved_ms": sum(r["cache"]["saved_ms"] or 0 for r in runs) / n,
        },
        "correct": sum(r["correct"] for r in runs) / n,
    }


def main() -> int:
    print(
        f"Model: {PROVIDER}/{MODEL}; order API {ORDER_API_SECONDS}s, quote API {QUOTE_API_SECONDS}s; {ROUNDS} round(s)"
    )
    warm_up()
    arms = {False: [], True: []}
    for round_index in range(ROUNDS):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for tool_cache in order:
            arms[tool_cache].append(run_queue(tool_cache=tool_cache))
    plain, cached = average(arms[False]), average(arms[True])

    def calls(run: dict) -> float:
        return sum(run["backend_calls"].values())

    saved = plain["seconds"] - cached["seconds"]
    print()
    print(f"{'':28}{'no cache':>12}{'tool cache':>12}")
    print(f"{'Queue time (s)':28}{plain['seconds']:>12.1f}{cached['seconds']:>12.1f}")
    for name in plain["backend_calls"]:
        print(
            f"{'Real calls: ' + name:28}{plain['backend_calls'][name]:>12g}{cached['backend_calls'][name]:>12g}"
        )
    print(f"{'Cache hits':28}{'—':>12}{cached['cache']['hits']:>12g}")
    print(
        f"{'Tool time saved (s)':28}{'—':>12}{cached['cache']['saved_ms'] / 1000:>12.1f}"
    )
    print(
        f"{'Correct answers':28}{plain['correct']:>9.0f}/{len(TICKETS)}{cached['correct']:>9.0f}/{len(TICKETS)}"
    )
    print(f"{'Each run (s)':28}{str(plain['runs']):>12}{str(cached['runs']):>12}")
    print(
        f"\nPer run, the cache cut real API calls from {calls(plain):g} to {calls(cached):g} and the queue time by {saved:.1f}s ({saved / plain['seconds']:.0%})."
    )
    out = Path(__file__).with_name("tool_cache_benchmark.json")
    out.write_text(
        json.dumps(
            {
                "model": f"{PROVIDER}/{MODEL}",
                "rounds": ROUNDS,
                "plain": arms[False],
                "cached": arms[True],
            },
            indent=2,
        )
    )
    print(f"Details: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
