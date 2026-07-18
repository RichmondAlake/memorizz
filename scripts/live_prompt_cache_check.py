"""Live integration test: prompt caching + dedup + embedding backfill.

Runs multi-turn conversations through MemAgent against the LOCAL ORACLE
database, once with OpenAI and once with Anthropic, and asserts:

1. Turn >= 2 reports cached prompt tokens (prompt cache is actually hitting).
2. Conversation rows get embeddings backfilled asynchronously.
3. Episodic retrieval + dedup pipeline runs without errors.
4. Duplicate (query, response) pairs are not double-written.

Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, MEMORIZZ_ORACLE_PASSWORD env
vars (optional MEMORIZZ_ORACLE_USER / MEMORIZZ_ORACLE_DSN overrides); the
memorizz_oracle container healthy on localhost:1521.

Usage: python scripts/live_prompt_cache_check.py
"""

import logging
import os
import sys
import time
import uuid

from memorizz.llms.anthropic import Anthropic
from memorizz.llms.openai import OpenAI
from memorizz.memagent import MemAgent
from memorizz.memory_provider.oracle.provider import OracleConfig, OracleProvider

logging.basicConfig(level=logging.WARNING)
logging.getLogger("memorizz").setLevel(logging.INFO)

ORACLE_KW = dict(
    user=os.getenv("MEMORIZZ_ORACLE_USER", "memorizz_user"),
    password=os.environ["MEMORIZZ_ORACLE_PASSWORD"],
    dsn=os.getenv("MEMORIZZ_ORACLE_DSN", "localhost:1521/FREEPDB1"),
)

# Padding pushes the static prefix well past every provider's minimum
# cacheable length (OpenAI: 1024 tokens; Anthropic sonnet-4-5: 1024).
PAD = (
    "\n\nBackground reference (integration fixture): "
    + "MemoRizz is a memory layer for AI agents supporting conversational, "
    "episodic, semantic, procedural, and entity memory backed by MongoDB, "
    "Oracle, and filesystem providers. " * 60
)

TURNS = [
    "Hi! My name is Richmond and I'm building a memory library called MemoRizz. Please remember that.",
    "What's my name and what am I building? Answer in one short sentence.",
    "Summarize everything you know about me in one sentence.",
]

FAILURES = []


def check(label, cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}: {msg}")
    if not cond:
        FAILURES.append(f"{label}: {msg}")


def run_provider(label, llm):
    print(f"\n=== {label} ===")
    provider = OracleProvider(OracleConfig(**ORACLE_KW))
    agent = MemAgent(
        model=llm,
        memory_provider=provider,
        instruction="You are a concise assistant used for an integration test." + PAD,
        agent_id=f"cache-test-{label}-{uuid.uuid4().hex[:6]}",
        semantic_cache=False,
    )

    usages = []
    for i, turn in enumerate(TURNS, 1):
        t0 = time.time()
        resp = agent.run(turn)
        usage = dict(agent.model.get_last_usage() or {})
        usages.append(usage)
        print(
            f"  turn {i}: {time.time() - t0:.1f}s prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"cached={usage.get('cached_tokens', 0)} :: {str(resp)[:70]!r}"
        )

    check(
        f"{label} response quality",
        "richmond" in str(resp).lower() or "memorizz" in str(resp).lower(),
        f"final answer references remembered facts: {str(resp)[:80]!r}",
    )
    cached_later = [u.get("cached_tokens", 0) or 0 for u in usages[1:]]
    check(
        f"{label} prompt cache",
        any(c > 0 for c in cached_later),
        f"cached tokens on turns 2+: {cached_later}",
    )

    # Async embedding backfill: rows written with embedding=None get vectors.
    time.sleep(6)
    memory_id = agent._current_memory_id
    with provider._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) FROM conversation_memory WHERE memory_id = :m",
            {"m": memory_id},
        )
        total, embedded = cur.fetchone()
    check(
        f"{label} embedding backfill",
        total > 0 and embedded > 0,
        f"{embedded}/{total} conversation rows embedded",
    )

    # Write-path dedup: identical (query, response) not double-written.
    agent.model.generate = lambda messages, tools=None, **_: "dup-guard-answer"
    agent.run("dedup probe question")
    with provider._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversation_memory WHERE memory_id = :m",
            {"m": memory_id},
        )
        before = cur.fetchone()[0]
    agent.run("dedup probe question")
    with provider._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversation_memory WHERE memory_id = :m",
            {"m": memory_id},
        )
        after = cur.fetchone()[0]
    check(
        f"{label} write dedup",
        after == before,
        f"rows before={before} after identical repeat={after}",
    )
    return usages


run_provider("openai", OpenAI(model="gpt-4o-mini"))
run_provider("anthropic", Anthropic(model="claude-sonnet-4-5", max_tokens=512))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ALL LIVE CHECKS PASSED")
