"""Mechanism probes for the context-efficiency work (live, Oracle-backed).

Five targeted scenarios validating that the cache/dedup mechanisms don't
hurt — and in places improve — long-horizon accuracy:

1. eviction-boundary recall   — fact pushed out of the history window must
                                 come back via episodic vector recall
2. dedup false-positive       — similar-but-distinct facts must BOTH survive
                                 the cosine/containment dedup filters
3. knowledge update           — the newer of two conflicting facts must win
4. cache neutrality           — Anthropic answers with prompt caching on vs
                                 off must both contain the planted fact
5. repeat-turn write guard    — identical re-asked questions with different
                                 intent must still be recorded and answered
                                 from the corrected fact

Requires: OPENAI_API_KEY (and ANTHROPIC_API_KEY for probe 4),
MEMORIZZ_ORACLE_PASSWORD (defaults match the local dev container), the
memorizz_oracle container healthy.

Usage: python scripts/memory_accuracy_probes.py
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timedelta

from memorizz.embeddings import get_embedding
from memorizz.enums import MemoryType, Role
from memorizz.llms.openai import OpenAI
from memorizz.memagent import MemAgent
from memorizz.memory_provider.oracle.provider import OracleConfig, OracleProvider

logging.basicConfig(level=logging.WARNING)

ORACLE_KW = dict(
    user=os.getenv("MEMORIZZ_ORACLE_USER", "memorizz_user"),
    password=os.getenv("MEMORIZZ_ORACLE_PASSWORD", "SecurePass123!"),
    dsn=os.getenv("MEMORIZZ_ORACLE_DSN", "localhost:1521/FREEPDB1"),
)

FAILURES = []


def check(label, cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {msg}")
    if not cond:
        FAILURES.append(label)


def make_agent(llm=None, **kwargs):
    provider = OracleProvider(OracleConfig(**ORACLE_KW))
    agent = MemAgent(
        model=llm or OpenAI(model="gpt-4o-mini"),
        memory_provider=provider,
        instruction=(
            "You are a concise assistant with excellent memory. Answer from "
            "what you know about this conversation; use your memory tools "
            "when the answer isn't in view."
        ),
        agent_id=f"probe-{uuid.uuid4().hex[:8]}",
        semantic_cache=False,
        **kwargs,
    )
    return agent, provider


def write_row(agent, role, content, memory_id, thread_id, ts, embed=True):
    """Write a conversation row directly (fast history seeding, like real
    accumulated history) and optionally embed it for episodic recall."""
    unit = agent.memory_manager.create_conversation_memory_unit(
        role=role,
        content=content,
        thread_id=thread_id,
        memory_id=memory_id,
        timestamp=ts,
        agent_id=agent.agent_id,
    )
    unit_id = agent.memory_manager.save_memory_unit(unit, memory_id)
    if unit_id and embed:
        try:
            agent.memory_provider.update_by_id(
                unit_id,
                {"embedding": get_embedding(content)},
                MemoryType.CONVERSATION_MEMORY,
            )
        except Exception:
            pass
    return unit_id


FILLER_TOPICS = [
    "the weather forecast for the weekend and whether to pack an umbrella",
    "a recipe for sourdough bread and hydration ratios",
    "training plans for a half marathon in autumn",
    "the difference between index funds and ETFs",
    "houseplant care for a monstera with yellowing leaves",
    "planning a road trip along the coast with two overnight stops",
    "learning Spanish with spaced repetition flashcards",
    "fixing a leaking kitchen tap without calling a plumber",
    "choosing a standing desk for a small home office",
    "the plot of a mystery novel set in a lighthouse",
]


def seed_history(agent, memory_id, thread_id, facts, filler_turns=60):
    """Seed planted facts early, then bury them under filler rows."""
    base = datetime.now() - timedelta(hours=6)
    step = 0
    for fact in facts:
        write_row(
            agent, Role.USER, fact, memory_id, thread_id, base + timedelta(seconds=step)
        )
        step += 1
        write_row(
            agent,
            Role.ASSISTANT,
            "Got it — noted.",
            memory_id,
            thread_id,
            base + timedelta(seconds=step),
        )
        step += 1
    for i in range(filler_turns):
        topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
        # Long filler so token budgets bite, forcing real eviction.
        user = (f"Let's talk more about {topic}. " * 4).strip()
        asst = (
            f"Sure — here are several detailed thoughts about {topic}, "
            "with considerations, trade-offs, and next steps. " * 3
        ).strip()
        write_row(
            agent,
            Role.USER,
            user,
            memory_id,
            thread_id,
            base + timedelta(seconds=step),
            embed=(i % 3 == 0),
        )
        step += 1
        write_row(
            agent,
            Role.ASSISTANT,
            asst,
            memory_id,
            thread_id,
            base + timedelta(seconds=step),
            embed=False,
        )
        step += 1
    agent.memory_manager.clear_conversation_cache(memory_id)


def scoped(agent):
    return agent._resolve_execution_state(None, None)


def count_rows(provider, memory_id):
    with provider._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversation_memory WHERE memory_id = :m",
            {"m": memory_id},
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
print("\n=== Probe 1: eviction-boundary recall ===")
agent, provider = make_agent()
agent._context_window_tokens = 8000  # tight budget → history window evicts
memory_id, thread_id = scoped(agent)
seed_history(
    agent,
    memory_id,
    thread_id,
    ["Important: the storage unit access code is 7391-alpha. Please remember it."],
    filler_turns=60,
)
answer = agent.run(
    "What is the storage unit access code?", memory_id=memory_id, thread_id=thread_id
)
window = agent._prepare_history_messages(
    agent.memory_manager.load_conversation_history(memory_id, limit=200),
    "sys",
    "q",
)
in_window = any("7391-alpha" in str(m.get("content", "")) for m in window)
check(
    "eviction recall",
    "7391-alpha" in str(answer),
    f"fact in window={in_window}; answer: {str(answer)[:80]!r}",
)

# ---------------------------------------------------------------------------
print("\n=== Probe 2: dedup false-positive (similar-but-distinct facts) ===")
agent, provider = make_agent()
agent._context_window_tokens = 8000
memory_id, thread_id = scoped(agent)
seed_history(
    agent,
    memory_id,
    thread_id,
    [
        "For the record: the budget for Project Apollo is $500,000.",
        "For the record: the budget for Project Artemis is $550,000.",
    ],
    filler_turns=60,
)
answer = str(
    agent.run(
        "What are the budgets for Project Apollo and Project Artemis?",
        memory_id=memory_id,
        thread_id=thread_id,
    )
)
has_apollo = "500,000" in answer or "500000" in answer or "$500" in answer
has_artemis = "550,000" in answer or "550000" in answer or "$550" in answer
check(
    "dedup keeps distinct facts",
    has_apollo and has_artemis,
    f"apollo={has_apollo} artemis={has_artemis}; answer: {answer[:100]!r}",
)

# ---------------------------------------------------------------------------
print("\n=== Probe 3: knowledge update (newer fact wins) ===")
agent, provider = make_agent()
agent._context_window_tokens = 8000
memory_id, thread_id = scoped(agent)
base = datetime.now() - timedelta(hours=8)
write_row(agent, Role.USER, "My dog is named Biscuit.", memory_id, thread_id, base)
write_row(
    agent,
    Role.ASSISTANT,
    "Biscuit — lovely name!",
    memory_id,
    thread_id,
    base + timedelta(seconds=1),
)
seed_history(agent, memory_id, thread_id, [], filler_turns=25)
later = datetime.now() - timedelta(hours=1)
write_row(
    agent,
    Role.USER,
    "Update: we renamed our dog — she's called Waffle now, not Biscuit.",
    memory_id,
    thread_id,
    later,
)
write_row(
    agent,
    Role.ASSISTANT,
    "Understood, Waffle it is.",
    memory_id,
    thread_id,
    later + timedelta(seconds=1),
)
seed_history(agent, memory_id, thread_id, [], filler_turns=25)
answer = str(
    agent.run("What is my dog's name now?", memory_id=memory_id, thread_id=thread_id)
)
check("knowledge update", "waffle" in answer.lower(), f"answer: {answer[:100]!r}")

# ---------------------------------------------------------------------------
print("\n=== Probe 4: cache neutrality (Anthropic, caching on vs off) ===")
if not os.getenv("ANTHROPIC_API_KEY"):
    print("  [SKIP] ANTHROPIC_API_KEY not set")
else:
    from memorizz.llms.anthropic import Anthropic

    answers = {}
    for label, enabled in (("cache-on", True), ("cache-off", False)):
        agent, provider = make_agent(
            llm=Anthropic(
                model="claude-sonnet-4-5", max_tokens=300, enable_prompt_caching=enabled
            ),
        )
        memory_id, thread_id = scoped(agent)
        agent.run(
            "My sister's name is Amara and she lives in Lagos.",
            memory_id=memory_id,
            thread_id=thread_id,
        )
        answers[label] = str(
            agent.run(
                "Where does my sister live, and what's her name?",
                memory_id=memory_id,
                thread_id=thread_id,
            )
        ).lower()
    ok = all("amara" in a and "lagos" in a for a in answers.values())
    usage = {}
    check(
        "cache neutrality",
        ok,
        f"on: {answers['cache-on'][:60]!r} | off: {answers['cache-off'][:60]!r}",
    )

# ---------------------------------------------------------------------------
print("\n=== Probe 5: repeat-turn write guard ===")
agent, provider = make_agent()
memory_id, thread_id = scoped(agent)
agent.run("My name is Richmond.", memory_id=memory_id, thread_id=thread_id)
a1 = str(agent.run("What's my name?", memory_id=memory_id, thread_id=thread_id))
rows_after_first = count_rows(provider, memory_id)
agent.run(
    "Actually, please call me Rich from now on.",
    memory_id=memory_id,
    thread_id=thread_id,
)
a2 = str(agent.run("What's my name?", memory_id=memory_id, thread_id=thread_id))
rows_after_second = count_rows(provider, memory_id)
check(
    "repeat question answered from corrected fact",
    "rich" in a2.lower(),
    f"first: {a1[:50]!r} | second: {a2[:50]!r}",
)
check(
    "legit repeat still recorded",
    rows_after_second > rows_after_first,
    f"rows {rows_after_first} -> {rows_after_second}",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} PROBE FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL MECHANISM PROBES PASSED")
