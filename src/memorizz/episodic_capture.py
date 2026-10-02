"""Save a coding agent's turns as episodic memory, and summarize a session.

Used by the Codex and Claude Code plugins (their hooks, locally) and by the
MCP server (``memorizz_record_turn`` and ``memorizz_summarize_session``, for a
hosted server). Turns are stored as conversation memory in the same shape
MemAgent writes, so the MemoRizz UI and agents read them like any other
conversation; secrets are removed first.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Longest message text kept per turn.
TURN_CHARS = 20_000


def record_turn(
    provider: Any,
    *,
    memory_id: str,
    thread_id: str,
    user_message: str,
    assistant_message: str,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[str]:
    """Store one turn (request and answer) and return the two record IDs."""
    from .enums import Role
    from .memagent.managers.memory_manager import MemoryManager
    from .metaharness.security import redact

    manager = MemoryManager(provider)
    saved: List[str] = []
    for role, text in ((Role.USER, user_message), (Role.ASSISTANT, assistant_message)):
        clean = str(redact(str(text or "").strip()))
        if not clean:
            continue
        if len(clean) > TURN_CHARS:
            clean = clean[:TURN_CHARS] + "…"
        unit = manager.create_conversation_memory_unit(
            role=role,
            content=clean,
            thread_id=thread_id,
            memory_id=memory_id,
            agent_id=agent_id,
            user_id=user_id,
        )
        try:
            from .embeddings import get_embedding

            unit.embedding = get_embedding(clean[:8_000])
        except Exception:
            pass  # Found by thread and keywords without one.
        record_id = manager.save_memory_unit(unit, memory_id)
        if record_id:
            saved.append(str(record_id))
    return saved


def default_model() -> Any:
    """MemoRizz's configured default LLM, or None when none is configured."""
    from .cli import agent_factory
    from .llms.llm_factory import create_llm_provider

    try:
        config = agent_factory.detect_llm_config()
        return create_llm_provider(config) if config else None
    except Exception:
        return None


SESSION_SUMMARY_PROMPT = (
    "Summarize this coding session so the next session can pick up where it "
    "left off. In at most 8 short bullets: what was asked, what changed (files, "
    "commands, migrations), decisions and their reasons, and open issues or next "
    "steps. Only facts from the transcript; no secrets; no commentary."
)


def _thread_turns(
    provider: Any, memory_id: str, thread_id: str, user_id: Optional[str]
) -> List[dict]:
    from .enums import MemoryType

    rows = [
        row
        for row in provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        if isinstance(row, dict)
        and row.get("memory_id") == memory_id
        and row.get("thread_id") == thread_id
        and (user_id is None or row.get("user_id") == user_id)
    ]
    rows.sort(key=lambda row: str(row.get("timestamp") or ""))
    return rows


def summarize_thread(
    provider: Any,
    *,
    memory_id: str,
    thread_id: str,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    model: Any = None,
) -> List[str]:
    """Summarize a coding session's turns into the summaries store (one summary
    per call, covering the whole thread) and return its ID. Nothing is made
    without a configured model or without turns."""
    from datetime import datetime, timezone

    from .enums import MemoryType

    turns = _thread_turns(provider, memory_id, thread_id, user_id)
    if not turns:
        return []
    model = model or default_model()
    if model is None:
        return []
    transcript = "\n".join(
        f"{str(turn.get('role') or 'user').upper()}: {str(turn.get('content') or '')[:4_000]}"
        for turn in turns
    )[-40_000:]
    text = model.generate(
        [
            {"role": "system", "content": SESSION_SUMMARY_PROMPT},
            {"role": "user", "content": transcript},
        ]
    )
    summary = str(text or "").strip()
    if not summary:
        return []
    embedding = None
    try:
        from .embeddings import get_embedding

        embedding = get_embedding(summary)
    except Exception:
        pass
    now = datetime.now(timezone.utc).isoformat()
    record_id = provider.store(
        {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "content": summary,
            "period_start": turns[0].get("timestamp"),
            "period_end": turns[-1].get("timestamp"),
            "memory_units_count": len(turns),
            "source_message_ids": [
                str(turn.get("_id") or turn.get("id") or "") for turn in turns
            ],
            "summary_type": "session",
            "created_at": now,
            "embedding": embedding,
        },
        MemoryType.SUMMARIES,
    )
    return [str(record_id)] if record_id else []


__all__ = [
    "SESSION_SUMMARY_PROMPT",
    "TURN_CHARS",
    "default_model",
    "record_turn",
    "summarize_thread",
]
