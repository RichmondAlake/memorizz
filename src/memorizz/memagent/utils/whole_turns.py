"""Recall a conversation turn whole: a request with its reply.

Each turn is stored, and embedded, as two rows. A new question is far closer
in meaning to earlier *questions* than to their answers, so a search for "what
was the email's subject line?" returned the requests to write the email, and
the agent's own earlier attempts at the question, and never the email. In a
memory several agents share that made a colleague's answer unfindable.

``whole_turns`` gives each recalled message the other half of its turn: a
request brings the reply that followed it, a reply brings the request it
answered, and two halves of one turn become one candidate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: A request is context for its reply; the reply is what the reader needs.
_REQUEST_CHARS = 240


def _text(row: Dict[str, Any]) -> str:
    content = row.get("content")
    if isinstance(content, dict):
        content = content.get("content") or content.get("text")
    value = row.get("text") if isinstance(row.get("text"), str) else content
    return str(value or "").strip()


def _role(row: Dict[str, Any]) -> str:
    content = row.get("content")
    nested = content.get("role") if isinstance(content, dict) else None
    return str(row.get("role") or nested or "").strip().lower()


def _id(row: Dict[str, Any]) -> str:
    return str(row.get("_id") or row.get("id") or "")


def _score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _partner(
    thread: Sequence[Dict[str, Any]], index: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The (request, reply) of the turn the row at ``index`` belongs to."""
    row = thread[index]
    if _role(row) == "user":
        for later in thread[index + 1 :]:
            role = _role(later)
            if role == "assistant":
                return row, later
            if role == "user":
                break
        return row, None
    if _role(row) == "assistant":
        for earlier in reversed(thread[:index]):
            role = _role(earlier)
            if role == "user":
                return earlier, row
            if role == "assistant":
                break
        return None, row
    return None, None


def whole_turns(
    rows: Sequence[Dict[str, Any]],
    read_thread: Callable[[str, str], Sequence[Dict[str, Any]]],
    *,
    skip_thread: Optional[str] = None,
    max_threads: int = 5,
    reader_agent_id: Optional[str] = None,
    own_turns: int = 1,
) -> List[Dict[str, Any]]:
    """Recalled conversation rows, each joined with the rest of its turn.

    ``read_thread(memory_id, thread_id)`` returns a thread's rows, oldest
    first. At most ``max_threads`` threads are read, those of the best
    matches; rows of other threads, of ``skip_thread`` (the conversation going
    on now, whose turns are already in the window) and rows with no partner
    are returned as they came.

    With ``reader_agent_id`` a turn says whose reply it was, the reader's own
    or another agent's. In a memory several agents share, "I couldn't find
    it" from the reader's own earlier attempt and the thing itself, written
    by a colleague, are otherwise the same kind of line. And where other
    agents' turns were recalled, only the reader's best ``own_turns`` of its
    own are kept (-1 keeps them all).
    """
    ranked = sorted(
        (row for row in rows if isinstance(row, dict)), key=_score, reverse=True
    )
    threads: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
    out: List[Dict[str, Any]] = []
    made: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    # One exchange is often on record twice, in the thread of the agent that
    # asked and the thread of the one that answered: the same reply, once.
    said: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in ranked:
        thread_id = str(row.get("thread_id") or "")
        memory_id = str(row.get("memory_id") or "")
        key = (memory_id, thread_id)
        if (
            not thread_id
            or thread_id == str(skip_thread or "")
            or _role(row) not in {"user", "assistant"}
        ):
            out.append(row)
            continue
        if key not in threads:
            if sum(found is not None for found in threads.values()) >= max_threads:
                out.append(row)
                continue
            try:
                threads[key] = [
                    item
                    for item in read_thread(memory_id, thread_id) or []
                    if isinstance(item, dict)
                ]
            except Exception:
                threads[key] = None
        thread = threads[key]
        index = next(
            (i for i, item in enumerate(thread or []) if _id(item) == _id(row)), None
        )
        if index is None:
            out.append(row)
            continue
        request, reply = _partner(thread, index)
        if request is None or reply is None:
            out.append(row)
            continue
        unit_key = (memory_id, thread_id, _id(reply))
        unit = made.get(unit_key)
        if unit is not None:
            # The other half of a turn already recalled: one candidate, with
            # the better of the two matches.
            if _score(row) > _score(unit):
                unit["score"] = _score(row)
                if row.get("embedding") is not None:
                    unit["embedding"] = row["embedding"]
            continue
        writer = str(reply.get("agent_id") or "")
        same = said.get((memory_id, " ".join(_text(reply).split())))
        if same is not None:
            if _score(row) > _score(same):
                same["score"] = _score(row)
                if row.get("embedding") is not None:
                    same["embedding"] = row["embedding"]
            made[unit_key] = same
            continue
        unit = dict(reply)
        if not reader_agent_id or not writer:
            answered = "Answered"
        elif writer == str(reader_agent_id):
            # Its own "I couldn't find it" is not evidence that the thing is
            # missing: a model that reads it as a record repeats it for good.
            answered = (
                "Your own answer at the time (not a record; what you can see "
                "now may hold more)"
            )
        else:
            answered = "Answered by another agent"
        unit["text"] = (
            f"Asked: {_clip(_text(request), _REQUEST_CHARS)}\n"
            f"{answered}: {_text(reply)}"
        )
        unit["score"] = _score(row)
        unit["embedding"] = row.get("embedding")
        unit["turn"] = {
            "request_id": _id(request),
            "reply_id": _id(reply),
            "by_reader": bool(reader_agent_id) and writer == str(reader_agent_id),
        }
        importance = [
            value
            for value in (request.get("importance"), reply.get("importance"))
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if importance:
            unit["importance"] = max(importance)
        made[unit_key] = unit
        said[(memory_id, " ".join(_text(reply).split()))] = unit
        out.append(unit)
    return _others_first(out, own_turns)


def with_turn_partners(
    rows: Sequence[Dict[str, Any]],
    read_thread: Callable[[str, str], Sequence[Dict[str, Any]]],
    *,
    max_threads: int = 10,
    visible: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Search results kept as the records they are, each conversation message
    given the other half of its turn.

    For an explicit search, where the caller gets records back (IDs it can
    fetch, correct or forget) and not prompt text. A request gains
    ``turn["reply"]`` (``id``, ``content``, ``agent_id``) and a reply gains
    ``turn["request"]`` (``id``, ``content``). A turn found by both halves
    appears once, at the place of the better one. Order is kept. ``visible``
    says whether the caller may see a partner row; one it may not is left
    out.
    """
    threads: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        thread_id = str(row.get("thread_id") or "")
        memory_id = str(row.get("memory_id") or "")
        key = (memory_id, thread_id)
        if not thread_id or _role(row) not in {"user", "assistant"}:
            out.append(row)
            continue
        if key not in threads:
            if sum(found is not None for found in threads.values()) >= max_threads:
                out.append(row)
                continue
            try:
                threads[key] = [
                    item
                    for item in read_thread(memory_id, thread_id) or []
                    if isinstance(item, dict)
                ]
            except Exception:
                threads[key] = None
        thread = threads[key] or []
        index = next(
            (i for i, item in enumerate(thread) if _id(item) == _id(row)), None
        )
        request, reply = _partner(thread, index) if index is not None else (None, None)
        if request is None or reply is None:
            out.append(row)
            continue
        turn_key = (memory_id, thread_id, _id(reply))
        if turn_key in seen:
            continue  # its other half is already in the results
        seen.add(turn_key)
        partner = reply if _role(row) == "user" else request
        if visible is not None and not visible(partner):
            out.append(row)
            continue
        shown = {"id": _id(partner), "content": _text(partner)}
        if partner is reply and reply.get("agent_id"):
            shown["agent_id"] = str(reply["agent_id"])
        out.append({**row, "turn": {"reply" if partner is reply else "request": shown}})
    return out


def _others_first(rows: List[Dict[str, Any]], own_turns: int) -> List[Dict[str, Any]]:
    """Where other agents' turns were recalled, keep only the reader's best
    ``own_turns`` of its own.

    What the reader said before is what it already knew; a colleague's turn
    is what it could not know. And its own earlier "I couldn't find it" is
    the closest match there is to the same question asked again: three of
    those took every place, and beside them a model repeats itself even with
    the thing in front of it.
    """

    def by_reader(row: Dict[str, Any]) -> Optional[bool]:
        turn = row.get("turn")
        return turn.get("by_reader") if isinstance(turn, dict) else None

    if own_turns < 0 or not any(by_reader(row) is False for row in rows):
        return rows
    kept: List[Dict[str, Any]] = []
    own = 0
    for row in rows:  # best match first
        if by_reader(row):
            own += 1
            if own > own_turns:
                continue
        kept.append(row)
    return kept


__all__ = ["whole_turns", "with_turn_partners"]
