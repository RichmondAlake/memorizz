"""Official-format dataset adapters for the local memory evaluation suite.

The loaders intentionally stop at a small normalized contract.  They do not
vendor benchmark data, call Hugging Face, or import an official repository's
runtime and its heavyweight dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

from ..longmemeval_v2 import trajectory_chunks
from .catalog import get_benchmark_spec
from .models import MemoryBenchmarkCase, MemoryDocument


class DatasetLoadError(ValueError):
    """Raised when a dataset path does not match the selected benchmark."""


_LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "common-sense",
    4: "single-hop",
    5: "adversarial",
}

_WEEKDAYS = {
    name.lower(): index
    for index, name in enumerate(
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    )
}


def _parse_locomo_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for format_string in ("%I:%M %p on %d %B, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(text, format_string)
        except ValueError:
            continue
    return None


def _relative_time_annotations(text: str, event_time: datetime | None) -> List[str]:
    """Resolve common relative-date phrases against a turn timestamp."""

    if event_time is None:
        return []
    lowered = str(text or "").casefold()
    resolved: List[tuple[str, datetime]] = []
    fixed = {
        "today": 0,
        "yesterday": -1,
        "tomorrow": 1,
        "last week": -7,
        "next week": 7,
    }
    for phrase, days in fixed.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            resolved.append((phrase, event_time + timedelta(days=days)))
    for direction, weekday_name in re.findall(
        r"\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    ):
        target = _WEEKDAYS[weekday_name]
        if direction == "last":
            days = (event_time.weekday() - target) % 7 or 7
            value = event_time - timedelta(days=days)
        else:
            days = (target - event_time.weekday()) % 7 or 7
            value = event_time + timedelta(days=days)
        resolved.append((f"{direction} {weekday_name}", value))
    output: List[str] = []
    seen = set()
    for phrase, value in resolved:
        key = (phrase, value.date().isoformat())
        if key in seen:
            continue
        seen.add(key)
        output.append(f"{phrase} = {value.date().isoformat()}")
    return output


_MEMORY_AGENT_COMPETENCIES = {
    "accurate-retrieval": {
        "event_qa",
        "eventqa",
        "ruler_qa1",
        "ruler_qa2",
        "longmemeval",
    },
    "test-time-learning": {
        "icl_banking",
        "icl_banking77",
        "icl_clinic",
        "icl_clinic150",
        "icl_nlu",
        "icl_trec_coarse",
        "icl_trec_fine",
        "recsys",
    },
    "long-range-understanding": {"detectiveqa", "detective_qa", "infbench"},
    "conflict-resolution": {
        "fact_mh",
        "fact_sh",
        "factconsolidation_mh",
        "factconsolidation_sh",
    },
}


def _stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLoadError(f"Unable to read JSON dataset {path}: {exc}") from exc


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DatasetLoadError(
                        f"Expected an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLoadError(f"Unable to read JSONL dataset {path}: {exc}") from exc
    return rows


def _records(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path) if path.suffix.lower() == ".json" else _read_jsonl(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("data", "rows", "samples", "records", "train", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(payload)]
    raise DatasetLoadError(f"Expected an object or list in {path}")


def _as_strings(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    output: List[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, (dict, list)):
            output.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        else:
            text = str(item).strip()
            if text:
                output.append(text)
    return tuple(output)


def _chunks(
    text: Any,
    *,
    source_id: str,
    max_chars: int,
    metadata: Mapping[str, Any] | None = None,
) -> Tuple[MemoryDocument, ...]:
    content = str(text or "").strip()
    if not content:
        return ()
    if len(content) <= max_chars:
        return (
            MemoryDocument(
                source_id=source_id,
                content=content,
                metadata=dict(metadata or {}),
            ),
        )

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()
    ]
    pieces: List[str] = []
    current = ""
    for paragraph in paragraphs or [content]:
        slices = [
            paragraph[offset : offset + max_chars]
            for offset in range(0, len(paragraph), max_chars)
        ]
        for piece in slices:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = piece
            else:
                current = candidate
    if current:
        pieces.append(current)
    return tuple(
        MemoryDocument(
            source_id=f"{source_id}#part-{index}",
            parent_source_id=source_id,
            content=piece,
            metadata=dict(metadata or {}),
        )
        for index, piece in enumerate(pieces, start=1)
    )


def _resolve_named_file(root: Path, names: Sequence[str]) -> Path:
    if root.is_file():
        return root
    for name in names:
        direct = root / name
        nested = root / "data" / name
        if direct.exists():
            return direct
        if nested.exists():
            return nested
    raise DatasetLoadError(
        f"Could not find any of {', '.join(names)} under dataset path {root}"
    )


def _balanced_limit(
    cases: Sequence[MemoryBenchmarkCase], limit: int
) -> List[MemoryBenchmarkCase]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    groups: "OrderedDict[str, deque[MemoryBenchmarkCase]]" = OrderedDict()
    for case in cases:
        groups.setdefault(case.category, deque()).append(case)
    selected: List[MemoryBenchmarkCase] = []
    while groups and len(selected) < limit:
        for category in list(groups):
            queue = groups[category]
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if not queue:
                groups.pop(category, None)
    return selected


def _locomo_documents(
    conversation: Mapping[str, Any], *, max_chars: int
) -> Tuple[MemoryDocument, ...]:
    documents: List[MemoryDocument] = []
    session_keys = sorted(
        (
            key
            for key in conversation
            if re.fullmatch(r"session_\d+", str(key))
            and isinstance(conversation.get(key), list)
        ),
        key=lambda value: int(str(value).split("_")[-1]),
    )
    for session_key in session_keys:
        session_index = int(session_key.split("_")[-1])
        raw_date = conversation.get(f"{session_key}_date_time") or "unknown date"
        event_time = _parse_locomo_datetime(raw_date)
        normalized_date = event_time.isoformat() if event_time else str(raw_date)
        for turn_index, turn in enumerate(conversation.get(session_key) or [], start=1):
            if not isinstance(turn, Mapping):
                continue
            speaker = turn.get("speaker") or "Unknown"
            text = turn.get("text") or ""
            caption = turn.get("blip_caption") or turn.get("caption")
            content = f"Date: {normalized_date}\n{speaker}: {text}"
            temporal_annotations = _relative_time_annotations(str(text), event_time)
            if temporal_annotations:
                content += "\nResolved dates: " + "; ".join(temporal_annotations)
            if caption:
                content += f"\nShared media: {caption}"
            source_id = f"D{session_index}:{turn_index}"
            documents.extend(
                _chunks(
                    content,
                    source_id=source_id,
                    max_chars=max_chars,
                    metadata={
                        "session_index": session_index,
                        "event_time": normalized_date,
                        "source_event_time": str(raw_date),
                        "resolved_dates": temporal_annotations,
                    },
                )
            )
    return tuple(documents)


def _evidence_ids(value: Any) -> Tuple[str, ...]:
    output: List[str] = []
    for item in value if isinstance(value, list) else [value]:
        for part in str(item or "").split(";"):
            match = re.search(r"D\d+:\d+", part.strip(), flags=re.IGNORECASE)
            if match:
                output.append(match.group(0).upper())
    return tuple(output)


def _evidence_text(
    conversation: Mapping[str, Any], evidence: Sequence[str]
) -> Tuple[str, ...]:
    output: List[str] = []
    for source_id in evidence:
        match = re.fullmatch(r"D(\d+):(\d+)", source_id, flags=re.IGNORECASE)
        if not match:
            continue
        turns = conversation.get(f"session_{int(match.group(1))}") or []
        turn_index = int(match.group(2)) - 1
        if 0 <= turn_index < len(turns) and isinstance(turns[turn_index], Mapping):
            turn = turns[turn_index]
            output.append(f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}")
    return tuple(output)


def _load_locomo_original(
    path: Path,
    *,
    benchmark_id: str,
    max_chars: int,
    scorer: str,
) -> List[MemoryBenchmarkCase]:
    file_path = _resolve_named_file(path, ("locomo10.json", "locomo.json"))
    rows = _records(file_path)
    cases: List[MemoryBenchmarkCase] = []
    for conversation_index, item in enumerate(rows):
        conversation = item.get("conversation")
        if not isinstance(conversation, Mapping):
            continue
        documents = _locomo_documents(conversation, max_chars=max_chars)
        corpus_id = f"locomo:{conversation_index}"
        for question_index, qa in enumerate(item.get("qa") or []):
            if not isinstance(qa, Mapping) or not qa.get("question"):
                continue
            category_value = qa.get("category")
            try:
                category_key = int(category_value)
            except (TypeError, ValueError):
                category_key = -1
            category = _LOCOMO_CATEGORIES.get(
                category_key, str(category_value or "uncategorized")
            )
            evidence = _evidence_ids(qa.get("evidence") or ())
            answers = _as_strings(qa.get("answer"))
            rubric = _evidence_text(conversation, evidence)
            if category == "adversarial" and not answers:
                rubric = (
                    "The correct behavior is to abstain because the memory does not contain the answer.",
                )
            cases.append(
                MemoryBenchmarkCase.create(
                    case_id=qa.get("id")
                    or f"{benchmark_id}:locomo:{conversation_index}:{question_index}",
                    benchmark_id=benchmark_id,
                    corpus_id=corpus_id,
                    category=category,
                    documents=documents,
                    question=qa.get("question"),
                    answers=answers,
                    relevant_source_ids=evidence,
                    rubric=rubric,
                    scorer=scorer,
                    metadata={
                        "source_dataset": "LoCoMo",
                        "upstream_category": category_value,
                    },
                )
            )
    if not cases:
        raise DatasetLoadError(f"No LoCoMo question records found in {file_path}")
    return cases


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _gap_days(value: Any) -> int:
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\s*"
        r"(week|weeks|month|months|year|years)\b",
        str(value or "").lower(),
    )
    if not match:
        return 0
    number, unit = match.groups()
    count = (
        int(number)
        if number.isdigit()
        else (1 if number in {"a", "an"} else _NUMBER_WORDS[number])
    )
    return count * (
        7 if unit.startswith("week") else 30 if unit.startswith("month") else 365
    )


def _parse_ab_dialogue(
    value: Any, *, speaker_a: str, speaker_b: str
) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for line in str(value or "").splitlines():
        match = re.match(r"^\s*([AB]):\s*(.+?)\s*$", line)
        if match:
            turns.append(
                {
                    "speaker": speaker_a if match.group(1) == "A" else speaker_b,
                    "text": match.group(2),
                }
            )
    return turns


def _locomo_session_times(conversation: Mapping[str, Any]) -> List[datetime]:
    values: List[datetime] = []
    index = 1
    while f"session_{index}" in conversation:
        raw = conversation.get(f"session_{index}_date_time")
        values.append(
            _parse_locomo_datetime(raw)
            or (datetime(1970, 1, 1) + timedelta(days=index))
        )
        index += 1
    return values


def _load_locomo_cognitive(path: Path, *, max_chars: int) -> List[MemoryBenchmarkCase]:
    locomo_path = _resolve_named_file(path, ("locomo10.json", "locomo.json"))
    plus_path = _resolve_named_file(path, ("locomo_plus.json",))
    locomo_rows = _records(locomo_path)
    plus_rows = _records(plus_path)
    if not locomo_rows:
        raise DatasetLoadError(f"No base LoCoMo conversations found in {locomo_path}")
    cases: List[MemoryBenchmarkCase] = []
    for index, plus in enumerate(plus_rows):
        base = locomo_rows[index % len(locomo_rows)]
        conversation = base.get("conversation") or {}
        if not isinstance(conversation, Mapping):
            continue
        speaker_a = str(conversation.get("speaker_a") or "A")
        speaker_b = str(conversation.get("speaker_b") or "B")
        cue_turns = _parse_ab_dialogue(
            plus.get("cue_dialogue"), speaker_a=speaker_a, speaker_b=speaker_b
        )
        documents = list(_locomo_documents(conversation, max_chars=max_chars))
        session_times = _locomo_session_times(conversation)
        query_time = (
            session_times[-1] if session_times else datetime(1970, 1, 1)
        ) + timedelta(days=7)
        cue_time = query_time - timedelta(days=_gap_days(plus.get("time_gap")))
        cue_index = None
        for session_index, session_time in enumerate(session_times):
            if session_time <= cue_time:
                cue_index = session_index
            else:
                break
        relevant: List[str] = []
        rubric: List[str] = []
        cue_documents: List[MemoryDocument] = []
        for turn_index, turn in enumerate(cue_turns, start=1):
            source_id = f"cue:{index}:{turn_index}"
            relevant.append(source_id)
            rubric.append(f"{turn['speaker']}: {turn['text']}")
            cue_documents.extend(
                _chunks(
                    f"Date: {cue_time.isoformat()}\n{turn['speaker']}: {turn['text']}",
                    source_id=source_id,
                    max_chars=max_chars,
                    metadata={
                        "event_time": cue_time.isoformat(),
                        "event_type": "cognitive_cue",
                        "semantic_group_id": f"cue:{index}",
                    },
                )
            )
        insertion = 0
        if cue_index is not None:
            insertion = next(
                (
                    document_index
                    for document_index, document in enumerate(documents)
                    if int(document.metadata.get("session_index") or 0) > cue_index + 1
                ),
                len(documents),
            )
        documents[insertion:insertion] = cue_documents
        trigger = str(plus.get("trigger_query") or "").strip()
        if not trigger:
            continue
        cases.append(
            MemoryBenchmarkCase.create(
                case_id=plus.get("id") or f"locomo-plus:cognitive:{index}",
                benchmark_id="locomo-plus",
                corpus_id=f"locomo-plus:cognitive:{index}",
                category="cognitive",
                documents=documents,
                question=trigger,
                relevant_source_ids=relevant,
                rubric=rubric,
                scorer="llm_judge",
                metadata={
                    "source_dataset": "LoCoMo-Plus",
                    "time_gap": plus.get("time_gap"),
                    "relation_type": plus.get("relation_type"),
                    "query_time": query_time.isoformat(),
                },
            )
        )
    if not cases:
        raise DatasetLoadError(f"No cognitive question records found in {plus_path}")
    return cases


def _load_locomo_plus(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    output: List[MemoryBenchmarkCase] = []
    if variant in {"original", "all"}:
        output.extend(
            _load_locomo_original(
                path,
                benchmark_id="locomo-plus",
                max_chars=max_chars,
                scorer="llm_judge",
            )
        )
    if variant in {"cognitive", "all"}:
        output.extend(_load_locomo_cognitive(path, max_chars=max_chars))
    return output


def _generic_documents(
    value: Any, *, corpus_id: str, max_chars: int
) -> Tuple[MemoryDocument, ...]:
    if isinstance(value, str):
        return _chunks(value, source_id=f"{corpus_id}:context", max_chars=max_chars)
    documents: List[MemoryDocument] = []
    if isinstance(value, Mapping):
        value = value.get("messages") or value.get("turns") or list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                role = (
                    item.get("speaker")
                    or item.get("role")
                    or item.get("author")
                    or "memory"
                )
                content = item.get("content") or item.get("text") or item.get("message")
                source_id = str(
                    item.get("id") or item.get("source_id") or f"{corpus_id}:{index}"
                )
                text = (
                    f"{role}: {content}"
                    if content
                    else json.dumps(item, ensure_ascii=False, sort_keys=True)
                )
            else:
                source_id = f"{corpus_id}:{index}"
                text = str(item)
            documents.extend(_chunks(text, source_id=source_id, max_chars=max_chars))
    return tuple(documents)


def _generic_agentmem_cases(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    file_path = path
    if path.is_dir():
        candidates = [
            path / f"{variant}.jsonl",
            path / f"{variant}.json",
            path / "data" / f"{variant}.jsonl",
            path / "data" / f"{variant}.json",
        ]
        selected = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
        if selected is None:
            raise DatasetLoadError(f"No normalized {variant} export found under {path}")
        file_path = selected
    cases: List[MemoryBenchmarkCase] = []
    corpus_cache: Dict[str, Tuple[MemoryDocument, ...]] = {}
    for index, row in enumerate(_records(file_path)):
        corpus_value = (
            row.get("corpus_id")
            or row.get("conversation_id")
            or row.get("dialogue_id")
            or index
        )
        corpus_id = f"agentmembench:{variant}:{corpus_value}"
        documents = corpus_cache.get(corpus_id)
        if documents is None:
            context = (
                row.get("context")
                or row.get("history")
                or row.get("conversation")
                or row.get("documents")
            )
            documents = _generic_documents(
                context, corpus_id=corpus_id, max_chars=max_chars
            )
            corpus_cache[corpus_id] = documents
        question = row.get("question") or row.get("query")
        if not question or not documents:
            continue
        cases.append(
            MemoryBenchmarkCase.create(
                case_id=row.get("id") or f"{corpus_id}:{index}",
                benchmark_id="agentmembench",
                corpus_id=corpus_id,
                category=row.get("category") or variant,
                documents=documents,
                question=question,
                answers=_as_strings(row.get("answers") or row.get("answer")),
                relevant_source_ids=_as_strings(
                    row.get("evidence") or row.get("relevant_source_ids")
                ),
                scorer="answer_f1",
                metadata={"source_dataset": variant},
            )
        )
    if not cases:
        raise DatasetLoadError(
            f"No normalized AgentMemBench cases found in {file_path}"
        )
    return cases


def _load_agentmembench(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    if variant == "locomo":
        return _load_locomo_original(
            path,
            benchmark_id="agentmembench",
            max_chars=max_chars,
            scorer="answer_f1",
        )
    return _generic_agentmem_cases(path, variant=variant, max_chars=max_chars)


def _longmemeval_category(value: Any) -> str:
    text = str(value or "uncategorized").strip().lower().replace("_", "-")
    text = re.sub(r"-abs$", "", text)
    aliases = {
        "static": "static-state-recall",
        "dynamic": "dynamic-state-tracking",
        "workflow": "workflow-knowledge",
        "gotcha": "environment-gotchas",
        "premise": "premise-awareness",
    }
    for prefix, category in aliases.items():
        if text.startswith(prefix):
            return category
    return text


def _load_longmemeval_v2(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    root = path.parent if path.is_file() else path
    questions_path = root / "questions.jsonl"
    trajectories_path = root / "trajectories.jsonl"
    tier, domain = variant.split("-", 1)
    haystack_path = root / "haystacks" / f"lme_v2_{tier}.json"
    for required in (questions_path, trajectories_path, haystack_path):
        if not required.exists():
            raise DatasetLoadError(f"Missing LongMemEval-V2 artifact: {required}")
    questions = [
        row
        for row in _read_jsonl(questions_path)
        if str(row.get("domain") or "").lower() == domain and not row.get("image")
    ]
    haystacks = _read_json(haystack_path)
    if not isinstance(haystacks, Mapping):
        raise DatasetLoadError(
            f"Expected question-to-trajectory mapping in {haystack_path}"
        )
    needed_ids = {
        str(trajectory_id)
        for question in questions
        for trajectory_id in haystacks.get(str(question.get("id")), [])
    }
    trajectories = {
        str(row.get("id")): row
        for row in _read_jsonl(trajectories_path)
        if str(row.get("id")) in needed_ids
    }
    corpus_cache: Dict[Tuple[str, ...], Tuple[MemoryDocument, ...]] = {}
    cases: List[MemoryBenchmarkCase] = []
    for question in questions:
        question_id = str(question.get("id"))
        trajectory_ids = tuple(str(item) for item in haystacks.get(question_id, []))
        if not trajectory_ids:
            continue
        documents = corpus_cache.get(trajectory_ids)
        if documents is None:
            rows: List[MemoryDocument] = []
            for trajectory_id in trajectory_ids:
                trajectory = trajectories.get(trajectory_id)
                if trajectory is None:
                    raise DatasetLoadError(
                        f"Trajectory {trajectory_id} referenced by {question_id} is missing"
                    )
                for index, content in enumerate(
                    trajectory_chunks(trajectory, max_chars=max_chars), start=1
                ):
                    rows.append(
                        MemoryDocument(
                            source_id=f"{trajectory_id}#part-{index}",
                            parent_source_id=trajectory_id,
                            content=content,
                            metadata={"trajectory_id": trajectory_id},
                        )
                    )
            documents = tuple(rows)
            corpus_cache[trajectory_ids] = documents
        cases.append(
            MemoryBenchmarkCase.create(
                case_id=question_id,
                benchmark_id="longmemeval-v2",
                corpus_id=f"longmemeval-v2:{tier}:{_stable_id(*trajectory_ids)}",
                category=_longmemeval_category(question.get("question_type")),
                documents=documents,
                question=question.get("question"),
                answers=_as_strings(question.get("answer")),
                relevant_source_ids=trajectory_ids,
                rubric=_as_strings(question.get("answer")),
                scorer="llm_judge",
                metadata={
                    "domain": domain,
                    "tier": tier,
                    "eval_function": question.get("eval_function"),
                    "question_type": question.get("question_type"),
                },
            )
        )
    if not cases:
        raise DatasetLoadError(
            f"No text-only LongMemEval-V2 questions found for domain {domain!r}"
        )
    return cases


def _walk_chat_messages(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if ("content" in value or "text" in value) and (
            "role" in value or "speaker" in value
        ):
            yield value
            return
        for nested in value.values():
            yield from _walk_chat_messages(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_chat_messages(nested)


def _walk_probe_records(
    value: Any, category: str = "uncategorized"
) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if value.get("question") or value.get("prompt"):
            yield category, value
            return
        for key, nested in value.items():
            yield from _walk_probe_records(nested, str(key))
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_probe_records(nested, category)


def _flatten_source_ids(value: Any) -> Tuple[str, ...]:
    output: List[str] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            output.extend(_flatten_source_ids(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            output.extend(_flatten_source_ids(nested))
    elif value is not None and str(value).strip():
        output.append(str(value).strip())
    return tuple(dict.fromkeys(output))


def _beam_scale_root(path: Path, variant: str) -> Path:
    aliases = {
        "128k": ("128K", "100K", "128k", "100k"),
        "500k": ("500K", "500k"),
        "1m": ("1M", "1m"),
        "10m": ("10M", "10m"),
    }[variant]
    roots = (path, path / "chats", path / "data", path / "data" / "chats")
    for root in roots:
        for name in aliases:
            candidate = root / name
            if candidate.is_dir():
                return candidate
    if path.is_dir() and (path / "probing_questions").exists():
        return path
    raise DatasetLoadError(f"Could not find BEAM {variant} chats under {path}")


def _load_beam(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    scale_root = _beam_scale_root(path, variant)
    chat_dirs = (
        [scale_root]
        if (scale_root / "probing_questions").exists()
        else sorted(child for child in scale_root.iterdir() if child.is_dir())
    )
    cases: List[MemoryBenchmarkCase] = []
    for chat_index, chat_dir in enumerate(chat_dirs):
        chat_path = next(
            (
                candidate
                for candidate in (
                    chat_dir / "chat_trunecated.json",
                    chat_dir / "chat.json",
                )
                if candidate.exists()
            ),
            None,
        )
        probe_path = chat_dir / "probing_questions" / "probing_questions.json"
        if chat_path is None or not probe_path.exists():
            continue
        documents: List[MemoryDocument] = []
        seen_ids: Dict[str, int] = defaultdict(int)
        for message_index, message in enumerate(
            _walk_chat_messages(_read_json(chat_path))
        ):
            parent_id = str(
                message.get("id") or message.get("chat_id") or message_index
            )
            seen_ids[parent_id] += 1
            source_id = (
                parent_id
                if seen_ids[parent_id] == 1
                else f"{parent_id}:{seen_ids[parent_id]}"
            )
            role = message.get("role") or message.get("speaker") or "memory"
            content = message.get("content") or message.get("text")
            documents.extend(
                _chunks(f"{role}: {content}", source_id=source_id, max_chars=max_chars)
            )
        if not documents:
            continue
        corpus_id = f"beam:{variant}:{chat_dir.name or chat_index}"
        for question_index, (category, probe) in enumerate(
            _walk_probe_records(_read_json(probe_path))
        ):
            question = probe.get("question") or probe.get("prompt")
            if not question:
                continue
            rubric = _as_strings(
                probe.get("rubric")
                or probe.get("grading_rubric")
                or probe.get("criteria")
            )
            answers = _as_strings(
                probe.get("ideal_answer")
                or probe.get("ideal_response")
                or probe.get("answer")
                or probe.get("reference_answer")
            )
            cases.append(
                MemoryBenchmarkCase.create(
                    case_id=probe.get("id")
                    or f"{corpus_id}:{category}:{question_index}",
                    benchmark_id="beam",
                    corpus_id=corpus_id,
                    category=str(category)
                    .strip()
                    .lower()
                    .replace("_", " ")
                    .replace(" ", "-"),
                    documents=documents,
                    question=question,
                    answers=answers,
                    relevant_source_ids=_flatten_source_ids(
                        probe.get("source_chat_ids")
                    ),
                    rubric=rubric or answers,
                    scorer="llm_judge",
                    metadata={"scale": variant, "chat": chat_dir.name},
                )
            )
    if not cases:
        raise DatasetLoadError(f"No BEAM chat/question pairs found under {scale_root}")
    return cases


def _memoryagent_source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    return (
        str(source or row.get("source") or row.get("dataset") or "unknown")
        .strip()
        .lower()
    )


def _memoryagent_competency(source: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    for competency, names in _MEMORY_AGENT_COMPETENCIES.items():
        if normalized in names or any(name in normalized for name in names):
            return competency
    return "unknown"


def _memoryagent_scorer(source: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    if "recsys" in normalized:
        return "recall_at_5"
    if normalized.startswith("icl_") or "detective" in normalized:
        return "exact_match"
    if any(token in normalized for token in ("event", "ruler", "fact")):
        return "substring_exact_match"
    if any(token in normalized for token in ("longmem", "infbench")):
        return "llm_judge"
    return "exact_match"


def _memoryagent_files(path: Path, variant: str) -> List[Path]:
    if path.is_file():
        return [path]
    extensions = {".json", ".jsonl"}
    all_files = sorted(
        file for file in path.rglob("*") if file.suffix.lower() in extensions
    )
    named = [
        file
        for file in all_files
        if variant.replace("-", "") in re.sub(r"[^a-z0-9]", "", file.stem.lower())
    ]
    return named or all_files


def _load_memoryagentbench(
    path: Path, *, variant: str, max_chars: int
) -> List[MemoryBenchmarkCase]:
    cases: List[MemoryBenchmarkCase] = []
    row_index = 0
    for file_path in _memoryagent_files(path, variant):
        for row in _records(file_path):
            source = _memoryagent_source(row)
            competency = _memoryagent_competency(source)
            if competency not in {variant, "unknown"}:
                continue
            context = row.get("context") or row.get("history") or row.get("documents")
            corpus_id = (
                f"memoryagentbench:{file_path.stem}:{row.get('id') or row_index}"
            )
            documents = _generic_documents(
                context, corpus_id=corpus_id, max_chars=max_chars
            )
            if not documents:
                row_index += 1
                continue
            questions_value = row.get("questions") or row.get("question") or []
            questions = (
                list(questions_value)
                if isinstance(questions_value, list)
                else [questions_value]
            )
            answers_value = (
                row.get("answers") if "answers" in row else row.get("answer")
            )
            for question_index, question in enumerate(questions):
                if not str(question or "").strip():
                    continue
                if len(questions) == 1:
                    answers = _as_strings(answers_value)
                elif isinstance(answers_value, list) and question_index < len(
                    answers_value
                ):
                    answers = _as_strings(answers_value[question_index])
                else:
                    answers = _as_strings(answers_value)
                cases.append(
                    MemoryBenchmarkCase.create(
                        case_id=f"{corpus_id}:{question_index}",
                        benchmark_id="memoryagentbench",
                        corpus_id=corpus_id,
                        category=source or variant,
                        documents=documents,
                        question=question,
                        answers=answers,
                        relevant_source_ids=_as_strings(
                            row.get("evidence") or row.get("relevant_source_ids")
                        ),
                        rubric=answers,
                        scorer=_memoryagent_scorer(source),
                        metadata={"competency": variant, "source_dataset": source},
                    )
                )
            row_index += 1
    if not cases:
        raise DatasetLoadError(
            f"No MemoryAgentBench records for competency {variant!r} found under {path}"
        )
    return cases


def load_benchmark_cases(
    benchmark_id: str,
    data_path: str | Path,
    *,
    variant: str | None = None,
    limit: int = 10,
    max_document_chars: int = 3_500,
    one_per_category: bool = False,
) -> List[MemoryBenchmarkCase]:
    """Load a balanced, deterministic subset from an official-format dataset."""
    spec = get_benchmark_spec(benchmark_id)
    selected_variant = str(variant or spec.default_variant).strip().lower()
    if selected_variant not in spec.variants:
        raise DatasetLoadError(
            f"Unknown {spec.name} variant {selected_variant!r}; choose one of {spec.variants}"
        )
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise DatasetLoadError(f"Dataset path does not exist: {path}")
    max_chars = max(256, int(max_document_chars))
    loaders = {
        "agentmembench": _load_agentmembench,
        "longmemeval-v2": _load_longmemeval_v2,
        "locomo-plus": _load_locomo_plus,
        "beam": _load_beam,
        "memoryagentbench": _load_memoryagentbench,
    }
    cases = loaders[spec.benchmark_id](
        path, variant=selected_variant, max_chars=max_chars
    )
    if one_per_category:
        selected: List[MemoryBenchmarkCase] = []
        seen = set()
        for case in cases:
            if case.category in seen:
                continue
            seen.add(case.category)
            selected.append(case)
        return selected
    return _balanced_limit(cases, int(limit))


__all__ = ["DatasetLoadError", "load_benchmark_cases"]
