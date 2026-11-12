# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tool learning: a tool's catalogue entry revised from how it was used.

A tool's entry in toolbox memory is written once, when it is registered.
Use then shows two kinds of gap:

- **calling**: the model reaches the tool and passes arguments it refuses
  (an id without its prefix, a date in the wrong shape);
- **selection**: a request the tool should have been reached by and was
  not. Either the tool served it although routing would not have offered it
  (the agent had to go looking), or the user had to ask again in the same
  conversation, in other words, before the tool was used. A request counts
  for the last tool its run called successfully (the action it was after),
  not for the lookups that came before.

A third gap is **confusion**: a request went to one tool (the run used it,
or routing ranked it first) and another tool turned out to serve it.

A review reads the agent's tool calls (the tool log, joined to the request
that led to each call in workflow memory) and asks the background model for
three small things per tool: a usage note for a repeated calling error,
example requests for the wordings routing missed, and a "Not for ...; use
<tool>" line for requests that keep going to it and belong to a neighbour.
The model only proposes. The code decides what is written:

- nothing the author wrote is rewritten: notes go under a "Learned from use"
  heading after the docstring, and the argument schema is never touched;
- a note must rest on ``min_failures`` failed calls the code can count;
- an example request must rest on ``min_misses`` requests routing really
  missed, and share a word with each;
- an example that pushes another tool out of the tools offered for a request
  that tool served is withdrawn when the revision is applied;
- neither may carry a literal argument value from the calls it rests on, or
  text aimed at the model;
- every revision is stored with its evidence and the ranks it changed, waits
  for a person unless ``auto_apply`` is given, and can be undone.

The search step reads the description and the example requests. The model is
shown the author's description, the usage notes, the "Not for" lines and an
example call shape. Example requests are retrieval keys, never instructions:
a user's wording is kept out of what the model reads unless
``show_learned_examples`` is switched on.

Example requests are marked by where they came from. One learned from real
requests replaces a synthetic one (written at registration) that says the
same thing, and when a tool is at its cap a learned example displaces the
oldest synthetic one instead of being refused.

Separately, each turn records which tools routing offered, so a review can
flag a tool that is offered and never called, or never offered at all.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .._time import utc_now_iso
from ..enums.memory_type import MemoryType
from ..long_term.procedural.toolbox.indexing import LEARNED_HEADING

logger = logging.getLogger(__name__)

TOOL_REVISION_RECORD = "learning_tool_revision"
TOOL_REVIEW_STATE_RECORD = "learning_tool_review_state"
TOOL_OFFERS_RECORD = "learning_tool_offers"
TOOL_SETTINGS_RECORD = "learning_tool_settings"
#: How a when-not-to-use line starts, and the line that shows a call's shape.
AVOID_PREFIX = "Not for "
CALL_EXAMPLE = "Example call:"
#: Where an example request on a tool row came from.
LEARNED, SYNTHETIC, UNKNOWN = "learned", "synthetic", "unknown"
#: Row field set when registration wrote the example requests.
QUERIES_SOURCE = "queries_source"
RETRIEVED_NEVER_CALLED = "retrieved_never_called"
NEVER_RETRIEVED = "never_retrieved"
#: Start of the line that shows the model the example requests learned from use.
ALSO_ASKED = "Also asked for as:"
AUTOMATIC_APPROVER = "memorizz:tool_learning"

_AIMED_AT_MODEL_RE = re.compile(
    r"\b(ignore|disregard|override|bypass|jailbreak|system prompt|developer message"
    r"|password|passwords|api[ _-]?keys?|credentials?|secrets?|sudo)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_ERROR_PREFIX = "Error executing tool:"


@dataclass
class ToolLearningConfig:
    """Bounds on what one review reads and may write."""

    #: Calls of a tool before it is reviewed at all.
    min_calls: int = 3
    #: Failed calls a usage note must rest on.
    min_failures: int = 2
    #: Missed requests an example request must rest on.
    min_misses: int = 2
    #: Usage notes kept on one tool; the oldest go first.
    max_notes: int = 4
    max_note_chars: int = 220
    #: Example requests kept on one tool, and added by one review.
    max_queries: int = 12
    max_new_queries: int = 4
    max_query_chars: int = 160
    #: Most recent calls read for one tool, and tools reviewed in one pass.
    max_calls_per_tool: int = 24
    max_tools: int = 8
    #: Requests that went to this tool and another served, before a
    #: when-not-to-use line is proposed; and such lines kept on one tool.
    min_confusions: int = 2
    max_avoid: int = 2
    #: Turns routing offered a tool in, with no call, before it is flagged
    #: "retrieved, never called"; and routed turns observed before a tool
    #: never offered is flagged "never retrieved".
    min_offers: int = 5
    min_turns: int = 20
    #: Show the model the example requests learned from use, on one "Also
    #: asked for as" line. Off: an example is a retrieval key, and a user's
    #: wording stays out of the prompt.
    show_learned_examples: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.show_learned_examples, bool):
            raise ValueError("show_learned_examples must be true or false")
        for name in (
            "min_confusions",
            "max_avoid",
            "min_offers",
            "min_turns",
            "min_calls",
            "min_failures",
            "min_misses",
            "max_notes",
            "max_note_chars",
            "max_queries",
            "max_new_queries",
            "max_query_chars",
            "max_calls_per_tool",
            "max_tools",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number of 1 or more")

    @classmethod
    def from_value(cls, value: Any) -> "ToolLearningConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            known = {
                key: value[key] for key in cls.__dataclass_fields__ if key in value
            }
            return cls(**known)
        return cls()


@dataclass
class ToolCall:
    """One call of a tool, as the review sees it."""

    tool: str
    ok: bool
    arguments: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: The request that led to the call ("" when only the tool log has it).
    query: str = ""
    thread_id: str = ""
    at: str = ""
    #: The tool log row or workflow the call was read from.
    source_id: str = ""
    #: Arguments of the next call of the same tool in that run that worked.
    recovered_with: Optional[Dict[str, Any]] = None
    #: The last call of its run that worked: the action the request was
    #: after. Earlier calls are preparation (a lookup before a change).
    final: bool = False
    #: The request made just before this one in the same conversation, when
    #: that run did not reach this tool: the user asked again, in other
    #: words, and this call is what the second wording reached.
    reask_of: str = ""
    #: The tool the earlier run ended with ("" when it called none that worked).
    reask_used: str = ""
    #: Where routing ranked the tool for ``query`` (None: not offered).
    rank: Optional[int] = None
    missed: bool = False
    #: Other tools the same run called successfully (its preparation).
    with_tools: List[str] = field(default_factory=list)


@dataclass
class ToolRevision:
    """One proposed or applied change to a tool's catalogue entry."""

    revision_id: str
    agent_id: str
    tool: str
    status: str = "proposed"  # proposed, applied, dismissed, undone
    notes_added: List[str] = field(default_factory=list)
    queries_added: List[str] = field(default_factory=list)
    #: "Not for <kind of request>; use <tool>" lines, from confused requests.
    avoid_added: List[str] = field(default_factory=list)
    #: The shape of a call that works, values replaced by their format.
    call_example: str = ""
    #: Synthetic example requests the new ones took the place of.
    queries_displaced: List[str] = field(default_factory=list)
    description_before: str = ""
    description_after: str = ""
    queries_before: List[str] = field(default_factory=list)
    queries_after: List[str] = field(default_factory=list)
    #: The calls the change rests on: ``{kind, query, error, argument_names,
    #: source_id, rank}``. Argument values are not kept.
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    #: What the model proposed and the code refused, each with the reason.
    refused: List[Dict[str, str]] = field(default_factory=list)
    #: Routing rank of the tool for each missed request, before and after.
    ranks: List[Dict[str, Any]] = field(default_factory=list)
    #: Requests another tool served that the new examples pushed that tool
    #: out of the offered tools for: ``{tool, query, before, after}``. When
    #: there is one, the examples are withdrawn and only the notes stay.
    displaced: List[Dict[str, Any]] = field(default_factory=list)
    queries_withdrawn: bool = False
    created_at: str = ""
    applied_at: Optional[str] = None
    approved_by: Optional[str] = None
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["record_type"] = TOOL_REVISION_RECORD
        data["record_id"] = self.revision_id
        data["timestamp"] = self.created_at
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolRevision":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)

    @property
    def empty(self) -> bool:
        return not (
            self.notes_added
            or self.queries_added
            or self.avoid_added
            or self.call_example
        )


@dataclass
class ToolReview:
    """What one review found: revisions, and why each other tool was skipped."""

    revisions: List[ToolRevision] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    #: What the model proposed for a tool and the code refused, with reasons,
    #: where nothing of the proposal was left.
    refused: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    #: Tools routing offers and nothing calls, or never offers: ``{tool,
    #: flag, offered_turns, calls, turns_observed, since}``. For a person
    #: to look at; a flag proposes nothing.
    flags: List[Dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    tokens_estimated: bool = False
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flags": [dict(flag) for flag in self.flags],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "skipped": dict(self.skipped),
            "refused": {tool: list(items) for tool, items in self.refused.items()},
            "tokens": self.tokens,
            "tokens_estimated": self.tokens_estimated,
            "dry_run": self.dry_run,
        }


# ------------------------------------------------------------- description


def split_description(description: Any) -> Tuple[str, List[str]]:
    """A stored description as (what the author or augmentation wrote, the
    usage notes learned since)."""
    text = str(description or "")
    index = text.find(LEARNED_HEADING)
    if index < 0:
        return text.strip(), []
    notes = [
        line.strip()[2:].strip()
        for line in text[index + len(LEARNED_HEADING) :].splitlines()
        if line.strip().startswith("- ")
    ]
    return text[:index].strip(), [note for note in notes if note]


def join_description(base: Any, notes: Iterable[str]) -> str:
    """The description stored and shown: ``base`` then the notes."""
    kept = [str(note).strip() for note in notes if str(note).strip()]
    text = str(base or "").strip()
    if not kept:
        return text
    block = LEARNED_HEADING + "\n" + "\n".join(f"- {note}" for note in kept)
    return f"{text}\n\n{block}" if text else block


def line_kind(line: Any) -> str:
    """What a learned line is: ``avoid`` (a "Not for" line), ``example_call``
    or a usage ``note``."""
    text = str(line or "").strip()
    if text.startswith(AVOID_PREFIX):
        return "avoid"
    if text.startswith(CALL_EXAMPLE):
        return "example_call"
    return "note"


def call_shape(calls: Iterable[Any]) -> str:
    """The shape of a call that works, from the calls that did: each argument
    with its value replaced by its format (``CW-######``, ``<number>``,
    ``<text>``). "" unless some argument has a fixed part worth showing. No
    value from a call is kept: digits are masked, and a value is shown as a
    format only when every call shares it and little of it is fixed."""
    worked = [call.arguments for call in calls if call.ok and call.arguments]
    if len(worked) < 2:
        return ""
    names = sorted(set(worked[0]).intersection(*worked[1:]))
    shape: Dict[str, str] = {}
    fixed = False
    for name in names:
        values = [arguments[name] for arguments in worked]
        if all(isinstance(value, bool) for value in values):
            shape[name] = "<true or false>"
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            shape[name] = "<number>"
        elif all(isinstance(value, str) for value in values):
            masks = {re.sub(r"[0-9]", "#", value.strip()) for value in values}
            mask = next(iter(masks))
            kept = re.sub(r"[#\s]", "", mask)
            if (
                len(masks) == 1
                and mask.count("#") >= 3
                and 1 <= len(kept) <= 6
                and len(mask) <= 24
            ):
                shape[name] = mask
                fixed = True
            else:
                shape[name] = "<text>"
        else:
            shape[name] = "<value>"
    if not fixed:
        return ""
    return CALL_EXAMPLE + " " + json.dumps(shape, sort_keys=True)


def note_tools_offered(provider: Any, agent_id: str, names: Iterable[str]) -> None:
    """Count one routed turn, and each tool routing offered in it. One small
    record per agent, read by :meth:`ToolLearner.usage_flags`."""
    from .store import LearningControlPlaneStore

    offered = sorted({str(name) for name in names if str(name).strip()})
    if provider is None or not agent_id or not offered:
        return
    store = LearningControlPlaneStore(provider, agent_id=str(agent_id))
    record_id = f"tool-offers-{agent_id}"
    record = store.lookup(record_id) or {}
    tools = record.get("tools")
    tools = dict(tools) if isinstance(tools, Mapping) else {}
    now = utc_now_iso()
    for name in offered:
        entry = tools.get(name)
        entry = dict(entry) if isinstance(entry, Mapping) else {}
        entry["turns"] = int(entry.get("turns") or 0) + 1
        entry["last"] = now
        tools[name] = entry
    store._put(
        {
            "record_type": TOOL_OFFERS_RECORD,
            "record_id": record_id,
            "turns": int(record.get("turns") or 0) + 1,
            "since": record.get("since") or now,
            "tools": tools,
        },
        immutable=False,
    )


def as_query_list(value: Any) -> List[str]:
    """Stored example requests as a list, whatever shape they were kept in."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = None
        value = parsed if isinstance(parsed, list) else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _words(text: Any) -> set:
    return set(_WORD_RE.findall(str(text or "").lower()))


_COMMON_WORDS = frozenset(
    "about after again against all also and any are been but can claim claims "
    "could does for from get has have her him his how into its just like more "
    "need not one our out please she that the their them then there these "
    "they this those was what when where which who with would you your".split()
)


def _content_words(text: Any) -> set:
    """Words that carry the request's meaning: everything but the commonest.
    Short ones stay, because house jargon is mostly short (SIU, R12)."""
    return {
        word for word in _words(text) if len(word) >= 3 and word not in _COMMON_WORDS
    }


def _same_wording(left: str, right: str) -> bool:
    """The same request apart from the numbers in it (a claim id, an amount)."""
    a = {word for word in _words(left) if not word.isdigit()}
    b = {word for word in _words(right) if not word.isdigit()}
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.8


def _same_text(left: str, right: str) -> bool:
    a, b = _words(left), _words(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.8


def _literal_values(arguments: Any) -> List[str]:
    """String forms of argument values long enough to identify something."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, bool) or value is None:
            return
        else:
            text = str(value).strip()
            if len(text) >= 4:
                found.append(text.casefold())

    walk(arguments)
    return found


def _parse_reply(reply: Any) -> Optional[Dict[str, Any]]:
    text = str(reply or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_REVIEW_PROMPT = """You maintain the catalogue entry of one tool an AI agent uses. \
The entry has two readers: a search step that decides which tools to offer for a \
request, and the model that then calls the tool. Below is the tool, and how its \
recent calls went.

Tool: {name}{signature}
Author's description:
{description}
Arguments: {arguments}
Usage notes already kept: {notes}
Example requests already kept: {queries}

FAILED CALLS (the arguments the tool refused, the error, and the arguments that \
worked next where there were any):
{failures}

MISSED REQUESTS (requests this tool should have been reached by and was not: \
either it served the request although the search step would not have offered \
it, or the user had to ask again in other words before it was used):
{misses}

CONFUSED REQUESTS (requests that went to this tool, because the agent used it \
or the search step ranked it first, and that another tool turned out to serve):
{confusions}

Reply with JSON only:
{{"notes": [{{"text": "...", "calls": [1, 2]}}], "queries": [{{"text": "...", "from": [1]}}], "avoid": [{{"text": "Not for ...; use <tool>.", "tool": "<tool>", "from": [1, 2]}}]}}

Rules:
- A note is one sentence that tells the caller how to pass the arguments so the \
call is not refused. Write one only for an error that appears in at least \
{min_failures} failed calls, and list those calls' numbers in "calls". At most 2 notes.
- Make a note exact enough that the next call works: name the fixed parts of \
the format (a prefix, a date pattern, a unit) as the error or the call that \
worked shows them, and leave out the part that changes from call to call. \
"claim_id is CLM- followed by the digits, never the digits alone" is a good \
note. "Use the correct format" is not.
- A query is a short request, in the user's own kind of wording and jargon, \
that this tool should be offered for. Write one only for a wording that at \
least {min_misses} missed requests share and that asks for what this tool \
does (a request that only passed through this tool on its way to something \
else is not one, and neither is an earlier request about a different task). \
Keep the words that made it hard to find, such as the team's own name for \
the task, and list the numbers it comes from in "from". At most \
{max_new_queries} queries.
- An avoid line says what this tool is NOT for, and which tool to use. Write \
one only when at least {min_confusions} confused requests were served by the \
same other tool: begin with "Not for", name the kind of request in the words \
those requests use, then "; use" and that tool's name. Put that tool's name \
in "tool" and the requests' numbers in "from". At most one line per other tool.
- Never copy a whole id, a person's or company's name, an amount or a date \
from the calls.
- Do not restate what the description, the notes or the example requests \
already say. Do not describe behaviour the calls do not show.
- With nothing to add, reply {{"notes": [], "queries": [], "avoid": []}}.
"""


class ToolLearner:
    """Reviews an agent's tool calls and revises its tools' catalogue entries."""

    def __init__(self, agent: Any, config: Any = None):
        self.agent = agent
        if agent.memory_provider is None:
            raise ValueError("Tool learning requires a memory provider")
        if not agent.agent_id:
            raise ValueError("Tool learning requires an agent_id")
        self.provider = agent.memory_provider
        self.agent_id = str(agent.agent_id)
        if isinstance(config, ToolLearningConfig):
            self.config = config
        else:
            # What configure() kept for this agent, under what the caller gave.
            given = dict(config) if isinstance(config, Mapping) else {}
            self.config = ToolLearningConfig.from_value({**self.settings(), **given})
        #: Routing's ranking of the tools for a request, within one review.
        self._ranked_cache: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------ store

    def _store(self) -> Any:
        from .store import LearningControlPlaneStore

        return LearningControlPlaneStore(self.provider, agent_id=self.agent_id)

    def _save(self, revision: ToolRevision) -> None:
        self._store()._put(revision.to_dict(), immutable=False)

    def list_revisions(
        self, tool: Optional[str] = None, status: Optional[str] = None
    ) -> List[ToolRevision]:
        """Stored revisions of this agent's tools, newest first."""
        rows = self._store().list_records(record_type=TOOL_REVISION_RECORD, limit=5000)
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        revisions = [ToolRevision.from_dict(row) for row in rows]
        if tool is not None:
            revisions = [item for item in revisions if item.tool == tool]
        if status is not None:
            revisions = [item for item in revisions if item.status == status]
        return revisions

    def get_revision(self, revision_id: str) -> Optional[ToolRevision]:
        record = self._store().get(revision_id)
        if record and record.get("record_type") == TOOL_REVISION_RECORD:
            return ToolRevision.from_dict(record)
        return None

    def settings(self) -> Dict[str, Any]:
        """Settings kept for this agent with :meth:`configure`."""
        try:
            record = self._store().lookup(f"tool-learning-settings-{self.agent_id}")
        except Exception as exc:
            logger.debug("Tool learning settings could not be read: %s", exc)
            return {}
        kept = (record or {}).get("settings")
        fields = ToolLearningConfig.__dataclass_fields__
        return (
            {key: value for key, value in kept.items() if key in fields}
            if isinstance(kept, Mapping)
            else {}
        )

    def configure(self, **settings: Any) -> ToolLearningConfig:
        """Keep settings for this agent's tool learning (any field of
        :class:`ToolLearningConfig`), and show the model what they now allow."""
        unknown = sorted(set(settings) - set(ToolLearningConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown tool learning setting: {', '.join(unknown)}")
        kept = {**self.settings(), **settings}
        self.config = ToolLearningConfig.from_value(kept)  # refuses a bad value
        self._store()._put(
            {
                "record_type": TOOL_SETTINGS_RECORD,
                "record_id": f"tool-learning-settings-{self.agent_id}",
                "settings": kept,
            },
            immutable=False,
        )
        shown = self.learned_notes()
        for tool in self.tool_rows():
            self._show_notes(tool, shown.get(tool, []))
        return self.config

    def _state_id(self) -> str:
        return f"tool-review-state-{self.agent_id}"

    def _reviewed_until(self) -> Dict[str, str]:
        record = self._store().lookup(self._state_id()) or {}
        tools = record.get("tools")
        return dict(tools) if isinstance(tools, Mapping) else {}

    def _mark_reviewed(self, marks: Mapping[str, str]) -> None:
        if not marks:
            return
        tools = self._reviewed_until()
        tools.update({name: at for name, at in marks.items() if at})
        self._store()._put(
            {
                "record_type": TOOL_REVIEW_STATE_RECORD,
                "record_id": self._state_id(),
                "tools": tools,
            },
            immutable=False,
        )

    # ------------------------------------------------------------- tool rows

    def _toolbox(self) -> Any:
        toolbox = getattr(self.agent, "toolbox", None)
        if toolbox is not None and hasattr(toolbox, "get_most_similar_tools"):
            return toolbox
        search = getattr(self.agent, "_toolbox_search", None)
        return search() if callable(search) else None

    def tool_rows(self) -> Dict[str, List[Dict[str, Any]]]:
        """This agent's toolbox rows by tool name: the rows a Toolbox
        registered for it. Rows shared by several agents, the copies saving
        an agent writes and MCP tools are left alone."""
        from ..long_term.procedural.toolbox.indexing import (
            NOT_AGENT_TOOLS,
            is_agent_mirror_row,
            row_type,
        )

        rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.provider.list_all(MemoryType.TOOLBOX) or []:
            if not isinstance(row, dict) or row.get("agent_id") != self.agent_id:
                continue
            if row_type(row) in NOT_AGENT_TOOLS:
                continue
            if is_agent_mirror_row(row, self.agent_id):
                continue
            name = str(row.get("name") or "").strip()
            if name:
                rows.setdefault(name, []).append(row)
        return rows

    @staticmethod
    def _row_id(row: Mapping[str, Any]) -> str:
        # The tool's own id first: on Oracle a row also has a row id, and
        # updates are addressed by the tool id.
        return str(row.get("tool_id") or row.get("_id") or row.get("id") or "")

    def _write_rows(
        self, rows: List[Dict[str, Any]], description: str, queries: List[str]
    ) -> None:
        from ..embeddings import get_embedding
        from ..long_term.procedural.toolbox.indexing import search_text
        from ..memory_provider.base import provider_manages_embeddings

        for row in rows:
            update: Dict[str, Any] = {"description": description, "queries": queries}
            if not provider_manages_embeddings(self.provider):
                update["embedding"] = get_embedding(search_text({**row, **update}))
            self.provider.update_by_id(self._row_id(row), update, MemoryType.TOOLBOX)

    def _show_notes(self, tool: str, notes: List[str]) -> None:
        """Put the notes on the description the model is shown for ``tool``."""
        manager = getattr(self.agent, "tool_manager", None)
        setter = getattr(manager, "set_learned_notes", None)
        if callable(setter):
            setter(tool, notes)

    # ------------------------------------------------------------------ calls

    def _rows_of(self, memory_type: Any) -> List[Dict[str, Any]]:
        try:
            rows = self.provider.list_all(memory_type) or []
        except Exception as exc:
            logger.warning("Tool learning could not read %s: %s", memory_type, exc)
            return []
        return [
            row
            for row in rows
            if isinstance(row, dict) and row.get("agent_id") == self.agent_id
        ]

    @staticmethod
    def _arguments(value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return {}
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _error_text(step: Mapping[str, Any]) -> str:
        error = step.get("error")
        text = "" if isinstance(error, bool) or not error else str(error)
        if not text:
            result = str(step.get("result") or "")
            if result.startswith(_ERROR_PREFIX):
                text = result
        if text.startswith(_ERROR_PREFIX):
            text = text[len(_ERROR_PREFIX) :]
        return _clip(text, 300) or ("the call failed" if error else "")

    def calls(self) -> Dict[str, List[ToolCall]]:
        """Recent calls of each tool, oldest first.

        Workflow memory gives each call its request. The tool log adds the
        calls workflow memory does not hold (an agent that keeps no
        workflows), without a request.
        """
        by_tool: Dict[str, List[ToolCall]] = {}
        seen = set()
        threads: Dict[str, List[Dict[str, Any]]] = {}

        def key(call: ToolCall) -> Tuple[str, str, str, bool]:
            return (
                call.thread_id,
                call.tool,
                json.dumps(call.arguments, sort_keys=True, default=str),
                call.ok,
            )

        for workflow in self._rows_of(MemoryType.WORKFLOW_MEMORY):
            steps = workflow.get("steps")
            if isinstance(steps, str):
                try:
                    steps = json.loads(steps)
                except (TypeError, ValueError):
                    steps = None
            if not isinstance(steps, Mapping):
                continue
            run: List[ToolCall] = []
            for label, step in steps.items():
                if not isinstance(step, Mapping):
                    continue
                tool = str(step.get("_id") or step.get("tool") or "").strip()
                if not tool:
                    tool = str(label).split(":", 1)[-1].strip()
                if not tool:
                    continue
                error = self._error_text(step)
                run.append(
                    ToolCall(
                        tool=tool,
                        ok=not error,
                        arguments=self._arguments(step.get("arguments")),
                        error=error,
                        query=str(workflow.get("user_query") or "").strip(),
                        thread_id=str(workflow.get("origin_thread_id") or ""),
                        at=str(
                            step.get("timestamp") or workflow.get("created_at") or ""
                        ),
                        source_id=str(
                            workflow.get("workflow_id") or workflow.get("_id") or ""
                        ),
                    )
                )
            for index, call in enumerate(run):
                if call.ok:
                    continue
                for later in run[index + 1 :]:
                    if later.tool == call.tool and later.ok:
                        call.recovered_with = later.arguments
                        break
            worked = [call for call in run if call.ok]
            if worked:
                worked[-1].final = True
            for call in run:
                call.with_tools = sorted(
                    {other.tool for other in worked if other.tool != call.tool}
                )
            for call in run:
                seen.add(key(call))
                by_tool.setdefault(call.tool, []).append(call)
            if run and run[0].thread_id:
                threads.setdefault(run[0].thread_id, []).append(
                    {
                        "at": str(workflow.get("created_at") or run[0].at),
                        "query": run[0].query,
                        "run": run,
                        "memory_id": workflow.get("memory_id"),
                        "user_id": workflow.get("user_id"),
                    }
                )

        self._mark_reasks(threads)

        for row in self._rows_of(MemoryType.TOOL_LOG):
            tool = str(row.get("tool_name") or "").strip()
            if not tool:
                continue
            ok = bool(row.get("success", True))
            call = ToolCall(
                tool=tool,
                ok=ok,
                arguments=self._arguments(row.get("arguments")),
                error=(
                    "" if ok else self._error_text({"error": row.get("error") or True})
                ),
                thread_id=str(row.get("thread_id") or ""),
                at=str(row.get("timestamp") or ""),
                source_id=str(row.get("tool_log_id") or row.get("_id") or ""),
            )
            if key(call) in seen:
                continue
            by_tool.setdefault(tool, []).append(call)

        for tool, items in by_tool.items():
            items.sort(key=lambda call: call.at)
            by_tool[tool] = items[-self.config.max_calls_per_tool :]
        return by_tool

    def _asked(self, thread_id: str, runs: List[Dict[str, Any]]) -> List[str]:
        """What the user asked in one conversation, in order. Read from the
        conversation, so a request the agent answered without a tool (a
        question back, say) is there too; without a readable conversation,
        the requests of its stored runs."""
        from .consolidation import (
            SEALED_STATE,
            _accepts,
            _row_text,
            _row_thread,
            is_suppressed,
        )

        fallback = [str(item["query"] or "") for item in runs]
        retrieve = getattr(
            self.provider, "retrieve_conversation_history_ordered_by_timestamp", None
        )
        memory_id = runs[0].get("memory_id")
        if not callable(retrieve) or not memory_id:
            return fallback
        kwargs: Dict[str, Any] = {"memory_id": memory_id}
        if _accepts(retrieve, "user_id"):
            kwargs["user_id"] = runs[0].get("user_id")
        if _accepts(retrieve, "thread_id"):
            kwargs["thread_id"] = thread_id
        if _accepts(retrieve, "include_embedding"):
            kwargs["include_embedding"] = False
        try:
            rows = retrieve(**kwargs) or []
        except Exception as exc:
            logger.debug("Tool learning could not read a conversation: %s", exc)
            return fallback
        asked = []
        for row in rows:
            if not isinstance(row, dict) or _row_thread(row) != str(thread_id):
                continue
            if str(row.get("retention_state") or "").lower() == SEALED_STATE:
                # A sealed conversation is not read: no re-asks from it.
                return []
            if is_suppressed(row) or str(row.get("role") or "").lower() != "user":
                continue
            text = " ".join(_row_text(row).split())
            if text:
                asked.append(text)
        return asked or fallback

    def _mark_reasks(self, threads: Dict[str, List[Dict[str, Any]]]) -> None:
        """Mark the calls a second wording reached: the user asked, the run
        did not reach the tool, and the next request in that conversation
        did. Only the most recent conversations are read."""
        recent = sorted(
            threads.items(),
            key=lambda item: max(run["at"] for run in item[1]),
            reverse=True,
        )[:40]
        for thread_id, runs in recent:
            runs.sort(key=lambda item: item["at"])
            if len(runs) < 2 and not runs[0].get("memory_id"):
                continue
            asked = self._asked(thread_id, runs)
            by_query = {" ".join(str(item["query"]).split()): item for item in runs}
            for item in runs:
                again = " ".join(str(item["query"]).split())
                reached = next((call for call in item["run"] if call.final), None)
                if reached is None or again not in asked:
                    continue
                index = len(asked) - 1 - asked[::-1].index(again)
                if index == 0:
                    continue
                first = asked[index - 1]
                if not first or first == again:
                    continue
                earlier = by_query.get(first, {}).get("run") or []
                if any(call.tool == reached.tool and call.ok for call in earlier):
                    continue
                ended = next((call for call in earlier if call.final), None)
                reached.reask_of = first
                reached.reask_used = ended.tool if ended else ""

    # ---------------------------------------------------------------- routing

    def _top_k(self) -> int:
        router = getattr(self.agent, "semantic_tool_router", None)
        try:
            return max(1, int(getattr(router, "top_k", 5)))
        except (TypeError, ValueError):
            return 5

    def rank(self, tool: str, query: str, limit: Optional[int] = None) -> Optional[int]:
        """Where routing ranks ``tool`` for ``query`` (1 is first; None when
        it is not among the ``limit`` closest)."""
        names = self.ranked(query, limit)
        if names is None:
            return None
        return names.index(tool) + 1 if tool in names else None

    def ranked(self, query: str, limit: Optional[int] = None) -> Optional[List[str]]:
        """The tools routing ranks closest to ``query``, best first (None
        when there is no toolbox to ask)."""
        toolbox = self._toolbox()
        if toolbox is None or not query:
            return None
        cache = self._ranked_cache if limit is None else None
        if cache is not None and query in cache:
            return cache[query]
        width = int(limit or max(self._top_k() * 3, 10))
        try:
            rows = toolbox.get_most_similar_tools(
                query, limit=width, agent_id=self.agent_id
            )
        except Exception as exc:
            logger.debug("Tool ranking failed for %r: %s", query, exc)
            return None
        names: List[str] = []
        for row in rows or []:
            name = str((row or {}).get("name") or "")
            if name and name not in names:
                names.append(name)
        if cache is not None:
            cache[query] = names
        return names

    def _confusions(
        self, calls: Mapping[str, List[ToolCall]], registered: Iterable[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Requests that went to one tool and another served, by the tool
        they went to. Two signs, both counted by the code: the run used the
        tool and the user had to ask again; or routing ranked the tool first
        and left the tool that served the request out of those it offered."""
        known = set(registered)
        top_k = self._top_k()
        found: Dict[str, List[Dict[str, Any]]] = {}
        seen = set()
        asked = 0

        def add(tool: str, query: str, served_by: str, how: str, call: ToolCall):
            key = (tool, served_by, query.casefold())
            if tool not in known or tool == served_by or not query or key in seen:
                return
            seen.add(key)
            found.setdefault(tool, []).append(
                {
                    "kind": "confused_request",
                    "query": query,
                    "served_by": served_by,
                    "how": how,
                    "source_id": call.source_id,
                    "at": call.at,
                }
            )

        for served_by in sorted(calls):
            if served_by not in known:
                continue
            for call in calls[served_by]:
                if not call.final:
                    continue
                if call.reask_of and call.reask_used:
                    add(call.reask_used, call.reask_of, served_by, "re_asked", call)
                if not call.query or asked >= 40:
                    continue
                asked += 1
                names = self.ranked(call.query) or []
                if not names or served_by in names[:top_k]:
                    continue
                first = names[0]
                # A tool the run itself called first is preparation (a lookup
                # before a change), not a wrong turn.
                if first not in call.with_tools:
                    add(first, call.query, served_by, "ranked_first", call)
        return found

    def _mark_misses(self, tool: str, calls: List[ToolCall]) -> None:
        top_k = self._top_k()
        ranked: Dict[str, Optional[int]] = {}
        for call in calls:
            if not call.final or not call.query:
                continue
            if call.query not in ranked:
                if len(ranked) >= 8:
                    continue
                ranked[call.query] = self.rank(tool, call.query)
            call.rank = ranked[call.query]
            call.missed = call.rank is None or call.rank > top_k

    # ----------------------------------------------------------------- review

    def review(
        self,
        tools: Optional[Iterable[str]] = None,
        *,
        dry_run: bool = False,
        auto_apply: bool = False,
        approved_by: Optional[str] = None,
    ) -> ToolReview:
        """Read the calls and propose a revision for each tool that needs one.

        A revision waits as ``proposed`` unless ``auto_apply`` is given. A dry
        run stores nothing and does not count as a review.
        """
        from .consolidation import text_generator
        from .refine import _TokenMeter

        review = ToolReview(dry_run=dry_run)
        model = getattr(self.agent, "background_model", None)
        generate = text_generator(model)
        rows = self.tool_rows()
        calls = self.calls()
        review.flags = self.usage_flags(calls=calls, rows=rows)
        wanted = set(tools) if tools is not None else None
        reviewed_until = self._reviewed_until()
        waiting = {item.tool for item in self.list_revisions(status="proposed")}
        marks: Dict[str, str] = {}
        candidates: List[Tuple[str, List[ToolCall]]] = []
        self._ranked_cache = {}
        confusions = self._confusions(calls, rows)
        latest: Dict[str, str] = {}

        for tool in sorted(set(calls) | set(confusions)):
            if wanted is not None and tool not in wanted:
                continue
            items = calls.get(tool, [])
            confused = confusions.get(tool, [])
            latest[tool] = max(
                [items[-1].at if items else "", *(item["at"] for item in confused)]
            )
            if tool not in rows:
                review.skipped[tool] = "not_registered_through_a_toolbox"
            elif tool in waiting:
                review.skipped[tool] = "revision_waiting_for_review"
            elif (
                len(items) < self.config.min_calls
                and len(confused) < self.config.min_confusions
            ):
                review.skipped[tool] = "too_few_calls"
            elif latest[tool] and latest[tool] <= reviewed_until.get(tool, ""):
                review.skipped[tool] = "no_new_calls"
            else:
                candidates.append((tool, items))
        for tool in sorted(wanted or ()):
            if tool not in calls and tool not in confusions:
                review.skipped[tool] = "no_calls"

        # Tools with the most failed calls first: they have the most to gain.
        candidates.sort(key=lambda item: -sum(1 for call in item[1] if not call.ok))
        for tool, _ in candidates[self.config.max_tools :]:
            review.skipped[tool] = "left_for_the_next_review"

        for tool, items in candidates[: self.config.max_tools]:
            self._mark_misses(tool, items)
            failures = [call for call in items if not call.ok]
            misses = self._misses(tool, items)
            confused = confusions.get(tool, [])[:8]
            if latest.get(tool):
                marks[tool] = latest[tool]
            if (
                len(failures) < self.config.min_failures
                and len(misses) < self.config.min_misses
                and len(confused) < self.config.min_confusions
            ):
                review.skipped[tool] = "nothing_to_fix"
                continue
            if generate is None:
                review.skipped[tool] = "no_model"
                continue
            row = rows[tool][0]
            prompt = self._prompt(tool, row, failures, misses, confused)
            with _TokenMeter(model) as meter:
                try:
                    # Built inside the meter, so the call it counts is the
                    # one made here.
                    reply = text_generator(model)(prompt)
                except Exception as exc:
                    logger.warning("Tool review of %s failed: %s", tool, exc)
                    review.skipped[tool] = "model_failed"
                    marks.pop(tool, None)
                    continue
            review.tokens += meter.tokens
            review.tokens_estimated = review.tokens_estimated or meter.estimated
            parsed = _parse_reply(reply)
            if parsed is None:
                review.skipped[tool] = "bad_reply"
                marks.pop(tool, None)
                continue
            revision = self._revision(
                tool, row, failures, misses, parsed, items, confused
            )
            revision.tokens = meter.tokens
            if revision.empty:
                review.skipped[tool] = (
                    "proposal_refused" if revision.refused else "nothing_to_add"
                )
                if revision.refused:
                    review.refused[tool] = list(revision.refused)
                continue
            review.revisions.append(revision)

        self._ranked_cache = None
        if dry_run:
            return review
        self._mark_reviewed(marks)
        for revision in review.revisions:
            self._save(revision)
            if auto_apply:
                applied = self.apply(
                    revision.revision_id, approved_by=approved_by or AUTOMATIC_APPROVER
                )
                revision.__dict__.update(applied.__dict__)
        return review

    def _misses(self, tool: str, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """Requests ``tool`` should have been reached by and was not, each
        once: a request it served that routing would not have offered it
        for, and a request the user had to make again before it was used."""
        seen, misses = set(), []
        for call in calls:
            if call.missed and call.query.casefold() not in seen:
                seen.add(call.query.casefold())
                misses.append(
                    {
                        "kind": "missed_request",
                        "query": call.query,
                        "rank": call.rank,
                        "source_id": call.source_id,
                    }
                )
        asked_again = 0
        for call in calls:
            if not call.reask_of or call.reask_of.casefold() in seen:
                continue
            if asked_again >= 6:
                break
            asked_again += 1
            seen.add(call.reask_of.casefold())
            misses.append(
                {
                    "kind": "re_asked_request",
                    "query": call.reask_of,
                    "then": call.query,
                    "used": call.reask_used,
                    "rank": self.rank(tool, call.reask_of),
                    "source_id": call.source_id,
                }
            )
        return misses

    def _prompt(
        self,
        tool: str,
        row: Mapping[str, Any],
        failures: List[ToolCall],
        misses: List[Dict[str, Any]],
        confused: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        from ..long_term.procedural.toolbox.indexing import argument_names

        base, notes = split_description(row.get("description") or row.get("docstring"))
        confusion_lines = [
            f"{number}. {json.dumps(_clip(item['query'], 240))}: "
            + (
                "the agent used this tool; the user asked again and "
                if item["how"] == "re_asked"
                else "the search step ranked this tool first; "
            )
            + f"{item['served_by']} served it"
            for number, item in enumerate(confused or [], 1)
        ]
        failure_lines = []
        for number, call in enumerate(failures, 1):
            line = (
                f"{number}. arguments {json.dumps(call.arguments, default=str)[:300]}"
                f" -> error: {call.error}"
            )
            if call.recovered_with is not None:
                line += (
                    " | worked next with "
                    f"{json.dumps(call.recovered_with, default=str)[:300]}"
                )
            failure_lines.append(line)
        miss_lines = []
        for number, miss in enumerate(misses, 1):
            asked = json.dumps(_clip(miss["query"], 240))
            if miss["kind"] == "re_asked_request":
                miss_lines.append(
                    f"{number}. {asked}: the agent used "
                    f"{miss.get('used') or 'no tool that worked'}; the user then "
                    f"asked {json.dumps(_clip(miss.get('then'), 240))}, which "
                    "reached this tool"
                )
            else:
                miss_lines.append(
                    f"{number}. {asked}: this tool served it, and the search step "
                    f"ranked it {miss.get('rank') or 'outside the closest tools'}"
                )
        return _REVIEW_PROMPT.format(
            name=tool,
            signature=str(row.get("signature") or ""),
            description=_clip(base, 1200) or "(none)",
            arguments=", ".join(argument_names(row.get("parameters"))) or "(none)",
            notes="; ".join(notes) or "(none)",
            queries="; ".join(as_query_list(row.get("queries"))) or "(none)",
            failures="\n".join(failure_lines) or "(none)",
            misses="\n".join(miss_lines) or "(none)",
            confusions="\n".join(confusion_lines) or "(none)",
            min_confusions=self.config.min_confusions,
            min_failures=self.config.min_failures,
            min_misses=self.config.min_misses,
            max_new_queries=self.config.max_new_queries,
        )

    def _revision(
        self,
        tool: str,
        row: Mapping[str, Any],
        failures: List[ToolCall],
        misses: List[Dict[str, Any]],
        parsed: Mapping[str, Any],
        calls: List[ToolCall],
        confused: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolRevision:
        confused = list(confused or [])
        base, lines = split_description(row.get("description") or row.get("docstring"))
        notes = [line for line in lines if line_kind(line) == "note"]
        avoid = [line for line in lines if line_kind(line) == "avoid"]
        shape = next((line for line in lines if line_kind(line) == "example_call"), "")
        queries = as_query_list(row.get("queries"))
        sources = self.example_sources(tool, row)
        revision = ToolRevision(
            revision_id=f"toolrev-{uuid.uuid4().hex[:12]}",
            agent_id=self.agent_id,
            tool=tool,
            description_before=join_description(base, lines),
            queries_before=list(queries),
            created_at=utc_now_iso(),
        )
        literals = set()
        for call in calls:
            literals.update(_literal_values(call.arguments))
            if call.recovered_with:
                literals.update(_literal_values(call.recovered_with))
        used_failures, used_misses, used_confusions = set(), set(), set()

        def refuse(kind: str, text: str, reason: str) -> None:
            revision.refused.append({"kind": kind, "text": text, "reason": reason})

        def cited(item: Mapping[str, Any], key: str, size: int) -> List[int]:
            numbers = item.get(key)
            if not isinstance(numbers, list):
                return []
            return sorted(
                {
                    int(number)
                    for number in numbers
                    if isinstance(number, int)
                    and not isinstance(number, bool)
                    and 1 <= number <= size
                }
            )

        def unsafe(text: str) -> Optional[str]:
            folded = text.casefold()
            if "\n" in text:
                return "more_than_one_line"
            if _AIMED_AT_MODEL_RE.search(text):
                return "aimed_at_the_model"
            if any(value in folded for value in literals):
                return "copies_an_argument_value"
            return None

        proposed_notes = parsed.get("notes")
        for item in (proposed_notes if isinstance(proposed_notes, list) else [])[:2]:
            if not isinstance(item, Mapping):
                continue
            text = " ".join(str(item.get("text") or "").split())
            if not text:
                continue
            numbers = cited(item, "calls", len(failures))
            reason = unsafe(text)
            if reason is None and len(text) > self.config.max_note_chars:
                reason = "too_long"
            if reason is None and len(numbers) < self.config.min_failures:
                reason = "too_few_failed_calls"
            if reason is None and any(
                _same_text(text, known) for known in [*notes, *revision.notes_added]
            ):
                reason = "already_noted"
            if reason:
                refuse("note", text, reason)
                continue
            revision.notes_added.append(text)
            used_failures.update(numbers)

        proposed_queries = parsed.get("queries")
        limit = self.config.max_new_queries
        kept_queries = list(queries)
        for item in proposed_queries if isinstance(proposed_queries, list) else []:
            if not isinstance(item, Mapping):
                continue
            text = " ".join(str(item.get("text") or "").split())
            if not text:
                continue
            numbers = cited(item, "from", len(misses))
            reason = unsafe(text)
            if reason is None and not 8 <= len(text) <= self.config.max_query_chars:
                reason = "wrong_length"
            if reason is None and not numbers:
                reason = "no_missed_request"
            if reason is None and len(numbers) < self.config.min_misses:
                reason = "too_few_missed_requests"
            if reason is None and not all(
                _content_words(text) & _content_words(misses[number - 1]["query"])
                for number in numbers
            ):
                reason = "not_in_the_cited_requests"
            # The same request already kept: a learned one stays as it is; a
            # synthetic one (written at registration) gives way to the real one.
            same = [known for known in kept_queries if _same_text(text, known)]
            if reason is None and any(
                sources.get(known) != SYNTHETIC or known in revision.queries_added
                for known in same
            ):
                reason = "already_kept"
            if reason is None and len(revision.queries_added) >= limit:
                reason = "no_room"
            giving_way = list(same)
            if reason is None and not giving_way:
                if len(kept_queries) >= self.config.max_queries:
                    # At the cap: the oldest synthetic example makes room.
                    oldest = next(
                        (
                            known
                            for known in kept_queries
                            if sources.get(known) == SYNTHETIC
                        ),
                        None,
                    )
                    if oldest is None:
                        reason = "no_room"
                    else:
                        giving_way = [oldest]
            if reason:
                refuse("query", text, reason)
                continue
            for known in giving_way:
                kept_queries.remove(known)
                revision.queries_displaced.append(known)
            kept_queries.append(text)
            revision.queries_added.append(text)
            used_misses.update(numbers)

        registered = set(self.tool_rows())
        proposed_avoid = parsed.get("avoid")
        for item in (proposed_avoid if isinstance(proposed_avoid, list) else [])[:4]:
            if not isinstance(item, Mapping):
                continue
            text = " ".join(str(item.get("text") or "").split())
            if not text:
                continue
            other = str(item.get("tool") or "").strip()
            numbers = cited(item, "from", len(confused))
            reason = unsafe(text)
            if reason is None and len(text) > self.config.max_note_chars:
                reason = "too_long"
            if reason is None and not text.startswith(AVOID_PREFIX):
                reason = "not_a_not_for_line"
            if reason is None and (other == tool or other not in registered):
                reason = "names_no_other_tool"
            if reason is None and other not in text:
                reason = "does_not_name_the_tool_to_use"
            if reason is None and any(
                confused[number - 1]["served_by"] != other for number in numbers
            ):
                reason = "cites_requests_another_tool_served"
            if reason is None and len(numbers) < self.config.min_confusions:
                reason = "too_few_confused_requests"
            if reason is None and not all(
                _content_words(text) & _content_words(confused[number - 1]["query"])
                for number in numbers
            ):
                reason = "not_in_the_cited_requests"
            if reason is None and any(
                other in known for known in [*avoid, *revision.avoid_added]
            ):
                reason = "already_noted"
            if reason:
                refuse("avoid", text, reason)
                continue
            revision.avoid_added.append(text)
            used_confusions.update(numbers)

        # The shape of a call that works, for a tool whose calls keep being
        # refused. Built by the code from the calls that worked.
        new_shape = (
            call_shape(calls) if len(failures) >= self.config.min_failures else ""
        )
        if new_shape and new_shape != shape and unsafe(new_shape) is None:
            revision.call_example = new_shape
            shape = new_shape

        kept_lines = [
            *[*notes, *revision.notes_added][-self.config.max_notes :],
            *[*avoid, *revision.avoid_added][-self.config.max_avoid :],
            *([shape] if shape else []),
        ]
        revision.description_after = join_description(base, kept_lines)
        revision.queries_after = kept_queries
        for number in sorted(used_failures):
            call = failures[number - 1]
            revision.evidence.append(
                {
                    "kind": "failed_call",
                    "query": _clip(call.query, 160),
                    "error": call.error,
                    "argument_names": sorted(call.arguments),
                    "source_id": call.source_id,
                }
            )
        for number in sorted(used_misses):
            miss = misses[number - 1]
            evidence = {
                "kind": miss["kind"],
                "query": _clip(miss["query"], 160),
                "rank": miss.get("rank"),
                "source_id": miss.get("source_id"),
            }
            if miss["kind"] == "re_asked_request":
                evidence["then"] = _clip(miss.get("then"), 160)
                evidence["used"] = miss.get("used") or ""
            revision.evidence.append(evidence)
            revision.ranks.append(
                {"query": _clip(miss["query"], 160), "before": miss.get("rank")}
            )
        for number in sorted(used_confusions):
            item = confused[number - 1]
            revision.evidence.append(
                {
                    "kind": "confused_request",
                    "query": _clip(item["query"], 160),
                    "served_by": item["served_by"],
                    "how": item["how"],
                    "source_id": item.get("source_id"),
                }
            )
        return revision

    def example_sources(
        self, tool: str, row: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, str]:
        """Where each example request kept on ``tool`` came from: ``learned``
        (a revision added it from real requests), ``synthetic`` (written when
        the tool was registered with augmentation) or ``unknown``. Only a
        synthetic one ever gives way to a learned one."""
        if row is None:
            rows = self.tool_rows().get(tool) or []
            row = rows[0] if rows else {}
        queries = as_query_list(row.get("queries"))
        learned = set()
        for revision in self.list_revisions(tool=tool, status="applied"):
            if not revision.queries_withdrawn:
                learned.update(revision.queries_added)
        marked = str(row.get(QUERIES_SOURCE) or "").lower()
        base = split_description(row.get("description"))[0]
        docstring = " ".join(str(row.get("docstring") or "").split())
        # A row from before the marker: augmentation is the only thing that
        # wrote a description other than the docstring.
        augmented = marked == "augmentation" or (
            not marked and bool(base) and " ".join(base.split()) != docstring
        )
        return {
            query: LEARNED if query in learned else SYNTHETIC if augmented else UNKNOWN
            for query in queries
        }

    # ------------------------------------------------------- apply and undo

    def _current(self, tool: str) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        rows = self.tool_rows().get(tool) or []
        if not rows:
            raise ValueError(f"No toolbox row of this agent is named {tool!r}")
        base, notes = split_description(
            rows[0].get("description") or rows[0].get("docstring")
        )
        return (
            rows,
            join_description(base, notes),
            as_query_list(rows[0].get("queries")),
        )

    def apply(self, revision_id: str, *, approved_by: str) -> ToolRevision:
        """Write a proposed revision to the tool's rows and re-embed them."""
        if not str(approved_by or "").strip():
            raise ValueError("approved_by is required")
        revision = self.get_revision(revision_id)
        if revision is None:
            raise ValueError(f"No tool revision {revision_id!r}")
        if revision.status != "proposed":
            raise ValueError(f"Revision {revision_id} is {revision.status}")
        rows, description, _ = self._current(revision.tool)
        if description != revision.description_before:
            revision.status = "dismissed"
            self._save(revision)
            raise ValueError(
                f"{revision.tool} was edited after the review; the revision "
                "was dropped. Review it again."
            )
        # The requests this revision rests on are the ones it says belong to
        # this tool, so another tool losing them (or the same request with
        # other numbers in it) is the point, not a harm.
        claimed = [str(item.get("query") or "") for item in revision.ranks]
        others = [
            item
            for item in (
                self._others_requests(revision.tool) if revision.queries_added else []
            )
            if not any(_same_wording(item[1], query) for query in claimed)
        ]
        self._write_rows(rows, revision.description_after, revision.queries_after)
        top_k = self._top_k()
        for tool, query, before in others:
            after = self.rank(tool, query)
            if after is None or after > top_k:
                revision.displaced.append(
                    {
                        "tool": tool,
                        "query": _clip(query, 160),
                        "before": before,
                        "after": after,
                    }
                )
        if revision.displaced:
            # The examples took another tool's requests: keep the notes only.
            revision.queries_withdrawn = True
            revision.queries_after = list(revision.queries_before)
            self._write_rows(rows, revision.description_after, revision.queries_after)
        for item in revision.ranks:
            item["after"] = self.rank(revision.tool, item.get("query") or "")
        revision.status = "applied"
        revision.applied_at = utc_now_iso()
        revision.approved_by = str(approved_by)
        self._save(revision)
        self._refresh_shown(revision.tool)
        return revision

    def _others_requests(self, tool: str) -> List[Tuple[str, str, int]]:
        """Recent requests other tools served and routing offered them for:
        ``(tool, request, rank)``. A revision of ``tool`` must leave each of
        them among the tools offered."""
        top_k = self._top_k()
        found: List[Tuple[str, str, int]] = []
        seen = set()
        for other, items in self.calls().items():
            if other == tool:
                continue
            for call in reversed(items):
                if not call.final or not call.query or call.query in seen:
                    continue
                if len(found) >= 12:
                    return found
                seen.add(call.query)
                rank = self.rank(other, call.query)
                if rank is not None and rank <= top_k:
                    found.append((other, call.query, rank))
        return found

    def dismiss(self, revision_id: str, *, approved_by: str) -> ToolRevision:
        """Turn a proposed revision down. Its calls are not reviewed again."""
        revision = self.get_revision(revision_id)
        if revision is None:
            raise ValueError(f"No tool revision {revision_id!r}")
        if revision.status != "proposed":
            raise ValueError(f"Revision {revision_id} is {revision.status}")
        revision.status = "dismissed"
        revision.approved_by = str(approved_by)
        self._save(revision)
        return revision

    def undo(self, tool: str, *, approved_by: str) -> ToolRevision:
        """Take back the newest applied revision of ``tool``."""
        if not str(approved_by or "").strip():
            raise ValueError("approved_by is required")
        applied = self.list_revisions(tool=tool, status="applied")
        if not applied:
            raise ValueError(f"{tool} has no applied revision to undo")
        revision = max(applied, key=lambda item: str(item.applied_at or ""))
        rows, description, _ = self._current(tool)
        if description != revision.description_after:
            raise ValueError(
                f"{tool} was edited after this revision; undo would lose the edit"
            )
        self._write_rows(rows, revision.description_before, revision.queries_before)
        revision.status = "undone"
        revision.approved_by = str(approved_by)
        self._save(revision)
        self._refresh_shown(tool)
        return revision

    def learned_notes(self) -> Dict[str, List[str]]:
        """What the model is shown for each tool beyond its own description:
        the lines the newest applied revision left (usage notes, "Not for"
        lines, an example call). With ``show_learned_examples`` on, also the
        example requests learned from use, on one "Also asked for as" line;
        by default those stay retrieval keys the model never reads."""
        newest: Dict[str, ToolRevision] = {}
        learned: Dict[str, List[str]] = {}
        applied = self.list_revisions(status="applied")
        applied.sort(key=lambda item: str(item.applied_at or ""))
        for revision in applied:
            newest[revision.tool] = revision
            if not revision.queries_withdrawn:
                learned.setdefault(revision.tool, []).extend(revision.queries_added)
        shown: Dict[str, List[str]] = {}
        for tool, revision in newest.items():
            lines = split_description(revision.description_after)[1]
            kept = [
                query
                for query in dict.fromkeys(learned.get(tool, []))
                if query in revision.queries_after
            ][-4:]
            if kept and self.config.show_learned_examples:
                lines = [
                    *lines,
                    ALSO_ASKED + " " + "; ".join(json.dumps(query) for query in kept),
                ]
            shown[tool] = lines
        return shown

    def _refresh_shown(self, tool: str) -> None:
        self._show_notes(tool, self.learned_notes().get(tool, []))

    # ------------------------------------------------------------ usage flags

    def offers(self) -> Dict[str, Any]:
        """What routing offered, counted per turn: ``{turns, since, tools:
        {name: {turns, last}}}``. Empty until a routed turn has run."""
        record = self._store().lookup(f"tool-offers-{self.agent_id}") or {}
        tools = record.get("tools")
        return {
            "turns": int(record.get("turns") or 0),
            "since": record.get("since"),
            "tools": dict(tools) if isinstance(tools, Mapping) else {},
        }

    def usage_flags(
        self,
        *,
        calls: Optional[Mapping[str, List[ToolCall]]] = None,
        rows: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tools worth a person's look, from what routing offered and what
        was called: ``retrieved_never_called`` (offered in ``min_offers``
        turns or more, and no call) and ``never_retrieved`` (never offered
        and never called, over ``min_turns`` routed turns or more). A flag
        proposes nothing."""
        try:
            offers = self.offers()
        except Exception as exc:
            logger.debug("Tool offers could not be read: %s", exc)
            return []
        if not offers["turns"]:
            return []
        calls = self.calls() if calls is None else calls
        rows = self.tool_rows() if rows is None else rows
        flags: List[Dict[str, Any]] = []
        for tool in sorted(rows):
            offered = int((offers["tools"].get(tool) or {}).get("turns") or 0)
            called = len(calls.get(tool, []))
            flag = None
            if offered >= self.config.min_offers and not called:
                flag = RETRIEVED_NEVER_CALLED
            elif (
                not offered and not called and offers["turns"] >= self.config.min_turns
            ):
                flag = NEVER_RETRIEVED
            if flag:
                flags.append(
                    {
                        "tool": tool,
                        "flag": flag,
                        "offered_turns": offered,
                        "calls": called,
                        "turns_observed": offers["turns"],
                        "since": offers["since"],
                    }
                )
        return flags


__all__ = [
    "AVOID_PREFIX",
    "CALL_EXAMPLE",
    "LEARNED_HEADING",
    "NEVER_RETRIEVED",
    "RETRIEVED_NEVER_CALLED",
    "TOOL_OFFERS_RECORD",
    "TOOL_REVISION_RECORD",
    "ToolCall",
    "ToolLearner",
    "ToolLearningConfig",
    "ToolReview",
    "ToolRevision",
    "as_query_list",
    "call_shape",
    "join_description",
    "line_kind",
    "note_tools_offered",
    "split_description",
]
