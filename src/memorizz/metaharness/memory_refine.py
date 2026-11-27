# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The refine cycle of a memory that harnesses share.

Every harness run given a ``memory_id`` leaves its answer there as a raw
conversation turn. Raw turns are kept for good: they are the record of who
said what. What later runs are handed, though, should be what the memory
*knows*, not three harnesses saying the same sentence. A cycle makes that:

1. **Findings.** A model reads the answers not yet distilled, beside the
   findings the memory already holds, and proposes durable facts. Each must
   quote the answer it came from, word for word, or it is dropped. A fact
   the memory already holds is *confirmed* (one more run agrees) instead of
   written again; one a newer answer corrects is *replaced*: the old note
   is kept, hidden, and points at its successor.
2. **Raw answers.** Each answer read is marked as distilled. It stays in
   conversation memory, in the timeline and on the memory's page; it is
   only left out of the context later runs start with.

The report has the shape of an agent's refine cycle
(:class:`memorizz.learning.refine.RefineRun`) and is stored the same way,
so one reader shows both. ``dry_run`` plans and changes nothing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .._env_io import memorizz_home
from .._time import utc_now_iso
from ..enums.memory_type import MemoryType
from ..learning.refine import REFINE_RUN_RECORD, RefineJob, RefineRun, _TokenMeter

logger = logging.getLogger(__name__)

FINDINGS_NAMESPACE = "harness_findings"
# "[Harness run: droid · succeeded]", "[Harness plan stage 1/4: Diagnose ·
# codex · succeeded]", "[Harness comparison run 2/3: pi · failed]".
SAVED_HEADER = re.compile(
    r"^\[Harness (?:run|plan stage|comparison run)(?: \d+/\d+)?: ([^\]]+)\]\n?"
)
SAVED_FOOTER = re.compile(r"\n?Harness run: ([0-9a-f-]{8,})\s*$")

#: Answers read in one cycle, and how much of each.
MAX_ANSWERS = 12
MAX_ANSWER_CHARS = 3000
#: Findings shown to the model so it can confirm or replace them.
MAX_KNOWN = 60
DEFAULT_AFTER_ANSWERS = 3

JOBS = (("findings", "Findings"), ("raw_answers", "Raw answers"))

PROMPT = """You keep the shared memory of a software project that several coding
agents work on. Below are answers those agents gave in recent runs, and the
findings the memory already holds.

Pull out what is worth remembering for later runs: facts about the project,
decisions, results, conventions, things that were tried and what happened.
A change a run made to the project's files is a result. Keep what it changed
and, when the answer says so, whether its check passed ("Files changed" and
"Verification passed" or "Verification failed" lines say this).
Skip anything that is not a lasting fact: a restated question, a description
of how the agent searched or read, "nothing was found", apologies, offers.

Answer with one JSON object and nothing else:
{"findings": [
  {"action": "add", "text": "<one self-contained fact, plain words>",
   "run": "<run id it came from>", "quote": "<exact words copied from that answer>"},
  {"action": "confirm", "note": "<id of a held finding>", "run": "<run id that agrees>",
   "quote": "<exact words copied from that answer>"},
  {"action": "replace", "note": "<id of a held finding>", "text": "<the corrected fact>",
   "run": "<run id>", "quote": "<exact words copied from that answer>",
   "why": "<what changed>"}
]}

Rules:
- One fact per finding. Write it so it makes sense with no other context.
- "quote" must be copied exactly from the answer of the run you name.
- If a held finding already says it, use "confirm", never "add".
- Use "replace" only when a newer answer says a held finding is no longer true.
- If nothing is worth keeping, answer {"findings": []}.

Findings the memory already holds:
__KNOWN__

Answers:
__ANSWERS__
"""


def _plain(text: Any) -> str:
    """Lower-case words only, so a quote matches through Markdown marks."""
    return " ".join(re.sub(r"[*_`#>\[\]()]", " ", str(text or "")).lower().split())


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("_id") or row.get("id") or "")


def parse_saved(content: Any) -> Dict[str, str]:
    """A saved harness answer taken apart: who wrote it, the run, the text,
    and the stage's name when it was a stage of a plan."""
    text = str(content or "")
    header, footer = SAVED_HEADER.match(text), SAVED_FOOTER.search(text)
    parts = [part.strip() for part in header.group(1).split(" · ")] if header else []
    status = parts.pop() if len(parts) > 1 else ""
    harness = parts.pop() if parts else ""
    return {
        "harness": harness,
        "status": status,
        "stage": parts[0] if parts and parts[0] != harness else "",
        "run_id": footer.group(1) if footer else "",
        "text": SAVED_FOOTER.sub("", SAVED_HEADER.sub("", text)).strip(),
    }


def refine_owner(memory_id: str) -> str:
    """The name reports of this memory's cycles are stored under."""
    return f"harness-memory:{memory_id}"


class RefineSettings:
    """Which model distils, and whether cycles start by themselves.

    Kept on this machine (``<MEMORIZZ_HOME>/harness-memory-refine.json``),
    like the judge's: the model and its key are this machine's.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or memorizz_home() / "harness-memory-refine.json"

    @staticmethod
    def clean(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        value = dict(value or {})
        provider = str(value.get("provider") or "").strip().lower()
        model = str(value.get("model") or "").strip()
        if len(provider) > 40 or len(model) > 240:
            raise ValueError("The model name is too long")
        try:
            after = int(value.get("after_answers") or DEFAULT_AFTER_ANSWERS)
        except (TypeError, ValueError) as exc:
            raise ValueError("after_answers must be a whole number") from exc
        return {
            "provider": provider,
            "model": model,
            # Off until a model is chosen: a cycle spends tokens.
            "auto": bool(value.get("auto")) and bool(provider and model),
            "after_answers": max(1, min(after, 50)),
        }

    def get(self) -> Dict[str, Any]:
        try:
            return self.clean(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return self.clean(None)

    def set(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        cleaned = self.clean({**self.get(), **dict(value)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
        return cleaned


def _default_model(settings: Mapping[str, Any]) -> Any:
    if not settings.get("provider") or not settings.get("model"):
        return None
    from ..llms.llm_factory import create_llm_provider

    return create_llm_provider(
        {"provider": settings["provider"], "model": settings["model"]}
    )


def _model_label(model: Any, settings: Mapping[str, Any]) -> str:
    """The model a cycle ran on, as its report names it: the one passed in,
    else the one in the settings."""
    config: Mapping[str, Any] = {}
    if isinstance(model, Mapping):
        config = model
    elif model is not None:
        reader = getattr(model, "get_config", None)
        try:
            found = reader() if callable(reader) else None
        except Exception:
            found = None
        config = found if isinstance(found, Mapping) else {}
        if not config.get("model") and getattr(model, "model", None):
            config = {**config, "model": getattr(model, "model")}
    else:
        config = settings
    name = str(config.get("model") or config.get("deployment_name") or "").strip()
    provider = str(config.get("provider") or "").strip()
    return " · ".join(part for part in (provider, name) if part) if name else ""


class HarnessMemoryRefine:
    """Plan and run refine cycles over the memories harnesses share."""

    def __init__(
        self,
        provider: Any,
        *,
        settings: Optional[RefineSettings] = None,
        model_factory: Callable[[Mapping[str, Any]], Any] = _default_model,
        task_of: Optional[Callable[[str], str]] = None,
    ):
        if provider is None:
            raise ValueError("Refining a memory requires a memory provider")
        self.provider = provider
        self.settings = settings or RefineSettings()
        self._model_factory = model_factory
        #: The request a run was given, by run id, to read its answer by.
        self._task_of = task_of or (lambda run_id: "")
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ----------------------------------------------------------------- reading

    def _rows(self, memory_type: MemoryType, user_id: Optional[str]) -> List[Dict]:
        try:
            rows = self.provider.list_all(memory_type, user_id=user_id)
        except TypeError:
            rows = self.provider.list_all(memory_type)
        return [row for row in rows or [] if isinstance(row, dict)]

    def answers(
        self, memory_id: str, user_id: Optional[str] = None, *, only_new: bool = False
    ) -> List[Dict[str, Any]]:
        """The raw answers harness runs saved to this memory, oldest first."""
        found = []
        for row in self._rows(MemoryType.CONVERSATION_MEMORY, user_id):
            if row.get("memory_id") != memory_id or row.get("user_id") != user_id:
                continue
            saved = parse_saved(row.get("content"))
            if not saved["harness"] or not saved["text"]:
                continue  # not a harness answer
            if only_new and row.get("distilled"):
                continue
            found.append(
                {
                    "id": _row_id(row),
                    "at": str(row.get("timestamp") or row.get("created_at") or ""),
                    "distilled": row.get("distilled") or None,
                    **saved,
                }
            )
        found.sort(key=lambda item: item["at"])
        return found

    def findings(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        *,
        include_replaced: bool = False,
    ) -> List[Dict[str, Any]]:
        """What the memory knows: its findings, most confirmed first."""
        found = []
        for row in self._rows(MemoryType.KNOWLEDGE_BASE, user_id):
            if (
                row.get("namespace") != FINDINGS_NAMESPACE
                or row.get("memory_id") != memory_id
                or row.get("user_id") != user_id
            ):
                continue
            replaced = bool(str(row.get("superseded_by") or "").strip())
            if replaced and not include_replaced:
                continue
            found.append(
                {
                    "id": _row_id(row),
                    "text": str(row.get("content") or ""),
                    "runs": list(row.get("source_runs") or []),
                    "harnesses": list(row.get("source_harnesses") or []),
                    "quote": str(row.get("quote") or ""),
                    "confirmations": int(row.get("confirmations") or 1),
                    "at": str(row.get("created_at") or row.get("timestamp") or ""),
                    "replaced_by": str(row.get("superseded_by") or ""),
                    "why_replaced": str(row.get("suppression_reason") or ""),
                    "replaces": str(row.get("replaces") or ""),
                }
            )
        found.sort(key=lambda item: (-item["confirmations"], item["at"]))
        return found

    # ---------------------------------------------------------------- planning

    def plan(
        self, memory_id: str, user_id: Optional[str] = None, *, model: Any = None
    ) -> Dict[str, Any]:
        """What a cycle would change. Nothing is written.

        ``ops`` are the changes that passed their checks; ``dropped`` the
        ones that did not, each with why.
        """
        from ..learning.consolidation import _parse_json, text_generator

        fresh = self.answers(memory_id, user_id, only_new=True)[:MAX_ANSWERS]
        known = self.findings(memory_id, user_id)[:MAX_KNOWN]
        plan: Dict[str, Any] = {
            "memory_id": memory_id,
            "user_id": user_id,
            "answers": fresh,
            "ops": [],
            "dropped": [],
            "warnings": [],
        }
        if not fresh:
            return plan
        if isinstance(model, Mapping):
            # ``model={"provider": ..., "model": ...}``, as an agent's llm_config.
            model = _default_model(model)
        generate = text_generator(model or self._model_factory(self.settings.get()))
        if generate is None:
            plan["warnings"].append("No model is set for refining harness memory.")
            return plan
        by_run = {item["run_id"]: item for item in fresh if item["run_id"]}
        by_note = {item["id"]: item for item in known}
        prompt = PROMPT.replace(
            "__KNOWN__",
            "\n".join(f"- [{item['id']}] {item['text']}" for item in known) or "(none)",
        ).replace(
            "__ANSWERS__",
            "\n\n".join(
                f"Run {item['run_id']} ({item['harness']})"
                + (
                    f"\nAsked: {self._task_of(item['run_id'])[:600]}"
                    if self._task_of(item["run_id"])
                    else ""
                )
                + f"\nAnswer: {item['text'][:MAX_ANSWER_CHARS]}"
                for item in fresh
            ),
        )
        try:
            answer = _parse_json(generate(prompt))
        except Exception as exc:
            plan["warnings"].append(f"The model call failed: {type(exc).__name__}")
            return plan
        if not isinstance(answer, dict) or not isinstance(answer.get("findings"), list):
            plan["warnings"].append("The model's answer could not be read.")
            return plan
        for item in answer["findings"]:
            op, why = self._checked(item, by_run, by_note, plan["ops"])
            if op is not None:
                plan["ops"].append(op)
            elif isinstance(item, dict) and why:
                plan["dropped"].append(
                    {
                        "text": str(item.get("text") or item.get("note") or "")[:300],
                        "why": why,
                    }
                )
        # A model tends to credit the first run that said a thing. Every
        # other answer of this cycle that holds the quoted words agrees too.
        for op in plan["ops"]:
            counted = set(op["runs"]) | set(
                (by_note.get(op["note"]) or {}).get("runs") or []
                if op["action"] == "confirm"
                else []
            )
            for item in fresh:
                if (
                    item["run_id"]
                    and item["run_id"] not in counted
                    and _plain(op["quote"]) in _plain(item["text"])
                ):
                    counted.add(item["run_id"])
                    op["runs"].append(item["run_id"])
                    op["harnesses"].append(item["harness"])
        return plan

    def _checked(
        self,
        item: Any,
        by_run: Mapping[str, Dict[str, Any]],
        by_note: Mapping[str, Dict[str, Any]],
        accepted: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """One proposed change, or why it is not taken."""
        from ..learning.consolidation import _same_statement

        if not isinstance(item, dict):
            return None, "not a finding"
        action = str(item.get("action") or "").strip().lower()
        source = by_run.get(str(item.get("run") or "").strip())
        if action not in {"add", "confirm", "replace"} or source is None:
            return None, "it names no answer read in this cycle"
        quote = " ".join(str(item.get("quote") or "").split())
        # The agents' rule for a stated fact: it is quoted from the
        # transcript, or it is not taken.
        if len(_plain(quote)) < 8 or _plain(quote) not in _plain(source["text"]):
            return None, "its quote is not in that run's answer"
        text = " ".join(str(item.get("text") or "").split())[:600]
        note = by_note.get(str(item.get("note") or "").strip())
        if action == "add":
            if not text:
                return None, "it has no text"
            # A model that adds what is already held is corrected here.
            note = next(
                (
                    held
                    for held in by_note.values()
                    if _same_statement(held["text"], text)
                ),
                None,
            )
            action = "confirm" if note else "add"
            twin = next(
                (
                    op
                    for op in accepted
                    if op["action"] != "confirm" and _same_statement(op["text"], text)
                ),
                None,
            )
            if not note and twin is not None:
                # Another run of this cycle says the same: it agrees.
                if source["run_id"] not in twin["runs"]:
                    twin["runs"].append(source["run_id"])
                    twin["harnesses"].append(source["harness"])
                return None, ""
        if action in {"confirm", "replace"} and note is None:
            return None, "it names no finding the memory holds"
        if action == "replace" and (not text or _same_statement(note["text"], text)):
            action = "confirm"
        if action == "confirm" and source["run_id"] in note["runs"]:
            return None, "that run is already counted for this finding"
        return (
            {
                "action": action,
                "text": text if action != "confirm" else note["text"],
                "note": note["id"] if note else "",
                "run": source["run_id"],
                "harness": source["harness"],
                # Every run of this cycle that says it, the first one quoted.
                "runs": [source["run_id"]],
                "harnesses": [source["harness"]],
                "quote": quote[:400],
                "why": " ".join(str(item.get("why") or "").split())[:300],
            },
            "",
        )

    # ---------------------------------------------------------------- applying

    def _write(self, op: Mapping[str, Any], plan: Mapping[str, Any], cycle: str) -> str:
        from ..long_term.semantic.knowledge_base import KnowledgeBase

        memory_id, user_id = plan["memory_id"], plan["user_id"]
        knowledge_base_id = KnowledgeBase(self.provider).ingest_knowledge(
            op["text"],
            FINDINGS_NAMESPACE,
            chunking_strategy="none",
            user_id=user_id,
            metadata={
                "memory_id": memory_id,
                "source": "harness_refine",
                "source_runs": list(op["runs"]),
                "source_harnesses": list(dict.fromkeys(op["harnesses"])),
                "quote": op["quote"],
                "confirmations": len(op["runs"]),
                "refine_run_id": cycle,
                **({"replaces": op["note"]} if op["action"] == "replace" else {}),
            },
            embeddings="optional",
        )
        for row in self._rows(MemoryType.KNOWLEDGE_BASE, user_id):
            if row.get("knowledge_base_id") == knowledge_base_id:
                return _row_id(row)
        return str(knowledge_base_id)

    def record_outcome(
        self,
        memory_id: str,
        user_id: Optional[str],
        handoff: Mapping[str, Any],
        task: str = "",
    ) -> Optional[str]:
        """Keep what the host saw a run do, as a finding, when the run ends.

        Which files a run changed and whether its check passed are observed
        by MemoRizz, not claimed by the harness, so they need no model and no
        refine cycle: an agent sharing the memory can answer "is it done, and
        was it checked?" at once. A run that changed nothing and had no check
        leaves no outcome; its answer waits for a cycle like any other."""
        files = [str(path) for path in handoff.get("files_changed") or [] if path]
        check = handoff.get("verification")
        check = check if isinstance(check, Mapping) and check.get("command") else None
        run_id = str(handoff.get("run_id") or "")
        harness = str(handoff.get("harness") or "")
        if not memory_id or not run_id or not harness or not (files or check):
            return None
        shown = ", ".join(files[:6]) + (
            f" and {len(files) - 6} more" if len(files) > 6 else ""
        )
        parts = [
            f"{harness} changed {shown}" if files else f"{harness} changed no files"
        ]
        if check:
            parts.append(
                "its check passed" if check.get("verified") else "its check failed"
            )
        else:
            parts.append("no check was run")
        text = " and ".join(parts) + "."
        asked = " ".join(str(task or "").split())
        if asked:
            text += f" Asked: {asked[:280]}{'…' if len(asked) > 280 else ''}"
        quote = (
            "Files changed: " + ", ".join(files)
            if files
            else f"Verification {'passed' if check.get('verified') else 'failed'}"
        )
        op = {
            "action": "add",
            "text": text,
            "runs": [run_id],
            "harnesses": [harness],
            "quote": quote,
        }
        return self._write(
            op, {"memory_id": memory_id, "user_id": user_id}, "run-outcome"
        )

    def apply(self, plan: Mapping[str, Any], cycle: str) -> Dict[str, int]:
        """Make a plan's changes. Returns how many of each were made."""
        done = {"added": 0, "confirmed": 0, "replaced": 0, "answers_distilled": 0}
        now = utc_now_iso()
        for op in plan.get("ops") or []:
            if op["action"] == "confirm":
                row = self.provider.retrieve_by_id(
                    op["note"], MemoryType.KNOWLEDGE_BASE
                )
                if not isinstance(row, dict):
                    continue
                runs = list(
                    dict.fromkeys([*(row.get("source_runs") or []), *op["runs"]])
                )
                harnesses = list(
                    dict.fromkeys(
                        [*(row.get("source_harnesses") or []), *op["harnesses"]]
                    )
                )
                self.provider.update_by_id(
                    op["note"],
                    {
                        "source_runs": runs,
                        "source_harnesses": harnesses,
                        "confirmations": len(runs),
                        "confirmed_at": now,
                    },
                    MemoryType.KNOWLEDGE_BASE,
                )
                done["confirmed"] += 1
                continue
            new_id = self._write(op, plan, cycle)
            if op["action"] == "add":
                done["added"] += 1
                continue
            # The corrected note stays, hidden, pointing at what replaced it.
            self.provider.update_by_id(
                op["note"],
                {
                    "retention_state": "suppressed",
                    "suppressed_from": "active",
                    "suppressed_at": now,
                    "suppression_reason": op["why"] or "a newer run corrected it",
                    "superseded_by": new_id,
                    "supersede_reason": "contradicts",
                },
                MemoryType.KNOWLEDGE_BASE,
            )
            done["replaced"] += 1
        for answer in plan.get("answers") or []:
            # Kept, and no longer handed to later runs beside its findings.
            if self.provider.update_by_id(
                answer["id"],
                {"distilled": {"refine_run_id": cycle, "at": now}},
                MemoryType.CONVERSATION_MEMORY,
            ):
                done["answers_distilled"] += 1
        return done

    # --------------------------------------------------------------- the cycle

    def _reports(self, memory_id: str) -> Any:
        from ..learning.store import LearningControlPlaneStore

        return LearningControlPlaneStore(
            self.provider, agent_id=refine_owner(memory_id)
        )

    def list_runs(self, memory_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Stored cycles of this memory, newest first."""
        try:
            runs = self._reports(memory_id).list_records(
                record_type=REFINE_RUN_RECORD, limit=2000
            )
        except Exception:
            return []
        runs.sort(key=lambda row: str(row.get("started_at") or ""), reverse=True)
        return runs[: max(1, int(limit))]

    def run(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        *,
        dry_run: bool = False,
        trigger: str = "manual",
        model: Any = None,
    ) -> Dict[str, Any]:
        """Run one cycle over this memory and return its report."""
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id is required to refine a memory")
        with self._guard:
            lock = self._locks.setdefault(memory_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise RuntimeError("A refine cycle of this memory is already running")
        try:
            return self._run(memory_id, user_id, dry_run, trigger, model)
        finally:
            lock.release()

    def _run(
        self,
        memory_id: str,
        user_id: Optional[str],
        dry_run: bool,
        trigger: str,
        model: Any,
    ) -> Dict[str, Any]:
        settings = self.settings.get()
        jobs = {name: RefineJob(name=name) for name, _ in JOBS}
        run = RefineRun(
            run_id=f"refine-{uuid.uuid4().hex[:12]}",
            agent_id=refine_owner(memory_id),
            trigger=trigger,
            started_at=utc_now_iso(),
            dry_run=bool(dry_run),
            scopes=[{"memory_id": memory_id, "user_id": user_id}],
            jobs=list(jobs.values()),
        )
        findings, raw = jobs["findings"], jobs["raw_answers"]
        started = time.perf_counter()
        label = _model_label(model, settings)
        if isinstance(model, Mapping):
            model = _default_model(model)
        model = model or self._model_factory(settings)
        meter = _TokenMeter(model) if model is not None else None
        try:
            if meter is not None:
                with meter:
                    plan = self.plan(memory_id, user_id, model=model)
                findings.tokens += meter.tokens
                findings.tokens_estimated = meter.estimated
            else:
                plan = self.plan(memory_id, user_id, model=model)
            findings.notes.extend(plan["warnings"])
            read = len(plan["answers"])
            if not read:
                findings.status = raw.status = "skipped"
                findings.notes.append("No new answers since the last cycle.")
                run.status, run.reason = "skipped", "nothing new"
            elif plan["warnings"] and not plan["ops"]:
                findings.status, raw.status = "failed", "skipped"
                run.status, run.reason = "failed", plan["warnings"][0]
            else:
                for op in plan["ops"]:
                    kind = {
                        "add": "added",
                        "confirm": "confirmed",
                        "replace": "replaced",
                    }[op["action"]]
                    findings.changes[kind] = findings.changes.get(kind, 0) + 1
                    findings.items.append(
                        {
                            "kind": kind,
                            "text": op["text"],
                            "detail": f"{', '.join(dict.fromkeys(op['harnesses']))} · run {op['run'][:8]}: “{op['quote']}”"
                            + (f" ({op['why']})" if op["why"] else ""),
                            "applied": not dry_run,
                        }
                    )
                for item in plan["dropped"]:
                    findings.changes["not_taken"] = (
                        findings.changes.get("not_taken", 0) + 1
                    )
                    findings.items.append(
                        {
                            "kind": "not_taken",
                            "text": item["text"],
                            "detail": item["why"],
                            "applied": False,
                        }
                    )
                if not dry_run:
                    done = self.apply(plan, run.run_id)
                    raw.changes["distilled"] = done["answers_distilled"]
                else:
                    raw.changes["would_distil"] = read
                raw.notes.append("Every answer stays in conversation memory.")
                findings.status = raw.status = "done"
                run.status = "dry_run" if dry_run else "done"
        except Exception as exc:
            logger.warning("Refining %s failed: %s", memory_id, exc, exc_info=True)
            findings.status = "failed"
            findings.notes.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            run.status, run.reason = "failed", type(exc).__name__
        findings.seconds = time.perf_counter() - started
        run.finished_at = utc_now_iso()
        report = run.to_dict()
        report["memory_id"] = memory_id
        report["model"] = label
        if not dry_run:
            try:
                self._reports(memory_id)._put(report, immutable=False)
            except Exception as exc:
                logger.warning("The refine report was not stored: %s", exc)
        return report

    # ------------------------------------------------------------- by itself

    def maybe_start(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Start a cycle in the background when enough new answers wait.

        Called after a run's answer is saved. It never blocks the run and
        never raises into it.
        """
        try:
            settings = self.settings.get()
            if not settings["auto"]:
                return False
            waiting = len(self.answers(memory_id, user_id, only_new=True))
            if waiting < settings["after_answers"]:
                return False
        except Exception:
            logger.debug("Refine check failed for %s", memory_id, exc_info=True)
            return False

        def work() -> None:
            try:
                self.run(memory_id, user_id, trigger="automatic")
            except RuntimeError:
                pass  # one is already running
            except Exception:
                logger.warning(
                    "Automatic refine of %s failed", memory_id, exc_info=True
                )

        threading.Thread(target=work, name="harness-memory-refine", daemon=True).start()
        return True


def distilled(row: Mapping[str, Any]) -> bool:
    """A row later runs are not handed: a raw answer already distilled into
    findings, or a finding a newer one replaced. Both stay in the store."""
    return bool(row.get("distilled")) or bool(
        str(row.get("superseded_by") or "").strip()
    )
