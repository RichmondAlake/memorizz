# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Refine cycle: an agent's memory upkeep as one ordered cycle, with one report.

Promotion, consolidation, persona learning and retention each used to fire on
their own counter, in no fixed order. A :class:`RefineCycle` calls the same
methods in a fixed order, because later jobs read what earlier ones cleaned:

1.  verify         check stored notes against the conversations they cite
2.  entities       merge duplicates and review attributes (under the source rule)
3.  stated_facts   add facts the user stated, each quoted from the transcript
4.  propagation    flag the skills and notes that still state a replaced value
5.  toolbox        review how the tools were called; propose revisions
6.  skill_staleness  deprecate skills whose tools no longer resolve
7.  promotion      gates, ranking, distillation and validation
8.  skill_review   the deviation flag, the holdout flag, stale references
9.  persona_learning  a goals sentence, and standing rules for their owner
10. compact        embed again what changed, so vectors match the text
11. retention      a retention plan

Verification comes first, so no later job builds on a note the transcript
no longer supports. Propagation follows the entity jobs, because that is
when values are replaced. Compaction comes after everything that edits text.
The top layer a model reads on every turn (persona, standing rules, active
skills) is changed here and nowhere else, so its changes land together, once
a cycle: that is the batching point for a prompt cache.

The cycle changes when the jobs run and in what order. It does not change
what any job may do: each keeps its own apply rule (``auto_apply`` and the
review queues), and guardrails stay out of reach. Capture, routing, reflection
and compaction are not part of it; they still run after each turn.

Each cycle writes one :class:`RefineRun`: when it ran, what started it, and per
job its status, changes by kind, tokens used and the plans left waiting for
review. Those plans are stored with the run, so a restart does not lose them.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .._time import utc_now_iso

logger = logging.getLogger(__name__)

REFINE_RUN_RECORD = "learning_refine_run"
AUTOMATIC_APPROVER = "memorizz:refine_cycle"
#: The jobs of a cycle, in the order they run, with a short label for each.
JOBS: Tuple[Tuple[str, str], ...] = (
    ("verify", "Verify notes"),
    ("entities", "Entities"),
    ("stated_facts", "Stated facts"),
    ("propagation", "Stale references"),
    ("toolbox", "Tool review"),
    ("skill_staleness", "Skill staleness"),
    ("promotion", "Promotion"),
    ("skill_review", "Skill review"),
    ("persona_learning", "Persona learning"),
    ("compact", "Compact and re-embed"),
    ("retention", "Retention"),
)
JOB_LABELS = dict(JOBS)
#: Consolidation op kinds that belong to the stated-facts job; the rest of a
#: consolidation plan is the entities job.
_FACT_KINDS = frozenset({"add_fact", "fact_not_applied", "reinforce_fact"})
#: Lines of detail kept per job; the counts are always complete.
MAX_ITEMS = 40


def describe_op(op: Any) -> Tuple[str, str]:
    """A consolidation change as a person reads it: what, and why."""
    details = op.details or {}
    subject = f"{details.get('entity')}.{details.get('attribute')}"
    if op.kind == "merge_entities":
        names = list(dict.fromkeys(str(n) for n in details.get("names") or [] if n))
        return (
            "Merged " + " and ".join(names) if len(names) > 1 else "Merged duplicates",
            op.reason,
        )
    if op.kind == "add_fact":
        quote = f"The user said: “{details.get('evidence')}”"
        if details.get("beside") is not None:
            return (
                f"Kept {subject} = {details.get('beside')}; noted "
                f"{details.get('value')} beside it",
                quote + ". The stored value came from your application.",
            )
        if details.get("replaces") is not None:
            return (
                f"{subject}: {details.get('replaces')} → {details.get('value')}",
                quote,
            )
        return f"Added {subject} = {details.get('value')}", quote
    if op.kind == "reinforce_fact":
        return (
            f"Confirmed {subject} = {details.get('value')}",
            f"The user said it again: “{details.get('evidence')}”",
        )
    if op.kind == "retire_attribute":
        return (
            f"Retired {subject} = {details.get('value')}",
            f"Same fact as {details.get('kept')} = {details.get('kept_value')}",
        )
    if op.kind == "suppress_duplicate":
        return f"Hid a duplicate {str(op.memory_type).replace('_', ' ')}", op.reason
    return str(op.kind).replace("_", " ").capitalize(), op.reason


@dataclass
class RefineJob:
    """What one job of a cycle did."""

    name: str
    status: str = "pending"  # done, skipped, failed, pending
    changes: Dict[str, int] = field(default_factory=dict)
    waiting: int = 0
    tokens: int = 0
    tokens_estimated: bool = False
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    #: What changed, one line each: ``{kind, text, detail, applied}``. The
    #: counts in ``changes`` are complete; this list stops at ``MAX_ITEMS``.
    items: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": JOB_LABELS.get(self.name, self.name),
            "status": self.status,
            "changes": dict(self.changes),
            "changed": sum(self.changes.values()),
            "waiting": self.waiting,
            "tokens": self.tokens,
            "tokens_estimated": self.tokens_estimated,
            "notes": list(self.notes),
            "seconds": round(self.seconds, 3),
            "items": list(self.items),
        }


@dataclass
class RefineRun:
    """One cycle: when it ran, what started it and what each job did."""

    run_id: str
    agent_id: str
    trigger: str  # manual, schedule
    started_at: str
    status: str = "running"  # running, done, failed, skipped, dry_run
    finished_at: Optional[str] = None
    dry_run: bool = False
    scopes: List[Dict[str, Any]] = field(default_factory=list)
    jobs: List[RefineJob] = field(default_factory=list)
    #: Plans a job left for a person: ``{kind, job, plan_id, memory_id,
    #: user_id, summary, plan}``. Stored here so they survive a restart.
    pending: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        jobs = [job.to_dict() for job in self.jobs]
        return {
            "record_type": REFINE_RUN_RECORD,
            "record_id": self.run_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "trigger": self.trigger,
            "status": self.status,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "timestamp": self.started_at,
            "scopes": list(self.scopes),
            "jobs": jobs,
            "pending": list(self.pending),
            "reason": self.reason,
            "changed": sum(job["changed"] for job in jobs),
            "waiting": len(self.pending),
            "tokens": sum(job["tokens"] for job in jobs),
            "tokens_estimated": any(job["tokens_estimated"] for job in jobs),
        }


class _TokenMeter:
    """Counts the tokens the background model uses while a job runs.

    Wraps the model's text calls for the length of the job. The model's own
    usage report is used when it has one; otherwise the count is estimated
    from the text (about four characters a token) and marked as such.
    """

    def __init__(self, model: Any):
        self.model = model
        self.tokens = 0
        self.estimated = False
        self._saved: Dict[str, Any] = {}

    def _count(self, prompt: Any, reply: Any) -> None:
        usage = None
        reader = getattr(self.model, "get_last_usage", None)
        if callable(reader):
            try:
                usage = reader()
            except Exception:
                usage = None
        if isinstance(usage, Mapping):
            total = usage.get("total_tokens")
            if total is None:
                total = (usage.get("prompt_tokens") or 0) + (
                    usage.get("completion_tokens") or 0
                )
            if total:
                self.tokens += int(total)
                return
        self.estimated = True
        self.tokens += (len(str(prompt or "")) + len(str(reply or ""))) // 4

    def __enter__(self) -> "_TokenMeter":
        for name in ("generate_text", "generate"):
            original = getattr(self.model, name, None)
            if not callable(original):
                continue

            def wrapped(*args: Any, __call: Callable = original, **kwargs: Any) -> Any:
                reply = __call(*args, **kwargs)
                self._count(args[0] if args else kwargs, reply)
                return reply

            try:
                # Whether the model object itself carried the method, or its
                # class did: that decides how it is put back.
                own = name in getattr(self.model, "__dict__", {})
                setattr(self.model, name, wrapped)
                self._saved[name] = (original, own)
            except Exception:
                continue
        return self

    def __exit__(self, *_: Any) -> None:
        for name, (original, own) in self._saved.items():
            try:
                if own:
                    setattr(self.model, name, original)
                else:
                    # Remove the wrapper so the class method shows again.
                    delattr(self.model, name)
            except Exception:
                try:
                    setattr(self.model, name, original)
                except Exception:
                    pass
        self._saved.clear()


class RefineCycle:
    """Runs an agent's upkeep jobs once, in order, and stores the report."""

    def __init__(self, agent: Any):
        self.agent = agent
        if agent.memory_provider is None:
            raise ValueError("A refine cycle requires a memory provider")
        if not agent.agent_id:
            raise ValueError("A refine cycle requires a saved agent (agent_id)")

    # ------------------------------------------------------------------ store

    def _store(self) -> Any:
        from .store import LearningControlPlaneStore

        return LearningControlPlaneStore(
            self.agent.memory_provider, agent_id=str(self.agent.agent_id)
        )

    def _save(self, run: RefineRun) -> None:
        try:
            self._store()._put(run.to_dict(), immutable=False)
        except Exception as exc:
            logger.warning("Refine run %s could not be stored: %s", run.run_id, exc)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Stored cycles of this agent, newest first."""
        runs = self._store().list_records(record_type=REFINE_RUN_RECORD, limit=5000)
        runs.sort(key=lambda row: str(row.get("started_at") or ""), reverse=True)
        return runs[: max(1, int(limit))]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self._store().get(run_id)
        if record and record.get("record_type") == REFINE_RUN_RECORD:
            return record
        return None

    def pending_plans(self) -> List[Dict[str, Any]]:
        """Plans left for review by stored cycles, newest first, each with
        the ``run_id`` of the cycle that made it."""
        waiting = []
        for run in self.list_runs(limit=50):
            for item in run.get("pending") or []:
                waiting.append({**item, "run_id": run.get("run_id")})
        return waiting

    def resolve_pending(self, run_id: str, plan_id: str) -> bool:
        """Take a reviewed plan off its cycle's waiting list."""
        run = self.get_run(run_id)
        if not run:
            return False
        remaining = [
            item
            for item in run.get("pending") or []
            if str(item.get("plan_id")) != str(plan_id)
        ]
        if len(remaining) == len(run.get("pending") or []):
            return False
        run["pending"] = remaining
        run["waiting"] = len(remaining)
        self._store()._put(run, immutable=False)
        return True

    # ----------------------------------------------------------------- scopes

    def scopes(
        self, memory_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Tuple[str, Optional[str]]]:
        """The (memory, user) pairs a cycle covers.

        The one named; otherwise every pair this process has seen a turn
        for, then the pairs of the last stored cycle, then the agent's own
        memories with no user. At most ten, so a cycle stays bounded.
        """
        agent = self.agent
        if memory_id:
            return [(str(memory_id), user_id)]
        found: List[Tuple[str, Optional[str]]] = []

        def add(memory: Any, user: Any) -> None:
            pair = (str(memory), user if user is None else str(user))
            if memory and pair not in found:
                found.append(pair)

        try:
            with agent._upkeep_lock:
                seen = list(agent._consolidation_state)
        except Exception:
            seen = []
        for memory, user in seen:
            add(memory, user)
        try:
            previous = self.list_runs(limit=1)
        except Exception:
            previous = []
        for scope in (previous[0].get("scopes") if previous else None) or []:
            add(scope.get("memory_id"), scope.get("user_id"))
        for memory in list(getattr(agent, "memory_ids", None) or [])[-5:]:
            add(memory, None)
        return found[:10]

    def has_new_activity(self, since: Optional[str]) -> bool:
        """Whether any conversation of this agent has a message newer than
        ``since`` (an ISO time). Unknown counts as yes: better a cycle with
        nothing to do than one that never runs."""
        if not since:
            return True
        from .._time import to_epoch

        try:
            cutoff = float(to_epoch(since))
        except Exception:
            return True
        agent = self.agent
        reader = getattr(
            agent.memory_provider,
            "retrieve_conversation_history_ordered_by_timestamp",
            None,
        )
        if not callable(reader):
            return True
        memories = list(getattr(agent, "memory_ids", None) or [])
        if not memories:
            return True
        for memory in memories[-20:]:
            try:
                rows = reader(memory) or []
            except Exception:
                return True
            for row in reversed(list(rows)[-3:]):
                stamp = row.get("timestamp") or row.get("created_at")
                try:
                    if stamp and float(to_epoch(stamp)) > cutoff:
                        return True
                except Exception:
                    return True
        return False

    # -------------------------------------------------------------------- run

    def run(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        skip_if_idle: bool = False,
    ) -> Dict[str, Any]:
        """Run one cycle and return its stored report.

        ``dry_run`` plans every job and applies nothing: plans are listed in
        the report and not kept for review. ``skip_if_idle`` (the scheduler
        sets it) skips the cycle when no conversation has a message newer
        than the last cycle.
        """
        agent = self.agent
        run = RefineRun(
            run_id=f"refine-{uuid.uuid4().hex[:20]}",
            agent_id=str(agent.agent_id),
            trigger=str(trigger or "manual")[:40],
            started_at=utc_now_iso(),
            dry_run=bool(dry_run),
            jobs=[RefineJob(name) for name, _ in JOBS],
        )
        if skip_if_idle:
            previous = self.list_runs(limit=1)
            last = previous[0].get("started_at") if previous else None
            if last and not self.has_new_activity(last):
                run.status = "skipped"
                run.reason = "No new conversation since the last cycle."
                run.finished_at = utc_now_iso()
                for job in run.jobs:
                    job.status = "skipped"
                self._save(run)
                return run.to_dict()
        scopes = self.scopes(memory_id, user_id)
        run.scopes = [{"memory_id": memory, "user_id": user} for memory, user in scopes]
        self._save(run)
        jobs = {job.name: job for job in run.jobs}
        steps: Tuple[Tuple[str, Callable[..., None]], ...] = (
            ("verify", self._verify),
            # Entities and stated facts come from one consolidation plan per
            # scope: one read of the memory, reported as two jobs.
            ("entities", self._consolidate),
            ("propagation", self._propagation),
            ("toolbox", self._toolbox),
            ("skill_staleness", self._skill_staleness),
            ("promotion", self._promotion),
            ("skill_review", self._skill_review),
            ("persona_learning", self._persona_learning),
            ("compact", self._compact),
            ("retention", self._retention),
        )
        failed = False
        for name, step in steps:
            job = jobs[name]
            started = time.monotonic()
            meter = _TokenMeter(agent.background_model)
            try:
                with meter:
                    step(run, jobs, scopes)
                if job.status == "pending":
                    job.status = "done"
            except Exception as exc:
                failed = True
                job.status = "failed"
                job.notes.append(f"{type(exc).__name__}: {str(exc)[:300]}")
                logger.warning("Refine job %s failed: %s", name, exc)
            job.seconds = time.monotonic() - started
            job.tokens += meter.tokens
            job.tokens_estimated = job.tokens_estimated or meter.estimated
            if name == "entities" and jobs["stated_facts"].status == "pending":
                jobs["stated_facts"].status = job.status
            self._save(run)
        run.status = "dry_run" if dry_run else ("failed" if failed else "done")
        run.finished_at = utc_now_iso()
        self._save(run)
        return run.to_dict()

    # ------------------------------------------------------------------- jobs

    @staticmethod
    def _add(job: RefineJob, kind: str, count: int = 1) -> None:
        if count:
            job.changes[kind] = job.changes.get(kind, 0) + int(count)

    @staticmethod
    def _item(
        job: RefineJob, kind: str, text: Any, detail: Any = "", applied: bool = True
    ) -> None:
        """One line of what a job changed, or would change, for the report."""
        if len(job.items) < MAX_ITEMS:
            job.items.append(
                {
                    "kind": kind,
                    "text": " ".join(str(text or "").split())[:240],
                    "detail": " ".join(str(detail or "").split())[:300],
                    "applied": bool(applied),
                }
            )

    def _setting(self, name: str, default: Any) -> Any:
        """One of the cycle's own settings, kept with the consolidation
        settings (``refine_verify``, ``refine_tool_review`` and so on)."""
        return getattr(getattr(self.agent, "memory_consolidation", None), name, default)

    def _verify(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        job = jobs["verify"]
        if not self._setting("refine_verify", True):
            job.status = "skipped"
            job.notes.append("Verification is off (refine_verify).")
            return
        if not scopes:
            job.status = "skipped"
            return
        from .verification import NoteVerifier, describe_finding

        verifier = NoteVerifier(self.agent)
        budget = int(self._setting("refine_verify_model_checks", 3) or 0)
        for memory, user in scopes:
            try:
                found = verifier.check(
                    memory, user, store=not run.dry_run, model_checks=budget
                )
            except NotImplementedError as exc:
                job.status = "skipped"
                job.notes.append(str(exc)[:200])
                return
            budget = max(0, budget - found["model_checked"])
            # A dry run reports every note that fails; a real one, the flags
            # it opened, so a note flagged before is not counted again.
            flagged = found["findings"] if run.dry_run else found["opened"]
            self._add(job, "notes_checked", found["checked"])
            for finding in flagged:
                self._add(
                    job,
                    ("would_flag_" if run.dry_run else "flagged_") + finding["kind"],
                )
                self._item(job, finding["kind"], *describe_finding(finding), False)
            if found["left"]:
                job.notes.append(
                    f"{found['left']} captured fact(s) left for the next cycle."
                )
        # A count of notes read is not a change; keep it out of the total.
        checked = job.changes.pop("notes_checked", 0)
        if checked:
            job.notes.insert(0, f"{checked} note(s) checked against their source.")

    def _propagation(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        job = jobs["propagation"]
        if getattr(self.agent, "propagate_supersessions", True) is False:
            job.status = "skipped"
            job.notes.append("Propagation is off (propagate_supersessions).")
            return
        if not scopes:
            job.status = "skipped"
            return
        from .propagation import SupersessionPropagator

        propagator = SupersessionPropagator(self.agent)
        seen: set = set()
        flags: List[Dict[str, Any]] = []
        closed: List[Dict[str, Any]] = []
        for memory, user in scopes:
            flags += propagator.sweep(
                memory_id=memory, user_id=user, preview=run.dry_run
            )
            closed += propagator.close_fixed(
                by=AUTOMATIC_APPROVER, memory_id=memory, preview=run.dry_run
            )
            if not run.dry_run:
                # Flags the entity jobs raised as they replaced a value.
                flags += [
                    flag
                    for flag in propagator.list("open", memory_id=memory)
                    if str(flag.get("flagged_at") or "") >= run.started_at
                ]
        for flag in flags:
            if flag.get("flag_id") in seen:
                continue
            seen.add(flag.get("flag_id"))
            self._add(
                job,
                ("would_flag_" if run.dry_run else "flagged_")
                + str(flag.get("target_type") or "record"),
            )
            self._item(job, "stale_reference", *self._describe_flag(flag), False)
        for flag in closed:
            self._add(job, "would_close" if run.dry_run else "closed")
            self._item(
                job,
                "stale_reference_closed",
                f"Closed the flag on {flag.get('target_type')} "
                f"“{flag.get('target_name')}”",
                f"It no longer says {flag.get('old_value')}",
                applied=not run.dry_run,
            )

    @staticmethod
    def _describe_flag(flag: Mapping[str, Any]) -> Tuple[str, str]:
        return (
            f"{str(flag.get('target_type') or 'record').capitalize()} "
            f"“{flag.get('target_name')}” still says {flag.get('old_value')}",
            f"{flag.get('entity')}.{flag.get('attribute')} is now "
            f"{flag.get('new_value')}. Nothing was rewritten: “{flag.get('excerpt')}”",
        )

    def _toolbox(self, run: RefineRun, jobs: Dict[str, RefineJob], _: Any) -> None:
        agent, job = self.agent, jobs["toolbox"]
        mode = str(self._setting("refine_tool_review", "propose") or "off")
        review_tools = getattr(agent, "review_tools", None)
        if mode == "off":
            job.status = "skipped"
            job.notes.append("Tool review is off (refine_tool_review).")
            return
        if getattr(agent, "toolbox", None) is None or not callable(review_tools):
            job.status = "skipped"
            job.notes.append("This agent has no toolbox to review.")
            return
        review = review_tools(
            dry_run=run.dry_run,
            auto_apply=mode == "auto" and not run.dry_run,
            approved_by=AUTOMATIC_APPROVER,
        )
        for revision in review.revisions:
            applied = revision.status == "applied"
            parts = [
                f"{len(lines)} {label}"
                for lines, label in (
                    (revision.notes_added, "usage note(s)"),
                    (revision.avoid_added, "“not for” line(s)"),
                    (revision.queries_added, "example request(s)"),
                    (
                        [revision.call_example] if revision.call_example else [],
                        "example call",
                    ),
                )
                if lines
            ]
            self._add(job, "tool_revisions" if applied else "planned_tool_revisions")
            self._item(
                job,
                "tool_revision",
                f"{revision.tool}: " + ", ".join(parts),
                " | ".join(
                    [
                        *revision.notes_added,
                        *revision.avoid_added,
                        *revision.queries_added,
                    ]
                ),
                applied=applied,
            )
            if applied or run.dry_run:
                continue
            job.waiting += 1
            run.pending.append(
                {
                    "kind": "tool_revision",
                    "job": "toolbox",
                    "plan_id": revision.revision_id,
                    "summary": f"A revision of tool {revision.tool} to review",
                }
            )
        for flag in review.flags:
            self._add(job, "usage_flags")
            self._item(
                job,
                "usage_flag",
                f"{flag.get('tool')}: {str(flag.get('flag')).replace('_', ' ')}",
                f"Offered in {flag.get('offered_turns')} turn(s), called "
                f"{flag.get('calls')} time(s), over {flag.get('turns_observed')} turn(s)",
                applied=False,
            )
        reasons: Dict[str, int] = {}
        for reason in review.skipped.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            job.notes.append(
                "Not revised: "
                + ", ".join(
                    f"{count} {reason.replace('_', ' ')}"
                    for reason, count in sorted(reasons.items())
                )
            )

    def _compact(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        job = jobs["compact"]
        if not self._setting("refine_compact", True):
            job.status = "skipped"
            job.notes.append("Compaction is off (refine_compact).")
            return
        if not scopes:
            job.status = "skipped"
            return
        from .compaction import Compactor

        compactor = Compactor(self.agent)
        prefix = "would_" if run.dry_run else ""
        for memory, user in scopes:
            done = compactor.run(
                memory,
                user,
                since=run.started_at,
                store=not run.dry_run,
                drop_suppressed_vectors=bool(
                    self._setting("refine_drop_suppressed_vectors", False)
                ),
            )
            if done["skipped"]:
                job.status = "skipped"
                job.notes.append(done["skipped"])
                return
            self._add(job, prefix + "re_embed_entities", done["entities"])
            self._add(job, prefix + "re_embed_notes", done["notes"])
            self._add(job, prefix + "drop_vectors", done["dropped"])
            for kind, text, detail in done["items"]:
                self._item(job, kind, text, detail, applied=not run.dry_run)
            if done["left"]:
                job.notes.append(f"{done['left']} record(s) left for the next cycle.")

    def _consolidate(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        agent = self.agent
        entities, facts = jobs["entities"], jobs["stated_facts"]
        if not scopes:
            entities.status = facts.status = "skipped"
            entities.notes.append("No memory to consolidate yet.")
            return
        applies = agent.memory_consolidation.auto_apply and not run.dry_run
        for memory, user in scopes:
            with agent._upkeep_lock:
                state = agent._consolidation_state.get((memory, user)) or {}
                threads = sorted(state.get("threads") or ())
                since = state.get("since")
                if state and not run.dry_run:
                    state.update(turns=0, since=time.time(), started=time.time())
                    state["threads"] = set()
            plan = agent.plan_memory_consolidation(
                memory, user_id=user, thread_ids=threads or None, since=since
            )
            for warning in getattr(plan, "warnings", None) or []:
                entities.notes.append(str(warning)[:200])
            fact_ops = [op for op in plan.ops if op.kind in _FACT_KINDS]
            other_ops = [op for op in plan.ops if op.kind not in _FACT_KINDS]
            if not plan.ops:
                agent._drop_consolidation_plan(plan.plan_id)
                continue
            for op in plan.ops:
                text, detail = describe_op(op)
                self._item(
                    facts if op.kind in _FACT_KINDS else entities,
                    op.kind,
                    text,
                    detail,
                    # A value kept beside the host's is recorded, not applied.
                    applied=applies and (op.details or {}).get("beside") is None,
                )
            if applies:
                result = agent.apply_memory_consolidation(
                    plan, approved_by=AUTOMATIC_APPROVER
                )
                for kind, count in (result.applied or {}).items():
                    self._add(facts if kind in _FACT_KINDS else entities, kind, count)
                for error in result.errors or []:
                    entities.notes.append(str(error)[:200])
                continue
            for target, ops in ((entities, other_ops), (facts, fact_ops)):
                for op in ops:
                    self._add(target, f"planned_{op.kind}")
                target.waiting += 1 if ops and not run.dry_run else 0
            if run.dry_run:
                # A dry run shows what would change and keeps nothing to review.
                agent._drop_consolidation_plan(plan.plan_id)
                continue
            run.pending.append(
                {
                    "kind": "consolidation",
                    "job": "entities" if other_ops else "stated_facts",
                    "plan_id": plan.plan_id,
                    "memory_id": memory,
                    "user_id": user,
                    "summary": f"{len(plan.ops)} change(s) to review",
                    "counts": dict(plan.counts),
                    "plan": plan.to_dict(),
                }
            )

    def _manager(self) -> Any:
        return getattr(self.agent, "continual_learning_manager", None)

    def _skill_staleness(
        self, run: RefineRun, jobs: Dict[str, RefineJob], _: Any
    ) -> None:
        job, manager = jobs["skill_staleness"], self._manager()
        if manager is None:
            job.status = "skipped"
            job.notes.append("Continual learning is off for this agent.")
            return
        if run.dry_run:
            stale = [
                skill.name
                for skill in manager.skillbox.list_skills()
                if skill.status.value == "active"
                and any(not manager._resolve_tool(name) for name in skill.tools_used)
            ]
            self._add(job, "planned_deprecations", len(stale))
            for name in stale:
                self._item(
                    job,
                    "deprecate_skill",
                    f"Would deprecate skill {name}",
                    "A tool it uses no longer resolves",
                    applied=False,
                )
            return
        deprecated = manager.monitor.staleness_sweep(manager._resolve_tool)
        self._add(job, "deprecated", len(deprecated))
        for skill_id in deprecated:
            skill = manager.skillbox.get_skill_by_id(str(skill_id))
            self._item(
                job,
                "deprecate_skill",
                f"Deprecated skill {getattr(skill, 'name', skill_id)}",
                "A tool it uses no longer resolves",
            )

    def _promotion(self, run: RefineRun, jobs: Dict[str, RefineJob], _: Any) -> None:
        job, manager = jobs["promotion"], self._manager()
        if manager is None:
            job.status = "skipped"
            job.notes.append("Continual learning is off for this agent.")
            return
        if run.dry_run:
            from ..long_term.procedural.workflow.canonicalization import (
                aggregate_trajectory_stats,
            )

            engine = manager.engine
            by_hash = engine._skills_by_hash()
            eligible = [
                stat
                for stat in aggregate_trajectory_stats(
                    self.agent.memory_provider, agent_id=self.agent.agent_id
                )
                if engine._eligibility(stat, by_hash)[0]
            ]
            self._add(job, "eligible_classes", len(eligible))
            for stat in eligible:
                runs = getattr(stat, "runs", None) or []
                self._item(
                    job,
                    "promote",
                    "Would distil a skill from "
                    f"“{getattr(runs[0], 'user_query', '') if runs else stat.canonical_hash[:12]}”",
                    f"{stat.executions} runs, {stat.success_rate:.0%} success",
                    applied=False,
                )
            return
        # The engine's cycle, without the staleness sweep job 3 just ran.
        report = manager.engine.run_promotion_cycle()
        manager._record_promoted_skills(report)
        manager.last_report = report
        self._add(job, "promoted", len(report.promoted))
        self._add(job, "rejected", len(report.rejected))
        for skill_id in report.promoted:
            skill = manager.skillbox.get_skill_by_id(skill_id)
            self._item(
                job,
                "promote",
                f"New skill {getattr(skill, 'name', skill_id)}",
                getattr(skill, "description", "") or "",
            )
        for _hash, reasons in report.rejected[:5]:
            job.notes.append("; ".join(str(reason) for reason in reasons)[:240])
            self._item(
                job,
                "reject",
                f"Rejected a candidate from class {str(_hash)[:12]}",
                "; ".join(str(reason) for reason in reasons),
                applied=False,
            )
        if manager.config.require_shadow and report.promoted:
            job.waiting = len(report.promoted)
            for skill_id in report.promoted:
                skill = manager.skillbox.get_skill_by_id(skill_id)
                run.pending.append(
                    {
                        "kind": "skill_activation",
                        "job": "promotion",
                        "plan_id": skill_id,
                        "summary": f"Skill {getattr(skill, 'name', skill_id)} is "
                        "in shadow, waiting for activation",
                    }
                )

    def _skill_review(self, run: RefineRun, jobs: Dict[str, RefineJob], _: Any) -> None:
        job, manager = jobs["skill_review"], self._manager()
        if manager is None:
            job.status = "skipped"
            job.notes.append("Continual learning is off for this agent.")
            return
        flags = manager.engine.skill_review_flags()
        self._add(job, "review_flags", len(flags))
        job.notes.extend(str(flag)[:300] for flag in flags[:10])
        for flag in flags:
            self._item(job, "review_flag", "Flagged for review", flag, applied=False)
        # A skill that still states a value entity memory has replaced is a
        # skill to review too. The propagation job found them; they are
        # listed here beside the other reasons to open a skill.
        if getattr(self.agent, "propagate_supersessions", True) is False:
            return
        try:
            from .propagation import SupersessionPropagator

            stale = SupersessionPropagator(self.agent).list("open", target_type="skill")
        except Exception as exc:
            logger.debug("Stale skill references could not be read: %s", exc)
            return
        self._add(job, "stale_references", len(stale))
        for flag in stale:
            self._item(job, "stale_reference", *self._describe_flag(flag), False)

    def _persona_learning(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        agent, job = self.agent, jobs["persona_learning"]
        manager = getattr(agent, "persona_manager", None)
        config = agent.persona_learning
        # A desk that shares one assistant can be read as a team
        # (across_users); otherwise several users are not one person.
        one_of_many = agent.multi_tenant and not getattr(config, "across_users", False)
        if (
            not config.enabled
            or one_of_many
            or manager is None
            or manager.configuration_persona is None
        ):
            job.status = "skipped"
            job.notes.append(
                "This agent serves several users, so it does not learn from one."
                if one_of_many
                else "Persona learning is off, or the agent has no persona."
            )
            return
        if not scopes:
            job.status = "skipped"
            return
        memory, user = scopes[0]
        # A dry run stores nothing: no review mark, no waiting proposal.
        review = (
            agent.review_persona(memory, user_id=user, dry_run=True)
            if run.dry_run
            else agent.review_persona(memory, user_id=user)
        )
        # Standing rules the same review proposed. A rule is never applied
        # by the cycle, whatever auto_apply says: it waits for its owner.
        for rule in getattr(review, "rule_proposals", None) or ():
            self._add(job, "planned_standing_rules")
            people = ", ".join(str(user) for user in getattr(rule, "users", ()) or ())
            self._item(
                job,
                "standing_rule",
                f"Standing rule: “{rule.rule}”",
                "From corrections"
                + (f" by {people}" if people else "")
                + ": "
                + " | ".join(
                    str(item.get("quote") or item.get("text") or "")[:120]
                    for item in list(rule.evidence or ())[:3]
                    if isinstance(item, dict)
                ),
                applied=False,
            )
            if run.dry_run:
                continue
            job.waiting += 1
            run.pending.append(
                {
                    "kind": "persona_rule",
                    "job": "persona_learning",
                    "plan_id": rule.proposal_id,
                    "memory_id": memory,
                    "user_id": user,
                    "summary": f"A standing rule for its owner to review: “{rule.rule}”",
                }
            )
        if review.proposal is None:
            # Why nothing was proposed; a proposal speaks for itself below.
            job.notes.append(f"{review.outcome}: {review.code}")
        if review.proposal is None:
            return
        proposal = review.proposal
        described = (
            f"Persona {proposal.field}: add “{proposal.addition}”",
            "Seen in: "
            + " | ".join(
                str(item.get("quote") or item.get("text") or item)[:120]
                if isinstance(item, dict)
                else str(item)[:120]
                for item in list(proposal.evidence or ())[:3]
            ),
        )
        if run.dry_run:
            self._add(job, "planned_persona_additions")
            self._item(job, "persona", *described, applied=False)
            return
        if config.auto_apply and not run.dry_run:
            agent.apply_persona_proposal(AUTOMATIC_APPROVER, review.proposal)
            self._add(job, "persona_additions")
            self._item(job, "persona", *described)
            return
        self._add(job, "planned_persona_additions")
        self._item(job, "persona", *described, applied=False)
        job.waiting += 1
        run.pending.append(
            {
                "kind": "persona",
                "job": "persona_learning",
                "plan_id": f"persona-{run.run_id}",
                "memory_id": memory,
                "user_id": user,
                "summary": "A persona change to review",
            }
        )

    def _retention(
        self, run: RefineRun, jobs: Dict[str, RefineJob], scopes: List[Tuple[str, Any]]
    ) -> None:
        agent, job = self.agent, jobs["retention"]
        if not scopes:
            job.status = "skipped"
            return
        for memory, user in scopes:
            report = agent.plan_memory_retention(memory, user_id=user)
            data = (
                report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
            )
            count = int(
                data.get("candidate_count") or len(data.get("candidates") or [])
            )
            if not count:
                continue
            # Retention never applies itself: a named person approves a plan.
            self._add(job, "planned_suppressions", count)
            self._item(
                job,
                "retention",
                f"{count} memory record(s) could be suppressed",
                "Waits for a named person to approve the retention plan",
                applied=False,
            )
            job.waiting += 1
            if not run.dry_run:
                run.pending.append(
                    {
                        "kind": "retention",
                        "job": "retention",
                        "plan_id": str(data.get("plan_id") or f"retention-{memory}"),
                        "memory_id": memory,
                        "user_id": user,
                        "summary": f"{count} memory record(s) could be suppressed",
                    }
                )


__all__ = [
    "AUTOMATIC_APPROVER",
    "REFINE_RUN_RECORD",
    "RefineCycle",
    "RefineJob",
    "RefineRun",
    "JOBS",
    "JOB_LABELS",
]
