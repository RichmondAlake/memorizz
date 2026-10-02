"""Durable single-node run and event storage for the MemoRizz meta-harness."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .._env_io import ensure_home, memorizz_home
from .models import (
    HarnessEvent,
    HarnessOrchestration,
    HarnessRun,
    HarnessStatus,
    utcnow_iso,
)


class HarnessRunStore(Protocol):
    def create(self, run: HarnessRun) -> HarnessRun:
        ...

    def get(self, run_id: str) -> Optional[HarnessRun]:
        ...

    def list(
        self, *, limit: int = 100, status: Optional[str] = None
    ) -> List[HarnessRun]:
        ...

    def update(self, run_id: str, **changes: Any) -> HarnessRun:
        ...

    def append_event(self, event: HarnessEvent) -> HarnessEvent:
        ...

    def replace(self, run: HarnessRun, events: List[HarnessEvent]) -> HarnessRun:
        ...

    def events(
        self, run_id: str, *, after: int = 0, limit: int = 1000
    ) -> List[HarnessEvent]:
        ...

    def acquire_workspace(self, workspace: str, run_id: str) -> bool:
        ...

    def release_workspace(self, workspace: str, run_id: str) -> None:
        ...

    def recover_interrupted(self) -> int:
        ...

    # Staged plans and comparisons. Optional: MetaHarness.start_plan and
    # start_compare need them; single runs do not.
    def create_orchestration(
        self, orchestration: HarnessOrchestration
    ) -> HarnessOrchestration:
        ...

    def get_orchestration(
        self, orchestration_id: str
    ) -> Optional[HarnessOrchestration]:
        ...

    def list_orchestrations(
        self, *, limit: int = 50, status: Optional[str] = None
    ) -> List[HarnessOrchestration]:
        ...

    def update_orchestration(
        self, orchestration_id: str, **changes: Any
    ) -> HarnessOrchestration:
        ...

    # Deleting finished runs and workflows. Optional: MetaHarness.delete_runs
    # and delete_orchestration need them.
    def delete(self, run_ids: List[str]) -> int:
        ...

    def delete_orchestration(self, orchestration_id: str) -> bool:
        ...

    def close(self) -> None:
        ...


class SQLiteHarnessRunStore:
    """Transactional SQLite state for one local/self-hosted worker."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            ensure_home()
            path = memorizz_home() / "harness-runs.sqlite3"
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS harness_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                harness TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS harness_runs_status_updated
                ON harness_runs(status, updated_at DESC);
            CREATE TABLE IF NOT EXISTS harness_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence),
                FOREIGN KEY (run_id) REFERENCES harness_runs(run_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS harness_workspace_leases (
                workspace TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES harness_runs(run_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS harness_orchestrations (
                orchestration_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS harness_orchestrations_updated
                ON harness_orchestrations(updated_at DESC);
            """
        )

    @staticmethod
    def _decode(row: sqlite3.Row | Dict[str, Any]) -> HarnessRun:
        payload = json.loads(row["payload"])
        return HarnessRun(**payload)

    def create(self, run: HarnessRun) -> HarnessRun:
        value = run.to_dict()
        with self._lock:
            self._connection.execute(
                "INSERT INTO harness_runs(run_id,status,harness,created_at,updated_at,payload) "
                "VALUES(?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.status.value,
                    run.harness,
                    run.created_at,
                    run.updated_at,
                    json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
        return run

    def get(self, run_id: str) -> Optional[HarnessRun]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM harness_runs WHERE run_id=?", (str(run_id),)
            ).fetchone()
        return self._decode(row) if row else None

    def list(
        self, *, limit: int = 100, status: Optional[str] = None
    ) -> List[HarnessRun]:
        bounded = max(1, min(int(limit), 10_000))
        with self._lock:
            if status:
                rows = self._connection.execute(
                    "SELECT payload FROM harness_runs WHERE status=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (str(status), bounded),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT payload FROM harness_runs ORDER BY updated_at DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
        return [self._decode(row) for row in rows]

    def update(self, run_id: str, **changes: Any) -> HarnessRun:
        with self._lock:
            # UI, CLI, MCP, and the worker can update the same row. Lock the
            # read-modify-write sequence across processes so a heartbeat can
            # never erase a concurrent durable cancellation request.
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM harness_runs WHERE run_id=?", (str(run_id),)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown harness run: {run_id}")
                payload = self._decode(row).to_dict()
                payload.update(changes)
                payload["updated_at"] = utcnow_iso()
                updated = HarnessRun(**payload)
                self._connection.execute(
                    "UPDATE harness_runs SET status=?,harness=?,updated_at=?,payload=? "
                    "WHERE run_id=?",
                    (
                        updated.status.value,
                        updated.harness,
                        updated.updated_at,
                        json.dumps(
                            updated.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                        updated.run_id,
                    ),
                )
                self._connection.execute("COMMIT")
                return updated
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def append_event(self, event: HarnessEvent) -> HarnessEvent:
        with self._lock:
            # A process-local mutex is not sufficient: UI, CLI, MCP, and worker
            # processes can append to the same run. BEGIN IMMEDIATE makes
            # sequence allocation and insertion one cross-process operation.
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if event.sequence is None:
                    row = self._connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) AS value FROM harness_events "
                        "WHERE run_id=?",
                        (event.run_id,),
                    ).fetchone()
                    event.sequence = int(row["value"] or 0) + 1
                self._connection.execute(
                    "INSERT INTO harness_events(run_id,sequence,timestamp,event_type,payload) "
                    "VALUES(?,?,?,?,?)",
                    (
                        event.run_id,
                        event.sequence,
                        event.timestamp,
                        event.type.value,
                        json.dumps(event.to_dict(), ensure_ascii=False, default=str),
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return event

    def replace(self, run: HarnessRun, events: List[HarnessEvent]) -> HarnessRun:
        """Write a run and all its events, replacing any earlier copy of both.

        For runs recorded from elsewhere, such as a coding agent's own session
        log, which are rebuilt whole each time they grow. One transaction, so
        a reader never sees the run with half its events.
        """
        value = json.dumps(
            run.to_dict(), ensure_ascii=False, sort_keys=True, default=str
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "INSERT INTO harness_runs(run_id,status,harness,created_at,updated_at,payload) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
                    "status=excluded.status,harness=excluded.harness,"
                    "updated_at=excluded.updated_at,payload=excluded.payload",
                    (
                        run.run_id,
                        run.status.value,
                        run.harness,
                        run.created_at,
                        run.updated_at,
                        value,
                    ),
                )
                self._connection.execute(
                    "DELETE FROM harness_events WHERE run_id=?", (run.run_id,)
                )
                for sequence, event in enumerate(events, start=1):
                    event.sequence = sequence
                    self._connection.execute(
                        "INSERT INTO harness_events(run_id,sequence,timestamp,event_type,payload) "
                        "VALUES(?,?,?,?,?)",
                        (
                            run.run_id,
                            sequence,
                            event.timestamp,
                            event.type.value,
                            json.dumps(
                                event.to_dict(), ensure_ascii=False, default=str
                            ),
                        ),
                    )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return run

    def events(
        self, run_id: str, *, after: int = 0, limit: int = 1000
    ) -> List[HarnessEvent]:
        bounded = max(1, min(int(limit), 10_000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM harness_events WHERE run_id=? AND sequence>? "
                "ORDER BY sequence ASC LIMIT ?",
                (str(run_id), max(0, int(after)), bounded),
            ).fetchall()
        events: List[HarnessEvent] = []
        for row in rows:
            events.append(HarnessEvent(**json.loads(row["payload"])))
        return events

    def event_counts(self, run_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """Actions per run by event type, counting each harness item once
        (some harnesses report an item when it starts and when it ends)."""
        ids = [str(run_id) for run_id in run_ids if run_id][:1000]
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, event_type, COUNT(DISTINCT COALESCE("
                "json_extract(payload, '$.data.id'), sequence)) AS n "
                f"FROM harness_events WHERE run_id IN ({marks}) "
                "GROUP BY run_id, event_type",
                ids,
            ).fetchall()
        counts: Dict[str, Dict[str, int]] = {}
        for row in rows:
            counts.setdefault(row["run_id"], {})[row["event_type"]] = int(row["n"])
        return counts

    def event_models(self, run_ids: List[str]) -> Dict[str, str]:
        """The model each run's harness reported in its events, latest first
        (Claude Code names it at start-up, Codex with its usage)."""
        ids = [str(run_id) for run_id in run_ids if run_id][:1000]
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, json_extract(payload, '$.data.model') AS model "
                f"FROM harness_events WHERE run_id IN ({marks}) "
                "AND event_type IN ('status', 'usage') "
                "AND json_type(payload, '$.data.model') = 'text' "
                "ORDER BY run_id, sequence DESC",
                ids,
            ).fetchall()
        models: Dict[str, str] = {}
        for row in rows:
            if row["model"] and row["run_id"] not in models:
                models[row["run_id"]] = str(row["model"])
        return models

    def acquire_workspace(self, workspace: str, run_id: str) -> bool:
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO harness_workspace_leases(workspace,run_id,acquired_at) "
                    "VALUES(?,?,?)",
                    (str(workspace), str(run_id), utcnow_iso()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def workspace_holder(self, workspace: str) -> Optional[str]:
        """The run holding a workspace's write lease, if any."""
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id FROM harness_workspace_leases WHERE workspace=?",
                (str(workspace),),
            ).fetchone()
        return str(row["run_id"]) if row else None

    def release_workspace(self, workspace: str, run_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM harness_workspace_leases WHERE workspace=? AND run_id=?",
                (str(workspace), str(run_id)),
            )

    def recover_interrupted(self) -> int:
        recovered = 0
        for run in self.list(limit=10_000):
            if run.status in {HarnessStatus.RUNNING, HarnessStatus.QUEUED}:
                self.update(
                    run.run_id,
                    status=HarnessStatus.INTERRUPTED.value,
                    finished_at=utcnow_iso(),
                    result={
                        "ok": False,
                        "status": HarnessStatus.INTERRUPTED.value,
                        "error_code": "host_restarted",
                        "error": "The local harness host restarted during this run.",
                    },
                )
                recovered += 1
        with self._lock:
            self._connection.execute("DELETE FROM harness_workspace_leases")
        # A plan or comparison is driven by a thread in the host process, so a
        # restart stops it even when its current run was only awaiting approval.
        for orchestration in self.list_orchestrations(limit=10_000):
            if not orchestration.status.terminal:
                self.update_orchestration(
                    orchestration.orchestration_id,
                    status=HarnessStatus.INTERRUPTED.value,
                    finished_at=utcnow_iso(),
                    error_code="host_restarted",
                    error=(
                        "The harness host restarted during this workflow; "
                        "later steps did not start."
                    ),
                )
        return recovered

    @staticmethod
    def _decode_orchestration(row: sqlite3.Row) -> HarnessOrchestration:
        return HarnessOrchestration(**json.loads(row["payload"]))

    @staticmethod
    def _encode_orchestration(orchestration: HarnessOrchestration) -> str:
        return json.dumps(
            orchestration.to_dict(), ensure_ascii=False, sort_keys=True, default=str
        )

    def create_orchestration(
        self, orchestration: HarnessOrchestration
    ) -> HarnessOrchestration:
        with self._lock:
            self._connection.execute(
                "INSERT INTO harness_orchestrations"
                "(orchestration_id,kind,status,created_at,updated_at,payload) "
                "VALUES(?,?,?,?,?,?)",
                (
                    orchestration.orchestration_id,
                    orchestration.kind,
                    orchestration.status.value,
                    orchestration.created_at,
                    orchestration.updated_at,
                    self._encode_orchestration(orchestration),
                ),
            )
        return orchestration

    def get_orchestration(
        self, orchestration_id: str
    ) -> Optional[HarnessOrchestration]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM harness_orchestrations WHERE orchestration_id=?",
                (str(orchestration_id),),
            ).fetchone()
        return self._decode_orchestration(row) if row else None

    def list_orchestrations(
        self, *, limit: int = 50, status: Optional[str] = None
    ) -> List[HarnessOrchestration]:
        bounded = max(1, min(int(limit), 10_000))
        with self._lock:
            if status:
                rows = self._connection.execute(
                    "SELECT payload FROM harness_orchestrations WHERE status=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (str(status), bounded),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT payload FROM harness_orchestrations "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
        return [self._decode_orchestration(row) for row in rows]

    def update_orchestration(
        self, orchestration_id: str, **changes: Any
    ) -> HarnessOrchestration:
        with self._lock:
            # The driver thread and a cancel request from another process can
            # both write; each passes only the fields it owns.
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM harness_orchestrations "
                    "WHERE orchestration_id=?",
                    (str(orchestration_id),),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown harness workflow: {orchestration_id}")
                payload = self._decode_orchestration(row).to_dict()
                payload.update(changes)
                payload["updated_at"] = utcnow_iso()
                updated = HarnessOrchestration(**payload)
                self._connection.execute(
                    "UPDATE harness_orchestrations SET status=?,updated_at=?,payload=? "
                    "WHERE orchestration_id=?",
                    (
                        updated.status.value,
                        updated.updated_at,
                        self._encode_orchestration(updated),
                        updated.orchestration_id,
                    ),
                )
                self._connection.execute("COMMIT")
                return updated
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def delete(self, run_ids: List[str]) -> int:
        """Delete runs with their events and workspace leases; returns how
        many runs were removed."""
        ids = list(dict.fromkeys(str(run_id) for run_id in run_ids if run_id))
        removed = 0
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for start in range(0, len(ids), 500):
                    chunk = ids[start : start + 500]
                    marks = ",".join("?" * len(chunk))
                    # Explicit, so stores created without foreign keys on
                    # lose them too.
                    self._connection.execute(
                        f"DELETE FROM harness_events WHERE run_id IN ({marks})", chunk
                    )
                    self._connection.execute(
                        f"DELETE FROM harness_workspace_leases WHERE run_id IN ({marks})",
                        chunk,
                    )
                    removed += self._connection.execute(
                        f"DELETE FROM harness_runs WHERE run_id IN ({marks})", chunk
                    ).rowcount
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return removed

    def delete_orchestration(self, orchestration_id: str) -> bool:
        """Delete a workflow record. Its runs are deleted separately."""
        with self._lock:
            return bool(
                self._connection.execute(
                    "DELETE FROM harness_orchestrations WHERE orchestration_id=?",
                    (str(orchestration_id),),
                ).rowcount
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["HarnessRunStore", "SQLiteHarnessRunStore"]
