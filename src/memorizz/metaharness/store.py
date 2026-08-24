"""Durable single-node run and event storage for the MemoRizz meta-harness."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .._env_io import ensure_home, memorizz_home
from .models import HarnessEvent, HarnessRun, HarnessStatus, utcnow_iso


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
        return recovered

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["HarnessRunStore", "SQLiteHarnessRunStore"]
