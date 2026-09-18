"""Local repair journal containing IDs and hashes, not memory text or tokens."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class NotionState:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.touch(mode=0o600, exist_ok=False)
        except FileExistsError:
            pass
        self._lock = threading.RLock()
        with self.connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS records (
                source_id TEXT NOT NULL, memory_type TEXT NOT NULL,
                page_id TEXT, fingerprint TEXT, status TEXT NOT NULL,
                PRIMARY KEY (source_id, memory_type))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY)"""
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def writer(self):
        # SQLite serializes writers across provider instances/processes sharing
        # this journal. Distributed writers with separate journals are not CAS.
        with self._lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def get(self, source_id, memory_type, connection=None):
        if connection is None:
            with self.connect() as conn:
                return self.get(source_id, memory_type, conn)
        row = connection.execute(
            "SELECT * FROM records WHERE source_id=? AND memory_type=?",
            (source_id, memory_type),
        ).fetchone()
        return dict(row) if row else None

    def put(
        self,
        source_id,
        memory_type,
        *,
        page_id=None,
        fingerprint=None,
        status="pending",
        connection=None,
    ):
        if connection is None:
            with self.connect() as conn:
                return self.put(
                    source_id,
                    memory_type,
                    page_id=page_id,
                    fingerprint=fingerprint,
                    status=status,
                    connection=conn,
                )
        connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_id, memory_type) "
            "DO UPDATE SET page_id=excluded.page_id, fingerprint=excluded.fingerprint, status=excluded.status",
            (source_id, memory_type, page_id, fingerprint, status),
        )

    def rows(self, *, pending_only=False):
        with self.connect() as connection:
            sql = "SELECT * FROM records"
            if pending_only:
                sql += " WHERE status NOT IN ('indexed', 'stored', 'deleted', 'not_created', 'abandoned')"
            return [dict(row) for row in connection.execute(sql)]

    def bind_source(self, data_source_id):
        with self.writer() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key='data_source_id'"
            ).fetchone()
            if row and json.loads(row[0]) != data_source_id:
                raise ValueError(
                    "A Notion journal cannot be shared between different data sources"
                )
            connection.execute(
                "INSERT OR IGNORE INTO settings VALUES ('data_source_id', ?)",
                (json.dumps(data_source_id),),
            )

    def activate_index(self, configuration):
        """Atomically record model/backend/type changes and require a rebuild."""
        encoded = json.dumps(configuration, sort_keys=True)
        with self.writer() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key='index_configuration'"
            ).fetchone()
            if not row or row[0] != encoded:
                connection.execute(
                    "UPDATE records SET status='pending' WHERE status IN ('indexed', 'stored')"
                )
            connection.execute(
                "INSERT INTO settings VALUES ('index_configuration', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (encoded,),
            )

    def has_pending(self, memory_type):
        with self.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM records WHERE memory_type=? AND status NOT IN ('indexed', 'stored', 'deleted', 'not_created', 'abandoned') LIMIT 1",
                    (memory_type,),
                ).fetchone()
                is not None
            )

    def webhook_seen(self, event_id):
        with self.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM webhook_events WHERE event_id=?", (event_id,)
                ).fetchone()
                is not None
            )

    def complete_webhook(self, event_id):
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO webhook_events VALUES (?)", (event_id,)
            )
            connection.execute(
                "DELETE FROM webhook_events WHERE rowid NOT IN (SELECT rowid FROM webhook_events ORDER BY rowid DESC LIMIT 10000)"
            )

    def setting(self, key, value=None):
        with self.connect() as connection:
            if value is not None:
                connection.execute(
                    "INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)),
                )
                return value
            row = connection.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None
