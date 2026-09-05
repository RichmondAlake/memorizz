"""SQLite/Oracle implementations of the same private observability index.

Oracle uses an explicit additive migration; all query values are bind variables.
SQLite lives beside, never inside, the semantic memory stores.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .index import SpanIndex, digest, resource_hash, validate_filters

FIELDS = (
    "application_id",
    "agent_id",
    "memory_id",
    "thread_id",
    "root_trace_id",
    "run_id",
    "turn_id",
    "kind",
    "status",
    "tool_name",
    "job_ref",
    "error_code",
)
TABLES = ("obs_spans", "obs_bundles", "obs_resources", "obs_previews", "obs_state")


class SQLSpanIndex(SpanIndex):
    oracle = False

    def table(self, name):
        if name not in TABLES:
            raise ValueError("Unknown observability table")
        return name

    def ready(self):
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT value FROM {self.table('obs_state')} WHERE name = :name",
                    {"name": "schema_version"},
                )
                row = cursor.fetchone()
                return bool(row and str(row[0]) == "1")
        except Exception:
            return False

    def _require_ready(self):
        if not self.ready():
            raise RuntimeError(
                "Observability index is not initialized; run the explicit migration"
            )

    def _insert(self, cursor, table, values):
        columns = ", ".join(values)
        binds = ", ".join(":" + key for key in values)
        if self.oracle:
            import oracledb

            clobs = {
                key: oracledb.DB_TYPE_CLOB
                for key in ("payload", "preview")
                if key in values
            }
            if clobs:
                cursor.setinputsizes(**clobs)
        cursor.execute(
            f"INSERT INTO {self.table(table)} ({columns}) VALUES ({binds})", values
        )

    def _lock_writer(self, cursor):
        if self.oracle:
            cursor.execute(
                f"SELECT value FROM {self.table('obs_state')} WHERE name = 'schema_version' FOR UPDATE"
            )
            cursor.fetchone()
        else:
            cursor.execute("BEGIN IMMEDIATE")

    def put(self, summary, records):
        self._require_ready()
        pending = "pending:" + summary["bundle_key"]
        with self.connection() as conn:
            cursor = conn.cursor()
            self._lock_writer(cursor)
            cursor.execute(
                f"DELETE FROM {self.table('obs_state')} WHERE name = :name",
                {"name": pending},
            )
            self._insert(cursor, "obs_state", {"name": pending, "value": "pending"})
            conn.commit()
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                self._lock_writer(cursor)
                cursor.execute(
                    f"SELECT fingerprint FROM {self.table('obs_bundles')} WHERE bundle_key = :key",
                    {"key": summary["bundle_key"]},
                )
                existing = cursor.fetchone()
                if existing:
                    if existing[0] != summary["fingerprint"]:
                        raise ValueError(
                            "Immutable bundle changed; record a new event or turn ID"
                        )
                    cursor.execute(
                        f"DELETE FROM {self.table('obs_state')} WHERE name = :name",
                        {"name": pending},
                    )
                    conn.commit()
                    return
                for row in records:
                    for table in ("obs_resources", "obs_previews", "obs_spans"):
                        cursor.execute(
                            f"DELETE FROM {self.table(table)} WHERE event_key = :key",
                            {"key": row["key"]},
                        )
                    metadata = row["metadata"]
                    values = {field: metadata.get(field) for field in FIELDS}
                    values["tool_name"] = metadata.get(
                        "logical_tool_name"
                    ) or metadata.get("tool_name")
                    values.update(
                        event_key=row["key"],
                        bundle_key=row["bundle_key"],
                        user_key=row["user_key"],
                        timestamp=row["timestamp"],
                        payload=json.dumps(metadata),
                        verified=int(metadata.get("verified") is True),
                    )
                    self._insert(cursor, "obs_spans", values)
                    for ref in row["resource_hashes"]:
                        self._insert(
                            cursor,
                            "obs_resources",
                            {"event_key": row["key"], "ref_hash": ref},
                        )
                    if row["preview"]:
                        self._insert(
                            cursor,
                            "obs_previews",
                            {
                                "event_key": row["key"],
                                "timestamp": row["timestamp"],
                                "preview": row["preview"],
                            },
                        )
                cursor.execute(
                    f"DELETE FROM {self.table('obs_bundles')} WHERE bundle_key = :key",
                    {"key": summary["bundle_key"]},
                )
                self._insert(
                    cursor,
                    "obs_bundles",
                    {
                        "bundle_key": summary["bundle_key"],
                        "timestamp": summary["timestamp"],
                        "source_record_id": summary["source_record_id"],
                        "fingerprint": summary["fingerprint"],
                        "payload": json.dumps(summary),
                    },
                )
                cursor.execute(
                    f"DELETE FROM {self.table('obs_state')} WHERE name = :name",
                    {"name": pending},
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def pending_count(self):
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table('obs_state')} WHERE value = :value",
                {"value": "pending"},
            )
            return cursor.fetchone()[0]

    def _predicate(self, filters, boundary=None):
        clauses, binds = [], {}

        def bind(value):
            name = "p" + str(len(binds))
            binds[name] = value
            return ":" + name

        def membership(column, values):
            return f"s.{column} IN ({', '.join(bind(value) for value in values)})"

        identity = []
        for key, column in (("agent_ids", "agent_id"), ("memory_ids", "memory_id")):
            if filters.get(key):
                identity.append(membership(column, filters[key]))
        if identity:
            clauses.append("(" + " OR ".join(identity) + ")")
        for key in (
            "application_id",
            "thread_id",
            "root_trace_id",
            "run_id",
            "turn_id",
            "tool_name",
        ):
            if filters.get(key) is not None:
                clauses.append(f"s.{key} = {bind(filters[key])}")
        if "user_id" in filters:
            clauses.append(f"s.user_key = {bind(digest(filters['user_id']))}")
        for key, column in (("event_kinds", "kind"), ("statuses", "status")):
            if filters.get(key):
                clauses.append(membership(column, filters[key]))
        if filters.get("success") is not None:
            clauses.append(
                membership(
                    "status", ["success"] if filters["success"] else ["error", "failed"]
                )
            )
        for key, operator in (("start_time", ">="), ("end_time", "<=")):
            if filters.get(key):
                clauses.append(f"s.timestamp {operator} {bind(filters[key])}")
        if filters.get("resource_refs"):
            refs = ", ".join(
                bind(resource_hash(value)) for value in filters["resource_refs"]
            )
            clauses.append(
                f"EXISTS (SELECT 1 FROM {self.table('obs_resources')} r WHERE r.event_key = s.event_key AND r.ref_hash IN ({refs}))"
            )
        if filters.get("query"):
            query = filters["query"]
            matches = [
                f"s.{column} = {bind(query)}"
                for column in (
                    "agent_id",
                    "thread_id",
                    "root_trace_id",
                    "run_id",
                    "turn_id",
                    "job_ref",
                    "error_code",
                )
            ]
            matches.append(f"s.user_key = {bind(digest(query))}")
            matches.append(
                f"EXISTS (SELECT 1 FROM {self.table('obs_resources')} r WHERE r.event_key = s.event_key AND r.ref_hash = {bind(resource_hash(query))})"
            )
            clauses.append("(" + " OR ".join(matches) + ")")
        if boundary:
            time_bind, key_bind = bind(boundary[0]), bind(boundary[1])
            clauses.append(
                f"(s.timestamp < {time_bind} OR (s.timestamp = {time_bind} AND s.event_key < {key_bind}))"
            )
        return " AND ".join(clauses) or "1=1", binds

    @staticmethod
    def _json(value):
        if hasattr(value, "read"):
            value = value.read()
        return json.loads(value)

    def _limit(self):
        return "FETCH FIRST :page_size ROWS ONLY" if self.oracle else "LIMIT :page_size"

    def select(self, filters, *, boundary=None, limit=251):
        self._require_ready()
        predicate, binds = self._predicate(filters, boundary)
        binds["page_size"] = limit
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT s.event_key, s.bundle_key, s.timestamp, s.payload FROM {self.table('obs_spans')} s WHERE {predicate} ORDER BY s.timestamp DESC, s.event_key DESC {self._limit()}",
                binds,
            )
            return [
                {
                    "key": row[0],
                    "bundle_key": row[1],
                    "timestamp": row[2],
                    "metadata": self._json(row[3]),
                }
                for row in cursor.fetchall()
            ]

    def summaries(self, *, limit=250, **filters):
        self._require_ready()
        predicate, binds = self._predicate(validate_filters(filters))
        binds["page_size"] = max(1, min(int(limit), 1000))
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT s.agent_id, s.thread_id, COUNT(*), COUNT(DISTINCT s.bundle_key), MAX(s.timestamp) FROM {self.table('obs_spans')} s WHERE {predicate} GROUP BY s.agent_id, s.thread_id ORDER BY MAX(s.timestamp) DESC {self._limit()}",
                binds,
            )
            return [
                {
                    "agent_id": row[0],
                    "thread_id": row[1],
                    "event_count": row[2],
                    "bundle_count": row[3],
                    "latest_timestamp": row[4],
                }
                for row in cursor.fetchall()
            ]

    def preview(self, event_id, **filters):
        self._require_ready()
        # Indexed identity fields plus event_id inside the bounded candidate
        # metadata keep this point lookup scoped without exposing other users.
        predicate, binds = self._predicate(validate_filters(filters))
        binds["event_id"] = event_id
        expression = (
            "JSON_VALUE(s.payload, '$.event_id')"
            if self.oracle
            else "json_extract(s.payload, '$.event_id')"
        )
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT p.preview FROM {self.table('obs_previews')} p JOIN {self.table('obs_spans')} s ON s.event_key = p.event_key WHERE {predicate} AND {expression} = :event_id",
                binds,
            )
            row = cursor.fetchone()
            if not row:
                return None
            return row[0].read() if hasattr(row[0], "read") else row[0]

    def retention(self, *, dry_run=True, **policy):
        self._require_ready()
        cutoffs = self.retention_cutoffs(**policy)
        spans = self.table("obs_spans")
        predicate = "((verified = 1 AND kind = 'verified_outcome' AND timestamp < :outcomes) OR ((verified = 0 OR kind <> 'verified_outcome') AND timestamp < :metadata))"
        event_binds = {k: cutoffs[k] for k in ("outcomes", "metadata")}
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {spans} WHERE {predicate}", event_binds
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table('obs_previews')} WHERE timestamp < :previews",
                {"previews": cutoffs["previews"]},
            )
            previews = cursor.fetchone()[0]
            if not dry_run:
                try:
                    self._lock_writer(cursor)
                    for table in ("obs_resources", "obs_previews"):
                        cursor.execute(
                            f"DELETE FROM {self.table(table)} WHERE event_key IN (SELECT event_key FROM {spans} WHERE {predicate})",
                            event_binds,
                        )
                    cursor.execute(
                        f"DELETE FROM {spans} WHERE {predicate}", event_binds
                    )
                    cursor.execute(
                        f"DELETE FROM {self.table('obs_previews')} WHERE timestamp < :previews",
                        {"previews": cutoffs["previews"]},
                    )
                    cursor.execute(
                        f"DELETE FROM {self.table('obs_bundles')} WHERE bundle_key NOT IN (SELECT bundle_key FROM {spans})"
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        return {
            "dry_run": dry_run,
            "metadata_events": count,
            "previews": previews,
            "cutoffs": cutoffs,
            "source_bundles_unchanged": True,
        }


class SQLiteSpanIndex(SQLSpanIndex):
    provider_name = "filesystem-sqlite"

    def __init__(self, root):
        self.path = Path(root) / "_observability" / "index.sqlite3"

    @contextmanager
    def connection(self):
        if not self.path.exists():
            raise RuntimeError("Observability index not initialized")
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            columns = ", ".join(field + " TEXT" for field in FIELDS)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS obs_spans (event_key TEXT PRIMARY KEY, bundle_key TEXT NOT NULL, user_key TEXT NOT NULL, timestamp TEXT NOT NULL, verified INTEGER NOT NULL, payload TEXT NOT NULL, {columns})"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS obs_bundles (bundle_key TEXT PRIMARY KEY, timestamp TEXT, source_record_id TEXT, fingerprint TEXT, payload TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS obs_resources (event_key TEXT NOT NULL, ref_hash TEXT NOT NULL, PRIMARY KEY(event_key, ref_hash))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS obs_previews (event_key TEXT PRIMARY KEY, timestamp TEXT NOT NULL, preview TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS obs_state (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            for name, fields in (
                ("agent_thread", "agent_id, thread_id"),
                ("root", "root_trace_id"),
                ("tenant", "user_key, application_id"),
                ("kind_status", "kind, status"),
                ("memory", "memory_id"),
                ("job", "job_ref"),
                ("error", "error_code"),
            ):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS obs_{name}_time ON obs_spans ({fields}, timestamp DESC, event_key DESC)"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS obs_ref ON obs_resources (ref_hash, event_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS obs_time ON obs_spans (timestamp DESC, event_key DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS obs_bundle_key ON obs_spans (bundle_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS obs_preview_time ON obs_previews (timestamp)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO obs_state VALUES ('schema_version', '1')"
            )
        self.path.chmod(0o600)
        return self.capabilities()


class OracleSpanIndex(SQLSpanIndex):
    provider_name = "oracle"
    oracle = True

    def __init__(self, provider):
        self.provider = provider
        import re

        schema = provider.config.schema or provider.config.user
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]*", schema):
            raise ValueError("Invalid Oracle schema identifier")
        self.schema = schema

    def table(self, name):
        return self.schema + "." + super().table(name)

    def connection(self):
        return self.provider._get_connection()

    def initialize(self):
        migration = (
            Path(__file__).parents[1]
            / "memory_provider"
            / "oracle"
            / "migrations"
            / "008_observability.sql"
        )
        statements = migration.read_text(encoding="utf-8").split("\n/\n")
        with self.connection() as conn:
            cursor = conn.cursor()
            # The migration is schema-qualified without altering pooled session state.
            for statement in statements:
                if statement.strip():
                    cursor.execute(statement.replace("__SCHEMA__", self.schema).strip())
            conn.commit()
        return self.capabilities()
