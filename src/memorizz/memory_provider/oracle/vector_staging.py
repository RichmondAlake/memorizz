"""Re-embed an Oracle store beside its live vectors, then switch over.

Every table with an ``EMBEDDING`` column gets a shadow ``EMBEDDING_NEXT``
column (any size) and an ``EMBEDDING_NEXT_SIG`` column (the signature of the
text each vector was made from). New vectors are written there while search
keeps reading ``EMBEDDING``; the switch then renames the columns, so the
live column is always ``EMBEDDING`` and every query keeps working:

    EMBEDDING      -> EMBEDDING_PREV  (kept, with EMBEDDING_PREV_SIG)
    EMBEDDING_NEXT -> EMBEDDING

A vector index on the live column is rebuilt on the new column before the
switch. Swapping back moves ``EMBEDDING_PREV`` into the staged slot and only
re-embeds rows whose text changed or that were written since. Oracle VECTOR
columns cannot change size, so this is also how a store changes size.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

NEXT, NEXT_SIG = "EMBEDDING_NEXT", "EMBEDDING_NEXT_SIG"
PREV, PREV_SIG = "EMBEDDING_PREV", "EMBEDDING_PREV_SIG"
# Never part of a memory: row mappers that read every column drop these.
SHADOW_COLUMNS = frozenset(c.lower() for c in (NEXT, NEXT_SIG, PREV, PREV_SIG))

# Columns that name exactly one row, besides the RAW ``id``.
_NATURAL_IDS = {
    MemoryType.WORKFLOW_MEMORY: "workflow_id",
    MemoryType.SUMMARIES: "summary_id",
    MemoryType.ENTITY_MEMORY: "entity_id",
    MemoryType.SEMANTIC_CACHE: "cache_key",
    MemoryType.TOOLBOX: "tool_id",
    MemoryType.SKILLBOX: "skill_id",
    MemoryType.PERSONAS: "persona_id",
}


def _raw(value: str) -> Optional[bytes]:
    try:
        return uuid.UUID(str(value)).bytes
    except (ValueError, TypeError, AttributeError):
        return None


class OracleVectorStaging:
    """Staged re-embedding for ``OracleProvider`` (see the module docstring)."""

    # The store pins one vector size (VECTOR(n) columns) and can change it.
    VECTOR_STAGING = {"fixed_size": True, "resize": True}

    # -- schema helpers -------------------------------------------------------

    def _vector_columns(self, cursor) -> Dict[str, Dict[str, Optional[int]]]:
        """``{TABLE: {COLUMN: declared size}}`` for the VECTOR columns of
        every table that has an ``EMBEDDING`` column."""
        owner = self._schema_owner()
        cursor.execute(
            """
            SELECT table_name, column_name
            FROM all_tab_columns
            WHERE owner = :owner AND data_type = 'VECTOR'
              AND table_name IN (
                SELECT table_name FROM all_tab_columns
                WHERE owner = :owner AND column_name = 'EMBEDDING'
              )
            ORDER BY table_name, column_id
            """,
            {"owner": owner},
        )
        found = cursor.fetchall()
        out: Dict[str, Dict[str, Optional[int]]] = {}
        for table, column in found:
            out.setdefault(table, {})[column] = self._declared_vector_dimension(
                cursor, owner, table, column
            )
        return out

    def _has_column(self, cursor, table: str, column: str) -> bool:
        cursor.execute(
            "SELECT COUNT(*) FROM all_tab_columns WHERE owner = :owner "
            "AND table_name = :t AND column_name = :c",
            {"owner": self._schema_owner(), "t": table, "c": column},
        )
        return cursor.fetchone()[0] > 0

    def _drop_columns(self, cursor, table: str, columns: Iterable[str]) -> None:
        present = [c for c in columns if self._has_column(cursor, table, c)]
        if present:
            cursor.execute(
                f"ALTER TABLE {self._qualified_table(table.lower())} "
                f"DROP ({', '.join(present)})"
            )

    def _vector_index_on(self, cursor, table: str, column: str) -> Optional[str]:
        cursor.execute(
            """
            SELECT i.index_name FROM all_ind_columns c
            JOIN all_indexes i
              ON i.owner = c.index_owner AND i.index_name = c.index_name
            WHERE c.table_owner = :owner AND c.table_name = :t
              AND c.column_name = :c AND i.index_type LIKE 'VECTOR%'
            """,
            {"owner": self._schema_owner(), "t": table, "c": column},
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _index_like(self, cursor, table: str, index: str, name: str) -> str:
        """CREATE statement for an index organized like ``index`` (HNSW or
        IVF, from the data dictionary) named ``name``, on the staged column,
        with the cosine distance MemoRizz searches by. DBMS_METADATA leaves
        the organization out of a vector index's DDL, and its parameters
        are not visible to the schema, so they are not carried over."""
        cursor.execute(
            "SELECT index_subtype FROM all_indexes WHERE owner = :owner AND index_name = :name",
            {"owner": self._schema_owner(), "name": index},
        )
        row = cursor.fetchone()
        subtype = str((row or [""])[0] or "").upper()
        organization = (
            "NEIGHBOR PARTITIONS"
            if "PARTITIONS" in subtype
            else "INMEMORY NEIGHBOR GRAPH"
        )
        return (
            f"CREATE VECTOR INDEX {self._qualified_table(name.lower())} "
            f"ON {self._qualified_table(table.lower())} ({NEXT}) "
            f"ORGANIZATION {organization} DISTANCE COSINE WITH TARGET ACCURACY 95"
        )

    def _forget_columns(self) -> None:
        """Drop every cache that remembers a table's columns or indexes."""
        self._table_columns_cache = {}
        self._archive_schema = {}
        self._vector_indexes_created = set()

    def _staging_table(self, memory_type: MemoryType) -> str:
        return memory_type.value.upper()

    # -- the staging interface (memorizz.finetune.staging) --------------------

    def vector_stage_begin(
        self, dimensions: int, *, reuse_previous: bool = False
    ) -> None:
        """Open an empty staged slot of ``dimensions`` on every vector table,
        or (``reuse_previous``) turn the kept previous vectors into it."""
        dimensions = int(dimensions)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            tables = self._vector_columns(cursor)
            if reuse_previous:
                sizes = {cols.get(PREV) for cols in tables.values() if PREV in cols}
                if sizes != {dimensions}:
                    raise RuntimeError(
                        f"No previous {dimensions}-dimension vectors are kept in this store"
                    )
            for table, cols in tables.items():
                qualified = self._qualified_table(table.lower())
                stale = self._vector_index_on(cursor, table, NEXT)
                if stale:
                    cursor.execute(f"DROP INDEX {self._qualified_table(stale.lower())}")
                self._drop_columns(cursor, table, (NEXT, NEXT_SIG))
                if reuse_previous and PREV in cols:
                    cursor.execute(
                        f"ALTER TABLE {qualified} RENAME COLUMN {PREV} TO {NEXT}"
                    )
                    if self._has_column(cursor, table, PREV_SIG):
                        cursor.execute(
                            f"ALTER TABLE {qualified} RENAME COLUMN {PREV_SIG} TO {NEXT_SIG}"
                        )
                    else:
                        cursor.execute(
                            f"ALTER TABLE {qualified} ADD ({NEXT_SIG} VARCHAR2(40))"
                        )
                else:
                    cursor.execute(
                        f"ALTER TABLE {qualified} ADD "
                        f"({NEXT} VECTOR({dimensions}, FLOAT32), {NEXT_SIG} VARCHAR2(40))"
                    )
            conn.commit()
        self._forget_columns()

    def vector_stage_keys(
        self, memory_type: MemoryType, candidates: Sequence[str]
    ) -> List[str]:
        """The keys ``vector_stage_signatures`` may list a row under."""
        keys = []
        natural = _NATURAL_IDS.get(memory_type)
        for value in candidates:
            raw = _raw(value)
            if raw is not None:
                keys.append("id:" + raw.hex().upper())
            if natural:
                keys.append(f"{natural}:{value}")
        return keys

    def vector_stage_signatures(self, memory_type: MemoryType) -> Dict[str, str]:
        """``{key: text signature}`` of the rows that have a staged vector."""
        table = self._staging_table(memory_type)
        natural = _NATURAL_IDS.get(memory_type)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if not self._has_column(cursor, table, NEXT):
                return {}
            if natural and not self._has_column(cursor, table, natural.upper()):
                natural = None
            cursor.execute(
                f"SELECT RAWTOHEX(id), {natural or 'NULL'}, {NEXT_SIG} "
                f"FROM {self._qualified_table(table.lower())} WHERE {NEXT} IS NOT NULL"
            )
            out: Dict[str, str] = {}
            for raw_hex, natural_value, sig in cursor:
                out["id:" + str(raw_hex).upper()] = sig or ""
                if natural and natural_value is not None:
                    out[f"{natural}:{natural_value}"] = sig or ""
            return out

    def vector_stage_put(
        self,
        memory_type: MemoryType,
        items: Sequence[Tuple[Sequence[str], Sequence[float], str]],
    ) -> List[bool]:
        """Write staged vectors: ``(candidate ids, vector, text signature)``
        each. A row is matched by its RAW id or its own id column, never by
        an id many rows share."""
        table = self._staging_table(memory_type)
        qualified = self._qualified_table(table.lower())
        natural = _NATURAL_IDS.get(memory_type)
        done: List[bool] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if natural and not self._has_column(cursor, table, natural.upper()):
                natural = None
            for candidates, vector, sig in items:
                ok = False
                binds = {"v": self._prepare_vector_value(list(vector)), "s": sig}
                for value in candidates:
                    tries = []
                    raw = _raw(value)
                    if raw is not None:
                        tries.append(("id", raw))
                    if natural:
                        tries.append((natural, str(value)))
                    for column, key in tries:
                        if column != "id":  # the RAW id is the primary key
                            cursor.execute("SAVEPOINT memorizz_stage_row")
                        cursor.execute(
                            f"UPDATE {qualified} SET {NEXT} = :v, {NEXT_SIG} = :s "
                            f"WHERE {column} = :k",
                            {**binds, "k": key},
                        )
                        if cursor.rowcount == 1:
                            ok = True
                            break
                        if cursor.rowcount > 1:  # not a key after all: undo it
                            cursor.execute("ROLLBACK TO SAVEPOINT memorizz_stage_row")
                    if ok:
                        break
                done.append(ok)
            conn.commit()
        return done

    def vector_stage_switch(self, staged: Iterable[MemoryType]) -> Dict[str, Any]:
        """Make the staged vectors live and keep the live ones as previous.

        Tables that were not re-embedded (``staged`` lists those that were)
        carry their vectors over when the size is unchanged, else lose them
        (reported). A vector index on the live column is first rebuilt on
        the staged column, so search never runs unindexed."""
        staged_tables = {self._staging_table(m) for m in staged}
        report: Dict[str, Any] = {"carried": {}, "dropped": {}, "indexes": []}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            tables = {
                t: c for t, c in self._vector_columns(cursor).items() if NEXT in c
            }
            if not tables:
                raise RuntimeError("Nothing is staged in this store")
            for table, cols in tables.items():
                if table in staged_tables:
                    continue
                qualified = self._qualified_table(table.lower())
                if cols.get("EMBEDDING") == cols.get(NEXT):
                    cursor.execute(
                        f"UPDATE {qualified} SET {NEXT} = EMBEDDING "
                        f"WHERE EMBEDDING IS NOT NULL AND {NEXT} IS NULL"
                    )
                    if cursor.rowcount:
                        report["carried"][table] = cursor.rowcount
                else:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {qualified} "
                        f"WHERE EMBEDDING IS NOT NULL AND {NEXT} IS NULL"
                    )
                    lost = cursor.fetchone()[0]
                    if lost:
                        report["dropped"][table] = lost
            conn.commit()
            # Indexes first (they take the time), then the renames back to back.
            rebuilt: Dict[str, Tuple[str, str]] = {}
            for table in tables:
                live_index = self._vector_index_on(cursor, table, "EMBEDDING")
                if not live_index:
                    continue
                # Vector indexes cannot be renamed: alternate two names.
                fresh = (
                    live_index[:-2]
                    if live_index.endswith("_N")
                    else f"{live_index[:120]}_N"
                )
                cursor.execute(self._index_like(cursor, table, live_index, fresh))
                rebuilt[table] = (live_index, fresh)
            for table in tables:
                qualified = self._qualified_table(table.lower())
                self._drop_columns(cursor, table, (PREV, PREV_SIG))
                cursor.execute(
                    f"ALTER TABLE {qualified} RENAME COLUMN EMBEDDING TO {PREV}"
                )
                cursor.execute(
                    f"ALTER TABLE {qualified} RENAME COLUMN {NEXT} TO EMBEDDING"
                )
                cursor.execute(
                    f"ALTER TABLE {qualified} RENAME COLUMN {NEXT_SIG} TO {PREV_SIG}"
                )
            conn.commit()
            # Switched. The old index now covers the previous vectors, which
            # nothing searches: drop it (best effort, the switch stands).
            for table, (old, fresh) in rebuilt.items():
                try:
                    cursor.execute(f"DROP INDEX {self._qualified_table(old.lower())}")
                except Exception as exc:
                    logger.warning(
                        "Could not drop the old vector index %s: %s", old, exc
                    )
                report["indexes"].append(f"{table}.{fresh}")
        self._forget_columns()
        return report

    def vector_stage_drop(self) -> None:
        """Discard the staged vectors (a swap that did not finish)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table, cols in self._vector_columns(cursor).items():
                if NEXT in cols:
                    stale = self._vector_index_on(cursor, table, NEXT)
                    if stale:
                        cursor.execute(
                            f"DROP INDEX {self._qualified_table(stale.lower())}"
                        )
                self._drop_columns(cursor, table, (NEXT, NEXT_SIG))
            conn.commit()
        self._forget_columns()

    def vector_previous(self) -> Optional[Dict[str, Any]]:
        """The kept previous vectors: ``{"dimensions", "rows"}``, or None."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            tables = {
                t: c for t, c in self._vector_columns(cursor).items() if PREV in c
            }
            if not tables:
                return None
            rows = 0
            for table in tables:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self._qualified_table(table.lower())} "
                    f"WHERE {PREV} IS NOT NULL"
                )
                rows += cursor.fetchone()[0]
            sizes = {c[PREV] for c in tables.values()}
            return {
                "dimensions": sizes.pop() if len(sizes) == 1 else None,
                "rows": rows,
            }

    def vector_previous_drop(self) -> None:
        """Forget the kept previous vectors."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table in self._vector_columns(cursor):
                self._drop_columns(cursor, table, (PREV, PREV_SIG))
            conn.commit()
        self._forget_columns()


__all__ = ["OracleVectorStaging", "SHADOW_COLUMNS"]
