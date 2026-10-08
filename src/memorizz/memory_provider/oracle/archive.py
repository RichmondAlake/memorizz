"""Oracle's lossless archive adapter, using native tables and JSON extensions.

The optional archive_extras column keeps portable fields for which a relational
table has no column. Native fields remain authoritative after subsequent edits.
No embedding service is called unless reembed=True.
"""

import json
import uuid
from datetime import datetime, timezone

from ...enums.memory_type import MemoryType
from ...memory_archive import ID_FIELDS, MemoryArchiveError

_RAW_TYPES = {
    MemoryType.CONVERSATION_MEMORY,
    MemoryType.KNOWLEDGE_BASE,
    MemoryType.SHORT_TERM_MEMORY,
    MemoryType.SHARED_MEMORY,
    MemoryType.SEMANTIC_CACHE,
}
_JSON = {
    "traits",
    "expertise",
    "parameters",
    "input_schema",
    "tool_policy",
    "aliases",
    "deprecated_arguments",
    "queries",
    "preconditions",
    "tools_used",
    "baseline",
    "stats",
    "source_workflow_ids",
    "steps",
    "outcome",
    "canonical_signature",
    "skills_activated",
    "shadow_evaluations",
    "access_list",
    "source_message_ids",
    "original_memory_ids",
    "attributes",
    "relations",
    "metadata",
    "linked_source_ids",
    "outcome_details",
    "archive_extras",
}


def target_id(kind, identifier):
    if kind in _RAW_TYPES:
        try:
            return str(uuid.UUID(identifier))
        except ValueError:
            return str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:{kind.value}:{identifier}")
            )
    return identifier


def load_extensions(provider):
    with provider._get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT table_name FROM user_tab_columns WHERE column_name = 'ARCHIVE_EXTRAS'"
        )
        provider._archive_types = {row[0].lower() for row in cursor}


def columns(provider, kind):
    cache = getattr(provider, "_archive_schema", None)
    if cache is None:
        provider._archive_schema = cache = {}
    if kind in cache:
        return cache[kind]
    with provider._get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT column_name, data_type, nullable, data_default, char_length FROM user_tab_columns WHERE table_name = :table_name ORDER BY column_id",
            {"table_name": kind.value.upper()},
        )
        cache[kind] = {
            name.lower(): (
                data_type,
                nullable,
                provider._read_lob_value(default),
                length,
            )
            for name, data_type, nullable, default, length in cursor
        }
    return cache[kind]


def list_records(provider, kind):
    if kind == MemoryType.MEMAGENT:
        return provider.list_all(kind)
    table = provider._get_table_name(kind)
    with provider._get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table}")
        return _decode(provider, kind, cursor)


def get_record(provider, kind, identifier):
    natural = ID_FIELDS.get(kind)
    table = provider._get_table_name(kind)
    with provider._get_connection() as connection:
        cursor = connection.cursor()
        if kind == MemoryType.SHARED_MEMORY:
            # Live coordination sessions are addressed by memory_id, which is
            # often itself a UUID. Archives also expose the physical row UUID.
            cursor.execute(
                f"SELECT * FROM {table} WHERE memory_id = :identifier",
                {"identifier": identifier},
            )
            records = _decode(provider, kind, cursor, parse_content=False)
            if records:
                return records[0]
        if natural:
            cursor.execute(
                f"SELECT * FROM {table} WHERE {natural} = :identifier",
                {"identifier": identifier},
            )
        else:
            try:
                raw_id = uuid.UUID(identifier).bytes
            except ValueError:
                raw_id = None
            if raw_id is not None:
                cursor.execute(
                    f"SELECT * FROM {table} WHERE id = :identifier",
                    {"identifier": raw_id},
                )
            else:
                return None
        records = _decode(provider, kind, cursor, parse_content=False)
        return records[0] if records else None


def _decode(provider, kind, cursor, *, parse_content=True):
    names = [field[0].lower() for field in cursor.description]
    records = []
    for raw in cursor:
        data = {}
        for key, item in zip(names, raw):
            item = provider._read_lob_value(item)
            if isinstance(item, bytes):
                item = str(uuid.UUID(bytes=item))
            elif isinstance(item, datetime):
                item = item.replace(tzinfo=timezone.utc).isoformat()
            elif key in _JSON or (
                parse_content and key == "content" and kind == MemoryType.SHARED_MEMORY
            ):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except ValueError:
                        pass
                elif key == "embedding" and item is not None:
                    item = list(item)
                elif (
                    key == "success"
                    and kind == MemoryType.TOOL_LOG
                    and item is not None
                ):
                    item = bool(item)
            data[key] = item
        extras = data.pop("archive_extras", None)
        if isinstance(extras, dict):
            data = {**extras, **data}
        key = ID_FIELDS.get(kind)
        identifier = data.get(key) if key else data.get("id")
        data["_id"] = data["id"] = str(identifier)
        records.append(data)
    return records


def validate_record(provider, kind, identifier, data):
    if kind == MemoryType.MEMAGENT:
        from ...memagent.models import MemAgentModel

        MemAgentModel(**data)
        return
    schema = columns(provider, kind)
    native = dict(data)
    if kind in ID_FIELDS:
        native.setdefault(ID_FIELDS[kind], identifier)
    for key, (datatype, nullable, default, length) in schema.items():
        if key in {"id", "archive_extras", "embedding"}:
            continue
        value = native.get(key)
        if nullable == "N" and default is None and (value is None or value == ""):
            raise MemoryArchiveError(f"{kind.value} requires field {key} on Oracle")
        if (
            datatype in {"VARCHAR2", "CHAR"}
            and isinstance(value, str)
            and len(value) > length
        ):
            raise MemoryArchiveError(
                f"{kind.value}.{key} exceeds the Oracle column length"
            )


def store_record(provider, kind, identifier, data, *, replace=False, reembed=False):
    if kind == MemoryType.MEMAGENT:
        from ...memagent.models import MemAgentModel

        # Taxonomy persona rows restore before the final agent configuration. Reusing
        # that row avoids native agent persistence creating a duplicate or
        # overwriting the persona's preserved metadata and timestamps.
        persona = data.get("persona")
        sync_persona = bool(persona)
        persona_id = persona.get("persona_id") if isinstance(persona, dict) else None
        if persona_id and provider.retrieve_by_id(persona_id, MemoryType.PERSONAS):
            sync_persona = False
        provider.store_memagent(
            MemAgentModel(**data),
            generate_embeddings=reembed,
            sync_tools=False,
            sync_persona=sync_persona,
        )
        return identifier
    validate_record(provider, kind, identifier, data)
    table = provider._get_table_name(kind)
    schema = columns(provider, kind)
    native = dict(data)
    natural = ID_FIELDS.get(kind)
    if natural:
        native.setdefault(natural, identifier)
    raw_id = (
        uuid.UUID(identifier).bytes
        if kind in _RAW_TYPES
        else uuid.uuid5(
            uuid.NAMESPACE_URL, f"memorizz-row:{kind.value}:{identifier}"
        ).bytes
    )
    native["id"] = raw_id
    extras = {
        key: item
        for key, item in data.items()
        if key not in schema and key not in {"_id", "id", "row_id", "embedding"}
    }
    # Preserve the type/shape of structured content that SQL stores as text.
    # Native columns are decoded normally, and this extension only holds fields
    # which are absent from the relational schema.
    with provider._get_connection() as connection:
        cursor = connection.cursor()
        if "archive_extras" not in schema:
            try:
                cursor.execute(
                    f"ALTER TABLE {table} ADD (archive_extras CLOB CHECK (archive_extras IS JSON))"
                )
            except Exception as exc:
                if "ORA-01430" not in str(exc):
                    raise
            provider._archive_schema.pop(kind, None)
            provider._table_columns_cache.pop(kind.value.upper(), None)
            schema = columns(provider, kind)
        provider._archive_types.add(kind.value)
        values = {"archive_extras": json.dumps(extras, ensure_ascii=False)}
        if "embedding" in schema:
            values["embedding"] = None  # Replacing content must invalidate old vectors.
        for key, (datatype, _, default, _) in schema.items():
            if key not in native or key == "embedding":
                continue
            item = native[key]
            if item is None and default is not None:
                continue  # Let the target's native default apply.
            if datatype.startswith("TIMESTAMP") or datatype == "DATE":
                if isinstance(item, str):
                    item = datetime.fromisoformat(item.replace("Z", "+00:00"))
                if isinstance(item, datetime) and item.tzinfo:
                    item = item.astimezone(timezone.utc).replace(tzinfo=None)
            elif isinstance(item, (dict, list)):
                item = json.dumps(item, ensure_ascii=False)
            elif isinstance(item, bool):
                item = int(item)
            values[key] = item
        if reembed and "embedding" in schema:
            text = data.get("content") or data.get("description") or data.get("name")
            if isinstance(text, str) and text:
                embedding = provider._generate_embedding_if_needed(text)
                if embedding is None:
                    raise MemoryArchiveError("Embedding generation failed")
                values["embedding"] = provider._prepare_vector_value(embedding)
                provider._set_vector_input_size(cursor, "embedding")
        selector = natural or "id"
        select_value = identifier if natural else raw_id
        cursor.execute(
            f"SELECT id FROM {table} WHERE {selector} = :identifier",
            {"identifier": select_value},
        )
        existing = cursor.fetchone()
        try:
            if existing:
                if not replace:
                    raise MemoryArchiveError(
                        "A destination record appeared after import preview"
                    )
                values.pop("id", None)
                cursor.execute(
                    f"UPDATE {table} SET {', '.join(key + ' = :' + key for key in values)} WHERE {selector} = :archive_identifier",
                    {**values, "archive_identifier": select_value},
                )
            else:
                cursor.execute(
                    f"INSERT INTO {table} ({', '.join(values)}) VALUES ({', '.join(':' + key for key in values)})",
                    values,
                )
            if kind == MemoryType.SUMMARIES:
                links = f"{provider.config.schema}.summary_message_links"
                cursor.execute(
                    f"DELETE FROM {links} WHERE summary_id = :summary",
                    {"summary": identifier},
                )
                for position, message in enumerate(
                    data.get("source_message_ids") or []
                ):
                    cursor.execute(
                        f"INSERT INTO {links} (summary_id, message_id, position) SELECT :summary, id, :position FROM {provider._get_table_name(MemoryType.CONVERSATION_MEMORY)} WHERE id = :message",
                        {
                            "summary": identifier,
                            "position": position,
                            "message": uuid.UUID(message).bytes,
                        },
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return identifier
