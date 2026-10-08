"""Notion document memory with an independently selected semantic provider."""

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, unquote

from ...enums.memory_type import MemoryType
from ..base import (
    _UNSET,
    MemoryProvider,
    MemoryProviderCapabilities,
    filter_tool_log_rows,
)
from ..vectors import VECTOR_SCOPE_FIELDS, validate_scope, validate_vector
from .client import (
    NotionAPIError,
    NotionClient,
    NotionError,
    NotionWriteUncertain,
    validate_transport_options,
)
from .state import NotionState

PROPERTIES = {
    "name": ("Name", "title"),
    "id": ("Memorizz ID", "rich_text"),
    "memory_type": ("Memory type", "select"),
    "content": ("Content", "rich_text"),
    "payload": ("Memorizz metadata", "rich_text"),
    "memory_id": ("Memory ID", "rich_text"),
    "user_id": ("User ID", "rich_text"),
    "thread_id": ("Thread ID", "rich_text"),
    "namespace": ("Namespace", "rich_text"),
    "agent_id": ("Agent ID", "rich_text"),
    "record_type": ("Record type", "rich_text"),
    "status": ("Status", "rich_text"),
    "timestamp": ("Timestamp", "rich_text"),
    "event_time": ("Event time", "date"),
    "application_id": ("Application ID", "rich_text"),
    "trace_memory_id": ("Trace memory ID", "rich_text"),
    "session_id": ("Session ID", "rich_text"),
}
MANAGED_FIELDS = VECTOR_SCOPE_FIELDS | {
    "record_type",
    "timestamp",
    "application_id",
    "trace_memory_id",
}
SEMANTIC_TYPES = frozenset(MemoryType) - {
    MemoryType.MEMAGENT,
    MemoryType.SHARED_MEMORY,
    MemoryType.TOOL_LOG,
}
DISPLAY_FIELDS = (
    "content",
    "text",
    "summary",
    "instruction",
    "instructions",
    "description",
    "query_text",
    "query",
    "attributes",
    "name",
)


class NotionIntegrityError(NotionError):
    """A managed record/schema is malformed or its protected identity changed."""


class NotionQueryLimitError(NotionError):
    """A remote scan was incomplete. Narrow/partition the query; never claim completeness."""


class NotionIndexingError(NotionError):
    def __init__(self, record_id, page_id):
        self.record_id, self.page_id = record_id, page_id
        super().__init__(
            f"Notion record {record_id} is saved, but semantic indexing is pending. "
            "Use repair_index() or retry with this same record ID."
        )


@dataclass
class NotionConfig:
    data_source_id: str
    token: Optional[str] = field(default=None, repr=False)
    state_path: Optional[Any] = None
    timeout: float = 30.0
    requests_per_second: float = 2.5
    max_retries: int = 4
    max_query_pages: int = 100
    max_semantic_candidates: int = 100
    read_only: bool = False
    semantic_memory_types: Optional[Any] = None

    def __post_init__(self):
        validate_transport_options(
            self.timeout, self.requests_per_second, self.max_retries
        )
        self.data_source_id = str(uuid.UUID(str(self.data_source_id)))
        self.token = self.token or os.environ.get("NOTION_TOKEN")
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("Set NOTION_TOKEN or supply NotionConfig(token=...)")
        for name, ceiling in (
            ("max_query_pages", 100),
            ("max_semantic_candidates", 1000),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= ceiling
            ):
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        if not isinstance(self.read_only, bool):
            raise ValueError("read_only must be boolean")
        self.semantic_memory_types = frozenset(
            MemoryType(value)
            for value in (
                SEMANTIC_TYPES
                if self.semantic_memory_types is None
                else self.semantic_memory_types
            )
        )
        if self.state_path is None:
            from ..._env_io import memorizz_home

            self.state_path = (
                memorizz_home() / "notion" / (self.data_source_id + ".sqlite3")
            )
        self.state_path = Path(self.state_path).expanduser().resolve()


def rich_text(value):
    text = str(value)
    if len(text) > 180_000:
        raise ValueError(
            "Notion text exceeds 180,000 characters; split the memory into chunks"
        )
    return [
        {"type": "text", "text": {"content": text[index : index + 2000]}}
        for index in range(0, len(text), 2000)
    ]


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "value"):
        return value.value
    if callable(value):
        return None
    raise TypeError(f"Notion memory cannot serialize {type(value).__name__}")


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _without_agent_credentials(value):
    """Persist references/configuration, never inline agent connection secrets."""
    secret_keys = {
        "api_key",
        "password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "client_secret",
        "secret_key",
        "x_api_key",
        "apikey",
        "api_token",
        "bearer_token",
        "api_secret",
        "headers",
    }
    if isinstance(value, dict):
        return {
            key: _without_agent_credentials(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in secret_keys
            and not str(key).lower().endswith(("_api_key", "_token", "_password"))
        }
    if isinstance(value, list):
        return [_without_agent_credentials(item) for item in value]
    return value


def _fingerprint(record):
    # Human-editable content and scope must match the indexed snapshot. Runtime
    # scores/provenance and original embeddings are not canonical Notion fields.
    return hashlib.sha256(
        _json(
            {
                key: value
                for key, value in record.items()
                if key not in {"embedding", "score", "source_id", "notion"}
            }
        ).encode()
    ).hexdigest()


def _parse_time(value):
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(value, timezone.utc)
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            parsed = parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(
                timezone.utc
            )
        return parsed
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _event_time(value):
    # Live Notion date properties discard seconds/subseconds. Use this column
    # only as a minute bucket; retain the precise timestamp in text/metadata.
    parsed = _parse_time(value)
    return (
        parsed.replace(second=0, microsecond=0).isoformat(timespec="milliseconds")
        if parsed is not None
        else None
    )


def _timestamp_text(value):
    parsed = _parse_time(value)
    if parsed is not None:
        return parsed.isoformat(timespec="microseconds")
    return "" if value is None else str(value)


def _validate_filter(query):
    if not isinstance(query, dict):
        raise ValueError("Notion filters must be dictionaries")
    for key, value in query.items():
        if key in {"$and", "$or"}:
            if not isinstance(value, list) or not value:
                raise ValueError("Compound filters must contain a non-empty list")
            for item in value:
                _validate_filter(item)
        elif key.startswith("$"):
            raise ValueError("Unsupported Notion filter operator")
        elif isinstance(value, dict):
            for operator, expected in value.items():
                if operator not in {"$eq", "$ne", "$in", "$nin"}:
                    raise ValueError("Unsupported Notion filter operator")
                if operator in {"$in", "$nin"} and not isinstance(expected, list):
                    raise ValueError("$in/$nin filters require lists")


def _matches(record, query):
    for key, expected in query.items():
        if key in {"$and", "$or"}:
            if not isinstance(expected, list) or not expected:
                raise ValueError("Compound filters must contain a non-empty list")
            results = [_matches(record, item) for item in expected]
            if not (all(results) if key == "$and" else any(results)):
                return False
            continue
        if key.startswith("$"):
            raise ValueError("Unsupported Notion filter operator")
        actual = record.get(key)
        if isinstance(expected, dict):
            for operator, value in expected.items():
                if operator == "$eq":
                    matched = actual == value
                elif operator == "$ne":
                    matched = actual != value
                elif operator in {"$in", "$nin"} and isinstance(value, list):
                    matched = (actual in value) == (operator == "$in")
                else:
                    raise ValueError(
                        "Notion filters support equality, $ne, $in, $nin, $and and $or"
                    )
                if not matched:
                    return False
        elif actual != expected:
            return False
    return True


class NotionProvider(MemoryProvider):
    """Notion is canonical; ``semantic_provider`` stores vectors and references.

    Constructors never provision remote resources. Use provision_notion_workspace
    explicitly, then pass its memory data-source ID. The injected provider is
    caller-owned and is not closed by this provider.
    """

    def __init__(self, config: NotionConfig, semantic_provider=None, *, client=None):
        self.config = config
        self.semantic_provider = semantic_provider
        self._owns_semantic_provider = False
        if (
            semantic_provider is not None
            and not semantic_provider.memory_capabilities().vector_store
        ):
            raise ValueError(
                "semantic_provider must implement the Memorizz vector-store contract"
            )
        self._client = client or NotionClient(
            config.token,
            timeout=config.timeout,
            requests_per_second=config.requests_per_second,
            max_retries=config.max_retries,
        )
        self._owns_client = client is None
        self._state = NotionState(config.state_path)
        try:
            self._state.bind_source(config.data_source_id)
            schema = self._client.request(
                "GET", "/data_sources/" + config.data_source_id
            )
            self.database_id = (schema.get("parent") or {}).get("database_id")
            self._properties = self._resolve_schema(schema)
            self._semantic_identity = (
                semantic_provider.semantic_identity()
                if semantic_provider is not None
                else None
            )
            self._index_configuration = {
                "semantic_identity": self._semantic_identity,
                "memory_types": sorted(
                    value.value for value in config.semantic_memory_types
                ),
            }
            if not config.read_only:
                self._state.activate_index(self._index_configuration)
        except Exception:
            if self._owns_client:
                self._client.close()
            raise

    def _resolve_schema(self, schema):
        saved = self._state.setting("property_ids") or {}
        properties = schema.get("properties") or {}
        by_id = {value.get("id"): value for value in properties.values()}
        resolved = {}
        for key, (name, kind) in PROPERTIES.items():
            prop = by_id.get(saved.get(key)) or properties.get(name)
            if (
                not isinstance(prop, dict)
                or prop.get("type") != kind
                or not prop.get("id")
            ):
                raise NotionIntegrityError(
                    f"Notion data source is missing the managed {name} ({kind}) property"
                )
            resolved[key] = prop["id"]
        self._state.setting("property_ids", resolved)
        return resolved

    def memory_capabilities(self):
        return MemoryProviderCapabilities(
            provider=type(self).__name__,
            result_scores=self.semantic_provider is not None,
            native_vector_search=False,
            native_hybrid_search=False,
            manages_embeddings=True,
            requires_live_read=True,
        )

    def _require_write(self):
        if self.config.read_only:
            raise PermissionError("The Notion provider is read-only")
        self._require_index_configuration()

    def _require_index_configuration(self):
        if self._state.setting("index_configuration") != self._index_configuration:
            raise NotionError(
                "Another provider changed this journal's index configuration; reconstruct the provider"
            )
        if (
            self.semantic_provider is not None
            and self.semantic_provider.semantic_identity() != self._semantic_identity
        ):
            raise NotionError(
                "Semantic provider/model changed; reconstruct the Notion provider and repair its index"
            )

    def _stored_status(self, memory_type):
        return (
            "indexed"
            if self.semantic_provider is not None
            and memory_type in self.config.semantic_memory_types
            else "stored"
        )

    def embed_text(self, text):
        if self.semantic_provider is None:
            raise NotImplementedError(
                "Configure semantic_provider= to generate embeddings"
            )
        self._require_index_configuration()
        return self.semantic_provider.embed_text(text)

    def _namespace(self, memory_type):
        return (
            "memorizz:notion:"
            + self.config.data_source_id
            + ":"
            + MemoryType(memory_type).value
        )

    def _property(self, page, key):
        identifier = self._properties[key]
        for name, prop in (page.get("properties") or {}).items():
            if prop.get("id") == identifier or name == identifier:
                return prop
        raise NotionIntegrityError(f"Managed Notion property {key} is missing")

    def _text(self, page, key):
        prop = self._property(page, key)
        kind = PROPERTIES[key][1]
        if kind == "select":
            return (prop.get("select") or {}).get("name") or ""
        values = prop.get(kind) or []
        # Retrieve-page responses can truncate rich-text properties. The
        # property-item endpoint, not the inline array, is authoritative here.
        if len(values) >= 25:
            values, cursor, seen = [], None, set()
            for _ in range(100):
                params = {"page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor
                result = self._client.request(
                    "GET",
                    "/pages/"
                    + str(uuid.UUID(page["id"]))
                    + "/properties/"
                    + quote(unquote(self._properties[key]), safe=""),
                    params=params,
                )
                values.extend(
                    item[kind] for item in result.get("results", []) if kind in item
                )
                if not result.get("has_more"):
                    break
                cursor = result.get("next_cursor")
                if not cursor or cursor in seen:
                    raise NotionIntegrityError(
                        "Notion property pagination did not advance"
                    )
                seen.add(cursor)
            else:
                raise NotionQueryLimitError(
                    "Notion property exceeds the pagination budget"
                )
        return "".join(
            value.get("plain_text", (value.get("text") or {}).get("content", ""))
            for value in values
        )

    def _decode(self, page, expected_type=None, expected_id=None):
        if page.get("archived") or page.get("in_trash"):
            return None
        parent = page.get("parent") or {}
        if str(parent.get("data_source_id", "")).replace(
            "-", ""
        ) != self.config.data_source_id.replace("-", ""):
            raise NotionIntegrityError(
                "Notion record is outside the configured data source"
            )
        identifier = self._text(page, "id")
        memory_type = self._text(page, "memory_type")
        if expected_id is not None and identifier != expected_id:
            raise NotionIntegrityError(
                "A Notion record's protected Memorizz ID changed"
            )
        if expected_type is not None and memory_type != MemoryType(expected_type).value:
            raise NotionIntegrityError(
                "A Notion record's protected memory type changed"
            )
        try:
            envelope = json.loads(self._text(page, "payload"))
            record = envelope["data"]
            valid = (
                envelope["format"] == "memorizz.notion.v1"
                and isinstance(record, dict)
                and envelope["id"] == identifier
                and envelope["memory_type"] == memory_type
            )
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise NotionIntegrityError(
                "Notion memory metadata is invalid; no stale fallback was used"
            )
        record = dict(record)
        display = envelope.get("display_field")
        if display:
            value = self._text(page, "content")
            try:
                record[display] = (
                    json.loads(value) if envelope.get("display_json") else value
                )
            except ValueError:
                raise NotionIntegrityError(
                    "The structured Content property is not valid JSON"
                ) from None
            for alias in envelope.get("display_aliases", []):
                if (display, alias) not in {
                    ("content", "text"),
                    ("text", "content"),
                    ("query_text", "query"),
                    ("query", "query_text"),
                }:
                    raise NotionIntegrityError("Invalid Notion display alias")
                record[alias] = record[display]
        # Name is editable when the original record has a name. Generated row
        # titles do not mutate the underlying memory's schema.
        if isinstance(record.get("name"), str):
            record["name"] = self._text(page, "name")
        for key in MANAGED_FIELDS:
            value = self._text(page, key)
            original = record.get(key)
            if key == "timestamp":
                value, original = _timestamp_text(value), _timestamp_text(original)
            # Scope and identity are managed properties. Do not let an edit of
            # a display column silently grant a record to another tenant.
            if value != ("" if original is None else str(original)):
                raise NotionIntegrityError(
                    f"Managed Notion {key} differs from its stored scope"
                )
        date = (self._property(page, "event_time").get("date") or {}).get("start")
        if _event_time(date) != _event_time(record.get("timestamp")):
            raise NotionIntegrityError(
                "Managed Notion event time differs from the stored timestamp"
            )
        record["_id"] = record["id"] = identifier
        return record

    def _encode(self, record, memory_type):
        data = dict(record)
        data.pop("embedding", None)
        data.pop("score", None)
        data.pop("notion", None)
        display = next((key for key in DISPLAY_FIELDS if key in data), None)
        value = data.get(display, "")
        display_json = not isinstance(value, str)
        content = _json(value) if display_json else value
        if display:
            data.pop(display)
        aliases = []
        for primary, alias in (
            ("content", "text"),
            ("text", "content"),
            ("query_text", "query"),
            ("query", "query_text"),
        ):
            if display == primary and alias in data and data[alias] == value:
                aliases.append(alias)
                data.pop(alias)
        # Name is retained in metadata so generated titles remain distinguishable.
        if "name" in record:
            data["name"] = record["name"]
        envelope = {
            "format": "memorizz.notion.v1",
            "id": record["id"],
            "memory_type": memory_type.value,
            "data": data,
            "display_field": display,
            "display_json": display_json,
            "display_aliases": aliases,
        }
        values = {
            "id": record["id"],
            "name": (
                record["name"]
                if isinstance(record.get("name"), str)
                else memory_type.value + ": " + record["id"]
            ),
            "content": content,
            "payload": _json(envelope),
        }
        for key in MANAGED_FIELDS:
            values[key] = "" if record.get(key) is None else str(record[key])
        # Canonical UTC text is sortable at full precision, unlike Notion dates.
        # The original timestamp representation remains in the record envelope.
        values["timestamp"] = _timestamp_text(record.get("timestamp"))
        properties = {
            self._properties[key]: {PROPERTIES[key][1]: rich_text(value)}
            for key, value in values.items()
        }
        properties[self._properties["memory_type"]] = {
            "select": {"name": memory_type.value}
        }
        date = _event_time(record.get("timestamp"))
        properties[self._properties["event_time"]] = {
            "date": {"start": date} if date else None
        }
        if len(_json(properties).encode()) > 440_000:
            raise ValueError(
                "Notion record is too large; split it into smaller memory records"
            )
        return properties

    def _condition(self, key, value):
        kind = PROPERTIES[key][1]
        if value is None or value == "":
            operation = {"is_empty": True}
        else:
            operation = {"equals": str(value)}
        return {"property": self._properties[key], kind: operation}

    def _pages(
        self,
        memory_type,
        *,
        query=None,
        user_id=_UNSET,
        memory_id=None,
        thread_id=None,
        namespace=None,
        limit=None,
        sorts=None,
        edited_since=None,
    ):
        conditions = [self._condition("memory_type", MemoryType(memory_type).value)]
        filters = dict(query or {})
        for key, value in (
            ("user_id", user_id),
            ("memory_id", memory_id),
            ("thread_id", thread_id),
            ("namespace", namespace),
        ):
            if value is not _UNSET and (key == "user_id" or value is not None):
                conditions.append(self._condition(key, value))
        for key, value in filters.items():
            mapped = "id" if key == "_id" else key
            if (
                mapped in PROPERTIES
                and mapped != "timestamp"
                and (value is None or isinstance(value, (str, int, float)))
            ):
                conditions.append(self._condition(mapped, value))
        if edited_since:
            conditions.append(
                {
                    "timestamp": "last_edited_time",
                    "last_edited_time": {"on_or_after": edited_since},
                }
            )
        cursor, seen, count = None, set(), 0
        for _ in range(self.config.max_query_pages):
            body = {"filter": {"and": conditions}, "page_size": min(100, limit or 100)}
            if sorts:
                body["sorts"] = sorts
            if cursor:
                body["start_cursor"] = cursor
            result = self._client.request(
                "POST",
                "/data_sources/" + self.config.data_source_id + "/query",
                body=body,
            )
            if (result.get("request_status") or {}).get("type") in {
                "incomplete",
                "partial",
            }:
                raise NotionQueryLimitError(
                    "Notion returned an incomplete query; narrow the scope"
                )
            for page in result.get("results", []):
                if page.get("object") not in {None, "page"}:
                    continue
                count += 1
                yield page
            if not result.get("has_more"):
                # The service can stop at its 10,000-result cap without exposing
                # another cursor. Never label that boundary as a complete scan.
                if count >= 10_000:
                    raise NotionQueryLimitError(
                        "Notion's 10,000-row query limit was reached; partition the query"
                    )
                return
            cursor = result.get("next_cursor")
            if not cursor or cursor in seen:
                raise NotionIntegrityError("Notion query pagination did not advance")
            seen.add(cursor)
        raise NotionQueryLimitError("Notion query budget exhausted; narrow the scope")

    def _find_page(self, identifier, memory_type):
        found = None
        for page in self._pages(memory_type, query={"id": identifier}, limit=2):
            if self._text(page, "id") != identifier:
                continue
            if found is not None:
                raise NotionIntegrityError(
                    "Duplicate Memorizz IDs in Notion require reconciliation"
                )
            found = page
        return found

    def _get_page(self, page_id):
        try:
            return self._client.request("GET", "/pages/" + str(uuid.UUID(page_id)))
        except NotionAPIError as exc:
            if exc.status == 404:
                return None
            raise

    def _scope(self, record):
        return {key: record.get(key) for key in VECTOR_SCOPE_FIELDS}

    def _index(self, record, memory_type, page_id, *, embedding=None):
        if (
            self.semantic_provider is None
            or memory_type not in self.config.semantic_memory_types
        ):
            return
        fields = {
            MemoryType.ENTITY_MEMORY: (
                "name",
                "entity_type",
                "description",
                "attributes",
                "relations",
            ),
            MemoryType.PERSONAS: (
                "name",
                "description",
                "role",
                "role_type",
                "background",
                "goals",
                "traits",
                "expertise",
                "content",
            ),
            MemoryType.WORKFLOW_MEMORY: (
                "name",
                "description",
                "steps",
                "workflow",
                "content",
            ),
            MemoryType.SEMANTIC_CACHE: ("query_text", "query"),
            MemoryType.SKILLBOX: ("name", "description", "preconditions", "queries"),
        }.get(memory_type, DISPLAY_FIELDS)
        fragments = list(
            dict.fromkeys(
                record[key] if isinstance(record[key], str) else _json(record[key])
                for key in fields
                if record.get(key) not in (None, "", {}, [])
            )
        )
        if not fragments:
            # Empty content must remove an old vector, not remain searchable.
            self.semantic_provider.delete_vector(
                self._namespace(memory_type), record["id"]
            )
            return
        vector = (
            validate_vector(embedding)
            if embedding is not None
            else self.semantic_provider.embed_text("\n".join(fragments))
        )
        self.semantic_provider.upsert_vector(
            self._namespace(memory_type),
            record["id"],
            vector,
            metadata={
                "page_id": page_id,
                "fingerprint": _fingerprint(record),
                "semantic_identity": self._semantic_identity,
            },
            scope=self._scope(record),
        )

    def store(
        self, data=None, memory_store_type=None, memory_id=None, memory_unit=None
    ):
        return self._store(data, memory_store_type, memory_id, memory_unit)

    def store_archive_record(
        self, memory_type, record_id, data, *, replace=False, reembed=False
    ):
        return self._store(
            dict(data, _id=record_id, id=record_id), memory_type, index_memory=reembed
        )

    def _store(
        self,
        data=None,
        memory_store_type=None,
        memory_id=None,
        memory_unit=None,
        *,
        merge=False,
        index_memory=True,
    ):
        self._require_write()
        if memory_unit is not None:
            data = (
                memory_unit.model_dump()
                if hasattr(memory_unit, "model_dump")
                else dict(memory_unit)
            )
            memory_store_type = (
                getattr(memory_unit, "memory_type", None)
                or data.get("memory_type")
                or memory_store_type
                or MemoryType.CONVERSATION_MEMORY
            )
        if data is None or memory_store_type is None:
            raise ValueError("Provide data and memory_store_type, or a memory_unit")
        memory_type = MemoryType(memory_store_type)
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        record = dict(data)
        # Legacy memory units may carry vectors from a different global model.
        # This composed provider owns embedding generation on its configured
        # semantic backend, so never mix caller/global vectors into that index.
        record.pop("embedding", None)
        embedding = None
        record = json.loads(_json(record))
        if memory_type == MemoryType.MEMAGENT:
            record = _without_agent_credentials(record)
        identifier = str(
            record.get("_id")
            or record.get("id")
            or (record.get("agent_id") if memory_type == MemoryType.MEMAGENT else None)
            or (
                record.get("cache_key")
                if memory_type == MemoryType.SEMANTIC_CACHE
                else None
            )
            or uuid.uuid4()
        )
        if not identifier or len(identifier) > 512:
            raise ValueError("Memory record IDs must contain 1–512 characters")
        record["id"] = record["_id"] = identifier
        if memory_type == MemoryType.MEMAGENT:
            record["agent_id"] = identifier
        if memory_id is not None:
            record.setdefault("memory_id", memory_id)
        error = None
        with self._state.writer() as connection:
            prior = self._state.get(identifier, memory_type.value, connection)
            if prior and prior["status"] == "creating":
                raise NotionWriteUncertain(
                    "A durable create intent already exists; call repair_index() instead of creating again"
                )
            page = (
                self._get_page(prior["page_id"])
                if prior and prior.get("page_id")
                else self._find_page(identifier, memory_type)
            )
            if prior and prior.get("page_id") and page is None:
                raise NotionIntegrityError(
                    "A previously known Notion page is inaccessible; refusing to recreate it"
                )
            if page and (page.get("archived") or page.get("in_trash")):
                raise NotionIntegrityError(
                    "This record was archived; use a new ID rather than resurrecting it"
                )
            if page:
                existing = self._decode(page, memory_type, identifier)
                if existing.get("immutable_trace") is True:
                    if merge:
                        raise PermissionError("An immutable trace cannot be updated")
                    return identifier
                if merge:
                    record = {**existing, **record}
            elif merge:
                return None
            if (
                not page
                and prior
                and prior["status"]
                in {"uncertain", "deleted", "delete_pending", "abandoned"}
            ):
                raise NotionWriteUncertain(
                    "This ID has an unresolved or deleted Notion write; reconcile it before creating"
                )
            record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            validate_scope(self._scope(record))
            properties = self._encode(record, memory_type)
            if page is None:
                # Commit intent BEFORE the external create. A process crash
                # after Notion accepts it must not erase our deduplication key.
                # Other writers sharing this journal see 'creating' and refuse
                # to issue another create for the same identity.
                self._state.put(
                    identifier,
                    memory_type.value,
                    status="creating",
                    connection=connection,
                )
                connection.commit()
            else:
                # A crash during PATCH also needs a durable reconciliation
                # record, even if no exception can reach this process.
                self._state.put(
                    identifier,
                    memory_type.value,
                    page_id=page["id"],
                    status="uncertain",
                    connection=connection,
                )
                connection.commit()
            try:
                if page:
                    page = self._client.request(
                        "PATCH", "/pages/" + page["id"], body={"properties": properties}
                    )
                else:
                    page = self._client.request(
                        "POST",
                        "/pages",
                        body={
                            "parent": {
                                "type": "data_source_id",
                                "data_source_id": self.config.data_source_id,
                            },
                            "properties": properties,
                        },
                    )
            except NotionWriteUncertain as exc:
                exc.record_id = identifier
                exc.page_id = page["id"] if page else None
                self._state.put(
                    identifier,
                    memory_type.value,
                    page_id=exc.page_id,
                    status="uncertain",
                    connection=connection,
                )
                error = exc
            except NotionAPIError as exc:
                if page is None:
                    self._state.put(
                        identifier,
                        memory_type.value,
                        status="not_created",
                        connection=connection,
                    )
                error = exc
            if error is None:
                self._state.put(
                    identifier,
                    memory_type.value,
                    page_id=page["id"],
                    status="pending",
                    connection=connection,
                )
                if not index_memory:
                    # The Notion value is durably stored. Keep its semantic
                    # projection pending until the user requests repair_index.
                    return identifier
                try:
                    self._index(record, memory_type, page["id"], embedding=embedding)
                except Exception:
                    error = NotionIndexingError(identifier, page["id"])
                else:
                    self._state.put(
                        identifier,
                        memory_type.value,
                        page_id=page["id"],
                        fingerprint=_fingerprint(record),
                        status=self._stored_status(memory_type),
                        connection=connection,
                    )
        if error is not None:
            raise error from None
        return identifier

    def retrieve_by_id(self, id, memory_store_type, *, user_id=_UNSET):
        memory_type = MemoryType(memory_store_type)
        prior = self._state.get(str(id), memory_type.value)
        page = (
            self._get_page(prior["page_id"])
            if prior and prior.get("page_id")
            else self._find_page(str(id), memory_type)
        )
        record = self._decode(page, memory_type, str(id)) if page else None
        if record is None or user_id is not _UNSET and record.get("user_id") != user_id:
            return None
        return record

    def retrieve_by_query(
        self,
        query,
        memory_store_type=None,
        limit=1,
        memory_id=None,
        memory_type=None,
        **kwargs,
    ):
        resolved = MemoryType(memory_type or memory_store_type)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if isinstance(query, str) or isinstance(query, dict) and "embedding" in query:
            return self._search(query, resolved, limit, memory_id=memory_id, **kwargs)
        if not isinstance(query, dict):
            raise ValueError(
                "query must be a dictionary filter or semantic search string"
            )
        return self._records(
            resolved, query=query, limit=limit, memory_id=memory_id, **kwargs
        )

    def _records(
        self,
        memory_type,
        *,
        query=None,
        limit=None,
        user_id=_UNSET,
        memory_id=None,
        thread_id=None,
        namespace=None,
        sorts=None,
        **kwargs,
    ):
        filters = dict(query or {})
        if user_id is not _UNSET:
            filters["user_id"] = user_id
        for key, value in (
            ("memory_id", memory_id),
            ("thread_id", thread_id),
            ("namespace", namespace),
        ):
            if value is not None:
                filters[key] = value
        for key in VECTOR_SCOPE_FIELDS - {
            "user_id",
            "memory_id",
            "thread_id",
            "namespace",
        }:
            if key in kwargs:
                filters[key] = kwargs[key]
        if kwargs.get("statuses") is not None:
            filters["status"] = {"$in": kwargs["statuses"]}
        # Validate operators even when the remote database is empty.
        _validate_filter(filters)
        records = []
        for page in self._pages(
            memory_type,
            query=filters,
            user_id=user_id,
            memory_id=memory_id,
            thread_id=thread_id,
            namespace=namespace,
            limit=limit,
            sorts=sorts,
        ):
            record = self._decode(page, memory_type)
            if record is not None and _matches(record, filters):
                records.append(record)
                if limit is not None and len(records) >= limit:
                    break
        return records

    def _search(self, query, memory_type, limit, *, memory_id=None, **kwargs):
        if self.semantic_provider is None:
            raise NotImplementedError(
                "Configure semantic_provider= to enable Notion semantic search"
            )
        self._require_index_configuration()
        if memory_type not in self.config.semantic_memory_types:
            return []
        if self._state.has_pending(memory_type.value):
            raise NotionError(
                "Semantic index has pending writes; call repair_index() before semantic retrieval"
            )
        if limit > self.config.max_semantic_candidates:
            raise ValueError("Semantic limit exceeds max_semantic_candidates")
        if isinstance(query, dict):
            embedding = validate_vector(query["embedding"])
            kwargs = {**query, **kwargs}
        else:
            if not query.strip():
                return []
            embedding = self.semantic_provider.embed_text(query)
        scope = {
            key: kwargs[key]
            for key in VECTOR_SCOPE_FIELDS
            if key in kwargs
            and (key in {"user_id", "agent_id"} or kwargs[key] is not None)
        }
        if memory_id is not None:
            scope["memory_id"] = memory_id
        if kwargs.get("statuses") is not None:
            scope["status"] = kwargs["statuses"]
        results, seen = [], set()
        size = min(max(limit * 2, 10), self.config.max_semantic_candidates)
        while True:
            hits = self.semantic_provider.query_vectors(
                self._namespace(memory_type),
                embedding,
                limit=size,
                scope=scope,
                include_embedding=bool(kwargs.get("include_embedding")),
            )
            for hit in hits:
                identifier = str(hit["source_id"])
                if identifier in seen:
                    continue
                seen.add(identifier)
                reference = hit.get("metadata") or {}
                if reference.get("semantic_identity") != self._semantic_identity:
                    raise NotionError(
                        "Semantic index model differs from the configured provider; run sync(force=True)"
                    )
                page_id = reference.get("page_id")
                page = self._get_page(page_id) if page_id else None
                # Never return cached vector metadata as if it were memory.
                record = self._decode(page, memory_type, identifier) if page else None
                if record is None or _fingerprint(record) != reference.get(
                    "fingerprint"
                ):
                    continue
                if not all(
                    record.get(key) in value
                    if isinstance(value, list)
                    else record.get(key) == value
                    for key, value in scope.items()
                ):
                    continue
                record["score"] = hit["score"]
                if kwargs.get("include_embedding") and "embedding" in hit:
                    record["embedding"] = hit["embedding"]
                record["source_id"] = identifier
                record["notion"] = {"page_id": page_id, "url": page.get("url")}
                results.append(record)
                if len(results) >= limit:
                    return results
            if len(hits) < size:
                return results
            if size >= self.config.max_semantic_candidates:
                raise NotionQueryLimitError(
                    "Semantic hydration budget exhausted; repair the index or narrow the scope"
                )
            size = min(size * 2, self.config.max_semantic_candidates)

    def retrieve_by_name(self, name, memory_store_type):
        rows = self.retrieve_by_query({"name": name}, memory_store_type, limit=1)
        return rows[0] if rows else None

    def list_all(self, memory_store_type, user_id=_UNSET):
        return self._records(MemoryType(memory_store_type), user_id=user_id)

    def update_by_id(self, id, data, memory_store_type):
        self._require_write()
        if any(key in data and str(data[key]) != str(id) for key in ("id", "_id")):
            raise ValueError("An update cannot change the record's identity")
        return (
            self._store(
                {**data, "id": str(id), "_id": str(id)}, memory_store_type, merge=True
            )
            is not None
        )

    def delete_by_id(self, id, memory_store_type):
        self._require_write()
        memory_type = MemoryType(memory_store_type)
        error, deleted = None, False
        with self._state.writer() as connection:
            prior = self._state.get(str(id), memory_type.value, connection)
            page = (
                self._get_page(prior["page_id"])
                if prior and prior.get("page_id")
                else self._find_page(str(id), memory_type)
            )
            if page and not (page.get("archived") or page.get("in_trash")):
                self._decode(page, memory_type, str(id))
                self._state.put(
                    str(id),
                    memory_type.value,
                    page_id=page["id"],
                    status="delete_uncertain",
                    connection=connection,
                )
                connection.commit()
                self._client.request(
                    "PATCH", "/pages/" + page["id"], body={"in_trash": True}
                )
                deleted = True
            self._state.put(
                str(id),
                memory_type.value,
                page_id=page["id"] if page else None,
                status="delete_pending",
                connection=connection,
            )
            try:
                if self.semantic_provider is not None:
                    self.semantic_provider.delete_vector(
                        self._namespace(memory_type), str(id)
                    )
            except Exception:
                error = NotionIndexingError(str(id), page["id"] if page else None)
            else:
                self._state.put(
                    str(id),
                    memory_type.value,
                    page_id=page["id"] if page else None,
                    status="deleted",
                    connection=connection,
                )
        if error:
            raise error from None
        return deleted

    def delete_by_name(self, name, memory_store_type):
        row = self.retrieve_by_name(name, memory_store_type)
        return self.delete_by_id(row["id"], memory_store_type) if row else False

    def delete_all(self, memory_store_type):
        self._require_write()
        # Resolve the full target set before deleting. A truncated scan must not
        # produce a deceptively successful partial delete_all operation.
        rows = self.list_all(memory_store_type)
        for row in rows:
            self.delete_by_id(row["id"], memory_store_type)
        return True

    def retrieve_conversation_history_ordered_by_timestamp(
        self, memory_id, memory_type=None, limit=None, user_id=_UNSET, thread_id=None
    ):
        rows = self._records(
            MemoryType(memory_type or MemoryType.CONVERSATION_MEMORY),
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )

        def timestamp(row):
            value = row.get("timestamp")
            if isinstance(value, (int, float)):
                return float(value)
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
            except (ValueError, TypeError):
                return 0.0

        rows.sort(key=lambda row: (timestamp(row), row["id"]))
        return rows[-limit:] if limit and limit > 0 else rows

    def list_tool_logs(
        self, *, memory_id=None, user_id=_UNSET, thread_id=None, limit=20
    ):
        rows = self._records(
            MemoryType.TOOL_LOG,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        return filter_tool_log_rows(
            rows, memory_id=memory_id, user_id=user_id, thread_id=thread_id, limit=limit
        )

    def query_memory_observations(
        self,
        memory_store_type,
        *,
        agent_id=None,
        memory_ids=None,
        user_id=_UNSET,
        application_id=None,
        exclude_ids=(),
        limit=200,
    ):
        from ..base import _observation_matches

        safe_limit = max(1, min(int(limit), 1000))
        queries = (
            [
                {"agent_id": str(agent_id)},
                {"agent_id": None, "owner_agent_id": str(agent_id)},
            ]
            if agent_id
            else []
        )
        queries.extend(
            {
                "memory_id": namespace,
                **({"agent_id": None, "owner_agent_id": None} if agent_id else {}),
            }
            for namespace in dict.fromkeys(memory_ids or ())
        )
        rows, seen = [], set()
        for query in queries or [{}]:
            if application_id is not None:
                query["application_id"] = application_id
            if exclude_ids or seen:
                query["id"] = {"$nin": list(set(exclude_ids) | seen)}
            for row in self._records(
                MemoryType(memory_store_type),
                query=query,
                user_id=user_id,
                limit=safe_limit - len(rows),
            ):
                if row["id"] not in seen and _observation_matches(
                    row,
                    agent_id=agent_id,
                    memory_ids=memory_ids,
                    user_id=user_id,
                    application_id=application_id,
                    exclude_ids=exclude_ids,
                ):
                    rows.append(row)
                    seen.add(row["id"])
                    if len(rows) == safe_limit:
                        return rows
        return rows

    def query_observability_records(
        self,
        memory_store_type,
        *,
        agent_ids=None,
        memory_ids=None,
        thread_id=None,
        user_id=_UNSET,
        application_id=None,
        record_type=None,
        tool_name=None,
        success=None,
        event_filters=None,
        start_time=None,
        end_time=None,
        limit=250,
        cursor=None,
    ):
        """One bounded, natively scoped Notion page; no remote list_all scan.

        Post-filters can produce an empty page with a continuation cursor. The
        caller must follow that cursor to examine the rest of the window.
        """
        started = time.perf_counter()
        memory_type = MemoryType(memory_store_type)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        conditions = [self._condition("memory_type", memory_type.value)]
        for key, value in (
            ("user_id", user_id),
            ("thread_id", thread_id),
            ("application_id", application_id),
            ("record_type", record_type),
        ):
            if value is not _UNSET and (key == "user_id" or value is not None):
                conditions.append(self._condition(key, value))
        owners = [
            self._condition("agent_id", value) for value in (agent_ids or []) if value
        ]
        for value in memory_ids or []:
            if value:
                owners.extend(
                    [
                        self._condition("memory_id", value),
                        self._condition("trace_memory_id", value),
                    ]
                )
        if len(owners) > 90:
            raise ValueError(
                "Too many observability scopes; partition agent/memory IDs"
            )
        if owners:
            conditions.append({"or": owners})
        time_bounds = []
        for value, operator in (
            (start_time, "on_or_after"),
            (end_time, "on_or_before"),
        ):
            if value is not None:
                date = _event_time(value)
                if date is None:
                    raise ValueError(
                        "Observability time bounds must be valid timestamps"
                    )
                time_bounds.append((operator, _parse_time(value)))
                conditions.append(
                    {
                        "property": self._properties["event_time"],
                        "date": {operator: date},
                    }
                )
        predicate = {"and": conditions}
        identity = hashlib.sha256(
            _json(
                [
                    self.config.data_source_id,
                    predicate,
                    tool_name,
                    success,
                    event_filters,
                    [(operator, bound.isoformat()) for operator, bound in time_bounds],
                ]
            ).encode()
        ).hexdigest()
        start_cursor, scanned = None, 0
        if cursor:
            try:
                if len(cursor) > 4096:
                    raise ValueError()
                decoded = json.loads(
                    base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
                )
                if (
                    decoded["v"] != 1
                    or decoded["scope"] != identity
                    or not isinstance(decoded["cursor"], str)
                ):
                    raise ValueError()
                start_cursor = decoded["cursor"]
                scanned = decoded["scanned"]
                if type(scanned) is not int or not 0 <= scanned < 10_000:
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    "Invalid or differently scoped Notion observability cursor"
                ) from None
        body = {
            "filter": predicate,
            "page_size": min(limit, 100),
            "sorts": [
                {"property": self._properties["timestamp"], "direction": "descending"},
                {"property": self._properties["id"], "direction": "descending"},
            ],
        }
        if start_cursor:
            body["start_cursor"] = start_cursor
        result = self._client.request(
            "POST", "/data_sources/" + self.config.data_source_id + "/query", body=body
        )
        if (result.get("request_status") or {}).get("type") in {
            "incomplete",
            "partial",
        }:
            raise NotionQueryLimitError(
                "Notion returned an incomplete observability query"
            )
        scanned += len(result.get("results", []))
        if scanned >= 10_000:
            raise NotionQueryLimitError(
                "Notion's 10,000-row limit was reached; partition the observability time window"
            )
        items = []
        for page in result.get("results", []):
            row = self._decode(page, memory_type)
            if row is None or user_id is not _UNSET and row.get("user_id") != user_id:
                continue
            if time_bounds:
                timestamp = _parse_time(row.get("timestamp"))
                if timestamp is None or any(
                    timestamp < bound
                    if operator == "on_or_after"
                    else timestamp > bound
                    for operator, bound in time_bounds
                ):
                    continue
            if thread_id is not None and str(row.get("thread_id") or "") != str(
                thread_id
            ):
                continue
            if record_type is not None and row.get("record_type") != record_type:
                continue
            if (
                application_id is not None
                and row.get("application_id") != application_id
            ):
                continue
            if (
                owners
                and row.get("agent_id") not in (agent_ids or [])
                and (row.get("trace_memory_id") or row.get("memory_id"))
                not in (memory_ids or [])
            ):
                continue
            if tool_name is not None and row.get("tool_name") != tool_name:
                continue
            if success is not None and row.get("success") is not success:
                continue
            if event_filters:
                from ...observability.store import ObservabilityStore

                payload = ObservabilityStore._payload(row) or row
                if any(
                    payload.get(key) != value for key, value in event_filters.items()
                ):
                    continue
            items.append(row)
        next_cursor = None
        if result.get("has_more"):
            next_value = result.get("next_cursor")
            if not next_value or next_value == start_cursor:
                raise NotionIntegrityError(
                    "Notion observability pagination did not advance"
                )
            next_cursor = (
                base64.urlsafe_b64encode(
                    _json(
                        {
                            "v": 1,
                            "scope": identity,
                            "cursor": next_value,
                            "scanned": scanned,
                        }
                    ).encode()
                )
                .decode()
                .rstrip("=")
            )
        return {
            "items": items,
            "next_cursor": next_cursor,
            "truncated": bool(next_cursor),
            "limit": min(limit, 100),
            "scanned_count": len(result.get("results", [])),
            "query_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "freshness": str(items[0].get("timestamp") or "") if items else None,
            "provider_native": True,
        }

    def supports_entity_memory(self):
        return True

    def store_memagent(self, memagent):
        data = (
            memagent.model_dump() if hasattr(memagent, "model_dump") else dict(memagent)
        )
        return self.store(data, MemoryType.MEMAGENT)

    def retrieve_memagent(self, agent_id):
        from ...memagent import MemAgentModel

        record = self.retrieve_by_id(agent_id, MemoryType.MEMAGENT)
        return MemAgentModel(**record) if record else None

    def list_memagents(self):
        from ...memagent import MemAgentModel

        return [MemAgentModel(**row) for row in self.list_all(MemoryType.MEMAGENT)]

    def delete_memagent(self, agent_id, cascade=False):
        if cascade:
            raise NotImplementedError(
                "Notion agent cascade deletion requires explicit scoped record deletion"
            )
        return self.delete_by_id(agent_id, MemoryType.MEMAGENT)

    def update_memagent_memory_ids(self, agent_id, memory_ids):
        return self.update_by_id(
            agent_id, {"memory_ids": list(memory_ids)}, MemoryType.MEMAGENT
        )

    def delete_memagent_memory_ids(self, agent_id):
        return self.update_memagent_memory_ids(agent_id, [])

    def compare_and_swap_shared_memory(self, memory_id, expected_content, content):
        raise NotImplementedError(
            "Notion does not supply the atomic compare-and-swap contract required for concurrent "
            "shared-memory delegation. The semantic provider is vector-only, not a coordination store."
        )

    def index_status(self):
        rows = self._state.rows()
        counts = {}
        tracked_by_type = {memory_type.value: 0 for memory_type in MemoryType}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row["page_id"] and row["status"] not in {
                "deleted",
                "delete_pending",
                "delete_uncertain",
            }:
                tracked_by_type[row["memory_type"]] += 1
        return {
            "data_source_id": self.config.data_source_id,
            "semantic_enabled": self.semantic_provider is not None,
            "tracked_records": len(rows),
            "counts": counts,
            "tracked_by_memory_type": tracked_by_type,
            "state_scope": "local_repair_journal",
            "notion_freshness_verified": False,
            "pending": sum(
                count
                for status, count in counts.items()
                if status
                not in {"indexed", "stored", "deleted", "not_created", "abandoned"}
            ),
        }

    def abandon_pending_create(self, record_id, memory_store_type, *, confirm=False):
        """Explicitly release an unresolved, reference-less create intent.

        This does NOT cancel an accepted Notion request or delete a page. Use
        only after operator investigation, and use a NEW ID for another write.
        A later full sync can discover a delayed Notion page; its data is kept.
        """
        self._require_write()
        if confirm is not True:
            raise ValueError(
                "Abandoning an ambiguous create requires confirm=True after operator investigation"
            )
        memory_type = MemoryType(memory_store_type)
        with self._state.writer() as connection:
            row = self._state.get(str(record_id), memory_type.value, connection)
            if (
                not row
                or row["page_id"]
                or row["status"] not in {"creating", "uncertain"}
            ):
                raise ValueError(
                    "Only an unresolved create without a known page can be abandoned"
                )
            if self._find_page(str(record_id), memory_type) is not None:
                raise ValueError("The Notion page exists; use repair_index() instead")
            self._state.put(
                str(record_id),
                memory_type.value,
                status="abandoned",
                connection=connection,
            )
        return True

    def sync_page(self, page_id, *, force=False):
        """Refresh one accessible managed page after a verified webhook/edit."""
        self._require_write()
        page_id = str(uuid.UUID(str(page_id)))
        with self._state.writer() as connection:
            page = self._get_page(page_id)
            outside_source = page is not None and str(
                (page.get("parent") or {}).get("data_source_id", "")
            ).replace("-", "") != self.config.data_source_id.replace("-", "")
            if (
                page is None
                or page.get("archived")
                or page.get("in_trash")
                or outside_source
            ):
                for row in self._state.rows():
                    if row["page_id"] == page_id:
                        self._remove_synced_vector(row, connection)
                return False
            self._sync_record(page, force=force, connection=connection)
            return True

    def _remove_synced_vector(self, row, connection):
        self._state.put(
            row["source_id"],
            row["memory_type"],
            page_id=row["page_id"],
            status="delete_pending",
            connection=connection,
        )
        connection.commit()
        if self.semantic_provider is not None:
            self.semantic_provider.delete_vector(
                self._namespace(row["memory_type"]), row["source_id"]
            )
        self._state.put(
            row["source_id"],
            row["memory_type"],
            page_id=row["page_id"],
            status="deleted",
            connection=connection,
        )

    def _sync_record(self, page, *, force=False, connection=None):
        if connection is None:
            with self._state.writer() as conn:
                return self._sync_record(page, force=force, connection=conn)
        page_id = page["id"]
        record = self._decode(page)
        memory_type = MemoryType(self._text(page, "memory_type"))
        prior = self._state.get(record["id"], memory_type.value, connection)
        if prior and prior["page_id"] and prior["page_id"] != page_id:
            raise NotionIntegrityError(
                "Duplicate Memorizz IDs refer to different Notion pages"
            )
        fingerprint = _fingerprint(record)
        if (
            not force
            and prior
            and prior["status"] in {"indexed", "stored"}
            and prior["fingerprint"] == fingerprint
        ):
            return False
        if prior and prior["fingerprint"] != fingerprint:
            # A paginated scan may predate an SDK write. Re-read changed rows
            # under the writer lock; unchanged rows need no extra HTTP request.
            latest = self._get_page(page_id)
            outside = (
                latest is not None
                and (latest.get("parent") or {}).get("data_source_id")
                != self.config.data_source_id
            )
            if (
                latest is None
                or latest.get("archived")
                or latest.get("in_trash")
                or outside
            ):
                self._remove_synced_vector(prior, connection)
                return False
            record = self._decode(latest, memory_type, record["id"])
            fingerprint = _fingerprint(record)
        self._state.put(
            record["id"], memory_type.value, page_id=page_id, connection=connection
        )
        connection.commit()
        try:
            self._index(record, memory_type, page_id)
        except Exception:
            raise NotionIndexingError(record["id"], page_id) from None
        self._state.put(
            record["id"],
            memory_type.value,
            page_id=page_id,
            fingerprint=fingerprint,
            status=self._stored_status(memory_type),
            connection=connection,
        )
        return True

    def repair_index(self):
        """Retry pending index work. Ambiguous absent creates remain unresolved."""
        self._require_write()
        repaired, unresolved = 0, 0
        for row in self._state.rows(pending_only=True):
            if row["status"] == "delete_uncertain":
                self.delete_by_id(row["source_id"], row["memory_type"])
                repaired += 1
                continue
            if row["status"] == "delete_pending":
                if self.semantic_provider is not None:
                    self.semantic_provider.delete_vector(
                        self._namespace(row["memory_type"]), row["source_id"]
                    )
                self._state.put(
                    row["source_id"],
                    row["memory_type"],
                    page_id=row["page_id"],
                    status="deleted",
                )
                repaired += 1
                continue
            page_id = row["page_id"]
            if not page_id:
                page = self._find_page(row["source_id"], row["memory_type"])
                page_id = page["id"] if page else None
            if page_id:
                self.sync_page(page_id)
                repaired += 1
            else:
                unresolved += 1
        return {"repaired": repaired, "unresolved": unresolved, **self.index_status()}

    def sync(self, memory_store_type=None, *, force=False):
        """Rebuild/repair from Notion, including human edits and revoked pages.

        This is explicit maintenance, not an unbounded scan in semantic search.
        It raises on incomplete scans; a successful result covers the selected
        memory types only. Unknown/deleted pages never become search results.
        """
        self._require_write()
        types = (
            [MemoryType(memory_store_type)]
            if memory_store_type is not None
            else list(MemoryType)
        )
        refreshed = 0
        seen = set()
        for memory_type in types:
            for page in self._pages(memory_type):
                seen.add(page["id"])
                refreshed += int(self._sync_record(page, force=force))
        for row in self._state.rows():
            if MemoryType(row["memory_type"]) in types and row["status"] != "deleted":
                if row["page_id"] in seen:
                    continue
                if row["page_id"]:
                    self.sync_page(row["page_id"])
        return {
            "refreshed": refreshed,
            "memory_types": [value.value for value in types],
            **self.index_status(),
        }

    def process_webhook(self, body, *, signature, verification_token):
        """Verify a raw Notion webhook and synchronize its latest page snapshot.

        Hosts own HTTPS routing/handshake and should enqueue this work after
        verification. Retried/out-of-order events fetch current Notion state;
        completed event IDs are deduplicated in the bounded local journal.
        """
        self._require_write()
        if not isinstance(body, bytes) or len(body) > 1_000_000:
            raise ValueError("Webhook body must be at most 1 MB of raw bytes")
        if not isinstance(verification_token, str) or not verification_token:
            raise ValueError("A webhook verification token is required")
        expected = (
            "sha256="
            + hmac.new(verification_token.encode(), body, hashlib.sha256).hexdigest()
        )
        if not isinstance(signature, str) or not hmac.compare_digest(
            expected, signature
        ):
            raise PermissionError("Invalid Notion webhook signature")
        try:
            event = json.loads(body)
            event_id = str(uuid.UUID(event["id"]))
        except (ValueError, TypeError, KeyError):
            raise ValueError("Invalid Notion webhook event") from None
        if self._state.webhook_seen(event_id):
            return {"duplicate": True, "event_id": event_id}
        entity = event.get("entity") or {}
        if not isinstance(entity, dict):
            raise ValueError("Invalid Notion webhook entity")
        if entity.get("type") != "page":
            return {"ignored": True, "event_id": event_id}
        try:
            page_id = str(uuid.UUID(entity["id"]))
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ValueError("Invalid Notion webhook page ID") from None
        refreshed = self.sync_page(page_id)
        self._state.complete_webhook(event_id)
        return {"event_id": event_id, "refreshed": refreshed}

    def close(self):
        if self._owns_client:
            self._client.close()
        if self._owns_semantic_provider:
            self.semantic_provider.close()
