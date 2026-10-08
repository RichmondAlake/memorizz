"""Versioned, taxonomy-aligned portable memory archives. No code is executed."""

from __future__ import annotations

import array
import copy
import hashlib
import json
import math
import os
import tempfile
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from .enums.memory_type import MemoryType
from .memory_provider.base import _UNSET
from .redaction import DROP, OMIT, SENSITIVE_KEY_NAMES, KeyMatcher, RedactionPolicy
from .redaction import redact as _redact

FORMAT = "memorizz.memory"
VERSION = 1
MAX_BYTES = 50 * 1024 * 1024
MAX_RECORDS = 100_000
ID_FIELDS = {
    MemoryType.MEMAGENT: "agent_id",
    MemoryType.PERSONAS: "persona_id",
    MemoryType.TOOLBOX: "tool_id",
    MemoryType.SKILLBOX: "skill_id",
    MemoryType.WORKFLOW_MEMORY: "workflow_id",
    MemoryType.SUMMARIES: "summary_id",
    MemoryType.ENTITY_MEMORY: "entity_id",
    MemoryType.TOOL_LOG: "tool_log_id",
    MemoryType.SEMANTIC_CACHE: "cache_key",
}
TAXONOMY = {
    "personas": "long_term.semantic",
    "toolbox": "long_term.procedural",
    "entity_memory": "long_term.semantic",
    "short_term_memory": "short_term",
    "knowledge_base": "long_term.semantic",
    "conversation_memory": "long_term.episodic",
    "workflow_memory": "long_term.procedural",
    "skillbox": "long_term.procedural",
    "agents": "agent_configuration",
    "shared_memory": "coordination",
    "summaries": "long_term.episodic",
    "semantic_cache": "short_term",
    "tool_log": "long_term.episodic",
}
# A credential name as the whole key or after "_"/"-" ("aws_secret_access_key").
_SECRET = KeyMatcher(suffix=SENSITIVE_KEY_NAMES)
_REFERENCES = {
    "_id",
    "id",
    "row_id",
    "record_id",
    "target_record_id",
    "event_id",
    "previous_event_id",
    "agent_id",
    "owner_agent_id",
    "initiator",
    "actor",
    "memory_id",
    "memory_ids",
    "thread_id",
    "thread_ids",
    "run_id",
    "turn_id",
    "root_trace_id",
    "span_id",
    "parent_span_id",
    "source_id",
    "source_ids",
    "source_record_ids",
    "parent_source_id",
    "linked_source_ids",
    "parent_id",
    "supersedes",
    "supersedes_id",
    "superseded_by",
    "duplicate_of",
    "derived_from",
    "source_message_ids",
    "original_memory_ids",
    "message_ids",
    "covered_message_ids",
    "source_workflow_ids",
    "exemplar_workflow_id",
    "promoted_skill_id",
    "knowledge_base_id",
    "knowledge_base_ids",
    "delegates",
    "access_list",
    "participant_agent_ids",
    "participants",
    "assigned_agent_id",
    "completed_by",
    "root_agent_id",
    "delegate_agent_ids",
    "sub_agent_ids",
    "trace_id",
    "session_id",
    *ID_FIELDS.values(),
}


class MemoryArchiveError(ValueError):
    """Invalid/unsupported archive or a conflict discovered before writes."""


def _json(value, depth=0):
    if depth > 64:
        raise MemoryArchiveError("Archive nesting exceeds 64 levels")
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {
            str(k): _json(v, depth + 1) for k, v in value.items() if not callable(v)
        }
    if isinstance(value, (list, tuple, array.array)):
        return [_json(v, depth + 1) for v in value if not callable(v)]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise MemoryArchiveError("Archive numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # BSON ObjectId and UUID are portable strings; executable objects are not.
    if isinstance(value, uuid.UUID) or type(value).__name__ in {"ObjectId", "JsonId"}:
        return str(value)
    raise MemoryArchiveError(f"Unsupported archive value: {type(value).__name__}")


def _schema_property(parent):
    # JSON-schema property names ("properties": {"api_key": {...}}) describe
    # a tool's interface and carry no secret.
    return parent.endswith((".properties", ".$defs"))


def _archive_rule(embeddings):
    def rule(key, value, parent):
        if key == "model" and not parent:
            return DROP  # the root model name is deployment-specific; report it
        if key in {"embedding", "vector"} and not embeddings:
            return OMIT  # vectors are dropped silently unless requested
        return None

    return rule


_SANITIZERS = {
    embeddings: RedactionPolicy(
        keys=_SECRET,
        mode=DROP,
        exempt=_schema_property,
        key_rule=_archive_rule(embeddings),
    )
    for embeddings in (False, True)
}


def _sanitize(value, removals, path="", *, embeddings=False):
    """Drop credential fields (reported into ``removals`` by dotted path)."""
    return _redact(value, _SANITIZERS[embeddings], path=path, on_drop=removals.append)


def _payload(row):
    content = row.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (ValueError, TypeError):
            pass
    if isinstance(content, dict) and str(content.get("record_type", "")).startswith(
        ("memory_history_", "observability_")
    ):
        return content
    return row


def _field(row, key):
    payload = _payload(row)
    # Observability/journal rows use a storage namespace equal to record_id;
    # the actual conversation/tenant identity is in their immutable payload.
    return payload.get(key, row.get(key)) if payload is not row else row.get(key)


def _rewrite(value, mapping, key=""):
    """Rewrite only documented identity/reference fields, never prose."""
    if isinstance(value, dict):
        return {
            (mapping.get(k, k) if key == "thread_titles" else k): _rewrite(
                v, mapping, k
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_rewrite(v, mapping, key) for v in value]
    if isinstance(value, str) and key in _REFERENCES:
        return mapping.get(value, value)
    return value


def _canonical(value):
    # JSON has one number type. In particular JavaScript reserializes 1.0 as
    # 1, 1e-6 as 0.000001, and exponent zero padding differently from Python.
    # Hash their shared numeric representation rather than a language's dumps.
    def encode(item):
        if isinstance(item, dict):
            return (
                "{"
                + ",".join(
                    json.dumps(key, ensure_ascii=False) + ":" + encode(item[key])
                    for key in sorted(item)
                )
                + "}"
            )
        if isinstance(item, list):
            return "[" + ",".join(encode(child) for child in item) + "]"
        if isinstance(item, float):
            if not math.isfinite(item):
                raise MemoryArchiveError("Archive numbers must be finite")
            if item == 0:
                return "0"
            number = repr(item).lower()
            if 1e-6 <= abs(item) < 1e21:
                number = format(Decimal(number), "f")
                return number.rstrip("0").rstrip(".") if "." in number else number
            mantissa, exponent = number.split("e")
            if mantissa.endswith(".0"):
                mantissa = mantissa[:-2]
            exponent = int(exponent)
            return mantissa + "e" + ("+" if exponent >= 0 else "") + str(exponent)
        return json.dumps(item, ensure_ascii=False, allow_nan=False)

    return encode(value).encode("utf-8")


def seal_archive(archive):
    """Recompute integrity after an intentional SDK transformation."""
    result = copy.deepcopy(archive)
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical(result)).hexdigest(),
    }
    return result


def validate_archive(archive, *, max_bytes=MAX_BYTES, max_records=MAX_RECORDS):
    value = _json(archive)
    if (
        not isinstance(value, dict)
        or value.get("format") != FORMAT
        or type(value.get("version")) is not int
        or value.get("version") != VERSION
    ):
        raise MemoryArchiveError("Expected memorizz.memory archive version 1")
    if len(_canonical(value)) > max_bytes:
        raise MemoryArchiveError("Archive exceeds the size limit")
    expected = seal_archive(value)["integrity"]
    if value.get("integrity") != expected:
        raise MemoryArchiveError(
            "Archive checksum does not match; the file may be incomplete or modified"
        )
    stores = value.get("stores")
    if set(value) != {
        "format",
        "version",
        "taxonomy",
        "manifest",
        "stores",
        "integrity",
    }:
        raise MemoryArchiveError("Unsupported archive envelope fields")
    if not isinstance(stores, dict) or set(stores) != set(TAXONOMY):
        raise MemoryArchiveError(
            "Archive stores must contain all 13 canonical taxonomy keys"
        )
    if value.get("taxonomy") != TAXONOMY:
        raise MemoryArchiveError("Archive taxonomy does not match version 1")
    count = 0
    for key, records in stores.items():
        if not isinstance(records, list):
            raise MemoryArchiveError(f"Store {key} must be an array")
        seen = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {"id", "data"}:
                raise MemoryArchiveError("Each archive record needs id and data")
            identifier = record["id"]
            if (
                not isinstance(identifier, str)
                or not identifier
                or len(identifier) > 512
                or identifier in seen
            ):
                raise MemoryArchiveError(f"Invalid or duplicate record ID in {key}")
            if not isinstance(record["data"], dict):
                raise MemoryArchiveError("Record data must be an object")
            natural = ID_FIELDS.get(MemoryType(key))
            if (
                natural
                and record["data"].get(natural) is not None
                and record["data"].get(natural) != identifier
            ):
                raise MemoryArchiveError(
                    "Natural and envelope record identities disagree"
                )
            if (
                record["data"].get("_id", identifier) != identifier
                or record["data"].get("id", identifier) != identifier
            ):
                raise MemoryArchiveError("Record envelope and data identities disagree")
            seen.add(identifier)
            count += 1
            if count > max_records:
                raise MemoryArchiveError("Archive exceeds the record limit")
    manifest = value.get("manifest")
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("record_count")) is not int
        or manifest.get("record_count") != count
        or manifest.get("counts") != {k: len(v) for k, v in stores.items()}
    ):
        raise MemoryArchiveError("Archive manifest counts do not match its records")
    if not {
        "created_at",
        "memorizz_version",
        "source_provider",
        "scope",
        "includes",
        "warnings",
    }.issubset(manifest):
        raise MemoryArchiveError("Archive manifest is incomplete")
    return value


def read_archive(path, *, max_bytes=MAX_BYTES):
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise MemoryArchiveError("Archive exceeds the size limit")
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MemoryArchiveError("Archive is not valid UTF-8 JSON") from exc
    return validate_archive(value, max_bytes=max_bytes)


def archive_json(archive):
    """Readable JSON when it fits, compact JSON near the file-size boundary."""
    data = json.dumps(archive, indent=2, ensure_ascii=False, allow_nan=False)
    if len(data.encode("utf-8")) > MAX_BYTES:
        data = _canonical(archive).decode("utf-8")
    return data


def write_archive(archive, path, *, overwrite=False):
    """Write a private file atomically. Existing files require explicit overwrite."""
    value = validate_archive(archive)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".memorizz-archive-", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(archive_json(value))
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


class MemoryArchive:
    """Export/import the complete selected memory graph through a provider.

    ``user_id`` omitted means trusted SDK administrator; explicit None is the
    anonymous tenant. Network/UI adapters must supply their authorized scope.
    Import defaults to preserving IDs and refusing existing records. A dry run
    validates the entire archive, checks references/conflicts and writes nothing.
    """

    def __init__(self, provider):
        self.provider = provider

    def _rows(self, kind):
        method = getattr(self.provider, "list_archive_records", None)
        rows = method(kind) if callable(method) else self.provider.list_all(kind)
        return [_json(row) for row in (rows or [])]

    def export(
        self,
        *,
        agent_id=None,
        memory_id=None,
        memory_types=None,
        include_delegates=True,
        include_history=True,
        include_context=True,
        include_embeddings=False,
        user_id=_UNSET,
        application_id=None,
        max_records=MAX_RECORDS,
        max_bytes=MAX_BYTES,
    ):
        from . import __version__

        selected = (
            {MemoryType(t) for t in memory_types}
            if memory_types is not None
            else set(MemoryType)
        )
        # Agent configs are needed to resolve private delegate namespaces.
        kinds = selected | ({MemoryType.MEMAGENT} if agent_id else set())
        raw = {kind: self._rows(kind) for kind in kinds}

        def tenant_row(row):
            return (user_id is _UNSET or _field(row, "user_id") == user_id) and (
                application_id is None
                or _field(row, "application_id") == application_id
            )

        # Agent configurations carry no user_id. A tenant-scoped export may
        # include an agent when one of its namespaces or knowledge bases holds
        # a record in the tenant's scope; an unscoped export sees every agent.
        tenant_namespaces, tenant_kb_ids = set(), set()
        for kind, rows in raw.items():
            if kind == MemoryType.MEMAGENT:
                continue
            for row in rows:
                if tenant_row(row):
                    tenant_namespaces.add(
                        _field(row, "memory_id") or row.get("namespace")
                    )
                    tenant_kb_ids.add(row.get("knowledge_base_id"))
        tenant_namespaces.discard(None)
        tenant_kb_ids.discard(None)

        def agent_in_scope(row):
            if (
                application_id is not None
                and _field(row, "application_id") != application_id
            ):
                return False
            if user_id is _UNSET or _field(row, "user_id") == user_id:
                return True
            return bool(
                tenant_namespaces.intersection(row.get("memory_ids") or [])
                or tenant_kb_ids.intersection(row.get("knowledge_base_ids") or [])
            )

        def allowed(row, kind=None):
            return (
                agent_in_scope(row) if kind == MemoryType.MEMAGENT else tenant_row(row)
            )

        agents = {
            str(row.get("agent_id")): row
            for row in raw.get(MemoryType.MEMAGENT, [])
            if agent_in_scope(row)
        }
        agent_ids = {agent_id} if agent_id else set()
        namespaces = {memory_id} if memory_id else set()
        kb_ids = set()
        if agent_id:
            if agent_id not in agents:
                raise MemoryArchiveError("Agent not found in the authorized scope")
            queue = [agent_id]
            while queue:
                current = queue.pop()
                agent = agents[current]
                namespaces.update(agent.get("memory_ids") or [])
                kb_ids.update(agent.get("knowledge_base_ids") or [])
                if include_delegates:
                    for child in agent.get("delegates") or []:
                        if child in agents and child not in agent_ids:
                            agent_ids.add(child)
                            queue.append(child)
        stores = {key: [] for key in TAXONOMY}
        removals, aliases, warnings = [], {}, []
        for kind in sorted(selected, key=lambda t: t.value):
            for row in raw.get(kind, []):
                if not allowed(row, kind):
                    continue
                payload = _payload(row)
                record_type = str(payload.get("record_type") or "")
                if record_type == "memory_history_head":
                    continue  # Derived write cursors are backend-local, not history.
                if not include_history and record_type.startswith("memory_history_"):
                    continue
                if not include_context and record_type.startswith("observability_"):
                    continue
                owner = _field(row, "agent_id") or _field(row, "owner_agent_id")
                namespace = _field(row, "memory_id") or row.get("namespace")
                if agent_id:
                    own = (
                        (row.get("agent_id") in agent_ids)
                        if kind == MemoryType.MEMAGENT
                        else (
                            owner in agent_ids
                            or (not owner and namespace in namespaces)
                            or row.get("knowledge_base_id") in kb_ids
                        )
                    )
                    if not own:
                        # Explicit shared-session participants are part of the team.
                        participants = (
                            payload.get("participant_agent_ids")
                            or payload.get("participants")
                            or []
                        )
                        if not set(participants).intersection(agent_ids):
                            continue
                if memory_id and kind != MemoryType.MEMAGENT and namespace != memory_id:
                    continue
                if (
                    memory_id
                    and kind == MemoryType.MEMAGENT
                    and memory_id not in (row.get("memory_ids") or [])
                ):
                    continue
                key = ID_FIELDS.get(kind)
                identifier = str(
                    (row.get(key) if key else None)
                    or row.get("_id")
                    or row.get("id")
                    or payload.get("record_id")
                    or ""
                )
                if not identifier:
                    raise MemoryArchiveError(
                        f"Provider returned a {kind.value} record without an ID"
                    )
                for alias in (
                    row.get("_id"),
                    row.get("id"),
                    row.get("row_id"),
                    row.get(key) if key else None,
                ):
                    if alias is not None:
                        aliases[str(alias)] = identifier
                data = _sanitize(row, removals, embeddings=include_embeddings)
                if payload is not row:
                    data["content"] = _sanitize(
                        payload, removals, embeddings=include_embeddings
                    )
                elif kind == MemoryType.SHARED_MEMORY and isinstance(
                    data.get("content"), str
                ):
                    try:
                        structured = json.loads(data["content"])
                        if isinstance(structured, (dict, list)):
                            data["content"] = _sanitize(
                                structured, removals, embeddings=include_embeddings
                            )
                    except ValueError:
                        pass
                data["_id"] = data["id"] = identifier
                stores[kind.value].append({"id": identifier, "data": data})
        for records in stores.values():
            for record in records:
                record["data"] = _rewrite(record["data"], aliases)
        available = {r["id"] for records in stores.values() for r in records}
        available.update(
            _payload(r["data"]).get("record_id")
            for records in stores.values()
            for r in records
        )
        external = set()
        for records in stores.values():
            for record in records:
                payload = _payload(record["data"])
                for key in (
                    "delegates",
                    "source_message_ids",
                    "source_record_ids",
                    "supersedes",
                    "supersedes_id",
                    "duplicate_of",
                    "previous_event_id",
                    "target_record_id",
                ):
                    references = payload.get(key) or record["data"].get(key) or []
                    for reference in (
                        references if isinstance(references, list) else [references]
                    ):
                        if isinstance(reference, str) and reference not in available:
                            external.add(reference)
        if external:
            warnings.append(
                f"{len(external)} explicit reference(s) point outside the archive, including deleted or unselected records."
            )
        if removals:
            warnings.append(
                "Credential fields and executable objects were omitted; reconnect providers/tools after restore."
            )
        warnings.append(
            "History contains recorded metadata and hashes, not deleted or historical memory values. Provider-specific projections may change version hashes after restore."
        )
        warnings.append(
            "This is a memory archive, not a database snapshot: harness SQLite runs, attachments and external services are not included. Concurrent writes can occur during export."
        )
        archive = seal_archive(
            {
                "format": FORMAT,
                "version": VERSION,
                "taxonomy": TAXONOMY,
                "manifest": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "memorizz_version": __version__,
                    "source_provider": type(self.provider).__name__,
                    "scope": {
                        "agent_id": agent_id,
                        "memory_id": memory_id,
                        "agent_ids": sorted(agent_ids),
                        "user_id": None if user_id is _UNSET else user_id,
                        "user_scoped": user_id is not _UNSET,
                        "application_id": application_id,
                    },
                    "counts": {k: len(v) for k, v in stores.items()},
                    "record_count": sum(len(v) for v in stores.values()),
                    "includes": {
                        "delegates": include_delegates,
                        "history": include_history,
                        "context": include_context,
                        "embeddings": include_embeddings,
                    },
                    "omitted_fields": sorted(set(removals)),
                    "external_references": sorted(external),
                    "warnings": warnings,
                },
                "stores": stores,
            }
        )
        return validate_archive(archive, max_bytes=max_bytes, max_records=max_records)

    def import_archive(
        self,
        archive,
        *,
        dry_run=True,
        conflict="error",
        id_strategy="preserve",
        target_memory_id=None,
        user_id=_UNSET,
        application_id=None,
        allowed_types=None,
        reembed=False,
    ):
        value = validate_archive(archive)
        if conflict not in {"error", "skip", "replace"} or id_strategy not in {
            "preserve",
            "new",
        }:
            raise MemoryArchiveError(
                "Choose conflict=error/skip/replace and id_strategy=preserve/new"
            )
        if getattr(self.provider, "read_only", False) and not dry_run:
            raise MemoryArchiveError("The target provider is read-only")
        allowed_types = (
            set(TAXONOMY)
            if allowed_types is None
            else {MemoryType(t).value for t in allowed_types}
        )
        mapping, rows, conflicts, skipped = {}, [], [], []
        warnings = list(value["manifest"].get("warnings") or []) + [
            "Import does not restore embeddings unless reembed=True. Restore is not atomic; a provider error may leave partial progress."
        ]
        original_rows = {}
        for kind, records in value["stores"].items():
            if records and kind not in allowed_types:
                raise MemoryArchiveError(
                    f"Importing {kind} is not authorized on this interface"
                )
            for record in records:
                data = record["data"]
                if user_id is not _UNSET and _field(data, "user_id") != user_id:
                    raise MemoryArchiveError(
                        "Archive contains records outside the authorized user scope"
                    )
                if (
                    application_id is not None
                    and _field(data, "application_id") != application_id
                ):
                    raise MemoryArchiveError(
                        "Archive contains records outside the authorized application scope"
                    )
                if id_strategy == "new":
                    mapping.setdefault(record["id"], str(uuid.uuid4()))

                    # Include journal logical IDs and each namespace, not prose.
                    def collect(node):
                        if isinstance(node, dict):
                            for key, item in node.items():
                                if (
                                    key
                                    in {
                                        "record_id",
                                        "event_id",
                                        "target_record_id",
                                        "memory_id",
                                        "thread_id",
                                        "run_id",
                                        "turn_id",
                                        "root_trace_id",
                                        "span_id",
                                        "trace_id",
                                        "session_id",
                                        "knowledge_base_id",
                                        "persona_id",
                                    }
                                    and isinstance(item, str)
                                    and item
                                    and not (key == "record_id" and "relation" in node)
                                ):
                                    mapping.setdefault(item, str(uuid.uuid4()))
                                if key in {
                                    "memory_ids",
                                    "knowledge_base_ids",
                                } and isinstance(item, list):
                                    for item_id in item:
                                        if isinstance(item_id, str):
                                            mapping.setdefault(
                                                item_id, str(uuid.uuid4())
                                            )
                                collect(item)
                        elif isinstance(node, list):
                            for item in node:
                                collect(item)

                    collect(data)
        if target_memory_id:
            namespaces = {
                str(_field(r["data"], "memory_id"))
                for records in value["stores"].values()
                for r in records
                if _field(r["data"], "memory_id")
            }
            if len(namespaces) > 1:
                raise MemoryArchiveError(
                    "--target-memory-id requires a single-namespace archive; cloning preserves separate delegate namespaces"
                )
            mapping.update({namespace: target_memory_id for namespace in namespaces})
        normalize = getattr(self.provider, "archive_target_id", None)
        if callable(normalize):
            for key, records in value["stores"].items():
                for record in records:
                    desired = mapping.get(record["id"], record["id"])
                    converted = normalize(MemoryType(key), desired)
                    if converted != desired:
                        mapping[record["id"]] = converted
        for kind, records in value["stores"].items():
            existing = (
                {
                    self._identity(MemoryType(kind), r): r
                    for r in self._rows(MemoryType(kind))
                }
                if records
                else {}
            )
            for record in records:
                identifier = mapping.get(record["id"], record["id"])
                data = _rewrite(copy.deepcopy(record["data"]), mapping)
                # Known credential keys are never reintroduced from handcrafted archives.
                data = _sanitize(data, [], embeddings=False)
                data["_id"] = data["id"] = identifier
                if kind == "agents":
                    data["agent_id"] = identifier
                    data["automations_enabled"] = False
                old = existing.get(identifier)
                original_rows[(kind, identifier)] = old
                if old is not None:
                    if (
                        user_id is not _UNSET and _field(old, "user_id") != user_id
                    ) or (
                        application_id is not None
                        and _field(old, "application_id") != application_id
                    ):
                        raise MemoryArchiveError(
                            "A destination ID belongs to another scope; use new IDs"
                        )
                    conflicts.append({"memory_type": kind, "id": identifier})
                    if conflict == "skip":
                        skipped.append({"memory_type": kind, "id": identifier})
                        continue
                rows.append((MemoryType(kind), identifier, data, old is not None))
        available_agents = {
            identifier for kind, identifier, _, _ in rows if kind == MemoryType.MEMAGENT
        }
        available_agents.update(
            self._identity(MemoryType.MEMAGENT, r)
            for r in self._rows(MemoryType.MEMAGENT)
            if (user_id is _UNSET or _field(r, "user_id") == user_id)
        )
        for kind, identifier, data, _ in rows:
            validate = getattr(self.provider, "validate_archive_record", None)
            if callable(validate):
                validate(kind, identifier, data)
            if kind == MemoryType.MEMAGENT:
                from .memagent.models import MemAgentModel

                MemAgentModel(**data)
                missing = set(data.get("delegates") or []) - available_agents
                if missing:
                    raise MemoryArchiveError(
                        "Archive has unavailable delegate references; export with delegates or restore them first"
                    )
        counts = {
            key: sum(1 for kind, *_ in rows if kind.value == key) for key in TAXONOMY
        }
        report = {
            "ok": not (conflicts and conflict == "error"),
            "dry_run": dry_run,
            "format": FORMAT,
            "version": VERSION,
            "planned": len(rows),
            "imported": 0,
            "skipped": len(skipped),
            "conflicts": conflicts,
            "counts": counts,
            "id_map": mapping,
            "warnings": warnings,
            "atomic": False,
            "errors": [],
        }
        if conflicts and conflict == "error":
            if dry_run:
                return report
            raise MemoryArchiveError(
                "Destination records already exist; choose skip, replace, or new IDs"
            )
        if dry_run:
            return report
        from .memory_history import (
            _emit_change,
            _suspend_recording,
            memory_change_context,
        )

        ordered = sorted(
            rows,
            key=lambda r: (
                r[0] != MemoryType.MEMAGENT,
                r[0] == MemoryType.SUMMARIES,
                r[0] == MemoryType.SHARED_MEMORY,
            ),
        )
        method = getattr(self.provider, "store_archive_record", None)
        with memory_change_context(source="archive_import"), _suspend_recording(
            self.provider
        ):
            for kind, identifier, data, replace in ordered:
                try:
                    first = (
                        {**data, "delegates": [], "persona": None}
                        if kind == MemoryType.MEMAGENT
                        else data
                    )
                    if callable(method):
                        stored = method(
                            kind, identifier, first, replace=replace, reembed=reembed
                        )
                    else:
                        stored = self.provider.store(first, memory_store_type=kind)
                    if str(stored) != identifier:
                        raise MemoryArchiveError(
                            "Provider changed the planned record ID"
                        )
                    report["imported"] += 1
                    if getattr(self.provider, "_memory_history_enabled", False):
                        try:
                            if kind != MemoryType.MEMAGENT:
                                _emit_change(
                                    self.provider,
                                    kind,
                                    identifier,
                                    original_rows[(kind.value, identifier)],
                                    first,
                                    "archive_import",
                                )
                        except Exception:
                            report["warnings"].append(
                                "A record was restored but its import history event could not be recorded."
                            )
                except Exception as exc:
                    report["ok"] = False
                    report["errors"].append(
                        {
                            "memory_type": kind.value,
                            "id": identifier,
                            "error_type": type(exc).__name__,
                        }
                    )
                    break  # Report committed progress; never claim a rollback.
            if report["ok"]:
                for kind, identifier, data, _ in ordered:
                    if kind == MemoryType.MEMAGENT:
                        try:
                            if callable(method):
                                method(
                                    kind, identifier, data, replace=True, reembed=False
                                )
                            else:
                                self.provider.store(data, memory_store_type=kind)
                            if getattr(self.provider, "_memory_history_enabled", False):
                                try:
                                    _emit_change(
                                        self.provider,
                                        kind,
                                        identifier,
                                        original_rows[(kind.value, identifier)],
                                        data,
                                        "archive_import",
                                    )
                                except Exception:
                                    report["warnings"].append(
                                        "An agent was restored but its import history event could not be recorded."
                                    )
                        except Exception as exc:
                            report["ok"] = False
                            report["errors"].append(
                                {
                                    "memory_type": kind.value,
                                    "id": identifier,
                                    "error_type": type(exc).__name__,
                                }
                            )
                            break
        return report

    @staticmethod
    def _identity(kind, row):
        key = ID_FIELDS.get(kind)
        return str(
            (row.get(key) if key else None) or row.get("_id") or row.get("id") or ""
        )

    def export_file(self, path, *, overwrite=False, **options):
        archive = self.export(**options)
        write_archive(archive, path, overwrite=overwrite)
        return archive["manifest"]

    def import_file(self, path, **options):
        return self.import_archive(read_archive(path), **options)
