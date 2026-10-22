# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Connecting the UI to a memory store, and describing that connection.

One code path connects for the connect form and at startup (from
``memorizz ui`` flags or the environment), so validation, read-only handling
and the client record are the same either way. The header on every page
shows the connection through ``connection_header``: the provider, a short
target with no credentials in it, and whether the UI is read-only.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .._env_io import resolve_oracle_in_database_embedding_from_env
from ..redaction import URI_CREDENTIALS, scrub_text
from .security import ReadOnlyProviderProxy, ui_read_only
from .state import _state, close_meta_harness

logger = logging.getLogger(__name__)

PROVIDER_TYPES = ("oracle", "mongodb", "filesystem", "notion")
PROVIDER_LABELS = {
    "oracle": "Oracle",
    "mongodb": "MongoDB",
    "filesystem": "FileSystem",
    "notion": "Notion",
}

# The connect form's fields. The password, URI and token are secrets: they
# are used to connect and never stored, logged or rendered.
CONNECT_FIELDS = (
    "oracle_user",
    "oracle_password",
    "oracle_dsn",
    "oracle_schema",
    "mongodb_uri",
    "mongodb_db_name",
    "filesystem_path",
    "notion_token",
    "notion_data_source_id",
    "notion_semantic_backend",
)
SECRET_FIELDS = frozenset({"oracle_password", "mongodb_uri", "notion_token"})

# Writes a read-only UI still accepts: signing in, connecting, and the trace
# reveal steps (which only read). The middleware refuses every other write;
# the page script leaves forms posting here enabled.
READ_ONLY_ALLOWED_POSTS = frozenset(
    {"/login", "/connect", "/traces/account/resolve", "/traces/reveal"}
)
READ_ONLY_TOOLTIP = "The UI is read-only. Changes are turned off."


def ui_is_read_only() -> bool:
    """The process-wide read-only switch, or a read-only connection."""
    return ui_read_only() or bool(_state.get("read_only"))


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


def _clean(params: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {}
    for key in CONNECT_FIELDS:
        value = params.get(key)
        values[key] = str(value).strip() if value not in (None, "") else None
    return values


def build_provider(
    provider_type: str, params: Mapping[str, Any], *, read_only: bool
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """``(provider, connection_info, provider_secrets)`` for one store.

    Blank secrets fall back to the environment (ORACLE_PASSWORD, MONGODB_URI,
    NOTION_TOKEN), as the connect form always has.
    """
    values = _clean(params)
    if provider_type == "oracle":
        user = values["oracle_user"]
        dsn = values["oracle_dsn"]
        password = values["oracle_password"] or os.environ.get("ORACLE_PASSWORD", "")
        if not all([user, password, dsn]):
            raise ValueError("Oracle requires user, password, and DSN")
        from ..memory_provider.oracle import OracleConfig, OracleProvider

        schema = values["oracle_schema"] or user
        in_database = resolve_oracle_in_database_embedding_from_env()
        config = OracleConfig(
            user=user,
            password=password,
            dsn=dsn,
            schema=schema,
            lazy_vector_indexes=True,
            in_database_embedding=in_database,
            # Read-only makes no schema changes: no tables, indexes or ONNX
            # model are created.
            read_only=read_only,
        )
        provider = OracleProvider(config)
        info = {
            "user": user,
            "dsn": dsn,
            "schema": schema,
            "embedding": _oracle_embedding_description(config),
        }
        secrets = {
            "oracle_user": user,
            "oracle_password": password,
            "oracle_dsn": dsn,
            "oracle_schema": schema,
        }
        return provider, info, secrets

    if provider_type == "mongodb":
        uri = values["mongodb_uri"] or os.environ.get("MONGODB_URI", "")
        if not uri:
            raise ValueError("MongoDB requires a URI")
        from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

        db_name = values["mongodb_db_name"] or "memorizz"
        provider = MongoDBProvider(
            MongoDBConfig(
                uri=uri,
                db_name=db_name,
                lazy_vector_indexes=True,
                read_only=read_only,
            )
        )
        return provider, {"uri": mask_uri(uri), "db_name": db_name}, {}

    if provider_type == "filesystem":
        path = values["filesystem_path"]
        if not path:
            raise ValueError("FileSystem requires a path")
        from ..memory_provider.filesystem import FileSystemConfig, FileSystemProvider

        provider = FileSystemProvider(FileSystemConfig(root_path=path))
        return provider, {"path": path}, {}

    if provider_type == "notion":
        from ..memory_provider.notion.factory import create_notion_provider_from_env

        backend = values["notion_semantic_backend"]
        provider = create_notion_provider_from_env(
            token=values["notion_token"],
            data_source_id=values["notion_data_source_id"],
            semantic_backend=backend,
            read_only=read_only,
        )
        info = {
            "data_source_id": provider.config.data_source_id,
            "semantic_backend": backend
            or os.getenv("MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem"),
        }
        return provider, info, {}

    raise ValueError(f"Unknown provider type: {provider_type}")


def _oracle_embedding_description(config: Any) -> str:
    if config.in_database_embedding:
        return "Oracle in-database"
    if config.embedding_provider:
        dimensions = (config.embedding_config or {}).get("dimensions")
        text = str(config.embedding_provider)
        return f"{text} ({dimensions} dimensions)" if dimensions else text
    return "Disabled"


def connect_provider(
    provider_type: str,
    params: Mapping[str, Any],
    *,
    read_only: bool,
    port: Optional[int] = None,
    source: str = "form",
) -> Any:
    """Connect the UI to one store; raise on failure, leaving the current
    connection in place. ``source`` is ``form`` or ``startup``."""
    from .embedding_check import check_store_embeddings, embedder_label

    provider, info, secrets = build_provider(provider_type, params, read_only=read_only)
    # Before the read-only proxy: adopting the store's embedder rebinds the
    # provider itself (in this process only; nothing is written).
    check = check_store_embeddings(provider, provider_type)
    if check.get("status") == "adopted" and "embedding" in info:
        info["embedding"] = embedder_label(check.get("adopted") or {})
    if read_only:
        provider = ReadOnlyProviderProxy(provider)
        info["access"] = "read-only"

    if _state["provider"]:
        close_meta_harness()
        try:
            _state["provider"].close()
        except Exception:
            pass

    _state["provider"] = provider
    _state["provider_type"] = provider_type
    _state["connection_info"] = info
    _state["provider_secrets"] = secrets
    _state["read_only"] = read_only
    _state["embedding_check"] = check
    _state["schema_note"] = schema_note(provider)
    _state["startup_connect"] = None
    logger.info(
        "Connected to %s provider (%s)",
        provider_type,
        "startup" if source == "startup" else "connect form",
    )
    # Other MemoRizz processes can then see this UI uses the store.
    from ..finetune import clients

    clients.register("ui", provider, port=port)
    return provider


def schema_note(provider: Any) -> Optional[str]:
    """What a read-only connection left missing, since it creates nothing."""
    missing = list(getattr(provider, "missing_tables", None) or [])
    if not missing:
        return None
    shown = ", ".join(missing[:6]) + (", …" if len(missing) > 6 else "")
    return (
        f"This schema lacks {len(missing)} table{'' if len(missing) == 1 else 's'} "
        f"this version of MemoRizz uses ({shown}). A read-only connection "
        "doesn't create them, so the pages that read them stay empty. "
        "Connect once without read-only, or run memorizz oracle setup, to add them."
    )


def disconnect_provider() -> None:
    close_meta_harness()
    if _state["provider"]:
        try:
            _state["provider"].close()
        except Exception:
            pass
    _state["provider"] = None
    _state["provider_type"] = None
    _state["connection_info"] = {}
    _state["provider_secrets"] = {}
    _state["read_only"] = False
    _state["embedding_check"] = None
    _state["schema_note"] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def format_connect_error(provider_type: Optional[str], exc: BaseException) -> str:
    """A connect failure as the connect page and the CLI show it."""
    from ..extras import missing_extra_message
    from .embedding_check import oracle_mismatch_error

    hint = missing_extra_message(exc)
    if hint:
        return hint
    message = scrub_text(
        str(exc).strip(),
        ((URI_CREDENTIALS, lambda match: f"{match.group('scheme')}[redacted]@"),),
    )
    mismatch = oracle_mismatch_error(message)
    if mismatch:
        return mismatch
    if "install_if_missing is disabled" in message and ui_is_read_only():
        return (
            f"{message} A read-only UI does not load an embedding model into the "
            "database. Set MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING=false and "
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER, _MODEL and _DIMENSIONS to the "
            "model that wrote this schema, then connect again."
        )
    if provider_type == "oracle":
        return _format_oracle_connect_error(message)
    return message or "Connection failed."


def _format_oracle_connect_error(message: str) -> str:
    """Map common Oracle connection failures to helpful messages."""
    lowered = message.lower()
    if "oracledb package is required" in lowered or "pip install oracledb" in lowered:
        from ..extras import missing_extra_message

        return missing_extra_message(ModuleNotFoundError(name="oracledb")) or message
    if "dpi-1047" in lowered or "oracle client library" in lowered:
        return (
            "Oracle client libraries are missing. Install Oracle Instant Client or "
            f"configure python-oracledb to use thin mode. Details: {message}"
        )
    connection_markers = (
        "ora-12541",
        "ora-12514",
        "ora-12170",
        "ora-12545",
        "ora-12543",
        "no listener",
        "connection refused",
        "econnrefused",
        "timed out",
        "dpy-6005",
    )
    if any(marker in lowered for marker in connection_markers):
        return (
            "Cannot reach Oracle at the DSN. Ensure the database is running and "
            "the host/port/service are correct. If using Docker, start the "
            f"container and confirm port 1521 is mapped. Details: {message}"
        )
    return message or "Oracle connection failed."


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _env(environ: Mapping[str, str], key: str) -> str:
    return str(environ.get(key, "") or "").strip()


def resolve_startup_connection(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """The store the environment names, for connecting at startup.

    ``MEMORIZZ_BACKEND`` picks the provider and its own variables complete it
    (ORACLE_USER, ORACLE_PASSWORD and ORACLE_DSN; MONGODB_URI; the CLI's
    FileSystem store, MEMORIZZ_MEMORY_ROOT or ~/.memorizz/memory; the Notion
    data source and token). Without it, a complete Oracle login, then a
    MongoDB URI, then MEMORIZZ_MEMORY_ROOT, is used, in the automations
    worker's order. Returns None when
    the environment names nothing, or ``{"provider_type", "error"}`` when it
    names a provider without everything it needs.
    ``MEMORIZZ_UI_AUTO_CONNECT=false`` turns this off.
    """
    env = os.environ if environ is None else environ
    if _env(env, "MEMORIZZ_UI_AUTO_CONNECT").lower() in {"0", "false", "no", "off"}:
        return None
    backend = _env(env, "MEMORIZZ_BACKEND").lower()
    if backend and backend not in PROVIDER_TYPES:
        return {
            "provider_type": None,
            "source": "environment",
            "error": (
                f"MEMORIZZ_BACKEND={backend} is not a memory provider "
                "(oracle, mongodb, filesystem or notion)."
            ),
            "params": {},
        }
    oracle_keys = ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN")
    if not backend:
        if all(_env(env, key) for key in oracle_keys):
            backend = "oracle"
        elif _env(env, "MONGODB_URI"):
            backend = "mongodb"
        elif _env(env, "MEMORIZZ_MEMORY_ROOT"):
            backend = "filesystem"
        else:
            return None

    required = {
        "oracle": oracle_keys,
        "mongodb": ("MONGODB_URI",),
        "filesystem": (),
        "notion": ("MEMORIZZ_NOTION_DATA_SOURCE_ID", "NOTION_TOKEN"),
    }[backend]
    missing = [key for key in required if not _env(env, key)]
    if missing:
        return {
            "provider_type": backend,
            "source": "environment",
            "error": (
                f"MEMORIZZ_BACKEND is {backend}, but "
                + " and ".join(missing)
                + (" is" if len(missing) == 1 else " are")
                + " not set, so the UI did not connect at startup."
            ),
            "params": {},
        }

    params: Dict[str, Any] = {}
    if backend == "oracle":
        params = {
            "oracle_user": _env(env, "ORACLE_USER"),
            "oracle_password": _env(env, "ORACLE_PASSWORD"),
            "oracle_dsn": _env(env, "ORACLE_DSN"),
            "oracle_schema": _env(env, "ORACLE_SCHEMA") or None,
        }
    elif backend == "mongodb":
        params = {
            "mongodb_uri": _env(env, "MONGODB_URI"),
            "mongodb_db_name": _env(env, "MONGODB_DB_NAME") or None,
        }
    elif backend == "filesystem":
        from .._env_io import memory_root

        root = _env(env, "MEMORIZZ_MEMORY_ROOT") or str(memory_root())
        params = {"filesystem_path": root}
    elif backend == "notion":
        params = {
            "notion_data_source_id": _env(env, "MEMORIZZ_NOTION_DATA_SOURCE_ID"),
            "notion_token": _env(env, "NOTION_TOKEN"),
            "notion_semantic_backend": _env(env, "MEMORIZZ_NOTION_SEMANTIC_BACKEND")
            or None,
        }
    return {"provider_type": backend, "source": "environment", "params": params}


def connect_at_startup(
    startup: Dict[str, Any], *, read_only: bool, port: Optional[int] = None
) -> Optional[str]:
    """Connect from ``startup`` (flags or the environment). Returns None on
    success, else the reason, which the connect page then shows. The
    secrets in ``startup`` are dropped either way."""
    provider_type = startup.get("provider_type")
    params = dict(startup.get("params") or {})
    for key in SECRET_FIELDS:
        (startup.get("params") or {}).pop(key, None)
    error = startup.get("error")
    if not error:
        try:
            connect_provider(
                provider_type, params, read_only=read_only, port=port, source="startup"
            )
            return None
        except Exception as exc:
            logger.warning("Startup connection failed: %s", type(exc).__name__)
            error = format_connect_error(provider_type, exc)
    _state["startup_connect"] = {
        "provider_type": provider_type,
        "source": startup.get("source") or "environment",
        "error": error,
        # Non-secret fields pre-fill the connect form.
        "fields": {
            key: value
            for key, value in params.items()
            if key not in SECRET_FIELDS and value
        },
    }
    return error


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def mask_uri(uri: str) -> str:
    """A connection URI with its credentials and query string removed:
    ``mongodb+srv://***@cluster0.example.net/memorizz``."""
    text = str(uri or "").strip()
    scheme, sep, rest = text.partition("://")
    if not sep:
        scheme, rest = "", text
    rest = rest.split("?", 1)[0].split("#", 1)[0]
    hidden = ""
    at = rest.rfind("@")
    if at >= 0:
        # The last @ ends the credentials, even when a password holds a /.
        rest, hidden = rest[at + 1 :], "***@"
    return f"{scheme}{sep}{hidden}{rest}"


def _shorten(text: str, limit: int = 56) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def oracle_dsn_target(dsn: str) -> str:
    """``host:port/service`` from an EZConnect string or a descriptor."""
    text = str(dsn or "").strip()
    if text.startswith("("):
        host = re.search(r"\(\s*HOST\s*=\s*([^)\s]+)", text, re.I)
        port = re.search(r"\(\s*PORT\s*=\s*(\d+)", text, re.I)
        service = re.search(r"\(\s*(?:SERVICE_NAME|SID)\s*=\s*([^)\s]+)", text, re.I)
        target = host.group(1) if host else "descriptor"
        if port:
            target += f":{port.group(1)}"
        if service:
            target += f"/{service.group(1)}"
        return target
    text = re.sub(r"^[a-z]+://", "", text, flags=re.I)
    text = text.split("?", 1)[0]
    if "@" in text:
        text = text.rsplit("@", 1)[-1]  # user/password@host/service
    return text


def mongodb_target(uri: str, db_name: Optional[str]) -> str:
    """``host/db``: credentials, extra hosts and options left out."""
    text = str(uri or "").strip()
    text = re.sub(r"^[a-z+]+://", "", text, flags=re.I)
    text = text.split("?", 1)[0]
    text = text[text.rfind("@") + 1 :]
    authority = text.split("/", 1)[0]
    hosts = [host for host in authority.split(",") if host]
    target = hosts[0] if hosts else "mongodb"
    if len(hosts) > 1:
        target += f" +{len(hosts) - 1}"
    return f"{target}/{db_name}" if db_name else target


def filesystem_target(path: str) -> str:
    """``~/…/memorizz/memory`` for a long path under the home directory."""
    from .local_stores import display_path

    try:
        shown = display_path(Path(str(path)).expanduser().resolve())
    except Exception:
        shown = str(path)
    parts = shown.split("/")
    if len(shown) > 32 and len(parts) > 3:
        head = parts[0] or "/"
        shown = f"{head.rstrip('/')}/…/" + "/".join(parts[-2:])
    return shown


def connection_target(
    provider_type: Optional[str], info: Mapping[str, Any]
) -> Tuple[str, str]:
    """``(short target, longer tooltip)`` for the header. Never a secret."""
    info = info or {}
    if provider_type == "oracle":
        user = str(info.get("user") or "")
        target = oracle_dsn_target(str(info.get("dsn") or ""))
        schema = str(info.get("schema") or "")
        short = f"{user}@{target}" if user else target
        title = f"Oracle {short}"
        if schema:
            title += f", schema {schema}"
            if schema.upper() != user.upper():
                short += f" · {schema}"
        return _shorten(short), title
    if provider_type == "mongodb":
        short = mongodb_target(str(info.get("uri") or ""), info.get("db_name"))
        return _shorten(short), f"MongoDB {short}"
    if provider_type == "filesystem":
        path = str(info.get("path") or "")
        short = filesystem_target(path) if path else "local store"
        from .local_stores import display_path

        try:
            full = display_path(Path(path).expanduser().resolve())
        except Exception:
            full = path
        return _shorten(short), f"FileSystem {full}"
    if provider_type == "notion":
        source = str(info.get("data_source_id") or "")
        backend = str(info.get("semantic_backend") or "")
        short = f"data source {source[:8]}…" if len(source) > 8 else source
        title = f"Notion data source {source}"
        if backend:
            short += f" · {backend} vectors"
            title += f", {backend} vectors"
        return _shorten(short), title
    return "", ""


def connection_header() -> Dict[str, Any]:
    """What the header on every page shows about the connection.

    Accounts restricted to trace inspection see the provider only: no host,
    user or path, and no link to the connect page they cannot open.
    """
    from .trace_access import current_principal

    provider_type = _state.get("provider_type")
    connected = _state.get("provider") is not None and bool(provider_type)
    label = PROVIDER_LABELS.get(str(provider_type), str(provider_type or ""))
    restricted = current_principal.get().restricted
    target, title = (
        connection_target(provider_type, _state.get("connection_info") or {})
        if connected and not restricted
        else ("", label)
    )
    check = _state.get("embedding_check") or {}
    return {
        "connected": connected,
        "provider_type": provider_type,
        "label": label,
        "target": target,
        "title": title,
        "href": None if restricted else "/connect",
        "read_only": ui_is_read_only(),
        "read_only_tooltip": READ_ONLY_TOOLTIP,
        "embedding_warning": (
            check.get("message")
            if check.get("status") == "mismatch" and not restricted
            else None
        ),
        # Writes the middleware still accepts while read-only; a host that
        # delegates persona actions accepts those too.
        "allowed_posts": sorted(
            READ_ONLY_ALLOWED_POSTS
            | (
                {"/persona-evolution/action"}
                if _state.get("persona_evolution_actions")
                else set()
            )
        ),
    }


__all__ = [
    "CONNECT_FIELDS",
    "PROVIDER_LABELS",
    "READ_ONLY_ALLOWED_POSTS",
    "READ_ONLY_TOOLTIP",
    "build_provider",
    "connect_at_startup",
    "connect_provider",
    "connection_header",
    "connection_target",
    "disconnect_provider",
    "format_connect_error",
    "mask_uri",
    "resolve_startup_connection",
    "ui_is_read_only",
]
