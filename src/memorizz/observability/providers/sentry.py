"""Sentry as an agent's observability provider.

Each finished turn is sent as one Sentry transaction, ``invoke_agent <name>``,
whose spans follow Sentry's agent conventions: ``gen_ai.chat`` for a model
call, ``gen_ai.execute_tool`` for a tool call, and ``memorizz.*`` for memory
reads and the other steps MemoRizz records. The Sentry trace id is the turn's
``root_trace_id`` without its dashes, so a MemoRizz trace and its Sentry trace
name each other. A turn that failed is also reported as an error event on the
same trace, which is what Sentry alerts on.

The DSN is read from ``MEMORIZZ_SENTRY_DSN`` or ``SENTRY_DSN``. MemoRizz sends
through a client of its own and never calls ``sentry_sdk.init``, so a host
application's Sentry setup is left alone; when no DSN is set and the host has
initialised Sentry, the host's client is used.

With ``SENTRY_SPOTLIGHT`` set, which ``sentry local run -- <command>`` does,
turns also go to Sentry's local development server. That needs no DSN and no
Sentry account: without a DSN the turns go to the local server only.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
import random
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlsplit

from .base import ObservabilityProvider

logger = logging.getLogger(__name__)

DSN_ENV = ("MEMORIZZ_SENTRY_DSN", "SENTRY_DSN")
LOCAL_ENV = "SENTRY_SPOTLIGHT"
LOCAL_DEFAULT_URL = "http://localhost:8969/stream"
#: Sentry's page that shows what a local server received (the address
#: ``sentry local`` prints).
LOCAL_UI = "https://local.sentry.dev/"
#: What a turn is marked with when it went to the local server only.
LOCAL_MARK = "sentry-local"
ORG_ENV = "SENTRY_ORG"
ORIGIN = "auto.ai.memorizz"
MAX_SPANS = 1000
_CONTENT_CHARS = 4000

_FAILED = {"error", "failed", "failure", "timeout", "deadline_exceeded"}
_NOT_FAILURES = {"cancelled", "canceled", "approval_required", "paused"}
_STATUS = {
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "timeout": "deadline_exceeded",
    "deadline_exceeded": "deadline_exceeded",
}
# Names the turn, not what happened in it: these travel as trace context or
# not at all.
_IDENTITY_FIELDS = frozenset(
    {
        "agent_id",
        "application_id",
        "run_id",
        "turn_id",
        "root_trace_id",
        "span_id",
        "parent_span_id",
        "memory_id",
        "thread_id",
        "expected_thread_id",
        "user_id",
        "schema_version",
        "timestamp",
        "trace_export",
        "prompt_cache_key",
        "prompt_cache_prefix",
        "request_context_fingerprint",
        "canonical_title_fingerprint",
        "client_title_fingerprint",
        "canonical_page_id",
        "client_page_id",
        "grounding_source_ids",
    }
)
# MemoRizz field -> Sentry's gen_ai attribute, on a model call.
_MODEL_ATTRIBUTES = {
    "model": "gen_ai.request.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "total_tokens": "gen_ai.usage.total_tokens",
    "cached_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_write_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "request_id": "gen_ai.response.id",
    "max_output_tokens": "gen_ai.request.max_tokens",
}
_USAGE_TOTALS = (
    ("input_tokens", "gen_ai.usage.input_tokens"),
    ("output_tokens", "gen_ai.usage.output_tokens"),
    ("total_tokens", "gen_ai.usage.total_tokens"),
    ("cached_tokens", "gen_ai.usage.cache_read.input_tokens"),
)


def sentry_trace_id(root_trace_id: Any) -> str:
    """The 32 hex characters Sentry files a MemoRizz trace under."""
    text = str(root_trace_id or "").strip()
    try:
        return uuid.UUID(text).hex
    except ValueError:
        return hashlib.md5(text.encode("utf-8")).hexdigest()  # nosec B324


def _span_id(value: Any) -> str:
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:16]  # nosec B324


def _moment(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _duration(events: List[Dict[str, Any]]) -> Optional[timedelta]:
    for event in reversed(events):
        value = event.get("duration_ms")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return timedelta(milliseconds=float(value))
    return None


def _failed(events: List[Dict[str, Any]]) -> bool:
    """Whether a span ended in a failure. A turn somebody stopped, or one
    waiting for an approval, did not succeed but is not a failure."""
    last = events[-1]
    status = str(last.get("status") or "").lower()
    if status in _NOT_FAILURES:
        return False
    return status in _FAILED or last.get("success") is False


def _status(events: List[Dict[str, Any]]) -> str:
    status = str(events[-1].get("status") or "").lower()
    if status in _STATUS:
        return _STATUS[status]
    return "internal_error" if _failed(events) else "ok"


def _kind(event: Dict[str, Any]) -> str:
    return str(event.get("trace_kind") or "trace").strip().lower() or "trace"


def _merged(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One span's fields: later events (the result) win over earlier ones."""
    from ..normalization import TRACE_METADATA_FIELDS

    merged: Dict[str, Any] = {}
    for event in events:
        for name in TRACE_METADATA_FIELDS:
            value = event.get(name)
            if value is None or value == "":
                continue
            if isinstance(value, (str, int, float, bool)):
                merged[name] = value
    return merged


def _content(events: List[Dict[str, Any]], kind: str) -> Optional[str]:
    for event in events:
        if _kind(event) == kind:
            text = str(event.get("content") or "")
            if text:
                return text[:_CONTENT_CHARS]
    return None


def _span(
    events: List[Dict[str, Any]],
    *,
    agent_name: str,
    include_content: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """``(op, description, data)`` for the events of one span."""
    kinds = {_kind(event) for event in events}
    fields = _merged(events)
    data: Dict[str, Any] = {
        f"memorizz.{name}": value
        for name, value in fields.items()
        if name not in _IDENTITY_FIELDS
    }
    data["gen_ai.agent.name"] = agent_name
    if kinds & {"model_call", "model_result"}:
        model = str(fields.get("model") or "model")
        data["gen_ai.operation.name"] = "chat"
        for name, attribute in _MODEL_ATTRIBUTES.items():
            if name in fields:
                data[attribute] = fields[name]
                data.pop(f"memorizz.{name}", None)
        data["gen_ai.response.model"] = str(fields.get("response_model") or model)
        if fields.get("provider"):
            provider = str(fields["provider"]).lower()
            data["gen_ai.provider.name"] = provider
            data["gen_ai.system"] = provider
        if fields.get("finish_reason"):
            data["gen_ai.response.finish_reasons"] = json.dumps(
                [str(fields["finish_reason"])]
            )
        return "gen_ai.chat", f"chat {model}", data
    if kinds & {"tool_call", "tool_result"}:
        tool = str(fields.get("logical_tool_name") or fields.get("tool_name") or "tool")
        data["gen_ai.operation.name"] = "execute_tool"
        data["gen_ai.tool.name"] = tool
        if include_content:
            for kind, attribute in (
                ("tool_call", "gen_ai.tool.call.arguments"),
                ("tool_result", "gen_ai.tool.call.result"),
            ):
                text = _content(events, kind)
                if text:
                    data[attribute] = text
        return "gen_ai.execute_tool", f"execute_tool {tool}", data
    kind = _kind(events[0])
    title = str(events[0].get("title") or kind).strip() or kind
    if fields.get("memory_type") and kind.startswith("memory"):
        title = f"{title} ({fields['memory_type']})"
    if include_content:
        text = _content(events, kind)
        if text:
            data["memorizz.content"] = text
    return f"memorizz.{kind}", title[:200], data


def build_sentry_events(
    trace_context: Dict[str, Any],
    events: List[Dict[str, Any]],
    *,
    agent_name: Optional[str] = None,
    include_content: bool = False,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """One turn as Sentry events: ``{"transaction": {...}, "error": {...}}``.

    ``error`` is None unless the turn failed. Nothing here needs the Sentry
    package: the events are plain dicts in Sentry's event format.
    """
    context = dict(trace_context or {})
    rows = [event for event in events or [] if isinstance(event, dict)]
    name = str(agent_name or context.get("agent_id") or "agent").strip() or "agent"
    root_key = str(context.get("root_trace_id") or "").strip()
    if not root_key:
        root_key = next(
            (str(row["root_trace_id"]) for row in rows if row.get("root_trace_id")),
            str(uuid.uuid4()),
        )
    trace_id = sentry_trace_id(root_key)
    root_span = _span_id(root_key)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("span_id") or row.get("trace_id") or f"event-{index}")
        groups.setdefault(key, []).append(row)
    root_events = groups.pop(root_key, [])

    moments = [
        moment for moment in (_moment(r.get("timestamp")) for r in rows) if moment
    ]
    fallback = _moment(context.get("timestamp")) or datetime.now(timezone.utc)
    started = min(moments) if moments else fallback
    finished = max(moments) if moments else fallback

    spans: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {}
    model: Optional[str] = None
    for key, captured in groups.items():
        if len(spans) >= MAX_SPANS:
            break
        times = sorted(
            moment
            for moment in (_moment(r.get("timestamp")) for r in captured)
            if moment
        )
        first, last = (times[0], times[-1]) if times else (started, started)
        duration = _duration(captured)
        if first == last:
            # A step recorded once, when it ended: its duration runs backwards.
            start, end = (last - duration, last) if duration else (last, last)
        else:
            start, end = first, last
            if duration and start + duration > end:
                end = start + duration
        started, finished = min(started, start), max(finished, end)
        op, description, data = _span(
            captured, agent_name=name, include_content=include_content
        )
        if op == "gen_ai.chat":
            model = model or data.get("gen_ai.request.model")
            for field_name, attribute in _USAGE_TOTALS:
                value = data.get(attribute)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[attribute] = totals.get(attribute, 0) + value
        parent = captured[0].get("parent_span_id")
        parent_id = (
            _span_id(parent)
            if parent and str(parent) != root_key and str(parent) in groups
            else root_span
        )
        spans.append(
            {
                "trace_id": trace_id,
                "span_id": _span_id(key),
                "parent_span_id": parent_id,
                "same_process_as_parent": True,
                "op": op,
                "description": description,
                "start_timestamp": start,
                "timestamp": end,
                "status": _status(captured),
                "origin": ORIGIN,
                "data": data,
            }
        )

    result = root_events or rows or [{}]
    status = _status(result) if (root_events or rows) else "ok"
    root_fields = _merged(root_events)
    data: Dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": name,
    }
    if context.get("thread_id"):
        data["gen_ai.conversation.id"] = str(context["thread_id"])
    if model:
        data["gen_ai.request.model"] = model
    for attribute, value in totals.items():
        data[attribute] = int(value) if float(value).is_integer() else value
    for field_name in ("agent_id", "application_id", "run_id", "turn_id", "memory_id"):
        if context.get(field_name):
            data[f"memorizz.{field_name}"] = str(context[field_name])
    data["memorizz.root_trace_id"] = root_key
    for field_name, value in root_fields.items():
        if field_name not in _IDENTITY_FIELDS:
            data[f"memorizz.{field_name}"] = value

    tags: Dict[str, str] = {}
    for row in rows:
        labels = row.get("trace_tags")
        if isinstance(labels, dict):
            for label, value in labels.items():
                tags[str(label)[:32]] = str(value)[:200]
    tags["memorizz.agent"] = name[:200]
    if context.get("agent_id"):
        tags["memorizz.agent_id"] = str(context["agent_id"])[:200]
    tags["memorizz.status"] = str(result[-1].get("status") or "unknown")[:200]

    transaction: Dict[str, Any] = {
        "type": "transaction",
        "platform": "python",
        "transaction": f"invoke_agent {name}"[:200],
        "transaction_info": {"source": "custom"},
        "contexts": {
            "trace": {
                "trace_id": trace_id,
                "span_id": root_span,
                "op": "gen_ai.invoke_agent",
                "description": f"invoke_agent {name}"[:200],
                "status": status,
                "origin": ORIGIN,
                "data": data,
            }
        },
        "tags": tags,
        "start_timestamp": started,
        "timestamp": finished,
        "spans": spans,
    }
    if environment:
        transaction["environment"] = environment
    if include_content and context.get("user_id"):
        transaction["user"] = {"id": str(context["user_id"])}

    error: Optional[Dict[str, Any]] = None
    if root_events and _failed(root_events):
        code = str(root_fields.get("error_code") or "").strip()
        message = f"{name}: agent turn failed" + (f" ({code})" if code else "")
        error = {
            "level": "error",
            "platform": "python",
            "logger": "memorizz.agent",
            "message": message[:500],
            "timestamp": finished,
            "transaction": transaction["transaction"],
            "contexts": {"trace": {"trace_id": trace_id, "span_id": root_span}},
            "tags": {**tags, **({"memorizz.error_code": code[:200]} if code else {})},
            "fingerprint": [
                "memorizz-agent-turn",
                str(context.get("agent_id") or name),
                code or "error",
            ],
        }
        if environment:
            error["environment"] = environment
        if "user" in transaction:
            error["user"] = dict(transaction["user"])
    return {"transaction": transaction, "error": error}


# ---- sending -------------------------------------------------------------

_CLIENTS: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Any] = {}
_CLIENTS_LOCK = threading.Lock()
_FLUSH_AT_EXIT = False


def _dsn() -> Optional[str]:
    for name in DSN_ENV:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def local_server_url() -> Optional[str]:
    """Where Sentry's local development server listens, when the process was
    told to send to it (``SENTRY_SPOTLIGHT``: a URL, or a truthy value for
    the default address). None when it was not."""
    value = os.getenv(LOCAL_ENV, "").strip()
    if not value or value.lower() in {"0", "false", "no", "off", "n", "f"}:
        return None
    if value.lower() in {"1", "true", "yes", "on", "y", "t"}:
        return LOCAL_DEFAULT_URL
    return value


def _sdk() -> Any:
    try:
        import sentry_sdk
    except ImportError:
        return None
    return sentry_sdk


def _flush_all(timeout: float = 2.0) -> None:
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
    for client in clients:
        try:
            client.flush(timeout=timeout)
        except Exception:  # pragma: no cover - best effort at exit
            logger.debug("Flushing Sentry failed", exc_info=True)


def _own_client(
    dsn: Optional[str], environment: Optional[str], local: Optional[str] = None
) -> Any:
    """MemoRizz's client for one DSN, one local server, or both. It installs
    no integrations, so the host process's excepthook, logging and HTTP
    clients stay as they were."""
    global _FLUSH_AT_EXIT
    key = (dsn, environment, local)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            sentry_sdk = _sdk()
            client = sentry_sdk.Client(
                dsn=dsn,
                environment=environment,
                default_integrations=False,
                auto_enabling_integrations=False,
                integrations=[],
                send_default_pii=False,
                traces_sample_rate=1.0,
                # The SDK posts each envelope to the local server as well.
                spotlight=local or False,
            )
            _CLIENTS[key] = client
            if not _FLUSH_AT_EXIT:
                _FLUSH_AT_EXIT = True
                atexit.register(_flush_all)
        return client


def reset_clients() -> None:
    """Close every client MemoRizz opened (after the DSN changes, in tests)."""
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        try:
            client.close(timeout=2.0)
        except Exception:
            logger.debug("Closing a Sentry client failed", exc_info=True)


def _dsn_parts(
    dsn: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(host, web origin, project id)`` of a DSN, each None when the DSN
    does not parse."""
    if not dsn:
        return None, None, None
    try:
        parts = urlsplit(dsn)
        port = parts.port
    except ValueError:
        return None, None, None
    if not parts.hostname:
        return None, None, None
    project = parts.path.rstrip("/").rsplit("/", 1)[-1] or None
    origin = f"{parts.scheme or 'https'}://{parts.hostname}" + (
        f":{port}" if port else ""
    )
    return parts.hostname, origin, project


class SentryObservability(ObservabilityProvider):
    """Send an agent's turns to Sentry (``observability_provider="sentry"``).

    Settings: ``org`` (the organisation slug, or ``SENTRY_ORG``) and
    ``project`` shape the dashboard link, ``environment`` names the Sentry
    environment, ``sample_rate`` sends a share of turns and
    ``include_content=True`` adds tool arguments, tool results and the user
    id. The DSN comes from ``MEMORIZZ_SENTRY_DSN`` or ``SENTRY_DSN`` and needs
    the ``sentry`` extra (:meth:`missing` gives the command that installs it).

    Under ``sentry local run -- <command>`` (or with ``SENTRY_SPOTLIGHT`` set)
    turns also go to Sentry's local development server, and need no DSN.
    """

    name = "sentry"
    external = True

    @property
    def label(self) -> str:  # type: ignore[override]
        return "Sentry (local)" if self.local_only else "Sentry"

    @property
    def local_only(self) -> bool:
        """True when turns go to Sentry's local development server and
        nowhere else: no DSN is set and the host has not set up Sentry."""
        return bool(local_server_url()) and not _dsn() and not self._host_dsn()

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._warned = False

    # -- readiness ----------------------------------------------------------

    def _client(self) -> Any:
        """The client turns are sent through, or None when there is none."""
        sentry_sdk = _sdk()
        if sentry_sdk is None:
            return None
        dsn = _dsn()
        local = local_server_url()
        if dsn:
            return _own_client(dsn, self.config.environment, local)
        try:
            client = sentry_sdk.get_client()
            if client.is_active() and client.dsn:
                return client
        except Exception:
            logger.debug("Reading the host's Sentry client failed", exc_info=True)
        if local:
            return _own_client(None, self.config.environment, local)
        return None

    def _host_dsn(self) -> Optional[str]:
        sentry_sdk = _sdk()
        if sentry_sdk is None:
            return None
        try:
            client = sentry_sdk.get_client()
            return client.dsn if client.is_active() else None
        except Exception:
            return None

    def missing(self) -> List[str]:
        missing: List[str] = []
        if _sdk() is None:
            from ...extras import install_command

            missing.append(f"the sentry extra ({install_command(['sentry'])})")
        if not _dsn() and not self._host_dsn() and not local_server_url():
            missing.append("SENTRY_DSN")
        return missing

    def status(self) -> Dict[str, Any]:
        status = super().status()
        # Where the local development server listens, when turns go there.
        status["local"] = local_server_url()
        status["local_only"] = self.local_only
        return status

    def export_mark(self) -> str:
        return LOCAL_MARK if self.local_only else self.name

    # -- links --------------------------------------------------------------

    def _base_url(self) -> Tuple[str, Optional[str]]:
        """The organisation's address and the project id traces are filed
        under."""
        host, origin, dsn_project = _dsn_parts(_dsn() or self._host_dsn())
        project = self.config.project or dsn_project
        org = self.config.org or os.getenv(ORG_ENV, "").strip() or None
        if host and not (host == "sentry.io" or host.endswith(".sentry.io")):
            # Self-hosted: the DSN's host is the web address too.
            return f"{origin}/organizations/{org or 'sentry'}", project
        if org:
            return f"https://{org}.sentry.io", project
        # Sentry resolves :orgslug to the signed-in person's organisation.
        return "https://sentry.io/orgredirect/organizations/:orgslug", project

    def dashboard_url(
        self, *, agent_id: Optional[str] = None, agent_name: Optional[str] = None
    ) -> Optional[str]:
        if self.config.url:
            return self.config.url
        if self.local_only:
            return f"{LOCAL_UI}?{urlencode({'url': local_server_url() or ''})}"
        base, project = self._base_url()
        query: Dict[str, str] = {}
        if project and str(project).isdigit():
            query["project"] = str(project)
        return f"{base}/explore/traces/" + (f"?{urlencode(query)}" if query else "")

    def trace_url(self, root_trace_id: Any, *, started_at: Any = None) -> Optional[str]:
        if not str(root_trace_id or "").strip() or self.local_only:
            return None
        base, _ = self._base_url()
        url = f"{base}/explore/traces/trace/{sentry_trace_id(root_trace_id)}/"
        moment = _moment(started_at)
        # Sentry finds a trace faster, and past its default window, when told
        # roughly when it happened.
        return url + (f"?timestamp={int(moment.timestamp())}" if moment else "")

    def trace_reference(self, trace_context: Dict[str, Any]) -> Optional[str]:
        root = (trace_context or {}).get("root_trace_id")
        return sentry_trace_id(root) if root else None

    # -- sending ------------------------------------------------------------

    def export_turn(
        self,
        trace_context: Dict[str, Any],
        events: List[Dict[str, Any]],
        *,
        agent_name: Optional[str] = None,
    ) -> Optional[str]:
        client = self._client()
        if client is None:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Sentry observability is on but cannot send: missing %s. "
                    "Traces are still kept in MemoRizz.",
                    ", ".join(self.missing()) or "a Sentry client",
                )
            return None
        rate = self.config.sample_rate
        if rate is not None and random.random() >= rate:  # nosec B311
            return None
        built = build_sentry_events(
            trace_context,
            events,
            agent_name=agent_name,
            include_content=self.config.include_content,
            environment=self.config.environment,
        )
        client.capture_event(built["transaction"])
        if built["error"] is not None:
            client.capture_event(built["error"])
        return built["transaction"]["contexts"]["trace"]["trace_id"]

    def flush(self, timeout: float = 2.0) -> None:
        client = self._client()
        if client is not None:
            try:
                client.flush(timeout=timeout)
            except Exception:
                logger.debug("Flushing Sentry failed", exc_info=True)

    def close(self) -> None:
        # The client is shared by every agent using this DSN: flush, keep it.
        self.flush()
