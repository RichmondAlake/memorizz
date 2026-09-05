# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import array
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import oracledb
except ImportError:
    raise ImportError(
        "oracledb package is required for Oracle provider. "
        "Install it with: pip install oracledb"
    )

from ...enums.memory_type import MemoryType
from ...long_term.semantic.persona.persona import Persona
from ...long_term.semantic.persona.role_type import RoleType
from ...memagent import MemAgentModel
from ..base import MemoryProvider, MemoryProviderCapabilities, filter_tool_log_rows

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Sentinel used to distinguish "user_id filter not supplied" from
# "user_id explicitly set to None" (which means anonymous/legacy scope).
class _UserIdUnset:
    """Unique sentinel so ``user_id=None`` is not conflated with "no filter"."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<user_id unset>"


_UNSET = _UserIdUnset()


# Memory types that are per-user data. Shared app-level config (agents,
# personas, toolbox, shared_memory) is intentionally NOT user-scoped.
_USER_SCOPED_MEMORY_TYPES = frozenset(
    {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
        MemoryType.TOOL_LOG,
        MemoryType.SKILLBOX,
    }
)


def _memory_type_supports_user_id(memory_type: Any) -> bool:
    """Return True when rows of this memory type carry a user_id column."""
    try:
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
    except Exception:
        return False
    return memory_type in _USER_SCOPED_MEMORY_TYPES


# ---------------------------------------------------------------------------
# Row→dict registry
# ---------------------------------------------------------------------------
#
# The relational read paths (``retrieve_by_id``, ``_list_all_from_table``,
# ``_vector_search`` and the agent-tool lookups) all share the same shape:
# SELECT an explicit column list, then map the positional row into a dict
# with per-column post-processing (LOB reads, JSON deserialization,
# timestamp formatting, RAW(16) → uuid strings, ...).  Historically every
# memory type re-implemented that mapping inline per path, and the copies
# drifted — e.g. ``retrieve_by_id`` returned raw CLOB locators for
# ``personas.background`` / ``toolbox.description`` while ``_vector_search``
# read them into strings.
#
# The specs below are now the single source of truth: each ``_Field``
# couples one SQL column with the dict key(s) it emits and the transform
# ("kind") applied to the raw value.  ``OracleProvider._apply_row_fields``
# interprets the kinds; the SELECT lists are built from the same specs so
# projection and mapping can never drift apart again.  Kinds:
#
#   plain          value emitted as-is
#   plain_or       ``value if value else default``
#   lob            ``_read_lob_value`` (CLOBs → str) — uniform on every path
#   lob_or         ``_read_lob_value(value) or default``
#   json           ``_deserialize_json_field``; falsy → default when given
#   json_pre       ``_deserialize_json_field(value) if value else default``
#                  (legacy pre-check variant kept for exact parity)
#   ts             ``value.isoformat() if value else None``
#   ts_attr        isoformat when available, otherwise the raw value
#   ts_str         isoformat when available, else ``str(value)`` if truthy
#   ts_str_nn      isoformat when available, else ``str(value)`` if not None
#   ts_opt         key only emitted when truthy (isoformat/str)
#   uuid           RAW(16) → canonical uuid string (None-safe)
#   int/float/bool coerce when not None, otherwise default
#   score          ``float(value)`` (vector-search score column)
#   epoch_now      ``value.timestamp()`` fallback ``time.time()``
#   vector         ``list(value)`` or None (always emitted)
#   vector_or_empty ``list(value)`` or []
#   vector_opt     emitted only when include_embedding and value is not None
#   vector_opt_raw as vector_opt but without the ``list()`` conversion
#
# Container defaults are passed as factories (``list``/``dict``) so each row
# gets a fresh instance.


class _Field:
    """One SELECT column and the dict emission(s) derived from it."""

    __slots__ = ("col", "emits")

    def __init__(self, col: str, emits: tuple):
        self.col = col
        self.emits = emits


def _c(col: str, key: Any = ..., kind: str = "plain", default: Any = None) -> _Field:
    """Build a field spec. ``key`` defaults to the column name; ``None``
    means the column is selected but not emitted; a tuple emits the same
    transformed value under several keys."""
    if key is ...:
        key = col
    if key is None:
        emits: tuple = ()
    elif isinstance(key, tuple):
        emits = tuple((k, kind, default) for k in key)
    else:
        emits = ((key, kind, default),)
    return _Field(col, emits)


def _cm(col: str, *emits: tuple) -> _Field:
    """Field spec emitting several keys with distinct transforms."""
    return _Field(col, emits)


def _select_list(fields: tuple) -> str:
    return ", ".join(f.col for f in fields)


_TS_CREATED = _c("created_at", kind="ts")
_TS_UPDATED = _c("updated_at", kind="ts")
_EMB_FULL = _c("embedding", kind="vector")
_EMB_OPT = _c("embedding", kind="vector_opt")
_EMB_HIDDEN = _c("embedding", key=None)
_SCORE_FIELD = _c(
    "(1 - VECTOR_DISTANCE(embedding, :query_vec, COSINE)) as score",
    key="score",
    kind="score",
)

_PERSONA_CORE = (
    _c("persona_id", key=("_id", "persona_id")),
    _c("name"),
    _c("role_type"),
    _c("background", kind="lob"),
    _c("traits", kind="json"),
    _c("expertise", kind="json"),
    _c("memory_id"),
    _c("agent_id"),
)

_TOOLBOX_HEAD = (
    _c("name"),
    _c("description", kind="lob"),
    _c("signature"),
    _c("docstring", kind="lob"),
    _c("tool_type"),
)

_TOOLBOX_SCHEMA = (
    _c("parameters", kind="json", default=dict),
    _c("input_schema", kind="json", default=dict),
    _c("tool_policy", kind="json", default=dict),
    _c("aliases", kind="json", default=list),
    _c("deprecated_arguments", kind="json", default=dict),
    _c("queries", kind="json", default=list),
    _c("import_reference"),
    _c("user_id"),
)

_SUMMARY_CORE = (
    _c("summary_id", key=("_id", "summary_id")),
    _c("content", kind="lob"),
    _c("source_message_ids", kind="json", default=list),
    _c("summary_type"),
    _c("memory_id"),
    _c("agent_id"),
    _c("user_id"),
    _c("thread_id"),
    _c("period_start", kind="float"),
    _c("period_end", kind="float"),
    _c("memory_units_count", kind="int", default=0),
)

_SKILLBOX_CORE = (
    _c("skill_id", key=("_id", "skill_id")),
    _c("name"),
    _c("description", kind="lob"),
    _c("content", kind="lob"),
    _c("preconditions", kind="json", default=list),
    _c("tools_used", kind="json", default=list),
    _c("queries", kind="json", default=list),
    _c("agent_id"),
    _c("user_id"),
    _c("source_canonical_hash"),
    _c("source_workflow_ids", kind="json", default=list),
    _c("exemplar_workflow_id"),
    _c("status"),
    _c("injection_role", default="user"),
    _c("version", kind="int", default=1),
    _c("promoted_at", kind="ts"),
    _c("demoted_at", kind="ts"),
    _c("demotion_reason", kind="lob"),
    _c("baseline", kind="json", default=dict),
    _c("stats", kind="json", default=dict),
)


def _workflow_body(steps_default: Any) -> tuple:
    return (
        _c("name"),
        _c("description", kind="lob"),
        _c("steps", kind="json", default=steps_default),
        _c("current_step"),
        _c("status"),
        _c("outcome", kind="json"),
        _c("memory_id"),
        _c("agent_id"),
        _c("user_id"),
        _c("user_query", kind="lob"),
    )


_WORKFLOW_TAIL = (
    _c("canonical_hash"),
    _c("canonical_signature", kind="json"),
    _c("step_count", kind="int"),
    _c("promoted_skill_id"),
    _c("skills_activated", kind="json", default=list),
    _c("shadow_evaluations", kind="json", default=list),
)


def _tool_log_fields(ts_kind: str) -> tuple:
    return (
        _c("id", key="_id", kind="uuid"),
        _c("tool_log_id"),
        _c("tool_name"),
        _c("arguments", kind="lob"),
        _c("result", kind="lob"),
        _c("success", kind="bool", default=True),
        _c("error", kind="lob"),
        _c("outcome"),
        _c("outcome_details", kind="json", default=dict),
        _c("timestamp", kind=ts_kind),
        _c("agent_id"),
        _c("tool_call_id"),
        _c("thread_id"),
        _c("memory_id"),
        _c("user_id"),
    )


_KNOWLEDGE_BASE_HEAD = (
    _c("id", key="_id", kind="uuid"),
    _c("memory_id"),
    _c("content", kind="lob"),
    _c("memory_type"),
    _c("importance"),
    _c("last_accessed", kind="ts_attr"),
    _c("access_count"),
    _c("agent_id"),
    _c("user_id"),
    _c("source_id"),
    _c("parent_source_id"),
    _c("linked_source_ids", kind="json", default=list),
    _c("metadata", kind="json", default=dict),
)

_SHORT_TERM_CORE = (
    _c("id", key="_id", kind="uuid"),
    _c("memory_id"),
    _c("content", kind="lob"),
    _c("memory_type"),
    _c("ttl"),
    _c("agent_id"),
    _c("user_id"),
)

# Chunking metadata (knowledge_base_id, namespace, chunk_*) is optional on
# older schemas — ``_knowledge_base_chunk_fields`` swaps in NULL projections
# until the additive startup migration or migration 005 has run, keeping the
# tuple shape stable during rolling upgrades.
_KB_CHUNK_COLUMNS = (
    "knowledge_base_id",
    "namespace",
    "chunk_index",
    "chunk_count",
    "chunking_strategy",
)

# retrieve_by_id: rows addressed by their logical string id. Workflows,
# tool logs (and friends) need these dedicated specs because the generic
# ``SELECT id, data`` fallback has no ``data`` column to read — without
# them, by-id loads silently returned None.
_BY_ID_SPECS: Dict[MemoryType, tuple] = {
    MemoryType.KNOWLEDGE_BASE: (
        "id",
        _KNOWLEDGE_BASE_HEAD + (_EMB_FULL, _TS_CREATED, _TS_UPDATED),
    ),
    MemoryType.SHORT_TERM_MEMORY: (
        "id",
        _SHORT_TERM_CORE + (_EMB_FULL, _TS_CREATED, _c("expires_at", kind="ts_attr")),
    ),
    MemoryType.PERSONAS: (
        "persona_id",
        _PERSONA_CORE + (_EMB_FULL, _TS_CREATED, _TS_UPDATED),
    ),
    MemoryType.TOOLBOX: (
        "tool_id",
        (_c("tool_id", key=("_id", "tool_id")),)
        + _TOOLBOX_HEAD
        + _TOOLBOX_SCHEMA
        + (
            _c("memory_id"),
            _c("agent_id"),
            _EMB_FULL,
            _TS_CREATED,
            _TS_UPDATED,
        ),
    ),
    MemoryType.SKILLBOX: (
        "skill_id",
        _SKILLBOX_CORE + (_EMB_FULL, _TS_CREATED, _TS_UPDATED),
    ),
    MemoryType.WORKFLOW_MEMORY: (
        "workflow_id",
        (_c("workflow_id", key=("_id", "workflow_id")),)
        + _workflow_body(steps_default=dict)
        + _WORKFLOW_TAIL
        + (_EMB_FULL, _TS_CREATED, _TS_UPDATED),
    ),
    MemoryType.TOOL_LOG: ("tool_log_id", _tool_log_fields("ts_str_nn")),
    MemoryType.SUMMARIES: (
        "summary_id",
        _SUMMARY_CORE + (_EMB_FULL, _TS_CREATED),
    ),
}

# _list_all_from_table: full-table scans. (fields, order_by)
_LIST_SPECS: Dict[MemoryType, tuple] = {
    MemoryType.ENTITY_MEMORY: (
        (
            _c("id", key="_id", kind="uuid"),
            _c("entity_id"),
            _c("name"),
            _c("entity_type"),
            _c("attributes", kind="json", default=list),
            _c("relations", kind="json", default=list),
            _c("metadata", kind="json", default=dict),
            _c("memory_id"),
            _c("agent_id"),
            _c("user_id"),
            _c("embedding", kind="vector_opt_raw"),
            _c("created_at", kind="ts_opt"),
            _c("updated_at", kind="ts_opt"),
        ),
        None,
    ),
    MemoryType.CONVERSATION_MEMORY: (
        (
            _c("id", key="_id", kind="uuid"),
            _c("memory_id"),
            _c("thread_id"),
            _c("role"),
            _c("content", kind="lob"),
            _c("timestamp", kind="ts_str"),
            _c("agent_id"),
            _c("user_id"),
            _c("summary_id"),
            _EMB_OPT,
        ),
        "timestamp",
    ),
    MemoryType.TOOLBOX: (
        (_c("id", key="_id", kind="uuid"), _c("tool_id"))
        + _TOOLBOX_HEAD
        + _TOOLBOX_SCHEMA
        + (
            _c("memory_id"),
            _c("agent_id"),
            _EMB_OPT,
            _c("created_at", key=None),
            _c("updated_at", key=None),
        ),
        None,
    ),
    MemoryType.SKILLBOX: (
        _SKILLBOX_CORE + (_EMB_OPT, _TS_CREATED, _TS_UPDATED),
        None,
    ),
    MemoryType.WORKFLOW_MEMORY: (
        (_c("id", key="_id", kind="uuid"), _c("workflow_id"))
        + _workflow_body(steps_default=None)
        + _WORKFLOW_TAIL
        + (_EMB_OPT, _TS_CREATED, _TS_UPDATED),
        None,
    ),
    MemoryType.SUMMARIES: (
        (_c("id", key="row_id", kind="uuid"),)
        + _SUMMARY_CORE
        + (_EMB_OPT, _c("created_at", kind="ts_opt")),
        None,
    ),
    MemoryType.TOOL_LOG: (_tool_log_fields("ts_str"), "timestamp"),
    MemoryType.KNOWLEDGE_BASE: (
        _KNOWLEDGE_BASE_HEAD
        + (
            _EMB_OPT,
            _c("created_at", kind="ts_attr"),
            _c("updated_at", kind="ts_attr"),
        ),
        None,
    ),
    MemoryType.SHORT_TERM_MEMORY: (
        _SHORT_TERM_CORE
        + (
            _EMB_OPT,
            _c("created_at", kind="ts_attr"),
            _c("expires_at", kind="ts_attr"),
        ),
        None,
    ),
    MemoryType.SHARED_MEMORY: (
        (
            _c("id", key="_id", kind="uuid"),
            _c("memory_id"),
            _c("content", kind="json"),
            _c("memory_type"),
            _c("scope"),
            _c("owner_agent_id"),
            _c("access_list", kind="json"),
            _EMB_OPT,
            _c("created_at", kind="ts_opt"),
            _c("updated_at", kind="ts_opt"),
        ),
        "created_at",
    ),
}

# _vector_search result rows (score column included).
_VECTOR_SPECS: Dict[MemoryType, tuple] = {
    MemoryType.SEMANTIC_CACHE: (
        _c("id", key="_id", kind="uuid"),
        _c("cache_key"),
        _c("query_text", kind="lob"),
        _c("response", kind="lob"),
        _c("scope"),
        _c("similarity_threshold", kind="float", default=0.85),
        # hit_count is also mapped to usage_count for API compatibility.
        _c("hit_count", key=("hit_count", "usage_count"), kind="int", default=0),
        _c("agent_id"),
        _c("memory_id"),
        _c("session_id"),
        _c("user_id"),
        _c("metadata", kind="json", default=dict),
        _c("embedding", kind="vector_or_empty"),
        _cm(
            "created_at",
            ("timestamp", "epoch_now", None),
            ("created_at", "plain", None),
        ),
        _c("expires_at"),
        _SCORE_FIELD,
    ),
    MemoryType.PERSONAS: _PERSONA_CORE
    + (_EMB_HIDDEN, _TS_CREATED, _TS_UPDATED, _SCORE_FIELD),
    MemoryType.TOOLBOX: (_c("tool_id", key=("_id", "tool_id")),)
    + _TOOLBOX_HEAD
    + _TOOLBOX_SCHEMA
    + (
        _c("memory_id"),
        _c("agent_id"),
        _EMB_HIDDEN,
        _TS_CREATED,
        _TS_UPDATED,
        _SCORE_FIELD,
    ),
    MemoryType.SKILLBOX: _SKILLBOX_CORE
    + (_EMB_HIDDEN, _TS_CREATED, _TS_UPDATED, _SCORE_FIELD),
    MemoryType.WORKFLOW_MEMORY: (_c("workflow_id", key=("_id", "workflow_id")),)
    + _workflow_body(steps_default=dict)
    + _WORKFLOW_TAIL
    + (_EMB_HIDDEN, _TS_CREATED, _TS_UPDATED, _SCORE_FIELD),
    MemoryType.SUMMARIES: (
        *_SUMMARY_CORE,
        _EMB_HIDDEN,
        _TS_CREATED,
        _SCORE_FIELD,
    ),
    MemoryType.ENTITY_MEMORY: (
        _c("entity_id", key=("_id", "entity_id")),
        _c("name"),
        _c("entity_type"),
        _c("attributes", kind="json", default=list),
        _c("relations", kind="json", default=list),
        _c("metadata", kind="json", default=dict),
        _c("memory_id"),
        _c("agent_id"),
        _c("user_id"),
        _EMB_HIDDEN,
        _TS_CREATED,
        _TS_UPDATED,
        _SCORE_FIELD,
    ),
    MemoryType.CONVERSATION_MEMORY: (
        _c("id", key="_id", kind="uuid"),
        _c("memory_id"),
        _c("thread_id"),
        _c("role"),
        _c("content", kind="lob"),
        _c("timestamp", kind="ts_attr"),
        _c("agent_id"),
        _c("user_id"),
        _c("summary_id"),
        _EMB_HIDDEN,
        _SCORE_FIELD,
    ),
    # Chunking metadata is appended at runtime (after the score column) so
    # the agent-scoped filter in ``knowledge_base_lookup`` has a real
    # ``knowledge_base_id`` to match against.
    MemoryType.KNOWLEDGE_BASE: _KNOWLEDGE_BASE_HEAD
    + (_EMB_HIDDEN, _TS_CREATED, _TS_UPDATED, _SCORE_FIELD),
    MemoryType.SHORT_TERM_MEMORY: _SHORT_TERM_CORE
    + (_EMB_HIDDEN, _TS_CREATED, _c("expires_at", kind="ts_attr"), _SCORE_FIELD),
    MemoryType.SHARED_MEMORY: (
        _c("id", key="_id", kind="uuid"),
        _c("memory_id"),
        _c("content", kind="json"),
        _c("memory_type"),
        _c("scope"),
        _c("owner_agent_id"),
        _c("access_list", kind="json"),
        _EMB_HIDDEN,
        _TS_CREATED,
        _TS_UPDATED,
        _SCORE_FIELD,
    ),
}

# Tool metadata as served to agents (retrieve_tools_for_agent and the
# semantic tool search): defaulted, LLM-facing key names ("type", not
# "tool_type").
_AGENT_TOOL_FIELDS = (
    _c("tool_id", key="_id", kind="plain_or"),
    _c("name", kind="plain_or", default="unknown"),
    _c("description", kind="lob_or", default=""),
    _c("signature", kind="plain_or", default=""),
    _c("docstring", kind="lob_or", default=""),
    _c("tool_type", key="type", kind="plain_or", default="function"),
    _c("parameters", kind="json", default=dict),
    _c("input_schema", kind="json", default=dict),
    _c("tool_policy", kind="json", default=dict),
    _c("aliases", kind="json", default=list),
    _c("deprecated_arguments", kind="json", default=dict),
    _c("queries", kind="json", default=list),
    _c("import_reference"),
    _c("user_id"),
    _c("memory_id", kind="plain_or"),
    _c("embedding", kind="vector"),
)

# Same projection as _AGENT_TOOL_FIELDS minus the embedding, in the column
# order retrieve_memagent has always used for an agent's private tools.
_MEMAGENT_TOOL_FIELDS = (
    _c("tool_id", key="_id"),
    _c("name"),
    _c("description", kind="lob_or", default=""),
    _c("signature", kind="plain_or", default=""),
    _c("docstring", kind="lob_or", default=""),
    _c("tool_type", key="type", kind="plain_or", default="function"),
    _c("memory_id"),
    _c("parameters", kind="json", default=dict),
    _c("input_schema", kind="json", default=dict),
    _c("tool_policy", kind="json", default=dict),
    _c("aliases", kind="json", default=list),
    _c("deprecated_arguments", kind="json", default=dict),
    _c("queries", kind="json", default=list),
    _c("import_reference"),
    _c("user_id"),
)


@dataclass
class OracleConfig:
    """Configuration for the Oracle provider."""

    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        schema: Optional[str] = None,
        lazy_vector_indexes: Optional[bool] = None,
        index_policy: str = "lazy",
        selected_vector_indexes: Optional[List[Union[str, MemoryType]]] = None,
        in_database_embedding: bool = True,
        embedding_provider=None,
        embedding_config: Dict[str, Any] = None,
        pool_min: int = 1,
        pool_max: int = 5,
        pool_increment: int = 1,
    ):
        """
        Initialize the Oracle provider with configuration settings.

        Parameters:
        -----------
        user : str
            Oracle database username
        password : str
            Oracle database password
        dsn : str
            Oracle DSN (Data Source Name) or connection string
            Examples:
            - "localhost:1521/FREEPDB1"
            - "myhost.example.com:1521/XEPDB1"
            - Full TNS string
        schema : str, optional
            Schema name for tables (default: None, uses username as schema)
        lazy_vector_indexes : bool
            If True, vector indexes are created only when needed
            If False, vector indexes are created immediately during initialization
            Default: False
        in_database_embedding : bool
            Use an ONNX embedding model inside Oracle AI Database when no
            explicit ``embedding_provider`` is supplied. The default is True.
            Model, dimensions, ONNX source, and installation behavior can be
            overridden through ``embedding_config``.
        embedding_provider : str or EmbeddingManager, optional
            Embedding provider to use. Can be:
            - EmbeddingManager instance (explicit injection)
            - String provider name ("openai", "ollama", "voyageai")
            - None (uses Oracle in-database embeddings by default)
        embedding_config : Dict[str, Any], optional
            Configuration for the embedding provider
            Example: {"model": "text-embedding-3-small", "dimensions": 512}
        pool_min : int
            Minimum number of connections in the pool (default: 1)
        pool_max : int
            Maximum number of connections in the pool (default: 5)
        pool_increment : int
            Number of connections to add when pool is exhausted (default: 1)
        """
        self.user = user
        self.password = password
        self.dsn = dsn
        self.schema = schema if schema is not None else user
        if lazy_vector_indexes is not None:
            index_policy = "lazy" if lazy_vector_indexes else "eager"
        normalized_policy = str(index_policy or "lazy").strip().lower()
        if normalized_policy not in {"none", "lazy", "selected", "eager"}:
            raise ValueError("index_policy must be one of: none, lazy, selected, eager")
        self.index_policy = normalized_policy
        # Compatibility attribute retained for applications that inspect it.
        self.lazy_vector_indexes = normalized_policy == "lazy"
        selected: set[MemoryType] = set()
        for item in selected_vector_indexes or []:
            if isinstance(item, MemoryType):
                selected.add(item)
                continue
            text = str(item).strip()
            try:
                selected.add(MemoryType(text))
            except ValueError:
                selected.add(MemoryType[text.upper()])
        self.selected_vector_indexes = selected
        self.in_database_embedding = bool(in_database_embedding)
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config or {}
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.pool_increment = pool_increment


class OracleProvider(MemoryProvider):
    """Oracle Database implementation of the MemoryProvider interface."""

    SUPPORTED_EMBEDDING_PROVIDERS = {
        "openai",
        "azure",
        "ollama",
        "voyageai",
        "huggingface",
    }

    VECTOR_INDEX_NAME_OVERRIDES = {
        MemoryType.CONVERSATION_MEMORY: "idx_conv_vec",
        MemoryType.KNOWLEDGE_BASE: "idx_kb_vec",
        MemoryType.SHORT_TERM_MEMORY: "idx_stm_vec",
        MemoryType.SEMANTIC_CACHE: "idx_cache_vec",
        MemoryType.ENTITY_MEMORY: "idx_entity_memory_vec",
    }
    NAME_FILTER_TYPES = {
        MemoryType.PERSONAS,
        MemoryType.TOOLBOX,
        MemoryType.SKILLBOX,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.ENTITY_MEMORY,
        MemoryType.MEMAGENT,
    }

    def memory_capabilities(self) -> MemoryProviderCapabilities:
        return MemoryProviderCapabilities(
            provider=type(self).__name__,
            batch_store=False,
            transactional_batch=False,
            scoped_search=True,
            result_scores=True,
            provenance=True,
            native_vector_search=True,
            native_hybrid_search=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        provision_if_missing: bool = False,
        index_policy: Optional[str] = None,
        selected_vector_indexes: Optional[List[Union[str, MemoryType]]] = None,
        **overrides: Any,
    ) -> "OracleProvider":
        """Construct a provider from the documented Oracle environment."""
        user = str(overrides.pop("user", os.getenv("ORACLE_USER", ""))).strip()
        password = str(
            overrides.pop("password", os.getenv("ORACLE_PASSWORD", ""))
        ).strip()
        dsn = str(overrides.pop("dsn", os.getenv("ORACLE_DSN", ""))).strip()
        if not user or not password or not dsn:
            raise ValueError(
                "ORACLE_USER, ORACLE_PASSWORD, and ORACLE_DSN are required"
            )
        if provision_if_missing:
            from .runtime import LocalOracleRuntime

            LocalOracleRuntime(
                user=user,
                password=password,
                dsn=dsn,
                provision_if_missing=True,
            ).ensure_ready()

        resolved_index_policy = (
            str(index_policy or os.getenv("MEMORIZZ_ORACLE_INDEX_POLICY", "") or "lazy")
            .strip()
            .lower()
        )

        if "in_database_embedding" not in overrides:
            from ..._env_io import resolve_oracle_in_database_embedding_from_env

            overrides[
                "in_database_embedding"
            ] = resolve_oracle_in_database_embedding_from_env()
        config = OracleConfig(
            user=user,
            password=password,
            dsn=dsn,
            index_policy=resolved_index_policy,
            selected_vector_indexes=selected_vector_indexes,
            **overrides,
        )
        return cls(config)

    def __init__(self, config: OracleConfig):
        """
        Initialize the Oracle provider with configuration settings.

        Parameters:
        -----------
        config : OracleConfig
            Configuration object containing connection parameters
        """
        self.config = config
        self._apply_embedding_defaults_from_env_if_needed(self.config)
        self._embedding_dimension_mismatch_detected = False
        self._embedding_mismatch_warning_emitted = False
        # Disable vector search per memory type once Oracle reports a vector dimension mismatch.
        self._vector_search_disabled_for = set()
        self._vector_dimension_mismatch_warning_emitted_for = set()
        # Cache of ``{table_name: {column, ...}}`` populated on first use.  We
        # consult this to skip ``user_id`` on schemas where the 001 migration
        # has not been run yet — see ``_table_has_column``.
        self._table_columns_cache: Dict[str, set] = {}

        # Initialize connection pool
        try:
            self.pool = oracledb.create_pool(
                user=config.user,
                password=config.password,
                dsn=config.dsn,
                min=config.pool_min,
                max=config.pool_max,
                increment=config.pool_increment,
            )
            logger.info("Oracle connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create Oracle connection pool: {e}")
            raise

        # Track which vector indexes have been created
        self._vector_indexes_created = set()
        self._vector_indexes_unavailable = set()
        self._vector_index_diagnostic_emitted = False

        # Process embedding provider configuration
        self._embedding_provider = self._setup_embedding_provider(config)

        # Create all memory store tables
        self._create_memory_stores()

        # Existing Oracle VECTOR columns have immutable dimensions. Fail
        # during provider construction instead of advertising a configured
        # model dimension and surfacing ORA-51803 on the first later write.
        if config.in_database_embedding:
            self.validate_vector_schema_dimensions()

        # Indexes are optional accelerators: exact VECTOR_DISTANCE search is
        # always available when a compatible VECTOR column exists.
        eager_types: Optional[List[MemoryType]] = None
        if config.index_policy == "eager":
            eager_types = list(MemoryType)
        elif config.index_policy == "selected":
            eager_types = list(config.selected_vector_indexes)
        if eager_types:
            try:
                self._create_vector_indexes_for_memory_stores(eager_types)
            except Exception as e:
                self._emit_vector_index_diagnostic(e)

    def _setup_embedding_provider(self, config: OracleConfig):
        """Setup the embedding provider based on configuration."""
        if config.embedding_provider is None and config.in_database_embedding:
            from ...embeddings import set_global_embedding_manager
            from .embedding import (
                OracleInDatabaseEmbeddingProvider,
                in_database_embedding_options,
            )

            provider = OracleInDatabaseEmbeddingProvider(
                self.pool,
                **in_database_embedding_options(config.embedding_config),
            )
            provider.ensure_model()
            set_global_embedding_manager(provider)
            logger.info(
                "Using Oracle in-database embedding model %s (%d dimensions)",
                provider.get_default_model(),
                provider.get_dimensions(),
            )
            return provider
        elif config.embedding_provider is None:
            return None
        elif isinstance(config.embedding_provider, str):
            try:
                from ...embeddings import EmbeddingManager, set_global_embedding_manager

                provider = EmbeddingManager(
                    config.embedding_provider, config.embedding_config
                )
                set_global_embedding_manager(provider)
                # Sanitize provider info to remove sensitive data
                provider_info = provider.get_provider_info()
                sanitized_info = provider_info.copy()
                if "config" in sanitized_info and isinstance(
                    sanitized_info["config"], dict
                ):
                    sanitized_config = sanitized_info["config"].copy()
                    # Remove sensitive keys
                    for key in ["api_key", "apiKey", "API_KEY", "token", "password"]:
                        if key in sanitized_config:
                            sanitized_config[key] = "***REDACTED***"
                    sanitized_info["config"] = sanitized_config
                logger.info(f"Created embedding provider: {sanitized_info}")
                return provider
            except Exception as e:
                logger.error(
                    f"Failed to create embedding provider '{config.embedding_provider}': {e}"
                )
                raise
        else:
            from ...embeddings import set_global_embedding_manager

            set_global_embedding_manager(config.embedding_provider)
            return config.embedding_provider

    @classmethod
    def _safe_int_from_env(cls, value: Optional[str]) -> Optional[int]:
        """Parse an integer environment value safely."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _resolve_embedding_defaults_from_env(
        cls,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        """
        Resolve Oracle embedding defaults from environment variables.

        Environment variables:
        - MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER
        - MEMORIZZ_DEFAULT_EMBEDDING_MODEL
        - MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS
        """
        provider = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "").strip().lower()
        if not provider:
            return None

        if provider not in cls.SUPPORTED_EMBEDDING_PROVIDERS:
            logger.warning(
                "Ignoring unsupported MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=%s. "
                "Supported: %s",
                provider,
                sorted(cls.SUPPORTED_EMBEDDING_PROVIDERS),
            )
            return None

        resolved: Dict[str, Any] = {}
        model = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "").strip()
        if model:
            resolved["model"] = model

        dims = cls._safe_int_from_env(
            os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS")
        )
        if dims:
            if provider == "voyageai":
                resolved["output_dimension"] = dims
            else:
                resolved["dimensions"] = dims

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if api_key:
                resolved["api_key"] = api_key
        elif provider == "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
            if api_key:
                resolved["api_key"] = api_key
            if endpoint:
                resolved["endpoint"] = endpoint
            if api_version:
                resolved["api_version"] = api_version
        elif provider == "voyageai":
            api_key = os.getenv("VOYAGE_API_KEY", "").strip()
            if api_key:
                resolved["api_key"] = api_key
        elif provider == "huggingface":
            token = os.getenv("HF_TOKEN", "").strip()
            if token:
                resolved["token"] = token
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
            if base_url:
                resolved["base_url"] = base_url

        return provider, resolved

    @classmethod
    def _apply_embedding_defaults_from_env_if_needed(cls, config: OracleConfig) -> None:
        """
        Apply default embedding provider/config from env when not explicitly provided.
        """
        if config.embedding_provider is not None or config.in_database_embedding:
            return

        defaults = cls._resolve_embedding_defaults_from_env()
        if not defaults:
            return

        provider, env_config = defaults
        merged_config = dict(env_config)
        if isinstance(config.embedding_config, dict):
            for key, value in config.embedding_config.items():
                if value not in (None, ""):
                    merged_config[key] = value

        config.embedding_provider = provider
        config.embedding_config = merged_config
        logger.info(
            "OracleProvider using default embedding provider from env: %s",
            provider,
        )

    @staticmethod
    def _is_embedding_dimension_mismatch_error(error_text: str) -> bool:
        """Return True when Oracle reports VECTOR dimension mismatch."""
        return any(
            code in error_text for code in ("ORA-51803", "ORA-51932", "ORA-42692")
        )

    @staticmethod
    def _is_vector_distance_dimension_mismatch_error(error_text: str) -> bool:
        """Return True when Oracle reports VECTOR_DISTANCE dimension mismatch."""
        return "ORA-51808" in error_text

    @staticmethod
    def _extract_dimension_pair_from_oracle_vector_error(
        error_text: str,
    ) -> Optional[tuple[int, int]]:
        """
        Extract the dimension pair from Oracle VECTOR errors like:
        "... different dimension counts (2050, 256)."
        """
        match = re.search(r"\((\d+)\s*,\s*(\d+)\)", str(error_text))
        if not match:
            return None
        try:
            return int(match.group(1)), int(match.group(2))
        except (TypeError, ValueError):
            return None

    def _handle_embedding_dimension_mismatch(
        self, store_name: str, exc: Exception
    ) -> None:
        """
        Mark this provider instance as embedding-write disabled after a dimension mismatch.
        """
        self._embedding_dimension_mismatch_detected = True
        if not self._embedding_mismatch_warning_emitted:
            self._embedding_mismatch_warning_emitted = True
            logger.warning(
                "%s embedding dimension mismatch detected. "
                "Disabling auto-embedding writes for this OracleProvider instance. "
                "Align MEMORIZZ_DEFAULT_EMBEDDING_* / OracleConfig embedding settings "
                "with the schema VECTOR dimensions (or use a separate schema). Error: %s",
                store_name,
                exc,
            )

    def _get_embedding_provider(self):
        """Get the embedding provider to use, with fallback logic."""
        if self._embedding_provider is not None:
            return self._embedding_provider
        else:
            from ...embeddings import get_embedding_manager

            return get_embedding_manager()

    def _get_embedding_dimensions_safe(self) -> int:
        """Safely get embedding dimensions with error handling."""
        try:
            if self._embedding_provider is not None:
                return self._embedding_provider.get_dimensions()
            else:
                from ...embeddings import get_embedding_dimensions

                return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Failed to get embedding dimensions: {e}")
            raise RuntimeError(
                "Cannot determine embedding dimensions. Please configure embeddings first using:\n"
                "configure_embeddings('openai', {'model': 'text-embedding-3-small', 'dimensions': 512})\n"
                "Or use lazy_vector_indexes=True to defer vector index creation."
            )

    def _generate_embedding_if_needed(
        self, content: str, existing_embedding=None
    ) -> Optional[List[float]]:
        """
        Generate embedding for content if needed.

        Args:
            content: Text content to embed
            existing_embedding: Existing embedding value (if any)

        Returns:
            Embedding vector or None
        """
        if self._embedding_dimension_mismatch_detected:
            # Prevent repeated Oracle VECTOR mismatch retries in long-running sessions.
            return None

        # If embedding already exists, use it (normalized to array.array so it binds
        # to a VECTOR column — a raw list raises ORA-01484 in thin mode).
        if existing_embedding is not None:
            return self._prepare_vector_value(existing_embedding)

        # If no embedding provider configured, return None
        if self._embedding_provider is None:
            return None

        # If no content to embed, return None
        if not content:
            return None

        # Generate embedding
        try:
            embedding = self._embedding_provider.get_embedding(content)
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return self._prepare_vector_value(embedding)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def get_observability_index(self):
        from ...observability.sql_index import OracleSpanIndex

        if not hasattr(self, "_observability_index"):
            self._observability_index = OracleSpanIndex(self)
        return self._observability_index

    def _get_connection(self):
        """Get a connection from the pool."""
        return self.pool.acquire()

    @staticmethod
    def _vector_dimension_from_ddl(ddl: str, column_name: str) -> Optional[int]:
        """Extract ``VECTOR(n, ...)`` for one column from Oracle table DDL."""
        pattern = re.compile(
            rf'"?{re.escape(str(column_name))}"?\s+VECTOR\s*\(\s*(\d+)\s*,',
            re.IGNORECASE,
        )
        match = pattern.search(str(ddl or ""))
        return int(match.group(1)) if match else None

    def get_vector_schema_dimensions(self) -> Dict[str, int]:
        """Return declared dimensions for schema ``EMBEDDING`` columns.

        ``DBMS_METADATA.GET_DDL`` is used instead of ``USER_VECTOR_COLUMNS``
        because the latter is not available in every Oracle AI Database
        release that supports the VECTOR type.
        """
        owner = str(self.config.schema or self.config.user).strip().upper()
        dimensions: Dict[str, int] = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT table_name, column_name
                    FROM all_tab_columns
                    WHERE owner = :owner
                      AND data_type = 'VECTOR'
                      AND column_name = 'EMBEDDING'
                    ORDER BY table_name, column_name
                    """,
                    {"owner": owner},
                )
                columns = cursor.fetchall()
                for table_name, column_name in columns:
                    cursor.execute(
                        """
                        SELECT DBMS_METADATA.GET_DDL(
                            'TABLE', :table_name, :owner
                        )
                        FROM dual
                        """,
                        {"table_name": table_name, "owner": owner},
                    )
                    row = cursor.fetchone()
                    ddl_value = row[0] if row else None
                    ddl = (
                        ddl_value.read()
                        if hasattr(ddl_value, "read")
                        else str(ddl_value or "")
                    )
                    dimension = self._vector_dimension_from_ddl(ddl, column_name)
                    if dimension is not None:
                        dimensions[f"{table_name}.{column_name}"] = dimension
                cursor.close()
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect Oracle VECTOR dimensions for schema "
                f"{owner}: {exc}"
            ) from exc
        return dimensions

    def validate_vector_schema_dimensions(
        self, expected_dimensions: Optional[int] = None
    ) -> Dict[str, int]:
        """Fail fast when existing VECTOR columns mismatch the embedder."""
        expected = int(
            expected_dimensions
            if expected_dimensions is not None
            else self._get_embedding_dimensions_safe()
        )
        declared = self.get_vector_schema_dimensions()
        mismatches = {
            name: dimension
            for name, dimension in declared.items()
            if dimension != expected
        }
        if mismatches:
            details = ", ".join(
                f"{name}={dimension}" for name, dimension in sorted(mismatches.items())
            )
            raise RuntimeError(
                "Oracle VECTOR dimension mismatch: the configured embedding "
                f"provider outputs {expected} dimensions, but schema "
                f"{self.config.schema or self.config.user} declares {details}. "
                "Use a fresh schema or migrate every affected VECTOR column; "
                "Oracle cannot reshape existing vector columns automatically."
            )
        return declared

    def _table_has_column(self, cursor: Any, table_name: str, column_name: str) -> bool:
        """Return True when a table column exists in the current schema."""
        try:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM user_tab_columns
                WHERE table_name = :table_name
                  AND column_name = :column_name
                """,
                {
                    "table_name": table_name.upper(),
                    "column_name": column_name.upper(),
                },
            )
            row = cursor.fetchone()
            return bool(row and int(row[0]) > 0)
        except Exception as exc:
            logger.debug(
                "Could not inspect schema column %s.%s: %s",
                table_name,
                column_name,
                exc,
            )
            return False

    def _get_table_name(self, memory_type: MemoryType) -> str:
        """Get the table name for a memory type."""
        return f"{self.config.schema}.{memory_type.value}"

    def _memory_type_has_column(
        self, memory_type: MemoryType, column_name: str
    ) -> bool:
        """Return True when the base table exposes ``column_name``.

        Used to safely skip optional columns (``user_id`` in particular) on
        schemas where the 001 migration has not been applied yet.
        """
        table_key = memory_type.value.upper()
        column_key = column_name.upper()
        if table_key in self._table_columns_cache:
            return column_key in self._table_columns_cache[table_key]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT column_name FROM user_tab_columns
                    WHERE table_name = :table_name
                    """,
                    {"table_name": table_key},
                )
                columns = {row[0].upper() for row in cursor.fetchall()}
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not introspect columns for %s: %s", table_key, exc)
            columns = set()

        self._table_columns_cache[table_key] = columns
        return column_key in columns

    def _create_memory_stores(self) -> None:
        """Create all memory store tables from the bundled relational schema.

        Historically this method built each table from an inline CREATE
        fallback that used a generic ``(id, data CLOB, embedding, ...)``
        shape. That shape didn't match the code — ``agents`` needs
        ``semantic_cache``, ``instruction``, ``application_mode``, etc. and
        ``knowledge_base`` needs the chunking columns — which surfaced as
        ``ORA-00904: "SEMANTIC_CACHE": invalid identifier`` on agent create
        and a missing Knowledge Base panel in the UI.

        The fix: drive table creation from ``schema_relational.sql`` — the
        same file ``memorizz oracle setup`` uses. One source of truth.

        Existing tables are preserved; Oracle returns ``ORA-00955`` for
        "already exists" which we swallow. Missing objects on a partially-
        initialized schema get filled in on the next connect.
        """
        # Reuse the SQL parser from setup.py so we apply the same embedding-
        # dimension substitution and comment-stripping logic everywhere.
        from pathlib import Path

        from .setup import parse_sql_file

        try:
            dimensions = self._get_embedding_dimensions_safe()
        except Exception:
            # Fallbacks in order: explicit env var → OpenAI 3-small default (256)
            # → legacy 1536. Picking 1536 blindly (what we used to do) causes
            # ORA-51803 whenever the runtime embedder actually produces
            # fewer dims, because Oracle VECTOR columns pin the dim count
            # at CREATE time.
            import os as _os

            env_dim = _os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS")
            if env_dim and env_dim.isdigit():
                dimensions = int(env_dim)
            else:
                dimensions = 256
            logger.warning(
                "Using default dimensions %d for table creation (set "
                "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS to override)",
                dimensions,
            )

        schema_path = Path(__file__).parent / "schema_relational.sql"
        if not schema_path.exists():
            logger.error(
                "schema_relational.sql not found at %s — cannot bootstrap tables",
                schema_path,
            )
            return

        statements = parse_sql_file(schema_path, embedding_dim=dimensions)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            created = skipped = failed = 0
            for stmt in statements:
                # Vector indexes are governed separately by index_policy.
                # Exact VECTOR_DISTANCE queries do not require an index.
                if re.match(r"^\s*CREATE\s+VECTOR\s+INDEX\b", stmt, re.I):
                    skipped += 1
                    continue
                try:
                    cursor.execute(stmt)
                    created += 1
                except Exception as exc:
                    msg = str(exc).upper()
                    statement = stmt.upper()
                    rolling_scope_index = "ORA-00904" in msg and any(
                        index_name in statement
                        for index_name in (
                            "IDX_SUMMARIES_MEMORY_THREAD",
                            "IDX_KB_NAMESPACE",
                        )
                    )
                    # ORA-00955: object already exists. ORA-01408: index
                    # already on those columns. ORA-02275: dupe fk
                    # constraint. A scope index can also run before the
                    # additive column migration on an upgraded schema; the
                    # migration below creates it after adding the column.
                    if rolling_scope_index or any(
                        code in msg for code in ("ORA-00955", "ORA-01408", "ORA-02275")
                    ):
                        skipped += 1
                    else:
                        failed += 1
                        if failed <= 3:
                            # Verbose for the first few so operators can
                            # diagnose a genuinely broken schema file.
                            logger.warning("Schema statement failed: %s", exc)
            conn.commit()

            logger.info(
                "Memory stores bootstrapped from schema_relational.sql "
                "(created=%d, skipped=%d, failed=%d)",
                created,
                skipped,
                failed,
            )

            # Keep the supplemental index helper — it handles vector indexes
            # and any cross-cutting indexes not expressed in the schema file.
            self._create_standard_indexes(cursor, conn)

        # Ensure existing tables have all the columns the provider expects.
        self._migrate_table_schemas()

    # Column definitions that the vector-search, store, and retrieve code
    # paths rely on.  Only columns beyond the generic set (id, data,
    # embedding, name, memory_id, agent_id, created_at, updated_at) are
    # listed — those already exist on every table created by
    # _create_memory_stores().
    _REQUIRED_COLUMNS: Dict[MemoryType, List[tuple]] = {
        MemoryType.ENTITY_MEMORY: [
            ("entity_id", "VARCHAR2(255)"),
            ("entity_type", "VARCHAR2(255)"),
            ("attributes", "CLOB"),
            ("relations", "CLOB"),
            ("metadata", "CLOB"),
        ],
        MemoryType.CONVERSATION_MEMORY: [
            ("memory_id", "VARCHAR2(255)"),
            ("thread_id", "VARCHAR2(255)"),
            ("role", "VARCHAR2(50)"),
            ("content", "CLOB"),
            ("timestamp", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("summary_id", "VARCHAR2(255)"),
        ],
        MemoryType.KNOWLEDGE_BASE: [
            ("memory_id", "VARCHAR2(255)"),
            ("content", "CLOB"),
            ("memory_type", "VARCHAR2(50)"),
            ("importance", "NUMBER(3,2)"),
            ("last_accessed", "TIMESTAMP"),
            ("access_count", "NUMBER(10) DEFAULT 0"),
            ("knowledge_base_id", "VARCHAR2(64)"),
            ("namespace", "VARCHAR2(255)"),
            ("chunk_index", "NUMBER(10) DEFAULT 0"),
            ("chunk_count", "NUMBER(10) DEFAULT 1"),
            ("chunking_strategy", "VARCHAR2(32)"),
            ("source_id", "VARCHAR2(512)"),
            ("parent_source_id", "VARCHAR2(512)"),
            ("linked_source_ids", "CLOB"),
            ("metadata", "CLOB"),
        ],
        MemoryType.SHORT_TERM_MEMORY: [
            ("memory_id", "VARCHAR2(255)"),
            ("content", "CLOB"),
            ("memory_type", "VARCHAR2(50)"),
            ("ttl", "NUMBER(10)"),
            ("expires_at", "TIMESTAMP"),
        ],
        MemoryType.PERSONAS: [
            ("persona_id", "VARCHAR2(255)"),
            ("role_type", "VARCHAR2(50)"),
            ("background", "CLOB"),
            ("traits", "CLOB"),
            ("expertise", "CLOB"),
        ],
        MemoryType.TOOLBOX: [
            ("tool_id", "VARCHAR2(255)"),
            ("description", "CLOB"),
            ("signature", "VARCHAR2(4000)"),
            ("docstring", "CLOB"),
            ("tool_type", "VARCHAR2(50)"),
            ("parameters", "CLOB"),
            ("input_schema", "CLOB"),
            ("tool_policy", "CLOB"),
            ("aliases", "CLOB"),
            ("deprecated_arguments", "CLOB"),
            ("queries", "CLOB"),
            ("import_reference", "VARCHAR2(1000)"),
            ("user_id", "VARCHAR2(255)"),
        ],
        MemoryType.SKILLBOX: [
            ("skill_id", "VARCHAR2(255)"),
            ("description", "CLOB"),
            ("content", "CLOB"),
            ("preconditions", "CLOB"),
            ("tools_used", "CLOB"),
            ("queries", "CLOB"),
            ("user_id", "VARCHAR2(255)"),
            ("source_canonical_hash", "VARCHAR2(255)"),
            ("source_workflow_ids", "CLOB"),
            ("exemplar_workflow_id", "VARCHAR2(255)"),
            ("status", "VARCHAR2(50)"),
            ("injection_role", "VARCHAR2(20) DEFAULT 'user' NOT NULL"),
            ("version", "NUMBER(10) DEFAULT 1"),
            ("promoted_at", "TIMESTAMP"),
            ("demoted_at", "TIMESTAMP"),
            ("demotion_reason", "CLOB"),
            ("baseline", "CLOB"),
            ("stats", "CLOB"),
        ],
        MemoryType.WORKFLOW_MEMORY: [
            ("workflow_id", "VARCHAR2(255)"),
            ("description", "CLOB"),
            ("steps", "CLOB"),
            ("current_step", "VARCHAR2(255)"),
            ("status", "VARCHAR2(50)"),
            ("outcome", "CLOB"),
            ("user_id", "VARCHAR2(255)"),
            ("user_query", "CLOB"),
            ("canonical_hash", "VARCHAR2(255)"),
            ("canonical_signature", "CLOB"),
            ("step_count", "NUMBER(10)"),
            ("promoted_skill_id", "VARCHAR2(255)"),
            ("skills_activated", "CLOB"),
            ("shadow_evaluations", "CLOB CHECK (shadow_evaluations IS JSON)"),
        ],
        MemoryType.SHARED_MEMORY: [
            ("content", "CLOB"),
            ("memory_type", "VARCHAR2(50)"),
            ("scope", "VARCHAR2(50)"),
            ("owner_agent_id", "VARCHAR2(255)"),
            ("access_list", "CLOB"),
        ],
        MemoryType.SUMMARIES: [
            ("summary_id", "VARCHAR2(255)"),
            ("content", "CLOB"),
            ("source_message_ids", "CLOB"),
            ("summary_type", "VARCHAR2(50)"),
            ("period_start", "NUMBER"),
            ("period_end", "NUMBER"),
            ("memory_units_count", "NUMBER(10) DEFAULT 0"),
            ("thread_id", "VARCHAR2(255)"),
        ],
        MemoryType.SEMANTIC_CACHE: [
            ("memory_id", "VARCHAR2(255)"),
            ("session_id", "VARCHAR2(255)"),
            ("user_id", "VARCHAR2(255)"),
            ("metadata", "CLOB"),
        ],
        MemoryType.TOOL_LOG: [
            ("tool_log_id", "VARCHAR2(255)"),
            ("tool_name", "VARCHAR2(255)"),
            ("arguments", "CLOB"),
            ("result", "CLOB"),
            ("success", "NUMBER(1) DEFAULT 1"),
            ("error", "CLOB"),
            ("outcome", "VARCHAR2(32) DEFAULT 'success'"),
            ("outcome_details", "CLOB"),
            ("timestamp", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("tool_call_id", "VARCHAR2(255)"),
            ("thread_id", "VARCHAR2(255)"),
        ],
    }

    def _migrate_table_schemas(self) -> None:
        """Add any missing columns to existing tables.

        Tables originally created with the generic schema (id, data,
        embedding, …) lack the specific columns that vector search and
        store methods expect.  This method inspects each table once at
        startup and issues ALTER TABLE ADD for any missing columns.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for memory_type, columns in self._REQUIRED_COLUMNS.items():
                table_bare = memory_type.value  # e.g. "entity_memory"
                table_full = self._get_table_name(memory_type)

                # Only migrate tables that already exist.
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM user_tables
                    WHERE table_name = UPPER(:1)
                    """,
                    (table_bare,),
                )
                if cursor.fetchone()[0] == 0:
                    continue

                for col_name, col_type in columns:
                    if self._table_has_column(cursor, table_bare, col_name):
                        continue
                    try:
                        cursor.execute(
                            f"ALTER TABLE {table_full} ADD ({col_name} {col_type})"
                        )
                        conn.commit()
                        logger.info(
                            "Migrated %s: added column %s", table_bare, col_name
                        )
                    except Exception as exc:
                        conn.rollback()
                        # ORA-01430 = column already exists (race / concurrent start)
                        if "ORA-01430" not in str(exc):
                            logger.warning(
                                "Could not add column %s to %s: %s",
                                col_name,
                                table_bare,
                                exc,
                            )

            # Canonicalize legacy summary source IDs after both columns exist.
            try:
                cursor.execute(
                    """
                    UPDATE summaries
                    SET source_message_ids = original_memory_ids
                    WHERE source_message_ids IS NULL
                      AND original_memory_ids IS NOT NULL
                    """
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.debug("Legacy summary source-ID migration skipped: %s", exc)

            # A normalized link table gives summary expansion a durable,
            # ordered relationship while source_message_ids remains a portable
            # document projection for non-relational providers.
            try:
                cursor.execute(
                    f"""
                    CREATE TABLE {self.config.schema}.summary_message_links (
                        summary_id VARCHAR2(255) NOT NULL,
                        message_id RAW(16) NOT NULL,
                        position NUMBER(10) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (summary_id, message_id),
                        CONSTRAINT fk_summary_link_summary FOREIGN KEY (summary_id)
                            REFERENCES {self.config.schema}.summaries(summary_id)
                            ON DELETE CASCADE,
                        CONSTRAINT fk_summary_link_message FOREIGN KEY (message_id)
                            REFERENCES {self.config.schema}.conversation_memory(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                if "ORA-00955" not in str(exc):
                    logger.warning("Could not create summary link table: %s", exc)

            # Add the exact-scope indexes after additive column migration so
            # upgraded installations get the same query plan as fresh schemas.
            for index_name, memory_type, columns in (
                (
                    "idx_summaries_memory_thread",
                    MemoryType.SUMMARIES,
                    ("memory_id", "thread_id"),
                ),
                ("idx_kb_namespace", MemoryType.KNOWLEDGE_BASE, ("namespace",)),
            ):
                if not all(
                    self._table_has_column(cursor, memory_type.value, column)
                    for column in columns
                ):
                    continue
                try:
                    cursor.execute(
                        f"CREATE INDEX {index_name} "
                        f"ON {self._get_table_name(memory_type)} "
                        f"({', '.join(columns)})"
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    if not any(code in str(exc) for code in ("ORA-00955", "ORA-01408")):
                        logger.warning("Could not create index %s: %s", index_name, exc)

    def _create_standard_indexes(self, cursor, conn):
        """Create standard B-tree indexes on commonly queried fields."""
        index_definitions = [
            ("name", ["name"]),
            ("memory_id", ["memory_id"]),
            ("agent_id", ["agent_id"]),
            ("created_at", ["created_at"]),
        ]

        for memory_type in MemoryType:
            table_name = self._get_table_name(memory_type)

            for idx_name, columns in index_definitions:
                full_idx_name = f"idx_{memory_type.value}_{idx_name}"
                try:
                    cursor.execute(
                        f"""
                        CREATE INDEX {full_idx_name}
                        ON {table_name} ({', '.join(columns)})
                        """
                    )
                    conn.commit()
                except Exception:
                    # Index might already exist
                    conn.rollback()

    def _create_vector_indexes_for_memory_stores(
        self, memory_types: Optional[List[MemoryType]] = None
    ) -> None:
        """Create vector indexes selected by the configured policy."""
        for memory_type in memory_types or list(MemoryType):
            try:
                self._ensure_vector_index(memory_type)
            except Exception as e:
                self._emit_vector_index_diagnostic(e)

    def _emit_vector_index_diagnostic(self, error: Exception) -> None:
        if self._vector_index_diagnostic_emitted:
            logger.debug("Additional Oracle vector-index failure: %s", error)
            return
        self._vector_index_diagnostic_emitted = True
        logger.warning(
            "Oracle vector indexes are unavailable; MemoRizz will use exact "
            "VECTOR_DISTANCE search. Check VECTOR_MEMORY_SIZE and privileges "
            "with provider.preflight(). First error: %s",
            error,
        )

    def _ensure_vector_index(self, memory_type: MemoryType):
        """Ensure vector index exists for a memory type."""
        index_name = self.VECTOR_INDEX_NAME_OVERRIDES.get(
            memory_type, f"idx_{memory_type.value}_vec"
        )
        index_key = index_name

        if index_key in self._vector_indexes_created:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if not self._table_has_column(cursor, memory_type.value, "embedding"):
                logger.debug(
                    "Skipping vector index %s: %s has no embedding column",
                    index_name,
                    memory_type.value,
                )
                self._vector_indexes_created.add(index_key)
                return

            table_name = self._get_table_name(memory_type)

            # Check if index exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM user_indexes
                WHERE index_name = UPPER(:1)
                """,
                (index_name,),
            )
            exists = cursor.fetchone()[0] > 0
            if not exists:
                try:
                    cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM all_ind_columns
                        WHERE table_owner = UPPER(:owner_name)
                          AND table_name = UPPER(:table_name)
                          AND column_name = 'EMBEDDING'
                        """,
                        {
                            "owner_name": self.config.schema,
                            "table_name": memory_type.value,
                        },
                    )
                    exists = cursor.fetchone()[0] > 0
                except Exception:
                    cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM user_ind_columns
                        WHERE table_name = UPPER(:table_name)
                          AND column_name = 'EMBEDDING'
                        """,
                        {"table_name": memory_type.value},
                    )
                    exists = cursor.fetchone()[0] > 0

            if not exists:
                try:
                    # Create vector index
                    # Oracle 23ai+ supports HNSW and IVF indexes
                    cursor.execute(
                        f"""
                        CREATE VECTOR INDEX {index_name}
                        ON {table_name} (embedding)
                        ORGANIZATION INMEMORY NEIGHBOR GRAPH
                        DISTANCE COSINE
                        WITH TARGET ACCURACY 95
                        """
                    )
                    conn.commit()
                    logger.info(f"Created vector index: {index_name}")
                    self._vector_indexes_created.add(index_key)
                except Exception as e:
                    error_str = str(e)
                    if "ORA-01408" in error_str:
                        logger.info(
                            f"Vector index already exists on {memory_type.value}.embedding"
                        )
                        self._vector_indexes_created.add(index_key)
                    else:
                        self._vector_indexes_unavailable.add(index_key)
                        self._vector_indexes_created.add(index_key)
                        self._emit_vector_index_diagnostic(e)
                    conn.rollback()
            else:
                self._vector_indexes_created.add(index_key)

    def _doc_to_dict(
        self, doc_data: str, include_embedding: bool = False
    ) -> Dict[str, Any]:
        """Convert JSON document to dictionary."""
        if not doc_data:
            return {}
        result = json.loads(doc_data)
        if not include_embedding and "embedding" in result:
            del result["embedding"]
        return result

    def _ensure_json_text(self, payload: Any) -> Optional[str]:
        """Convert payloads to JSON strings for storage in ``IS JSON`` columns.

        Notes:
        - If payload is already a JSON string, it is returned as-is.
        - If payload is a non-JSON string (e.g. ``success``), it is JSON-encoded
          as a string (e.g. ``\"success\"``) so it passes ``IS JSON`` checks.
        """
        if payload is None:
            return None
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return None
            try:
                json.loads(text)
                return text
            except Exception:
                return json.dumps(text)
        # Sanitize payload to handle JsonId objects before JSON serialization
        sanitized = self._sanitize_for_json(payload)
        return json.dumps(sanitized)

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """Recursively convert objects into JSON-serializable structures, handling JsonId objects."""
        from datetime import datetime

        if isinstance(value, datetime):
            return value.isoformat()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Handle oracledb JsonId objects
        # Check both by isinstance and by type name for robustness
        try:
            from oracledb import JsonId

            if isinstance(value, JsonId):
                return str(value)
        except (ImportError, AttributeError):
            pass

        # Also check by type name in case isinstance fails
        type_str = str(type(value))
        class_name = getattr(value, "__class__", None)
        class_name_str = str(class_name) if class_name else ""
        if "JsonId" in type_str or "JsonId" in class_name_str:
            try:
                return str(value)
            except Exception:
                pass

        if hasattr(value, "as_json"):
            try:
                result = value.as_json()
                if isinstance(result, str):
                    return json.loads(result)
                return result
            except Exception:
                pass

        if hasattr(value, "as_dict"):
            return OracleProvider._sanitize_for_json(value.as_dict())

        if hasattr(value, "value"):
            return OracleProvider._sanitize_for_json(value.value)

        if isinstance(value, dict):
            return {k: OracleProvider._sanitize_for_json(v) for k, v in value.items()}

        if isinstance(value, list):
            return [OracleProvider._sanitize_for_json(item) for item in value]

        # For bytes, try to convert to UUID string if it's 16 bytes
        if isinstance(value, bytes) and len(value) == 16:
            try:
                return str(uuid.UUID(bytes=value))
            except (ValueError, TypeError):
                pass

        return str(value)

    def _deserialize_json_field(self, value: Any):
        """Convert Oracle LOB/string JSON fields to Python objects."""
        if value is None:
            return None
        if hasattr(value, "as_json"):
            try:
                text = value.as_json()
                return json.loads(text) if isinstance(text, str) else text
            except Exception:
                try:
                    return value.as_json()
                except Exception:
                    pass
        if hasattr(value, "as_dict"):
            try:
                return value.as_dict()
            except Exception:
                return None
        if hasattr(value, "value"):
            try:
                return self._deserialize_json_field(value.value)
            except Exception:
                return value.value
        if hasattr(value, "read"):
            value = value.read()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _read_lob_value(value: Any) -> Any:
        """Read Oracle LOB values into plain Python types."""
        if value is None:
            return None
        if hasattr(value, "read"):
            value = value.read()
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value
        return value

    def _apply_row_fields(
        self, fields: tuple, row: tuple, include_embedding: bool = True
    ) -> Dict[str, Any]:
        """Map a positional row onto a dict as described by a field spec.

        This is the single row→dict interpreter behind ``retrieve_by_id``,
        ``_list_all_from_table``, ``_vector_search`` and the agent-tool
        lookups (see the registry at module level for the kind semantics).
        ``include_embedding`` only affects ``vector_opt*`` fields.
        """
        doc: Dict[str, Any] = {}
        for value, field in zip(row, fields):
            for key, kind, default in field.emits:
                if kind == "plain":
                    doc[key] = value
                elif kind == "plain_or":
                    doc[key] = value if value else default
                elif kind == "lob":
                    doc[key] = self._read_lob_value(value)
                elif kind == "lob_or":
                    doc[key] = self._read_lob_value(value) or default
                elif kind == "json":
                    parsed = self._deserialize_json_field(value)
                    if default is not None and not parsed:
                        parsed = default()
                    doc[key] = parsed
                elif kind == "json_pre":
                    doc[key] = (
                        self._deserialize_json_field(value) if value else default()
                    )
                elif kind == "ts":
                    doc[key] = value.isoformat() if value else None
                elif kind == "ts_attr":
                    doc[key] = (
                        value.isoformat() if hasattr(value, "isoformat") else value
                    )
                elif kind == "ts_str":
                    if hasattr(value, "isoformat"):
                        doc[key] = value.isoformat()
                    elif value:
                        doc[key] = str(value)
                    else:
                        doc[key] = value
                elif kind == "ts_str_nn":
                    if hasattr(value, "isoformat"):
                        doc[key] = value.isoformat()
                    elif value is not None:
                        doc[key] = str(value)
                    else:
                        doc[key] = None
                elif kind == "ts_opt":
                    if value:
                        doc[key] = (
                            value.isoformat()
                            if hasattr(value, "isoformat")
                            else str(value)
                        )
                elif kind == "uuid":
                    doc[key] = str(uuid.UUID(bytes=value)) if value else None
                elif kind == "int":
                    doc[key] = int(value) if value is not None else default
                elif kind == "float":
                    doc[key] = float(value) if value is not None else default
                elif kind == "bool":
                    doc[key] = bool(value) if value is not None else default
                elif kind == "score":
                    doc[key] = float(value)
                elif kind == "epoch_now":
                    doc[key] = (
                        value.timestamp()
                        if hasattr(value, "timestamp")
                        else time.time()
                    )
                elif kind == "vector":
                    doc[key] = list(value) if value is not None else None
                elif kind == "vector_or_empty":
                    doc[key] = list(value) if value is not None else []
                elif kind == "vector_opt":
                    if include_embedding and value is not None:
                        doc[key] = list(value)
                elif kind == "vector_opt_raw":
                    if include_embedding and value is not None:
                        doc[key] = value
                else:  # pragma: no cover - registry authoring error
                    raise ValueError(f"Unknown field kind: {kind}")
        return doc

    def _knowledge_base_chunk_fields(self) -> tuple:
        """Chunking-metadata fields for knowledge_base projections.

        Gracefully returns NULL projections when migration 002 hasn't run,
        so pre-migration schemas keep a stable tuple shape.
        """
        has_chunking = all(
            self._memory_type_has_column(MemoryType.KNOWLEDGE_BASE, col)
            for col in _KB_CHUNK_COLUMNS
        )
        if has_chunking:
            return tuple(_c(col) for col in _KB_CHUNK_COLUMNS)
        return tuple(_c("NULL", key=col) for col in _KB_CHUNK_COLUMNS)

    @staticmethod
    def _coerce_timestamp(value: Any) -> Any:
        """Normalize ISO-8601 strings to datetime for TIMESTAMP binds.

        ``to_dict`` serializers (Skill, Workflow) emit datetimes as ISO
        strings; Oracle's implicit string→TIMESTAMP conversion depends on
        the session NLS format, so bind real datetimes instead.
        """
        if value is None:
            return None
        from datetime import datetime

        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _prepare_vector_value(embedding: Any) -> Any:
        """Normalize embedding values for VECTOR bindings."""
        if embedding is None:
            return None
        if isinstance(embedding, array.array):
            return embedding
        if isinstance(embedding, list):
            return array.array("f", embedding)
        return embedding

    @staticmethod
    def _normalize_raw_uuid(value: Any) -> bytes:
        """Normalize UUID-like values into RAW(16) bytes for Oracle tables."""
        if isinstance(value, uuid.UUID):
            return value.bytes

        if isinstance(value, (bytes, bytearray, memoryview)):
            raw_value = bytes(value)
            if len(raw_value) == 16:
                return raw_value
            try:
                return uuid.UUID(raw_value.decode("utf-8")).bytes
            except Exception as exc:
                raise ValueError("Invalid UUID bytes payload") from exc

        if isinstance(value, str):
            return uuid.UUID(value).bytes

        raise ValueError(f"Unsupported UUID payload type: {type(value).__name__}")

    @staticmethod
    def _set_vector_input_size(cursor, param_name: str) -> None:
        """Hint vector binding type for Oracle drivers that support it."""
        try:
            cursor.setinputsizes(**{param_name: oracledb.DB_TYPE_VECTOR})
        except AttributeError:
            cursor.setinputsizes(**{param_name: array.array})

    def store(
        self,
        data: Dict[str, Any] = None,
        memory_store_type: MemoryType = None,
        memory_id: str = None,
        memory_unit: Any = None,
    ) -> str:
        """
        Store data in Oracle.

        Parameters:
        -----------
        data : Dict[str, Any], optional
            The document to be stored (legacy parameter)
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_id : str, optional
            The memory ID to associate with (new parameter)
        memory_unit : MemoryUnit, optional
            The memory unit object to store (new parameter)

        Returns:
        --------
        str
            The ID of the inserted/updated document
        """
        # Handle new calling style (memory_unit + memory_id)
        if memory_unit is not None:
            # Convert memory_unit to dict
            if isinstance(memory_unit, dict):
                data = memory_unit  # Already a dict, use as-is
            elif hasattr(memory_unit, "model_dump"):
                data = memory_unit.model_dump()
            elif hasattr(memory_unit, "dict"):
                data = memory_unit.dict()
            elif hasattr(memory_unit, "__dict__"):
                data = memory_unit.__dict__
            else:
                # Fallback: try to convert to dict
                data = dict(memory_unit)

            # Add memory_id if provided
            if memory_id:
                data["memory_id"] = memory_id

            # Determine memory_store_type from memory_unit
            if hasattr(memory_unit, "memory_type"):
                memory_store_type = memory_unit.memory_type
            elif "memory_type" in data:
                memory_store_type = data["memory_type"]
            else:
                memory_store_type = MemoryType.CONVERSATION_MEMORY

        # Validate we have required parameters
        if data is None or memory_store_type is None:
            raise ValueError(
                "Either (data, memory_store_type) or (memory_unit) must be provided"
            )

        # ``memory_id`` is part of the public provider contract for both the
        # legacy ``data=`` form and the newer ``memory_unit=`` form.  The
        # latter already copied it above, but the former silently dropped the
        # scope and made scoped vector retrieval return no rows.  Keep an
        # explicitly supplied value in ``data`` authoritative, matching the
        # filesystem provider's compatibility behavior.
        if memory_id is not None and isinstance(data, dict):
            data = dict(data)
            data.setdefault("memory_id", memory_id)

        # Ensure memory_store_type is MemoryType enum
        if isinstance(memory_store_type, str):
            memory_store_type = MemoryType(memory_store_type)

        if memory_store_type == MemoryType.MEMAGENT:
            from ...memagent import MemAgentModel

            agent = MemAgentModel(**data)
            return self.store_memagent(agent)
        elif memory_store_type == MemoryType.PERSONAS:
            return self._store_persona(data)
        elif memory_store_type == MemoryType.TOOLBOX:
            return self._store_toolbox(data)
        elif memory_store_type == MemoryType.SKILLBOX:
            return self._store_skillbox(data)
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self._store_conversation_memory(data)
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            return self._store_knowledge_base(data)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self._store_short_term_memory(data)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self._store_workflow_memory(data)
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            return self._store_shared_memory(data)
        elif memory_store_type == MemoryType.SUMMARIES:
            return self._store_summary(data)
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            return self._store_semantic_cache(data)
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            return self._store_entity_memory(data)
        elif memory_store_type == MemoryType.TOOL_LOG:
            return self._store_tool_log(data)
        else:
            raise ValueError(f"Unsupported memory type: {memory_store_type}")

    # ===== BASE-TABLE STORAGE METHODS =====

    def _store_persona(self, data: Dict[str, Any]) -> str:
        """Store persona directly in its base table."""
        persona_id = (
            data.get("persona_id")
            or data.get("personaId")
            or data.get("name")
            or str(uuid.uuid4())
        )
        role_value = data.get("role") or data.get("role_type") or data.get("roleType")
        if hasattr(role_value, "value"):
            role_value = role_value.value

        name = data.get("name", "Unnamed Persona")
        background = data.get("background", "")
        goals = data.get("goals", "")
        persona_text = f"{name}: {background} {goals}".strip()
        embedding = self._generate_embedding_if_needed(
            persona_text, existing_embedding=data.get("embedding")
        )

        table_name = self._get_table_name(MemoryType.PERSONAS)
        traits = data.get("traits")
        expertise = data.get("expertise")
        memory_id = data.get("memory_id") or data.get("memoryId")
        agent_id = data.get("agent_id") or data.get("agentId")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM {table_name} WHERE persona_id = :persona_id",
                {"persona_id": persona_id},
            )
            existing = cursor.fetchone()

            common_params = {
                "persona_id": persona_id,
                "name": name,
                "role_type": role_value,
                "background": background,
                "memory_id": memory_id,
                "agent_id": agent_id,
                "traits": json.dumps(self._sanitize_for_json(traits))
                if traits is not None
                else None,
                "expertise": json.dumps(self._sanitize_for_json(expertise))
                if expertise is not None
                else None,
            }
            if embedding is not None:
                common_params["embedding"] = self._prepare_vector_value(embedding)

            if existing:
                set_parts = [
                    "name = :name",
                    "role_type = :role_type",
                    "background = :background",
                    "memory_id = :memory_id",
                    "agent_id = :agent_id",
                    "traits = :traits",
                    "expertise = :expertise",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                if embedding is not None:
                    set_parts.append("embedding = :embedding")
                cursor.execute(
                    f"UPDATE {table_name} SET {', '.join(set_parts)} "
                    "WHERE persona_id = :persona_id",
                    common_params,
                )
            else:
                cols = [
                    "id",
                    "persona_id",
                    "name",
                    "role_type",
                    "background",
                    "memory_id",
                    "agent_id",
                    "traits",
                    "expertise",
                ]
                vals = [
                    ":id",
                    ":persona_id",
                    ":name",
                    ":role_type",
                    ":background",
                    ":memory_id",
                    ":agent_id",
                    ":traits",
                    ":expertise",
                ]
                params = dict(common_params)
                params["id"] = uuid.uuid4().bytes
                if embedding is not None:
                    cols.append("embedding")
                    vals.append(":embedding")
                cursor.execute(
                    f"INSERT INTO {table_name} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)})",
                    params,
                )
            conn.commit()

        return persona_id

    def _store_toolbox(self, data: Dict[str, Any]) -> str:
        """Store toolbox entry directly in its base table."""
        tool_id = (
            data.get("tool_id")
            or data.get("toolId")
            or data.get("_id")
            or data.get("name")
            or str(uuid.uuid4())
        )
        # Callers (e.g. Toolbox.register_tool) may hand us a uuid.UUID; oracledb
        # can't bind that type (DPY-3002), so coerce the id to text.
        tool_id = str(tool_id)
        tool_type = (
            data.get("tool_type")
            or data.get("toolType")
            or data.get("type")
            or "function"
        )

        name = data.get("name", "unknown_tool")
        description = data.get("description", "")
        signature = data.get("signature", "")
        docstring = data.get("docstring", "")
        parameters = data.get("parameters")
        input_schema = data.get("input_schema") or data.get("inputSchema")
        if not isinstance(input_schema, dict):
            if isinstance(parameters, dict) and parameters.get("type") == "object":
                input_schema = dict(parameters)
                parameters = input_schema.get("properties", {})
            else:
                input_schema = {
                    "type": "object",
                    "properties": parameters if isinstance(parameters, dict) else {},
                    "required": list(data.get("required") or []),
                    "additionalProperties": False,
                }
        input_schema = dict(input_schema)
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        input_schema.setdefault("required", list(data.get("required") or []))
        input_schema["additionalProperties"] = False
        memory_id = data.get("memory_id") or data.get("memoryId")
        agent_id = data.get("agent_id") or data.get("agentId")

        tool_text = f"{name}: {description} {signature} {docstring}".strip()
        embedding = self._generate_embedding_if_needed(
            tool_text, existing_embedding=data.get("embedding")
        )

        table_name = self._get_table_name(MemoryType.TOOLBOX)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM {table_name} WHERE tool_id = :tool_id",
                {"tool_id": tool_id},
            )
            existing = cursor.fetchone()

            common_params: Dict[str, Any] = {
                "tool_id": tool_id,
                "name": name,
                "description": description,
                "signature": signature,
                "docstring": docstring,
                "tool_type": tool_type,
                "parameters": json.dumps(self._sanitize_for_json(parameters))
                if parameters is not None
                else None,
                "input_schema": json.dumps(self._sanitize_for_json(input_schema)),
                "tool_policy": json.dumps(
                    self._sanitize_for_json(data.get("tool_policy") or {})
                ),
                "aliases": json.dumps(
                    self._sanitize_for_json(data.get("aliases") or [])
                ),
                "deprecated_arguments": json.dumps(
                    self._sanitize_for_json(data.get("deprecated_arguments") or {})
                ),
                "queries": json.dumps(
                    self._sanitize_for_json(data.get("queries") or [])
                ),
                "import_reference": data.get("import_reference"),
                "user_id": data.get("user_id"),
                "memory_id": memory_id,
                "agent_id": agent_id,
            }
            if embedding is not None:
                common_params["embedding"] = self._prepare_vector_value(embedding)

            if existing:
                set_parts = [
                    "name = :name",
                    "description = :description",
                    "signature = :signature",
                    "docstring = :docstring",
                    "tool_type = :tool_type",
                    "parameters = :parameters",
                    "input_schema = :input_schema",
                    "tool_policy = :tool_policy",
                    "aliases = :aliases",
                    "deprecated_arguments = :deprecated_arguments",
                    "queries = :queries",
                    "import_reference = :import_reference",
                    "user_id = :user_id",
                    "memory_id = :memory_id",
                    "agent_id = :agent_id",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                if embedding is not None:
                    set_parts.append("embedding = :embedding")
                cursor.execute(
                    f"UPDATE {table_name} SET {', '.join(set_parts)} "
                    "WHERE tool_id = :tool_id",
                    common_params,
                )
            else:
                cols = [
                    "id",
                    "tool_id",
                    "name",
                    "description",
                    "signature",
                    "docstring",
                    "tool_type",
                    "parameters",
                    "input_schema",
                    "tool_policy",
                    "aliases",
                    "deprecated_arguments",
                    "queries",
                    "import_reference",
                    "user_id",
                    "memory_id",
                    "agent_id",
                ]
                vals = [f":{c}" for c in cols]
                params = dict(common_params)
                params["id"] = uuid.uuid4().bytes
                if embedding is not None:
                    cols.append("embedding")
                    vals.append(":embedding")
                cursor.execute(
                    f"INSERT INTO {table_name} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)})",
                    params,
                )
            conn.commit()

        return tool_id

    def _store_skillbox(self, data: Dict[str, Any]) -> str:
        """Store skillbox entry directly in its base table."""
        skill_id = (
            data.get("skill_id")
            or data.get("skillId")
            or data.get("_id")
            or str(uuid.uuid4())
        )
        # Callers may hand us a uuid.UUID; oracledb can't bind that type
        # (DPY-3002), so coerce the id to text.
        skill_id = str(skill_id)

        status = data.get("status") or "candidate"
        if hasattr(status, "value"):
            status = status.value
        injection_role = data.get("injection_role") or "user"
        if hasattr(injection_role, "value"):
            injection_role = injection_role.value

        name = data.get("name", "unknown_skill")
        description = data.get("description", "")
        content = data.get("content", "")
        preconditions = data.get("preconditions")
        tools_used = data.get("tools_used")
        queries = data.get("queries")
        source_workflow_ids = data.get("source_workflow_ids")
        baseline = data.get("baseline")
        stats = data.get("stats")
        version = data.get("version")
        agent_id = data.get("agent_id") or data.get("agentId")

        # Skill documents arrive with an applicability embedding computed by
        # the Skill class (name/description/preconditions/queries — never
        # ``content``); prefer it and only embed the same applicability text
        # when it is missing.
        skill_text = " ".join(
            [
                str(name),
                str(description),
                " ".join(str(p) for p in (preconditions or [])),
                " ".join(str(q) for q in (queries or [])),
            ]
        ).strip()
        embedding = self._generate_embedding_if_needed(
            skill_text, existing_embedding=data.get("embedding")
        )

        table_name = self._get_table_name(MemoryType.SKILLBOX)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM {table_name} WHERE skill_id = :skill_id",
                {"skill_id": skill_id},
            )
            existing = cursor.fetchone()

            common_params: Dict[str, Any] = {
                "skill_id": skill_id,
                "name": name,
                "description": description,
                "content": content,
                "preconditions": json.dumps(self._sanitize_for_json(preconditions))
                if preconditions is not None
                else None,
                "tools_used": json.dumps(self._sanitize_for_json(tools_used))
                if tools_used is not None
                else None,
                "queries": json.dumps(self._sanitize_for_json(queries))
                if queries is not None
                else None,
                "agent_id": agent_id,
                "user_id": data.get("user_id"),
                "source_canonical_hash": data.get("source_canonical_hash"),
                "source_workflow_ids": json.dumps(
                    self._sanitize_for_json(source_workflow_ids)
                )
                if source_workflow_ids is not None
                else None,
                "exemplar_workflow_id": data.get("exemplar_workflow_id"),
                "status": status,
                "injection_role": str(injection_role),
                "version": int(version) if version is not None else 1,
                "promoted_at": self._coerce_timestamp(data.get("promoted_at")),
                "demoted_at": self._coerce_timestamp(data.get("demoted_at")),
                "demotion_reason": data.get("demotion_reason"),
                "baseline": json.dumps(self._sanitize_for_json(baseline))
                if baseline is not None
                else None,
                "stats": json.dumps(self._sanitize_for_json(stats))
                if stats is not None
                else None,
            }
            if embedding is not None:
                common_params["embedding"] = self._prepare_vector_value(embedding)

            if existing:
                set_parts = [
                    "name = :name",
                    "description = :description",
                    "content = :content",
                    "preconditions = :preconditions",
                    "tools_used = :tools_used",
                    "queries = :queries",
                    "agent_id = :agent_id",
                    "user_id = :user_id",
                    "source_canonical_hash = :source_canonical_hash",
                    "source_workflow_ids = :source_workflow_ids",
                    "exemplar_workflow_id = :exemplar_workflow_id",
                    "status = :status",
                    "injection_role = :injection_role",
                    "version = :version",
                    "promoted_at = :promoted_at",
                    "demoted_at = :demoted_at",
                    "demotion_reason = :demotion_reason",
                    "baseline = :baseline",
                    "stats = :stats",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                if embedding is not None:
                    set_parts.append("embedding = :embedding")
                cursor.execute(
                    f"UPDATE {table_name} SET {', '.join(set_parts)} "
                    "WHERE skill_id = :skill_id",
                    common_params,
                )
            else:
                cols = [
                    "id",
                    "skill_id",
                    "name",
                    "description",
                    "content",
                    "preconditions",
                    "tools_used",
                    "queries",
                    "agent_id",
                    "user_id",
                    "source_canonical_hash",
                    "source_workflow_ids",
                    "exemplar_workflow_id",
                    "status",
                    "injection_role",
                    "version",
                    "promoted_at",
                    "demoted_at",
                    "demotion_reason",
                    "baseline",
                    "stats",
                ]
                vals = [f":{c}" for c in cols]
                params = dict(common_params)
                params["id"] = uuid.uuid4().bytes
                if embedding is not None:
                    cols.append("embedding")
                    vals.append(":embedding")
                cursor.execute(
                    f"INSERT INTO {table_name} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)})",
                    params,
                )
            conn.commit()

        return skill_id

    def _upsert_persona_row(
        self,
        cursor,
        *,
        persona_id: str,
        name: str,
        role_type: Optional[str],
        background: str,
        memory_id: Optional[str],
        agent_id: Optional[str],
        embedding: Optional[List[float]],
        traits: Any = None,
        expertise: Any = None,
    ) -> None:
        """Insert or update a personas row using an existing cursor."""
        cursor.execute(
            "SELECT id FROM personas WHERE persona_id = :persona_id",
            {"persona_id": persona_id},
        )
        existing = cursor.fetchone()

        params: Dict[str, Any] = {
            "persona_id": persona_id,
            "name": name,
            "role_type": role_type,
            "background": background,
            "memory_id": memory_id,
            "agent_id": agent_id,
            "traits": json.dumps(self._sanitize_for_json(traits))
            if traits is not None
            else None,
            "expertise": json.dumps(self._sanitize_for_json(expertise))
            if expertise is not None
            else None,
        }
        if embedding is not None:
            params["embedding"] = self._prepare_vector_value(embedding)

        if existing:
            set_parts = [
                "name = :name",
                "role_type = :role_type",
                "background = :background",
                "memory_id = :memory_id",
                "agent_id = :agent_id",
                "traits = :traits",
                "expertise = :expertise",
                "updated_at = CURRENT_TIMESTAMP",
            ]
            if embedding is not None:
                set_parts.append("embedding = :embedding")
            cursor.execute(
                f"UPDATE personas SET {', '.join(set_parts)} "
                "WHERE persona_id = :persona_id",
                params,
            )
        else:
            cols = [
                "id",
                "persona_id",
                "name",
                "role_type",
                "background",
                "memory_id",
                "agent_id",
                "traits",
                "expertise",
            ]
            insert_params = dict(params)
            insert_params["id"] = uuid.uuid4().bytes
            if embedding is not None:
                cols.append("embedding")
            vals = [f":{c}" for c in cols]
            cursor.execute(
                f"INSERT INTO personas ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)})",
                insert_params,
            )

    def _upsert_toolbox_row(
        self,
        cursor,
        *,
        tool_id: str,
        name: str,
        description: str,
        signature: str,
        docstring: str,
        tool_type: str,
        memory_id: Optional[str],
        agent_id: Optional[str],
        embedding: Optional[List[float]],
        parameters: Any = None,
        required: Any = None,
        input_schema: Any = None,
        tool_policy: Any = None,
        aliases: Any = None,
        deprecated_arguments: Any = None,
        queries: Any = None,
        import_reference: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Insert or update one complete, strict toolbox schema."""
        cursor.execute(
            "SELECT id FROM toolbox WHERE tool_id = :tool_id",
            {"tool_id": tool_id},
        )
        existing = cursor.fetchone()

        schema = (
            input_schema
            if isinstance(input_schema, dict)
            else {
                "type": "object",
                "properties": parameters if isinstance(parameters, dict) else {},
                "required": list(required or []),
                "additionalProperties": False,
            }
        )
        schema = dict(schema)
        schema.setdefault("type", "object")
        schema.setdefault(
            "properties", parameters if isinstance(parameters, dict) else {}
        )
        schema.setdefault("required", list(required or []))
        schema["additionalProperties"] = False
        params: Dict[str, Any] = {
            "tool_id": tool_id,
            "name": name,
            "description": description,
            "signature": signature,
            "docstring": docstring,
            "tool_type": tool_type,
            "memory_id": memory_id,
            "agent_id": agent_id,
            "parameters": json.dumps(self._sanitize_for_json(parameters))
            if parameters is not None
            else None,
            "input_schema": json.dumps(self._sanitize_for_json(schema)),
            "tool_policy": json.dumps(self._sanitize_for_json(tool_policy or {})),
            "aliases": json.dumps(self._sanitize_for_json(aliases or [])),
            "deprecated_arguments": json.dumps(
                self._sanitize_for_json(deprecated_arguments or {})
            ),
            "queries": json.dumps(self._sanitize_for_json(queries or [])),
            "import_reference": import_reference,
            "user_id": user_id,
        }
        if embedding is not None:
            # Normalize to array.array so it binds to the VECTOR column (a raw list
            # raises ORA-01484 in thin mode); _prepare_vector_value is the shared helper.
            params["embedding"] = self._prepare_vector_value(embedding)

        if existing:
            set_parts = [
                "name = :name",
                "description = :description",
                "signature = :signature",
                "docstring = :docstring",
                "tool_type = :tool_type",
                "parameters = :parameters",
                "input_schema = :input_schema",
                "tool_policy = :tool_policy",
                "aliases = :aliases",
                "deprecated_arguments = :deprecated_arguments",
                "queries = :queries",
                "import_reference = :import_reference",
                "user_id = :user_id",
                "memory_id = :memory_id",
                "agent_id = :agent_id",
                "updated_at = CURRENT_TIMESTAMP",
            ]
            if embedding is not None:
                set_parts.append("embedding = :embedding")
            cursor.execute(
                f"UPDATE toolbox SET {', '.join(set_parts)} "
                "WHERE tool_id = :tool_id",
                params,
            )
        else:
            cols = [
                "id",
                "tool_id",
                "name",
                "description",
                "signature",
                "docstring",
                "tool_type",
                "parameters",
                "input_schema",
                "tool_policy",
                "aliases",
                "deprecated_arguments",
                "queries",
                "import_reference",
                "user_id",
                "memory_id",
                "agent_id",
            ]
            insert_params = dict(params)
            insert_params["id"] = uuid.uuid4().bytes
            if embedding is not None:
                cols.append("embedding")
            vals = [f":{c}" for c in cols]
            cursor.execute(
                f"INSERT INTO toolbox ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)})",
                insert_params,
            )

    def _store_conversation_memory_impl(self, data: Dict[str, Any]) -> str:
        """Store conversation memory directly in the base table."""
        memory_id = data.get("memory_id") or str(uuid.uuid4())
        thread_value = data.get("thread_id") or data.get("conversation_id")
        embedding = self._generate_embedding_if_needed(
            content=data.get("content", ""), existing_embedding=data.get("embedding")
        )
        table_name = self._get_table_name(MemoryType.CONVERSATION_MEMORY)
        has_user_id = self._memory_type_has_column(
            MemoryType.CONVERSATION_MEMORY, "user_id"
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            row_id = uuid.uuid4().bytes

            columns = ["id", "memory_id", "thread_id", "role", "content", "agent_id"]
            placeholders = [
                ":id",
                ":memory_id",
                ":thread_id",
                ":role",
                ":content",
                ":agent_id",
            ]
            bind_vars: Dict[str, Any] = {
                "id": row_id,
                "memory_id": memory_id,
                "thread_id": thread_value,
                "role": data.get("role"),
                "content": data.get("content"),
                "agent_id": data.get("agent_id"),
            }
            if has_user_id:
                columns.append("user_id")
                placeholders.append(":user_id")
                bind_vars["user_id"] = data.get("user_id")
            if embedding is not None:
                columns.append("embedding")
                placeholders.append(":embedding")
                bind_vars["embedding"] = embedding

            # Let Oracle use DEFAULT CURRENT_TIMESTAMP for the timestamp column.
            sql = (
                f"INSERT INTO {table_name} ("
                + ", ".join(columns)
                + ") VALUES ("
                + ", ".join(placeholders)
                + ")"
            )
            try:
                cursor.execute(sql, bind_vars)
            except Exception as exc:
                error_str = str(exc)
                # Schema drift — if user_id got detected but the actual column
                # vanished between introspection and insert, drop it and retry.
                if (
                    has_user_id
                    and "ORA-00904" in error_str
                    and "USER_ID" in error_str.upper()
                ):
                    self._table_columns_cache.pop(
                        MemoryType.CONVERSATION_MEMORY.value.upper(), None
                    )
                    columns = [c for c in columns if c != "user_id"]
                    placeholders = [p for p in placeholders if p != ":user_id"]
                    bind_vars.pop("user_id", None)
                    sql = (
                        f"INSERT INTO {table_name} ("
                        + ", ".join(columns)
                        + ") VALUES ("
                        + ", ".join(placeholders)
                        + ")"
                    )
                    cursor.execute(sql, bind_vars)
                else:
                    raise
            conn.commit()
            # Return the ROW id (matching MongoDB's inserted_id semantics),
            # not the grouping memory_id. Callers use this as the per-unit
            # handle — e.g. the conversation-embedding backfill updates the
            # row by this id; returning memory_id here made that update
            # match EVERY row in the conversation and overwrite all their
            # embeddings with the last message's vector.
            return str(uuid.UUID(bytes=row_id))

    def _store_conversation_memory(self, data: Dict[str, Any]) -> str:
        """Store conversation memory directly in the base table."""
        return self._store_conversation_memory_impl(data)

    def _insert_base_row(
        self,
        memory_type: MemoryType,
        required_columns: Dict[str, Any],
        optional_columns: Optional[Dict[str, Any]] = None,
        *,
        embedding: Optional[Any] = None,
    ) -> None:
        """Generic helper that writes a row directly to a base table.

        ``required_columns`` always land in the INSERT; entries in
        ``optional_columns`` are dropped when the column does not exist on the
        target table (most commonly ``user_id`` on unmigrated schemas).
        """
        table_name = self._get_table_name(memory_type)
        bind_vars: Dict[str, Any] = dict(required_columns)
        columns = list(required_columns.keys())
        placeholders = [f":{name}" for name in columns]

        optional_columns = optional_columns or {}
        skipped_optional: List[str] = []
        for column, value in optional_columns.items():
            if self._memory_type_has_column(memory_type, column):
                columns.append(column)
                placeholders.append(f":{column}")
                bind_vars[column] = value
            else:
                skipped_optional.append(column)

        if embedding is not None:
            columns.append("embedding")
            placeholders.append(":embedding")
            # Oracle rejects Python lists on plain SQL inserts with
            # ORA-01484 ("arrays can only be bound to PL/SQL statements").
            # Normalize to array.array("f", ...) so it binds as a VECTOR.
            bind_vars["embedding"] = self._prepare_vector_value(embedding)

        sql = (
            f"INSERT INTO {table_name} ("
            + ", ".join(columns)
            + ") VALUES ("
            + ", ".join(placeholders)
            + ")"
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if embedding is not None:
                    # Hint the driver that :embedding is a VECTOR.
                    self._set_vector_input_size(cursor, "embedding")
                cursor.execute(sql, bind_vars)
            except Exception as exc:
                error_str = str(exc).upper()
                # Schema drift safety net: if the column vanished between
                # introspection and insert, drop it and retry once.
                if "ORA-00904" in error_str and optional_columns:
                    self._table_columns_cache.pop(memory_type.value.upper(), None)
                    retry_optional = {
                        k: v
                        for k, v in optional_columns.items()
                        if k.upper() not in error_str
                    }
                    if retry_optional != optional_columns:
                        return self._insert_base_row(
                            memory_type,
                            required_columns,
                            retry_optional,
                            embedding=embedding,
                        )
                # Re-surface unexpected errors so the caller can log context.
                if skipped_optional:
                    logger.debug(
                        "Insert into %s dropped optional columns %s before failing: %s",
                        table_name,
                        skipped_optional,
                        exc,
                    )
                raise
            conn.commit()

    def _store_knowledge_base(self, data: Dict[str, Any]) -> str:
        """Store knowledge base entry directly in its base table."""
        memory_id = data.get("memory_id") or str(uuid.uuid4())
        supplied_record_id = data.get("id") or data.get("_id")
        try:
            record_id = self._normalize_raw_uuid(supplied_record_id)
        except (ValueError, TypeError, AttributeError):
            record_id = uuid.uuid4().bytes
        embedding = self._generate_embedding_if_needed(
            content=data.get("content", ""), existing_embedding=data.get("embedding")
        )
        # Chunking metadata is optional at the storage layer so an installation
        # that hasn't run migration 002 still ingests cleanly (the extra fields
        # are dropped silently). Run 002_knowledge_base_chunking.sql to enable
        # namespace + per-ingest grouping.
        chunk_index = data.get("chunk_index")
        chunk_count = data.get("chunk_count")
        self._insert_base_row(
            MemoryType.KNOWLEDGE_BASE,
            required_columns={
                "id": record_id,
                "memory_id": memory_id,
                "content": data.get("content"),
                "memory_type": data.get("memory_type"),
                "importance": data.get("importance", 1.0),
                "last_accessed": data.get("last_accessed"),
                "access_count": data.get("access_count", 0),
                "agent_id": data.get("agent_id"),
            },
            optional_columns={
                "user_id": data.get("user_id"),
                "knowledge_base_id": data.get("knowledge_base_id"),
                "namespace": data.get("namespace"),
                "chunk_index": int(chunk_index) if chunk_index is not None else None,
                "chunk_count": int(chunk_count) if chunk_count is not None else None,
                "chunking_strategy": data.get("chunking_strategy"),
                "source_id": data.get("source_id"),
                "parent_source_id": data.get("parent_source_id"),
                "linked_source_ids": json.dumps(
                    list(data.get("linked_source_ids") or []), ensure_ascii=False
                ),
                "metadata": json.dumps(
                    data.get("metadata") or {}, ensure_ascii=False, sort_keys=True
                ),
            },
            embedding=embedding,
        )
        # ``memory_id`` groups many records. Return the physical row identity,
        # matching the MemoryProvider contract and enabling reliable MCP
        # get/update/delete round trips for a single record.
        return str(uuid.UUID(bytes=record_id))

    def _store_short_term_memory(self, data: Dict[str, Any]) -> str:
        """Store short-term memory directly in its base table."""
        memory_id = data.get("memory_id") or str(uuid.uuid4())
        supplied_record_id = data.get("id") or data.get("_id")
        try:
            record_id = self._normalize_raw_uuid(supplied_record_id)
        except (ValueError, TypeError, AttributeError):
            record_id = uuid.uuid4().bytes
        embedding = self._generate_embedding_if_needed(
            content=data.get("content", ""), existing_embedding=data.get("embedding")
        )
        self._insert_base_row(
            MemoryType.SHORT_TERM_MEMORY,
            required_columns={
                "id": record_id,
                "memory_id": memory_id,
                "content": data.get("content"),
                "memory_type": data.get("memory_type"),
                "ttl": data.get("ttl"),
                "agent_id": data.get("agent_id"),
                "expires_at": data.get("expires_at"),
            },
            optional_columns={"user_id": data.get("user_id")},
            embedding=embedding,
        )
        return str(uuid.UUID(bytes=record_id))

    def _store_workflow_memory(self, data: Dict[str, Any]) -> str:
        """Store workflow memory directly in its base table."""
        workflow_id = data.get("workflow_id") or str(uuid.uuid4())

        # Embed the workflow's *stable identity* (name + description) so it is
        # retrievable by intent via ``retrieve_workflows_by_query`` (which runs a
        # VECTOR_DISTANCE search over this column). We deliberately do NOT embed
        # steps/outcome — those carry volatile, arbitrary tool output.
        embed_text = " ".join(
            str(x) for x in (data.get("name"), data.get("description")) if x
        ).strip()
        embedding = self._generate_embedding_if_needed(
            embed_text, existing_embedding=data.get("embedding")
        )

        step_count = data.get("step_count")
        self._insert_base_row(
            MemoryType.WORKFLOW_MEMORY,
            required_columns={
                "id": uuid.uuid4().bytes,
                "workflow_id": workflow_id,
                "name": data.get("name"),
                "description": data.get("description"),
                "current_step": data.get("current_step", 0),
                "status": data.get("status", "pending"),
                "memory_id": data.get("memory_id"),
                "agent_id": data.get("agent_id"),
            },
            optional_columns={
                "user_id": data.get("user_id"),
                "user_query": data.get("user_query"),
                "canonical_hash": data.get("canonical_hash"),
                "step_count": int(step_count) if step_count is not None else None,
                "promoted_skill_id": data.get("promoted_skill_id"),
            },
            embedding=embedding,
        )

        # Structured workflow fields are JSON-constrained CLOB columns and
        # go in a follow-up UPDATE.
        if (
            data.get("steps")
            or data.get("outcome")
            or data.get("canonical_signature")
            or data.get("skills_activated")
            or data.get("shadow_evaluations")
        ):
            with self._get_connection() as conn:
                cursor = conn.cursor()
                update_parts = []
                params = {"workflow_id": workflow_id}

                if data.get("steps"):
                    update_parts.append("steps = :steps")
                    params["steps"] = self._ensure_json_text(data["steps"])

                if data.get("outcome"):
                    update_parts.append("outcome = :outcome")
                    params["outcome"] = self._ensure_json_text(data["outcome"])

                if data.get("canonical_signature"):
                    update_parts.append("canonical_signature = :canonical_signature")
                    params["canonical_signature"] = self._ensure_json_text(
                        data["canonical_signature"]
                    )

                if data.get("skills_activated"):
                    update_parts.append("skills_activated = :skills_activated")
                    params["skills_activated"] = self._ensure_json_text(
                        data["skills_activated"]
                    )

                if data.get("shadow_evaluations"):
                    update_parts.append("shadow_evaluations = :shadow_evaluations")
                    params["shadow_evaluations"] = self._ensure_json_text(
                        data["shadow_evaluations"]
                    )

                if update_parts:
                    cursor.execute(
                        f"""
                        UPDATE workflow_memory
                        SET {', '.join(update_parts)}
                        WHERE workflow_id = :workflow_id
                    """,
                        params,
                    )
                    conn.commit()

        return workflow_id

    def _store_shared_memory(self, data: Dict[str, Any]) -> str:
        """Store shared memory directly to base table."""
        memory_id = data.get("memory_id") or str(uuid.uuid4())
        table_name = self._get_table_name(MemoryType.SHARED_MEMORY)
        immutable = (
            data.get("immutable_trace") is True
            and data.get("record_type") == "observability_trace_bundle"
        )

        # Sanitize content to handle JsonId objects
        content = data.get("content")
        if content is not None:
            if isinstance(content, str):
                # Already a string, use as-is
                pass
            else:
                # Convert to JSON string
                content = self._ensure_json_text(self._sanitize_for_json(content))

        # Get embedding from data if provided (don't auto-generate for shared memory)
        # Shared memory typically doesn't need embeddings, and they cause binding issues
        embedding = data.get("embedding")

        # Convert embedding to array.array for Oracle if present
        # Oracle requires arrays to be array.array type, not Python lists
        if embedding is not None and not isinstance(embedding, array.array):
            if isinstance(embedding, list):
                embedding = array.array("f", embedding)
            else:
                # If it's already an array.array or other type, use as-is
                pass

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if record exists
            cursor.execute(
                f"SELECT id FROM {table_name} WHERE memory_id = :memory_id",
                {"memory_id": memory_id},
            )
            existing = cursor.fetchone()

            if existing:
                if immutable:
                    return memory_id
                # Update existing record
                update_fields = ["content = :content", "updated_at = SYSTIMESTAMP"]
                params = {"memory_id": memory_id, "content": content}

                if embedding is not None:
                    update_fields.append("embedding = :embedding")
                    params["embedding"] = embedding
                    # Tell Oracle that embedding is an array type (VECTOR type)
                    try:
                        cursor.setinputsizes(embedding=oracledb.DB_TYPE_VECTOR)
                    except AttributeError:
                        # Fallback: use array type if DB_TYPE_VECTOR doesn't exist
                        cursor.setinputsizes(embedding=array.array)

                if data.get("access_list"):
                    access_list_json = self._ensure_json_text(
                        self._sanitize_for_json(data["access_list"])
                    )
                    update_fields.append("access_list = :access_list")
                    params["access_list"] = access_list_json

                cursor.execute(
                    f"""
                    UPDATE {table_name}
                    SET {', '.join(update_fields)}
                    WHERE memory_id = :memory_id
                """,
                    params,
                )
            else:
                # Insert new record
                id_bytes = (
                    uuid.uuid5(uuid.NAMESPACE_URL, "memorizz:trace:" + memory_id).bytes
                    if immutable
                    else uuid.uuid4().bytes
                )
                insert_fields = [
                    "id",
                    "memory_id",
                    "content",
                    "memory_type",
                    "scope",
                    "owner_agent_id",
                    "created_at",
                    "updated_at",
                ]
                insert_values = [
                    ":id",
                    ":memory_id",
                    ":content",
                    ":memory_type",
                    ":scope",
                    ":owner_agent_id",
                    "SYSTIMESTAMP",
                    "SYSTIMESTAMP",
                ]
                params = {
                    "id": id_bytes,
                    "memory_id": memory_id,
                    "content": content,
                    "memory_type": data.get(
                        "memory_type", MemoryType.SHARED_MEMORY.value
                    ),
                    "scope": data.get("scope", "global"),
                    "owner_agent_id": data.get("owner_agent_id"),
                }

                if embedding is not None:
                    insert_fields.append("embedding")
                    insert_values.append(":embedding")
                    params["embedding"] = embedding
                    # Tell Oracle that embedding is an array type (VECTOR type)
                    try:
                        cursor.setinputsizes(embedding=oracledb.DB_TYPE_VECTOR)
                    except AttributeError:
                        # Fallback: use array type if DB_TYPE_VECTOR doesn't exist
                        cursor.setinputsizes(embedding=array.array)

                if data.get("access_list"):
                    access_list_json = self._ensure_json_text(
                        self._sanitize_for_json(data["access_list"])
                    )
                    insert_fields.append("access_list")
                    insert_values.append(":access_list")
                    params["access_list"] = access_list_json

                try:
                    cursor.execute(
                        f"INSERT INTO {table_name} ({', '.join(insert_fields)}) VALUES ({', '.join(insert_values)})",
                        params,
                    )
                except oracledb.IntegrityError as exc:
                    if not immutable or getattr(exc.args[0], "code", None) != 1:
                        raise
                    # Deterministic RAW primary key serializes concurrent retries.
                    conn.rollback()

            conn.commit()

        return memory_id

    def _store_summary(self, data: Dict[str, Any]) -> str:
        """Atomically store a summary, its source links, and compaction markers."""
        summary_id = str(data.get("summary_id") or str(uuid.uuid4()))
        source_message_ids = list(
            data.get("source_message_ids") or data.get("original_memory_ids") or []
        )
        embedding = self._generate_embedding_if_needed(
            content=data.get("content", ""), existing_embedding=data.get("embedding")
        )
        table_name = self._get_table_name(MemoryType.SUMMARIES)
        conversation_table = self._get_table_name(MemoryType.CONVERSATION_MEMORY)
        links_table = f"{self.config.schema}.summary_message_links"
        values: Dict[str, Any] = {
            "id": uuid.uuid4().bytes,
            "summary_id": summary_id,
            "content": data.get("content"),
            "source_message_ids": json.dumps(source_message_ids),
            "summary_type": data.get("summary_type", "general"),
            "memory_id": data.get("memory_id"),
            "agent_id": data.get("agent_id"),
            "user_id": data.get("user_id"),
            "thread_id": data.get("thread_id"),
            "period_start": data.get("period_start"),
            "period_end": data.get("period_end"),
            "memory_units_count": data.get(
                "memory_units_count", len(source_message_ids)
            ),
        }
        columns = list(values)
        if embedding is not None:
            columns.append("embedding")
            values["embedding"] = self._prepare_vector_value(embedding)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if embedding is not None:
                    self._set_vector_input_size(cursor, "embedding")
                cursor.execute(
                    f"INSERT INTO {table_name} ({', '.join(columns)}) "
                    f"VALUES ({', '.join(':' + column for column in columns)})",
                    values,
                )

                for position, message_id in enumerate(source_message_ids):
                    try:
                        raw_message_id = uuid.UUID(str(message_id)).bytes
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid source message ID '{message_id}'"
                        ) from exc
                    predicates = ["id = :message_id", "summary_id IS NULL"]
                    marker_params: Dict[str, Any] = {
                        "message_id": raw_message_id,
                        "summary_id": summary_id,
                    }
                    if data.get("memory_id") is not None:
                        predicates.append("memory_id = :summary_memory_id")
                        marker_params["summary_memory_id"] = data.get("memory_id")
                    if data.get("user_id") is None:
                        predicates.append("user_id IS NULL")
                    else:
                        predicates.append("user_id = :summary_user_id")
                        marker_params["summary_user_id"] = data.get("user_id")
                    cursor.execute(
                        f"UPDATE {conversation_table} SET summary_id = :summary_id "
                        f"WHERE {' AND '.join(predicates)}",
                        marker_params,
                    )
                    if cursor.rowcount != 1:
                        raise ValueError(
                            "Source message was missing, out of scope, or already "
                            f"compacted: {message_id}"
                        )
                    cursor.execute(
                        f"INSERT INTO {links_table} "
                        "(summary_id, message_id, position) "
                        "VALUES (:summary_id, :message_id, :position)",
                        {
                            "summary_id": summary_id,
                            "message_id": raw_message_id,
                            "position": position,
                        },
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return summary_id

    def store_summary_with_links(self, data: Dict[str, Any]) -> str:
        """Public atomic compaction primitive used by ``MemoryManager``."""
        return self._store_summary(data)

    def _store_semantic_cache(self, data: Dict[str, Any]) -> str:
        """Store semantic cache entry directly in its base table."""
        cache_key = data.get("cache_key") or str(uuid.uuid4())
        embedding = self._generate_embedding_if_needed(
            content=data.get("query_text", ""), existing_embedding=data.get("embedding")
        )
        required: Dict[str, Any] = {
            "id": uuid.uuid4().bytes,
            "cache_key": cache_key,
            "query_text": data.get("query_text"),
            "response": data.get("response"),
            "scope": data.get("scope", "global"),
            "similarity_threshold": data.get("similarity_threshold", 0.8),
            "hit_count": data.get("hit_count", 0),
            "agent_id": data.get("agent_id"),
            "expires_at": data.get("expires_at"),
        }
        optional: Dict[str, Any] = {
            "memory_id": data.get("memory_id"),
            "session_id": data.get("session_id"),
            "user_id": data.get("user_id"),
            "metadata": self._ensure_json_text(data.get("metadata") or {}),
        }
        # Cache keys are deterministic per query/scope. Refresh an existing
        # row instead of violating the unique constraint or leaving stale
        # persistent metadata when the same answer is admitted again.
        refresh_payload = {
            **{
                key: value
                for key, value in required.items()
                if key not in {"id", "cache_key"}
            },
            **optional,
            "embedding": embedding,
        }
        if self._update_semantic_cache_by_key(str(cache_key), refresh_payload):
            return str(cache_key)
        try:
            self._insert_base_row(
                MemoryType.SEMANTIC_CACHE,
                required_columns=required,
                optional_columns=optional,
                embedding=embedding,
            )
        except Exception as exc:
            # Another process may have admitted the same deterministic key
            # between the UPDATE and INSERT. Resolve that race as an upsert;
            # all other failures retain their original traceback.
            if "ORA-00001" not in str(
                exc
            ).upper() or not self._update_semantic_cache_by_key(
                str(cache_key), refresh_payload
            ):
                raise
        return cache_key

    def _store_entity_memory(self, data: Dict[str, Any]) -> str:
        """Store entity memory directly in the base table."""
        entity_id = data.get("entity_id") or str(uuid.uuid4())
        record_id = data.get("id") or data.get("_id")
        if record_id:
            try:
                record_id = self._normalize_raw_uuid(record_id)
            except (ValueError, TypeError, AttributeError):
                record_id = uuid.uuid4().bytes
        else:
            record_id = uuid.uuid4().bytes

        attributes = self._ensure_json_text(data.get("attributes"))
        relations = self._ensure_json_text(data.get("relations"))
        metadata = self._ensure_json_text(data.get("metadata"))

        embedding = data.get("embedding")
        if isinstance(embedding, list):
            embedding = array.array("f", embedding)

        params = {
            "id": record_id,
            "entity_id": entity_id,
            "name": data.get("name"),
            "entity_type": data.get("entity_type"),
            "attributes": attributes,
            "relations": relations,
            "metadata": metadata,
            "memory_id": data.get("memory_id"),
            "agent_id": data.get("agent_id"),
            "embedding": embedding,
        }

        table_name = self._get_table_name(MemoryType.ENTITY_MEMORY)
        include_user_id = self._memory_type_has_column(
            MemoryType.ENTITY_MEMORY, "user_id"
        )
        if include_user_id:
            params["user_id"] = data.get("user_id")

        user_id_update = (
            ",\n                user_id = :user_id" if include_user_id else ""
        )
        user_id_insert_col = ", user_id" if include_user_id else ""
        user_id_insert_val = ", :user_id" if include_user_id else ""

        merge_sql = f"""
            MERGE INTO {table_name} tgt
            USING (SELECT :entity_id AS entity_id FROM dual) src
            ON (tgt.entity_id = src.entity_id)
            WHEN MATCHED THEN UPDATE SET
                name = :name,
                entity_type = :entity_type,
                attributes = :attributes,
                relations = :relations,
                metadata = :metadata,
                memory_id = :memory_id,
                agent_id = :agent_id{user_id_update},
                embedding = :embedding,
                updated_at = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN INSERT (
                id, entity_id, name, entity_type, attributes,
                relations, metadata, memory_id, agent_id{user_id_insert_col},
                embedding, created_at, updated_at
            ) VALUES (
                :id, :entity_id, :name, :entity_type, :attributes,
                :relations, :metadata, :memory_id, :agent_id{user_id_insert_val},
                :embedding, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            scope_columns = "memory_id, user_id" if include_user_id else "memory_id"
            scope_sql = f"""
                SELECT {scope_columns}
                FROM {table_name}
                WHERE entity_id = :entity_id
                FOR UPDATE
            """

            def _read_and_validate_scope() -> Any:
                cursor.execute(scope_sql, {"entity_id": entity_id})
                scope = cursor.fetchone()
                if not scope:
                    return None
                existing_memory_id = scope[0]
                existing_user_id = scope[1] if include_user_id else None
                memory_scope_changed = existing_memory_id != params.get("memory_id")
                user_scope_changed = include_user_id and existing_user_id != params.get(
                    "user_id"
                )
                if memory_scope_changed or user_scope_changed:
                    raise PermissionError(
                        "Entity ownership mismatch: entity_id already belongs to "
                        "a different memory or user scope"
                    )
                return scope

            _read_and_validate_scope()

            def _execute_merge() -> None:
                try:
                    cursor.execute(merge_sql, params)
                except Exception as merge_exc:
                    if params.get(
                        "embedding"
                    ) is not None and self._is_embedding_dimension_mismatch_error(
                        str(merge_exc)
                    ):
                        params["embedding"] = None
                        self._handle_embedding_dimension_mismatch(
                            "Entity memory", merge_exc
                        )
                        cursor.execute(merge_sql, params)
                        return
                    raise

            try:
                _execute_merge()
            except Exception as exc:
                if "ORA-00001" not in str(exc).upper():
                    raise
                # A concurrent request may have inserted the same canonical
                # entity after our initial SELECT. Retry only after the winning
                # row is visible and its exact tenant ownership is validated.
                # A collision in another scope raises PermissionError above.
                if not _read_and_validate_scope():
                    raise
                _execute_merge()
            conn.commit()

        return entity_id

    def _store_tool_log(self, data: Dict[str, Any]) -> str:
        """Store tool log directly in the base table."""
        tool_log_id = data.get("tool_log_id") or str(uuid.uuid4())
        return self._store_tool_log_impl(data, tool_log_id)

    def _store_tool_log_impl(self, data: Dict[str, Any], tool_log_id: str) -> str:
        """Insert directly into the tool_log base table."""
        record_id = uuid.uuid4().bytes
        required: Dict[str, Any] = {
            "id": record_id,
            "tool_log_id": tool_log_id,
            "tool_name": data.get("tool_name", ""),
            "arguments": data.get("arguments", ""),
            "result": data.get("result", ""),
            "success": 1 if data.get("success", True) else 0,
            "error": data.get("error"),
            "outcome": data.get("outcome")
            or ("success" if data.get("success", True) else "error"),
            "outcome_details": json.dumps(
                data.get("outcome_details") or {},
                ensure_ascii=False,
                default=str,
            ),
            "timestamp": self._coerce_timestamp_bind(data.get("timestamp")),
            "agent_id": data.get("agent_id"),
            "tool_call_id": data.get("tool_call_id", ""),
            "thread_id": data.get("thread_id", ""),
            "memory_id": data.get("memory_id", ""),
        }
        self._insert_base_row(
            MemoryType.TOOL_LOG,
            required_columns=required,
            optional_columns={"user_id": data.get("user_id")},
        )
        return tool_log_id

    @staticmethod
    def _coerce_timestamp_bind(value: Any) -> Any:
        """Return a value safe to bind into an Oracle TIMESTAMP column.

        Python ``datetime`` objects map natively via oracledb. ISO-8601 strings
        (what ``memory_manager`` emits) trigger ORA-01843 because Oracle falls
        back to ``NLS_DATE_FORMAT`` for implicit string→TIMESTAMP conversion,
        which doesn't understand ``"2026-04-17T12:34:56+00:00"``.
        """
        from datetime import datetime

        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            # ``fromisoformat`` accepts offsets in 3.11+; strip a trailing Z
            # for broader support.
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                return None
        return value

    def retrieve_by_query(
        self,
        query: Union[Dict[str, Any], str],
        memory_store_type: MemoryType = None,
        limit: int = 1,
        include_embedding: bool = False,
        memory_id: str = None,
        memory_type: Union[str, MemoryType] = None,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve documents from Oracle by query.

        Parameters:
        -----------
        query : Union[Dict[str, Any], str]
            The query to use for retrieval
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_type : Union[str, MemoryType], optional
            The type of memory store (new parameter, takes precedence)
        memory_id : str, optional
            Filter results to specific memory_id
        limit : int
            The maximum number of documents to return
        include_embedding : bool
            Whether to include the embedding field in the results

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved documents, or None if not found
        """
        # Handle new calling style: memory_type takes precedence over memory_store_type
        if memory_type is not None:
            if isinstance(memory_type, str):
                memory_store_type = MemoryType(memory_type)
            else:
                memory_store_type = memory_type

        if memory_store_type is None:
            raise ValueError("Either memory_store_type or memory_type must be provided")

        # Extract user_id up-front so we can apply it consistently across every
        # branch (dict queries, vector search, legacy cache paths, etc.).
        user_id_filter = kwargs.pop("user_id", _UNSET)

        # If memory_id filter is provided, add it to the query
        if memory_id is not None:
            if isinstance(query, dict):
                query = {**query, "memory_id": memory_id}
            else:
                # For string queries (semantic search), store memory_id for filtering
                kwargs["memory_id"] = memory_id

        # Fold user_id into the query/kwargs for downstream helpers that expect
        # the filter as part of the query dict (relational paths) or as an
        # explicit keyword (vector-search paths).
        if user_id_filter is not _UNSET:
            if isinstance(query, dict) and _memory_type_supports_user_id(
                memory_store_type
            ):
                query = {**query, "user_id": user_id_filter}
            kwargs["user_id"] = user_id_filter

        # Handle special cases with vector search
        if memory_store_type == MemoryType.PERSONAS:
            return self.retrieve_persona_by_query(query, limit=limit)
        elif memory_store_type == MemoryType.TOOLBOX:
            return self.retrieve_toolbox_item(
                query,
                limit,
                agent_id=kwargs.get("agent_id", _UNSET),
                user_id=kwargs.get("user_id", _UNSET),
            )
        elif memory_store_type == MemoryType.SKILLBOX:
            return self.retrieve_skillbox_item(
                query,
                limit,
                statuses=kwargs.get("statuses"),
                agent_id=kwargs.get("agent_id", _UNSET),
                user_id=kwargs.get("user_id", _UNSET),
            )
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.retrieve_workflow_by_query(
                query, limit, user_id=kwargs.get("user_id", _UNSET)
            )
        elif memory_store_type == MemoryType.SUMMARIES:
            return self.retrieve_summaries_by_query(
                query, limit, user_id=kwargs.get("user_id", _UNSET)
            )
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            if isinstance(query, dict):
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            from ...embeddings import get_embedding

            try:
                embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for entity query: {e}")
                return []

            return self._vector_search(
                MemoryType.ENTITY_MEMORY,
                embedding,
                limit=limit,
                memory_id=kwargs.get("memory_id"),
                user_id=kwargs.get("user_id", _UNSET),
            )
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            if isinstance(query, dict):
                # Filter query for loading existing cache entries
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            else:
                # String query for semantic similarity search
                return self.find_similar_cache_entries(query, limit=limit, **kwargs)
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            # Knowledge-base retrieval is always semantic. Callers either
            # pass a raw string (we embed it here) or a dict carrying a
            # pre-computed ``embedding`` plus optional scalar filters like
            # ``namespace`` or ``limit``. Routing the dict to
            # ``_retrieve_by_filter`` was wrong — it tried to build
            # ``WHERE embedding = :embedding AND limit = :limit`` and
            # Oracle rejected ``limit`` as an unknown column (ORA-00904).
            namespace_filter: Optional[str] = kwargs.get("namespace") or None
            query_embedding = None
            if isinstance(query, dict):
                query_embedding = query.get("embedding")
                namespace_filter = query.get("namespace") or namespace_filter
                # Allow the caller's dict to override the outer limit arg.
                dict_limit = query.get("limit")
                if isinstance(dict_limit, int) and dict_limit > 0:
                    limit = dict_limit
            elif isinstance(query, str) and query.strip():
                from ...embeddings import get_embedding

                try:
                    query_embedding = get_embedding(query)
                except Exception as exc:
                    logger.error("Failed to embed knowledge_base query: %s", exc)
                    return []

            if query_embedding is None:
                # No embedding to search against — nothing meaningful to return.
                return []

            rows = self._vector_search(
                MemoryType.KNOWLEDGE_BASE,
                query_embedding,
                limit=limit,
                filters=({"namespace": namespace_filter} if namespace_filter else None),
                memory_id=kwargs.get("memory_id"),
                user_id=kwargs.get("user_id", _UNSET),
            )
            return rows
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            # Dict → straight filter; string → episodic semantic recall via
            # VECTOR_DISTANCE over per-turn embeddings (rows without an
            # embedding are excluded by the ``embedding IS NOT NULL`` guard
            # in _vector_search, so unmigrated/unbackfilled rows degrade to
            # "no matches" rather than erroring).
            if isinstance(query, dict):
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            if isinstance(query, str) and query.strip():
                from ...embeddings import get_embedding

                try:
                    query_embedding = get_embedding(query)
                except Exception as exc:
                    logger.error("Failed to embed conversation_memory query: %s", exc)
                    return []
                return self._vector_search(
                    MemoryType.CONVERSATION_MEMORY,
                    query_embedding,
                    limit=limit,
                    filters=(
                        {"thread_id": str(kwargs["thread_id"])}
                        if kwargs.get("thread_id") is not None
                        else None
                    ),
                    memory_id=kwargs.get("memory_id"),
                    user_id=kwargs.get("user_id", _UNSET),
                )
            return []
        else:
            # Standard query
            if isinstance(query, dict):
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            else:
                return []

    def _retrieve_by_filter(
        self,
        query: Dict[str, Any],
        memory_store_type: MemoryType,
        limit: int,
        include_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents by filter criteria using base tables."""
        table_name = self._get_table_name(memory_store_type)

        # Drop ``user_id`` from the query predicate when the column has not
        # been added yet (unmigrated schemas). Defaults to keeping it so
        # tenant isolation is not silently weakened post-migration.
        query_has_user_id = "user_id" in query
        user_id_col_present = not query_has_user_id or self._memory_type_has_column(
            memory_store_type, "user_id"
        )
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build WHERE clause from query (using actual column names)
            where_clauses = []
            params = {}

            for key, value in query.items():
                if key == "_id":
                    continue
                if key == "user_id" and not user_id_col_present:
                    # Silently drop user_id predicate on unmigrated schemas.
                    continue
                # Tenant isolation: ``user_id=None`` must match SQL ``IS NULL``,
                # not ``= NULL`` (which silently never matches in Oracle).
                if key == "user_id" and value is None:
                    where_clauses.append("user_id IS NULL")
                    continue
                where_clauses.append(f"{key} = :{key}")
                params[key] = value

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Select specific columns based on memory type
            if memory_store_type == MemoryType.SEMANTIC_CACHE:
                cache_columns = [
                    "id",
                    "cache_key",
                    "query_text",
                    "response",
                    "scope",
                    "similarity_threshold",
                    "hit_count",
                    "agent_id",
                ]
                for optional_column in (
                    "memory_id",
                    "session_id",
                    "user_id",
                    "metadata",
                ):
                    if self._memory_type_has_column(
                        MemoryType.SEMANTIC_CACHE, optional_column
                    ):
                        cache_columns.append(optional_column)
                cache_columns.extend(["embedding", "created_at", "expires_at"])
                columns = ", ".join(cache_columns)
            else:
                columns = "*"

            sql = f"""
            SELECT {columns}
            FROM {table_name}
            WHERE {where_sql}
            FETCH FIRST :limit ROWS ONLY
            """

            params["limit"] = limit
            cursor.execute(sql, params)

            results = []
            for row in cursor:
                # Convert row to dict based on memory type
                if memory_store_type == MemoryType.SEMANTIC_CACHE:
                    values = dict(zip(cache_columns, row))
                    created_at = values.get("created_at")
                    timestamp = (
                        created_at.timestamp()
                        if hasattr(created_at, "timestamp")
                        else time.time()
                    )

                    # Handle CLOB columns - they return LOB objects that need to be read
                    raw_query = values.get("query_text")
                    raw_response = values.get("response")
                    query_text = (
                        raw_query.read() if hasattr(raw_query, "read") else raw_query
                    )
                    response = (
                        raw_response.read()
                        if hasattr(raw_response, "read")
                        else raw_response
                    )

                    doc = {
                        "_id": str(uuid.UUID(bytes=values["id"])),
                        "cache_key": values.get("cache_key"),
                        "query_text": query_text,
                        "response": response,
                        "scope": values.get("scope"),
                        "similarity_threshold": (
                            float(values["similarity_threshold"])
                            if values.get("similarity_threshold") is not None
                            else 0.85
                        ),
                        "hit_count": int(values.get("hit_count") or 0),
                        "usage_count": (
                            int(values.get("hit_count") or 0)
                        ),  # Map hit_count to usage_count
                        "agent_id": values.get("agent_id"),
                        "memory_id": values.get("memory_id"),
                        "session_id": values.get("session_id"),
                        "user_id": values.get("user_id"),
                        "metadata": self._deserialize_json_field(values.get("metadata"))
                        or {},
                        "timestamp": timestamp,
                        "created_at": created_at,
                        "expires_at": values.get("expires_at"),
                    }
                    if include_embedding and values.get("embedding") is not None:
                        doc["embedding"] = list(values["embedding"])
                else:
                    # Generic handling for other types
                    columns = [desc[0].lower() for desc in cursor.description]
                    doc = dict(zip(columns, row))
                    if memory_store_type == MemoryType.ENTITY_MEMORY:
                        for key in ("attributes", "relations", "metadata"):
                            doc[key] = self._deserialize_json_field(doc.get(key))
                    if not include_embedding:
                        doc.pop("embedding", None)

                results.append(doc)

            return results if results else None

    def retrieve_by_id(
        self, id: str, memory_store_type: MemoryType
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from Oracle by ID.

        Parameters:
        -----------
        id : str
            The document ID
        memory_store_type : MemoryType
            The type of memory store

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found
        """
        # Special handling for MEMAGENT
        if memory_store_type == MemoryType.MEMAGENT:
            return self.retrieve_memagent(id)

        # Registry-driven types are addressed by their logical string id
        # (persona_id / tool_id / skill_id / workflow_id / tool_log_id).
        # These need dedicated projections because the generic
        # ``SELECT id, data`` fallback below has no ``data`` column to read
        # — without them, by-id loads silently returned None (workflow
        # skill exemplars and the LLM-facing ``retrieve_tool_log_entry``
        # compact-reference pattern in particular broke).
        spec = _BY_ID_SPECS.get(memory_store_type)
        if spec is not None:
            id_column, fields = spec
            bound_id = id
            if id_column == "id":
                try:
                    bound_id = self._normalize_raw_uuid(id)
                except (ValueError, TypeError, AttributeError):
                    return None
            table_name = self._get_table_name(memory_store_type)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT {_select_list(fields)}
                    FROM {table_name}
                    WHERE {id_column} = :{id_column}
                    """,
                    {id_column: bound_id},
                )
                row = cursor.fetchone()
                if row is None and memory_store_type == MemoryType.WORKFLOW_MEMORY:
                    try:
                        row_id = uuid.UUID(str(id)).bytes
                    except (ValueError, TypeError):
                        row_id = None
                    if row_id is not None:
                        cursor.execute(
                            f"""
                            SELECT {_select_list(fields)}
                            FROM {table_name}
                            WHERE id = :row_id
                            """,
                            {"row_id": row_id},
                        )
                        row = cursor.fetchone()
                if not row:
                    return None
                return self._apply_row_fields(fields, row)
        if memory_store_type == MemoryType.ENTITY_MEMORY:
            table_name = self._get_table_name(memory_store_type)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE entity_id = :entity_id
                    """,
                    {"entity_id": id},
                )
                row = cursor.fetchone()
                if row:
                    columns = [desc[0].lower() for desc in cursor.description]
                    record = dict(zip(columns, row))
                    for key in ("attributes", "relations", "metadata"):
                        record[key] = self._deserialize_json_field(record.get(key))
                    return record
                return None

        # Conversation rows: RAW(16) row id, addressed by canonical uuid str.
        # Previously fell into the generic ``SELECT id, data`` fallback below
        # (no ``data`` column exists) and silently returned None — breaking
        # by-id message reconstruction (summary expansion, tenant checks).
        if memory_store_type == MemoryType.CONVERSATION_MEMORY:
            try:
                row_id = (
                    bytes(id)
                    if isinstance(id, (bytes, bytearray))
                    else uuid.UUID(str(id)).bytes
                )
            except (ValueError, TypeError):
                return None
            table_name = self._get_table_name(memory_store_type)
            has_user_id = self._memory_type_has_column(
                MemoryType.CONVERSATION_MEMORY, "user_id"
            )
            has_summary_id = self._memory_type_has_column(
                MemoryType.CONVERSATION_MEMORY, "summary_id"
            )
            optional_columns = []
            if has_user_id:
                optional_columns.append("user_id")
            if has_summary_id:
                optional_columns.append("summary_id")
            optional_projection = (
                ", " + ", ".join(optional_columns) if optional_columns else ""
            )
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        f"""
                        SELECT id, memory_id, thread_id, role, content,
                               timestamp, agent_id{optional_projection}
                        FROM {table_name}
                        WHERE id = :id
                        """,
                        {"id": row_id},
                    )
                except Exception as exc:
                    logger.debug("retrieve_by_id(conversation_memory) failed: %s", exc)
                    return None
                row = cursor.fetchone()
                if not row:
                    return None
                result = {
                    "_id": str(uuid.UUID(bytes=row[0])),
                    "memory_id": row[1],
                    "thread_id": row[2],
                    "role": row[3],
                    "content": self._read_lob_value(row[4]),
                    "timestamp": row[5].isoformat()
                    if hasattr(row[5], "isoformat")
                    else row[5],
                    "agent_id": row[6],
                }
                offset = 7
                if has_user_id:
                    result["user_id"] = row[offset]
                    offset += 1
                if has_summary_id:
                    result["summary_id"] = row[offset]
                return result

        # For shared memory, query base table directly
        if memory_store_type == MemoryType.SHARED_MEMORY:
            table_name = self._get_table_name(memory_store_type)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, memory_id, content, memory_type, scope, owner_agent_id,
                           embedding, created_at, updated_at, access_list
                    FROM {table_name}
                    WHERE memory_id = :memory_id
                    """,
                    {"memory_id": id},
                )
                row = cursor.fetchone()
                if row:
                    # Handle CLOB content
                    content = row[2]
                    if hasattr(content, "read"):
                        content = content.read()
                    elif content:
                        content = self._deserialize_json_field(content)

                    # Handle access_list
                    access_list = row[9] if len(row) > 9 else None
                    if access_list and hasattr(access_list, "read"):
                        access_list = access_list.read()
                    if access_list:
                        access_list = self._deserialize_json_field(access_list)

                    result = {
                        "_id": str(uuid.UUID(bytes=row[0])),
                        "memory_id": row[1],
                        "content": content,
                        "memory_type": row[3],
                        "scope": row[4],
                        "owner_agent_id": row[5],
                        "created_at": row[7].isoformat() if row[7] else None,
                        "updated_at": row[8].isoformat() if row[8] else None,
                    }

                    if row[6] is not None:  # embedding
                        result["embedding"] = list(row[6])

                    if access_list:
                        result["access_list"] = access_list

                    return result
                return None

        # Generic base-table fallback for memory types without a dedicated
        # branch above. Most tables no longer have a ``data`` JSON column,
        # so this path is deliberately conservative — failures return None
        # rather than raise, and specific memory types should add their own
        # branch above when they need reliable by-id lookup.
        table_name = self._get_table_name(memory_store_type)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                doc_id = uuid.UUID(id).bytes
                cursor.execute(
                    f"SELECT id, data FROM {table_name} WHERE id = :id",
                    {"id": doc_id},
                )
                row = cursor.fetchone()
                if row:
                    doc = self._doc_to_dict(row[1])
                    doc["_id"] = str(uuid.UUID(bytes=row[0]))
                    return doc
            except Exception as exc:
                logger.debug(
                    "retrieve_by_id fallback failed for %s/%s: %s",
                    memory_store_type.value,
                    id,
                    exc,
                )
            return None

    def retrieve_by_name(
        self, name: str, memory_store_type: MemoryType, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        if memory_store_type not in self.NAME_FILTER_TYPES:
            return None
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {table_name}
                WHERE name = :name
                FETCH FIRST 1 ROWS ONLY
                """,
                {"name": name},
            )
            row = cursor.fetchone()
            if not row:
                return None
            columns = [desc[0].lower() for desc in cursor.description]
            record = dict(zip(columns, row))
            if memory_store_type == MemoryType.ENTITY_MEMORY:
                for key in ("attributes", "relations", "metadata"):
                    record[key] = self._deserialize_json_field(record.get(key))
            if not include_embedding and "embedding" in record:
                record.pop("embedding", None)
            return record

    def delete_observability_bundle(self, record_id, fingerprint):
        from ...observability.index import digest
        from ...observability.normalization import read_payload

        row = self.retrieve_by_id(record_id, MemoryType.SHARED_MEMORY)
        payload = read_payload(row) if row else None
        if (
            not payload
            or payload.get("record_type") != "observability_trace_bundle"
            or digest(payload) != fingerprint
        ):
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.setinputsizes(expected=oracledb.DB_TYPE_CLOB)
            cursor.execute(
                f"DELETE FROM {self._get_table_name(MemoryType.SHARED_MEMORY)} WHERE memory_id = :record_id AND DBMS_LOB.COMPARE(content, :expected) = 0",
                {"record_id": record_id, "expected": row["content"]},
            )
            removed = cursor.rowcount == 1
            conn.commit()
            return removed

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """Delete a document by ID."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if memory_store_type == MemoryType.PERSONAS:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE persona_id = :persona_id",
                    {"persona_id": id},
                )
                conn.commit()
                return cursor.rowcount > 0

            if memory_store_type == MemoryType.TOOLBOX:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE tool_id = :tool_id",
                    {"tool_id": id},
                )
                conn.commit()
                return cursor.rowcount > 0

            if memory_store_type == MemoryType.SKILLBOX:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE skill_id = :skill_id",
                    {"skill_id": id},
                )
                conn.commit()
                return cursor.rowcount > 0

            if memory_store_type == MemoryType.WORKFLOW_MEMORY:
                params = {"workflow_id": str(id)}
                try:
                    params["row_id"] = uuid.UUID(str(id)).bytes
                except (ValueError, TypeError):
                    cursor.execute(
                        f"DELETE FROM {table_name} " "WHERE workflow_id = :workflow_id",
                        params,
                    )
                else:
                    cursor.execute(
                        f"DELETE FROM {table_name} "
                        "WHERE workflow_id = :workflow_id OR id = :row_id",
                        params,
                    )
                conn.commit()
                return cursor.rowcount > 0

            if memory_store_type == MemoryType.MEMAGENT:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE agent_id = :agent_id",
                    {"agent_id": id},
                )
                conn.commit()
                return cursor.rowcount > 0

            try:
                doc_id = uuid.UUID(id).bytes
            except (ValueError, Exception):
                return False

            cursor.execute(f"DELETE FROM {table_name} WHERE id = :id", {"id": doc_id})
            conn.commit()

            return cursor.rowcount > 0

    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """Delete a document by name."""
        if memory_store_type not in self.NAME_FILTER_TYPES:
            return False
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"DELETE FROM {table_name} WHERE name = :name", {"name": name}
            )
            conn.commit()

            return cursor.rowcount > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """Delete all documents within a memory store type."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()

            return cursor.rowcount > 0

    def list_all(
        self,
        memory_store_type: MemoryType,
        include_embedding: bool = False,
        user_id: Any = _UNSET,
    ) -> List[Dict[str, Any]]:
        """List all documents within a memory store type.

        Parameters:
        -----------
        memory_store_type : MemoryType
            Which memory store to enumerate.
        include_embedding : bool
            Whether to keep ``embedding`` fields in the returned dicts.
        user_id : str, optional
            Tenant scope. When provided (including ``None`` for the
            anonymous/legacy scope), results are restricted to rows whose
            stored ``user_id`` matches. When the sentinel default is used no
            ``user_id`` filter is applied (preserves legacy callers).
        """
        table_name = self._get_table_name(memory_store_type)
        base_results = self._list_all_from_table(
            table_name, memory_store_type, include_embedding
        )
        return self._apply_user_id_filter(base_results, memory_store_type, user_id)

    def list_tool_logs(
        self,
        memory_id: Optional[str] = None,
        user_id: Any = _UNSET,
        limit: int = 20,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recent tool-log rows for a memory/thread (most-recent first).

        Native counterpart to the MongoDB provider's indexed query. Reads the
        tool_log rows via :meth:`list_all` and narrows in Python with the shared
        :func:`filter_tool_log_rows` — behaviour-equivalent to the manager's
        former fallback, plus the ``thread_id`` scoping the digest relies on.
        """
        rows = self.list_all(MemoryType.TOOL_LOG)
        return filter_tool_log_rows(
            rows,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
        )

    @staticmethod
    def _apply_user_id_filter(
        rows: List[Dict[str, Any]],
        memory_store_type: MemoryType,
        user_id: Any,
    ) -> List[Dict[str, Any]]:
        """Strict client-side user_id filter. ``_UNSET`` disables filtering."""
        if user_id is _UNSET:
            return rows
        if not _memory_type_supports_user_id(memory_store_type):
            return rows
        return [row for row in rows if row.get("user_id") == user_id]

    def _list_all_from_table(
        self,
        table_name: str,
        memory_store_type: MemoryType,
        include_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all documents from a base table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Registry-driven types: these tables have individual columns,
            # not a generic ``data`` column, so build the projection and the
            # row→dict mapping from the shared field specs.
            spec = _LIST_SPECS.get(memory_store_type)
            if spec is not None:
                fields, order_by = spec
                if memory_store_type == MemoryType.KNOWLEDGE_BASE:
                    fields = fields + self._knowledge_base_chunk_fields()
                query = f"""
                    SELECT {_select_list(fields)}
                    FROM {table_name}
                """
                if order_by:
                    query += f" ORDER BY {order_by}"
                cursor.execute(query)
                results = []
                for row in cursor:
                    try:
                        results.append(
                            self._apply_row_fields(fields, row, include_embedding)
                        )
                    except Exception as row_error:
                        logger.warning(
                            "Error processing %s row: %s",
                            memory_store_type.value,
                            row_error,
                        )
                        continue
                return results
            if memory_store_type == MemoryType.MEMAGENT:
                # Agents are stored across `agents`, `agent_llm_configs`,
                # `agent_memories`, and `personas`. Reuse ``retrieve_memagent``
                # per row so every agent comes back fully hydrated instead of
                # relying on a (non-existent) ``data`` JSON column.
                try:
                    cursor.execute("SELECT agent_id FROM agents")
                    agent_ids = [row[0] for row in cursor.fetchall() if row and row[0]]
                except Exception as exc:
                    logger.error("Failed to list agent_ids from agents table: %s", exc)
                    return []

                results: List[Dict[str, Any]] = []
                for agent_id in agent_ids:
                    try:
                        agent_model = self.retrieve_memagent(agent_id)
                    except Exception as exc:
                        logger.warning(
                            "Failed to hydrate memagent %s: %s", agent_id, exc
                        )
                        continue
                    if agent_model is None:
                        continue
                    doc = (
                        agent_model.model_dump()
                        if hasattr(agent_model, "model_dump")
                        else agent_model.__dict__
                    )
                    if not include_embedding:
                        doc.pop("embedding", None)
                    results.append(doc)
                return results
            elif memory_store_type in (
                MemoryType.PERSONAS,
                MemoryType.SHORT_TERM_MEMORY,
                MemoryType.SEMANTIC_CACHE,
            ):
                # These tables have type-specific columns, not a generic
                # `data` JSON. They aren't exercised by the UI's list_all
                # path yet, so return [] quietly rather than spamming
                # ORA-00904 on every Connect. Add a proper SELECT branch
                # here when these memory types need full listing.
                logger.debug(
                    "list_all for %s is not implemented for Oracle; returning empty list",
                    memory_store_type.value,
                )
                return []
            else:
                # Unknown memory type — surface rather than swallow.
                try:
                    query = f"SELECT id, data FROM {table_name}"
                    cursor.execute(query)
                    results = []
                    for row in cursor:
                        doc = self._doc_to_dict(row[1], include_embedding)
                        if row[0]:
                            doc["_id"] = str(uuid.UUID(bytes=row[0]))
                        results.append(doc)
                    return results
                except Exception as table_error:
                    logger.error(
                        f"Failed to query base table {table_name}: {table_error}"
                    )
                    return []

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        include_embedding: bool = False,
        memory_type: Union[str, MemoryType] = None,
        limit: int = None,
        user_id: Any = _UNSET,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The memory ID to retrieve history for
        include_embedding : bool
            Whether to include embeddings
        memory_type : Union[str, MemoryType], optional
            Type of memory (defaults to CONVERSATION_MEMORY)
        limit : int, optional
            Maximum number of entries to return
        user_id : str, optional
            Multi-tenant scope. When provided (including explicit ``None``),
            results are restricted to rows whose stored ``user_id`` matches.
            When left at the sentinel default, no user_id filter is applied
            (legacy behavior).
        thread_id : str, optional
            Exact conversation thread to retrieve. When omitted, rows from
            every thread in the memory scope are returned.
        """
        # Default to CONVERSATION_MEMORY if not specified
        if memory_type is None:
            memory_type = MemoryType.CONVERSATION_MEMORY
        elif isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        # Query from base table for proper timestamp ordering
        table_name = self._get_table_name(memory_type)

        # Build an optional user_id predicate. We introspect the column set
        # first so unmigrated schemas (no ``user_id`` yet) are tolerated
        # without noisy retry-on-exception logs.
        params = {"memory_id": memory_id}
        user_clause = ""
        if user_id is not _UNSET and self._memory_type_has_column(
            memory_type, "user_id"
        ):
            if user_id is None:
                user_clause = " AND user_id IS NULL"
            else:
                user_clause = " AND user_id = :user_id_scope"
                params["user_id_scope"] = user_id

        with self._get_connection() as conn:
            cursor = conn.cursor()
            has_user_id = self._memory_type_has_column(memory_type, "user_id")
            has_summary_id = self._memory_type_has_column(memory_type, "summary_id")
            optional_projection = ""
            if has_user_id:
                optional_projection += ", user_id"
            if has_summary_id:
                optional_projection += ", summary_id"

            # Try the current schema first, then tolerate pre-migration
            # deployments that lack user_id and/or still call thread_id
            # conversation_id. Thread filtering stays native in every form.
            attempts = [("thread_id", True)]
            if user_clause:
                attempts.append(("thread_id", False))
            attempts.append(("conversation_id", True))
            if user_clause:
                attempts.append(("conversation_id", False))

            last_error = None
            for thread_column, include_user_scope in attempts:
                attempt_params = {"memory_id": memory_id}
                clauses = ["memory_id = :memory_id"]
                if include_user_scope and user_clause:
                    clauses.append(user_clause.replace(" AND ", "", 1))
                    if "user_id_scope" in params:
                        attempt_params["user_id_scope"] = params["user_id_scope"]
                if thread_id is not None:
                    clauses.append(f"{thread_column} = :thread_id")
                    attempt_params["thread_id"] = str(thread_id)
                sql = f"""
                    SELECT id, memory_id, {thread_column}, role, content,
                           timestamp, agent_id{optional_projection}, embedding
                    FROM {table_name}
                    WHERE {' AND '.join(clauses)}
                    ORDER BY timestamp
                """
                try:
                    cursor.execute(sql, attempt_params)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error

            results = []
            columns = [description[0].lower() for description in cursor.description]
            for row in cursor:
                values = dict(zip(columns, row))
                # Handle CLOB content properly
                content = values.get("content")
                if hasattr(content, "read"):
                    content = content.read()

                # Handle timestamp conversion
                timestamp = values.get("timestamp")
                if hasattr(timestamp, "isoformat"):
                    timestamp = timestamp.isoformat()
                elif timestamp:
                    timestamp = str(timestamp)

                result = {
                    "_id": str(uuid.UUID(bytes=values["id"])),
                    "memory_id": values.get("memory_id"),
                    "thread_id": values.get(thread_column),
                    "role": values.get("role"),
                    "content": content,
                    "timestamp": timestamp,
                    "agent_id": values.get("agent_id"),
                }
                if has_user_id:
                    result["user_id"] = values.get("user_id")
                if has_summary_id:
                    result["summary_id"] = values.get("summary_id")

                if include_embedding and values.get("embedding") is not None:
                    result["embedding"] = list(values["embedding"])

                results.append(result)

            if limit is not None:
                try:
                    limit_value = int(limit)
                except (TypeError, ValueError):
                    limit_value = 0
                if limit_value > 0:
                    return results[-limit_value:]
            return results

    def _update_relational_by_id(
        self, id_value: str, data: Dict[str, Any], memory_store_type: MemoryType
    ) -> bool:
        """Update relational tables directly for memory types that avoid DV updates."""
        update_config = {
            MemoryType.CONVERSATION_MEMORY: {
                # Conversation rows are addressed by their RAW(16) row id —
                # "memory_id" here is the conversation GROUPING id shared by
                # every row of a thread, so using it as the WHERE key made
                # any update (embedding backfill, summary marking) rewrite
                # the entire conversation.
                "id_field": "id",
                "id_is_raw_uuid": True,
                "fields": {
                    "thread_id",
                    "role",
                    "content",
                    "timestamp",
                    "agent_id",
                    "summary_id",
                    "embedding",
                },
                "json_fields": set(),
                "has_updated_at": False,
            },
            MemoryType.KNOWLEDGE_BASE: {
                "id_field": "id",
                "id_is_raw_uuid": True,
                "fields": {
                    "content",
                    "memory_type",
                    "importance",
                    "last_accessed",
                    "access_count",
                    "agent_id",
                    "embedding",
                },
                "json_fields": set(),
                "has_updated_at": True,
            },
            MemoryType.PERSONAS: {
                "id_field": "persona_id",
                "fields": {
                    "name",
                    "role_type",
                    "background",
                    "traits",
                    "expertise",
                    "memory_id",
                    "agent_id",
                    "user_id",
                    "embedding",
                },
                "json_fields": {"traits", "expertise"},
                "has_updated_at": True,
                "field_map": {"role": "role_type"},
            },
            MemoryType.SHORT_TERM_MEMORY: {
                "id_field": "id",
                "id_is_raw_uuid": True,
                "fields": {
                    "content",
                    "memory_type",
                    "ttl",
                    "agent_id",
                    "expires_at",
                    "embedding",
                },
                "json_fields": set(),
                "has_updated_at": False,
            },
            MemoryType.TOOLBOX: {
                "id_field": "tool_id",
                "fields": {
                    "name",
                    "description",
                    "signature",
                    "docstring",
                    "tool_type",
                    "parameters",
                    "input_schema",
                    "tool_policy",
                    "aliases",
                    "deprecated_arguments",
                    "queries",
                    "import_reference",
                    "memory_id",
                    "agent_id",
                    "embedding",
                },
                "json_fields": {
                    "parameters",
                    "input_schema",
                    "tool_policy",
                    "aliases",
                    "deprecated_arguments",
                    "queries",
                },
                "has_updated_at": True,
                "field_map": {"type": "tool_type"},
            },
            MemoryType.SKILLBOX: {
                "id_field": "skill_id",
                "fields": {
                    "status",
                    "injection_role",
                    "version",
                    "promoted_at",
                    "demoted_at",
                    "demotion_reason",
                    "baseline",
                    "stats",
                },
                "json_fields": {"baseline", "stats"},
                "timestamp_fields": {"promoted_at", "demoted_at"},
                "has_updated_at": True,
            },
            MemoryType.WORKFLOW_MEMORY: {
                "id_field": "workflow_id",
                "allow_raw_id": True,
                "fields": {
                    "name",
                    "description",
                    "current_step",
                    "status",
                    "memory_id",
                    "agent_id",
                    "user_query",
                    "steps",
                    "outcome",
                    "canonical_hash",
                    "canonical_signature",
                    "step_count",
                    "promoted_skill_id",
                    "skills_activated",
                    "shadow_evaluations",
                    "embedding",
                },
                "json_fields": {
                    "steps",
                    "outcome",
                    "canonical_signature",
                    "skills_activated",
                    "shadow_evaluations",
                },
                "has_updated_at": True,
            },
            MemoryType.SUMMARIES: {
                "id_field": "summary_id",
                "fields": {
                    "content",
                    "summary_type",
                    "memory_id",
                    "agent_id",
                    "user_id",
                    "source_message_ids",
                    "period_start",
                    "period_end",
                    "memory_units_count",
                    "embedding",
                },
                "json_fields": {"source_message_ids"},
                "has_updated_at": False,
            },
            MemoryType.ENTITY_MEMORY: {
                "id_field": "entity_id",
                "fields": {
                    "name",
                    "entity_type",
                    "attributes",
                    "relations",
                    "metadata",
                    "memory_id",
                    "agent_id",
                    "embedding",
                },
                "json_fields": {"attributes", "relations", "metadata"},
                "has_updated_at": True,
            },
            MemoryType.SEMANTIC_CACHE: {
                "id_field": "cache_key",
                "fields": {
                    "query_text",
                    "response",
                    "scope",
                    "similarity_threshold",
                    "hit_count",
                    "agent_id",
                    "memory_id",
                    "session_id",
                    "user_id",
                    "expires_at",
                    "metadata",
                    "embedding",
                },
                "json_fields": {"metadata"},
                "has_updated_at": False,
            },
            MemoryType.TOOL_LOG: {
                "id_field": "tool_log_id",
                "fields": {
                    "tool_name",
                    "arguments",
                    "result",
                    "success",
                    "error",
                    "timestamp",
                    "agent_id",
                    "tool_call_id",
                    "thread_id",
                    "memory_id",
                },
                "json_fields": set(),
                "has_updated_at": False,
            },
            MemoryType.SHARED_MEMORY: {
                "id_field": "memory_id",
                "fields": {
                    "content",
                    "memory_type",
                    "scope",
                    "owner_agent_id",
                    "access_list",
                    "embedding",
                },
                "json_fields": {"access_list"},
                "has_updated_at": True,
            },
        }

        # ``user_id`` is tenant scope — add it as an allowed field on every
        # memory type that carries the column so callers can update scope
        # during data migrations. ``_maybe_set_user_id`` below checks the
        # actual column presence before emitting the SQL so unmigrated
        # schemas are tolerated.
        for cfg in update_config.values():
            cfg["fields"] = set(cfg["fields"]) | {"user_id"}

        config = update_config.get(memory_store_type)
        if config is None:
            return False

        if not isinstance(data, dict):
            logger.error("Update data must be a dict, got %s", type(data))
            return False

        id_field = config["id_field"]
        allowed_fields = config["fields"]
        json_fields = config["json_fields"]
        timestamp_fields = config.get("timestamp_fields", set())
        field_map = config.get("field_map", {})
        has_updated_at = config["has_updated_at"]

        sanitized_data = self._sanitize_for_json(data)
        if not isinstance(sanitized_data, dict):
            logger.error(
                "Update data is not a dict after sanitization: %s",
                type(sanitized_data),
            )
            return False

        set_clauses = []
        bound_id_value = id_value
        if config.get("id_is_raw_uuid"):
            # RAW(16) primary keys bind as bytes; callers hold the row id in
            # its canonical string form (str(uuid.UUID(bytes=...))).
            try:
                if isinstance(id_value, (bytes, bytearray)):
                    bound_id_value = bytes(id_value)
                else:
                    bound_id_value = uuid.UUID(str(id_value)).bytes
            except (ValueError, AttributeError, TypeError):
                logger.debug(
                    "update_by_id: %r is not a valid row uuid for %s — no-op",
                    id_value,
                    memory_store_type.value,
                )
                return False
        params = {id_field: bound_id_value}
        where_clause = f"{id_field} = :{id_field}"
        if config.get("allow_raw_id"):
            try:
                params["row_id"] = uuid.UUID(str(id_value)).bytes
            except (ValueError, AttributeError, TypeError):
                pass
            else:
                where_clause = f"({where_clause} OR id = :row_id)"
        skip_user_id = "user_id" in sanitized_data and not self._memory_type_has_column(
            memory_store_type, "user_id"
        )

        for key, value in sanitized_data.items():
            mapped_key = field_map.get(key, key)
            if mapped_key in ("id", "_id", id_field):
                continue
            if mapped_key not in allowed_fields:
                continue
            if mapped_key == "user_id" and skip_user_id:
                # Column not yet added — silently drop to tolerate pre-migration schemas.
                continue
            if mapped_key in json_fields:
                value = self._ensure_json_text(value)
            elif mapped_key in timestamp_fields:
                # _sanitize_for_json serialized any datetime to an ISO string;
                # TIMESTAMP columns need a real datetime bind.
                value = self._coerce_timestamp(value)
            elif mapped_key == "content" and not isinstance(value, str):
                value = self._ensure_json_text(value)

            if mapped_key == "embedding":
                value = self._prepare_vector_value(value)
                params["embedding"] = value
                set_clauses.append("embedding = :embedding")
                continue

            params[mapped_key] = value
            set_clauses.append(f"{mapped_key} = :{mapped_key}")

        if not set_clauses:
            return False

        if has_updated_at:
            set_clauses.append("updated_at = SYSTIMESTAMP")

        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if "embedding" in params:
                self._set_vector_input_size(cursor, "embedding")
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE {where_clause}
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_by_id(
        self, id: str, data: Dict[str, Any], memory_store_type: MemoryType
    ) -> bool:
        """Update a document by ID."""
        # For semantic cache and shared memory, update base table directly
        if memory_store_type == MemoryType.SEMANTIC_CACHE:
            return self._update_semantic_cache_by_key(id, data)

        if memory_store_type == MemoryType.SHARED_MEMORY:
            return self._update_shared_memory_by_id(id, data)

        if memory_store_type == MemoryType.MEMAGENT:
            if not isinstance(data, dict):
                return False
            memory_ids = data.get("memory_ids")
            if memory_ids is not None:
                updated = self.update_memagent_memory_ids(id, memory_ids)
                data = {k: v for k, v in data.items() if k != "memory_ids"}
                if not data:
                    return updated

        if memory_store_type in {
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.KNOWLEDGE_BASE,
            MemoryType.PERSONAS,
            MemoryType.SHORT_TERM_MEMORY,
            MemoryType.TOOLBOX,
            MemoryType.SKILLBOX,
            MemoryType.WORKFLOW_MEMORY,
            MemoryType.SUMMARIES,
            MemoryType.ENTITY_MEMORY,
        }:
            return self._update_relational_by_id(id, data, memory_store_type)

        # Memory types with a dedicated update path are handled above. Every
        # other type falls through here; all of those now use base-table
        # writes via their ``_store_*`` methods, so a generic update_by_id
        # is no longer meaningful. Log and return False so callers see a
        # no-op rather than an exception.
        logger.debug(
            "update_by_id: no relational update path for %s — skipping",
            memory_store_type.value,
        )
        return False

    def _update_shared_memory_by_id(self, memory_id: str, data: Dict[str, Any]) -> bool:
        """Update shared memory entry directly in base table by memory_id."""
        table_name = self._get_table_name(MemoryType.SHARED_MEMORY)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build SET clause from data
            set_clauses = ["updated_at = SYSTIMESTAMP"]
            params = {"memory_id": memory_id}

            for key, value in data.items():
                # Skip fields that we handle specially or don't want to update
                if key in ("id", "memory_id", "_id", "updated_at"):
                    continue

                if key == "content":
                    # Sanitize and convert content to JSON string
                    if value is not None:
                        if isinstance(value, str):
                            params["content"] = value
                        else:
                            params["content"] = self._ensure_json_text(
                                self._sanitize_for_json(value)
                            )
                        set_clauses.append("content = :content")
                elif key == "access_list":
                    # Sanitize and convert access_list to JSON string
                    if value is not None:
                        params["access_list"] = self._ensure_json_text(
                            self._sanitize_for_json(value)
                        )
                        set_clauses.append("access_list = :access_list")
                else:
                    # Map snake_case to database column names
                    db_key = key
                    if key == "owner_agent_id":
                        db_key = "owner_agent_id"
                    set_clauses.append(f"{db_key} = :{key}")
                    params[key] = value

            if len(set_clauses) == 1:  # Only updated_at
                return False

            set_sql = ", ".join(set_clauses)

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {set_sql}
                WHERE memory_id = :memory_id
                """,
                params,
            )
            conn.commit()

            return cursor.rowcount > 0

    def _update_semantic_cache_by_key(
        self, id_value: str, data: Dict[str, Any]
    ) -> bool:
        """Update semantic cache by row UUID or deterministic cache key."""
        table_name = self._get_table_name(MemoryType.SEMANTIC_CACHE)

        allowed = {
            "query_text",
            "response",
            "scope",
            "similarity_threshold",
            "hit_count",
            "agent_id",
            "memory_id",
            "session_id",
            "user_id",
            "expires_at",
            "metadata",
            "embedding",
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Generated cache keys and physical row IDs are both UUID-shaped;
            # bind both and let the exact row/cache-key predicate resolve it.
            try:
                id_bytes = uuid.UUID(id_value).bytes
            except ValueError:
                id_bytes = None

            set_clauses = []
            params = {"id": id_bytes, "cache_key": str(id_value)}

            for key, value in data.items():
                if key not in allowed:
                    continue
                if key in {
                    "memory_id",
                    "session_id",
                    "user_id",
                    "metadata",
                } and not self._memory_type_has_column(MemoryType.SEMANTIC_CACHE, key):
                    continue
                if key == "metadata":
                    value = self._ensure_json_text(value or {})
                elif key == "embedding":
                    value = self._prepare_vector_value(value)
                elif key == "expires_at":
                    value = self._coerce_timestamp(value)
                set_clauses.append(f"{key} = :{key}")
                params[key] = value

            if not set_clauses:
                return False

            set_sql = ", ".join(set_clauses)
            if "embedding" in params:
                self._set_vector_input_size(cursor, "embedding")
            sql = f"""
                UPDATE {table_name}
                SET {set_sql}
                WHERE id = :id OR cache_key = :cache_key
            """
            try:
                cursor.execute(sql, params)
            except Exception as exc:
                if params.get(
                    "embedding"
                ) is not None and self._is_embedding_dimension_mismatch_error(str(exc)):
                    params["embedding"] = None
                    self._handle_embedding_dimension_mismatch(
                        "Semantic cache update", exc
                    )
                    cursor.execute(sql, params)
                else:
                    raise
            conn.commit()

            return cursor.rowcount > 0

    def recommended_vector_memory_size(self) -> Dict[str, Any]:
        """Estimate a conservative vector-memory budget for indexed rows."""
        dimensions = self.get_vector_schema_dimensions()
        default_dimension = next(iter(dimensions.values())) if dimensions else 256
        row_count = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT NVL(SUM(NVL(num_rows, 0)), 0)
                    FROM user_tables
                    WHERE table_name IN (
                        SELECT DISTINCT table_name
                        FROM user_tab_columns
                        WHERE data_type = 'VECTOR'
                    )
                    """
                )
                row = cursor.fetchone()
                row_count = int((row or [0])[0] or 0)
        except Exception:
            row_count = 0
        # HNSW graph + vectors + build headroom. A 256 MiB floor prevents a
        # misleadingly tiny recommendation on empty development schemas.
        estimated_bytes = max(
            256 * 1024 * 1024,
            int(max(row_count, 1) * default_dimension * 4 * 3.0),
        )
        gib = max(1, (estimated_bytes + (1024**3 - 1)) // 1024**3)
        return {
            "estimated_rows": row_count,
            "dimensions": default_dimension,
            "recommended": f"{gib}G",
            "estimated_bytes": estimated_bytes,
        }

    def preflight(self) -> Dict[str, Any]:
        """Return one structured Oracle readiness and capability report."""
        report: Dict[str, Any] = {
            "ok": True,
            "dsn": self.config.dsn,
            "registered_service": str(self.config.dsn).rsplit("/", 1)[-1],
            "schema": self.config.schema,
            "index_policy": self.config.index_policy,
            "diagnostics": [],
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                def one(sql: str, params: Optional[Dict[str, Any]] = None):
                    cursor.execute(sql, params or {})
                    return cursor.fetchone()

                try:
                    try:
                        product = one(
                            """
                            SELECT product, version, version_full
                            FROM product_component_version
                            WHERE ROWNUM = 1
                            """
                        )
                    except Exception:
                        # Older Oracle releases do not project VERSION_FULL.
                        product = one(
                            """
                            SELECT product, version
                            FROM product_component_version
                            WHERE ROWNUM = 1
                            """
                        )
                    report["database_product"] = product[0] if product else None
                    report["database_version"] = product[1] if product else None
                    report["version_full"] = (
                        str(product[2])
                        if product and len(product) > 2 and product[2]
                        else None
                    )
                except Exception as exc:
                    report["diagnostics"].append(f"database version unavailable: {exc}")

                # PRODUCT_COMPONENT_VERSION may expose only the compatibility
                # version (for example 23.0.0.0.0). VERSION_FULL retains the
                # actual Release Update patch level (for example 23.26.0.0.0).
                if not report.get("version_full"):
                    try:
                        full_version = one("SELECT version_full FROM v$instance")
                        report["version_full"] = (
                            str(full_version[0])
                            if full_version and full_version[0]
                            else None
                        )
                    except Exception as exc:
                        report["version_full"] = report.get("database_version")
                        report["diagnostics"].append(
                            f"full database version unavailable: {exc}"
                        )

                try:
                    pdb = one("SELECT SYS_CONTEXT('USERENV', 'CON_NAME') FROM dual")
                    report["pdb"] = pdb[0] if pdb else None
                    open_state = None
                    if report["pdb"]:
                        try:
                            state = one(
                                "SELECT open_mode FROM v$pdbs WHERE name = :pdb_name",
                                {"pdb_name": report["pdb"]},
                            )
                            open_state = state[0] if state else None
                        except Exception as state_exc:
                            report["diagnostics"].append(
                                f"PDB open mode unavailable: {state_exc}"
                            )
                    report["pdb_open_state"] = open_state or "unknown"
                except Exception as exc:
                    report["pdb"] = None
                    report["pdb_open_state"] = "unknown"
                    report["diagnostics"].append(f"PDB state unavailable: {exc}")

                try:
                    cursor.execute("SELECT privilege FROM session_privs")
                    privileges = sorted(str(row[0]) for row in cursor.fetchall())
                    report["schema_privileges"] = privileges
                    # Oracle does not expose a ``CREATE INDEX`` system
                    # privilege for indexes in the current schema.  A schema
                    # owner with CREATE TABLE and tablespace quota can create
                    # indexes on its own tables, so requiring the nonexistent
                    # privilege made every otherwise healthy preflight fail.
                    required = {"CREATE SESSION", "CREATE TABLE"}
                    report["missing_schema_privileges"] = sorted(
                        required.difference(privileges)
                    )
                except Exception as exc:
                    report["schema_privileges"] = []
                    report["diagnostics"].append(
                        f"schema privileges unavailable: {exc}"
                    )

                try:
                    cursor.execute(
                        "SELECT model_name FROM user_mining_models ORDER BY model_name"
                    )
                    report["database_embedding_models"] = [
                        str(row[0]) for row in cursor.fetchall()
                    ]
                except Exception:
                    report["database_embedding_models"] = []

                try:
                    vector_memory = one(
                        """
                        SELECT value FROM v$parameter
                        WHERE name = 'vector_memory_size'
                        """
                    )
                    report["vector_memory_size"] = (
                        vector_memory[0] if vector_memory else None
                    )
                except Exception as exc:
                    report["vector_memory_size"] = None
                    report["diagnostics"].append(
                        f"VECTOR_MEMORY_SIZE unavailable: {exc}"
                    )

                try:
                    cursor.execute(
                        """
                        SELECT index_name, table_name, status, index_type
                        FROM user_indexes
                        WHERE index_name LIKE 'IDX%VEC%'
                        ORDER BY table_name, index_name
                        """
                    )
                    report["vector_indexes"] = [
                        {
                            "name": row[0],
                            "table": row[1],
                            "status": row[2],
                            "type": row[3],
                        }
                        for row in cursor.fetchall()
                    ]
                except Exception as exc:
                    report["vector_indexes"] = []
                    report["diagnostics"].append(
                        f"vector index status unavailable: {exc}"
                    )
        except Exception as exc:
            report["ok"] = False
            report["diagnostics"].append(f"connection failed: {exc}")
            return report

        try:
            report["vector_dimensions"] = self.get_vector_schema_dimensions()
        except Exception as exc:
            report["vector_dimensions"] = {}
            report["diagnostics"].append(str(exc))
        provider = self._embedding_provider
        report["embedding"] = {
            "configured": provider is not None,
            "provider": type(provider).__name__ if provider is not None else None,
            "model": (
                provider.get_default_model()
                if provider is not None and hasattr(provider, "get_default_model")
                else None
            ),
            "dimensions": (
                provider.get_dimensions()
                if provider is not None and hasattr(provider, "get_dimensions")
                else None
            ),
        }
        embedding_dimensions = report["embedding"].get("dimensions")
        vector_dimensions = dict(report.get("vector_dimensions") or {})
        dimension_mismatches: Dict[str, int] = {}
        if embedding_dimensions is not None:
            try:
                expected_dimensions = int(embedding_dimensions)
                dimension_mismatches = {
                    name: int(dimension)
                    for name, dimension in vector_dimensions.items()
                    if int(dimension) != expected_dimensions
                }
                report["embedding_dimension_compatible"] = not dimension_mismatches
            except (TypeError, ValueError):
                report["embedding_dimension_compatible"] = None
        else:
            report["embedding_dimension_compatible"] = None
        report["embedding_dimension_mismatches"] = dimension_mismatches
        if dimension_mismatches:
            report["ok"] = False
            declared = ", ".join(
                f"{name}={dimension}"
                for name, dimension in sorted(dimension_mismatches.items())
            )
            report["diagnostics"].append(
                "embedding dimension mismatch: configured provider outputs "
                f"{embedding_dimensions}, while Oracle declares {declared}; "
                "align the embedding configuration or migrate the VECTOR columns "
                "before writes"
            )
        report["recommended_vector_memory_size"] = self.recommended_vector_memory_size()
        report["exact_search_fallback"] = True
        report["index_acceleration_available"] = bool(report.get("vector_indexes"))
        if report.get("missing_schema_privileges"):
            report["ok"] = False
        return report

    def set_vector_memory_size(
        self,
        size: str,
        *,
        admin_user: Optional[str] = None,
        admin_password: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set VECTOR_MEMORY_SIZE when the supplied connection is authorized."""
        normalized = str(size or "").strip().upper()
        if not re.fullmatch(r"[1-9][0-9]*[MG]", normalized):
            raise ValueError("size must use a positive Oracle size such as '1G'")
        owns_connection = bool(admin_user or admin_password)
        if owns_connection and not (admin_user and admin_password):
            raise ValueError("admin_user and admin_password must be supplied together")
        conn = (
            oracledb.connect(
                user=admin_user,
                password=admin_password,
                dsn=self.config.dsn,
            )
            if owns_connection
            else self._get_connection()
        )
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"ALTER SYSTEM SET VECTOR_MEMORY_SIZE = {normalized} SCOPE=SPFILE"
            )
            return {
                "ok": True,
                "value": normalized,
                "restart_required": True,
            }
        finally:
            conn.close()

    def delete_scope(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = _UNSET,
        agent_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Transactionally delete all package-owned data in an exact scope."""
        normalized_agents = sorted(
            {str(value).strip() for value in (agent_ids or []) if str(value).strip()}
        )
        if memory_id is None and user_id is _UNSET and not normalized_agents:
            raise ValueError(
                "delete_scope requires memory_id, user_id, or at least one agent_id"
            )

        table_fields: List[tuple[str, Optional[str], Optional[str], Optional[str]]] = [
            ("conversation_memory", "memory_id", "user_id", "agent_id"),
            ("semantic_cache", "memory_id", "user_id", "agent_id"),
            ("workflow_memory", "memory_id", "user_id", "agent_id"),
            ("tool_log", "memory_id", "user_id", "agent_id"),
            ("skillbox", None, "user_id", "agent_id"),
            ("summaries", "memory_id", "user_id", "agent_id"),
            ("entity_memory", "memory_id", "user_id", "agent_id"),
            ("knowledge_base", "memory_id", "user_id", "agent_id"),
            ("short_term_memory", "memory_id", "user_id", "agent_id"),
            ("toolbox", "memory_id", "user_id", "agent_id"),
            ("personas", "memory_id", None, "agent_id"),
            ("shared_memory", "memory_id", None, "owner_agent_id"),
            ("automation_jobs", None, None, "agent_id"),
        ]
        counts: Dict[str, int] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if normalized_agents:
                    agent_binds = []
                    automation_params: Dict[str, Any] = {}
                    for index, agent_id in enumerate(normalized_agents):
                        key = f"automation_agent_{index}"
                        agent_binds.append(f":{key}")
                        automation_params[key] = agent_id
                    agent_in = ", ".join(agent_binds)
                    cursor.execute(
                        f"""
                        DELETE FROM {self.config.schema}.automation_deliveries
                        WHERE run_id IN (
                            SELECT run_id
                            FROM {self.config.schema}.automation_runs
                            WHERE job_id IN (
                                SELECT job_id
                                FROM {self.config.schema}.automation_jobs
                                WHERE agent_id IN ({agent_in})
                            )
                        )
                        """,
                        automation_params,
                    )
                    counts["automation_deliveries"] = max(int(cursor.rowcount or 0), 0)
                    cursor.execute(
                        f"""
                        DELETE FROM {self.config.schema}.automation_runs
                        WHERE job_id IN (
                            SELECT job_id
                            FROM {self.config.schema}.automation_jobs
                            WHERE agent_id IN ({agent_in})
                        )
                        """,
                        automation_params,
                    )
                    counts["automation_runs"] = max(int(cursor.rowcount or 0), 0)

                for table, memory_col, user_col, agent_col in table_fields:
                    cursor.execute(
                        "SELECT COUNT(*) FROM user_tables WHERE table_name = UPPER(:1)",
                        (table,),
                    )
                    if not cursor.fetchone()[0]:
                        continue
                    predicates: List[str] = []
                    params: Dict[str, Any] = {}
                    if memory_id is not None and memory_col:
                        predicates.append(f"{memory_col} = :scope_memory_id")
                        params["scope_memory_id"] = memory_id
                    if user_id is not _UNSET and user_col:
                        if user_id is None:
                            predicates.append(f"{user_col} IS NULL")
                        else:
                            predicates.append(f"{user_col} = :scope_user_id")
                            params["scope_user_id"] = user_id
                    if normalized_agents and agent_col:
                        binds = []
                        for index, agent_id in enumerate(normalized_agents):
                            key = f"scope_agent_{index}"
                            binds.append(f":{key}")
                            params[key] = agent_id
                        predicates.append(f"{agent_col} IN ({', '.join(binds)})")
                    if not predicates:
                        continue
                    cursor.execute(
                        f"DELETE FROM {self.config.schema}.{table} "
                        f"WHERE {' AND '.join(predicates)}",
                        params,
                    )
                    counts[table] = max(int(cursor.rowcount or 0), 0)

                if memory_id is not None:
                    cursor.execute(
                        f"DELETE FROM {self.config.schema}.agent_memories "
                        "WHERE memory_id = :scope_memory_id",
                        {"scope_memory_id": memory_id},
                    )
                    counts["agent_memories"] = max(int(cursor.rowcount or 0), 0)

                if normalized_agents:
                    binds = []
                    params = {}
                    for index, agent_id in enumerate(normalized_agents):
                        key = f"delete_agent_{index}"
                        binds.append(f":{key}")
                        params[key] = agent_id
                    cursor.execute(
                        f"DELETE FROM {self.config.schema}.agents "
                        f"WHERE agent_id IN ({', '.join(binds)})",
                        params,
                    )
                    counts["agents"] = max(int(cursor.rowcount or 0), 0)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {
            "ok": True,
            "scope": {
                "memory_id": memory_id,
                "user_id": None if user_id is _UNSET else user_id,
                "user_id_supplied": user_id is not _UNSET,
                "agent_ids": normalized_agents,
            },
            "counts": counts,
            "total_deleted": sum(counts.values()),
        }

    def close(self) -> None:
        """Close the connection pool."""
        pool = getattr(self, "pool", None)
        if pool is None:
            return

        # Explicit provider shutdown owns the pool lifecycle. A non-forced
        # python-oracledb close raises DPY-1005 when LOB-backed results or
        # background work still hold a checked-out connection, which makes
        # notebook/application teardown fail even though no more work is
        # expected from this provider.
        self.pool = None
        pool.close(force=True)
        logger.info("Oracle connection pool closed")

    # ===== VECTOR SEARCH METHODS =====

    def retrieve_persona_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve personas using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(MemoryType.PERSONAS, embedding, limit=limit)

    def retrieve_toolbox_item(
        self,
        query: Dict[str, Any],
        limit: int = 1,
        *,
        agent_id: Any = _UNSET,
        user_id: Any = _UNSET,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve toolbox items using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        filters: Dict[str, Any] = {}
        if agent_id is not _UNSET:
            filters["agent_id"] = agent_id
        return self._vector_search(
            MemoryType.TOOLBOX,
            embedding,
            limit=limit,
            filters=filters,
            user_id=user_id,
        )

    def retrieve_skillbox_item(
        self,
        query: Dict[str, Any],
        limit: int = 1,
        *,
        statuses: Optional[List[str]] = None,
        agent_id: Any = _UNSET,
        user_id: Any = _UNSET,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve skillbox items using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(
                "Failed to generate a skillbox query embedding (%s)",
                type(e).__name__,
            )
            return []

        filters: Dict[str, Any] = {}
        if statuses:
            # Passive evaluation requests exactly SHADOW. Keep this scalar
            # so Oracle applies it in SQL before FETCH FIRST top-k.
            filters["status"] = list(statuses)[0]
        if agent_id is not _UNSET:
            filters["agent_id"] = agent_id
        return self._vector_search(
            MemoryType.SKILLBOX,
            embedding,
            limit=limit,
            filters=filters,
            user_id=user_id,
        )

    def retrieve_skillbox_candidates(
        self,
        query: str,
        *,
        limit: int,
        statuses: List[str],
        agent_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Oracle vector search with lifecycle and tenant pre-filters."""
        return list(
            self.retrieve_skillbox_item(
                query,
                limit,
                statuses=statuses,
                agent_id=agent_id,
                user_id=user_id,
            )
            or []
        )

    def retrieve_workflow_by_query(
        self, query: Dict[str, Any], limit: int = 1, user_id: Any = _UNSET
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve workflows using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(
            MemoryType.WORKFLOW_MEMORY, embedding, limit=limit, user_id=user_id
        )

    def retrieve_summaries_by_query(
        self, query: Dict[str, Any], limit: int = 1, user_id: Any = _UNSET
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve summaries using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(
            MemoryType.SUMMARIES, embedding, limit=limit, user_id=user_id
        )

    def find_similar_cache_entries(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Find semantically similar cache entries using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for semantic cache query: {e}")
            return []

        # Extract filters
        filters = {}
        if "agent_id" in kwargs and kwargs["agent_id"] is not None:
            filters["agent_id"] = kwargs["agent_id"]
        if "memory_id" in kwargs and kwargs["memory_id"] is not None:
            filters["memory_id"] = kwargs["memory_id"]
        if "session_id" in kwargs and kwargs["session_id"] is not None:
            filters["session_id"] = kwargs["session_id"]

        # user_id is tracked separately so None (anonymous) is applied as a
        # real IS NULL predicate instead of being dropped with truthy filters.
        user_id = kwargs.get("user_id", _UNSET)

        return self._vector_search(
            MemoryType.SEMANTIC_CACHE,
            embedding,
            limit=limit,
            filters=filters,
            user_id=user_id,
        )

    def _vector_search(
        self,
        memory_type: MemoryType,
        query_embedding: List[float],
        limit: int = 5,
        filters: Dict[str, Any] = None,
        memory_id: str = None,
        user_id: Any = _UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Oracle's VECTOR_DISTANCE function.

        Parameters:
        -----------
        memory_type : MemoryType
            The type of memory store to search
        query_embedding : List[float]
            The query embedding vector
        limit : int
            Maximum number of results to return
        filters : Dict[str, Any]
            Additional filter criteria
        memory_id : str
            Memory ID to filter by

        Returns:
        --------
        List[Dict[str, Any]]
            List of similar documents with scores
        """
        if memory_type in self._vector_search_disabled_for:
            return []

        # Ensure vector index exists (lazy creation)
        if self.config.index_policy == "lazy":
            self._ensure_vector_index(memory_type)

        table_name = self._get_table_name(memory_type)

        allowed_filters = {
            MemoryType.PERSONAS: {"memory_id", "agent_id", "name", "role_type"},
            MemoryType.TOOLBOX: {
                "memory_id",
                "agent_id",
                "user_id",
                "name",
                "tool_type",
            },
            MemoryType.SKILLBOX: {"agent_id", "user_id", "name", "status"},
            MemoryType.WORKFLOW_MEMORY: {
                "memory_id",
                "agent_id",
                "status",
                "name",
                "user_id",
            },
            MemoryType.SUMMARIES: {
                "memory_id",
                "agent_id",
                "summary_type",
                "thread_id",
                "user_id",
            },
            MemoryType.SEMANTIC_CACHE: {
                "cache_key",
                "scope",
                "agent_id",
                "memory_id",
                "session_id",
                "user_id",
            },
            MemoryType.ENTITY_MEMORY: {
                "memory_id",
                "agent_id",
                "entity_type",
                "name",
                "user_id",
            },
            MemoryType.CONVERSATION_MEMORY: {
                "memory_id",
                "agent_id",
                "thread_id",
                "role",
                "user_id",
            },
            MemoryType.KNOWLEDGE_BASE: {
                "memory_id",
                "agent_id",
                "memory_type",
                "namespace",
                "user_id",
            },
            MemoryType.SHORT_TERM_MEMORY: {
                "memory_id",
                "agent_id",
                "memory_type",
                "user_id",
            },
            MemoryType.SHARED_MEMORY: {"memory_id", "owner_agent_id", "scope"},
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()

            filter_clauses = ["embedding IS NOT NULL"]
            params = {"limit": limit}
            allowed = allowed_filters.get(memory_type, set())

            if memory_id and "memory_id" in allowed:
                filter_clauses.append("memory_id = :memory_id")
                params["memory_id"] = memory_id

            if filters:
                for key, value in filters.items():
                    if key in allowed:
                        if value is None:
                            filter_clauses.append(f"{key} IS NULL")
                        else:
                            filter_clauses.append(f"{key} = :{key}")
                            params[key] = value

            # Tenant isolation: user_id filter is strict — None becomes IS NULL,
            # not "match everything", to prevent cross-tenant leaks on vector
            # search paths. _UNSET means "no user_id scoping supplied"
            # (legacy callers) — we leave the query unfiltered. Skip entirely
            # when the column doesn't exist yet (pre-migration schemas).
            apply_user_scope = (
                user_id is not _UNSET
                and "user_id" in allowed
                and self._memory_type_has_column(memory_type, "user_id")
            )
            if apply_user_scope:
                if user_id is None:
                    filter_clauses.append("user_id IS NULL")
                else:
                    filter_clauses.append("user_id = :user_id_scope")
                    params["user_id_scope"] = user_id

            where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"

            params["query_vec"] = array.array("f", query_embedding)

            try:
                fields = _VECTOR_SPECS.get(memory_type)
                if fields is None:
                    logger.warning("Vector search not supported for %s", memory_type)
                    return []
                if memory_type == MemoryType.KNOWLEDGE_BASE:
                    fields = fields + self._knowledge_base_chunk_fields()

                sql = f"""
                SELECT
                    {_select_list(fields)}
                FROM {table_name}
                WHERE {where_clause}
                ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                FETCH FIRST :limit ROWS ONLY
                """
                if memory_type == MemoryType.CONVERSATION_MEMORY:
                    try:
                        cursor.execute(sql, params)
                    except Exception:
                        # Backward compat: unmigrated schema still has conversation_id
                        sql = sql.replace("thread_id", "conversation_id")
                        cursor.execute(sql, params)
                else:
                    cursor.execute(sql, params)

                return [self._apply_row_fields(fields, row) for row in cursor]
            except Exception as e:
                error_str = str(e)
                if self._is_vector_distance_dimension_mismatch_error(error_str):
                    self._vector_search_disabled_for.add(memory_type)
                    if (
                        memory_type
                        not in self._vector_dimension_mismatch_warning_emitted_for
                    ):
                        self._vector_dimension_mismatch_warning_emitted_for.add(
                            memory_type
                        )
                        dims = self._extract_dimension_pair_from_oracle_vector_error(
                            error_str
                        )
                        if dims:
                            # Our SQL calls VECTOR_DISTANCE(embedding, :query_vec, COSINE)
                            # so Oracle typically reports (stored_embedding_dim, query_dim).
                            stored_dim, query_dim = dims
                            logger.warning(
                                "Vector search dimension mismatch for %s: stored embeddings=%s dims, query vector=%s dims. "
                                "This usually means your embedding model/dimensions changed after the Oracle VECTOR columns were created, "
                                "or the table contains mixed-dimension embeddings. Align your embedding configuration with the schema (or rebuild/re-embed). Error: %s",
                                memory_type.value,
                                stored_dim,
                                query_dim,
                                e,
                            )
                        else:
                            logger.warning(
                                "Vector search dimension mismatch for %s. Align embedding dimensions with the Oracle schema (or rebuild/re-embed). Error: %s",
                                memory_type.value,
                                e,
                            )
                    return []

                if "ORA-00904" in error_str:
                    # Column missing — usually ``USER_ID`` on a schema where
                    # migration 001 has not been applied yet. Invalidate the
                    # column cache for this table and, if the query was
                    # user_id-scoped, retry without that predicate instead of
                    # disabling vector search entirely. Only disable when the
                    # missing column is something we can't gracefully drop.
                    self._table_columns_cache.pop(memory_type.value.upper(), None)
                    upper_err = error_str.upper()
                    if "USER_ID" in upper_err and user_id is not _UNSET:
                        logger.warning(
                            "user_id column not present on %s; retrying vector search without user scope. "
                            "Run migrations/001_add_user_id.sql to enable tenant scoping.",
                            memory_type.value,
                        )
                        return self._vector_search(
                            memory_type,
                            query_embedding,
                            limit=limit,
                            filters=filters,
                            memory_id=memory_id,
                            user_id=_UNSET,
                        )
                    self._vector_search_disabled_for.add(memory_type)
                    logger.warning(
                        "Vector search disabled for %s due to missing column: %s",
                        memory_type.value,
                        e,
                    )
                    return []

                logger.error(f"Vector search failed: {e}")
                return []

    # ===== MEMAGENT METHODS =====

    def store_memagent(self, memagent: "MemAgentModel") -> str:
        """Store a memagent."""
        memagent_dict = memagent.model_dump()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Generate or use existing agent_id
            agent_id_str = (
                memagent_dict.get("_id")
                or memagent_dict.get("agent_id")
                or str(uuid.uuid4())
            )
            has_is_favorite_column = self._table_has_column(
                cursor, "agents", "is_favorite"
            )
            has_verbose_column = self._table_has_column(cursor, "agents", "verbose")

            cursor.execute(
                "SELECT id FROM agents WHERE agent_id = :agent_id",
                {"agent_id": agent_id_str},
            )
            existing_agent_row = cursor.fetchone()

            agent_embedding = memagent_dict.get("embedding")
            agent_params: Dict[str, Any] = {
                "agent_id": agent_id_str,
                "name": memagent_dict.get("name"),
                "instruction": memagent_dict.get("instruction"),
                "application_mode": memagent_dict.get("application_mode", "assistant"),
                "max_steps": memagent_dict.get("max_steps", 20),
                "tool_access": memagent_dict.get("tool_access", "private"),
                "semantic_cache": 1 if memagent_dict.get("semantic_cache") else 0,
            }
            if has_verbose_column:
                agent_params["verbose"] = 1 if memagent_dict.get("verbose") else 0
            if has_is_favorite_column:
                agent_params["is_favorite"] = (
                    1 if memagent_dict.get("is_favorite") else 0
                )
            if agent_embedding is not None:
                agent_params["embedding"] = agent_embedding

            if existing_agent_row:
                set_parts = [
                    "name = :name",
                    "instruction = :instruction",
                    "application_mode = :application_mode",
                    "max_steps = :max_steps",
                    "tool_access = :tool_access",
                    "semantic_cache = :semantic_cache",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                if has_verbose_column:
                    set_parts.append("verbose = :verbose")
                if has_is_favorite_column:
                    set_parts.append("is_favorite = :is_favorite")
                if agent_embedding is not None:
                    set_parts.append("embedding = :embedding")
                cursor.execute(
                    f"UPDATE agents SET {', '.join(set_parts)} "
                    "WHERE agent_id = :agent_id",
                    agent_params,
                )
            else:
                cols = [
                    "id",
                    "agent_id",
                    "name",
                    "instruction",
                    "application_mode",
                    "max_steps",
                    "tool_access",
                    "semantic_cache",
                ]
                insert_params = dict(agent_params)
                insert_params["id"] = uuid.uuid4().bytes
                if has_verbose_column:
                    cols.append("verbose")
                if has_is_favorite_column:
                    cols.append("is_favorite")
                if agent_embedding is not None:
                    cols.append("embedding")
                vals = [f":{c}" for c in cols]
                cursor.execute(
                    f"INSERT INTO agents ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)})",
                    insert_params,
                )

            # Get the internal UUID for foreign key relationships
            cursor.execute(
                "SELECT id FROM agents WHERE agent_id = :agent_id",
                {"agent_id": agent_id_str},
            )
            row = cursor.fetchone()
            agent_uuid = row[0] if row else uuid.UUID(agent_id_str).bytes

            # Store persona in the personas base table
            if memagent_dict.get("persona"):
                persona = memagent_dict["persona"]
                persona_dict = (
                    persona.__dict__ if hasattr(persona, "__dict__") else persona
                )

                persona_text = f"{persona_dict.get('name', '')}: {persona_dict.get('background', '')} {persona_dict.get('goals', '')}"
                persona_embedding = None
                if self._embedding_provider and persona_text.strip():
                    try:
                        embedding_result = self._embedding_provider.get_embedding(
                            persona_text
                        )
                        if embedding_result and isinstance(embedding_result, list):
                            persona_embedding = embedding_result
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for persona: {e}")

                persona_id_value = persona_dict.get("persona_id") or persona_dict.get(
                    "name", str(uuid.uuid4())
                )
                role_raw = persona_dict.get("role")
                if isinstance(role_raw, str):
                    role_type_value = role_raw
                elif hasattr(role_raw, "value"):
                    role_type_value = role_raw.value
                else:
                    role_type_value = "general"

                self._upsert_persona_row(
                    cursor,
                    persona_id=persona_id_value,
                    name=persona_dict.get("name", "Unnamed Persona"),
                    role_type=role_type_value,
                    background=persona_dict.get("background", ""),
                    memory_id=persona_dict.get("memory_id"),
                    agent_id=agent_id_str,
                    embedding=persona_embedding,
                    traits=persona_dict.get("traits"),
                    expertise=persona_dict.get("expertise"),
                )

            # Store tools in the toolbox base table
            if memagent_dict.get("tools"):
                for tool_meta in memagent_dict["tools"]:
                    tool_text = f"{tool_meta.get('name', '')}: {tool_meta.get('description', '')} {tool_meta.get('signature', '')}"
                    tool_embedding = None
                    if self._embedding_provider and tool_text.strip():
                        try:
                            embedding_result = self._embedding_provider.get_embedding(
                                tool_text
                            )
                            if embedding_result and isinstance(embedding_result, list):
                                tool_embedding = embedding_result
                        except Exception as e:
                            logger.warning(
                                f"Failed to generate embedding for tool: {e}"
                            )

                    raw_tool_id = tool_meta.get("_id") or tool_meta.get(
                        "name", str(uuid.uuid4())
                    )
                    tool_id = f"{agent_id_str}:{raw_tool_id}"

                    self._upsert_toolbox_row(
                        cursor,
                        tool_id=tool_id,
                        name=tool_meta.get("name", "unknown_tool"),
                        description=tool_meta.get("description", ""),
                        signature=tool_meta.get("signature", ""),
                        docstring=tool_meta.get("docstring", ""),
                        tool_type=tool_meta.get("type", "function"),
                        memory_id=tool_meta.get("memory_id"),
                        agent_id=agent_id_str,
                        embedding=tool_embedding,
                        parameters=tool_meta.get("parameters"),
                        required=tool_meta.get("required"),
                        input_schema=tool_meta.get("input_schema"),
                        tool_policy=tool_meta.get("tool_policy"),
                        aliases=tool_meta.get("aliases"),
                        deprecated_arguments=tool_meta.get("deprecated_arguments"),
                        queries=tool_meta.get("queries"),
                        import_reference=tool_meta.get("import_reference"),
                        user_id=tool_meta.get("user_id"),
                    )

            # Store LLM config in its own table
            llm_config = memagent_dict.get("llm_config") or {}
            additional_cfg = dict(llm_config.get("additional_config", {}) or {})

            application_id_val = memagent_dict.get("application_id")
            if isinstance(application_id_val, str) and application_id_val.strip():
                additional_cfg["application_id"] = application_id_val.strip()
            else:
                additional_cfg.pop("application_id", None)

            sandbox_val = memagent_dict.get("sandbox_provider")
            if sandbox_val:
                additional_cfg["sandbox_provider"] = sandbox_val
            else:
                additional_cfg.pop("sandbox_provider", None)

            browser_control_val = memagent_dict.get("browser_control")
            if browser_control_val:
                additional_cfg["browser_control"] = browser_control_val
            else:
                additional_cfg.pop("browser_control", None)

            additional_cfg["meta_harness"] = bool(
                memagent_dict.get("meta_harness", False)
            )
            meta_harness_mode_val = memagent_dict.get("meta_harness_mode")
            if meta_harness_mode_val:
                additional_cfg["meta_harness_mode"] = meta_harness_mode_val
            else:
                additional_cfg.pop("meta_harness_mode", None)
            default_harness_val = memagent_dict.get("default_harness") or "auto"
            additional_cfg["default_harness"] = default_harness_val
            harness_config_val = memagent_dict.get("harness_config")
            if isinstance(harness_config_val, dict) and harness_config_val:
                additional_cfg["harness_config"] = harness_config_val
            else:
                additional_cfg.pop("harness_config", None)

            if "skill_paths" in memagent_dict:
                skill_paths_val = memagent_dict.get("skill_paths")
                if skill_paths_val:
                    additional_cfg["skill_paths"] = skill_paths_val
                else:
                    additional_cfg.pop("skill_paths", None)

            if "mcp_servers" in memagent_dict:
                mcp_servers_val = memagent_dict.get("mcp_servers")
                if mcp_servers_val:
                    additional_cfg["mcp_servers"] = mcp_servers_val
                else:
                    additional_cfg.pop("mcp_servers", None)

            if "memory_types" in memagent_dict:
                memory_types_val = memagent_dict.get("memory_types")
                if memory_types_val:
                    additional_cfg["memory_types"] = memory_types_val
                else:
                    additional_cfg.pop("memory_types", None)

            # Agent runtime governance belongs beside the LLM configuration so
            # existing Oracle schemas can round-trip the complete 0.5 surface
            # without adding a grab-bag of nullable columns to ``agents``.
            for config_key in (
                "semantic_cache_config",
                "tool_result_policy",
                "context_policy",
                "retrieval_policy",
                "delegation_config",
                "skill_retrieval_config",
                "semantic_layer_config",
                "context_window_tokens",
            ):
                config_value = memagent_dict.get(config_key)
                if config_value not in (None, {}, []):
                    additional_cfg[config_key] = config_value
                else:
                    additional_cfg.pop(config_key, None)
            additional_cfg["skill_retrieval"] = bool(
                memagent_dict.get("skill_retrieval", False)
            )

            internet_provider_val = memagent_dict.get("internet_access_provider")
            if internet_provider_val:
                additional_cfg["internet_access_provider"] = internet_provider_val
            else:
                additional_cfg.pop("internet_access_provider", None)

            if "internet_access_config" in memagent_dict:
                internet_config_val = memagent_dict.get("internet_access_config")
                if internet_config_val:
                    additional_cfg["internet_access_config"] = internet_config_val
                else:
                    additional_cfg.pop("internet_access_config", None)

            skills_provider_val = memagent_dict.get("skills_marketplace_provider")
            if skills_provider_val:
                additional_cfg["skills_marketplace_provider"] = skills_provider_val
            else:
                additional_cfg.pop("skills_marketplace_provider", None)

            if "skills_marketplace_config" in memagent_dict:
                skills_config_val = memagent_dict.get("skills_marketplace_config")
                if skills_config_val:
                    additional_cfg["skills_marketplace_config"] = skills_config_val
                else:
                    additional_cfg.pop("skills_marketplace_config", None)

            if "self_aware" in memagent_dict:
                additional_cfg["self_aware"] = bool(
                    memagent_dict.get("self_aware", False)
                )

            if "self_aware_config" in memagent_dict:
                self_aware_cfg_val = memagent_dict.get("self_aware_config")
                if isinstance(self_aware_cfg_val, dict):
                    additional_cfg["self_aware_config"] = self_aware_cfg_val
                else:
                    additional_cfg.pop("self_aware_config", None)

            if "continual_learning" in memagent_dict:
                additional_cfg["continual_learning"] = bool(
                    memagent_dict.get("continual_learning", False)
                )

            if "continual_learning_config" in memagent_dict:
                cl_cfg_val = memagent_dict.get("continual_learning_config")
                if isinstance(cl_cfg_val, dict):
                    additional_cfg["continual_learning_config"] = cl_cfg_val
                else:
                    additional_cfg.pop("continual_learning_config", None)

            if "learning_control_plane" in memagent_dict:
                additional_cfg["learning_control_plane"] = bool(
                    memagent_dict.get("learning_control_plane", False)
                )

            if "learning_control_plane_config" in memagent_dict:
                control_cfg = memagent_dict.get("learning_control_plane_config")
                if isinstance(control_cfg, dict):
                    additional_cfg["learning_control_plane_config"] = control_cfg
                else:
                    additional_cfg.pop("learning_control_plane_config", None)

            if "automations_enabled" in memagent_dict:
                additional_cfg["automations_enabled"] = bool(
                    memagent_dict.get("automations_enabled", True)
                )

            if "default_timezone" in memagent_dict:
                tz_val = memagent_dict.get("default_timezone")
                if isinstance(tz_val, str) and tz_val.strip():
                    additional_cfg["default_timezone"] = tz_val.strip()
                else:
                    additional_cfg.pop("default_timezone", None)

            if has_is_favorite_column:
                additional_cfg.pop("is_favorite", None)
            else:
                additional_cfg["is_favorite"] = bool(
                    memagent_dict.get("is_favorite", False)
                )

            if llm_config or additional_cfg:
                llm_data = {
                    "agent_id": agent_uuid,
                    "provider": llm_config.get("provider", "openai"),
                    "model": llm_config.get("model", "gpt-4o-mini"),
                    "temperature": llm_config.get("temperature"),
                    "max_tokens": llm_config.get("max_tokens"),
                    "top_p": llm_config.get("top_p"),
                    "frequency_penalty": llm_config.get("frequency_penalty"),
                    "presence_penalty": llm_config.get("presence_penalty"),
                    "additional_config": json.dumps(additional_cfg),
                }

                try:
                    cursor.execute(
                        """
                        INSERT INTO agent_llm_configs (agent_id, provider, model, temperature,
                                                       max_tokens, top_p, frequency_penalty,
                                                       presence_penalty, additional_config)
                        VALUES (:agent_id, :provider, :model, :temperature, :max_tokens,
                               :top_p, :frequency_penalty, :presence_penalty, :additional_config)
                    """,
                        llm_data,
                    )
                except Exception as e:
                    if "ORA-00001" in str(e):
                        cursor.execute(
                            """
                            UPDATE agent_llm_configs
                            SET provider = :provider, model = :model, temperature = :temperature,
                                max_tokens = :max_tokens, top_p = :top_p,
                                frequency_penalty = :frequency_penalty,
                                presence_penalty = :presence_penalty,
                                additional_config = :additional_config
                            WHERE agent_id = :agent_id
                        """,
                            llm_data,
                        )

            # Store memory_ids (many-to-many relationship table)
            if memagent_dict.get("memory_ids"):
                cursor.execute(
                    "DELETE FROM agent_memories WHERE agent_id = :agent_id",
                    {"agent_id": agent_uuid},
                )
                for memory_id in memagent_dict["memory_ids"]:
                    cursor.execute(
                        """
                        INSERT INTO agent_memories (agent_id, memory_id)
                        VALUES (:agent_id, :memory_id)
                    """,
                        {"agent_id": agent_uuid, "memory_id": memory_id},
                    )

            # Store knowledge_base_ids (many-to-many via agent_knowledge_bases).
            # Always DELETE first so detach (empty list) actually clears the
            # links — otherwise a None-on-empty short-circuit would leak stale rows.
            kb_ids = memagent_dict.get("knowledge_base_ids") or []
            # Skip gracefully on pre-migration schemas where the table doesn't
            # exist — callers still see the SDK behavior, just not persisted.
            try:
                cursor.execute(
                    "DELETE FROM agent_knowledge_bases WHERE agent_id = :agent_id",
                    {"agent_id": agent_uuid},
                )
                for kb_id in kb_ids:
                    cursor.execute(
                        """
                        INSERT INTO agent_knowledge_bases (agent_id, knowledge_base_id)
                        VALUES (:agent_id, :knowledge_base_id)
                        """,
                        {"agent_id": agent_uuid, "knowledge_base_id": str(kb_id)},
                    )
            except oracledb.DatabaseError as exc:
                (err,) = exc.args
                if err.code == 942:  # ORA-00942: table does not exist
                    logger.warning(
                        "agent_knowledge_bases table missing — run the updated "
                        "schema_relational.sql (knowledge_base_ids not persisted)"
                    )
                else:
                    raise

            # Store delegates (many-to-many relationship table)
            if memagent_dict.get("delegates"):
                cursor.execute(
                    "DELETE FROM agent_delegates WHERE agent_id = :agent_id",
                    {"agent_id": agent_uuid},
                )
                for delegate_id in memagent_dict["delegates"]:
                    try:
                        delegate_uuid = uuid.UUID(delegate_id).bytes
                        cursor.execute(
                            """
                            INSERT INTO agent_delegates (agent_id, delegate_agent_id)
                            VALUES (:agent_id, :delegate_agent_id)
                        """,
                            {
                                "agent_id": agent_uuid,
                                "delegate_agent_id": delegate_uuid,
                            },
                        )
                    except Exception:
                        pass

            # Update IS JSON columns
            # Update persona traits/expertise if present
            if memagent_dict.get("persona"):
                persona = memagent_dict["persona"]
                persona_dict = (
                    persona.__dict__ if hasattr(persona, "__dict__") else persona
                )
                persona_id = persona_dict.get("persona_id") or persona_dict.get(
                    "name", str(uuid.uuid4())
                )

                traits = persona_dict.get("traits")
                expertise = persona_dict.get("expertise")
                if traits or expertise:
                    cursor.execute(
                        """
                        UPDATE personas
                        SET traits = :traits, expertise = :expertise
                        WHERE persona_id = :persona_id
                    """,
                        {
                            "traits": json.dumps(traits) if traits else None,
                            "expertise": json.dumps(expertise) if expertise else None,
                            "persona_id": persona_id,
                        },
                    )

            # Update tool parameters if present (IS JSON column)
            if memagent_dict.get("tools"):
                for tool_meta in memagent_dict["tools"]:
                    if tool_meta.get("parameters"):
                        raw_tool_id = tool_meta.get("_id") or tool_meta.get(
                            "name", str(uuid.uuid4())
                        )
                        tool_id = f"{agent_id_str}:{raw_tool_id}"
                        cursor.execute(
                            """
                            UPDATE toolbox
                            SET parameters = :parameters
                            WHERE tool_id = :tool_id
                        """,
                            {
                                "parameters": json.dumps(
                                    tool_meta.get("parameters", {})
                                ),
                                "tool_id": tool_id,
                            },
                        )

            conn.commit()
            return agent_id_str

    def retrieve_tools_for_agent(
        self,
        agent_id: str,
        tool_access: str = "private",
        query: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tools for an agent based on tool_access mode.

        Args:
            agent_id: The agent's ID
            tool_access: 'private' or 'public'/'global'
            query: Optional query for semantic tool search
            top_k: Number of tools to retrieve (for public/semantic search)

        Returns:
            List of tool metadata dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            tools = []

            if tool_access == "private":
                # Retrieve only tools belonging to this agent
                cursor.execute(
                    f"""
                    SELECT {_select_list(_AGENT_TOOL_FIELDS)}
                    FROM toolbox
                    WHERE agent_id = :agent_id
                      AND (tool_type IS NULL OR LOWER(tool_type) <> 'mcp_server_config')
                      AND name IS NOT NULL
                    ORDER BY created_at DESC
                """,
                    {"agent_id": agent_id},
                )

                for row in cursor.fetchall():
                    tools.append(self._toolbox_row_to_dict(row))

            elif tool_access in ("public", "global"):
                # If query provided, use vector search for semantic matching
                if query and self._embedding_provider:
                    tools = self._retrieve_tools_by_semantic_search(query, top_k)
                else:
                    # Otherwise, retrieve all tools (or top_k most recent)
                    cursor.execute(
                        f"""
                        SELECT {_select_list(_AGENT_TOOL_FIELDS)}
                        FROM toolbox
                        WHERE (tool_type IS NULL OR LOWER(tool_type) <> 'mcp_server_config')
                          AND name IS NOT NULL
                        ORDER BY created_at DESC
                        FETCH FIRST {top_k} ROWS ONLY
                    """
                    )

                    for row in cursor.fetchall():
                        tools.append(self._toolbox_row_to_dict(row))

            logger.info(
                f"Retrieved {len(tools)} tools for agent {agent_id} (access: {tool_access})"
            )
            return tools

    def _retrieve_tools_by_semantic_search(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tools using vector similarity search.

        Args:
            query: The search query
            top_k: Number of tools to return

        Returns:
            List of tool metadata dictionaries
        """
        try:
            # Generate embedding for the query
            query_embedding = self._embedding_provider.get_embedding(query)
            if not query_embedding or not isinstance(query_embedding, list):
                logger.warning("Failed to generate query embedding for tool search")
                return []

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use Oracle's VECTOR_DISTANCE for similarity search
                # Convert Python list to array.array for Oracle VECTOR binding
                # Use 'f' (float32) to match the VECTOR column format in Oracle
                query_vec = array.array("f", query_embedding)

                num_tool_cols = len(_AGENT_TOOL_FIELDS)
                cursor.execute(
                    f"""
                    SELECT {_select_list(_AGENT_TOOL_FIELDS)},
                           VECTOR_DISTANCE(embedding, :query_embedding, COSINE) as distance
                    FROM toolbox
                    WHERE embedding IS NOT NULL
                      AND (tool_type IS NULL OR LOWER(tool_type) <> 'mcp_server_config')
                      AND name IS NOT NULL
                    ORDER BY distance
                    FETCH FIRST {top_k} ROWS ONLY
                """,
                    {"query_embedding": query_vec},
                )

                tools = []
                for row in cursor.fetchall():
                    # The trailing column is the vector distance.
                    tool_dict = self._toolbox_row_to_dict(row[:num_tool_cols])
                    tool_dict["similarity_distance"] = (
                        float(row[num_tool_cols])
                        if row[num_tool_cols] is not None
                        else None
                    )
                    tools.append(tool_dict)

                logger.info(
                    f"Found {len(tools)} tools via semantic search for query: {query[:50]}..."
                )
                return tools

        except Exception as e:
            logger.error(f"Semantic tool search failed: {e}")
            return []

    def _toolbox_row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """Convert a toolbox table row (``_AGENT_TOOL_FIELDS`` projection)
        to an agent-facing tool dictionary."""
        return self._apply_row_fields(_AGENT_TOOL_FIELDS, row)

    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """Delete a memagent from the memory provider."""
        if cascade:
            memagent = self.retrieve_memagent(agent_id)
            if memagent is None:
                raise ValueError(f"MemAgent with id {agent_id} not found")

            # Delete all memory units associated with the memagent
            for memory_id in memagent.memory_ids:
                for memory_type in MemoryType:
                    self._delete_memory_units_by_memory_id(memory_id, memory_type)

        return self.delete_by_id(agent_id, MemoryType.MEMAGENT)

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Update the memory_ids of a memagent."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM agents WHERE agent_id = :agent_id",
                    {"agent_id": agent_id},
                )
                row = cursor.fetchone()
                if not row:
                    logger.error(
                        "Cannot update memory_ids for unknown agent_id=%s", agent_id
                    )
                    return False

                agent_uuid = row[0]
                cursor.execute(
                    "DELETE FROM agent_memories WHERE agent_id = :agent_id",
                    {"agent_id": agent_uuid},
                )

                for memory_id in memory_ids:
                    cursor.execute(
                        """
                        INSERT INTO agent_memories (agent_id, memory_id)
                        VALUES (:agent_id, :memory_id)
                    """,
                        {"agent_id": agent_uuid, "memory_id": memory_id},
                    )

                conn.commit()
                return True
        except Exception as exc:
            logger.error(
                "Failed to update memagent memory_ids for %s: %s", agent_id, exc
            )
            return False

    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """Delete the memory_ids of a memagent."""
        return self.update_by_id(agent_id, {"memory_ids": []}, MemoryType.MEMAGENT)

    def list_memagents(self) -> List["MemAgentModel"]:
        """List all memagents in the Oracle database."""
        documents = self.list_all(MemoryType.MEMAGENT)
        agents = []
        fallback_favorites: Dict[str, bool] = {}
        additional_cfg_by_agent: Dict[str, Dict[str, Any]] = {}

        if documents:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    has_is_favorite_column = self._table_has_column(
                        cursor, "agents", "is_favorite"
                    )
                    cursor.execute(
                        """
                        SELECT a.agent_id, c.additional_config
                        FROM agents a
                        LEFT JOIN agent_llm_configs c ON c.agent_id = a.id
                        """
                    )
                    for row in cursor.fetchall():
                        agent_key = row[0]
                        cfg = self._deserialize_json_field(row[1]) or {}
                        if isinstance(cfg, dict):
                            additional_cfg_by_agent[agent_key] = cfg
                            if not has_is_favorite_column:
                                fallback_favorites[agent_key] = bool(
                                    cfg.get("is_favorite", False)
                                )
            except Exception:
                fallback_favorites = {}
                additional_cfg_by_agent = {}

        for doc in documents:
            # Use agent_id from the document, fall back to _id if not present
            agent_id = doc.get("agent_id") or str(doc.get("_id"))
            favorite_value = doc.get("is_favorite")
            if favorite_value is None:
                favorite_value = fallback_favorites.get(agent_id, False)
            cfg = additional_cfg_by_agent.get(agent_id, {})
            self_aware_value = bool(cfg.get("self_aware", False))
            self_aware_cfg_value = cfg.get("self_aware_config")
            if not isinstance(self_aware_cfg_value, dict):
                self_aware_cfg_value = None
            continual_learning_value = bool(cfg.get("continual_learning", False))
            continual_learning_cfg_value = cfg.get("continual_learning_config")
            if not isinstance(continual_learning_cfg_value, dict):
                continual_learning_cfg_value = None
            learning_control_plane_value = bool(
                cfg.get("learning_control_plane", False)
            )
            learning_control_plane_cfg_value = cfg.get("learning_control_plane_config")
            if not isinstance(learning_control_plane_cfg_value, dict):
                learning_control_plane_cfg_value = None
            automations_enabled_value = cfg.get("automations_enabled", True)
            if automations_enabled_value is None:
                automations_enabled_value = True
            automations_enabled_value = bool(automations_enabled_value)
            default_timezone_value = cfg.get("default_timezone")
            if not isinstance(default_timezone_value, str):
                default_timezone_value = None
            else:
                default_timezone_value = default_timezone_value.strip() or None
            agent = MemAgentModel(
                name=doc.get("name"),
                application_id=cfg.get("application_id"),
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                memory_types=doc.get("memory_types"),
                max_steps=doc.get("max_steps") or 20,  # Default to 20 if None
                memory_ids=doc.get("memory_ids") or [],
                agent_id=agent_id,
                is_favorite=bool(favorite_value),
                tools=doc.get("tools"),
                knowledge_base_ids=doc.get("knowledge_base_ids"),
                semantic_cache=bool(doc.get("semantic_cache", False)),
                semantic_cache_config=cfg.get("semantic_cache_config"),
                tool_result_policy=cfg.get("tool_result_policy"),
                context_policy=cfg.get("context_policy"),
                retrieval_policy=cfg.get("retrieval_policy"),
                delegation_config=cfg.get("delegation_config"),
                skill_retrieval=bool(cfg.get("skill_retrieval", False)),
                skill_retrieval_config=cfg.get("skill_retrieval_config"),
                semantic_layer_config=cfg.get("semantic_layer_config"),
                context_window_tokens=cfg.get("context_window_tokens"),
                browser_control=cfg.get("browser_control"),
                meta_harness=bool(cfg.get("meta_harness", False)),
                meta_harness_mode=cfg.get("meta_harness_mode"),
                default_harness=cfg.get("default_harness", "auto"),
                harness_config=cfg.get("harness_config"),
                self_aware=self_aware_value,
                continual_learning=continual_learning_value,
                continual_learning_config=continual_learning_cfg_value,
                learning_control_plane=learning_control_plane_value,
                learning_control_plane_config=learning_control_plane_cfg_value,
                self_aware_config=self_aware_cfg_value,
                automations_enabled=automations_enabled_value,
                default_timezone=default_timezone_value,
                memory_provider=self,
            )

            # Construct persona if present. Use from_dict so stored goals/
            # background aren't double-merged with role defaults and
            # version/evolution_history/storage_id round-trip correctly.
            if doc.get("persona"):
                agent.persona = Persona.from_dict(doc.get("persona"))

            agents.append(agent)

        return agents

    def supports_entity_memory(self) -> bool:
        """Oracle provider fully supports entity memory operations."""
        return True

    def migrate_entity_memory_user_scope(self, *, memory_id: str, user_id: str) -> int:
        """Atomically adopt anonymous entity rows in one Oracle scope."""
        if not self._memory_type_has_column(MemoryType.ENTITY_MEMORY, "user_id"):
            raise RuntimeError(
                "Oracle entity migration requires migrations/001_add_user_id.sql"
            )
        table_name = self._get_table_name(MemoryType.ENTITY_MEMORY)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET user_id = :user_id,
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_id = :memory_id
                  AND user_id IS NULL
                """,
                {"memory_id": str(memory_id), "user_id": str(user_id)},
            )
            migrated = int(cursor.rowcount or 0)
            conn.commit()
        return migrated

    def retrieve_memagent(self, agent_id: str) -> "MemAgentModel":
        """Retrieve a memagent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            has_is_favorite_column = self._table_has_column(
                cursor, "agents", "is_favorite"
            )

            # Query directly from the base table
            if has_is_favorite_column:
                cursor.execute(
                    """
                    SELECT id, agent_id, name, instruction, application_mode, max_steps,
                           tool_access, semantic_cache, is_favorite, verbose, embedding, created_at, updated_at
                    FROM agents
                    WHERE agent_id = :agent_id
                """,
                    {"agent_id": agent_id},
                )
            else:
                cursor.execute(
                    """
                    SELECT id, agent_id, name, instruction, application_mode, max_steps,
                           tool_access, semantic_cache, verbose, embedding, created_at, updated_at
                    FROM agents
                    WHERE agent_id = :agent_id
                """,
                    {"agent_id": agent_id},
                )

            row = cursor.fetchone()
            if not row:
                return None

            # Build agent JSON from row
            # Handle CLOB fields - they return LOB objects that need to be read
            agent_uuid = row[0]
            instruction = row[3]
            if hasattr(instruction, "read"):
                instruction = instruction.read()

            agent_json = {
                "agentId": row[1],
                "name": row[2],
                "instruction": instruction,
                "applicationMode": row[4],
                "maxSteps": row[5],
                "toolAccess": row[6],
                "semanticCache": row[7],
                "isFavorite": row[8] if has_is_favorite_column else 0,
                "verbose": row[9] if has_is_favorite_column else row[8],
                "embedding": row[10] if has_is_favorite_column else row[9],
                "createdAt": (
                    row[11].isoformat() if has_is_favorite_column and row[11] else None
                )
                or (
                    row[10].isoformat()
                    if not has_is_favorite_column and row[10]
                    else None
                ),
                "updatedAt": (
                    row[12].isoformat() if has_is_favorite_column and row[12] else None
                )
                or (
                    row[11].isoformat()
                    if not has_is_favorite_column and row[11]
                    else None
                ),
            }

            # Query LLM config (in its own table)
            llm_config = None
            if agent_uuid:
                cursor.execute(
                    """
                    SELECT provider, model, temperature, max_tokens, top_p,
                           frequency_penalty, presence_penalty, additional_config
                    FROM agent_llm_configs
                    WHERE agent_id = :agent_id
                """,
                    {"agent_id": agent_uuid},
                )
                llm_row = cursor.fetchone()
                if llm_row:
                    llm_config = {
                        "provider": llm_row[0],
                        "model": llm_row[1],
                        "temperature": llm_row[2],
                        "max_tokens": llm_row[3],
                        "top_p": llm_row[4],
                        "frequency_penalty": llm_row[5],
                        "presence_penalty": llm_row[6],
                        "additional_config": self._deserialize_json_field(llm_row[7])
                        or {},
                    }

                # Query memory_ids
                cursor.execute(
                    """
                    SELECT memory_id FROM agent_memories
                    WHERE agent_id = :agent_id
                    ORDER BY created_at
                """,
                    {"agent_id": agent_uuid},
                )
                memory_ids = [row[0] for row in cursor.fetchall()]

                # Query knowledge_base_ids (tolerate missing table on pre-
                # migration schemas — returns empty list, not an error).
                try:
                    cursor.execute(
                        """
                        SELECT knowledge_base_id FROM agent_knowledge_bases
                        WHERE agent_id = :agent_id
                        ORDER BY created_at
                        """,
                        {"agent_id": agent_uuid},
                    )
                    knowledge_base_ids = [row[0] for row in cursor.fetchall()]
                except oracledb.DatabaseError as exc:
                    (err,) = exc.args
                    if err.code == 942:  # ORA-00942: table does not exist
                        knowledge_base_ids = []
                    else:
                        raise
            else:
                memory_ids = []
                knowledge_base_ids = []

            # Query persona separately from personas table
            persona = None
            cursor.execute(
                """
                SELECT persona_id, name, role_type, background, memory_id, traits, expertise
                FROM personas
                WHERE agent_id = :agent_id
            """,
                {"agent_id": agent_id},
            )
            persona_row = cursor.fetchone()

            if persona_row:
                from ...long_term.semantic.persona.persona import Persona

                # Convert role_type string to enum
                role_type_str = persona_row[2] or "general"
                role = RoleType.GENERAL
                for role_enum in RoleType:
                    if role_enum.value == role_type_str:
                        role = role_enum
                        break

                traits = self._deserialize_json_field(persona_row[5])
                expertise = self._deserialize_json_field(persona_row[6])

                background_text = self._read_lob_value(persona_row[3]) or ""
                persona = Persona(
                    name=persona_row[1],
                    role=role,
                    goals=background_text,
                    background=background_text,
                    persona_id=persona_row[0],
                )
                if traits:
                    persona.traits = traits
                if expertise:
                    persona.expertise = expertise

            # Query tools separately from toolbox table
            tool_access = agent_json.get("toolAccess", "private")
            tools = None

            if tool_access == "private":
                cursor.execute(
                    f"""
                    SELECT {_select_list(_MEMAGENT_TOOL_FIELDS)}
                    FROM toolbox
                    WHERE agent_id = :agent_id
                      AND (tool_type IS NULL OR LOWER(tool_type) <> 'mcp_server_config')
                      AND name IS NOT NULL
                """,
                    {"agent_id": agent_id},
                )
                tool_rows = cursor.fetchall()

                if tool_rows:
                    tools = [
                        self._apply_row_fields(_MEMAGENT_TOOL_FIELDS, tool_row)
                        for tool_row in tool_rows
                    ]
            elif tool_access in ("public", "global"):
                # For public tools, load from all available tools
                tools = self.retrieve_tools_for_agent(
                    agent_id=agent_id, tool_access=tool_access, query=None, top_k=50
                )

            # Extract extended config from llm_config's additional_config
            sandbox_provider = None
            browser_control = None
            meta_harness = False
            meta_harness_mode = None
            default_harness = "auto"
            harness_config = None
            skill_paths = None
            mcp_servers = None
            internet_access_provider = None
            internet_access_config = None
            skills_marketplace_provider = None
            skills_marketplace_config = None
            memory_types = None
            self_aware = False
            self_aware_config = None
            continual_learning = False
            continual_learning_config = None
            learning_control_plane = False
            learning_control_plane_config = None
            automations_enabled = True
            default_timezone = None
            favorite_from_cfg = None
            semantic_cache_config = None
            tool_result_policy = None
            context_policy = None
            retrieval_policy = None
            delegation_config = None
            skill_retrieval = False
            skill_retrieval_config = None
            semantic_layer_config = None
            context_window_tokens = None
            application_id = None
            if llm_config:
                additional_cfg = llm_config.get("additional_config") or {}
                sandbox_provider = additional_cfg.pop("sandbox_provider", None)
                browser_control = additional_cfg.pop("browser_control", None)
                meta_harness = bool(additional_cfg.pop("meta_harness", False))
                meta_harness_mode = additional_cfg.pop("meta_harness_mode", None)
                default_harness = additional_cfg.pop("default_harness", "auto")
                harness_config = additional_cfg.pop("harness_config", None)
                skill_paths = additional_cfg.pop("skill_paths", None)
                mcp_servers = additional_cfg.pop("mcp_servers", None)
                internet_access_provider = additional_cfg.pop(
                    "internet_access_provider", None
                )
                internet_access_config = additional_cfg.pop(
                    "internet_access_config", None
                )
                skills_marketplace_provider = additional_cfg.pop(
                    "skills_marketplace_provider", None
                )
                skills_marketplace_config = additional_cfg.pop(
                    "skills_marketplace_config", None
                )
                memory_types = additional_cfg.pop("memory_types", None)
                semantic_cache_config = additional_cfg.pop(
                    "semantic_cache_config", None
                )
                tool_result_policy = additional_cfg.pop("tool_result_policy", None)
                context_policy = additional_cfg.pop("context_policy", None)
                retrieval_policy = additional_cfg.pop("retrieval_policy", None)
                delegation_config = additional_cfg.pop("delegation_config", None)
                skill_retrieval = bool(additional_cfg.pop("skill_retrieval", False))
                skill_retrieval_config = additional_cfg.pop(
                    "skill_retrieval_config", None
                )
                semantic_layer_config = additional_cfg.pop(
                    "semantic_layer_config", None
                )
                context_window_tokens = additional_cfg.pop(
                    "context_window_tokens", None
                )
                self_aware = bool(additional_cfg.pop("self_aware", False))
                self_aware_config = additional_cfg.pop("self_aware_config", None)
                if not isinstance(self_aware_config, dict):
                    self_aware_config = None
                continual_learning = bool(
                    additional_cfg.pop("continual_learning", False)
                )
                continual_learning_config = additional_cfg.pop(
                    "continual_learning_config", None
                )
                if not isinstance(continual_learning_config, dict):
                    continual_learning_config = None
                learning_control_plane = bool(
                    additional_cfg.pop("learning_control_plane", False)
                )
                learning_control_plane_config = additional_cfg.pop(
                    "learning_control_plane_config", None
                )
                if not isinstance(learning_control_plane_config, dict):
                    learning_control_plane_config = None
                automations_enabled = additional_cfg.pop("automations_enabled", True)
                if automations_enabled is None:
                    automations_enabled = True
                automations_enabled = bool(automations_enabled)
                default_timezone = additional_cfg.pop("default_timezone", None)
                if not isinstance(default_timezone, str):
                    default_timezone = None
                else:
                    default_timezone = default_timezone.strip() or None
                favorite_from_cfg = additional_cfg.pop("is_favorite", None)
                application_id = additional_cfg.pop("application_id", None)

            # Create MemAgentModel from JSON
            memagent = MemAgentModel(
                name=agent_json.get("name"),
                application_id=application_id,
                instruction=agent_json.get("instruction"),
                application_mode=agent_json.get("applicationMode", "assistant"),
                memory_types=memory_types,
                max_steps=agent_json.get("maxSteps", 20),
                tool_access=tool_access,
                memory_ids=memory_ids,
                knowledge_base_ids=knowledge_base_ids or None,
                agent_id=agent_json.get("agentId"),
                is_favorite=bool(
                    agent_json.get("isFavorite")
                    if has_is_favorite_column
                    else favorite_from_cfg
                    if favorite_from_cfg is not None
                    else False
                ),
                llm_config=llm_config,
                semantic_cache=bool(agent_json.get("semanticCache")),
                semantic_cache_config=semantic_cache_config,
                tool_result_policy=tool_result_policy,
                context_policy=context_policy,
                retrieval_policy=retrieval_policy,
                delegation_config=delegation_config,
                skill_retrieval=skill_retrieval,
                skill_retrieval_config=skill_retrieval_config,
                semantic_layer_config=semantic_layer_config,
                context_window_tokens=context_window_tokens,
                tools=tools if tools else None,
                persona=persona,
                sandbox_provider=sandbox_provider,
                browser_control=browser_control,
                meta_harness=meta_harness,
                meta_harness_mode=meta_harness_mode,
                default_harness=default_harness,
                harness_config=harness_config,
                skill_paths=skill_paths,
                mcp_servers=mcp_servers,
                internet_access_provider=internet_access_provider,
                internet_access_config=internet_access_config,
                skills_marketplace_provider=skills_marketplace_provider,
                skills_marketplace_config=skills_marketplace_config,
                self_aware=self_aware,
                self_aware_config=self_aware_config,
                continual_learning=continual_learning,
                continual_learning_config=continual_learning_config,
                learning_control_plane=learning_control_plane,
                learning_control_plane_config=learning_control_plane_config,
                automations_enabled=automations_enabled,
                default_timezone=default_timezone,
                memory_provider=self,
            )

            return memagent

    # Tables that do not have a memory_id column.
    _TYPES_WITHOUT_MEMORY_ID = frozenset(
        {MemoryType.MEMAGENT, MemoryType.SEMANTIC_CACHE}
    )

    def _delete_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ):
        """Delete all memory units associated with a memory_id."""
        if memory_type in self._TYPES_WITHOUT_MEMORY_ID:
            return

        table_name = self._get_table_name(memory_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE memory_id = :memory_id",
                    {"memory_id": memory_id},
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                if "ORA-00904" in str(exc):
                    logger.debug(
                        "Skipping cleanup for %s: memory_id column not present",
                        memory_type.value,
                    )
                else:
                    raise
