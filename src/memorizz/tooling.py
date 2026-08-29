# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tool schemas, policies, result offloading, and progressive disclosure."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    get_type_hints,
)

from pydantic import TypeAdapter

from .tool_outcomes import (
    ToolOutcome,
    ToolOutcomeStatus,
    ToolResult,
    normalize_tool_result,
)


def _json_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def serialize_tool_result(value: Any) -> str:
    """Serialize a tool result once, preserving JSON structure where possible."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    except (TypeError, ValueError):
        return str(value)


def _inline_local_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a self-contained annotation schema with local refs expanded.

    ``TypeAdapter`` places ``$defs`` beside the schema it generates. Tool
    parameters are generated one annotation at a time, so leaving those defs
    inside an individual property produces refs such as
    ``#/$defs/EntityAttributeInput`` that incorrectly point at the eventual
    tool-schema root. Inlining keeps provider tool schemas valid and makes the
    nested fields explicit to the model. Recursive input models are deliberately
    reduced to an object at the recursive edge; recursive tool arguments are not
    a useful or safe model-facing contract.
    """
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict) or not definitions:
        return schema

    def expand(value: Any, active: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [expand(item, active) for item in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        prefix = "#/$defs/"
        if isinstance(ref, str) and ref.startswith(prefix):
            definition_name = ref[len(prefix) :]
            overlay = {key: item for key, item in value.items() if key != "$ref"}
            target = definitions.get(definition_name)
            if isinstance(target, dict):
                if definition_name in active:
                    return {
                        "type": "object",
                        **{key: expand(item, active) for key, item in overlay.items()},
                    }
                return expand(
                    {**target, **overlay},
                    (*active, definition_name),
                )

        return {
            key: expand(item, active) for key, item in value.items() if key != "$defs"
        }

    return expand(schema)


def _schema_for_annotation(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    try:
        schema = TypeAdapter(annotation).json_schema(mode="validation")
        if isinstance(schema, dict):
            return _inline_local_schema_refs(schema)
    except Exception:
        pass
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array", "items": {}},
        dict: {"type": "object"},
    }
    return dict(mapping.get(annotation, {"type": "string"}))


def callable_json_schema(function: Callable[..., Any]) -> Dict[str, Any]:
    """Generate a complete JSON Schema from a trusted callable signature."""
    signature = inspect.signature(function)
    # ``from __future__ import annotations`` is common in applications and in
    # MemoRizz itself.  ``inspect.signature`` deliberately leaves those
    # annotations as strings; feeding the strings to Pydantic loses details
    # such as Literal enums, Optional unions, nested models, and constrained
    # collection types. Resolve the callable's annotations as one namespace so
    # the resulting schema remains complete across persistence round-trips.
    try:
        resolved_hints = get_type_hints(function, include_extras=True)
    except Exception:
        # A callable may legitimately reference a local type that is no longer
        # resolvable. Preserve the previous best-effort behaviour for only
        # those unusual callables rather than failing tool registration.
        resolved_hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        schema = _schema_for_annotation(resolved_hints.get(name, parameter.annotation))
        schema.setdefault("description", f"Parameter {name}")
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            try:
                json.dumps(parameter.default, default=_json_default)
                schema["default"] = _json_default(parameter.default)
            except Exception:
                pass
        properties[name] = schema
    result: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


@dataclass(frozen=True)
class ToolPolicy:
    """Runtime governance metadata attached to a trusted callable."""

    deterministic: bool = True
    side_effects: bool = False
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    domains: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    deprecated_arguments: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "side_effects": self.side_effects,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason,
            "domains": list(self.domains),
            "aliases": list(self.aliases),
            "deprecated_arguments": dict(self.deprecated_arguments),
        }


def governed_tool(
    *,
    deterministic: bool = True,
    side_effects: bool = False,
    requires_approval: Optional[bool] = None,
    approval_reason: Optional[str] = None,
    domains: Sequence[str] = (),
    aliases: Sequence[str] = (),
    deprecated_arguments: Optional[Mapping[str, str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a callable with explicit tool and cache governance metadata."""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        policy = ToolPolicy(
            deterministic=bool(deterministic),
            side_effects=bool(side_effects),
            requires_approval=(
                bool(side_effects)
                if requires_approval is None
                else bool(requires_approval)
            ),
            approval_reason=approval_reason,
            domains=tuple(str(item) for item in domains if str(item).strip()),
            aliases=tuple(str(item) for item in aliases if str(item).strip()),
            deprecated_arguments=dict(deprecated_arguments or {}),
        )
        setattr(function, "__memorizz_tool_policy__", policy)
        return function

    return decorator


def policy_for_callable(function: Optional[Callable[..., Any]]) -> ToolPolicy:
    value = getattr(function, "__memorizz_tool_policy__", None)
    if isinstance(value, ToolPolicy):
        return value
    if isinstance(value, dict):
        return ToolPolicy(**value)
    return ToolPolicy()


@dataclass
class ToolResultPolicy:
    """Control when full tool output is replaced by an auditable pointer."""

    offload_above_chars: int = 8_000
    offload_above_tokens: Optional[int] = None
    expansion_tool_names: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"retrieve_tool_log_entry", "expand_tool_result"}
        )
    )
    digest_chars: int = 360

    def __post_init__(self) -> None:
        self.offload_above_chars = max(256, int(self.offload_above_chars))
        if self.offload_above_tokens is not None:
            self.offload_above_tokens = max(64, int(self.offload_above_tokens))
        self.digest_chars = max(80, min(int(self.digest_chars), 2_000))
        self.expansion_tool_names = frozenset(
            str(name).strip() for name in self.expansion_tool_names if str(name).strip()
        )

    @classmethod
    def from_value(cls, value: Optional["ToolResultPolicy" | Dict[str, Any]]):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("tool_result_policy must be ToolResultPolicy, dict, or None")

    def should_offload(self, tool_name: str, serialized_result: str) -> bool:
        if str(tool_name) in self.expansion_tool_names:
            return False
        if len(serialized_result) > self.offload_above_chars:
            return True
        if self.offload_above_tokens is not None:
            # A conservative provider-independent approximation.
            return (len(serialized_result) + 3) // 4 > self.offload_above_tokens
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable JSON representation for agent persistence."""
        return {
            "offload_above_chars": self.offload_above_chars,
            "offload_above_tokens": self.offload_above_tokens,
            "expansion_tool_names": sorted(self.expansion_tool_names),
            "digest_chars": self.digest_chars,
        }

    def pointer(
        self,
        *,
        tool_name: str,
        tool_log_id: str,
        serialized_result: str,
        digest: str,
        identifiers: str = "",
        tool_call_id: Optional[str] = None,
    ) -> str:
        value = {
            "ok": True,
            "offloaded": True,
            "tool_name": tool_name,
            "tool_log_id": tool_log_id,
            "tool_call_id": tool_call_id,
            "result_sha256": hashlib.sha256(
                serialized_result.encode("utf-8", errors="replace")
            ).hexdigest(),
            "result_chars": len(serialized_result),
            "digest": str(digest or "")[: self.digest_chars],
            "identifiers": identifiers or None,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "expand_with": {
                "tool": "retrieve_tool_log_entry",
                "arguments": {"tool_log_id": tool_log_id},
            },
        }
        return json.dumps(
            {key: item for key, item in value.items() if item is not None},
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class ContextPolicy:
    """Bounded context and progressive-disclosure defaults for an agent."""

    progressive_tool_disclosure: bool = True
    tool_top_k: int = 5
    approval_ttl_seconds: int = 900
    max_tool_invocations_per_turn: int = 20
    max_tool_attempts_per_call: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_top_k", max(1, min(int(self.tool_top_k), 50)))
        object.__setattr__(
            self,
            "approval_ttl_seconds",
            max(1, min(int(self.approval_ttl_seconds), 86_400)),
        )
        object.__setattr__(
            self,
            "max_tool_invocations_per_turn",
            max(1, int(self.max_tool_invocations_per_turn)),
        )
        object.__setattr__(
            self,
            "max_tool_attempts_per_call",
            max(1, int(self.max_tool_attempts_per_call)),
        )

    @classmethod
    def from_value(cls, value: Optional["ContextPolicy" | Dict[str, Any]]):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("context_policy must be ContextPolicy, dict, or None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "progressive_tool_disclosure": self.progressive_tool_disclosure,
            "tool_top_k": self.tool_top_k,
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "max_tool_invocations_per_turn": self.max_tool_invocations_per_turn,
            "max_tool_attempts_per_call": self.max_tool_attempts_per_call,
        }


def tool_metadata_to_openai(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize persisted or native metadata into a strict OpenAI tool schema."""
    if metadata.get("type") == "function" and isinstance(
        metadata.get("function"), dict
    ):
        function = dict(metadata["function"])
        name = str(function.get("name") or "").strip()
        if not name:
            return None
        schema = function.get("parameters")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        schema = dict(schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        function["parameters"] = schema
        return {**metadata, "type": "function", "function": function}

    name = str(metadata.get("name") or "").strip()
    if not name:
        return None
    explicit_schema = metadata.get("input_schema") or metadata.get("json_schema")
    if isinstance(explicit_schema, str):
        try:
            explicit_schema = json.loads(explicit_schema)
        except json.JSONDecodeError:
            explicit_schema = None
    if isinstance(explicit_schema, dict):
        schema = dict(explicit_schema)
    else:
        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        schema = {
            "type": "object",
            "properties": parameters,
            "required": list(metadata.get("required") or []),
        }
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(metadata.get("description") or "No description"),
            "parameters": schema,
        },
    }


class SemanticToolRouter:
    """Progressively disclose and strictly dispatch a bounded set of tools."""

    DISCOVERY_TOOL = "discover_tools"
    INVOCATION_TOOL = "invoke_tool"

    @dataclass
    class _TurnState:
        selected: set[str] = field(default_factory=set)
        call_attempts: Dict[str, int] = field(default_factory=dict)
        successful_calls: set[str] = field(default_factory=set)
        invocation_count: int = 0
        active_user_id: Optional[str] = None

    def _state(self) -> "SemanticToolRouter._TurnState":
        state = self._turn_state_var.get()
        if state is None:
            state = self._TurnState()
            self._turn_state_var.set(state)
        return state

    @property
    def _selected(self) -> set[str]:
        return self._state().selected

    @_selected.setter
    def _selected(self, value: Iterable[str]) -> None:
        self._state().selected = set(value)

    @property
    def _call_attempts(self) -> Dict[str, int]:
        return self._state().call_attempts

    @_call_attempts.setter
    def _call_attempts(self, value: Mapping[str, int]) -> None:
        self._state().call_attempts = dict(value)

    @property
    def _successful_calls(self) -> set[str]:
        return self._state().successful_calls

    @_successful_calls.setter
    def _successful_calls(self, value: Iterable[str]) -> None:
        self._state().successful_calls = set(value)

    @property
    def _invocation_count(self) -> int:
        return self._state().invocation_count

    @_invocation_count.setter
    def _invocation_count(self, value: int) -> None:
        self._state().invocation_count = int(value)

    @property
    def _active_user_id(self) -> Optional[str]:
        return self._state().active_user_id

    @_active_user_id.setter
    def _active_user_id(self, value: Optional[str]) -> None:
        self._state().active_user_id = value

    def __init__(
        self,
        tool_manager: Any,
        *,
        toolbox: Any = None,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        enabled: bool = True,
        always_visible: Optional[Iterable[str]] = None,
        aliases: Optional[Mapping[str, str]] = None,
        deprecated_arguments: Optional[Mapping[str, Mapping[str, str]]] = None,
        max_invocations_per_turn: int = 20,
        max_attempts_per_call: int = 2,
    ) -> None:
        self._turn_state_var: ContextVar[
            Optional[SemanticToolRouter._TurnState]
        ] = ContextVar(f"memorizz_router_turn_{id(self)}", default=None)
        self.tool_manager = tool_manager
        self.toolbox = toolbox
        self.agent_id = agent_id
        self.top_k = max(1, min(int(top_k), 50))
        self.enabled = bool(enabled)
        self.always_visible = set(always_visible or ())
        self.aliases = {str(key): str(value) for key, value in (aliases or {}).items()}
        self.deprecated_arguments = {
            str(tool): {str(old): str(new) for old, new in mapping.items()}
            for tool, mapping in (deprecated_arguments or {}).items()
        }
        self.max_invocations_per_turn = max(1, int(max_invocations_per_turn))
        self.max_attempts_per_call = max(1, int(max_attempts_per_call))
        self._selected: set[str] = set()
        self._call_attempts: Dict[str, int] = {}
        self._successful_calls: set[str] = set()
        self._invocation_count = 0
        self._active_user_id: Optional[str] = None

    def begin_turn(self, *, user_id: Optional[str] = None) -> None:
        self._turn_state_var.set(self._TurnState(active_user_id=user_id))

    def _metadata(self) -> List[Dict[str, Any]]:
        values = self.tool_manager.get_tool_metadata() or []
        return [dict(item) for item in values if isinstance(item, dict)]

    @staticmethod
    def _name(metadata: Dict[str, Any]) -> str:
        if metadata.get("type") == "function" and isinstance(
            metadata.get("function"), dict
        ):
            return str(metadata["function"].get("name") or "")
        return str(metadata.get("name") or "")

    def _resolve_name(self, name: str) -> str:
        normalized = str(name or "").strip()
        return self.aliases.get(normalized, normalized)

    def _lexical_candidates(self, query: str, limit: int) -> List[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "at",
            "for",
            "from",
            "in",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", str(query).lower())
            if term not in stop_words
        }
        scored: List[tuple[int, str]] = []
        for metadata in self._metadata():
            name = self._name(metadata)
            if not name or name in {self.DISCOVERY_TOOL, self.INVOCATION_TOOL}:
                continue
            haystack = " ".join([name, str(metadata.get("description") or "")]).lower()
            score = sum(1 for term in terms if term in haystack)
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for score, name in scored if score > 0][:limit]

    def _semantic_candidates(
        self, query: str, *, user_id: Optional[str], limit: int
    ) -> List[str]:
        if self.toolbox is None:
            return []
        try:
            rows = self.toolbox.get_most_similar_tools(
                query,
                limit=max(limit * 5, 25),
                user_id=user_id,
                agent_id=self.agent_id,
            )
        except TypeError:
            rows = self.toolbox.get_most_similar_tools(query, limit=max(limit * 5, 25))
        except Exception:
            return []
        names: List[str] = []
        registered = set(self.tool_manager.list_tools())
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if self.agent_id is not None and row.get("agent_id") not in {
                None,
                self.agent_id,
            }:
                continue
            if row.get("user_id") != user_id:
                continue
            name = self._resolve_name(str(row.get("name") or ""))
            if name in registered and name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    def select(
        self, query: str, *, user_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        bounded = max(1, min(int(limit or self.top_k), self.top_k))
        candidates = self.preview(query, user_id=user_id, limit=bounded)
        self._selected.update(candidates)
        return candidates

    def preview(
        self, query: str, *, user_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """Select candidates without mutating the active turn allowlist."""
        bounded = max(1, min(int(limit or self.top_k), self.top_k))
        semantic = self._semantic_candidates(query, user_id=user_id, limit=bounded)
        candidates = semantic or self._lexical_candidates(query, bounded)
        return candidates[:bounded]

    @staticmethod
    def _meta_schemas() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": SemanticToolRouter.DISCOVERY_TOOL,
                    "description": "Find relevant registered tools without receiving every schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": SemanticToolRouter.INVOCATION_TOOL,
                    "description": "Invoke one tool returned by discover_tools using strict arguments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def schemas_for_turn(
        self, query: str, *, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return [
                schema
                for metadata in self._metadata()
                if (schema := tool_metadata_to_openai(metadata)) is not None
            ]
        self.select(query, user_id=user_id)
        visible = self.always_visible | self._selected
        schemas = self._meta_schemas()
        for metadata in self._metadata():
            if self._name(metadata) not in visible:
                continue
            schema = tool_metadata_to_openai(metadata)
            if schema is not None:
                schemas.append(schema)
        schemas.sort(key=lambda item: item["function"]["name"])
        return schemas

    def discover_tools(
        self, query: str, limit: int = 5, *, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        names = self.select(query, user_id=user_id, limit=limit)
        by_name = {self._name(item): item for item in self._metadata()}
        tools = []
        for name in names:
            schema = tool_metadata_to_openai(by_name.get(name, {}))
            if schema is not None:
                tools.append(
                    {
                        "name": name,
                        "description": schema["function"].get("description"),
                        "input_schema": schema["function"]["parameters"],
                    }
                )
        return {"ok": True, "query": query, "tools": tools, "count": len(tools)}

    def normalize_invocation(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, Dict[str, Any], List[str]]:
        name = self._resolve_name(tool_name)
        # With progressive disclosure disabled, ``schemas_for_turn`` exposes
        # every registered tool. Invocation must mirror that contract rather
        # than requiring a selection that can only be made through the hidden
        # ``discover_tools`` meta-tool.
        if (
            self.enabled
            and name not in self._selected
            and name not in self.always_visible
        ):
            raise PermissionError(
                f"Tool '{name}' was not disclosed for this turn; call discover_tools first"
            )
        if name in {self.DISCOVERY_TOOL, self.INVOCATION_TOOL}:
            raise PermissionError("Meta-tools cannot invoke themselves")
        values = dict(arguments)
        warnings: List[str] = []
        mapping = self.deprecated_arguments.get(name, {})
        for old, new in mapping.items():
            if old not in values:
                continue
            if new in values:
                raise TypeError(
                    f"Arguments '{old}' and '{new}' cannot both be supplied"
                )
            values[new] = values.pop(old)
            warnings.append(f"Argument '{old}' is deprecated; use '{new}'")

        function = self.tool_manager.get_tool_callable(name)
        if function is None:
            raise LookupError(f"Tool '{name}' has no trusted callable binding")
        signature = inspect.signature(function)
        signature.bind(**values)
        return name, values, warnings

    def invoke_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Dict[str, Any]:
        try:
            name, values, warnings = self.normalize_invocation(tool_name, arguments)
        except (LookupError, PermissionError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "error_code": "invalid_tool_invocation",
                "error": str(exc),
            }

        prepared = self.prepare_invocation(name, values)
        if not prepared.get("ok"):
            return prepared
        call_hash = str(prepared["call_hash"])
        result, _outcome = self.tool_manager.execute_tool(name, values)
        result, tool_outcome = normalize_tool_result(result)
        failed = not tool_outcome.ok
        self.record_invocation(call_hash, success=not failed)
        return {
            "ok": not failed,
            "tool_name": name,
            "result": result,
            "outcome": tool_outcome.to_dict(),
            "call_hash": call_hash,
            "warnings": warnings,
        }

    def prepare_invocation(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Reserve one bounded invocation without executing the callable."""
        if self._invocation_count >= self.max_invocations_per_turn:
            return {
                "ok": False,
                "error_code": "tool_budget_exhausted",
                "error": "The per-turn routed tool invocation budget was exhausted",
            }
        name = str(tool_name)
        values = dict(arguments)
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
        call_hash = hashlib.sha256(f"{name}\n{canonical}".encode()).hexdigest()
        if call_hash in self._successful_calls:
            return {
                "ok": False,
                "error_code": "duplicate_tool_call",
                "error": "This exact tool call already succeeded in the current turn",
                "call_hash": call_hash,
            }
        attempts = self._call_attempts.get(call_hash, 0)
        if attempts >= self.max_attempts_per_call:
            return {
                "ok": False,
                "error_code": "tool_retry_limit",
                "error": "This exact tool call reached its bounded retry limit",
                "call_hash": call_hash,
            }
        self._call_attempts[call_hash] = attempts + 1
        self._invocation_count += 1
        return {"ok": True, "call_hash": call_hash}

    def record_invocation(self, call_hash: str, *, success: bool) -> None:
        """Record completion for duplicate-call protection."""
        if success:
            self._successful_calls.add(call_hash)

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return the minimal JSON-safe state needed to resume a turn."""
        return {
            "selected": sorted(self._selected),
            "call_attempts": dict(self._call_attempts),
            "successful_calls": sorted(self._successful_calls),
            "invocation_count": self._invocation_count,
            "user_id": self._active_user_id,
        }

    def restore_state(self, value: Optional[Mapping[str, Any]]) -> None:
        """Restore a previously checkpointed routing budget."""
        state = dict(value or {})
        self._selected = {str(item) for item in state.get("selected") or []}
        self._call_attempts = {
            str(key): int(count)
            for key, count in dict(state.get("call_attempts") or {}).items()
        }
        self._successful_calls = {
            str(item) for item in state.get("successful_calls") or []
        }
        self._invocation_count = max(0, int(state.get("invocation_count") or 0))
        self._active_user_id = state.get("user_id")


__all__ = [
    "ContextPolicy",
    "SemanticToolRouter",
    "ToolOutcome",
    "ToolOutcomeStatus",
    "ToolPolicy",
    "ToolResult",
    "ToolResultPolicy",
    "callable_json_schema",
    "governed_tool",
    "normalize_tool_result",
    "policy_for_callable",
    "serialize_tool_result",
    "tool_metadata_to_openai",
]
