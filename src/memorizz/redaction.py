# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""One redaction engine for every credential-scrubbing surface.

MCP tool traffic, harness reports, memory archives, the local UI and the
observability validator all need to recognise the same credential-bearing
keys and token shapes, but differ in how a key is anchored (whole name,
suffix or substring), what happens to a match (mask, drop or pseudonymise)
and which string leaves are scrubbed. The vocabulary lives here once; each
surface declares a :class:`RedactionPolicy` and calls :func:`redact`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Mapping, Optional, Sequence, Tuple

# --- Sensitive key names --------------------------------------------------------

#: Regex fragments naming credential-bearing keys: the union of every surface's
#: former list. Anchoring (whole key, suffix or substring) is a per-surface
#: choice made through :class:`KeyMatcher`.
SENSITIVE_KEY_NAMES: Tuple[str, ...] = (
    r"api[_-]?key",
    r"access[_-]?token",
    r"refresh[_-]?token",
    r"bearer[_-]?token",
    r"client[_-]?secret",
    r"secret[_-]?access[_-]?key",
    r"private[_-]?key",
    r"code[_-]?verifier",
    r"credentials?",
    r"authorization",
    r"password",
    r"passwd",
    r"secret",
    r"token",
    r"cookie",
)


def _compile(template: str, names: Sequence[str]) -> Optional[re.Pattern]:
    if not names:
        return None
    return re.compile(template % "|".join(names), re.I)


class KeyMatcher:
    """Decide whether a key names a credential.

    ``exact`` names must be the whole key (surrounding whitespace ignored),
    ``suffix`` names must be the whole key or follow a ``_``/``-`` separator
    (``aws_secret_access_key``), and ``search`` names may appear anywhere in
    the key. Matching ignores case.
    """

    def __init__(
        self,
        *,
        exact: Sequence[str] = (),
        suffix: Sequence[str] = (),
        search: Sequence[str] = (),
    ) -> None:
        self._exact = _compile(r"\s*(?:%s)\s*", exact)
        self._suffix = _compile(r"(?:.*[_-])?(?:%s)", suffix)
        self._search = _compile(r"(?:%s)", search)

    def __call__(self, key: Any) -> bool:
        text = str(key)
        return bool(
            (self._exact is not None and self._exact.fullmatch(text))
            or (self._suffix is not None and self._suffix.fullmatch(text))
            or (self._search is not None and self._search.search(text))
        )


_TOKEN_TELEMETRY_KEYS = re.compile(
    r"^(?:[a-z0-9]+_)*(?:input|output|total|prompt|completion|reasoning|"
    r"cached|candidate|context|memory|evidence|thinking|write)_tokens(?:_details)?"
    r"(?:_(?:mean|median|min|max|sum|stdev|sample_stdev|p(?:50|90|95|99)|"
    r"mean_delta|paired_deltas))?$|"
    r"^(?:tokens_(?:used|saved|avoided)|token_estimates?|token_count)$|"
    r"^[a-z0-9_]+_(?:token_count|token_budget)$|^per_million_tokens$",
    re.I,
)


def is_token_telemetry(key: str, value: Any) -> bool:
    """Whether ``key``/``value`` is a token *count*, not a similarly named credential.

    Suitable as a :attr:`RedactionPolicy.preserve` hook for surfaces that
    match ``token`` as a substring.
    """
    # camelCase counters (pi's ``totalTokens``) match their snake_case form.
    normalized_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized_key = normalized_key.replace("-", "_").lower()
    if normalized_key == "token_reporting":
        return isinstance(value, bool) or value is None
    if normalized_key in {"tokens", "token_usage"}:
        # A block of counters (Hermes reports ``tokens: {input, output, ...}``).
        return isinstance(value, Mapping) and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            or item is None
            for item in value.values()
        )
    if not _TOKEN_TELEMETRY_KEYS.fullmatch(normalized_key):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Mapping, list, tuple)) or value is None:
        return True
    return isinstance(value, str) and value.strip().isdigit()


# --- Sensitive value shapes -----------------------------------------------------

_EMAIL_CORE = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
_BEARER = r"(?i:Bearer\s+[A-Za-z0-9._~+/=-]{8,})"
_SECRET_TOKEN = "|".join(
    (
        # Vendor key prefixes separated by "-" or "_" (OpenAI/Anthropic sk-,
        # Stripe pk, Tavily, E2B, npm, PyPI), not glued to other alphanumerics.
        r"(?i:(?<![A-Za-z0-9])(?:sk|pk|tvly|e2b|npm|pypi)[-_]"
        r"[A-Za-z0-9._-]{12,}(?![A-Za-z0-9]))",
        # OpenAI/Anthropic, GitHub and Slack tokens as whole words.
        r"\b(?:sk-(?:ant-)?|gh[oprsu]_|xox[baprs]-)[A-Za-z0-9_-]{8,}\b",
        # Short forms recognised by the observability validator.
        r"(?i:\b(?:sk-|ghp_|xoxb-)[\w-]{8,})",
    )
)

#: ``Bearer <token>`` with at least eight token characters.
BEARER_TOKEN = re.compile(_BEARER)
#: API keys and access tokens recognised by vendor prefix.
SECRET_TOKEN = re.compile(_SECRET_TOKEN)
#: Either of the above: the credential scrub applied to free text.
SENSITIVE_VALUE = re.compile(f"{_BEARER}|{_SECRET_TOKEN}")
#: ``scheme://user:password@`` URI credentials; ``scheme`` is a named group.
URI_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@", re.I
)
#: ``?api_key=…`` style query-string secrets; group 1 is the ``key=`` prefix.
QUERY_SECRET = re.compile(r"([?&](?:api[_-]?key|token|secret|password)=)[^&#\s]+", re.I)
#: An e-mail address, for detection (may touch adjacent punctuation).
EMAIL_ADDRESS = re.compile(_EMAIL_CORE, re.I)
#: An e-mail address bounded so a replacement never eats neighbouring text.
EMAIL = re.compile(rf"(?<![\w.+-]){_EMAIL_CORE}(?![\w.-])")


def scrub_text(text: str, rules: Sequence[Tuple[re.Pattern, Any]]) -> str:
    """Apply ``(pattern, replacement)`` substitutions to ``text`` in order."""
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


# --- Policies -------------------------------------------------------------------

MASK = "mask"
DROP = "drop"
PSEUDONYM = "pseudonym"
#: Extra verdicts a ``key_rule`` may return: drop a key without reporting it,
#: or keep a key regardless of the key patterns.
OMIT = "omit"
KEEP = "keep"


@dataclass(frozen=True)
class RedactionPolicy:
    """How one surface treats sensitive keys and string leaves.

    A key recognised by ``keys`` is masked with ``replacement`` (``mode=MASK``),
    removed (``DROP``) or replaced by ``pseudonym(value)`` (``PSEUDONYM``).
    ``preserve(key, value)`` exempts a matching key (token telemetry);
    ``exempt(parent_path)`` exempts every key directly under a dotted path
    (JSON-schema property names); ``key_rule(key, value, parent_path)`` may
    return an action for any key before the patterns are consulted.
    ``pseudonym_keys`` are always pseudonymised when their value is truthy.
    ``values`` are ``(pattern, replacement)`` pairs applied in order to every
    string leaf. Tuples come back as lists unless ``tuples="tuple"``.
    """

    keys: KeyMatcher
    mode: str = MASK
    replacement: str = "[REDACTED]"
    pseudonym: Optional[Callable[[Any], Any]] = None
    pseudonym_keys: FrozenSet[str] = frozenset()
    preserve: Optional[Callable[[str, Any], bool]] = None
    exempt: Optional[Callable[[str], bool]] = None
    key_rule: Optional[Callable[[str, Any, str], Optional[str]]] = None
    values: Tuple[Tuple[re.Pattern, Any], ...] = ()
    tuples: str = "list"

    def __post_init__(self) -> None:
        if self.mode not in {MASK, DROP, PSEUDONYM}:
            raise ValueError(f"Unknown redaction mode: {self.mode!r}")
        if (self.mode == PSEUDONYM or self.pseudonym_keys) and self.pseudonym is None:
            raise ValueError("A pseudonym hook is required to pseudonymise")
        if self.tuples not in {"list", "tuple"}:
            raise ValueError(f"Unknown tuple handling: {self.tuples!r}")


class _Walker:
    __slots__ = ("policy", "on_drop")

    def __init__(self, policy: RedactionPolicy, on_drop) -> None:
        self.policy = policy
        self.on_drop = on_drop

    def decide(self, key: str, value: Any, parent: str) -> Optional[str]:
        policy = self.policy
        if policy.key_rule is not None:
            verdict = policy.key_rule(key, value, parent)
            if verdict is not None:
                return verdict
        if (
            policy.keys(key)
            and not (policy.exempt is not None and policy.exempt(parent))
            and not (policy.preserve is not None and policy.preserve(key, value))
        ):
            return policy.mode
        if key in policy.pseudonym_keys and value:
            return PSEUDONYM
        return None

    def apply(self, action: str, value: Any) -> Any:
        if action == MASK:
            return self.policy.replacement
        if action == PSEUDONYM:
            return self.policy.pseudonym(value)
        return None

    def descend(self, value: Any, location: str) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for raw_key, child in value.items():
                key = str(raw_key)
                action = self.decide(key, child, location)
                if action is None or action == KEEP:
                    child_location = f"{location}.{key}" if location else key
                    result[key] = self.descend(child, child_location)
                elif action == DROP:
                    if self.on_drop is not None:
                        self.on_drop(f"{location}.{key}" if location else key)
                elif action != OMIT:
                    result[key] = self.apply(action, child)
            return result
        if isinstance(value, (list, tuple)):
            items = [self.descend(item, location) for item in value]
            if isinstance(value, tuple) and self.policy.tuples == "tuple":
                return tuple(items)
            return items
        if isinstance(value, str) and self.policy.values:
            return scrub_text(value, self.policy.values)
        return value


def redact(
    value: Any,
    policy: RedactionPolicy,
    *,
    key: Any = "",
    path: str = "",
    on_drop: Optional[Callable[[str], Any]] = None,
) -> Any:
    """Redact ``value`` under ``policy``.

    ``key`` is the name ``value`` was stored under, when known: a sensitive
    top-level key masks or pseudonymises the whole value (``None`` when the
    policy drops). ``path`` is the dotted location of ``value`` and ``on_drop``
    receives the dotted location of every key the policy drops.
    """
    walker = _Walker(policy, on_drop)
    name = str(key)
    if name:
        action = walker.decide(name, value, path)
        if action is not None and action != KEEP:
            return walker.apply(action, value)
    location = f"{path}.{name}" if path and name else (name or path)
    return walker.descend(value, location)


__all__ = [
    "BEARER_TOKEN",
    "DROP",
    "EMAIL",
    "EMAIL_ADDRESS",
    "KEEP",
    "KeyMatcher",
    "MASK",
    "OMIT",
    "PSEUDONYM",
    "QUERY_SECRET",
    "RedactionPolicy",
    "SECRET_TOKEN",
    "SENSITIVE_KEY_NAMES",
    "SENSITIVE_VALUE",
    "URI_CREDENTIALS",
    "is_token_telemetry",
    "redact",
    "scrub_text",
]
