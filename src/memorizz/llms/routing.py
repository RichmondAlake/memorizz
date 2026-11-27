# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""How a model gateway chooses who answers a request.

A gateway such as OpenRouter serves one model from several providers, and can
fall back to other models. :class:`ModelRouting` is that choice as plain
settings: it is saved with the agent's ``llm_config`` and holds no secrets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Union

SORTS = ("price", "throughput", "latency")
DATA_COLLECTION = ("allow", "deny")


def _names(value: Any, what: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        names = [str(item).strip() for item in value]
    except TypeError:
        raise ValueError(f"{what} must be a list of names") from None
    return [name for name in dict.fromkeys(names) if name]


@dataclass
class ModelRouting:
    """Routing rules for one model on a gateway.

    Every field is optional; an empty routing leaves the gateway's defaults.

    fallback_models:
        Models to try, in order, when the agent's model fails (an error, a
        rate limit, an outage). The answer is billed at the rate of the model
        that gave it.
    providers:
        Providers to try first, in order (``["anthropic", "google-vertex"]``).
    only / ignore:
        The only providers allowed, and providers never to use.
    allow_fallbacks:
        ``False`` keeps a request on ``providers`` and fails instead of
        moving to another provider.
    sort:
        Prefer the cheapest (``"price"``), the fastest in tokens per second
        (``"throughput"``) or the quickest to start (``"latency"``).
    data_collection:
        ``"deny"`` uses only providers that do not keep or train on prompts.
    zero_data_retention:
        ``True`` uses only endpoints that retain nothing.
    require_parameters:
        ``True`` uses only providers that support every parameter sent, such
        as tools or a response format.
    quantizations:
        Accepted quantization levels (``["fp8", "int4"]``).
    max_price:
        The most to pay, in dollars per million tokens:
        ``{"prompt": 1, "completion": 2}``. A request over it does not run.
    """

    fallback_models: List[str] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    only: List[str] = field(default_factory=list)
    ignore: List[str] = field(default_factory=list)
    allow_fallbacks: Optional[bool] = None
    sort: Optional[str] = None
    data_collection: Optional[str] = None
    zero_data_retention: Optional[bool] = None
    require_parameters: Optional[bool] = None
    quantizations: List[str] = field(default_factory=list)
    max_price: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.fallback_models = _names(self.fallback_models, "fallback_models")
        self.providers = _names(self.providers, "providers")
        self.only = _names(self.only, "only")
        self.ignore = _names(self.ignore, "ignore")
        self.quantizations = _names(self.quantizations, "quantizations")
        if self.sort is not None:
            self.sort = str(self.sort).strip().lower()
            if self.sort not in SORTS:
                raise ValueError(f"sort must be one of {', '.join(SORTS)}")
        if self.data_collection is not None:
            self.data_collection = str(self.data_collection).strip().lower()
            if self.data_collection not in DATA_COLLECTION:
                raise ValueError("data_collection must be 'allow' or 'deny'")
        both = set(self.only) & set(self.ignore)
        if both:
            raise ValueError(
                f"A provider cannot be both allowed and ignored: {sorted(both)}"
            )
        if self.max_price is not None:
            if not isinstance(self.max_price, Mapping):
                raise ValueError("max_price must map 'prompt'/'completion' to a price")
            self.max_price = {
                str(key): float(value) for key, value in self.max_price.items()
            }

    @classmethod
    def coerce(
        cls, value: Union["ModelRouting", Mapping[str, Any], None]
    ) -> Optional["ModelRouting"]:
        """A routing from one, from a saved dict, or None for none."""
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("routing must be a ModelRouting or a dict")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown routing settings: {', '.join(unknown)}")
        return cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        """Only what was set, for saving with the agent."""
        out: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None or value == []:
                continue
            out[item.name] = (
                list(value)
                if isinstance(value, list)
                else dict(value)
                if isinstance(value, dict)
                else value
            )
        return out

    def __bool__(self) -> bool:
        return bool(self.to_dict())

    def to_request(self) -> Dict[str, Any]:
        """The fields a gateway's chat request takes for these rules."""
        provider: Dict[str, Any] = {}
        if self.providers:
            provider["order"] = list(self.providers)
        if self.only:
            provider["only"] = list(self.only)
        if self.ignore:
            provider["ignore"] = list(self.ignore)
        if self.allow_fallbacks is not None:
            provider["allow_fallbacks"] = bool(self.allow_fallbacks)
        if self.sort:
            provider["sort"] = self.sort
        if self.data_collection:
            provider["data_collection"] = self.data_collection
        if self.zero_data_retention is not None:
            provider["zdr"] = bool(self.zero_data_retention)
        if self.require_parameters is not None:
            provider["require_parameters"] = bool(self.require_parameters)
        if self.quantizations:
            provider["quantizations"] = list(self.quantizations)
        if self.max_price:
            provider["max_price"] = dict(self.max_price)
        body: Dict[str, Any] = {}
        if self.fallback_models:
            body["fallback_models"] = list(self.fallback_models)
        if provider:
            body["provider"] = provider
        return body


__all__ = ["ModelRouting"]
