"""Deterministic semantic catalog and governed query planner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    Dimension,
    LineageEdge,
    Measure,
    Relationship,
    SemanticModel,
    SemanticQuery,
    ValidatedQueryPlan,
)

_SAFE_SOURCE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]*(\.[A-Za-z_][A-Za-z0-9_$#]*)?$")
_OPERATORS = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "in": "IN",
    "not_in": "NOT IN",
}


class SemanticCatalog:
    """Own immutable model versions and emit parameterized validated plans."""

    def __init__(
        self, models: Optional[Iterable[SemanticModel | Dict[str, Any]]] = None
    ):
        self._models: Dict[tuple[str, str], SemanticModel] = {}
        self._active: Dict[str, str] = {}
        for model in models or []:
            self.register(model, activate=True)

    def register(
        self,
        model: SemanticModel | Dict[str, Any],
        *,
        activate: bool = True,
    ) -> SemanticModel:
        value = (
            model
            if isinstance(model, SemanticModel)
            else SemanticModel.model_validate(model)
        )
        for entity in value.entities:
            if not _SAFE_SOURCE.fullmatch(entity.source):
                raise ValueError(f"unsafe entity source '{entity.source}'")
        key = (value.name, value.version)
        if key in self._models and self._models[key] != value:
            raise ValueError(
                f"semantic model {value.name}@{value.version} is immutable"
            )
        self._models[key] = value
        if activate:
            self._active[value.name] = value.version
        return value

    def get(self, name: str, version: Optional[str] = None) -> SemanticModel:
        resolved = version or self._active.get(name)
        if not resolved or (name, resolved) not in self._models:
            raise KeyError(f"unknown semantic model '{name}@{resolved or 'active'}'")
        return self._models[(name, resolved)]

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "version": version,
                "active": self._active.get(name) == version,
            }
            for name, version in sorted(self._models)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete versioned catalog without executable code."""
        return {
            "format": "memorizz-semantic-catalog-v1",
            "active": dict(self._active),
            "models": [
                model.model_dump(mode="json")
                for _, model in sorted(self._models.items())
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SemanticCatalog":
        """Restore a catalog from its validated, data-only representation."""
        if payload.get("format") != "memorizz-semantic-catalog-v1":
            raise ValueError("unsupported semantic catalog format")
        catalog = cls(payload.get("models") or [])
        active = dict(payload.get("active") or {})
        for name, version in active.items():
            catalog.get(str(name), str(version))
        catalog._active = {str(name): str(version) for name, version in active.items()}
        return catalog

    def describe(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        model = self.get(name, version)
        return {
            "name": model.name,
            "version": model.version,
            "entities": [item.model_dump() for item in model.entities],
            "measures": [item.model_dump() for item in model.measures],
            "dimensions": [item.model_dump() for item in model.dimensions],
            "relationships": [item.model_dump() for item in model.relationships],
            "policies": [item.model_dump() for item in model.policies],
            "synonyms": [item.model_dump() for item in model.synonyms],
        }

    @staticmethod
    def _resolve_name(model: SemanticModel, kind: str, value: str) -> str:
        synonyms = {
            item.alias: item.target for item in model.synonyms if item.kind == kind
        }
        return synonyms.get(value, value)

    @staticmethod
    def _join_path(
        relationships: List[Relationship], entities: List[str]
    ) -> List[Relationship]:
        if len(entities) <= 1:
            return []
        joined = {entities[0]}
        remaining = set(entities[1:])
        selected: List[Relationship] = []
        while remaining:
            queue = deque((entity, []) for entity in joined)
            visited = set(joined)
            found = None
            while queue and found is None:
                current, path = queue.popleft()
                for relationship in relationships:
                    if relationship.from_entity == current:
                        other = relationship.to_entity
                    elif relationship.to_entity == current:
                        other = relationship.from_entity
                    else:
                        continue
                    if other in visited:
                        continue
                    next_path = [*path, relationship]
                    if other in remaining:
                        found = (other, next_path)
                        break
                    visited.add(other)
                    queue.append((other, next_path))
            if found is None:
                raise ValueError("requested semantic entities are not connected")
            entity, path = found
            for relationship in path:
                if relationship not in selected:
                    selected.append(relationship)
                joined.update({relationship.from_entity, relationship.to_entity})
            remaining.remove(entity)
        return selected

    def plan(
        self,
        model_name: str,
        query: SemanticQuery | Dict[str, Any],
        *,
        version: Optional[str] = None,
    ) -> ValidatedQueryPlan:
        model = self.get(model_name, version)
        request = (
            query
            if isinstance(query, SemanticQuery)
            else SemanticQuery.model_validate(query)
        )
        measures = {item.name: item for item in model.measures}
        dimensions = {item.name: item for item in model.dimensions}
        requested_measures: List[Measure] = []
        requested_dimensions: List[Dimension] = []
        for raw in request.measures:
            name = self._resolve_name(model, "measure", raw)
            if name not in measures:
                raise ValueError(f"unknown measure '{raw}'")
            requested_measures.append(measures[name])
        for raw in request.dimensions:
            name = self._resolve_name(model, "dimension", raw)
            if name not in dimensions:
                raise ValueError(f"unknown dimension '{raw}'")
            requested_dimensions.append(dimensions[name])

        fields: Dict[str, Measure | Dimension] = {**measures, **dimensions}
        resolved_filters = []
        for item in request.filters:
            field_name = self._resolve_name(model, "dimension", item.field)
            if field_name not in fields:
                field_name = self._resolve_name(model, "measure", item.field)
            if field_name not in fields:
                raise ValueError(f"unknown semantic filter field '{item.field}'")
            resolved_filters.append((fields[field_name], item))

        entity_names: List[str] = []
        for item in [
            *requested_dimensions,
            *requested_measures,
            *(f[0] for f in resolved_filters),
        ]:
            if item.entity not in entity_names:
                entity_names.append(item.entity)
        entity_map = {item.name: item for item in model.entities}
        joins = self._join_path(model.relationships, entity_names)
        for relationship in joins:
            for entity in (relationship.from_entity, relationship.to_entity):
                if entity not in entity_names:
                    entity_names.append(entity)

        roles = set(request.roles)
        applied_policies = []
        where = []
        for policy in model.policies:
            if policy.entity not in entity_names:
                continue
            if policy.allowed_roles and not roles.intersection(policy.allowed_roles):
                raise PermissionError(
                    f"principal is not authorized by semantic policy '{policy.name}'"
                )
            applied_policies.append(policy.name)
            if policy.row_filter:
                where.append(f"({policy.row_filter})")

        projections = []
        groups = []
        for item in requested_dimensions:
            projections.append(f"{item.expression} AS {item.name}")
            groups.append(item.expression)
        aggregation = {
            "sum": "SUM",
            "avg": "AVG",
            "min": "MIN",
            "max": "MAX",
            "count": "COUNT",
            "count_distinct": "COUNT(DISTINCT",
        }
        for item in requested_measures:
            if item.aggregation == "count_distinct":
                expression = f"COUNT(DISTINCT {item.expression})"
            else:
                expression = f"{aggregation[item.aggregation]}({item.expression})"
            projections.append(f"{expression} AS {item.name}")

        params: Dict[str, Any] = {}
        for index, (field, item) in enumerate(resolved_filters):
            operator = _OPERATORS[item.operator]
            key = f"filter_{index}"
            if item.operator in {"in", "not_in"}:
                if not isinstance(item.value, list) or not item.value:
                    raise ValueError(f"filter '{item.field}' requires a non-empty list")
                binds = []
                for item_index, value in enumerate(item.value):
                    bind = f"{key}_{item_index}"
                    binds.append(f":{bind}")
                    params[bind] = value
                where.append(f"{field.expression} {operator} ({', '.join(binds)})")
            else:
                params[key] = item.value
                where.append(f"{field.expression} {operator} :{key}")

        base = entity_map[entity_names[0]]
        sql = f"SELECT {', '.join(projections)} FROM {base.source}"
        joined = {base.name}
        pending = list(joins)
        while pending:
            progress = False
            for relationship in list(pending):
                if relationship.from_entity in joined:
                    target = relationship.to_entity
                elif relationship.to_entity in joined:
                    target = relationship.from_entity
                else:
                    continue
                sql += f" JOIN {entity_map[target].source} ON {relationship.on}"
                joined.add(target)
                pending.remove(relationship)
                progress = True
            if not progress:
                raise ValueError("could not construct a deterministic join plan")
        if where:
            sql += " WHERE " + " AND ".join(where)
        if groups and requested_measures:
            sql += " GROUP BY " + ", ".join(groups)
        valid_aliases = {
            item.name for item in [*requested_dimensions, *requested_measures]
        }
        order = []
        for raw in request.order_by:
            name, _, direction = raw.partition(":")
            if name not in valid_aliases or direction.lower() not in {
                "",
                "asc",
                "desc",
            }:
                raise ValueError(f"invalid semantic order field '{raw}'")
            order.append(f"{name} {direction.upper() or 'ASC'}")
        if order:
            sql += " ORDER BY " + ", ".join(order)
        params["result_limit"] = request.limit
        sql += " FETCH FIRST :result_limit ROWS ONLY"

        canonical = json.dumps(
            {
                "model": model.model_dump(mode="json"),
                "query": request.model_dump(mode="json"),
                "sql": sql,
                "parameters": params,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        lineage = list(model.lineage)
        for item in [*requested_dimensions, *requested_measures]:
            lineage.append(
                LineageEdge(
                    source=f"{item.entity}.{item.expression}",
                    target=item.name,
                    transform=(item.aggregation if isinstance(item, Measure) else None),
                )
            )
        return ValidatedQueryPlan(
            model_name=model.name,
            model_version=model.version,
            plan_id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            sql=sql,
            parameters=params,
            entities=entity_names,
            measures=[item.name for item in requested_measures],
            dimensions=[item.name for item in requested_dimensions],
            relationships=[item.name for item in joins],
            policies=applied_policies,
            lineage=lineage,
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        fd, temporary = tempfile.mkstemp(prefix=".semantic-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SemanticCatalog":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.from_dict(payload)
