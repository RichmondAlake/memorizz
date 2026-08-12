"""Versioned governed semantic-layer contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_SEMANTIC_NAME = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SemanticEntity(BaseModel):
    name: str
    source: str
    primary_key: str
    description: str = ""


class Measure(BaseModel):
    name: str
    entity: str
    expression: str
    aggregation: Literal["sum", "avg", "min", "max", "count", "count_distinct"]
    description: str = ""
    data_type: str = "number"


class Dimension(BaseModel):
    name: str
    entity: str
    expression: str
    description: str = ""
    data_type: str = "string"


class Relationship(BaseModel):
    name: str
    from_entity: str
    to_entity: str
    on: str
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one"]


class SemanticPolicy(BaseModel):
    name: str
    entity: str
    allowed_roles: List[str] = Field(default_factory=list)
    row_filter: Optional[str] = None
    description: str = ""


class SemanticSynonym(BaseModel):
    alias: str
    target: str
    kind: Literal["entity", "measure", "dimension"]


class LineageEdge(BaseModel):
    source: str
    target: str
    transform: Optional[str] = None


class SemanticModel(BaseModel):
    name: str
    version: str
    entities: List[SemanticEntity]
    measures: List[Measure] = Field(default_factory=list)
    dimensions: List[Dimension] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    policies: List[SemanticPolicy] = Field(default_factory=list)
    synonyms: List[SemanticSynonym] = Field(default_factory=list)
    lineage: List[LineageEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "version")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("semantic model name/version cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_references(self):
        def unique(values, label):
            names = [item.name for item in values]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} names are not allowed")
            return set(names)

        entities = unique(self.entities, "entity")
        measures = unique(self.measures, "measure")
        dimensions = unique(self.dimensions, "dimension")
        unique(self.relationships, "relationship")
        unique(self.policies, "policy")
        governed_names = [
            *(item.name for item in self.entities),
            *(item.name for item in self.measures),
            *(item.name for item in self.dimensions),
            *(item.name for item in self.relationships),
            *(item.name for item in self.policies),
            *(item.alias for item in self.synonyms),
        ]
        invalid = [
            name for name in governed_names if not _SEMANTIC_NAME.fullmatch(name)
        ]
        if invalid:
            raise ValueError(
                "semantic names must be safe identifiers: " + ", ".join(invalid)
            )
        for item in [*self.measures, *self.dimensions, *self.policies]:
            if item.entity not in entities:
                raise ValueError(f"unknown semantic entity '{item.entity}'")
        for item in self.relationships:
            if item.from_entity not in entities or item.to_entity not in entities:
                raise ValueError(
                    f"relationship '{item.name}' references unknown entity"
                )
        targets = {
            "entity": entities,
            "measure": measures,
            "dimension": dimensions,
        }
        aliases = set()
        for item in self.synonyms:
            if item.alias in aliases:
                raise ValueError(f"duplicate semantic synonym '{item.alias}'")
            aliases.add(item.alias)
            if item.target not in targets[item.kind]:
                raise ValueError(
                    f"synonym '{item.alias}' references unknown {item.kind}"
                )
        return self


class SemanticFilter(BaseModel):
    field: str
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"]
    value: Any


class SemanticQuery(BaseModel):
    measures: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[SemanticFilter] = Field(default_factory=list)
    order_by: List[str] = Field(default_factory=list)
    limit: int = 100
    principal_id: Optional[str] = None
    roles: List[str] = Field(default_factory=list)

    @field_validator("limit")
    @classmethod
    def bounded_limit(cls, value: int) -> int:
        if not 1 <= int(value) <= 10_000:
            raise ValueError("semantic query limit must be between 1 and 10000")
        return int(value)

    @model_validator(mode="after")
    def require_projection(self):
        if not self.measures and not self.dimensions:
            raise ValueError("semantic query requires a measure or dimension")
        return self


class ValidatedQueryPlan(BaseModel):
    valid: Literal[True] = True
    model_name: str
    model_version: str
    plan_id: str
    fingerprint: str
    sql: str
    parameters: Dict[str, Any]
    entities: List[str]
    measures: List[str]
    dimensions: List[str]
    relationships: List[str]
    policies: List[str]
    lineage: List[LineageEdge]
