"""Neo4j semantic index models.

These models mirror rows returned by Neo4j's SHOW VECTOR/FULLTEXT INDEXES
commands and the requests used to create those indexes.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.neo4j.types import normalize_neo4j_value


SemanticIndexType = Literal["VECTOR", "FULLTEXT"]
SEMANTIC_INDEX_TYPES: tuple[SemanticIndexType, ...] = get_args(SemanticIndexType)


class IndexError(Exception):
    """Base exception for index operations."""


class IndexNotFoundError(IndexError):
    """Raised when an index cannot be found by name."""

    def __init__(self, index_name: str) -> None:
        self.index_name = index_name
        super().__init__(f"Index with name '{index_name}' not found.")


class IndexDuplicateError(IndexError):
    """Raised when multiple indexes share the same name."""

    def __init__(self, index_name: str) -> None:
        self.index_name = index_name
        super().__init__(
            f"Multiple indexes with name '{index_name}' found, "
            "which is unexpected. Please check the database integrity."
        )


class IndexCreationError(IndexError):
    """Raised when index creation fails."""

    def __init__(self, index_name: str, detail: str = "") -> None:
        self.index_name = index_name
        self.detail = detail
        msg = f"Failed to create index '{index_name}'."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class IndexTypeError(IndexError):
    """Raised when an index is not of the expected type."""

    def __init__(self, index_name: str, expected: str, actual: str) -> None:
        self.index_name = index_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Index '{index_name}' is of type '{actual}', "
            f"but '{expected}' was expected."
        )


class EntityTypeNotSupportedError(IndexError):
    """Raised when an unsupported entity type is used."""

    def __init__(self, entity_type: str) -> None:
        self.entity_type = entity_type
        super().__init__(f"Unsupported entity type '{entity_type}'.")


class VectorIndexConfiguration(BaseModel):
    dimensions: int = Field(default=768, ge=1, le=16384, serialization_alias="vector.dimensions")
    similarity_function: Literal["COSINE", "EUCLIDEAN"] = Field(default="COSINE", serialization_alias="vector.similarity_function")
    quantization_enabled: bool = Field(default=True, serialization_alias="vector.quantization.enabled")
    hnsw_max_connections: int = Field(default=16, ge=1, le=512, serialization_alias="vector.hnsw.m")
    hnsw_ef_construction: int = Field(default=100, ge=1, le=3200, serialization_alias="vector.hnsw.ef_construction")


class FullTextIndexConfiguration(BaseModel):
    analyzer: str = Field(default="english", serialization_alias="fulltext.analyzer")
    eventually_consistent: bool = Field(default=False, serialization_alias="fulltext.eventually_consistent")


class Options(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    index_config: VectorIndexConfiguration | FullTextIndexConfiguration = Field(alias="indexConfig")


class BaseIndex(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)

    id: int
    name: str
    state: str
    population_percent: float
    type: SemanticIndexType
    entity_type: Literal["NODE", "RELATIONSHIP"]
    labels_or_types: list[str] = Field(..., min_length=1)
    properties: list[str] = Field(..., min_length=1)
    index_provider: str
    options: Options
    create_statement: str
    owning_constraint: str | None = None
    last_read: str | None = None
    read_count: int | None = None
    tracked_since: str | None = None
    failure_message: str = ""

    @classmethod
    def from_row(cls, row: dict) -> Self:
        for key, val in row.items():
            row[key] = normalize_neo4j_value(val)
        return cls.model_validate(row)

    def convert_to_dedicated_index(self) -> VectorIndexInfo | FullTextIndexInfo:
        data = self.model_dump(by_alias=True)
        if self.type == "VECTOR":
            return VectorIndexInfo.model_validate(data)
        if self.type == "FULLTEXT":
            return FullTextIndexInfo.model_validate(data)
        raise ValueError(f"Unknown index type: {self.type}")


class VectorIndexInfo(BaseIndex):
    type: Literal["VECTOR"] = "VECTOR"  # type: ignore[reportIncompatibleVariableOverride]

    labels_or_types: list[str] = Field(..., min_length=1, max_length=1)
    properties: list[str] = Field(..., min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_vector_options(self) -> Self:
        if not isinstance(self.options.index_config, VectorIndexConfiguration):
            raise ValueError("VectorIndexInfo must have VectorIndexConfiguration in options")
        return self


class FullTextIndexInfo(BaseIndex):
    type: Literal["FULLTEXT"] = "FULLTEXT"  # type: ignore[reportIncompatibleVariableOverride]


BaseIndex.model_rebuild()


class CreateIndexRequest(BaseModel):
    """Request to create a Neo4j semantic index."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)

    name: str
    type: SemanticIndexType
    entity_type: Literal["NODE", "RELATIONSHIP"]
    on_label_or_type: list[str] = Field(..., min_length=1)
    on_property: list[str] = Field(..., min_length=1)
    index_config: VectorIndexConfiguration | FullTextIndexConfiguration | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def set_default_index_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("index_config") is None and data.get("indexConfig") is None:
            t = data.get("type")
            if t == "VECTOR":
                data["indexConfig"] = VectorIndexConfiguration()
            elif t == "FULLTEXT":
                data["indexConfig"] = FullTextIndexConfiguration()
        return data

    @model_validator(mode="after")
    def validate_index_config(self) -> Self:
        if self.type == "VECTOR":
            if not isinstance(self.index_config, VectorIndexConfiguration):
                raise ValueError("For VECTOR index type, index_config must be VectorIndexConfiguration")
        elif self.type == "FULLTEXT":
            if not isinstance(self.index_config, FullTextIndexConfiguration):
                raise ValueError("For FULLTEXT index type, index_config must be FullTextIndexConfiguration")
        return self

    @model_validator(mode="after")
    def validate_labels_and_properties(self) -> Self:
        if self.type == "VECTOR":
            if len(self.on_label_or_type) != 1:
                raise ValueError("VECTOR index must have exactly one label/type in on_label_or_type")
            if len(self.on_property) != 1:
                raise ValueError("VECTOR index must have exactly one property in on_property")
        return self


@dataclass(frozen=True)
class IndexQueryResult:
    """Result of a Neo4j index query."""

    entities: list[dict[str, Any]]
    scores: list[float]


@dataclass(frozen=True)
class EmbedResult:
    """Result of requesting embedding generation for a vector index."""

    index_info: VectorIndexInfo
    pending_updates: int
    status: str


# Helpers

def normalize_index_token(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_").lower()

