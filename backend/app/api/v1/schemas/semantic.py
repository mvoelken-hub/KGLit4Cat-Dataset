from typing import Any

from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus
from app.domain.semantics import TraversalDirection, VocabQuery, VocabQueryResult


class VocabTermSchemeResponse(BaseModel):
    rdf_type: str = Field(...)
    properties: list[str] = Field(default_factory=list)
    applicable_relationships: list[str] = Field(default_factory=list)
    count: int = Field(...)

class VocabSchemeInfoResponse(BaseModel):
    identifier: str
    source: str
    rdf_format: str
    num_triples: int
    description: str | None = None
    vocab_term_schemes: list[VocabTermSchemeResponse] = Field(default_factory=list)

class VocabEmbeddingUpdateResponse(BaseModel):
    pending_updates: int
    task_status: TaskStatus


class VocabQueryRequest(BaseModel):
    rdf_type: str
    vector_query: str | None = None
    fulltext_query: str | None = None
    allowed_rel_types: list[str] | None = None
    traversal_direction: TraversalDirection = "undirected"
    vector_top_k: int = Field(default=10, ge=1)
    fulltext_top_k: int = Field(default=10, ge=1)
    seed_top_k: int = Field(default=5, ge=1)
    max_hops: int = Field(default=1, ge=0)
    max_statements_per_seed: int = Field(default=12, ge=1)
    vector_weight: float = Field(default=1.0, gt=0)
    fulltext_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=1)

    def to_domain(self) -> VocabQuery:
        return VocabQuery.model_validate(self.model_dump())


class VocabSeedResponse(BaseModel):
    uri: str
    rdf_type: str
    rrf_score: float
    vector_score: float | None = None
    vector_rank: int | None = None
    fulltext_score: float | None = None
    fulltext_rank: int | None = None


class VocabGraphStatementResponse(BaseModel):
    subject_uri: str
    predicate: str
    object_uri: str


class CompactVocabResourceResponse(BaseModel):
    uri: str
    rdf_types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class VocabQueryResultResponse(BaseModel):
    identifier: str
    rdf_type: str
    seeds: list[VocabSeedResponse] = Field(default_factory=list)
    graph_statements: list[VocabGraphStatementResponse] = Field(default_factory=list)
    resources: dict[str, CompactVocabResourceResponse] = Field(default_factory=dict)


def _vocab_query_result_response(result: VocabQueryResult) -> VocabQueryResultResponse:
    return VocabQueryResultResponse.model_validate(result.model_dump())
