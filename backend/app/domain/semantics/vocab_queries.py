from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.semantics.controlled_vocabularies import VocabResource


TraversalDirection = Literal["outgoing", "incoming", "undirected"]
VocabSearchSource = Literal["vector", "fulltext"]


class VocabQuery(BaseModel):
    """Domain-level request for querying one vocabulary resource type."""

    rdf_type: str
    vector_query: str | None = None
    fulltext_query: str | None = None
    allowed_rel_types: list[str] | None = None
    traversal_direction: TraversalDirection = "undirected"
    vector_top_k: int = Field(default=10, ge=1)
    fulltext_top_k: int = Field(default=10, ge=1)
    seed_top_k: int = Field(default=5, ge=1)
    max_hops: int = Field(default=1, ge=0)
    max_statements_per_seed: int = Field(default=50, ge=1)
    vector_weight: float = Field(default=1.0, gt=0)
    fulltext_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=1)

    @model_validator(mode="after")
    def require_query_text(self) -> "VocabQuery":
        if not self.vector_query and not self.fulltext_query:
            raise ValueError("At least one of vector_query or fulltext_query must be provided.")
        return self


class VocabSearchCandidate(BaseModel):
    uri: str
    score: float
    rank: int = Field(..., ge=1)
    source: VocabSearchSource


class VocabSeed(BaseModel):
    uri: str
    rdf_type: str
    rrf_score: float
    vector_score: float | None = None
    vector_rank: int | None = None
    fulltext_score: float | None = None
    fulltext_rank: int | None = None


class VocabGraphStatement(BaseModel):
    subject_uri: str
    predicate: str
    object_uri: str


class CompactVocabResource(BaseModel):
    uri: str
    rdf_types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class VocabQueryResult(BaseModel):
    identifier: str
    rdf_type: str
    seeds: list[VocabSeed] = Field(default_factory=list)
    graph_statements: list[VocabGraphStatement] = Field(default_factory=list)
    resources: dict[str, CompactVocabResource] = Field(default_factory=dict)


def fuse_vocab_candidates(
    *,
    rdf_type: str,
    vector_candidates: list[VocabSearchCandidate],
    fulltext_candidates: list[VocabSearchCandidate],
    seed_top_k: int,
    vector_weight: float,
    fulltext_weight: float,
    rrf_k: int,
) -> list[VocabSeed]:
    """Fuse vector and full-text rankings with weighted reciprocal rank fusion."""

    by_uri: dict[str, VocabSeed] = {}

    def seed_for(uri: str) -> VocabSeed:
        if uri not in by_uri:
            by_uri[uri] = VocabSeed(uri=uri, rdf_type=rdf_type, rrf_score=0.0)
        return by_uri[uri]

    for candidate in vector_candidates:
        seed = seed_for(candidate.uri)
        seed.vector_score = candidate.score
        seed.vector_rank = candidate.rank
        seed.rrf_score += vector_weight * (1 / (rrf_k + candidate.rank))

    for candidate in fulltext_candidates:
        seed = seed_for(candidate.uri)
        seed.fulltext_score = candidate.score
        seed.fulltext_rank = candidate.rank
        seed.rrf_score += fulltext_weight * (1 / (rrf_k + candidate.rank))

    def best_rank(seed: VocabSeed) -> int:
        ranks = [
            rank
            for rank in (seed.vector_rank, seed.fulltext_rank)
            if rank is not None
        ]
        return min(ranks) if ranks else 0

    return sorted(
        by_uri.values(),
        key=lambda seed: (-seed.rrf_score, best_rank(seed), seed.uri),
    )[:seed_top_k]


def compact_vocab_resource(resource: VocabResource) -> CompactVocabResource:
    properties = {
        key: value
        for key, value in resource.properties.items()
        if key not in {"embedding", "embeddings"}
    }
    return CompactVocabResource(
        uri=resource.uri,
        rdf_types=resource.rdf_types,
        properties=properties,
    )


def compact_vocab_query_result(
    *,
    identifier: str,
    rdf_type: str,
    seeds: list[VocabSeed],
    graph_statements: list[VocabGraphStatement],
    resources: list[VocabResource],
) -> VocabQueryResult:
    seen_statements: set[tuple[str, str, str]] = set()
    compact_statements: list[VocabGraphStatement] = []

    for statement in graph_statements:
        key = (statement.subject_uri, statement.predicate, statement.object_uri)
        if key in seen_statements:
            continue
        seen_statements.add(key)
        compact_statements.append(statement)

    resource_map = {
        resource.uri: compact_vocab_resource(resource)
        for resource in sorted(resources, key=lambda item: item.uri)
    }

    return VocabQueryResult(
        identifier=identifier,
        rdf_type=rdf_type,
        seeds=seeds,
        graph_statements=compact_statements,
        resources=resource_map,
    )
