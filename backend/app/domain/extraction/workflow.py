from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.extraction.extraction_context import ExtractionContext
from app.domain.extraction.file_ranking import RankedFile
from app.domain.semantics import VocabQuery, VocabQueryResult


class ChunkingRequiredError(Exception):
    pass


class ExtractionResultNotFoundError(Exception):
    pass


class ExtractionValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("Projected extraction result did not satisfy the selected profile.")
        self.errors = errors


class ExtractionChunkRef(BaseModel):
    chunk_index: int = Field(..., ge=0)
    file_path: str
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)


class ExtractionVocabQueryConfig(BaseModel):
    qualitative_vocab_identifiers: list[str] = Field(default_factory=list)
    vector_top_k: int = Field(default=12, ge=1)
    fulltext_top_k: int = Field(default=12, ge=1)
    seed_top_k: int = Field(default=6, ge=1)
    max_hops: int = Field(default=1, ge=0)
    max_statements_per_seed: int = Field(default=12, ge=1)
    traversal_direction: str = "undirected"
    vector_weight: float = Field(default=1.0, gt=0)
    fulltext_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=1)
    quantitative_vector_top_k: int = Field(default=12, ge=1)
    quantitative_fulltext_top_k: int = Field(default=12, ge=1)
    quantitative_seed_top_k: int = Field(default=6, ge=1)
    quantitative_max_hops: int = Field(default=0, ge=0)
    quantitative_max_statements_per_seed: int = Field(default=12, ge=1)
    quantitative_traversal_direction: str = "undirected"
    quantitative_vector_weight: float = Field(default=1.0, gt=0)
    quantitative_fulltext_weight: float = Field(default=1.0, gt=0)
    quantitative_rrf_k: int = Field(default=60, ge=1)


class ExtractionVocabQueryRecord(BaseModel):
    query_id: str
    kind: str
    source_value: str
    source_context: dict[str, Any] = Field(default_factory=dict)
    vocabulary_identifier: str
    rdf_type: str
    query: VocabQuery
    status: str = Field("pending", pattern="^(pending|running|completed|failed)$")
    result: VocabQueryResult | None = None
    error: str | None = None
    duration_ms: float | None = None


class ExtractionChunkResult(ExtractionChunkRef):
    status: str = Field("pending", pattern="^(pending|running|completed|failed)$")
    extraction_context: ExtractionContext | None = None
    error: str | None = None
    response_duration_ms: float | None = None
    context_tokens: int | None = None
    vocab_queries: list[ExtractionVocabQueryRecord] = Field(default_factory=list)


class ExtractionRunState(BaseModel):
    profile_identifier: str | None = None
    vocab_query_config: ExtractionVocabQueryConfig = Field(default_factory=ExtractionVocabQueryConfig)
    ranked_files: list[RankedFile] = Field(default_factory=list)
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class ExtractionRunProgress(BaseModel):
    stage: str = "pending"
    processed_chunks: int = 0
    total_chunks: int = 0
    normalized_quantities: int = 0
    normalized_qualitative_attributes: int = 0
    interim_context: ExtractionContext | None = None
    vocab_query_config: ExtractionVocabQueryConfig = Field(default_factory=ExtractionVocabQueryConfig)
    ranked_files: list[RankedFile] = Field(default_factory=list)
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)
    current_chunk: ExtractionChunkRef | None = None
    warnings: list[str] = Field(default_factory=list)


class ExtractionRunResult(BaseModel):
    document: dict[str, Any]
    extraction_context: ExtractionContext
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
