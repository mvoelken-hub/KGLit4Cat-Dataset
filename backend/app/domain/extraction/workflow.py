from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.extraction.extraction_context import ExtractionContext
from app.domain.extraction.file_ranking import RankedFile


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


class ExtractionChunkResult(ExtractionChunkRef):
    status: str = Field("pending", pattern="^(pending|running|completed|failed)$")
    extraction_context: ExtractionContext | None = None
    error: str | None = None
    response_duration_ms: float | None = None
    context_tokens: int | None = None


class ExtractionRunState(BaseModel):
    ranked_files: list[RankedFile] = Field(default_factory=list)
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class ExtractionRunProgress(BaseModel):
    stage: str = "pending"
    processed_chunks: int = 0
    total_chunks: int = 0
    normalized_quantities: int = 0
    normalized_qualitative_attributes: int = 0
    interim_context: ExtractionContext | None = None
    ranked_files: list[RankedFile] = Field(default_factory=list)
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)
    current_chunk: ExtractionChunkRef | None = None
    warnings: list[str] = Field(default_factory=list)


class ExtractionRunResult(BaseModel):
    document: dict[str, Any]
    extraction_context: ExtractionContext
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
