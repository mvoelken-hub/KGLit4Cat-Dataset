from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.extraction.extraction_context import ExtractionContext


class ChunkingRequiredError(Exception):
    pass


class ExtractionResultNotFoundError(Exception):
    pass


class ExtractionValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("Projected extraction result did not satisfy the selected profile.")
        self.errors = errors


class ExtractionRunProgress(BaseModel):
    stage: str = "pending"
    processed_chunks: int = 0
    total_chunks: int = 0
    normalized_quantities: int = 0
    normalized_qualitative_attributes: int = 0
    interim_context: ExtractionContext | None = None
    warnings: list[str] = Field(default_factory=list)


class ExtractionRunResult(BaseModel):
    document: dict[str, Any]
    extraction_context: ExtractionContext
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
