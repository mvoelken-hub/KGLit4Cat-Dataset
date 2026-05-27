from typing import Any

from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus
from app.domain.extraction import (
    ExtractionContext,
    ExtractionRunProgress,
    ExtractionRunResult,
)


class ExtractionRunRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    profile_identifier: str = Field(
        ...,
        description="Identifier of the registered extraction profile.",
    )
    qualitative_vocab_identifiers: list[str] | None = Field(
        default=None,
        description="Vocabulary identifiers to use for qualitative attribute normalization.",
    )
    resume: bool = Field(
        default=False,
        description="Resume from persisted extraction context state instead of clearing previous partial results.",
    )


class ExtractionRunResponse(BaseModel):
    status: TaskStatus
    result: ExtractionRunResult | None = None
    progress: ExtractionRunProgress | None = None


class ExtractionProgressResponse(BaseModel):
    status: TaskStatus
    progress: ExtractionRunProgress | None = None


class ExtractionResultResponse(BaseModel):
    document: dict[str, Any]
    extraction_context: ExtractionContext
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)


def _extraction_run_response(
    *,
    status: TaskStatus,
    result: ExtractionRunResult | None = None,
    progress: ExtractionRunProgress | None = None,
) -> ExtractionRunResponse:
    return ExtractionRunResponse(status=status, result=result, progress=progress)


def _extraction_result_response(result: ExtractionRunResult) -> ExtractionResultResponse:
    return ExtractionResultResponse.model_validate(result.model_dump(mode="json"))
