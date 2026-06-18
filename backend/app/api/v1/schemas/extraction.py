from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus
from app.domain.extraction import (
    ChunkRepairMode,
    CompleteWorkflowProgress,
    EvidenceCriticGranularity,
    ExtractionRunProgress,
    ExtractionRunResult,
    ExtractionVocabQueryConfig,
)
from app.api.v1.schemas.datasources import DataPackageResponse


class ExtractionRunRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    profile_identifier: str | None = Field(
        default=None,
        description=(
            "Identifier of the registered extraction profile. Required for profile, "
            "grounding, and complete runs; omitted for profile-agnostic context runs."
        ),
    )
    qualitative_vocab_identifiers: list[str] | None = Field(
        default=None,
        description="Vocabulary identifiers to use for qualitative attribute normalization.",
    )
    resume: bool = Field(
        default=False,
        description="Resume from persisted extraction context state instead of clearing previous partial results.",
    )
    force_profile_rebuild: bool = Field(
        default=False,
        description=(
            "When resuming to a profile stage, clear generated draft/projection artifacts "
            "and rebuild the machine profile draft from persisted evidence."
        ),
    )
    target_stage: Literal["context", "profile", "grounding", "complete"] = Field(
        default="complete",
        description="Workflow stage to run up to.",
    )
    chunking_strategy: str = Field(
        default="semantic",
        description="Chunking strategy branch to read: 'semantic' or 'fixed_tokens'.",
    )
    chunk_repair_mode: ChunkRepairMode = Field(
        default="deferred",
        description="How to handle repairable chunk structured-output failures.",
    )
    evidence_critic_granularity: EvidenceCriticGranularity = Field(
        default="per_chunk",
        description="How to batch independent evidence critic assessments.",
    )


class ExtractionRunResponse(BaseModel):
    status: TaskStatus
    result: ExtractionRunResult | None = None
    progress: ExtractionRunProgress | None = None


class ExtractionProgressResponse(BaseModel):
    status: TaskStatus
    progress: ExtractionRunProgress | None = None


class InitialContextRunRequest(BaseModel):
    force_rerun: bool = Field(
        default=False,
        description="Clear previous extraction outputs and regenerate file summaries, ranking, and run overview.",
    )


class CompleteWorkflowProgressResponse(BaseModel):
    status: TaskStatus
    progress: CompleteWorkflowProgress | None = None


class CompleteWorkflowRunResponse(BaseModel):
    status: TaskStatus
    data_package: DataPackageResponse
    progress: CompleteWorkflowProgress | None = None
    progress_url: str
    result_url: str


class VocabQueryConfigUpdateRequest(ExtractionVocabQueryConfig):
    pass


class CuratedDocumentUpdateRequest(BaseModel):
    profile_identifier: str = Field(..., description="Identifier of the selected extraction profile.")
    document: dict[str, Any] = Field(..., description="Edited curated profile document.")


class CurationFieldActionRequest(BaseModel):
    action: Literal["select_vocab_term", "mark_unresolved"]
    json_path: str = Field(..., description="RFC 6901 JSON Pointer to the curated field.")
    selected_uri: str | None = Field(default=None, description="Selected vocabulary term URI.")
    selected_title: str | None = Field(default=None, description="Selected vocabulary term label.")
    vocabulary_identifier: str | None = Field(default=None, description="Source vocabulary identifier.")


class ExtractionResultResponse(ExtractionRunResult):
    pass


def _extraction_run_response(
    *,
    status: TaskStatus,
    result: ExtractionRunResult | None = None,
    progress: ExtractionRunProgress | None = None,
) -> ExtractionRunResponse:
    return ExtractionRunResponse(status=status, result=result, progress=progress)


def _extraction_result_response(result: ExtractionRunResult) -> ExtractionResultResponse:
    return ExtractionResultResponse.model_validate(result.model_dump(mode="json"))
