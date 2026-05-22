from typing import Any

from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus
from app.domain.extraction import (
    InitialContext,
)


class InitialContextRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    max_files_to_read: int = Field(
        12,
        ge=1,
        le=100,
        description="Maximum number of files the agent should inspect.",
    )
    max_chars_per_file: int = Field(
        3000,
        ge=1,
        le=50000,
        description="Maximum characters the agent should read from any one file.",
    )


class InitialDraftRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    profile_identifier: str = Field(
        ...,
        description="Identifier of the registered extraction profile.",
    )


class PatchDraftRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    profile_identifier: str = Field(
        ...,
        description="Identifier of the registered extraction profile.",
    )
    num_chunks_per_turn: int= Field(
        1,
        ge=1,
        description="Number of chunks to include in each patch agent call.",
    )
    auto_resolve: bool = Field(
        default=False,
        description="Automatically send unresolved review items to the resolve agent as patch artifacts are produced.",
    )


class PatchDraftResponse(BaseModel):
    draft: dict[str, Any]
    status: TaskStatus


class SaveDraftRequest(BaseModel):
    data_package_id: str = Field(..., description="ID of the uploaded data package.")
    draft: dict[str, Any] = Field(..., description="Updated draft object to persist.")


class ProtectedFieldsRequest(BaseModel):
    fields: list[str] = Field(default_factory=list, description="Top-level field paths to protect from patching.")


class PatchProgressResponse(BaseModel):
    status: TaskStatus
    progress: dict[str, Any] | None = None


class PatchArtifactsResponse(BaseModel):
    patches: list[dict[str, Any]] = Field(default_factory=list)
    quality_reports: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_facts: list[dict[str, Any]] = Field(default_factory=list)


class PatchReviewState(BaseModel):
    resolved_item_ids: list[str] = Field(default_factory=list)
    unmapped_assignments: dict[str, str] = Field(default_factory=dict)
    resolution_notes: dict[str, str] = Field(default_factory=dict)
    resolved_at: dict[str, str] = Field(default_factory=dict)


class PatchReviewItemRequest(BaseModel):
    id: str
    kind: str
    path: str
    detail: str | None = None
    issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    patch: dict[str, Any] | None = None
    fact: str | None = None
    reason: str | None = None
    confidence: float | None = None
    file_name: str | None = None


class PatchReviewResolutionRequest(BaseModel):
    profile_identifier: str = Field(..., description="Identifier of the registered extraction profile.")
    review_items: list[PatchReviewItemRequest] = Field(default_factory=list)


class PatchReviewDecisionResponse(BaseModel):
    id: str
    outcome: str
    note: str
    target_path: str | None = None


class PatchReviewResolutionResponse(BaseModel):
    draft: dict[str, Any]
    review_state: PatchReviewState
    resolved_count: int = 0
    unresolved_item_ids: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    resolution_decisions: list[PatchReviewDecisionResponse] = Field(default_factory=list)
    resolution_log: list[str] = Field(default_factory=list)


def _initial_context_response(initial_context: InitialContext) -> InitialContext:
    return initial_context


def _initial_draft_response(initial_draft: dict[str, Any]) -> dict[str, Any]:
    return initial_draft


def _patch_draft_response(
    draft: dict[str, Any],
    status: TaskStatus,
) -> PatchDraftResponse:
    return PatchDraftResponse(draft=draft, status=status)
