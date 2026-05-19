from typing import Any

from pydantic import BaseModel, Field

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
    num_chunks_per_turn: int | None = Field(
        None,
        ge=1,
        le=50,
        description="Optional number of chunks to include in each patch agent call.",
    )


def _initial_context_response(initial_context: InitialContext) -> InitialContext:
    return initial_context


def _initial_draft_response(initial_draft: dict[str, Any]) -> dict[str, Any]:
    return initial_draft


def _patch_draft_response(draft: dict[str, Any]) -> dict[str, Any]:
    return draft
