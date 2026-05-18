from typing import Any

from pydantic import BaseModel, Field

from app.domain.extraction import (
    InitialContext,
    JsonLdExportResult,
    ProfileManifest,
    ProfileValidationIssue,
    ProfileValidationResult,
)


class ProfileManifestResponse(BaseModel):
    identifier: str
    source: str
    source_type: str
    schema_url: str | None = None
    schema_file_name: str | None = None
    target_class: str
    checksum: str
    version: str | None = None
    enrichable_fields: list[str] = Field(default_factory=list)


class ProfileDocumentRequest(BaseModel):
    document: dict[str, Any]


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


class ProfileValidationIssueResponse(BaseModel):
    path: str
    message: str
    schema_path: str


class ProfileValidationResponse(BaseModel):
    valid: bool
    errors: list[ProfileValidationIssueResponse] = Field(default_factory=list)


class JsonLdExportResponse(BaseModel):
    document: dict[str, Any]
    triple_count: int


def _initial_context_response(initial_context: InitialContext) -> InitialContext:
    return initial_context


def _profile_manifest_response(manifest: ProfileManifest) -> ProfileManifestResponse:
    return ProfileManifestResponse.model_validate(manifest.model_dump(mode="json"))


def _profile_validation_response(result: ProfileValidationResult) -> ProfileValidationResponse:
    return ProfileValidationResponse(
        valid=result.valid,
        errors=[_profile_validation_issue_response(issue) for issue in result.errors],
    )


def _profile_validation_issue_response(
    issue: ProfileValidationIssue,
) -> ProfileValidationIssueResponse:
    return ProfileValidationIssueResponse.model_validate(issue.model_dump(mode="json"))


def _jsonld_export_response(result: JsonLdExportResult) -> JsonLdExportResponse:
    return JsonLdExportResponse.model_validate(result.model_dump(mode="json"))
