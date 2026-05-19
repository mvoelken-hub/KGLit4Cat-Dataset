from typing import Any

from pydantic import BaseModel, Field

from app.domain.profiles import (
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
