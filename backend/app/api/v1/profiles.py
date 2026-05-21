from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile, status
from pydantic import HttpUrl

from app.api.v1.schemas import (
    JsonLdExportResponse,
    ProfileDocumentRequest,
    ProfileManifestResponse,
    ProfileValidationResponse,
    _jsonld_export_response,
    _profile_manifest_response,
    _profile_validation_response,
)
from app.dependencies import get_profile_service
from app.domain.profiles import (
    InvalidProfileIdentifierError,
    ProfileAlreadyExistsError,
    ProfileCompatibilityError,
    ProfileNotFoundError,
    ProfileSourceError,
)
from app.services.profile_service import ProfileService


router = APIRouter(prefix="/profiles", tags=["Profiles"])


def _raise_profile_error(exc: Exception) -> None:
    if isinstance(exc, ProfileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(exc, ProfileAlreadyExistsError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        (
            InvalidProfileIdentifierError,
            ProfileCompatibilityError,
            ProfileSourceError,
            ValueError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise exc


@router.post(
    "",
    response_model=ProfileManifestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_profile(
    identifier: Annotated[str, Form(...)],
    target_class: Annotated[str, Form()] = "Dataset",
    schema_url: Annotated[HttpUrl | None, Form()] = None,
    schema_file: Annotated[UploadFile | None, File()] = None,
    version: Annotated[str | None, Form()] = None,
    enrichable_fields: Annotated[list[str] | None, Form()] = None,
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        manifest = await profile_service.register_profile(
            identifier=identifier,
            target_class=target_class,
            schema_url=str(schema_url) if schema_url else None,
            schema_file=schema_file,
            version=version,
            enrichable_fields=enrichable_fields,
        )
        return _profile_manifest_response(manifest)
    except Exception as exc:
        _raise_profile_error(exc)


@router.get("", response_model=list[ProfileManifestResponse])
async def list_profiles(
    profile_service: ProfileService = Depends(get_profile_service),
):
    return [
        _profile_manifest_response(profile)
        for profile in profile_service.list_profiles()
    ]


@router.get("/{identifier}/json-schema")
async def get_profile_json_schema(
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        return profile_service.load_json_schema(identifier)
    except Exception as exc:
        _raise_profile_error(exc)


@router.get("/{identifier}/jsonld-context")
async def get_profile_jsonld_context(
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        return profile_service.load_jsonld_context(identifier)
    except Exception as exc:
        _raise_profile_error(exc)


@router.post(
    "/{identifier}/validate",
    response_model=ProfileValidationResponse,
)
async def validate_profile_document(
    request: ProfileDocumentRequest,
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        result = profile_service.validate_document(
            identifier=identifier,
            document=request.document,
        )
        return _profile_validation_response(result)
    except Exception as exc:
        _raise_profile_error(exc)


@router.post(
    "/{identifier}/jsonld",
    response_model=JsonLdExportResponse,
)
async def export_profile_document_jsonld(
    request: ProfileDocumentRequest,
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        result = profile_service.export_jsonld(
            identifier=identifier,
            document=request.document,
        )
        return _jsonld_export_response(result)
    except Exception as exc:
        _raise_profile_error(exc)


@router.get("/{identifier}", response_model=ProfileManifestResponse)
async def get_profile(
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        return _profile_manifest_response(profile_service.get_profile(identifier))
    except Exception as exc:
        _raise_profile_error(exc)


@router.delete("/{identifier}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    identifier: str = Path(..., description="Profile identifier."),
    profile_service: ProfileService = Depends(get_profile_service),
):
    try:
        profile_service.delete_profile(identifier)
    except Exception as exc:
        _raise_profile_error(exc)
