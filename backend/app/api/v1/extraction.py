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
from app.dependencies import get_extraction_service
from app.domain.extraction import (
    InvalidProfileIdentifierError,
    ProfileAlreadyExistsError,
    ProfileCompatibilityError,
    ProfileNotFoundError,
    ProfileSourceError,
)
from app.services.extraction_service import ExtractionService


router = APIRouter(prefix="/extraction", tags=["Extraction"])


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
    "/profiles",
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
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        manifest = await extraction_service.register_profile(
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


@router.get("/profiles", response_model=list[ProfileManifestResponse])
async def list_profiles(
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    return [
        _profile_manifest_response(profile)
        for profile in extraction_service.list_profiles()
    ]


@router.get("/profiles/{identifier}/json-schema")
async def get_profile_json_schema(
    identifier: str = Path(..., description="Profile identifier."),
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        return extraction_service.get_json_schema(identifier)
    except Exception as exc:
        _raise_profile_error(exc)


@router.get("/profiles/{identifier}/jsonld-context")
async def get_profile_jsonld_context(
    identifier: str = Path(..., description="Profile identifier."),
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        return extraction_service.get_jsonld_context(identifier)
    except Exception as exc:
        _raise_profile_error(exc)


@router.post(
    "/profiles/{identifier}/validate",
    response_model=ProfileValidationResponse,
)
async def validate_profile_document(
    request: ProfileDocumentRequest,
    identifier: str = Path(..., description="Profile identifier."),
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        result = extraction_service.validate_document(
            identifier=identifier,
            document=request.document,
        )
        return _profile_validation_response(result)
    except Exception as exc:
        _raise_profile_error(exc)


@router.post(
    "/profiles/{identifier}/jsonld",
    response_model=JsonLdExportResponse,
)
async def export_profile_document_jsonld(
    request: ProfileDocumentRequest,
    identifier: str = Path(..., description="Profile identifier."),
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        result = extraction_service.export_jsonld(
            identifier=identifier,
            document=request.document,
        )
        return _jsonld_export_response(result)
    except Exception as exc:
        _raise_profile_error(exc)


@router.get("/profiles/{identifier}", response_model=ProfileManifestResponse)
async def get_profile(
    identifier: str = Path(..., description="Profile identifier."),
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        return _profile_manifest_response(extraction_service.get_profile(identifier))
    except Exception as exc:
        _raise_profile_error(exc)
