from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas import (
    InitialContextRequest,
    InitialDraftRequest,
    PatchDraftRequest,
    _initial_context_response,
    _initial_draft_response,
    _patch_draft_response,
)
from app.dependencies import get_extraction_service
from app.domain.datasources import (
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
)
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContextRequiredError,
    InitialContext,
    PatchDraftPrerequisiteError,
)
from app.domain.profiles import (
    InvalidProfileIdentifierError,
    ProfileAlreadyExistsError,
    ProfileCompatibilityError,
    ProfileNotFoundError,
    ProfileSourceError,
)
from app.services.extraction_service import ExtractionService
from pydantic_ai.exceptions import AgentRunError


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


def _raise_extraction_error(exc: Exception) -> None:
    if isinstance(exc, (DataPackageIdNotFoundError, DataPackageZipNotFoundError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(exc, InitialContextRequiredError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if isinstance(exc, (PatchDraftPrerequisiteError, ChunkingRequiredError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if isinstance(exc, AgentRunError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    _raise_profile_error(exc)


@router.post("/initial-context", response_model=InitialContext)
async def extract_initial_context(
    request: InitialContextRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        result = await extraction_service.extract_initial_context(
            data_package_id=request.data_package_id,
            max_files_to_read=request.max_files_to_read,
            max_chars_per_file=request.max_chars_per_file,
        )
        return _initial_context_response(result)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/initial-draft", response_model=dict[str, Any])
async def extract_initial_draft(
    request: InitialDraftRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        result = await extraction_service.extract_initial_draft(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
        )
        return _initial_draft_response(result)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/patch-draft", response_model=dict[str, Any])
async def patch_initial_draft(
    request: PatchDraftRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        result = await extraction_service.patch_initial_draft(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
            num_chunks_per_turn=request.num_chunks_per_turn,
        )
        return _patch_draft_response(result)
    except Exception as exc:
        _raise_extraction_error(exc)
