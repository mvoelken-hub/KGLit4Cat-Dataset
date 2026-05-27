from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas import (
    ExtractionProgressResponse,
    ExtractionRunRequest,
    ExtractionRunResponse,
    _extraction_result_response,
    _extraction_run_response,
)
from app.dependencies import get_extraction_service
from app.domain.datasources import (
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
)
from app.domain.extraction import (
    ChunkingRequiredError,
    ExtractionResultNotFoundError,
    ExtractionValidationError,
)
from app.domain.profiles import (
    InvalidProfileIdentifierError,
    ProfileCompatibilityError,
    ProfileNotFoundError,
    ProfileSourceError,
)
from app.ollama.errors import CompletionError
from app.services.extraction_service import ExtractionService


router = APIRouter(prefix="/extraction", tags=["Extraction"])


def _raise_extraction_error(exc: Exception) -> None:
    if isinstance(exc, (DataPackageIdNotFoundError, DataPackageZipNotFoundError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ProfileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ExtractionResultNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ChunkingRequiredError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ExtractionValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors,
        ) from exc
    if isinstance(exc, CompletionError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
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


@router.post("/run", response_model=ExtractionRunResponse)
async def run_extraction(
    request: ExtractionRunRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionRunResponse:
    try:
        result, task_status = await extraction_service.run_extraction(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
            qualitative_vocab_identifiers=request.qualitative_vocab_identifiers,
        )
        _, progress = await extraction_service.get_extraction_progress(
            data_package_id=request.data_package_id,
        )
        return _extraction_run_response(
            status=task_status,
            result=result,
            progress=progress,
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/run/{data_package_id}/progress", response_model=ExtractionProgressResponse)
async def get_extraction_progress(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionProgressResponse:
    status_value, progress = await extraction_service.get_extraction_progress(
        data_package_id=data_package_id,
    )
    return ExtractionProgressResponse(status=status_value, progress=progress)


@router.get("/result/{data_package_id}")
async def get_extraction_result(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        return _extraction_result_response(
            await extraction_service.get_extraction_result(
                data_package_id=data_package_id,
            )
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/{data_package_id}/token-usage")
async def get_token_usage(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> dict[str, Any]:
    return await extraction_service.get_token_usage(data_package_id=data_package_id)

