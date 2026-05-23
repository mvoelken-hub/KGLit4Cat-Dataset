from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas import (
    InitialContextRequest,
    InitialDraftRequest,
    PatchDraftRequest,
    PatchDraftResponse,
    ProtectedFieldsRequest,
    PatchProgressResponse,
    PatchArtifactsResponse,
    PatchReviewState,
    PatchReviewResolutionRequest,
    PatchReviewResolutionResponse,
    SaveDraftRequest,
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


@router.get("/initial-context/{data_package_id}")
async def get_existing_initial_context(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> InitialContext | None:
    return await extraction_service.get_existing_initial_context(
        data_package_id=data_package_id,
    )


@router.put("/initial-context/{data_package_id}", response_model=InitialContext)
async def save_initial_context(
    data_package_id: str,
    request: InitialContext,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        await extraction_service.save_initial_context(
            data_package_id=data_package_id,
            initial_context=request,
        )
        return _initial_context_response(request)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/initial-draft/{data_package_id}")
async def get_existing_initial_draft(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> dict[str, Any] | None:
    return await extraction_service.get_existing_initial_draft(
        data_package_id=data_package_id,
    )


@router.get("/initial-draft/{data_package_id}/protected-fields")
async def get_protected_fields(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> list[str]:
    return await extraction_service.get_protected_fields(data_package_id=data_package_id)


@router.post("/initial-draft/{data_package_id}/protected-fields")
async def set_protected_fields(
    data_package_id: str,
    request: ProtectedFieldsRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> list[str]:
    await extraction_service.set_protected_fields(
        data_package_id=data_package_id,
        protected_fields=request.fields,
    )
    return request.fields


@router.get("/patch-draft/{data_package_id}/progress")
async def get_patch_progress(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> PatchProgressResponse:
    status, progress = await extraction_service.get_patch_progress(data_package_id=data_package_id)
    return PatchProgressResponse(status=status, progress=progress)


@router.get("/{data_package_id}/token-usage")
async def get_token_usage(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> dict[str, Any]:
    return await extraction_service.get_token_usage(data_package_id=data_package_id)


@router.get("/patch-draft/{data_package_id}/artifacts")
async def get_patch_artifacts(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> PatchArtifactsResponse:
    artifacts = await extraction_service.get_patch_artifacts(data_package_id=data_package_id)
    return PatchArtifactsResponse(**artifacts)


@router.get("/patch-draft/{data_package_id}/patches")
async def get_patch_files(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> list[dict[str, Any]]:
    return await extraction_service.get_patch_files(data_package_id=data_package_id)


@router.get("/patch-draft/{data_package_id}/quality-reports")
async def get_patch_quality_reports(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> list[dict[str, Any]]:
    return await extraction_service.get_patch_quality_reports(
        data_package_id=data_package_id,
    )


@router.get("/patch-draft/{data_package_id}/unmapped-facts")
async def get_unmapped_facts(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> list[dict[str, Any]]:
    return await extraction_service.get_unmapped_facts(data_package_id=data_package_id)


@router.get("/patch-draft/{data_package_id}/review-state")
async def get_patch_review_state(
    data_package_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> PatchReviewState:
    state = await extraction_service.get_patch_review_state(
        data_package_id=data_package_id,
    )
    return PatchReviewState(**state)


@router.put("/patch-draft/{data_package_id}/review-state")
async def save_patch_review_state(
    data_package_id: str,
    request: PatchReviewState,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> PatchReviewState:
    state = await extraction_service.save_patch_review_state(
        data_package_id=data_package_id,
        review_state=request.model_dump(mode="json"),
    )
    return PatchReviewState(**state)


@router.post("/patch-draft/{data_package_id}/resolve-review")
async def resolve_patch_review(
    data_package_id: str,
    request: PatchReviewResolutionRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> PatchReviewResolutionResponse: # type: ignore
    try:
        result = await extraction_service.resolve_patch_review_items(
            data_package_id=data_package_id,
            profile_identifier=request.profile_identifier,
            review_items=[item.model_dump(mode="json") for item in request.review_items],
        )
        return PatchReviewResolutionResponse(**result)
    except Exception as exc:
        _raise_extraction_error(exc)


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


@router.put("/initial-draft/{data_package_id}", response_model=dict[str, Any])
async def save_initial_draft(
    data_package_id: str,
    request: SaveDraftRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        await extraction_service.save_initial_draft(
            data_package_id=data_package_id,
            draft=request.draft,
        )
        return request.draft
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/patch-draft", response_model=PatchDraftResponse)
async def patch_initial_draft(
    request: PatchDraftRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service),
):
    try:
        draft, task_status = await extraction_service.patch_initial_draft(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
            num_chunks_per_turn=request.num_chunks_per_turn,
            auto_resolve=request.auto_resolve,
        )
        return _patch_draft_response(draft, task_status)
    except Exception as exc:
        _raise_extraction_error(exc)
