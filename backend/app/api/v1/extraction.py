import json
from io import BytesIO
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.task_registry import TaskStatus
from app.api.v1.schemas import (
    CompleteWorkflowProgressResponse,
    CompleteWorkflowRunResponse,
    CuratedDocumentUpdateRequest,
    CurationFieldActionRequest,
    ExtractionProgressResponse,
    ExtractionRunRequest,
    ExtractionRunResponse,
    InitialContextRunRequest,
    ProfileProjectionRunRequest,
    VocabQueryConfigUpdateRequest,
    _data_package_response,
    _extraction_result_response,
    _extraction_run_response,
)
from app.dependencies import get_datasource_service, get_workflow_service
from app.domain.datasources import (
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
    InvalidDataPackageZipFileError,
    MultipleDataPackageZipFilesError,
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
from app.services.datasource_service import DataSourceService
from app.services.workflow_service import WorkflowService


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


def _parse_qualitative_vocab_identifiers(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in stripped.split(",")]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("qualitative_vocab_identifiers must be a JSON string array or a comma-separated string.")
    identifiers = [item.strip() for item in parsed if item.strip()]
    return identifiers or None


def _raise_workflow_upload_error(exc: Exception) -> None:
    if isinstance(exc, (InvalidDataPackageZipFileError, MultipleDataPackageZipFilesError)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    _raise_extraction_error(exc)


@router.post("/stages/evidence", response_model=ExtractionRunResponse)
async def run_evidence_stage(
    request: ExtractionRunRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionRunResponse:
    try:
        if request.target_stage == "profile":
            raise ValueError(
                "Use /api/v1/extraction/stages/profile to build generated drafts from persisted evidence."
            )
        result, task_status = await workflow_service.run_extraction(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
            qualitative_vocab_identifiers=request.qualitative_vocab_identifiers,
            resume=request.resume,
            force_profile_rebuild=request.force_profile_rebuild,
            target_stage=request.target_stage,
            chunking_strategy=request.chunking_strategy,
            chat_model=request.chat_model,
            chunk_repair_mode=request.chunk_repair_mode,
            evidence_critic_granularity=request.evidence_critic_granularity,
        )
        _, progress = await workflow_service.get_extraction_progress(
            data_package_id=request.data_package_id,
            chunking_strategy=request.chunking_strategy,
            chat_model=request.chat_model,
        )
        return _extraction_run_response(
            status=task_status,
            result=result,
            progress=progress,
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/stages/profile", response_model=ExtractionRunResponse)
async def run_profile_projection_stage(
    request: ProfileProjectionRunRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionRunResponse:
    try:
        result, task_status = await workflow_service.run_profile_projection(
            data_package_id=request.data_package_id,
            profile_identifier=request.profile_identifier,
            qualitative_vocab_identifiers=request.qualitative_vocab_identifiers,
            resume=request.resume,
            force_rebuild=request.force_rebuild,
            chunking_strategy=request.chunking_strategy,
            chat_model=request.chat_model,
        )
        _, progress = await workflow_service.get_extraction_progress(
            data_package_id=request.data_package_id,
            chunking_strategy=request.chunking_strategy,
            chat_model=request.chat_model,
        )
        return _extraction_run_response(
            status=task_status,
            result=result,
            progress=progress,
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/stages/evidence/{data_package_id}/progress", response_model=ExtractionProgressResponse)
async def get_evidence_stage_progress(
    data_package_id: str,
    chunking_strategy: Literal["semantic", "fixed_tokens"] | None = None,
    chat_model: str | None = None,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    status_value, progress = await workflow_service.get_extraction_progress(
        data_package_id=data_package_id,
        chunking_strategy=chunking_strategy,
        chat_model=chat_model,
    )
    return ExtractionProgressResponse(status=status_value, progress=progress)


@router.post("/stages/orientation/{data_package_id}", response_model=ExtractionProgressResponse)
async def run_orientation_stage(
    data_package_id: str,
    request: InitialContextRunRequest | None = None,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    try:
        status_value = await workflow_service.run_initial_context(
            data_package_id=data_package_id,
            force_rerun=request.force_rerun if request else False,
        )
        _, progress = await workflow_service.get_initial_context_progress(
            data_package_id=data_package_id,
        )
        return ExtractionProgressResponse(status=status_value, progress=progress)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/stages/orientation/{data_package_id}/progress", response_model=ExtractionProgressResponse)
async def get_orientation_stage_progress(
    data_package_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    status_value, progress = await workflow_service.get_initial_context_progress(
        data_package_id=data_package_id,
    )
    return ExtractionProgressResponse(status=status_value, progress=progress)


@router.post("/workflows/{data_package_id}/pause", response_model=ExtractionProgressResponse)
async def pause_extraction(
    data_package_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    status_value, progress = await workflow_service.pause_extraction(
        data_package_id=data_package_id,
    )
    return ExtractionProgressResponse(status=status_value, progress=progress)


@router.patch("/stages/grounding/{data_package_id}/config", response_model=ExtractionProgressResponse)
async def update_vocab_query_config(
    data_package_id: str,
    request: VocabQueryConfigUpdateRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    try:
        progress = await workflow_service.update_vocab_query_config(
            data_package_id=data_package_id,
            config=request,
        )
        return ExtractionProgressResponse(status=TaskStatus.UNKNOWN, progress=progress)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.put("/stages/curation/{data_package_id}/document", response_model=ExtractionProgressResponse)
async def update_curated_document(
    data_package_id: str,
    request: CuratedDocumentUpdateRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    try:
        progress = await workflow_service.update_curated_document(
            data_package_id=data_package_id,
            profile_identifier=request.profile_identifier,
            document=request.document,
        )
        return ExtractionProgressResponse(status=TaskStatus.UNKNOWN, progress=progress)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/stages/curation/{data_package_id}/field", response_model=ExtractionProgressResponse)
async def apply_curation_field_action(
    data_package_id: str,
    request: CurationFieldActionRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ExtractionProgressResponse:
    try:
        progress = await workflow_service.apply_curation_field_action(
            data_package_id=data_package_id,
            action=request.action,
            json_path=request.json_path,
            selected_uri=request.selected_uri,
            selected_title=request.selected_title,
            vocabulary_identifier=request.vocabulary_identifier,
        )
        return ExtractionProgressResponse(status=TaskStatus.UNKNOWN, progress=progress)
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post(
    "/workflows",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CompleteWorkflowRunResponse,
)
async def run_complete_workflow(
    file: UploadFile = File(...),
    profile_identifier: str = Form(...),
    qualitative_vocab_identifiers: str | None = Form(
        default=None,
        description="Optional JSON string array or comma-separated vocabulary identifiers.",
    ),
    buffer_window_size: int = Form(
        default=1,
        ge=0,
        description="Number of lines to include as buffer before and after each chunk.",
    ),
    semantic_chunking_threshold: float = Form(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Threshold for semantic chunking quality (0-100).",
    ),
    chunking_strategy: str = Form(
        default="semantic",
        description="Chunking strategy: 'semantic' uses embedding-based breakpoints, 'fixed_tokens' splits by configured token count.",
    ),
    fixed_tokens_per_chunk: int = Form(
        default=1024,
        ge=1,
        description="Target tokens per chunk when chunking_strategy is 'fixed_tokens'.",
    ),
    min_tokens_per_chunk: int = Form(
        default=128,
        ge=1,
        description="Minimum token count for a chunk after post-processing; smaller chunks are merged with neighbors.",
    ),
    max_tokens_per_chunk: int = Form(
        default=1024,
        ge=1,
        description="Maximum token count for a chunk after post-processing; larger chunks are split.",
    ),
    replace_existing_chunks: bool = Form(
        default=False,
        description="Replace previously persisted chunks for a package with the same deterministic id.",
    ),
    resume: bool = Form(
        default=False,
        description="Resume a persisted extraction run instead of clearing previous partial results.",
    ),
    force_rerun: bool = Form(
        default=False,
        description="Clear persisted extraction artifacts and schedule a fresh complete workflow for the same deterministic package id.",
    ),
    datasource_service: DataSourceService = Depends(get_datasource_service),
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> CompleteWorkflowRunResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )
    if min_tokens_per_chunk > max_tokens_per_chunk:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_tokens_per_chunk must be less than or equal to max_tokens_per_chunk.",
        )

    try:
        vocab_identifiers = _parse_qualitative_vocab_identifiers(
            qualitative_vocab_identifiers
        )
        data_package = datasource_service.save_data_package(
            BytesIO(await file.read()),
            file.filename,
        )
        workflow_status = await workflow_service.run_complete_workflow(
            data_package_id=data_package.id,
            profile_identifier=profile_identifier,
            qualitative_vocab_identifiers=vocab_identifiers,
            buffer_window_size=buffer_window_size,
            semantic_chunking_threshold=semantic_chunking_threshold,
            chunking_strategy=chunking_strategy,
            fixed_tokens_per_chunk=fixed_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
            max_tokens_per_chunk=max_tokens_per_chunk,
            replace_existing_chunks=replace_existing_chunks,
            resume=resume,
            force_rerun=force_rerun,
        )
        _, progress = await workflow_service.get_complete_workflow_progress(
            data_package_id=data_package.id,
        )
        return CompleteWorkflowRunResponse(
            status=workflow_status,
            data_package=_data_package_response(data_package),
            progress=progress,
            progress_url=f"/api/v1/extraction/workflows/{data_package.id}/progress",
            result_url=f"/api/v1/extraction/results/{data_package.id}",
        )
    except Exception as exc:
        _raise_workflow_upload_error(exc)


@router.get(
    "/workflows/{data_package_id}/progress",
    response_model=CompleteWorkflowProgressResponse,
)
async def get_complete_workflow_progress(
    data_package_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> CompleteWorkflowProgressResponse:
    status_value, progress = await workflow_service.get_complete_workflow_progress(
        data_package_id=data_package_id,
    )
    return CompleteWorkflowProgressResponse(status=status_value, progress=progress)


@router.post("/stages/grounding/{data_package_id}/run")
async def run_grounding_stage(
    data_package_id: str,
    chunking_strategy: Literal["semantic", "fixed_tokens"] | None = None,
    chat_model: str | None = None,
    workflow_service: WorkflowService = Depends(get_workflow_service),
):
    """Run the vocabulary grounding stage as a standalone step.

    Grounds the persisted profile draft without rebuilding the projection or
    requirement-enrichment stages.
    """
    try:
        return _extraction_result_response(
            await workflow_service.run_grounding_stage(
                data_package_id=data_package_id,
                chunking_strategy=chunking_strategy,
                chat_model=chat_model,
            )
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/stages/grounding/{data_package_id}/rerun")
async def rerun_vocab_queries(
    data_package_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return _extraction_result_response(
            await workflow_service.rerun_vocab_queries(data_package_id=data_package_id)
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.post("/stages/grounding/{data_package_id}/queries/{query_id}/rerun")
async def rerun_vocab_query(
    data_package_id: str,
    query_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return _extraction_result_response(
            await workflow_service.rerun_vocab_queries(
                data_package_id=data_package_id,
                query_id=query_id,
            )
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/results/{data_package_id}")
async def get_extraction_result(
    data_package_id: str,
    chunking_strategy: Literal["semantic", "fixed_tokens"] | None = None,
    chat_model: str | None = None,
    workflow_service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return _extraction_result_response(
            await workflow_service.get_extraction_result(
                data_package_id=data_package_id,
                chunking_strategy=chunking_strategy,
                chat_model=chat_model,
            )
        )
    except Exception as exc:
        _raise_extraction_error(exc)


@router.get("/workflows/{data_package_id}/token-usage")
async def get_token_usage(
    data_package_id: str,
    chunking_strategy: Literal["semantic", "fixed_tokens"] | None = None,
    chat_model: str | None = None,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> dict[str, Any]:
    return await workflow_service.get_token_usage(data_package_id=data_package_id, chunking_strategy=chunking_strategy, chat_model=chat_model)

