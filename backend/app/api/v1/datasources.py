from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Query


from app.dependencies import get_datasource_service
from app.domain.datasources import (
    DataPackage,
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
    InvalidDataPackageZipFileError,
    MultipleDataPackageZipFilesError,
)
from app.api.v1.schemas import (
    DataPackageResponse,
    FileEntryResponse,
    FileEntryContentResponse,
    _data_package_response,
    ChunkingRequest,
    ChunkResponse,
    ChunkRequestResponse
)
from app.services.datasource_service import DataSourceService

router = APIRouter(prefix="/datasources", tags=["Datasources"])



def _raise_datasource_error(exc: Exception) -> None:
    if isinstance(exc, (DataPackageIdNotFoundError, DataPackageZipNotFoundError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(exc, (InvalidDataPackageZipFileError, MultipleDataPackageZipFilesError)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise exc


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DataPackageResponse)
async def upload_data_package(
    file: UploadFile = File(...),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    try:
        data_package = datasource_service.save_data_package(
            BytesIO(await file.read()),
            file.filename,
        )
        return _data_package_response(data_package)
    
    except Exception as exc:
        _raise_datasource_error(exc)


@router.get("/{id}")
async def get_data_package(
    id: str,
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        data_package = datasource_service.get_data_package(id)
        return _data_package_response(data_package)
    
    except Exception as exc:
        _raise_datasource_error(exc)

    
@router.get("", response_model=list[DataPackageResponse])
async def list_data_packages(
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    return [_data_package_response(dp) for dp in datasource_service.list_data_packages()]


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_package(
    id: str,
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        datasource_service.delete_data_package(id)
    except Exception as exc:
        _raise_datasource_error(exc)

@router.get("/{id}/files/{file_path:path}", response_model=FileEntryContentResponse)
async def get_file_entry_content(
    id: str,
    file_path: str,
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        data_package = datasource_service.get_data_package(id)
        file_entry = data_package.get_file_entry(file_path)
        return FileEntryContentResponse(
            file_path=file_entry.file_path,
            file_name=file_entry.file_name,
            file_extension=file_entry.file_extension,
            content=file_entry.get_extracted_content()
        )
    except Exception as exc:
        _raise_datasource_error(exc)


@router.get("/{id}/chunks/status")
async def get_chunk_status(
    id: str,
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        chunks = datasource_service.get_completed_content_chunks_by_file(id)
        return {"has_chunks": bool(chunks), "file_count": len(chunks)}
    except Exception as exc:
        _raise_datasource_error(exc)


@router.get("/{id}/chunks", response_model=list[list[ChunkResponse]])
async def get_data_package_chunks(
    id: str,
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        chunks_by_file = datasource_service.get_content_chunks_by_file(id)
        return [[ChunkResponse(**chunk.model_dump()) for chunk in chunks] for chunks in chunks_by_file]
    except Exception as exc:
        _raise_datasource_error(exc)


@router.post("/chunk", response_model=ChunkRequestResponse)
async def chunk_file_entries_in_data_package(
    chunking_request: Annotated[ChunkingRequest, Query(..., description="Chunking parameters")],
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    try:
        chunks_by_file, status = await datasource_service.chunk_file_entries_in_data_package(
            data_package_id=chunking_request.id,
            buffer_window_size=chunking_request.buffer_window_size,
            embedding_batch_size=chunking_request.embedding_batch_size,
            semantic_chunking_threshold=chunking_request.semantic_chunking_threshold,
            replace_existing_chunks=chunking_request.replace_existing_chunks
        )
        return ChunkRequestResponse(
            chunks=[[ChunkResponse(**chunk.model_dump()) for chunk in chunks] for chunks in chunks_by_file],
            status=status
        )
    
    except Exception as exc:
        _raise_datasource_error(exc)
