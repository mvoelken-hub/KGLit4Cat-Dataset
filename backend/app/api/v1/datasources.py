from io import BytesIO
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import get_datasource_service
from app.models.datasources import (
    DataPackage,
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
    InvalidDataPackageZipFileError,
    MultipleDataPackageZipFilesError,
)
from app.api.v1.schemas import (
    DataPackageResponse
)
from app.services.datasource_service import DataSourceService

router = APIRouter(prefix="/datasources", tags=["Datasources"])


def _data_package_response(data_package: DataPackage) -> DataPackageResponse:
    return DataPackageResponse(**data_package.dump_without_raw_content())


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