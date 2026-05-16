from pydantic import BaseModel, Field

from app.domain.datasources import DataPackage

from app.core.task_registry import TaskStatus

class FileEntryContentResponse(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    content: str

class FileEntryResponse(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    byte_size: int

class DataPackageResponse(BaseModel):
    file_name: str
    id: str
    files: list[FileEntryResponse] = Field(..., min_length=1)

def _data_package_response(data_package: DataPackage) -> DataPackageResponse:
    return DataPackageResponse(**data_package.dump_without_raw_content())

class ChunkingRequest(BaseModel):
    id: str = Field(..., description="ID of the data package")
    buffer_window_size: int = Field(1, ge=0, description="Number of lines to include as buffer before and after each chunk")
    embedding_batch_size: int = Field(32, ge=1, description="Number of chunks to process in each embedding batch")
    semantic_chunking_threshold: float = Field(95.0, ge=0.0, le=100.0, description="Threshold for semantic chunking quality (0-100)")

class ChunkResponse(BaseModel):
    content: str
    data_package_id: str
    file_path: str
    start_idx: int
    end_idx: int
    summary: str | None = None

class ChunkRequestResponse(BaseModel):
    chunks: list[list[ChunkResponse]]
    status: TaskStatus

