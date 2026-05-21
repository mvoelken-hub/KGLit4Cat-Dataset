from pydantic import BaseModel, Field, model_validator

from app.domain.datasources import DataPackage
from app.domain.datasources.text_quality import TextQualityConfig

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
    files: list[FileEntryResponse] = Field(..., min_length=1) # TODO: remove this field for listing all data packages, only include it for retrieving a single data package

def _data_package_response(data_package: DataPackage) -> DataPackageResponse:
    return DataPackageResponse(**data_package.dump_without_raw_content())

class ChunkingRequest(BaseModel):
    id: str = Field(..., description="ID of the data package")
    buffer_window_size: int = Field(1, ge=0, description="Number of lines to include as buffer before and after each chunk")
    semantic_chunking_threshold: float = Field(95.0, ge=0.0, le=100.0, description="Threshold for semantic chunking quality (0-100)")
    replace_existing_chunks: bool = Field(False, description="Replace previously persisted chunks with a new chunking run")
    protected_line_indices: dict[str, list[int]] = Field(default_factory=dict, description="Map of file_path -> list of 0-based line indices to always keep regardless of text quality filter")
    text_quality_config: TextQualityConfig | None = Field(None, description="Optional per-request tuning of the text-quality classifier")

    @model_validator(mode='before')
    @classmethod
    def _parse_json_strings(cls, data: dict) -> dict:
        """Allow these nested fields to arrive as JSON strings from query params."""
        import json
        for key in ('protected_line_indices', 'text_quality_config'):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    data[key] = {} if key == 'protected_line_indices' else None
        return data

class ChunkResponse(BaseModel):
    content: str
    data_package_id: str
    file_path: str
    start_idx: int
    end_idx: int
    filtered_line_indices: list[int] = []
    summary: str | None = None

class ChunkRequestResponse(BaseModel):
    chunks: list[list[ChunkResponse]]
    status: TaskStatus

