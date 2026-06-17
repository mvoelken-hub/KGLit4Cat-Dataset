from pydantic import BaseModel, Field, model_validator

from app.domain.datasources import DataPackage
from app.domain.datasources.chunking import ChunkPostProcessingMetadata
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
    files: list[FileEntryResponse] = Field(..., min_length=1)

def _data_package_response(data_package: DataPackage) -> DataPackageResponse:
    return DataPackageResponse(**data_package.dump_without_raw_content())

class ChunkingRequest(BaseModel):
    id: str = Field(..., description="ID of the data package")
    buffer_window_size: int = Field(1, ge=0, description="Number of lines to include as buffer before and after each chunk")
    semantic_chunking_threshold: float = Field(95.0, ge=0.0, le=100.0, description="Threshold for semantic chunking quality (0-100)")
    replace_existing_chunks: bool = Field(False, description="Replace previously persisted chunks with a new chunking run")
    protected_line_indices: dict[str, list[int]] = Field(default_factory=dict, description="Map of file_path -> list of 0-based line indices to always keep regardless of text quality filter")
    text_quality_config: TextQualityConfig | None = Field(None, description="Optional per-request tuning of the text-quality classifier")
    embedding_num_gpu: int | None = Field(None, ge=-1, le=999, description="Optional Ollama num_gpu override for this chunking run's embedding requests. Use -1 for auto/all GPU and 0 for CPU only.")
    chunking_strategy: str = Field("semantic", description="Chunking strategy: 'semantic' uses embedding-based breakpoints, 'fixed_tokens' splits by configured token count")
    fixed_tokens_per_chunk: int = Field(1024, ge=1, description="Target tokens per chunk when chunking_strategy is 'fixed_tokens'")
    min_tokens_per_chunk: int = Field(128, ge=1, description="Minimum token count for a chunk after post-processing; smaller chunks are merged with neighbors")
    max_tokens_per_chunk: int = Field(1024, ge=1, description="Maximum token count for a chunk after post-processing; larger chunks are split")

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

    @model_validator(mode='after')
    def _validate_token_bounds(self):
        if self.min_tokens_per_chunk > self.max_tokens_per_chunk:
            raise ValueError("min_tokens_per_chunk must be less than or equal to max_tokens_per_chunk.")
        return self

class ChunkResponse(BaseModel):
    content: str
    data_package_id: str
    file_path: str
    start_idx: int
    end_idx: int
    filtered_line_indices: list[int] = []
    summary: str | None = None
    post_processing: ChunkPostProcessingMetadata = Field(default_factory=ChunkPostProcessingMetadata)

class ChunkRequestResponse(BaseModel):
    chunks: list[list[ChunkResponse]]
    status: TaskStatus

