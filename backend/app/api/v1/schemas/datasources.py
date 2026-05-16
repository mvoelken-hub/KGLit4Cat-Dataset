from pydantic import BaseModel, Field

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