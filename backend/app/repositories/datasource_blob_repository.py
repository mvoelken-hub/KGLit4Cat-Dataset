from typing import Protocol
from io import BytesIO
from pathlib import Path

from app.domain.datasources import DataPackage, ContentChunk

class DataSourceBlobRepository(Protocol):
    def save_data_package(self, data: BytesIO, id: str, file_name: str) -> Path:
        ...

    def load_data_package(self, id: str) -> DataPackage:
        ...

    def delete_data_package(self, id: str) -> None:
        ...

    def list_data_packages(self) -> list[DataPackage]:
        ...

    def save_content_chunks(self, chunks: list[ContentChunk]) -> None:
        ...

    def delete_content_chunks(self, data_package_id: str) -> None:
        ...

    def load_content_chunks_by_file_path(self, data_package_id: str, file_path: str) -> list[ContentChunk]:
        ...
