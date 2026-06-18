from io import BytesIO
from pathlib import Path
from typing import Protocol

from app.domain.datasources import ContentChunk, DataPackage


class DataSourceBlobRepository(Protocol):
    def save_data_package(self, data: BytesIO, id: str, file_name: str) -> Path:
        ...

    def load_data_package(self, id: str) -> DataPackage:
        ...

    def delete_data_package(self, id: str) -> None:
        ...

    def list_data_packages(self) -> list[DataPackage]:
        ...

    def save_content_chunks(self, chunks: list[ContentChunk], chunking_strategy: str = "semantic") -> None:
        ...

    def delete_content_chunks(
        self,
        data_package_id: str,
        chunking_strategy: str | None = None,
    ) -> None:
        ...

    def load_content_chunks_by_file_path(
        self,
        data_package_id: str,
        file_path: str,
        chunking_strategy: str = "semantic",
    ) -> list[ContentChunk]:
        ...
