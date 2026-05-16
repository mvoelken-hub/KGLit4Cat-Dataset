from typing import Protocol
from io import BytesIO
from pathlib import Path

from app.models.datasources import DataPackage, DataPackageIdNotFoundError

class DataSourceBlobRepository(Protocol):
    def save_data_package(self, data: BytesIO, id: str, file_name: str) -> Path:
        ...

    def load_data_package(self, id: str) -> DataPackage:
        ...

    def delete_data_package(self, id: str) -> None:
        ...

    def list_data_package_ids(self) -> list[str]:
        ...

