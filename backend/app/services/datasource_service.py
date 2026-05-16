from io import BytesIO

from app.models.datasources import DataPackage, FileEntry
from app.repositories.datasource_blob_repository import DataSourceBlobRepository

class DataSourceService:
    def __init__(self, blob_repository: DataSourceBlobRepository):
        self.blob_repository = blob_repository

    def save_data_package(self, data: BytesIO, file_name: str) -> DataPackage:
        
        data_package = DataPackage.from_bytes(data, file_name)
        self.blob_repository.save_data_package(data, data_package.id, file_name)
        
        return data_package

    def get_data_package(self, id: str) -> DataPackage:
        return self.blob_repository.load_data_package(id)

    def list_data_package_ids(self) -> list[str]:
        return self.blob_repository.list_data_package_ids()
    
    def delete_data_package(self, id: str) -> None:
        self.blob_repository.delete_data_package(id)

    def get_file_entry(self, id: str, file_path: str) -> FileEntry:
        data_package = self.get_data_package(id)
        return data_package.get_file_entry(file_path)
