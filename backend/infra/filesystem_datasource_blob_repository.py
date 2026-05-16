from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

from app.domain.datasources import (
    DataPackage,
    FileEntry,
    InvalidDataPackageFileNameError,
    DataPackageZipNotFoundError,
    DataPackageIdNotFoundError,
    MultipleDataPackageZipFilesError,
    ContentChunk
)

class FileSystemDataSourceBlobRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    # DataSourceBlobRepository protocol methods
    
    def save_data_package(self, data: BytesIO, id: str, file_name: str) -> Path:
        zip_path = self._create_zip_path(id, file_name)
        with open(zip_path, "wb") as f:
            f.write(data.getbuffer())

        return zip_path
    
    def load_data_package(self, id: str) -> DataPackage:
        zip_path = self._get_zip_path(id)
        with open(zip_path, "rb") as f:
            data = BytesIO(f.read())
            
        file_name = zip_path.name
        return DataPackage.from_bytes(data, file_name)
        
    def delete_data_package(self, id: str) -> None:
        # Delete the ZIP file for the data package
        zip_path = self._get_zip_path(id)
        zip_path.unlink(missing_ok=True)

        # Also delete any associated content chunks
        chunk_dir = self.base_path / id / "chunks"
        if chunk_dir.exists() and chunk_dir.is_dir():
            for chunk_file in chunk_dir.glob("*.json"):
                chunk_file.unlink(missing_ok=True)

        # Remove the empty directories
        chunk_dir.rmdir()
        zip_path.parent.rmdir()

    def list_data_packages(self) -> list[DataPackage]:
        if not self.base_path.exists() or not self.base_path.is_dir():
            return []
        data_packages = []
        for zip_path in self.base_path.glob("*/**/*.zip"):
            with open(zip_path, "rb") as f:
                data = BytesIO(f.read())
            file_name = zip_path.name
            try:
                data_package = DataPackage.from_bytes(data, file_name)
                data_packages.append(data_package)
            except InvalidDataPackageFileNameError:
                continue
        return data_packages

    def save_content_chunks(self, chunks: list[ContentChunk]) -> None:
        if not chunks:
            return
        
        data_package_id = chunks[0].data_package_id
        chunk_group_id = chunks[0].chunk_group_id
        chunk_dir = self._create_content_chunk_dir(data_package_id)
        chunk_path = chunk_dir / f"{chunk_group_id}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json_chunks = [chunk.model_dump_json() for chunk in chunks]
            json.dump(json_chunks, f, ensure_ascii=False, indent=2)
    
    def load_content_chunks_by_file_path(self, data_package_id: str, file_path: str) -> list[ContentChunk]:
        chunk_group_id = ContentChunk.get_chunk_group_id_from_file_path(file_path)
        chunks_dir = self._create_content_chunk_dir(data_package_id)
        if not chunks_dir.exists() or not chunks_dir.is_dir():
            return []
        chunk_path = chunks_dir / f"{chunk_group_id}.json"
        if not chunk_path.exists():
            return []
        with open(chunk_path, "r", encoding="utf-8") as f:
            json_chunks = json.load(f)
            return [ContentChunk.model_validate_json(json_chunk) for json_chunk in json_chunks]

    # Helper methods for internal use

    def _create_content_chunk_dir(self, data_package_id: str) -> Path:
        chunk_dir = self.base_path / data_package_id / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return chunk_dir

    def _create_zip_path(self, id: str, file_name: str) -> Path:
        zip_path = self.base_path / id / f"{file_name}.zip"
        if not str(zip_path).startswith(str(self.base_path)):
            raise InvalidDataPackageFileNameError("Invalid file path for data package: " + str(file_name))
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        return zip_path
    
    def _get_zip_path(self, id: str) -> Path:
        zip_id_path = self.base_path / id
        if not zip_id_path.exists() or not zip_id_path.is_dir():
            raise DataPackageIdNotFoundError(f"Data package id {id} not found")
        
        zip_files = list(zip_id_path.glob("*.zip"))

        if not zip_files:
            raise DataPackageZipNotFoundError(f"No ZIP file found for data package with id {id}")
        if len(zip_files) > 1:
            raise MultipleDataPackageZipFilesError(f"Multiple ZIP files found for data package with id {id}")
        return zip_files[0]


    def _walk_zip_file(self, data: BytesIO, target_file_path: str, current_path: str = "") -> BytesIO | None:
        with ZipFile(data) as zip_file:
            for zip_info in zip_file.infolist():
                if zip_info.is_dir():
                    continue

                file_entry_path = DataPackage.resolve_file_path(zip_info.filename, current_path)
                
                if file_entry_path == target_file_path:
                    with zip_file.open(zip_info) as f:
                        return BytesIO(f.read())
                    
                if not file_entry_path.endswith(".zip"):
                    continue

                with zip_file.open(zip_info) as nested_file:
                    nested_data = BytesIO(nested_file.read())
                    return self._walk_zip_file(nested_data, target_file_path, file_entry_path)
                


    

    
