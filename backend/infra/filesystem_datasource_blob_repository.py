import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.domain.datasources import (
    ContentChunk,
    DataPackage,
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
    InvalidDataPackageFileNameError,
    InvalidDataPackageZipFileError,
    MultipleDataPackageZipFilesError,
)


class FileSystemDataSourceBlobRepository:
    def __init__(self, uploads_path: Path, output_path: Path | None = None):
        self.uploads_path = uploads_path
        self.output_path = output_path or uploads_path

    def save_data_package(self, data: BytesIO, id: str, file_name: str) -> Path:
        zip_path = self._create_zip_path(id, file_name)
        with open(zip_path, "wb") as f:
            f.write(data.getbuffer())
        return zip_path

    def load_data_package(self, id: str) -> DataPackage:
        zip_path = self._get_zip_path(id)
        with open(zip_path, "rb") as f:
            data = BytesIO(f.read())
        return DataPackage.from_bytes(data, zip_path.name)

    def delete_data_package(self, id: str) -> None:
        zip_path = self._get_zip_path(id)
        zip_path.unlink(missing_ok=True)
        package_dir = zip_path.parent
        if package_dir.exists():
            shutil.rmtree(package_dir)
        chunks_dir = self.output_path / id / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)

    def list_data_packages(self) -> list[DataPackage]:
        if not self.uploads_path.exists() or not self.uploads_path.is_dir():
            return []
        data_packages: list[DataPackage] = []
        for zip_path in self.uploads_path.glob("*/*.zip"):
            with open(zip_path, "rb") as f:
                data = BytesIO(f.read())
            try:
                data_packages.append(DataPackage.from_bytes(data, zip_path.name))
            except (InvalidDataPackageZipFileError, InvalidDataPackageFileNameError, BadZipFile):
                continue
        return data_packages

    def save_content_chunks(self, chunks: list[ContentChunk], chunking_strategy: str = "semantic") -> None:
        if not chunks:
            return
        data_package_id = chunks[0].data_package_id
        chunk_group_id = chunks[0].chunk_group_id
        chunk_dir = self._content_chunk_dir(data_package_id, chunking_strategy)
        chunk_path = chunk_dir / f"{chunk_group_id}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump_json() for chunk in chunks], f, ensure_ascii=False, indent=2)

    def delete_content_chunks(
        self,
        data_package_id: str,
        chunking_strategy: str | None = None,
    ) -> None:
        chunk_dir = (
            self.output_path / data_package_id / "chunks"
            if chunking_strategy is None
            else self.output_path / data_package_id / "chunks" / self._safe_strategy(chunking_strategy)
        )
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

    def load_content_chunks_by_file_path(
        self,
        data_package_id: str,
        file_path: str,
        chunking_strategy: str = "semantic",
    ) -> list[ContentChunk]:
        chunk_group_id = ContentChunk.get_chunk_group_id_from_file_path(file_path)
        chunk_path = self._content_chunk_dir(data_package_id, chunking_strategy) / f"{chunk_group_id}.json"
        if not chunk_path.exists():
            return []
        with open(chunk_path, "r", encoding="utf-8") as f:
            return [ContentChunk.model_validate_json(item) for item in json.load(f)]

    def _content_chunk_dir(self, data_package_id: str, chunking_strategy: str) -> Path:
        chunk_dir = self.output_path / data_package_id / "chunks" / self._safe_strategy(chunking_strategy)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return chunk_dir

    def _create_zip_path(self, id: str, file_name: str) -> Path:
        zip_id_path = self.uploads_path / id
        zip_id_path.mkdir(parents=True, exist_ok=True)
        return zip_id_path / f"{file_name}.zip"

    def _get_zip_path(self, id: str) -> Path:
        zip_id_path = self.uploads_path / id
        if not zip_id_path.exists() or not zip_id_path.is_dir():
            raise DataPackageIdNotFoundError(f"Data package id {id} not found")
        zip_files = list(zip_id_path.glob("*.zip"))
        if not zip_files:
            raise DataPackageZipNotFoundError(f"No ZIP file found for data package id {id}")
        if len(zip_files) > 1:
            raise MultipleDataPackageZipFilesError(f"Multiple ZIP files found for data package id {id}")
        return zip_files[0]

    def _walk_zip_file(
        self,
        data: BytesIO,
        target_file_path: str,
        current_path: str = "",
    ) -> BytesIO | None:
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
                result = self._walk_zip_file(nested_data, target_file_path, file_entry_path)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _safe_strategy(value: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", value or "semantic")
