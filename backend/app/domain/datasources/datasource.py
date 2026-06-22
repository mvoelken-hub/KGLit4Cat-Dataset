from typing import Literal
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from hashlib import sha256

from pydantic import BaseModel, Field, computed_field, field_validator

from app.domain.datasources.errors import (
    InvalidDataPackageZipFileError,
    FileEntryNotFoundError,
)
from app.domain.datasources.file_types import (
    FileType,
    determine_file_type,
    extract_text_from_file
)

class FileEntry(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    raw_content: bytes

    @computed_field
    @property
    def file_type(self) -> FileType:
        return determine_file_type(self.file_extension)
    
    def is_data_package(self) -> bool:
        return self.file_type == FileType.ARCHIVE

    def is_chunkable(self) -> bool:
        """Return whether this file can provide source text for chunking."""
        return self.file_type != FileType.IMAGE
    
    def get_extracted_content(self) -> str:
        return extract_text_from_file(content=self.raw_content, file_name=self.file_name)
    

class DataPackage(BaseModel):
    file_name: str
    files: list[FileEntry] = Field(..., min_length=1)

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str):
        if v.lower().endswith(".zip"):
            return v[:-4]
        return v

    @computed_field
    @property
    def id(self) -> str:
        hash_input = self.file_name + "".join(self.get_file_path_list())
        return sha256(hash_input.encode()).hexdigest()[:8]

    @classmethod
    def from_bytes(cls, data: BytesIO, file_name: str, root_path: str = "") -> "DataPackage":
        try:
            with ZipFile(data) as zip_file:
                return cls.from_zip_file(zip_file, file_name, root_path)
        except BadZipFile as e:
            raise InvalidDataPackageZipFileError(f"Invalid ZIP archive {file_name}: {e}")

    @classmethod
    def from_zip_file(cls, zip_file: ZipFile, file_name: str, root_path: str = "") -> "DataPackage":
        normalized_file_name = cls.normalize_zip_file_name(file_name)
        file_entries: list[FileEntry] = []
        for zip_info in zip_file.infolist():
            if zip_info.is_dir():
                continue
            content = b""
            with zip_file.open(zip_info) as f:
                content = f.read()
            file_entry = FileEntry(
                file_path=cls.resolve_file_path(zip_info.filename, root_path),
                file_name=zip_info.filename.split("/")[-1],
                file_extension="." + zip_info.filename.split(".")[-1],
                raw_content=content
            )
            if file_entry.is_data_package():
                with zip_file.open(zip_info) as nested_file:
                    nested_data = BytesIO(nested_file.read())
                    nested_data_package = cls.from_bytes(nested_data, file_name=file_entry.file_name, root_path=file_entry.file_path)
                    file_entries.extend(nested_data_package.files)
            else:
                file_entries.append(file_entry)
        return cls(
            file_name=normalized_file_name,
            files=file_entries
        )

    @staticmethod
    def normalize_zip_file_name(file_name: str) -> str:
        if not file_name.lower().endswith(".zip"):
            raise InvalidDataPackageZipFileError(f"Data package file name must end with .zip: {file_name}")
        return file_name[:-4]
    
    @staticmethod
    def resolve_file_path(file_name: str, current_path: str) -> str:
        return current_path + "/" + file_name if current_path else file_name
    
    def get_file_path_list(self) -> list[str]:
        return [file.file_path for file in self.files]
    
    def get_file_entry(self, file_path: str) -> FileEntry:
        for file in self.files:
            if isinstance(file, FileEntry) and file.file_path == file_path:
                return file
        raise FileEntryNotFoundError(f"File entry with path {file_path} not found in data package {self.file_name}")
    
    def dump_without_raw_content(self) -> dict:
        reduced_dict = self.model_dump(
            mode="json",
            exclude={
                "files": {
                    "__all__": {"raw_content"}
                }
            }
        )

        for red_file_entry, src_file_entry in zip(reduced_dict["files"], self.files):
            red_file_entry["byte_size"] = len(src_file_entry.raw_content)

        return reduced_dict

    
DataPackage.model_rebuild()
FileEntry.model_rebuild()

