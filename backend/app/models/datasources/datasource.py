from typing import Literal
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from hashlib import sha256

from pydantic import BaseModel, Field, computed_field

from app.models.datasources.errors import (
    InvalidDataPackageZipFileError,
    FileEntryNotFoundError
)

class FileEntry(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    raw_content: bytes
    
    def is_data_package(self) -> bool:
        return self.file_extension == ".zip"
    

class DataPackage(BaseModel):
    file_name: str
    file_extension: Literal[".zip"]
    files: list[FileEntry] = Field(..., min_length=1)

    @computed_field
    @property
    def id(self) -> str:
        hash_input = self.file_name + "".join(self.get_file_path_list())
        return sha256(hash_input.encode()).hexdigest()

    @classmethod
    def from_bytes(cls, data: BytesIO, file_name: str, root_path: str = "") -> "DataPackage":
        try:
            with ZipFile(data) as zip_file:
                return cls.from_zip_file(zip_file, file_name, root_path)
        except BadZipFile as e:
            raise InvalidDataPackageZipFileError(f"Invalid ZIP archive {file_name}: {e}")

    @classmethod
    def from_zip_file(cls, zip_file: ZipFile, file_name: str, root_path: str = "") -> "DataPackage":
        file_entries: list[FileEntry] = []
        for zip_info in zip_file.infolist():
            if zip_info.is_dir():
                continue
            content = b""
            with zip_file.open(zip_info) as f:
                content = f.read()
            file_entry = FileEntry(
                file_path=DataPackage.resolve_file_path(zip_info.filename, root_path),
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
            file_name=file_name,
            file_extension=".zip",
            files=file_entries
        )

    def get_file_path_list(self) -> list[str]:
        return [file.file_path for file in self.files]
    
    def get_file_entry(self, file_path: str) -> FileEntry:
        for file in self.files:
            if isinstance(file, FileEntry) and file.file_path == file_path:
                return file
        raise FileEntryNotFoundError(f"File entry with path {file_path} not found in data package {self.file_name}")
    
    def dump_without_raw_content(self) -> dict:
        return self.model_dump(
            mode="json",
            exclude={
                "files": {
                    "__all__": {"raw_content"}
                }
            }
        )
    
    @staticmethod
    def resolve_file_path(file_name: str, current_path: str) -> str:
        return current_path + "/" + file_name if current_path else file_name
    
DataPackage.model_rebuild()
FileEntry.model_rebuild()

if __name__ == "__main__":
    from pprint import pprint as print

    # Example usage
    with open("C:\\Users\\simcl\\Sciebo\\Masterarbeit_Simon_Clemens\\04_Datensätze\\only_NMR_Files\\1H-1H_STM125.zip", "rb") as f:
        data = BytesIO(f.read())
        data_package = DataPackage.from_bytes(data, file_name="1H-1H_STM125.zip")
    
    print(data_package.dump_without_raw_content())
    print(data_package.get_file_path_list())
    file_entry = data_package.get_file_entry("241.zip/241/pdata/1/parm.txt")
    if file_entry:
        print(file_entry.model_dump())
