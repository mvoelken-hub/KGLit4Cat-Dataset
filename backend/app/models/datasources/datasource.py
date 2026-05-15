from typing import Literal
from zipfile import ZipFile
from io import BytesIO

from pydantic import BaseModel, Field

class FileEntry(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    byte_size: int = Field(..., gt=0)
    
    def is_data_package(self) -> bool:
        return self.file_extension == ".zip"

class DataPackage(BaseModel):
    title: str
    file_extension: Literal[".zip"]
    byte_size: int = Field(..., gt=0)
    files: list["FileEntry | DataPackage"] = Field(..., min_length=1)

    @classmethod
    def from_bytes(cls, data: BytesIO, title: str, root_path: str = "") -> "DataPackage":
        with ZipFile(data) as zip_file:
            file_entries = []
            for zip_info in zip_file.infolist():
                if zip_info.is_dir():
                    continue
                file_entry = FileEntry(
                    file_path=root_path + "/" + zip_info.filename if root_path else zip_info.filename,
                    file_name=zip_info.filename.split("/")[-1],
                    file_extension="." + zip_info.filename.split(".")[-1],
                    byte_size=zip_info.file_size
                )
                if file_entry.is_data_package():
                    with zip_file.open(zip_info) as nested_file:
                        nested_data = BytesIO(nested_file.read())
                        nested_data_package = cls.from_bytes(nested_data, title=file_entry.file_name, root_path=file_entry.file_path)
                        file_entries.append(nested_data_package)
                else:
                    file_entries.append(file_entry)
            return cls(
                title=title,
                file_extension=".zip",
                byte_size=len(data.getvalue()),
                files=file_entries
            )

    def get_file_path_list(self) -> list[str]:
        file_paths = []
        for file in self.files:
            if isinstance(file, FileEntry):
                file_paths.append(file.file_path)
            elif isinstance(file, DataPackage):
                file_paths.extend(file.get_file_path_list())
        return file_paths
    
    def get_file_entry_info(self, file_path: str) -> FileEntry | None:
        for file in self.files:
            if isinstance(file, FileEntry) and file.file_path == file_path:
                return file
            elif isinstance(file, DataPackage):
                nested_info = file.get_file_entry_info(file_path)
                if nested_info is not None:
                    return nested_info
        return None

if __name__ == "__main__":
    from pprint import pprint as print

    # Example usage
    with open("C:\\Users\\simcl\\Sciebo\\Masterarbeit_Simon_Clemens\\04_Datensätze\\only_NMR_Files\\1H-1H_STM125.zip", "rb") as f:
        data = BytesIO(f.read())
        data_package = DataPackage.from_bytes(data, title="Example Data Package")
    
    print(data_package)
    print(data_package.get_file_path_list())
    print(data_package.get_file_entry_info("241.zip/241/pdata/1/parm.txt"))
