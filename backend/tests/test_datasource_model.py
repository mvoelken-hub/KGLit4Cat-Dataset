from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import unittest

from app.models.datasources import DataPackage, InvalidDataPackageZipFileError
from infra.filesystem_datasource_blob_repository import FileSystemDataSourceBlobRepository


def zip_bytes(entries: dict[str, bytes]) -> BytesIO:
    data = BytesIO()
    with ZipFile(data, "w") as zip_file:
        for file_path, content in entries.items():
            zip_file.writestr(file_path, content)
    data.seek(0)
    return data


class DataPackageTests(unittest.TestCase):
    def test_from_bytes_accepts_zip_file_name_and_stores_stem(self):
        data_package = DataPackage.from_bytes(
            zip_bytes({"sample.txt": b"content"}),
            file_name="1H_NMR_-1H_NMR_clean.zip",
        )

        self.assertEqual(data_package.file_name, "1H_NMR_-1H_NMR_clean")
        self.assertEqual(data_package.get_file_path_list(), ["sample.txt"])

    def test_from_bytes_rejects_non_zip_file_name(self):
        with self.assertRaises(InvalidDataPackageZipFileError):
            DataPackage.from_bytes(
                zip_bytes({"sample.txt": b"content"}),
                file_name="sample.txt",
            )

    def test_filesystem_repository_loads_saved_data_package(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))
            data = zip_bytes({"sample.txt": b"content"})
            data_package = DataPackage.from_bytes(data, file_name="sample.zip")

            repository.save_data_package(data, data_package.id, data_package.file_name)
            loaded_data_package = repository.load_data_package(data_package.id)

        self.assertEqual(loaded_data_package.file_name, "sample")
        self.assertEqual(loaded_data_package.get_file_path_list(), ["sample.txt"])


if __name__ == "__main__":
    unittest.main()
