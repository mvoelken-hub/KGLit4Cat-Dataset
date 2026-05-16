from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import unittest

import pymupdf

from app.domain.datasources import DataPackage, InvalidDataPackageZipFileError
from app.domain.datasources.file_types import FileType, determine_file_type, extract_text_from_file
from infra.filesystem_datasource_blob_repository import FileSystemDataSourceBlobRepository


def zip_bytes(entries: dict[str, bytes]) -> BytesIO:
    data = BytesIO()
    with ZipFile(data, "w") as zip_file:
        for file_path, content in entries.items():
            zip_file.writestr(file_path, content)
    data.seek(0)
    return data


def pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


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

    def test_extract_text_from_file_dispatches_pdf_without_dot_suffix(self):
        extracted = extract_text_from_file(
            content=pdf_bytes("Catalyst test content"),
            file_name="email_Jun23-2017_241_1.pdf",
        )

        self.assertIn("Catalyst test content", extracted)

    def test_nested_zip_pdf_entry_extracts_text_by_request_path(self):
        pdf_path = "241/pdata/1/email_Jun23-2017_241_1.pdf"
        inner_zip = zip_bytes({pdf_path: pdf_bytes("Nested catalyst content")})
        data_package = DataPackage.from_bytes(
            zip_bytes({"241.zip": inner_zip.getvalue()}),
            file_name="outer.zip",
        )

        file_entry = data_package.get_file_entry(f"241.zip/{pdf_path}")

        self.assertEqual(file_entry.file_extension, ".pdf")
        self.assertIn("Nested catalyst content", file_entry.get_extracted_content())

    def test_extract_text_from_file_dispatches_csv_without_dot_suffix(self):
        extracted = extract_text_from_file(
            content=b"name,value\nsample,42\n",
            file_name="sample.csv",
        )

        self.assertIn("Sheet1/1: sample", extracted)
        self.assertIn("sample,42", extracted)

    def test_determine_file_type_accepts_dotless_extensions(self):
        self.assertEqual(determine_file_type("pdf"), FileType.PDF)
        self.assertEqual(determine_file_type("csv"), FileType.TABLE)


if __name__ == "__main__":
    unittest.main()
