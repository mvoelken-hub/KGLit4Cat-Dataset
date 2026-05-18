import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.api.v1.extraction import router
from app.dependencies import get_extraction_service
from app.domain.datasources import DataPackage, DataPackageIdNotFoundError, FileEntry
from app.domain.extraction import InitialContext
from app.services.extraction_service import ExtractionService
from infra.filesystem_extraction_output_repository import (
    FileSystemExtractionOutputRepository,
)


INITIAL_CONTEXT_OUTPUT = {
    "device_name": "Gas chromatograph",
    "device_model": "GC-42",
    "entities_analyzed": ["sample-a"],
    "analytical_technique": "gas chromatography",
    "file_relationships": [],
    "metadata_sources": [
        {
            "file_path": "metadata.txt",
            "source_type": "metadata text",
            "description": "Contains instrument and sample metadata.",
            "extracted_fields": ["device_name", "analytical_technique"],
            "evidence": "Instrument: GC-42",
            "confidence": 0.9,
        }
    ],
    "keywords": ["gas chromatography", "sample-a"],
    "summary": "The package contains gas chromatography metadata for sample-a.",
}


def make_data_package() -> DataPackage:
    return DataPackage(
        file_name="context-package",
        files=[
            FileEntry(
                file_path="metadata.txt",
                file_name="metadata.txt",
                file_extension=".txt",
                raw_content=b"Instrument: GC-42\nTechnique: gas chromatography\n",
            )
        ],
    )


class FakeProfileRepository:
    pass


class FakeDataSourceService:
    def __init__(self, data_package: DataPackage):
        self.data_package = data_package
        self.requested_id: str | None = None

    def get_data_package(self, id: str) -> DataPackage:
        self.requested_id = id
        return self.data_package


class FakeOllamaClient:
    def __init__(self, output: dict):
        self.agent_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(output),
        )


class FakeOutputRepository:
    def __init__(self):
        self.saved: dict | None = None

    def save_initial_context(
        self,
        *,
        workflow_id: str,
        initial_context: InitialContext,
    ) -> None:
        self.saved = {
            "workflow_id": workflow_id,
            "initial_context": initial_context,
        }

    def load_initial_context(self, workflow_id: str) -> InitialContext:
        if self.saved is None or self.saved["workflow_id"] != workflow_id:
            raise FileNotFoundError(workflow_id)
        return self.saved["initial_context"]


class FakeExtractionService:
    def __init__(self, result: InitialContext | Exception):
        self.result = result
        self.request: dict | None = None

    async def extract_initial_context(
        self,
        *,
        data_package_id: str,
        max_files_to_read: int = 12,
        max_chars_per_file: int = 3000,
    ) -> InitialContext:
        self.request = {
            "data_package_id": data_package_id,
            "max_files_to_read": max_files_to_read,
            "max_chars_per_file": max_chars_per_file,
        }
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class InitialContextExtractionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_initial_context_loads_datasource_and_runs_agent(self):
        datasource_service = FakeDataSourceService(make_data_package())
        output_repository = FakeOutputRepository()
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(INITIAL_CONTEXT_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        result = await service.extract_initial_context(
            data_package_id="package-id",
            max_files_to_read=3,
            max_chars_per_file=200,
        )

        self.assertEqual(datasource_service.requested_id, "package-id")
        self.assertEqual(result.device_model, "GC-42")
        self.assertEqual(result.analytical_technique, "gas chromatography")
        self.assertEqual(output_repository.saved["workflow_id"], "package-id")  # type: ignore[index]
        self.assertEqual(
            output_repository.saved["initial_context"].device_model,  # type: ignore[index]
            "GC-42",
        )


class FileSystemExtractionOutputRepositoryTests(unittest.TestCase):
    def test_initial_context_round_trip(self):
        context = InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT)
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            repository.save_initial_context(
                workflow_id="package-id",
                initial_context=context,
            )
            loaded = repository.load_initial_context("package-id")

        self.assertEqual(loaded, context)

    def test_rejects_output_paths_that_escape_base_directory(self):
        context = InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT)
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            with self.assertRaises(ValueError):
                repository.save_initial_context(
                    workflow_id="../escape",
                    initial_context=context,
                )


class InitialContextExtractionApiTests(unittest.TestCase):
    def make_client(self, service: FakeExtractionService) -> TestClient:
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_extraction_service] = lambda: service
        return TestClient(app)

    def test_initial_context_endpoint_returns_context(self):
        context = InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT)
        service = FakeExtractionService(context)
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/initial-context",
            json={
                "data_package_id": "package-id",
                "max_files_to_read": 5,
                "max_chars_per_file": 1000,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device_model"], "GC-42")
        self.assertEqual(service.request["data_package_id"], "package-id")  # type: ignore[index]

    def test_initial_context_endpoint_maps_missing_datasource_to_404(self):
        service = FakeExtractionService(DataPackageIdNotFoundError("missing package"))
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/initial-context",
            json={"data_package_id": "missing"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "missing package")


if __name__ == "__main__":
    unittest.main()
