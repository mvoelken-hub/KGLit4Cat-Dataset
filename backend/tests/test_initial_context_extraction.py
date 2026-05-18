import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.api.v1.extraction import router
from app.dependencies import get_extraction_service
from app.domain.datasources import ContentChunk, DataPackage, DataPackageIdNotFoundError, FileEntry
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContext,
    InitialContextRequiredError,
    PatchDraftPrerequisiteError,
    ProfileManifest,
)
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

INITIAL_DRAFT_OUTPUT = {
    "title": "Gas chromatography dataset for sample-a",
    "description": "Initial draft for a gas chromatography package.",
    "keywords": ["gas chromatography", "sample-a"],
}

PATCHED_DRAFT_OUTPUT = {
    "title": "Gas chromatography dataset for sample-a",
    "description": "Updated with chunk evidence.",
    "keywords": ["gas chromatography", "sample-a", "chunk-keyword"],
}

MERGE_PATCH_OUTPUT = {
    "patch": {
        "description": "Updated with chunk evidence.",
        "keywords": ["chunk-keyword"],
    }
}

PROFILE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "$defs": {
        "Dataset": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        }
    },
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
    def __init__(self):
        self.requested_manifest_identifier: str | None = None
        self.requested_schema_identifier: str | None = None
        self.manifest = ProfileManifest(
            identifier="test-profile",
            source="profile.yaml",
            source_type="upload",
            schema_file_name="profile.yaml",
            target_class="Dataset",
            checksum="sha256:test",
        )

    def get_profile_manifest(self, identifier: str) -> ProfileManifest | None:
        self.requested_manifest_identifier = identifier
        if identifier != self.manifest.identifier:
            return None
        return self.manifest

    def load_json_schema(self, identifier: str) -> dict:
        self.requested_schema_identifier = identifier
        return PROFILE_JSON_SCHEMA


class FakeDataSourceService:
    def __init__(self, data_package: DataPackage):
        self.data_package = data_package
        self.requested_id: str | None = None
        self.chunks_by_file: list[list[ContentChunk]] = []

    def get_data_package(self, id: str) -> DataPackage:
        self.requested_id = id
        return self.data_package

    def get_completed_content_chunks_by_file(
        self,
        data_package_id: str,
    ) -> list[list[ContentChunk]]:
        self.requested_id = data_package_id
        return self.chunks_by_file


class FakeOllamaClient:
    def __init__(self, output: dict):
        self.agent_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(output),
        )


class FakeOutputRepository:
    def __init__(self):
        self.saved: dict | None = None
        self.initial_context: InitialContext | None = None
        self.initial_draft: dict | None = None
        self.draft: dict | None = None
        self.patches: dict[str, dict] = {}

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
        self.initial_context = initial_context

    def load_initial_context(self, workflow_id: str) -> InitialContext:
        if self.initial_context is None:
            raise FileNotFoundError(workflow_id)
        return self.initial_context

    def save_initial_draft(
        self,
        *,
        workflow_id: str,
        initial_draft: dict,
    ) -> None:
        self.saved = {
            "workflow_id": workflow_id,
            "initial_draft": initial_draft,
        }
        self.initial_draft = initial_draft

    def load_initial_draft(self, workflow_id: str) -> dict:
        if self.initial_draft is None:
            raise FileNotFoundError(workflow_id)
        return self.initial_draft

    def save_draft(
        self,
        *,
        workflow_id: str,
        draft: dict,
    ) -> None:
        self.saved = {
            "workflow_id": workflow_id,
            "draft": draft,
        }
        self.draft = draft

    def load_draft(self, workflow_id: str) -> dict:
        if self.draft is None:
            raise FileNotFoundError(workflow_id)
        return self.draft

    def save_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict,
    ) -> None:
        self.patches[patch_file_name] = patch


class FakeExtractionService:
    def __init__(self, result: InitialContext | dict | Exception):
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
        return self.result  # type: ignore[return-value]

    async def extract_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
    ) -> dict:
        self.request = {
            "data_package_id": data_package_id,
            "profile_identifier": profile_identifier,
        }
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]

    async def patch_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        num_chunks_per_turn: int | None = None,
    ) -> dict:
        self.request = {
            "data_package_id": data_package_id,
            "profile_identifier": profile_identifier,
            "num_chunks_per_turn": num_chunks_per_turn,
        }
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]


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

    async def test_extract_initial_draft_loads_context_profile_and_persists_draft(self):
        datasource_service = FakeDataSourceService(make_data_package())
        output_repository = FakeOutputRepository()
        output_repository.initial_context = InitialContext.model_validate(
            INITIAL_CONTEXT_OUTPUT
        )
        profile_repository = FakeProfileRepository()
        service = ExtractionService(
            profile_repository,  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(INITIAL_DRAFT_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        result = await service.extract_initial_draft(
            data_package_id="package-id",
            profile_identifier="test-profile",
        )

        self.assertEqual(datasource_service.requested_id, "package-id")
        self.assertEqual(profile_repository.requested_manifest_identifier, "test-profile")
        self.assertEqual(profile_repository.requested_schema_identifier, "test-profile")
        self.assertEqual(result["title"], "Gas chromatography dataset for sample-a")
        self.assertEqual(output_repository.initial_draft, INITIAL_DRAFT_OUTPUT)

    async def test_extract_initial_draft_requires_existing_initial_context(self):
        output_repository = FakeOutputRepository()
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(INITIAL_DRAFT_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with self.assertRaises(InitialContextRequiredError):
            await service.extract_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
            )

    async def test_patch_initial_draft_requires_existing_artifacts_and_chunks(self):
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(MERGE_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=FakeOutputRepository(),
        )

        with self.assertRaises(PatchDraftPrerequisiteError):
            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
            )

    async def test_patch_initial_draft_requires_completed_chunks(self):
        output_repository = FakeOutputRepository()
        output_repository.initial_context = InitialContext.model_validate(
            INITIAL_CONTEXT_OUTPUT
        )
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(MERGE_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with self.assertRaises(ChunkingRequiredError):
            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
            )

    async def test_patch_initial_draft_persists_draft_and_patches(self):
        datasource_service = FakeDataSourceService(make_data_package())
        datasource_service.chunks_by_file = [
            [
                ContentChunk(
                    content="Chunk says chunk-keyword.",
                    data_package_id="package-id",
                    file_path="metadata.txt",
                    start_idx=0,
                    end_idx=0,
                )
            ]
        ]
        output_repository = FakeOutputRepository()
        output_repository.initial_context = InitialContext.model_validate(
            INITIAL_CONTEXT_OUTPUT
        )
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(MERGE_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        result = await service.patch_initial_draft(
            data_package_id="package-id",
            profile_identifier="test-profile",
            num_chunks_per_turn=1,
        )

        self.assertEqual(result["description"], "Updated with chunk evidence.")
        self.assertEqual(output_repository.draft, result)
        self.assertEqual(list(output_repository.patches.values())[0], MERGE_PATCH_OUTPUT["patch"])


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

    def test_initial_draft_round_trip(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            repository.save_initial_draft(
                workflow_id="package-id",
                initial_draft=INITIAL_DRAFT_OUTPUT,
            )
            loaded = repository.load_initial_draft("package-id")

        self.assertEqual(loaded, INITIAL_DRAFT_OUTPUT)

    def test_draft_and_patch_round_trip(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            repository.save_draft(workflow_id="package-id", draft=PATCHED_DRAFT_OUTPUT)
            repository.save_patch(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                patch=MERGE_PATCH_OUTPUT["patch"],
            )

            loaded = repository.load_draft("package-id")
            patch_path = (
                Path(temporary_directory)
                / "package-id"
                / "patches"
                / "patch_1.json"
            )
            patch_exists = patch_path.exists()
            patch = json.loads(patch_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, PATCHED_DRAFT_OUTPUT)
        self.assertTrue(patch_exists)
        self.assertEqual(patch, MERGE_PATCH_OUTPUT["patch"])

    def test_missing_initial_context_raises_file_not_found(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            with self.assertRaises(FileNotFoundError):
                repository.load_initial_context("package-id")

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

    def test_initial_draft_endpoint_returns_draft(self):
        service = FakeExtractionService(INITIAL_DRAFT_OUTPUT)
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/initial-draft",
            json={
                "data_package_id": "package-id",
                "profile_identifier": "test-profile",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], INITIAL_DRAFT_OUTPUT["title"])
        self.assertEqual(service.request["data_package_id"], "package-id")  # type: ignore[index]
        self.assertEqual(service.request["profile_identifier"], "test-profile")  # type: ignore[index]

    def test_initial_draft_endpoint_maps_missing_initial_context_to_409(self):
        service = FakeExtractionService(
            InitialContextRequiredError(
                "Initial context output not found. Run /api/v1/extraction/initial-context first."
            )
        )
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/initial-draft",
            json={
                "data_package_id": "package-id",
                "profile_identifier": "test-profile",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("/initial-context", response.json()["detail"])

    def test_patch_draft_endpoint_returns_draft(self):
        service = FakeExtractionService(PATCHED_DRAFT_OUTPUT)
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/patch-draft",
            json={
                "data_package_id": "package-id",
                "profile_identifier": "test-profile",
                "num_chunks_per_turn": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["description"], "Updated with chunk evidence.")
        self.assertEqual(service.request["num_chunks_per_turn"], 2)  # type: ignore[index]

    def test_patch_draft_endpoint_maps_prerequisites_to_409(self):
        service = FakeExtractionService(ChunkingRequiredError("chunk first"))
        client = self.make_client(service)

        response = client.post(
            "/api/v1/extraction/patch-draft",
            json={
                "data_package_id": "package-id",
                "profile_identifier": "test-profile",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "chunk first")


if __name__ == "__main__":
    unittest.main()
