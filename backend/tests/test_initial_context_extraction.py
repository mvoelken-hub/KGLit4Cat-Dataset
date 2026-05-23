import json
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.api.v1.extraction import router
from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.dependencies import get_extraction_service
from app.domain.datasources import ContentChunk, DataPackage, DataPackageIdNotFoundError, FileEntry
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContext,
    InitialContextRequiredError,
    PatchDraftPrerequisiteError,
)
from app.domain.extraction.patch_quality import CandidateQualityRating, PatchQualityReport, UnmappedFact
from app.domain.extraction.review_resolution import PatchReviewResolution
from app.domain.extraction.artifacts import PatchCandidate
from app.domain.profiles import ProfileManifest
from app.services.extraction_service import ExtractionService
from infra.filesystem_extraction_output_repository import (
    FileSystemExtractionOutputRepository,
)


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


async def _completed_task() -> None:
    return None


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

FIELD_PATCH_OUTPUT = {
    "candidates": [
        {
            "field_path": "description",
            "patch": {"description": "Updated with chunk evidence."},
            "confidence": 0.9,
            "reasoning": "Chunk contains updated description.",
            "source_evidence": ["Chunk says chunk-keyword."],
        },
        {
            "field_path": "keywords",
            "patch": {"keywords": ["chunk-keyword"]},
            "confidence": 0.85,
            "reasoning": "Chunk contains keyword evidence.",
            "source_evidence": ["Chunk says chunk-keyword."],
        },
    ]
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
    def __init__(self, json_schema: dict | None = None):
        self.requested_manifest_identifier: str | None = None
        self.requested_schema_identifier: str | None = None
        self.json_schema = json_schema or PROFILE_JSON_SCHEMA
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

    def get_profile(self, identifier: str) -> ProfileManifest:
        manifest = self.get_profile_manifest(identifier)
        if manifest is None:
            raise AssertionError(f"Unexpected profile identifier: {identifier}")
        return manifest

    def load_json_schema(self, identifier: str) -> dict:
        self.requested_schema_identifier = identifier
        return self.json_schema


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
        self.draft_saves: list[dict] = []
        self.patches: dict[str, dict] = {}
        self.raw_patches: dict[str, dict] = {}
        self.accepted_patches: dict[str, dict] = {}
        self.candidates: dict[str, list] = {}
        self.quality_reports: dict[str, PatchQualityReport] = {}
        self.unmapped_facts: dict[str, list[UnmappedFact]] = {}
        self.protected_fields: list[str] = []
        self.review_state: dict = {
            "resolved_item_ids": [],
            "unmapped_assignments": {},
            "resolution_notes": {},
            "resolved_at": {},
        }

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
        self.draft_saves.append(json.loads(json.dumps(draft)))

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

    def save_raw_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict,
    ) -> None:
        self.raw_patches[patch_file_name] = patch

    def save_accepted_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict,
    ) -> None:
        self.accepted_patches[patch_file_name] = patch

    def save_candidates(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        candidates: list,
    ) -> None:
        self.candidates[patch_file_name] = candidates

    def save_quality_report(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        quality_report: PatchQualityReport,
    ) -> None:
        self.quality_reports[patch_file_name] = quality_report

    def save_unmapped_facts(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        unmapped_facts: list[UnmappedFact],
    ) -> None:
        self.unmapped_facts[patch_file_name] = unmapped_facts

    def save_protected_fields(
        self,
        *,
        workflow_id: str,
        protected_fields: list[str],
    ) -> None:
        self.protected_fields = protected_fields

    def load_protected_fields(self, workflow_id: str) -> list[str]:
        return self.protected_fields

    def save_patch_review_state(
        self,
        *,
        workflow_id: str,
        review_state: dict,
    ) -> None:
        self.review_state = review_state

    def load_patch_review_state(self, workflow_id: str) -> dict:
        return self.review_state

    def load_patch_files(self, workflow_id: str) -> list[dict]:
        artifacts = [
            {"file_name": file_name, "artifact_type": "patch", "content": patch}
            for file_name, patch in sorted(self.patches.items())
        ]
        artifacts.extend(
            {
                "file_name": file_name.replace(".json", ".candidates.json"),
                "artifact_type": "candidates",
                "content": [
                    candidate.model_dump(mode="json")
                    if hasattr(candidate, "model_dump")
                    else candidate
                    for candidate in candidates
                ],
            }
            for file_name, candidates in sorted(self.candidates.items())
        )
        return artifacts

    def load_patch_quality_reports(self, workflow_id: str) -> list[dict]:
        return [
            {
                "file_name": file_name,
                "content": report.model_dump(mode="json"),
            }
            for file_name, report in sorted(self.quality_reports.items())
        ]

    def load_unmapped_facts(self, workflow_id: str) -> list[dict]:
        facts: list[dict] = []
        for file_name, items in sorted(self.unmapped_facts.items()):
            facts.extend(
                {"file_name": file_name, **item.model_dump(mode="json")}
                for item in items
            )
        return facts

    def load_completed_patch_file_names(self, workflow_id: str) -> set[str]:
        return set(self.patches)

    def clear_patch_artifacts(self, workflow_id: str) -> None:
        self.draft = None
        self.patches = {}
        self.raw_patches = {}
        self.accepted_patches = {}
        self.candidates = {}
        self.quality_reports = {}
        self.unmapped_facts = {}


class FakeExtractionService:
    def __init__(self, result: InitialContext | dict | tuple[dict, TaskStatus] | Exception):
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
        auto_resolve: bool = False,
    ) -> tuple[dict, TaskStatus]:
        self.request = {
            "data_package_id": data_package_id,
            "profile_identifier": profile_identifier,
            "num_chunks_per_turn": num_chunks_per_turn,
            "auto_resolve": auto_resolve,
        }
        if isinstance(self.result, Exception):
            raise self.result
        if isinstance(self.result, tuple):
            return self.result
        return self.result, TaskStatus.COMPLETED  # type: ignore[return-value]

    async def get_patch_progress(
        self,
        data_package_id: str,
    ) -> tuple[TaskStatus, dict | None]:
        self.request = {"data_package_id": data_package_id}
        return TaskStatus.UNKNOWN, None

    async def get_patch_artifacts(self, data_package_id: str) -> dict:
        self.request = {"data_package_id": data_package_id}
        return {"patches": [], "quality_reports": [], "unmapped_facts": []}

    async def get_patch_files(self, data_package_id: str) -> list[dict]:
        self.request = {"data_package_id": data_package_id}
        return []

    async def get_patch_quality_reports(self, data_package_id: str) -> list[dict]:
        self.request = {"data_package_id": data_package_id}
        return []

    async def get_unmapped_facts(self, data_package_id: str) -> list[dict]:
        self.request = {"data_package_id": data_package_id}
        return []

    async def get_patch_review_state(self, data_package_id: str) -> dict:
        self.request = {"data_package_id": data_package_id}
        return {
            "resolved_item_ids": [],
            "unmapped_assignments": {},
            "resolution_notes": {},
            "resolved_at": {},
        }

    async def save_patch_review_state(
        self,
        *,
        data_package_id: str,
        review_state: dict,
    ) -> dict:
        self.request = {
            "data_package_id": data_package_id,
            "review_state": review_state,
        }
        return review_state


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
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        profile_repository = FakeProfileRepository()
        service = ExtractionService(
            profile_repository,  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(INITIAL_DRAFT_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )
        task_name = service._patch_draft_task_name("package-id")
        await task_registry.create_task(
            coro=_completed_task(),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        await task_registry.wait_for_task(task_name, timeout=1.0)

        result = await service.extract_initial_draft(
            data_package_id="package-id",
            profile_identifier="test-profile",
        )

        self.assertEqual(datasource_service.requested_id, "package-id")
        self.assertEqual(profile_repository.requested_manifest_identifier, "test-profile")
        self.assertEqual(profile_repository.requested_schema_identifier, "test-profile")
        self.assertEqual(result["title"], "Gas chromatography dataset for sample-a")
        self.assertEqual(output_repository.initial_draft, INITIAL_DRAFT_OUTPUT)
        self.assertIsNone(task_registry.get_task_info(task_name))

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
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=FakeOutputRepository(),
            task_registry=task_registry,
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
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )

        with self.assertRaises(ChunkingRequiredError):
            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
            )

    async def test_patch_initial_draft_restarts_stale_completed_task_without_artifacts(self):
        from unittest import mock as unittest_mock

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
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="All candidates accepted.",
        )
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )
        task_name = service._patch_draft_task_name("package-id")
        await task_registry.create_task(
            coro=_completed_task(),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        await task_registry.wait_for_task(task_name, timeout=1.0)
        self.assertEqual(
            await service.get_patch_progress("package-id"),
            (TaskStatus.UNKNOWN, None),
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            draft, status = await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(task_name, timeout=2.0)

        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(draft, INITIAL_DRAFT_OUTPUT)
        self.assertTrue(output_repository.patches)

    async def test_patch_initial_draft_persists_draft_and_patches(self):
        from unittest import mock as unittest_mock

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

        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="All candidates accepted.",
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
            service = ExtractionService(
                FakeProfileRepository(),  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                datasource_service=datasource_service,  # type: ignore[arg-type]
                ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
                output_repository=output_repository,
                task_registry=task_registry,
            )

            draft, status = await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        result = output_repository.draft
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(draft, INITIAL_DRAFT_OUTPUT)
        self.assertIsNotNone(result)
        self.assertEqual(result["description"], "Updated with chunk evidence.")
        self.assertEqual(output_repository.draft, result)
        self.assertEqual(output_repository.draft_saves[0], INITIAL_DRAFT_OUTPUT)
        self.assertEqual(list(output_repository.patches.values())[0], {"description": "Updated with chunk evidence.", "keywords": ["chunk-keyword"]})
        task_info = task_registry.get_task_info(service._patch_draft_task_name("package-id"))
        self.assertEqual(task_info.progress["batch_no"], 1)  # type: ignore[union-attr,index]
        self.assertEqual(task_info.progress["total_batches"], 1)  # type: ignore[union-attr,index]

    async def test_patch_initial_draft_starts_auto_resolve_before_patching_finishes(self):
        from unittest import mock as unittest_mock

        datasource_service = FakeDataSourceService(make_data_package())
        datasource_service.chunks_by_file = [
            [
                ContentChunk(
                    content="First chunk says chunk-keyword.",
                    data_package_id="package-id",
                    file_path="metadata.txt",
                    start_idx=0,
                    end_idx=0,
                ),
                ContentChunk(
                    content="Second chunk says chunk-keyword.",
                    data_package_id="package-id",
                    file_path="metadata.txt",
                    start_idx=1,
                    end_idx=1,
                ),
            ]
        ]
        output_repository = FakeOutputRepository()
        output_repository.initial_context = InitialContext.model_validate(
            INITIAL_CONTEXT_OUTPUT
        )
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="All candidates accepted.",
        )
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )
        resolve_patch_counts: list[int] = []

        async def fake_auto_resolve(
            *,
            data_package_id: str,
            profile_identifier: str,
            on_token_usage=None,
        ) -> None:
            resolve_patch_counts.append(len(output_repository.patches))
            await asyncio.sleep(0.01)

        service._auto_resolve_review_items = fake_auto_resolve  # type: ignore[method-assign]

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
                auto_resolve=True,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        self.assertEqual(len(output_repository.patches), 2)
        self.assertGreaterEqual(len(resolve_patch_counts), 2)
        self.assertEqual(resolve_patch_counts[0], 1)

    async def test_patch_initial_draft_progress_includes_token_usage_summary(self):
        from unittest import mock as unittest_mock

        class FakeUsage:
            input_tokens = 120
            output_tokens = 30
            total_tokens = 150
            requests = 1

        datasource_service = FakeDataSourceService(make_data_package())
        datasource_service.chunks_by_file = [
            [
                ContentChunk(
                    file_path="metadata.txt",
                    chunk_index=0,
                    text="Chunk says chunk-keyword.",
                )
            ]
        ]
        output_repository = FakeOutputRepository()
        output_repository.initial_context = InitialContext.model_validate(
            INITIAL_CONTEXT_OUTPUT
        )
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="All candidates accepted.",
        )

        async def fake_quality_review(**kwargs):
            kwargs["on_token_usage"]("patch_quality", FakeUsage(), 1)
            return accept_report

        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=datasource_service,  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            side_effect=fake_quality_review,
        ):
            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        progress = task_registry.get_task_info(
            service._patch_draft_task_name("package-id"),
        ).progress
        token_usage = progress["token_usage"]  # type: ignore[index]
        self.assertIn("patch_extraction", token_usage["agents"])
        self.assertEqual(
            token_usage["agents"]["patch_quality"]["average_total_tokens_per_patch"],
            150,
        )
        self.assertGreater(token_usage["combined"]["average_total_tokens_per_patch"], 150)

    async def test_auto_resolve_marks_progress_active_while_running(self):
        from unittest import mock as unittest_mock

        output_repository = FakeOutputRepository()
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        output_repository.draft = INITIAL_DRAFT_OUTPUT
        output_repository.candidates["patch_1.json"] = [
            {
                "field_path": "description",
                "patch": {"description": "Updated with chunk evidence."},
                "confidence": 0.5,
                "reasoning": "Low confidence candidate needs resolver review.",
                "source_evidence": ["Chunk says chunk-keyword."],
            }
        ]
        output_repository.quality_reports["patch_1.json"] = PatchQualityReport(
            overall_decision="reject",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="reject",
                    issues=[],
                )
            ],
            summary="Resolver should review the candidate.",
        )
        task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
            output_repository=output_repository,
            task_registry=task_registry,
        )
        task_name = service._patch_draft_task_name("package-id")
        await task_registry.create_task(
            asyncio.sleep(1),
            TaskType.WORKFLOW,
            task_name,
        )
        resolver_started = asyncio.Event()
        release_resolver = asyncio.Event()

        class FakeUsage:
            input_tokens = 80
            output_tokens = 20
            total_tokens = 100
            requests = 1

        token_usage_totals: dict[str, dict[str, int]] = {}

        def record_usage(agent_name, usage, patch_count):
            service._record_patch_token_usage(
                token_usage_totals,
                agent_name=agent_name,
                usage=usage,
                patch_count=patch_count,
            )
            service._update_patch_token_usage_progress(
                data_package_id="package-id",
                token_usage=token_usage_totals,
            )

        async def fake_resolve(**kwargs):
            resolver_started.set()
            await release_resolver.wait()
            kwargs["on_token_usage"]("auto_resolve", FakeUsage(), 1)
            return PatchReviewResolution(
                final_draft=INITIAL_DRAFT_OUTPUT,
                item_decisions=[
                    {
                        "id": "matched:patch_1.json:description:0",
                        "outcome": "excluded",
                        "note": "Insufficient confidence.",
                        "target_path": "description",
                    }
                ],
            )

        try:
            with unittest_mock.patch(
                "app.services.extraction_service.resolve_patch_review_items",
                side_effect=fake_resolve,
            ):
                resolve_task = asyncio.create_task(
                    service._auto_resolve_review_items(
                        data_package_id="package-id",
                        profile_identifier="test-profile",
                        on_token_usage=record_usage,
                    )
                )
                await asyncio.wait_for(resolver_started.wait(), timeout=2.0)
                task_info = task_registry.get_task_info(task_name)
                self.assertTrue(task_info.progress["resolution_active"])  # type: ignore[union-attr,index]

                release_resolver.set()
                result = await asyncio.wait_for(resolve_task, timeout=2.0)

            task_info = task_registry.get_task_info(task_name)
            self.assertFalse(task_info.progress["resolution_active"])  # type: ignore[union-attr,index]
            self.assertEqual(task_info.progress["resolution_resolved_count"], 1)  # type: ignore[union-attr,index]
            self.assertEqual(task_info.progress["resolution_unresolved_item_ids"], [])  # type: ignore[union-attr,index]
            self.assertEqual(
                task_info.progress["token_usage"]["agents"]["auto_resolve"]["average_total_tokens_per_patch"],  # type: ignore[union-attr,index]
                100,
            )
            self.assertEqual(result["resolved_count"], 1)
        finally:
            await task_registry.cancel_task(task_name)

    async def test_patch_initial_draft_normalizes_duplicate_object_arrays_before_saving(self):
        from unittest import mock as unittest_mock

        rich_schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$defs": {
                "Attribute": {
                    "type": "object",
                    "properties": {
                        "has_quantity_type": {"type": "string"},
                        "unit": {"type": "string"},
                        "value": {"type": "number"},
                        "title": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "Activity": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "has_quantitative_attribute": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Attribute"},
                        },
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
                "Dataset": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "was_generated_by": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Activity"},
                        },
                    },
                    "required": ["id", "was_generated_by"],
                    "additionalProperties": False,
                },
            },
        }
        initial_draft = {
            "id": "dataset",
            "was_generated_by": [
                {
                    "id": "activity-nmr",
                    "has_quantitative_attribute": [
                        {
                            "has_quantity_type": "frequency",
                            "unit": "MHz",
                            "value": 400.0,
                            "title": "Magnetic Field Strength",
                        }
                    ],
                }
            ],
        }
        patch_output = {
            "candidates": [
                {
                    "field_path": "was_generated_by",
                    "patch": {
                        "was_generated_by": [
                            {
                                "id": "activity-nmr",
                                "has_quantitative_attribute": [
                                    {
                                        "has_quantity_type": "frequency",
                                        "unit": "MHz",
                                        "value": 400.0,
                                        "title": "Magnetic Field Strength",
                                    },
                                    {
                                        "has_quantity_type": "frequency",
                                        "unit": "MHz",
                                        "value": 400.0,
                                        "title": "Magnetic Field Strength",
                                    },
                                ],
                            }
                        ]
                    },
                    "confidence": 0.95,
                    "reasoning": "Chunk repeats the field strength.",
                    "source_evidence": ["400 MHz"],
                }
            ]
        }
        datasource_service = FakeDataSourceService(make_data_package())
        datasource_service.chunks_by_file = [
            [
                ContentChunk(
                    content="400 MHz",
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
        output_repository.initial_draft = initial_draft
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="was_generated_by",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="Accepted.",
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
            service = ExtractionService(
                FakeProfileRepository(rich_schema),  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                datasource_service=datasource_service,  # type: ignore[arg-type]
                ollama_client=FakeOllamaClient(patch_output),  # type: ignore[arg-type]
                output_repository=output_repository,
                task_registry=task_registry,
            )

            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        activity = output_repository.draft["was_generated_by"][0]
        self.assertEqual(
            activity["has_quantitative_attribute"],
            [
                {
                    "has_quantity_type": "frequency",
                    "unit": "MHz",
                    "value": 400.0,
                    "title": "Magnetic Field Strength",
                }
            ],
        )

    async def test_patch_initial_draft_does_not_merge_protected_fields(self):
        from unittest import mock as unittest_mock

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
        output_repository.protected_fields = ["description"]

        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="Unprotected candidates accepted.",
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
            service = ExtractionService(
                FakeProfileRepository(),  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                datasource_service=datasource_service,  # type: ignore[arg-type]
                ollama_client=FakeOllamaClient(FIELD_PATCH_OUTPUT),  # type: ignore[arg-type]
                output_repository=output_repository,
                task_registry=task_registry,
            )

            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        self.assertEqual(
            output_repository.draft["description"],
            INITIAL_DRAFT_OUTPUT["description"],
        )
        self.assertEqual(
            output_repository.draft["keywords"],
            ["gas chromatography", "sample-a", "chunk-keyword"],
        )
        self.assertEqual(
            list(output_repository.patches.values())[0],
            {"keywords": ["chunk-keyword"]},
        )

    async def test_patch_initial_draft_rejects_revisions_touching_protected_fields(self):
        from unittest import mock as unittest_mock

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
        output_repository.protected_fields = ["description"]

        revise_report = PatchQualityReport(
            overall_decision="revise",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="keywords",
                    decision="revise",
                    issues=[],
                    revised_patch={
                        "description": "Should not be merged.",
                        "keywords": ["chunk-keyword"],
                    },
                ),
            ],
            summary="Revision touches a protected field.",
        )

        with unittest_mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=revise_report,
        ):
            task_registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]
            service = ExtractionService(
                FakeProfileRepository(),  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                datasource_service=datasource_service,  # type: ignore[arg-type]
                ollama_client=FakeOllamaClient(
                    {
                        "candidates": [
                            {
                                "field_path": "keywords",
                                "patch": {"keywords": ["chunk-keyword"]},
                                "confidence": 0.85,
                                "reasoning": "Chunk contains keyword evidence.",
                                "source_evidence": ["Chunk says chunk-keyword."],
                            },
                        ]
                    }
                ),  # type: ignore[arg-type]
                output_repository=output_repository,
                task_registry=task_registry,
            )

            await service.patch_initial_draft(
                data_package_id="package-id",
                profile_identifier="test-profile",
                num_chunks_per_turn=1,
            )
            await task_registry.wait_for_task(
                service._patch_draft_task_name("package-id"),
                timeout=2.0,
            )

        self.assertEqual(output_repository.draft, INITIAL_DRAFT_OUTPUT)
        self.assertEqual(list(output_repository.patches.values())[0], {})

    async def test_resolve_patch_review_includes_full_draft_change(self):
        from unittest import mock as unittest_mock

        output_repository = FakeOutputRepository()
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        final_draft = {
            **INITIAL_DRAFT_OUTPUT,
            "description": "Resolved with source evidence.",
        }
        resolution = PatchReviewResolution(
            final_draft=final_draft,
            item_decisions=[
                {
                    "id": "matched:patch:description:0",
                    "outcome": "included",
                    "note": "Added the sourced description.",
                    "target_path": "description",
                },
                {
                    "id": "unmapped:temperature",
                    "outcome": "unresolved",
                    "note": "No confident target field.",
                },
            ],
        )
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with unittest_mock.patch(
            "app.services.extraction_service.resolve_patch_review_items",
            return_value=resolution,
        ):
            result = await service.resolve_patch_review_items(
                data_package_id="package-id",
                profile_identifier="test-profile",
                review_items=[
                    {
                        "id": "matched:patch:description:0",
                        "kind": "matched",
                        "path": "description",
                    },
                    {
                        "id": "unmapped:temperature",
                        "kind": "unmapped",
                        "path": "",
                        "fact": "Temperature was 300 K.",
                    },
                ],
            )

        self.assertEqual(result["resolved_count"], 2)
        self.assertEqual(result["unresolved_item_ids"], [])
        self.assertEqual(result["draft"], final_draft)
        self.assertEqual(output_repository.draft, final_draft)
        self.assertEqual(
            output_repository.review_state["resolved_item_ids"],
            ["matched:patch:description:0", "unmapped:temperature"],
        )
        self.assertEqual(
            output_repository.review_state["resolution_notes"]["matched:patch:description:0"],
            "included: Added the sourced description. Target: description.",
        )
        self.assertIn(
            "excluded: The resolution agent marked this item unresolved",
            output_repository.review_state["resolution_notes"]["unmapped:temperature"],
        )
        self.assertTrue(result["resolution_log"])

    def test_auto_resolve_review_item_ids_match_frontend_shape(self):
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
        )

        review_items = service._build_patch_review_items_from_artifacts(
            artifacts={
                "patches": [
                    {
                        "file_name": "patch.candidates.json",
                        "artifact_type": "candidates",
                        "content": [
                            {
                                "field_path": "was_generated_by",
                                "patch": {"was_generated_by": [{"name": ["Instrument"]}]},
                                "confidence": 0.9,
                                "reasoning": "Candidate needs quality review.",
                                "source_evidence": ["Instrument evidence"],
                            }
                        ],
                    }
                ],
                "quality_reports": [
                    {
                        "file_name": "patch.quality_report.json",
                        "content": {
                            "candidate_ratings": [
                                {
                                    "field_path": "was_generated_by",
                                    "decision": "reject",
                                    "issues": [
                                        {
                                            "issue_type": "unsupported_fact",
                                            "severity": "major",
                                            "explanation": "Not supported by source.",
                                            "suggested_target_path": "description",
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "unmapped_facts": [
                    {
                        "file_name": "patch.unmapped_facts.json",
                        "fact": "Temperature was 300 K.",
                        "reason": "No target field.",
                        "source_hint": "line 4",
                    }
                ],
            },
            existing_state={
                "resolved_item_ids": [],
                "unmapped_assignments": {},
                "resolution_notes": {},
                "resolved_at": {},
            },
        )

        self.assertEqual(
            [item["id"] for item in review_items],
            [
                "matched:patch.json:was_generated_by:0",
                "unmapped:patch.unmapped_facts.json|Temperature was 300 K.|No target field.|line 4",
            ],
        )
        self.assertEqual(review_items[0]["issues"], [
            "unsupported_fact (major) Not supported by source. Suggested field: description",
        ])

    async def test_resolve_patch_review_excludes_without_changing_draft(self):
        from unittest import mock as unittest_mock

        output_repository = FakeOutputRepository()
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        resolution = PatchReviewResolution(
            final_draft=INITIAL_DRAFT_OUTPUT,
            item_decisions=[
                {
                    "id": "matched:patch:instrument:0",
                    "outcome": "excluded",
                    "note": "Source instrument conflicts with the NMR context.",
                    "target_path": "was_generated_by",
                }
            ],
        )
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with unittest_mock.patch(
            "app.services.extraction_service.resolve_patch_review_items",
            return_value=resolution,
        ):
            result = await service.resolve_patch_review_items(
                data_package_id="package-id",
                profile_identifier="test-profile",
                review_items=[
                    {
                        "id": "matched:patch:instrument:0",
                        "kind": "matched",
                        "path": "was_generated_by",
                    },
                ],
            )

        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["unresolved_item_ids"], [])
        self.assertEqual(output_repository.draft, INITIAL_DRAFT_OUTPUT)
        self.assertEqual(
            output_repository.review_state["resolution_notes"]["matched:patch:instrument:0"],
            "excluded: Source instrument conflicts with the NMR context. Target: was_generated_by.",
        )

    async def test_resolve_patch_review_excludes_missing_agent_decision(self):
        from unittest import mock as unittest_mock

        output_repository = FakeOutputRepository()
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        resolution = PatchReviewResolution(
            final_draft=INITIAL_DRAFT_OUTPUT,
            item_decisions=[
                {
                    "id": "matched:patch:description:0",
                    "outcome": "already_present",
                    "note": "Description is already represented.",
                    "target_path": "description",
                }
            ],
        )
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with unittest_mock.patch(
            "app.services.extraction_service.resolve_patch_review_items",
            return_value=resolution,
        ):
            result = await service.resolve_patch_review_items(
                data_package_id="package-id",
                profile_identifier="test-profile",
                review_items=[
                    {
                        "id": "matched:patch:description:0",
                        "kind": "matched",
                        "path": "description",
                    },
                    {
                        "id": "matched:patch:keywords:1",
                        "kind": "matched",
                        "path": "keywords",
                    },
                ],
            )

        self.assertEqual(result["resolved_count"], 2)
        self.assertEqual(result["unresolved_item_ids"], [])
        self.assertEqual(
            output_repository.review_state["resolved_item_ids"],
            ["matched:patch:description:0", "matched:patch:keywords:1"],
        )
        self.assertIn(
            "excluded: The resolution agent did not return a decision",
            output_repository.review_state["resolution_notes"]["matched:patch:keywords:1"],
        )
        self.assertTrue(
            any("Converted 1 missing or unresolved decisions" in entry for entry in result["resolution_log"]),
        )

    async def test_resolve_patch_review_rejects_invalid_full_draft(self):
        from unittest import mock as unittest_mock

        output_repository = FakeOutputRepository()
        output_repository.initial_draft = INITIAL_DRAFT_OUTPUT
        resolution = PatchReviewResolution(
            final_draft={
                **INITIAL_DRAFT_OUTPUT,
                "unknown_field": "Invalid extra field.",
            },
            item_decisions=[
                {
                    "id": "matched:patch:description:0",
                    "outcome": "included",
                    "note": "Added extra field.",
                }
            ],
        )
        service = ExtractionService(
            FakeProfileRepository(),  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
            ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
            output_repository=output_repository,
        )

        with unittest_mock.patch(
            "app.services.extraction_service.resolve_patch_review_items",
            return_value=resolution,
        ):
            result = await service.resolve_patch_review_items(
                data_package_id="package-id",
                profile_identifier="test-profile",
                review_items=[
                    {
                        "id": "matched:patch:description:0",
                        "kind": "matched",
                        "path": "description",
                    },
                ],
            )

        self.assertEqual(result["resolved_count"], 0)
        self.assertEqual(result["unresolved_item_ids"], ["matched:patch:description:0"])
        self.assertTrue(result["validation_errors"])
        self.assertTrue(result["resolution_log"])
        self.assertIsNone(output_repository.draft)
        self.assertEqual(output_repository.review_state["resolved_item_ids"], [])

    async def test_resolve_patch_review_requires_schema_valid_checksum_distribution(self):
        from unittest import mock as unittest_mock

        checksum_schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$defs": {
                "Dataset": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "dataset_distribution": {
                            "type": ["array", "null"],
                            "items": {"$ref": "#/$defs/Distribution"},
                        },
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
                "Distribution": {
                    "type": "object",
                    "properties": {
                        "title": {"type": ["array", "null"], "items": {"type": "string"}},
                        "access_URL": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Resource"},
                        },
                        "checksum": {
                            "anyOf": [
                                {"$ref": "#/$defs/Checksum"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["access_URL"],
                    "additionalProperties": False,
                },
                "Resource": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                "Checksum": {
                    "type": "object",
                    "properties": {
                        "algorithm": {"$ref": "#/$defs/ChecksumAlgorithm"},
                        "checksum_value": {
                            "type": "string",
                            "pattern": "([0-9a-fA-F]{2})*",
                        },
                    },
                    "required": ["algorithm", "checksum_value"],
                    "additionalProperties": False,
                },
                "ChecksumAlgorithm": {
                    "type": "object",
                    "properties": {
                        "title": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        }
        initial_draft = {"title": "NMR dataset"}
        invalid_draft = {
            "title": "NMR dataset",
            "dataset_distribution": [
                {
                    "title": ["Peak list data"],
                    "access_URL": [{"id": "resource/peak-jdx"}],
                    "checksum": {
                        "algorithm": {"id": "term/sha1", "title": "SHA-1"},
                        "checksum_value": "454a83afceb23229e0bd94b4a91cedbd",
                    },
                }
            ],
        }
        valid_draft = {
            "title": "NMR dataset",
            "dataset_distribution": [
                {
                    "title": ["Peak list data"],
                    "access_URL": [{"id": "resource/peak-jdx"}],
                    "checksum": {
                        "algorithm": {"title": "SHA-1"},
                        "checksum_value": "454a83afceb23229e0bd94b4a91cedbd",
                    },
                }
            ],
        }

        async def run_resolution(final_draft: dict) -> tuple[dict, FakeOutputRepository]:
            output_repository = FakeOutputRepository()
            output_repository.initial_draft = initial_draft
            service = ExtractionService(
                FakeProfileRepository(checksum_schema),  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                datasource_service=FakeDataSourceService(make_data_package()),  # type: ignore[arg-type]
                ollama_client=FakeOllamaClient({}),  # type: ignore[arg-type]
                output_repository=output_repository,
            )
            resolution = PatchReviewResolution(
                final_draft=final_draft,
                item_decisions=[
                    {
                        "id": "matched:patch:distribution:0",
                        "outcome": "included",
                        "note": "Added checksum distribution.",
                        "target_path": "dataset_distribution",
                    }
                ],
            )
            with unittest_mock.patch(
                "app.services.extraction_service.resolve_patch_review_items",
                return_value=resolution,
            ):
                result = await service.resolve_patch_review_items(
                    data_package_id="package-id",
                    profile_identifier="test-profile",
                    review_items=[
                        {
                            "id": "matched:patch:distribution:0",
                            "kind": "matched",
                            "path": "dataset_distribution",
                        },
                    ],
                )
            return result, output_repository

        invalid_result, invalid_repository = await run_resolution(invalid_draft)
        self.assertEqual(invalid_result["resolved_count"], 0)
        self.assertEqual(
            invalid_result["unresolved_item_ids"],
            ["matched:patch:distribution:0"],
        )
        self.assertIsNone(invalid_repository.draft)

        valid_result, valid_repository = await run_resolution(valid_draft)
        self.assertEqual(valid_result["resolved_count"], 1)
        self.assertEqual(valid_result["unresolved_item_ids"], [])
        self.assertEqual(valid_repository.draft, valid_draft)
        self.assertEqual(
            valid_repository.draft["dataset_distribution"][0]["checksum"]["algorithm"],
            {"title": "SHA-1"},
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
                patch={"description": "Updated with chunk evidence.", "keywords": ["chunk-keyword"]},
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
        self.assertEqual(patch, {"description": "Updated with chunk evidence.", "keywords": ["chunk-keyword"]})

    def test_patch_artifact_loaders_return_saved_outputs(self):
        quality_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[],
            summary="Accepted.",
        )
        candidate = PatchCandidate(
            field_path="description",
            patch={"description": "Updated with chunk evidence."},
            confidence=0.9,
            reasoning="Chunk contains an updated description.",
            source_evidence=["Chunk says chunk-keyword."],
        )
        unmapped_fact = UnmappedFact(
            fact="Temperature was 300 K.",
            reason="No matching schema field.",
        )

        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))
            repository.save_patch(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                patch={"description": "Updated with chunk evidence."},
            )
            repository.save_raw_patch(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                patch={"description": "Raw update."},
            )
            repository.save_accepted_patch(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                patch={"description": "Updated with chunk evidence."},
            )
            repository.save_candidates(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                candidates=[candidate],
            )
            repository.save_quality_report(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                quality_report=quality_report,
            )
            repository.save_unmapped_facts(
                workflow_id="package-id",
                patch_file_name="patch_1.json",
                unmapped_facts=[unmapped_fact],
            )

            patches = repository.load_patch_files("package-id")
            quality_reports = repository.load_patch_quality_reports("package-id")
            unmapped_facts = repository.load_unmapped_facts("package-id")

        self.assertEqual(
            [artifact["artifact_type"] for artifact in patches],
            ["accepted_patch", "candidates", "patch", "raw_patch"],
        )
        self.assertEqual(quality_reports[0]["file_name"], "patch_1.quality_report.json")
        self.assertEqual(quality_reports[0]["content"]["summary"], "Accepted.")
        self.assertEqual(
            unmapped_facts[0]["file_name"],
            "patch_1.unmapped_facts.json",
        )
        self.assertEqual(unmapped_facts[0]["fact"], "Temperature was 300 K.")

    def test_patch_review_state_round_trip(self):
        state = {
            "resolved_item_ids": ["patch:description:0"],
            "unmapped_assignments": {"unmapped:1": "description"},
            "resolution_notes": {"patch:description:0": "Reviewed manually."},
            "resolved_at": {"patch:description:0": "2026-05-19T12:00:00.000Z"},
        }
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemExtractionOutputRepository(Path(temporary_directory))

            repository.save_patch_review_state(
                workflow_id="package-id",
                review_state=state,
            )
            loaded = repository.load_patch_review_state("package-id")

        self.assertEqual(loaded, state)

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
        service = FakeExtractionService((PATCHED_DRAFT_OUTPUT, TaskStatus.RUNNING))
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
        self.assertEqual(response.json()["draft"]["description"], "Updated with chunk evidence.")
        self.assertEqual(response.json()["status"], "running")
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

    def test_patch_progress_endpoint_returns_unknown_when_task_is_missing(self):
        service = FakeExtractionService(INITIAL_DRAFT_OUTPUT)
        client = self.make_client(service)

        response = client.get("/api/v1/extraction/patch-draft/package-id/progress")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "unknown", "progress": None})

    def test_patch_artifact_endpoints_return_lists(self):
        service = FakeExtractionService(INITIAL_DRAFT_OUTPUT)
        client = self.make_client(service)

        aggregate = client.get("/api/v1/extraction/patch-draft/package-id/artifacts")
        patches = client.get("/api/v1/extraction/patch-draft/package-id/patches")
        quality_reports = client.get(
            "/api/v1/extraction/patch-draft/package-id/quality-reports"
        )
        unmapped_facts = client.get(
            "/api/v1/extraction/patch-draft/package-id/unmapped-facts"
        )

        self.assertEqual(aggregate.status_code, 200)
        self.assertEqual(
            aggregate.json(),
            {"patches": [], "quality_reports": [], "unmapped_facts": []},
        )
        self.assertEqual(patches.status_code, 200)
        self.assertEqual(patches.json(), [])
        self.assertEqual(quality_reports.status_code, 200)
        self.assertEqual(quality_reports.json(), [])
        self.assertEqual(unmapped_facts.status_code, 200)
        self.assertEqual(unmapped_facts.json(), [])

    def test_patch_review_state_endpoints_round_trip_shape(self):
        service = FakeExtractionService(INITIAL_DRAFT_OUTPUT)
        client = self.make_client(service)
        state = {
            "resolved_item_ids": ["patch:description:0"],
            "unmapped_assignments": {"unmapped:1": "description"},
            "resolution_notes": {"patch:description:0": "Reviewed manually."},
            "resolved_at": {"patch:description:0": "2026-05-19T12:00:00.000Z"},
        }

        saved = client.put(
            "/api/v1/extraction/patch-draft/package-id/review-state",
            json=state,
        )
        loaded = client.get(
            "/api/v1/extraction/patch-draft/package-id/review-state",
        )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json(), state)
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(
            loaded.json(),
            {
                "resolved_item_ids": [],
                "unmapped_assignments": {},
                "resolution_notes": {},
                "resolved_at": {},
            },
        )


if __name__ == "__main__":
    unittest.main()
