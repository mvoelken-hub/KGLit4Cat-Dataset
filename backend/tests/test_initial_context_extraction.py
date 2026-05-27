import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.task_registry import TaskRegistry, TaskStatus
from app.domain.datasources import ContentChunk, DataPackage, FileEntry
from app.domain.extraction import (
    ChunkingRequiredError,
    ExtractionContext,
    ExtractionRunResult,
    FileRankingResult,
    RankedFile,
)
from app.ollama.completion import CompletionResult
from app.ollama.usage import RunUsage
from app.services.extraction_service import ExtractionService


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def exception(self, *_args, **_kwargs):
        pass


class FakeDataSourceService:
    def __init__(self, chunks_by_file: list[list[ContentChunk]]):
        self.data_package = DataPackage(
            file_name="package",
            files=[
                FileEntry(
                    file_path="README.md",
                    file_name="README.md",
                    file_extension=".md",
                    raw_content=b"metadata",
                )
            ],
        )
        self.chunks_by_file = chunks_by_file

    def get_data_package(self, _data_package_id: str) -> DataPackage:
        return self.data_package

    def get_completed_content_chunks_by_file(self, _data_package_id: str):
        return self.chunks_by_file


class FakeProfileService:
    schema = {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "string"}},
        "additionalProperties": True,
    }

    def get_profile(self, identifier: str):
        return SimpleNamespace(identifier=identifier, target_class="Dataset")

    def load_json_schema(self, _identifier: str):
        return self.schema

    def validate_document(self, *, identifier: str, document: dict):
        errors = [] if isinstance(document.get("id"), str) else [SimpleNamespace(path="$.id", message="required")]
        return SimpleNamespace(valid=not errors, errors=errors)


class FakeOutputRepository:
    def __init__(self):
        self.context: ExtractionContext | None = None
        self.contexts: list[ExtractionContext] = []
        self.result: ExtractionRunResult | None = None
        self.warnings: list[str] = []
        self.token_usage: dict[str, dict[str, int]] = {}

    def save_extraction_context(self, *, workflow_id: str, extraction_context: ExtractionContext):
        self.context = extraction_context
        self.contexts.append(extraction_context)

    def load_extraction_context(self, workflow_id: str) -> ExtractionContext:
        if self.context is None:
            raise FileNotFoundError
        return self.context

    def save_extraction_result(self, *, workflow_id: str, result: ExtractionRunResult):
        self.result = result

    def load_extraction_result(self, workflow_id: str) -> ExtractionRunResult:
        if self.result is None:
            raise FileNotFoundError
        return self.result

    def save_extraction_warnings(self, *, workflow_id: str, warnings: list[str]):
        self.warnings = warnings

    def load_extraction_warnings(self, workflow_id: str) -> list[str]:
        return self.warnings

    def save_token_usage(self, *, workflow_id: str, token_usage: dict[str, dict[str, int]]):
        self.token_usage = token_usage

    def load_token_usage(self, workflow_id: str) -> dict[str, dict[str, int]]:
        return self.token_usage

    def clear_extraction_run(self, workflow_id: str):
        self.context = None
        self.contexts = []
        self.result = None
        self.warnings = []
        self.token_usage = {}


def make_chunk(start_idx: int = 0, content: str = "sample measured at 20 C") -> ContentChunk:
    return ContentChunk(
        content=content,
        data_package_id="package-id",
        file_path="README.md",
        start_idx=start_idx,
        end_idx=start_idx,
    )


def make_service(chunks_by_file: list[list[ContentChunk]]):
    task_registry = TaskRegistry(SimpleNamespace(), FakeLogger())  # type: ignore[arg-type]
    output_repository = FakeOutputRepository()
    service = ExtractionService(
        FakeProfileService(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        FakeDataSourceService(chunks_by_file),  # type: ignore[arg-type]
        SimpleNamespace(chat_model="chat", max_context_length=4096),  # type: ignore[arg-type]
        output_repository,
        task_registry,
        semantic_service=None,
    )
    return service, task_registry, output_repository


class ExtractionServiceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_extraction_requires_completed_chunks(self):
        service, _, _ = make_service([])

        with self.assertRaises(ChunkingRequiredError):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )

    async def test_run_extraction_processes_task_and_persists_interim_context(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        outputs = [
            CompletionResult(
                output=FileRankingResult(files=[RankedFile(rank=1, file_path="README.md")]),
                usage=RunUsage(requests=1, input_tokens=10, output_tokens=2),
            ),
            CompletionResult(
                output=ExtractionContext.model_validate(
                    {
                        "datasets": [
                            {
                                "identifier": "dataset-one",
                                "description": "First partial dataset.",
                                "keywords": ["one"],
                            }
                        ]
                    }
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            CompletionResult(
                output=ExtractionContext.model_validate(
                    {
                        "datasets": [
                            {
                                "identifier": "dataset-two",
                                "description": "Second partial dataset.",
                                "keywords": ["two"],
                            }
                        ]
                    }
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            ),
        ]

        async def fake_generate(*_args, **_kwargs):
            return outputs.pop(0)

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.context)
        self.assertGreaterEqual(len(output_repository.contexts), 2)
        self.assertEqual(output_repository.contexts[0].datasets[0].identifier, "dataset-one")
        self.assertEqual(len(output_repository.contexts[1].datasets), 2)
        self.assertIsNotNone(output_repository.result)
        self.assertEqual(output_repository.result.document["id"], "dataset")
        self.assertIn("chunk_extraction", output_repository.token_usage)

    async def test_progress_returns_persisted_interim_context(self):
        service, _, output_repository = make_service([[make_chunk()]])
        output_repository.save_extraction_context(
            workflow_id="package-id",
            extraction_context=ExtractionContext.model_validate(
                {
                    "datasets": [
                        {
                            "identifier": "interim-dataset",
                            "description": "Persisted partial context.",
                        }
                    ]
                }
            ),
        )

        status, progress = await service.get_extraction_progress(
            data_package_id="package-id",
        )

        self.assertEqual(status, TaskStatus.UNKNOWN)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "interim_context")
        self.assertIsNotNone(progress.interim_context)
        self.assertEqual(
            progress.interim_context.datasets[0].identifier,
            "interim-dataset",
        )


if __name__ == "__main__":
    unittest.main()
