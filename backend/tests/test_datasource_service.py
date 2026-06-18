from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.core.task_registry import TaskStatus
from app.domain.datasources import ContentChunk, DataPackage, FileEntry
from app.services.datasource_service import DataSourceService


def make_data_package() -> DataPackage:
    return DataPackage(
        file_name="package",
        files=[
            FileEntry(
                file_path="metadata.txt",
                file_name="metadata.txt",
                file_extension=".txt",
                raw_content=b"Instrument: GC-42\n",
            )
        ],
    )


def make_chunk() -> ContentChunk:
    return ContentChunk(
        content="Instrument: GC-42\n",
        data_package_id="package-id",
        file_path="metadata.txt",
        start_idx=0,
        end_idx=0,
    )


class FakeBlobRepository:
    def __init__(self, chunks_by_file_path: dict[str, list[ContentChunk]]):
        self.data_package = make_data_package()
        self.chunks_by_file_path = chunks_by_file_path
        self.loaded_chunk_paths: list[str] = []
        self.deleted_chunk_package_ids: list[str] = []
        self.deleted_chunk_strategies: list[str | None] = []
        self.saved_chunks: list[ContentChunk] = []
        self.saved_chunk_strategies: list[str] = []

    def load_data_package(self, id: str) -> DataPackage:
        return self.data_package

    def load_content_chunks_by_file_path(
        self,
        data_package_id: str,
        file_path: str,
        chunking_strategy: str = "semantic",
    ) -> list[ContentChunk]:
        self.loaded_chunk_paths.append(file_path)
        return self.chunks_by_file_path.get(file_path, [])

    def delete_content_chunks(self, data_package_id: str, chunking_strategy: str | None = None) -> None:
        self.deleted_chunk_package_ids.append(data_package_id)
        self.deleted_chunk_strategies.append(chunking_strategy)
        self.chunks_by_file_path.clear()

    def save_content_chunks(self, content_chunks: list[ContentChunk], chunking_strategy: str = "semantic") -> None:
        self.saved_chunks.extend(content_chunks)
        self.saved_chunk_strategies.append(chunking_strategy)


class FakeOllamaClient:
    def __init__(self):
        self.embedding_num_gpu_values: list[int | None] = []

    async def get_embeddings(self, input: list[str], num_gpu: int | None = None):
        self.embedding_num_gpu_values.append(num_gpu)
        return [[float(index), 1.0] for index, _ in enumerate(input)]


class FakeTaskRegistry:
    def __init__(self, status: TaskStatus | None):
        self.status = status
        self.requested_name: str | None = None
        self.created_task_names: list[str] = []

    def get_task_info(self, name: str):
        self.requested_name = name
        if self.status is None:
            return None
        return SimpleNamespace(status=self.status)

    async def create_task(self, coro, type, name: str):
        self.created_task_names.append(name)
        coro.close()
        return SimpleNamespace()


class FakeSettings:
    embedding_batch_size = 32
    max_context_length = 4096
    ollama_chat_tokenizer = ""


class DataSourceServiceTests(unittest.IsolatedAsyncioTestCase):
    def make_service(
        self,
        *,
        chunks_by_file_path: dict[str, list[ContentChunk]],
        task_status: TaskStatus | None,
        ollama_client=None,
    ) -> tuple[DataSourceService, FakeBlobRepository, FakeTaskRegistry]:
        blob_repository = FakeBlobRepository(chunks_by_file_path)
        task_registry = FakeTaskRegistry(status=task_status)
        service = DataSourceService(
            blob_repository=blob_repository,  # type: ignore[arg-type]
            settings=FakeSettings(),  # type: ignore[arg-type]
            ollama_client=ollama_client,  # type: ignore[arg-type]
            task_registry=task_registry,  # type: ignore[arg-type]
        )
        return service, blob_repository, task_registry

    def test_completed_chunks_can_be_loaded_after_task_registry_restart(self):
        chunk = make_chunk()
        service, _, task_registry = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=None,
        )

        result = service.get_completed_content_chunks_by_file("package-id")

        self.assertEqual(result, [[chunk]])
        self.assertEqual(
            task_registry.requested_name,
            "chunking:file_entries:package-id",
        )

    def test_running_chunk_task_blocks_persisted_chunks(self):
        chunk = make_chunk()
        service, blob_repository, _ = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=TaskStatus.RUNNING,
        )

        result = service.get_completed_content_chunks_by_file("package-id")

        self.assertEqual(result, [])
        self.assertEqual(blob_repository.loaded_chunk_paths, [])

    def test_completed_chunk_task_loads_persisted_chunks(self):
        chunk = make_chunk()
        service, _, _ = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=TaskStatus.COMPLETED,
        )

        result = service.get_completed_content_chunks_by_file("package-id")

        self.assertEqual(result, [[chunk]])

    async def test_chunk_request_returns_persisted_chunks_after_task_registry_restart(self):
        chunk = make_chunk()
        service, _, task_registry = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=None,
        )

        result, status = await service.chunk_file_entries_in_data_package(
            data_package_id="package-id",
            buffer_window_size=1,
            semantic_chunking_threshold=95.0,
        )

        self.assertEqual(result, [[chunk]])
        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertEqual(task_registry.created_task_names, [])

    async def test_chunk_request_starts_task_when_registry_missing_and_no_chunks(self):
        service, _, task_registry = self.make_service(
            chunks_by_file_path={},
            task_status=None,
        )

        result, status = await service.chunk_file_entries_in_data_package(
            data_package_id="package-id",
            buffer_window_size=1,
            semantic_chunking_threshold=95.0,
        )

        self.assertEqual(result, [])
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(
            task_registry.created_task_names,
            ["chunking:file_entries:package-id"],
        )

    async def test_chunking_task_passes_embedding_gpu_override_to_ollama(self):
        ollama_client = FakeOllamaClient()
        service, blob_repository, _ = self.make_service(
            chunks_by_file_path={},
            task_status=None,
            ollama_client=ollama_client,
        )
        blob_repository.data_package.files[0].raw_content = "\n".join(
            f"measurement line {index} contains useful catalyst metadata"
            for index in range(110)
        ).encode()

        await service._run_chunking_task(
            data_package_id="package-id",
            buffer_window_size=1,
            semantic_chunking_threshold=95.0,
            embedding_num_gpu=0,
        )

        self.assertTrue(blob_repository.saved_chunks)
        self.assertEqual(set(ollama_client.embedding_num_gpu_values), {0})

    async def test_chunking_task_uses_explicit_token_bounds(self):
        ollama_client = FakeOllamaClient()
        service, blob_repository, _ = self.make_service(
            chunks_by_file_path={},
            task_status=None,
            ollama_client=ollama_client,
        )
        captured: dict[str, object] = {}

        async def fake_create_chunks(**kwargs):
            captured.update(kwargs)
            return [make_chunk()]

        with patch.object(
            ContentChunk,
            "create_chunks_for_file_entry",
            side_effect=fake_create_chunks,
        ):
            await service._run_chunking_task(
                data_package_id="package-id",
                buffer_window_size=1,
                semantic_chunking_threshold=95.0,
                min_tokens_per_chunk=64,
                max_tokens_per_chunk=1500,
            )

        self.assertEqual(captured["min_tokens_per_chunk"], 64)
        self.assertEqual(captured["max_tokens_per_chunk"], 1500)
        self.assertIn("token_budgeter", captured)

    async def test_chunk_request_replace_existing_starts_task_and_deletes_chunks(self):
        chunk = make_chunk()
        service, blob_repository, task_registry = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=None,
        )

        result, status = await service.chunk_file_entries_in_data_package(
            data_package_id="package-id",
            buffer_window_size=1,
            semantic_chunking_threshold=95.0,
            replace_existing_chunks=True,
        )

        self.assertEqual(result, [])
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(blob_repository.deleted_chunk_package_ids, ["package-id"])
        self.assertEqual(
            task_registry.created_task_names,
            ["chunking:file_entries:package-id"],
        )

    async def test_chunk_request_replace_existing_restarts_completed_task(self):
        chunk = make_chunk()
        service, blob_repository, task_registry = self.make_service(
            chunks_by_file_path={"metadata.txt": [chunk]},
            task_status=TaskStatus.COMPLETED,
        )

        result, status = await service.chunk_file_entries_in_data_package(
            data_package_id="package-id",
            buffer_window_size=1,
            semantic_chunking_threshold=95.0,
            replace_existing_chunks=True,
        )

        self.assertEqual(result, [])
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(blob_repository.deleted_chunk_package_ids, ["package-id"])
        self.assertEqual(
            task_registry.created_task_names,
            ["chunking:file_entries:package-id"],
        )


if __name__ == "__main__":
    unittest.main()
