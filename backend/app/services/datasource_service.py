from functools import partial
from io import BytesIO

from app.core.config import Settings
from app.ollama.client import OllamaClientWrapper
from app.core.task_registry import TaskRegistry, TaskInfo, TaskType, TaskStatus

from app.domain.datasources import DataPackage, FileEntry, ContentChunk, FileEntryNotFoundError
from app.domain.datasources.text_quality import TextQualityConfig, classify_text_line
from app.domain.token_budget import PromptTokenBudgeter
from app.repositories.datasource_blob_repository import DataSourceBlobRepository

class DataSourceService:
    def __init__(self, blob_repository: DataSourceBlobRepository, settings: Settings, ollama_client: OllamaClientWrapper, task_registry: TaskRegistry):
        self.blob_repository = blob_repository
        self.settings = settings
        self.ollama_client = ollama_client
        self.task_registry = task_registry

    def save_data_package(self, data: BytesIO, file_name: str) -> DataPackage:
        
        data_package = DataPackage.from_bytes(data, file_name)

        # Note: file_name in DataPackage is stored without the .zip extension
        self.blob_repository.save_data_package(data, data_package.id, data_package.file_name)
        
        return data_package

    def get_data_package(self, id: str) -> DataPackage:
        return self.blob_repository.load_data_package(id)

    def list_data_packages(self) -> list[DataPackage]:
        return self.blob_repository.list_data_packages()

    def delete_data_package(self, id: str) -> None:
        self.blob_repository.delete_data_package(id)

    @staticmethod
    def chunk_task_name(data_package_id: str) -> str:
        return f"chunking:file_entries:{data_package_id}"

    def get_file_entry(self, id: str, file_path: str) -> FileEntry:
        data_package = self.get_data_package(id)
        return data_package.get_file_entry(file_path)

    def get_content_chunks_by_file(
        self,
        data_package_id: str,
        chunking_strategy: str = "semantic",
    ) -> list[list[ContentChunk]]:
        return self._load_content_chunks_by_file(data_package_id, chunking_strategy)

    def get_completed_content_chunks_by_file(
        self,
        data_package_id: str,
        chunking_strategy: str = "semantic",
    ) -> list[list[ContentChunk]]:
        task_name = self.chunk_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status != TaskStatus.COMPLETED:
            return []

        return self._load_content_chunks_by_file(data_package_id, chunking_strategy)

    def get_chunk_task_status(self, data_package_id: str) -> TaskStatus:
        task_name = self.chunk_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None:
            return task_info.status

        if self._load_content_chunks_by_file(data_package_id, "semantic"):
            return TaskStatus.COMPLETED

        return TaskStatus.UNKNOWN

    def _load_content_chunks_by_file(
        self,
        data_package_id: str,
        chunking_strategy: str = "semantic",
    ) -> list[list[ContentChunk]]:
        data_package = self.get_data_package(data_package_id)
        content_chunks_by_file: list[list[ContentChunk]] = []
        for file_entry in data_package.files:
            if not file_entry.is_chunkable():
                continue
            content_chunks = self.blob_repository.load_content_chunks_by_file_path(
                data_package_id,
                file_entry.file_path,
                chunking_strategy,
            )
            if content_chunks:
                content_chunks_by_file.append(content_chunks)

        return content_chunks_by_file
    
    async def chunk_file_entries_in_data_package(
        self,
        data_package_id: str,
        buffer_window_size: int,
        semantic_chunking_threshold: float,
        replace_existing_chunks: bool = False,
        protected_line_indices: dict[str, list[int]] | None = None,
        text_quality_config: TextQualityConfig | None = None,
        embedding_num_gpu: int | None = None,
        chunking_strategy: str = "semantic",
        fixed_tokens_per_chunk: int = 1024,
        min_tokens_per_chunk: int = 128,
        max_tokens_per_chunk: int = 1024,
    ) -> tuple[list[list[ContentChunk]], TaskStatus]:
        if min_tokens_per_chunk > max_tokens_per_chunk:
            raise ValueError("min_tokens_per_chunk must be less than or equal to max_tokens_per_chunk.")
        
        TASK_NAME = self.chunk_task_name(data_package_id)

        task_info: TaskInfo | None = self.task_registry.get_task_info(TASK_NAME)

        if task_info is None:
            if not replace_existing_chunks:
                content_chunks_by_file = self._load_content_chunks_by_file(
                    data_package_id,
                    chunking_strategy,
                )
                if content_chunks_by_file:
                    return content_chunks_by_file, TaskStatus.COMPLETED

            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=replace_existing_chunks,
                protected_line_indices=protected_line_indices,
                text_quality_config=text_quality_config,
                embedding_num_gpu=embedding_num_gpu,
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
            return [], TaskStatus.RUNNING

        if replace_existing_chunks and task_info.status != TaskStatus.RUNNING:
            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=True,
                protected_line_indices=protected_line_indices,
                text_quality_config=text_quality_config,
                embedding_num_gpu=embedding_num_gpu,
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
            return [], TaskStatus.RUNNING

        if task_info.status == TaskStatus.CANCELLED:
            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=False,
                protected_line_indices=protected_line_indices,
                text_quality_config=text_quality_config,
                embedding_num_gpu=embedding_num_gpu,
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
            return [], TaskStatus.RUNNING        
        
        elif task_info.status == TaskStatus.CRASHED:
            exception = task_info.task.exception()
            raise exception if exception else Exception("Chunking task crashed without an exception.")
        
        if task_info.status != TaskStatus.COMPLETED and task_info.status != TaskStatus.RUNNING:
            raise Exception(f"Unexpected task status: {task_info.status}")
        
        # From here the task is either RUNNING or COMPLETED
        
        data_package = self.get_data_package(data_package_id)
        files = data_package.files

        if not files:
            raise FileEntryNotFoundError("No file entries found in the data package.")

        content_chunks_by_file: list[list[ContentChunk]] = []        

        for file_entry in files:
            if not file_entry.is_chunkable():
                continue
            content_chunks = self.blob_repository.load_content_chunks_by_file_path(
                data_package_id,
                file_entry.file_path,
                chunking_strategy,
            )
            if not content_chunks:
                continue
            content_chunks_by_file.append(content_chunks)

        return content_chunks_by_file, task_info.status

    async def _start_chunking_task(
        self,
        *,
        task_name: str,
        data_package_id: str,
        buffer_window_size: int,
        semantic_chunking_threshold: float,
        delete_existing_chunks: bool,
        protected_line_indices: dict[str, list[int]] | None = None,
        text_quality_config: TextQualityConfig | None = None,
        embedding_num_gpu: int | None = None,
        chunking_strategy: str = "semantic",
        fixed_tokens_per_chunk: int = 1024,
        min_tokens_per_chunk: int = 128,
        max_tokens_per_chunk: int = 1024,
    ) -> None:
        if delete_existing_chunks:
            self.blob_repository.delete_content_chunks(data_package_id, chunking_strategy)

        await self.task_registry.create_task(
            coro=self._run_chunking_task(
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                protected_line_indices=protected_line_indices,
                text_quality_config=text_quality_config,
                embedding_num_gpu=embedding_num_gpu,
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
            ),
            type=TaskType.CHUNKING,
            name=task_name
        )

    # Task runner

    async def _run_chunking_task(
        self,
        data_package_id: str,
        buffer_window_size: int,
        semantic_chunking_threshold: float,
        protected_line_indices: dict[str, list[int]] | None = None,
        text_quality_config: TextQualityConfig | None = None,
        embedding_num_gpu: int | None = None,
        chunking_strategy: str = "semantic",
        fixed_tokens_per_chunk: int = 1024,
        min_tokens_per_chunk: int = 128,
        max_tokens_per_chunk: int = 1024,
    ):
        data_package = self.get_data_package(data_package_id)
        files = data_package.files

        if not files:
            raise FileEntryNotFoundError("No file entries found in the data package.")

        # Bake the config into the classification function so the chunking
        # domain layer stays free of config-awareness.
        if text_quality_config is not None:
            classification_func = partial(classify_text_line, config=text_quality_config)
        else:
            classification_func = classify_text_line
        embedding_func = (
            partial(self.ollama_client.get_embeddings, num_gpu=embedding_num_gpu)
            if embedding_num_gpu is not None
            else self.ollama_client.get_embeddings
        )
        token_budgeter = PromptTokenBudgeter.from_tokenizer_source(
            getattr(self.settings, "ollama_chat_tokenizer", ""),
            hf_token=getattr(self.settings, "hf_token", ""),
        )
        
        for file_entry in files:
            if not file_entry.is_chunkable():
                continue
            file_protected = protected_line_indices.get(file_entry.file_path) if protected_line_indices else None
            content_chunks = await ContentChunk.create_chunks_for_file_entry(
                data_package_id=data_package_id,
                file_entry=file_entry,
                embedding_func=embedding_func,
                text_classification_func=classification_func,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=self.settings.embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                protected_line_indices=file_protected,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
                token_budgeter=token_budgeter,
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
            )
            self.blob_repository.save_content_chunks(content_chunks, chunking_strategy)


