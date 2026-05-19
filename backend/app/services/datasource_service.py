from io import BytesIO

from app.core.config import Settings
from app.ollama.client import OllamaClientWrapper
from app.core.task_registry import TaskRegistry, TaskInfo, TaskType, TaskStatus

from app.domain.datasources import DataPackage, FileEntry, ContentChunk, FileEntryNotFoundError
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

    def get_file_entry(self, id: str, file_path: str) -> FileEntry:
        data_package = self.get_data_package(id)
        return data_package.get_file_entry(file_path)

    def get_completed_content_chunks_by_file(
        self,
        data_package_id: str,
    ) -> list[list[ContentChunk]]:
        task_name = f"chunking:file_entries:{data_package_id}"
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status != TaskStatus.COMPLETED:
            return []

        return self._load_content_chunks_by_file(data_package_id)

    def _load_content_chunks_by_file(
        self,
        data_package_id: str,
    ) -> list[list[ContentChunk]]:
        data_package = self.get_data_package(data_package_id)
        content_chunks_by_file: list[list[ContentChunk]] = []
        for file_entry in data_package.files:
            content_chunks = self.blob_repository.load_content_chunks_by_file_path(
                data_package_id,
                file_entry.file_path,
            )
            if content_chunks:
                content_chunks_by_file.append(content_chunks)

        return content_chunks_by_file
    
    async def chunk_file_entries_in_data_package(
        self,
        data_package_id: str,
        buffer_window_size: int,
        embedding_batch_size: int,
        semantic_chunking_threshold: float,
        replace_existing_chunks: bool = False,
    ) -> tuple[list[list[ContentChunk]], TaskStatus]:
        
        TASK_NAME = f"chunking:file_entries:{data_package_id}"

        task_info: TaskInfo | None = self.task_registry.get_task_info(TASK_NAME)

        if task_info is None:
            if not replace_existing_chunks:
                content_chunks_by_file = self._load_content_chunks_by_file(data_package_id)
                if content_chunks_by_file:
                    return content_chunks_by_file, TaskStatus.COMPLETED

            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=replace_existing_chunks,
            )
            return [], TaskStatus.RUNNING

        if replace_existing_chunks and task_info.status != TaskStatus.RUNNING:
            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=True,
            )
            return [], TaskStatus.RUNNING

        if task_info.status == TaskStatus.CANCELLED:
            await self._start_chunking_task(
                task_name=TASK_NAME,
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                delete_existing_chunks=False,
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
            content_chunks = self.blob_repository.load_content_chunks_by_file_path(data_package_id, file_entry.file_path)
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
        embedding_batch_size: int,
        semantic_chunking_threshold: float,
        delete_existing_chunks: bool,
    ) -> None:
        if delete_existing_chunks:
            self.blob_repository.delete_content_chunks(data_package_id)

        await self.task_registry.create_task(
            coro=self._run_chunking_task(
                data_package_id=data_package_id,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold
            ),
            type=TaskType.CHUNKING,
            name=task_name
        )

    # Task runner

    async def _run_chunking_task(self, data_package_id: str, buffer_window_size: int, embedding_batch_size: int, semantic_chunking_threshold: float):
        data_package = self.get_data_package(data_package_id)
        files = data_package.files

        if not files:
            raise FileEntryNotFoundError("No file entries found in the data package.")
        
        for file_entry in files:
            content_chunks = await ContentChunk.create_chunks_for_file_entry(
                data_package_id=data_package_id,
                file_entry=file_entry,
                embedding_func=self.ollama_client.get_embeddings,
                buffer_window_size=buffer_window_size,
                embedding_batch_size=embedding_batch_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
            )
            self.blob_repository.save_content_chunks(content_chunks)
