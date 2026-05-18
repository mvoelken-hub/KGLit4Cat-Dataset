from pydantic import HttpUrl
from fastapi import UploadFile


from app.core.config import Settings
from app.ollama.client import OllamaClientWrapper
from app.core.task_registry import TaskRegistry, TaskInfo, TaskType, TaskStatus

from app.domain.semantics import (
    VocabSchemeInfo,
    SerializedRdfGraph,
    LoadedRdfGraph,
    load_rdf_graph,
    VocabAlreadyExistsError,
    VocabResource
)
from app.repositories.semantic_graph_repository import SemanticGraphRepository

class SemanticService:

    def __init__(self, semantic_graph_repository: SemanticGraphRepository, settings: Settings, ollama_client: OllamaClientWrapper, task_registry: TaskRegistry):
        self.semantic_graph_repository = semantic_graph_repository
        self.settings = settings
        self.ollama_client = ollama_client
        self.task_registry = task_registry

    async def import_vocabulary(self, rdf_source: HttpUrl | UploadFile, identifier: str) -> VocabSchemeInfo:

        existing_identifiers = await self.semantic_graph_repository.list_vocabulary_identifiers()
        if identifier in existing_identifiers:
            raise VocabAlreadyExistsError(f"A vocabulary with identifier '{identifier}' already exists.")

        _rdf_source: HttpUrl | SerializedRdfGraph
        if isinstance(rdf_source, UploadFile):
            file_content = await rdf_source.read()
            _rdf_source = SerializedRdfGraph(
                data=file_content,
                file_name=rdf_source.filename,
            )
        else:
            _rdf_source = rdf_source

        loaded_rdf_graph = await load_rdf_graph(_rdf_source, identifier)
        vocab_scheme_info = VocabSchemeInfo.from_loaded_graph(loaded_rdf_graph)

        await self.semantic_graph_repository.import_vocabulary(vocab_scheme_info, loaded_rdf_graph.graph)
        await self._create_vocab_indexes(identifier)

        vocab_info_with_terms = await self.get_vocabulary(identifier)
        if vocab_info_with_terms is None:
            raise Exception("Failed to retrieve vocabulary after import.")
        
        return vocab_info_with_terms
    
    async def get_vocabulary(self, identifier: str) -> VocabSchemeInfo | None:
        return await self.semantic_graph_repository.get_vocabulary(identifier)

    async def list_vocabularies(self) -> list[str]:
        return await self.semantic_graph_repository.list_vocabulary_identifiers()
    
    async def delete_vocabulary(self, identifier: str) -> None:
        vocab_info = await self.get_vocabulary(identifier)
        if not vocab_info:
            raise ValueError(f"Vocabulary with identifier '{identifier}' not found.")

        # Note: Indexes must be deleted before the vocbulary nodes
        await self._delete_vocab_indexes(identifier)
        await self.semantic_graph_repository.delete_vocabulary(identifier)        

    async def generate_embeddings_for_vocabulary(self, identifier: str) -> tuple[int, TaskStatus]:
        vocab_scheme_info = await self.get_vocabulary(identifier)
        if not vocab_scheme_info:
            raise ValueError(f"Vocabulary with identifier '{identifier}' not found.")
        
        TASK_NAME = f"embedding:vocab:{identifier}"

        task_info: TaskInfo | None = self.task_registry.get_task_info(TASK_NAME)

        pending_updates = await self.semantic_graph_repository.check_pending_embedding_updates(identifier)

        if not task_info or task_info.status == TaskStatus.CANCELLED:
            await self.task_registry.create_task(
                coro=self._run_embedding_generation(pending_vocab_resources=pending_updates),
                type=TaskType.EMBEDDING,
                name=TASK_NAME
            )
            return len(pending_updates), TaskStatus.RUNNING
        
        elif task_info.status == TaskStatus.CRASHED:
            exception = task_info.task.exception()
            raise exception if exception else Exception("Embedding generation task crashed without an exception.")
        
        else:
            return len(pending_updates), task_info.status
        

    # task runners

    async def _run_embedding_generation(self, pending_vocab_resources: list[VocabResource]) -> None:

        batch_size = self.settings.embedding_batch_size
        
        for i in range(0, len(pending_vocab_resources), batch_size):
            batch = pending_vocab_resources[i:i+batch_size]
            texts = [resource.to_embedding_str() for resource in batch]
            embeddings = await self.ollama_client.get_embeddings(texts)
            for resource, embedding in zip(batch, embeddings):
                resource.embedding = embedding
            await self.semantic_graph_repository.update_resource_embeddings(batch)
    
    # internal helpers

    async def _create_vocab_indexes(self, identifier: str) -> None:
        await self.semantic_graph_repository.create_vocab_indexes(identifier)
        
    async def _delete_vocab_indexes(self, identifier: str) -> None:
        await self.semantic_graph_repository.delete_vocab_indexes(identifier)

