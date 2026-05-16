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
    VocabAlreadyExistsError
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
        return vocab_scheme_info
    
    async def get_vocabulary(self, identifier: str) -> VocabSchemeInfo | None:
        return await self.semantic_graph_repository.get_vocabulary(identifier)