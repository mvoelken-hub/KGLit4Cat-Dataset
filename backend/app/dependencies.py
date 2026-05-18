from app.core.config import Settings, settings
from app.core.task_registry import TaskRegistry, task_registry
from app.neo4j.driver import Neo4jDriver, neo4j_driver
from app.ollama.client import OllamaClientWrapper, ollama_client

def get_settings() -> Settings:
    return settings


def get_task_registry() -> TaskRegistry:
    return task_registry


def get_neo4j_driver() -> Neo4jDriver:
    return neo4j_driver


def get_ollama_client() -> OllamaClientWrapper:
    return ollama_client

from infra.filesystem_datasource_blob_repository import FileSystemDataSourceBlobRepository
from app.services.datasource_service import DataSourceService


datasource_blob_repository = FileSystemDataSourceBlobRepository(settings.uploads_dir)
datasource_service = DataSourceService(
    datasource_blob_repository,
    settings,
    ollama_client,
    task_registry
)

def get_datasource_service() -> DataSourceService:
    return datasource_service


from infra.filesystem_extraction_profile_repository import FileSystemExtractionProfileRepository
from app.services.extraction_service import ExtractionService

extraction_profile_repository = FileSystemExtractionProfileRepository(settings.dcat_profiles_dir)
extraction_service = ExtractionService(
    extraction_profile_repository,
    settings,
)

def get_extraction_service() -> ExtractionService:
    return extraction_service


from infra.neo4j_semantic_graph_repository import Neo4jSemanticGraphRepository
from app.services.semantic_service import SemanticService

semantic_graph_repository = Neo4jSemanticGraphRepository(neo4j_driver, ollama_client)
semantic_service = SemanticService(
    semantic_graph_repository,
    settings,
    ollama_client,
    task_registry
)

def get_semantic_service() -> SemanticService:
    return semantic_service
