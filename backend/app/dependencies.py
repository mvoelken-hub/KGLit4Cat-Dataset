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


datasource_blob_repository = FileSystemDataSourceBlobRepository(
    settings.uploads_dir,
    settings.output_dir,
)
datasource_service = DataSourceService(
    datasource_blob_repository,
    settings,
    ollama_client,
    task_registry
)

def get_datasource_service() -> DataSourceService:
    return datasource_service


from infra.filesystem_profile_repository import FileSystemProfileRepository
from infra.filesystem_extraction_output_repository import FileSystemExtractionOutputRepository
from app.services.extraction_service import ExtractionService
from app.services.profile_service import ProfileService
from infra.neo4j_semantic_graph_repository import Neo4jSemanticGraphRepository
from app.services.semantic_service import SemanticService

profile_repository = FileSystemProfileRepository(settings.dcat_profiles_dir)
profile_service = ProfileService(profile_repository)
extraction_output_repository = FileSystemExtractionOutputRepository(settings.output_dir)

semantic_graph_repository = Neo4jSemanticGraphRepository(neo4j_driver, ollama_client)
semantic_service = SemanticService(
    semantic_graph_repository,
    settings,
    ollama_client,
    task_registry
)

extraction_service = ExtractionService(
    profile_service,
    settings,
    datasource_service,
    ollama_client,
    extraction_output_repository,
    task_registry,
    semantic_service,
)

def get_profile_service() -> ProfileService:
    return profile_service

def get_extraction_service() -> ExtractionService:
    return extraction_service


def get_semantic_service() -> SemanticService:
    return semantic_service
