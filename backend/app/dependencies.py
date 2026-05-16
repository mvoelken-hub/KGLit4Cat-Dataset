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
