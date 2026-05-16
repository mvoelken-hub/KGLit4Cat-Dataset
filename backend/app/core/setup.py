from logging import Logger
from typing import Literal

from app.neo4j.driver import Neo4jDriver
from app.ollama.client import OllamaClientWrapper

from app.core.config import Settings
from app.core.task_registry import TaskRegistry, TaskType


async def pull_ollama_models(ollama_client: OllamaClientWrapper, settings: Settings):
    await ollama_client.pull_models([
        settings.ollama_embed_model,
        settings.ollama_chat_model,
    ])

async def load_ollama_models(ollama_client: OllamaClientWrapper, model: Literal["embedding", "chat", "both", "none"] = "none"):
    if model in ("embedding", "both"):
        await ollama_client.verify_embedding()
    if model in ("chat", "both"):
        await ollama_client.verify_chat()

async def import_inital_vocab():    
    pass


async def start_setup(
    settings: Settings,
    logger: Logger,
    ollama_client: OllamaClientWrapper,
    neo4j_driver: Neo4jDriver,
    task_registry: TaskRegistry
) -> None:

    # Ollama bootstrap: Pull models and optionally load them into memory on startup. This ensures that the models are ready to use when the first requests arrive, avoiding latency spikes due to model loading during request handling.

    if settings.skip_model_pull:
        logger.info("Skipping Ollama model pull because skip_model_pull is enabled.")
    else:
        await task_registry.create_task(pull_ollama_models(ollama_client=ollama_client, settings=settings), name="startup:pull_ollama_models:01", type=TaskType.STARTUP)
        await pull_ollama_models(ollama_client=ollama_client, settings=settings)

    if settings.load_ollama_models_on_startup:
        await task_registry.create_task(load_ollama_models(ollama_client=ollama_client, model="both"), name="startup:load_ollama_models:01", type=TaskType.STARTUP)
    else:
        logger.info("Skipping Ollama model warm-up because load_ollama_models_on_startup is disabled.")

    # Neo4j connection and bootstrap: Wait for Neo4j connection and perform any necessary startup tasks like creating constraints or indexes. This ensures that the database is ready to handle queries when the first requests arrive.

    await neo4j_driver.wait_for_connection(100.0, 2.0)

    if settings.skip_initial_vocab_import:
        logger.info("Skipping initial vocabulary import because skip_initial_vocab_import is enabled.")
    else:
        await task_registry.create_task(import_inital_vocab(), name="startup:import_initial_vocabs:01", type=TaskType.STARTUP)
