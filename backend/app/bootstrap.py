import asyncio
from enum import Enum
from logging import Logger

from app.core.initial_vocabs import INITIAL_VOCABS
from app.neo4j.driver import Neo4jDriver
from app.ollama.client import OllamaClientWrapper

from app.core.config import Settings
from app.core.task_registry import TaskRegistry, TaskType, TaskStatus

from app.services.semantic_service import SemanticService
from app.domain.semantics import VocabAlreadyExistsError

class InitialTasks(Enum):
    PULL_OLLAMA_MODEL = "startup:pull_ollama_models:01"
    BOOTSTRAP_INITIAL_VOCABS = "startup:import_initial_vocabs:01"


async def pull_ollama_models(ollama_client: OllamaClientWrapper, settings: Settings):
    await ollama_client.pull_models([
        settings.ollama_embed_model,
        settings.ollama_chat_model,
    ])


async def import_initial_vocab(semantic_service: SemanticService, logger: Logger | None = None):
    for vocab in INITIAL_VOCABS:
        if logger:
            logger.info("Importing initial vocabulary %s from %s", vocab.identifier, vocab.rdf_source)
        try:
            await semantic_service.import_vocabulary(rdf_source=vocab.rdf_source, identifier=vocab.identifier)
        except VocabAlreadyExistsError:
            if logger:
                logger.info("Initial vocabulary already exists: %s", vocab.identifier)

        pending_updates, status = await semantic_service.generate_embeddings_for_vocabulary(vocab.identifier)
        while status == TaskStatus.RUNNING:
            if logger:
                logger.info("Generating embeddings for %s. Pending updates: %s", vocab.identifier, pending_updates)
            await asyncio.sleep(10)
            pending_updates, status = await semantic_service.generate_embeddings_for_vocabulary(vocab.identifier)
        if logger:
            logger.info("Finished initial vocabulary %s with embedding status %s", vocab.identifier, status.value)
            


async def start_setup(
    settings: Settings,
    logger: Logger,
    ollama_client: OllamaClientWrapper,
    neo4j_driver: Neo4jDriver,
    task_registry: TaskRegistry,
    semantic_service: SemanticService
) -> None:

    # Pull Ollama models

    if settings.skip_model_pull:
        logger.info("Skipping Ollama model pull because skip_model_pull is enabled.")
    else:
        await task_registry.create_task(
            pull_ollama_models(ollama_client=ollama_client, settings=settings),
            name=InitialTasks.PULL_OLLAMA_MODEL.value,
            type=TaskType.STARTUP
        )

    # Neo4j connection and Graph bootstrap

    await neo4j_driver.wait_for_connection(60.0, 5.0)
    logger.info("Removing Resource-only semantic graph nodes.")
    await semantic_service.cleanup_untyped_resources()

    if settings.skip_initial_vocab_import:
        logger.info("Skipping initial vocabulary import because skip_initial_vocab_import is enabled.")
    else:
        await task_registry.create_task(
            import_initial_vocab(semantic_service, logger=logger),
            name=InitialTasks.BOOTSTRAP_INITIAL_VOCABS.value,
            type=TaskType.STARTUP
        )


async def run_initial_vocab_bootstrap() -> None:
    from app.core.config import settings
    from app.core.logging import logger, setup_logging
    from app.core.task_registry import task_registry
    from app.dependencies import get_semantic_service
    from app.neo4j.driver import neo4j_driver
    from app.ollama.client import ollama_client

    setup_logging(runtime_dir=settings.runtime_dir)
    logger.info("Starting initial vocabulary bootstrap.")
    try:
        await neo4j_driver.wait_for_connection(60.0, 5.0)
        semantic_service = get_semantic_service()
        logger.info("Removing Resource-only semantic graph nodes.")
        await semantic_service.cleanup_untyped_resources()
        await import_initial_vocab(semantic_service, logger=logger)
        logger.info("Initial vocabulary bootstrap completed.")
    finally:
        await task_registry.cancel_all_tasks()
        await neo4j_driver.close()
        await ollama_client.close()


def main() -> None:
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "initial-vocabs":
        asyncio.run(run_initial_vocab_bootstrap())
        return

    raise SystemExit("Usage: python -m app.bootstrap initial-vocabs")


if __name__ == "__main__":
    main()
