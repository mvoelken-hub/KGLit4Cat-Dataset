from logging import Logger
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from app.neo4j.driver import Neo4jDriver
from app.ollama.client import OllamaClientWrapper

from app.core.config import Settings
from app.core.task_registry import TaskRegistry, TaskType, TaskStatus

from app.services.semantic_service import SemanticService
from app.domain.semantics import VocabAlreadyExistsError

class InitialVocab(BaseModel):
    rdf_source: HttpUrl = Field(..., description="The URL of the RDF source to import.")
    identifier: str = Field(..., description="A unique identifier for the vocab, used for referencing it in the system.")


INITIAL_VOCABS = [
    InitialVocab(
        rdf_source=HttpUrl("https://nfdi4cat.github.io/voc4cat/v2026-02-24/voc4cat.ttl"),
        identifier="https://w3id.org/nfdi4cat/voc4cat",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/quantitykind/"),
        identifier="http://qudt.org/vocab/quantitykind",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/unit/"),
        identifier="http://qudt.org/vocab/unit",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/constant"),
        identifier="http://qudt.org/vocab/constant",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://raw.githubusercontent.com/rsc-ontologies/rsc-cmo/master/chmo.owl"),
        identifier="http://purl.obolibrary.org/obo/chmo.owl",
    ),
    InitialVocab(
        rdf_source=HttpUrl("http://nmrML.org/nmrCV.owl"),
        identifier="http://nmrML.org/nmrCV",
    ),
]



async def pull_ollama_models(ollama_client: OllamaClientWrapper, settings: Settings):
    await ollama_client.pull_models([
        settings.ollama_embed_model,
        settings.ollama_chat_model,
    ])

async def load_ollama_models(ollama_client: OllamaClientWrapper, task_registry: TaskRegistry, model: Literal["embedding", "chat", "both", "none"] = "none"):

    pull_models_task = task_registry.get_task_info("startup:pull_ollama_models:01")
    if pull_models_task and pull_models_task.status != TaskStatus.COMPLETED:
        await task_registry.wait_for_task("startup:pull_ollama_models:01")
    
    if model in ("embedding", "both"):
        await ollama_client.verify_embedding()
    if model in ("chat", "both"):
        await ollama_client.verify_chat()

async def import_initial_vocab(semantic_service: SemanticService):    
    for vocab in INITIAL_VOCABS:
        try:
            await semantic_service.import_vocabulary(rdf_source=vocab.rdf_source, identifier=vocab.identifier)
        except VocabAlreadyExistsError:
            pass


async def start_setup(
    settings: Settings,
    logger: Logger,
    ollama_client: OllamaClientWrapper,
    neo4j_driver: Neo4jDriver,
    task_registry: TaskRegistry,
    semantic_service: SemanticService
) -> None:

    # Ollama bootstrap: Pull models and optionally load them into memory on startup. This ensures that the models are ready to use when the first requests arrive, avoiding latency spikes due to model loading during request handling.

    if settings.skip_model_pull:
        logger.info("Skipping Ollama model pull because skip_model_pull is enabled.")
    else:
        await task_registry.create_task(pull_ollama_models(ollama_client=ollama_client, settings=settings), name="startup:pull_ollama_models:01", type=TaskType.STARTUP)

    if settings.load_ollama_models_on_startup:
        await task_registry.create_task(load_ollama_models(ollama_client=ollama_client,task_registry=task_registry, model="both"), name="startup:load_ollama_models:01", type=TaskType.STARTUP)
    else:
        logger.info("Skipping Ollama model warm-up because load_ollama_models_on_startup is disabled.")

    # Neo4j connection and bootstrap

    await neo4j_driver.wait_for_connection(100.0, 2.0)

    if settings.skip_initial_vocab_import:
        logger.info("Skipping initial vocabulary import because skip_initial_vocab_import is enabled.")
    else:
        await task_registry.create_task(import_initial_vocab(semantic_service), name="startup:import_initial_vocabs:01", type=TaskType.STARTUP)
