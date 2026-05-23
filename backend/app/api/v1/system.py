import asyncio
from typing import Any

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.config import Settings
from app.core.task_registry import TaskRegistry
from app.dependencies import (
    get_neo4j_driver,
    get_ollama_client,
    get_settings,
    get_task_registry,
)
from app.neo4j.driver import Neo4jDriver
from app.ollama.client import OllamaClientWrapper

from app.api.v1.schemas import (
    TaskResponse,
    _serialize_task
)

router = APIRouter(tags=["System"])


class OllamaRuntimeConfigPatch(BaseModel):
    chat_model: str | None = Field(None, min_length=1)
    embedding_model: str | None = Field(None, min_length=1)
    max_context_length: int | None = Field(None, ge=512, le=262144)
    embedding_batch_size: int | None = Field(None, ge=1, le=2048)
    embedding_num_gpu: int | None = Field(None, ge=-1, le=999)

#region --- Helper functions for health checks and task serialization ---

def _format_check_error(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


async def _check_neo4j(neo4j_driver: Neo4jDriver, settings: Settings) -> dict[str, Any]:
    try:
        await neo4j_driver.verify_connection()
        return {
            "status": "ok",
            "uri": settings.neo4j_uri,
            "database": settings.db_names["default"],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "uri": settings.neo4j_uri,
            "database": settings.db_names["default"],
            "error": _format_check_error(exc),
        }


async def _check_ollama_embedding(ollama_client: OllamaClientWrapper, settings: Settings) -> dict[str, Any]:
    try:
        await ollama_client.verify_embedding()
        return {
            "status": "ok",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_embed_model,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_embed_model,
            "error": _format_check_error(exc),
        }


async def _check_ollama_chat(ollama_client: OllamaClientWrapper, settings: Settings) -> dict[str, Any]:
    try:
        await ollama_client.verify_chat()
        return {
            "status": "ok",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_chat_model,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_chat_model,
            "error": _format_check_error(exc),
        }


#endregion 

#region --- API endpoints ---

@router.get("/health")
async def health_check(
    response: Response,
    settings: Settings = Depends(get_settings),
    neo4j_driver: Neo4jDriver = Depends(get_neo4j_driver),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    """
    Health check endpoint to verify that the API dependencies are reachable.
    """
    neo4j_check, embedding_check, chat_check = await asyncio.gather(
        _check_neo4j(neo4j_driver, settings),
        _check_ollama_embedding(ollama_client, settings),
        _check_ollama_chat(ollama_client, settings),
    )

    checks = {
        "neo4j": neo4j_check,
        "ollama_embedding": embedding_check,
        "ollama_chat": chat_check,
    }

    healthy = all(check["status"] == "ok" for check in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    }


@router.get("/settings")
async def get_current_settings(settings: Settings = Depends(get_settings)):
    """
    Return current application settings.
    """
    return settings.model_dump(mode="json")


@router.get("/llm-budget")
async def get_llm_budget(
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    max_context_length = int(getattr(ollama_client, "max_context_length", settings.max_context_length))
    input_token_budget = int(max_context_length * 0.75)
    return {
        "chat_model": getattr(ollama_client, "chat_model", settings.ollama_chat_model),
        "max_context_length": max_context_length,
        "input_token_budget": input_token_budget,
        "input_target_ratio": 0.75,
        "warning_threshold": 0.8,
        "danger_threshold": 1.0,
    }


def _is_local_ollama(settings: Settings) -> bool:
    hostname = getattr(settings, "ollama_hostname", None)
    return not hostname or str(hostname).strip().lower() in {"localhost", "127.0.0.1", "::1"}


def _embedding_gpu_label(value: int) -> str:
    if value == -1:
        return "auto"
    if value == 0:
        return "cpu"
    return f"{value} layers"


async def _running_model_summary(ollama_client: OllamaClientWrapper) -> dict[str, Any]:
    try:
        running = await ollama_client.list_running_models()
    except Exception as exc:
        return {
            "available": False,
            "error": _format_check_error(exc),
            "models": [],
        }

    models = []
    for item in getattr(running, "models", []) or []:
        models.append(
            {
                "model": getattr(item, "model", None),
                "size": getattr(item, "size", None),
                "size_vram": getattr(item, "size_vram", None),
                "expires_at": getattr(item, "expires_at", None),
            }
        )
    return {
        "available": True,
        "models": models,
    }


def _ollama_config_payload(settings: Settings, ollama_client: OllamaClientWrapper, running: dict[str, Any]) -> dict[str, Any]:
    max_context_length = int(getattr(ollama_client, "max_context_length", settings.max_context_length))
    input_token_budget = int(max_context_length * 0.75)
    embedding_num_gpu = int(getattr(ollama_client, "embed_num_gpu", getattr(settings, "ollama_embed_num_gpu", -1)))
    chat_model = getattr(ollama_client, "chat_model", settings.ollama_chat_model)
    embedding_model = getattr(ollama_client, "embed_model", settings.ollama_embed_model)
    running_models = running.get("models", []) if running.get("available") else []
    loaded_names = {item.get("model") for item in running_models if isinstance(item, dict)}
    return {
        "host": {
            "base_url": settings.ollama_base_url,
            "is_local": _is_local_ollama(settings),
            "server_settings_note": "Server-side Ollama settings must be changed on the Ollama host.",
            "flash_attention": getattr(settings, "ollama_flash_attention", False),
            "kv_cache_type": getattr(settings, "ollama_kv_cache_type", "f16"),
        },
        "runtime": {
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "max_context_length": max_context_length,
            "input_token_budget": input_token_budget,
            "input_target_ratio": 0.75,
            "embedding_batch_size": settings.embedding_batch_size,
            "embedding_num_gpu": embedding_num_gpu,
            "embedding_gpu_label": _embedding_gpu_label(embedding_num_gpu),
            "resets_on_api_restart": True,
        },
        "running": {
            **running,
            "chat_model_loaded": chat_model in loaded_names,
            "embedding_model_loaded": embedding_model in loaded_names,
        },
        "warnings": [
            "Runtime changes apply only to future SIMONE calls and reset when the API restarts.",
            "Lower context length or chunks per turn when average input tokens approach the input budget.",
            "Use embedding_num_gpu=0 or a lower embedding batch size if embeddings cause repeated GPU offloads.",
        ],
    }


@router.get("/ollama-config")
async def get_ollama_config(
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    running = await _running_model_summary(ollama_client)
    return _ollama_config_payload(settings, ollama_client, running)


@router.patch("/ollama-config/runtime")
async def update_ollama_runtime_config(
    patch: OllamaRuntimeConfigPatch,
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    updates = patch.model_dump(exclude_unset=True)
    for key in ("chat_model", "embedding_model"):
        if key in updates and updates[key] is not None:
            updates[key] = updates[key].strip()
            if not updates[key]:
                raise HTTPException(status_code=422, detail=f"{key} cannot be blank")
    if "chat_model" in updates:
        settings.ollama_chat_model = updates["chat_model"]
    if "embedding_model" in updates:
        settings.ollama_embed_model = updates["embedding_model"]
    if "max_context_length" in updates:
        settings.max_context_length = updates["max_context_length"]
    if "embedding_batch_size" in updates:
        settings.embedding_batch_size = updates["embedding_batch_size"]
    if "embedding_num_gpu" in updates:
        settings.ollama_embed_num_gpu = updates["embedding_num_gpu"]

    ollama_client.update_runtime_config(
        chat_model=updates.get("chat_model"),
        embed_model=updates.get("embedding_model"),
        max_context_length=updates.get("max_context_length"),
        embed_num_gpu=updates.get("embedding_num_gpu"),
    )
    running = await _running_model_summary(ollama_client)
    return _ollama_config_payload(settings, ollama_client, running)


@router.get("/tasks", response_model=list[TaskResponse])
async def get_tasks(task_registry: TaskRegistry = Depends(get_task_registry)):
    """
    Return all currently registered background tasks.
    """
    return [_serialize_task(name, task_info) for name, task_info in task_registry.tasks.items()]

#endregion
