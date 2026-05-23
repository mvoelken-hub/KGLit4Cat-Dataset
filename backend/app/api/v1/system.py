import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Response, status

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
async def get_llm_budget(settings: Settings = Depends(get_settings)):
    return {
        "chat_model": settings.ollama_chat_model,
        "max_context_length": settings.max_context_length,
        "warning_threshold": 0.8,
        "danger_threshold": 1.0,
    }


@router.get("/tasks", response_model=list[TaskResponse])
async def get_tasks(task_registry: TaskRegistry = Depends(get_task_registry)):
    """
    Return all currently registered background tasks.
    """
    return [_serialize_task(name, task_info) for name, task_info in task_registry.tasks.items()]

#endregion
