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
from app.ollama.runtime import (
    apply_runtime_config,
    available_model_summary,
    clean_model_name,
    format_ollama_error,
    ollama_config_payload,
    run_performance_test,
)

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


class OllamaModelRequest(BaseModel):
    model: str = Field(..., min_length=1)


class OllamaPerformanceTestRequest(BaseModel):
    chat_model: str | None = Field(None, min_length=1)
    embedding_model: str | None = Field(None, min_length=1)
    max_context_length: int | None = Field(None, ge=512, le=262144)
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


@router.get("/ollama-config")
async def get_ollama_config(
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    return await ollama_config_payload(settings, ollama_client)


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
    apply_runtime_config(settings, ollama_client, updates)
    return await ollama_config_payload(settings, ollama_client)


@router.get("/ollama-models")
async def get_ollama_models(
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    return await available_model_summary(ollama_client)


@router.post("/ollama-models/pull")
async def pull_ollama_model(
    request: OllamaModelRequest,
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    try:
        model = clean_model_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        await ollama_client.pull_models([model])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=format_ollama_error(exc)) from exc
    return await available_model_summary(ollama_client)


@router.delete("/ollama-models/{model:path}")
async def delete_ollama_model(
    model: str,
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    try:
        model = clean_model_name(model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        await ollama_client.delete_model(model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=format_ollama_error(exc)) from exc
    return await available_model_summary(ollama_client)


@router.post("/ollama-config/runtime/performance-test")
async def run_ollama_performance_test(
    request: OllamaPerformanceTestRequest | None = None,
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClientWrapper = Depends(get_ollama_client),
):
    request = request or OllamaPerformanceTestRequest()
    try:
        return await run_performance_test(
            settings,
            ollama_client,
            chat_model=request.chat_model,
            embedding_model=request.embedding_model,
            max_context_length=request.max_context_length,
            embedding_num_gpu=request.embedding_num_gpu,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/tasks", response_model=list[TaskResponse])
async def get_tasks(task_registry: TaskRegistry = Depends(get_task_registry)):
    """
    Return all currently registered background tasks.
    """
    return [_serialize_task(name, task_info) for name, task_info in task_registry.tasks.items()]

#endregion
