import asyncio
from typing import Any

from app.core.config import Settings
from app.ollama.client import OllamaClientWrapper


def format_ollama_error(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def clean_model_name(model: str) -> str:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError("model cannot be blank")
    return cleaned


def is_local_ollama(settings: Settings) -> bool:
    hostname = getattr(settings, "ollama_hostname", None)
    return not hostname or str(hostname).strip().lower() in {"localhost", "127.0.0.1", "::1"}


def embedding_gpu_label(value: int) -> str:
    if value == -1:
        return "auto"
    if value == 0:
        return "cpu"
    return f"{value} layers"


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def serialize_details(details: Any) -> dict[str, Any] | None:
    if not details:
        return None
    if hasattr(details, "model_dump"):
        return details.model_dump(mode="json")
    if isinstance(details, dict):
        return details
    return {
        key: getattr(details, key, None)
        for key in ("format", "family", "families", "parameter_size", "quantization_level")
        if getattr(details, key, None) is not None
    }


def model_name(item: dict[str, Any]) -> str | None:
    return item.get("model") or item.get("name")


def model_kind(model_name_value: str, capabilities: list[str]) -> str:
    normalized_capabilities = {capability.lower() for capability in capabilities}
    normalized_name = model_name_value.lower()
    if (
        "embedding" in normalized_capabilities
        or "embed" in normalized_name
        or "bge" in normalized_name
        or "nomic-embed" in normalized_name
    ):
        return "embedding"
    if "completion" in normalized_capabilities or "vision" in normalized_capabilities or "tools" in normalized_capabilities:
        return "chat"
    return "unknown"


def is_cloud_model(model_name_value: str, size: Any) -> bool:
    try:
        numeric_size = float(size or 0)
    except (TypeError, ValueError):
        numeric_size = None
    return "cloud" in model_name_value.lower() or numeric_size == 0


def serialize_available_model(item: Any, show_response: Any = None) -> dict[str, Any]:
    name = get_value(item, "model") or get_value(item, "name") or ""
    size = get_value(item, "size")
    capabilities = list(get_value(show_response, "capabilities", []) or [])
    return {
        "name": get_value(item, "name"),
        "model": name,
        "size": size,
        "digest": get_value(item, "digest"),
        "modified_at": get_value(item, "modified_at"),
        "details": serialize_details(get_value(item, "details")),
        "capabilities": capabilities,
        "kind": model_kind(name, capabilities),
        "is_cloud": is_cloud_model(name, size),
    }


def processor_split(size: Any, size_vram: Any) -> dict[str, Any]:
    try:
        total = float(size or 0)
        vram = float(size_vram or 0)
    except (TypeError, ValueError):
        return {
            "processor": "unknown",
            "gpu_percent": None,
            "cpu_percent": None,
        }
    if total <= 0:
        return {
            "processor": "unknown",
            "gpu_percent": None,
            "cpu_percent": None,
        }
    gpu_percent = max(0, min(100, round((vram / total) * 100)))
    cpu_percent = max(0, 100 - gpu_percent)
    if gpu_percent >= 99:
        processor = "100% GPU"
    elif gpu_percent <= 1:
        processor = "100% CPU"
    else:
        processor = f"{cpu_percent}%/{gpu_percent}% CPU/GPU"
    return {
        "processor": processor,
        "gpu_percent": gpu_percent,
        "cpu_percent": cpu_percent,
    }


async def available_model_summary(ollama_client: OllamaClientWrapper) -> dict[str, Any]:
    try:
        response = await ollama_client.list_models()
    except Exception as exc:
        return {
            "available": False,
            "error": format_ollama_error(exc),
            "models": [],
        }
    items = list(getattr(response, "models", []) or [])
    show_results = await asyncio.gather(
        *[
            ollama_client.show_model(get_value(item, "model") or get_value(item, "name"))
            for item in items
            if get_value(item, "model") or get_value(item, "name")
        ],
        return_exceptions=True,
    )
    show_by_name: dict[str, Any] = {}
    for item, show_result in zip(items, show_results, strict=False):
        name = get_value(item, "model") or get_value(item, "name")
        if name and not isinstance(show_result, Exception):
            show_by_name[name] = show_result
    return {
        "available": True,
        "models": [
            serialize_available_model(
                item,
                show_by_name.get(get_value(item, "model") or get_value(item, "name")),
            )
            for item in items
        ],
    }


async def running_model_summary(ollama_client: OllamaClientWrapper) -> dict[str, Any]:
    try:
        running = await ollama_client.list_running_models()
    except Exception as exc:
        return {
            "available": False,
            "error": format_ollama_error(exc),
            "models": [],
        }

    models = []
    for item in getattr(running, "models", []) or []:
        size = get_value(item, "size")
        size_vram = get_value(item, "size_vram")
        models.append(
            {
                "name": get_value(item, "name"),
                "model": get_value(item, "model") or get_value(item, "name"),
                "size": size,
                "size_vram": size_vram,
                "context_length": get_value(item, "context_length"),
                "expires_at": get_value(item, "expires_at"),
                **processor_split(size, size_vram),
            }
        )
    return {
        "available": True,
        "models": models,
    }


def diagnose_ollama_runtime(
    chat_model: str,
    embedding_model: str,
    running: dict[str, Any],
) -> dict[str, Any]:
    if not running.get("available"):
        return {
            "status": "unknown",
            "summary": "Could not inspect running Ollama models.",
            "recommendations": ["Run the switch test after Ollama is reachable."],
        }
    running_models = running.get("models", [])
    loaded_names = {model_name(item) for item in running_models if isinstance(item, dict)}
    chat_loaded = chat_model in loaded_names
    embed_loaded = embedding_model in loaded_names
    chat_running = next((item for item in running_models if isinstance(item, dict) and model_name(item) == chat_model), None)
    chat_gpu_percent = chat_running.get("gpu_percent") if chat_running else None
    if chat_loaded and isinstance(chat_gpu_percent, int | float) and chat_gpu_percent < 99:
        return {
            "status": "chat-pressure",
            "summary": f"Configured chat model is only {chat_running.get('processor')} resident, so it does not fit completely on the GPU.",
            "recommendations": [
                "Reduce runtime max context length for chat calls.",
                "If chat still does not fit, change OLLAMA_KV_CACHE_TYPE on the Ollama host.",
            ],
        }
    if chat_loaded and embed_loaded:
        return {
            "status": "ok",
            "summary": "Configured chat and embedding models are currently resident.",
            "recommendations": [],
        }
    if chat_loaded or embed_loaded:
        return {
            "status": "partial",
            "summary": "Only one configured model is currently resident. Fit together is unknown until tested.",
            "recommendations": ["Run the switch test to check whether chat and embedding calls force reloads."],
        }
    return {
        "status": "unknown",
        "summary": "Neither configured model is currently resident. Fit together is unknown until tested.",
        "recommendations": ["Run the switch test before changing host-side settings."],
    }


async def ollama_config_payload(settings: Settings, ollama_client: OllamaClientWrapper) -> dict[str, Any]:
    running, available_models = await asyncio.gather(
        running_model_summary(ollama_client),
        available_model_summary(ollama_client),
    )
    max_context_length = int(getattr(ollama_client, "max_context_length", settings.max_context_length))
    input_token_budget = int(max_context_length * 0.75)
    embedding_num_gpu = int(getattr(ollama_client, "embed_num_gpu", getattr(settings, "ollama_embed_num_gpu", -1)))
    chat_model = getattr(ollama_client, "chat_model", settings.ollama_chat_model)
    embedding_model = getattr(ollama_client, "embed_model", settings.ollama_embed_model)
    is_local = is_local_ollama(settings)
    running_models = running.get("models", []) if running.get("available") else []
    loaded_names = {model_name(item) for item in running_models if isinstance(item, dict)}
    host_payload: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "is_local": is_local,
        "server_settings_note": "Server-side Ollama settings must be changed on the Ollama host.",
    }
    if is_local:
        host_payload.update(
            {
                "flash_attention": getattr(settings, "ollama_flash_attention", False),
                "kv_cache_type": getattr(settings, "ollama_kv_cache_type", "f16"),
            }
        )
    return {
        "host": host_payload,
        "runtime": {
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "max_context_length": max_context_length,
            "input_token_budget": input_token_budget,
            "input_target_ratio": 0.75,
            "embedding_batch_size": settings.embedding_batch_size,
            "embedding_num_gpu": embedding_num_gpu,
            "embedding_gpu_label": embedding_gpu_label(embedding_num_gpu),
            "resets_on_api_restart": True,
        },
        "models": available_models,
        "running": {
            **running,
            "chat_model_loaded": chat_model in loaded_names,
            "embedding_model_loaded": embedding_model in loaded_names,
            "both_configured_models_loaded": chat_model in loaded_names and embedding_model in loaded_names,
        },
        "diagnostics": diagnose_ollama_runtime(chat_model, embedding_model, running),
        "warnings": [
            "Runtime changes apply only to future SIMONE calls and reset when the API restarts.",
            "Lower context length or chunks per turn when average input tokens approach the input budget.",
            "Use embedding_num_gpu=0 or a lower embedding batch size if embeddings cause repeated GPU offloads.",
        ],
    }


def apply_runtime_config(
    settings: Settings,
    ollama_client: OllamaClientWrapper,
    updates: dict[str, Any],
) -> None:
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


def snapshot_loaded_names(snapshot: dict[str, Any]) -> set[str]:
    if not snapshot.get("available"):
        return set()
    return {
        name
        for item in snapshot.get("models", [])
        if isinstance(item, dict) and (name := model_name(item))
    }


def duration_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000, 1)
    except (TypeError, ValueError):
        return None


async def performance_step(
    ollama_client: OllamaClientWrapper,
    *,
    key: str,
    label: str,
    model: str,
    is_embedding: bool,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
) -> dict[str, Any]:
    result = await ollama_client.ping_model(
        model,
        num_ctx=num_ctx,
        is_embedding=is_embedding,
        num_gpu=num_gpu,
    )
    snapshot = await running_model_summary(ollama_client)
    return {
        "key": key,
        "label": label,
        "model": model,
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "load_duration_ms": duration_ms(result.get("load_duration_ns")),
        "snapshot": snapshot,
    }


def performance_recommendations(
    *,
    chat_model: str,
    embedding_model: str,
    initial: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if not initial.get("available") or any(not step["snapshot"].get("available") for step in steps):
        return {
            "status": "unknown",
            "summary": "The switch test ran, but loaded model state could not be inspected consistently.",
            "recommendations": ["Retry after Ollama /api/ps is reachable."],
        }
    embed_first, chat_step, embed_second, final_chat_step = steps
    after_embed = snapshot_loaded_names(embed_first["snapshot"])
    after_chat = snapshot_loaded_names(chat_step["snapshot"])
    after_second_embed = snapshot_loaded_names(embed_second["snapshot"])
    after_final_chat = snapshot_loaded_names(final_chat_step["snapshot"])
    recommendations: list[str] = []
    status = "ok"

    chat_displaced_by_embed = chat_model in after_chat and chat_model not in after_second_embed
    embed_displaced_by_chat = embedding_model in after_embed and embedding_model not in after_chat
    embed_displaced_by_final_chat = embedding_model in after_second_embed and embedding_model not in after_final_chat
    both_resident_after_second_embed = chat_model in after_second_embed and embedding_model in after_second_embed
    both_resident_after_final = chat_model in after_final_chat and embedding_model in after_final_chat
    slow_final_chat_load = (final_chat_step.get("load_duration_ms") or 0) >= 1000
    steady_state_ok = both_resident_after_second_embed and both_resident_after_final and not slow_final_chat_load

    if not steady_state_ok and (chat_displaced_by_embed or embed_displaced_by_final_chat or not both_resident_after_second_embed):
        status = "reload-pressure"
        recommendations.append("Set embeddings to CPU with embedding_num_gpu=0.")
        recommendations.append("Reduce embedding batch size to lower transient memory pressure.")
    if not chat_step["success"] or not final_chat_step["success"] or (chat_model not in after_final_chat) or slow_final_chat_load:
        status = "reload-pressure" if status != "ok" else "chat-pressure"
        recommendations.append("Reduce runtime max context length for chat calls.")
        recommendations.append("If chat still does not fit, change OLLAMA_KV_CACHE_TYPE on the Ollama host.")
    if not recommendations and not both_resident_after_final:
        status = "unknown"
        recommendations.append("Both models were not visible after the final snapshot; repeat the test before changing settings.")

    return {
        "status": status,
        "summary": (
            "Configured chat and embedding models stayed resident."
            if status == "ok"
            else "The test found evidence of model reload pressure or insufficient residency."
        ),
        "both_resident_after_final": both_resident_after_final,
        "chat_displaced_by_embedding": chat_displaced_by_embed and not steady_state_ok,
        "embedding_displaced_by_chat": (embed_displaced_by_chat or embed_displaced_by_final_chat) and not steady_state_ok,
        "recommendations": recommendations,
    }


async def run_performance_test(
    settings: Settings,
    ollama_client: OllamaClientWrapper,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    max_context_length: int | None = None,
    embedding_num_gpu: int | None = None,
) -> dict[str, Any]:
    active_chat_model = clean_model_name(chat_model or getattr(ollama_client, "chat_model", settings.ollama_chat_model))
    active_embedding_model = clean_model_name(embedding_model or getattr(ollama_client, "embed_model", settings.ollama_embed_model))
    active_context_length = max_context_length or int(getattr(ollama_client, "max_context_length", settings.max_context_length))
    active_embedding_num_gpu = embedding_num_gpu
    if active_embedding_num_gpu is None:
        active_embedding_num_gpu = int(getattr(ollama_client, "embed_num_gpu", getattr(settings, "ollama_embed_num_gpu", -1)))
    embed_num_gpu_option = active_embedding_num_gpu if active_embedding_num_gpu != -1 else None

    initial = await running_model_summary(ollama_client)
    steps = [
        await performance_step(
            ollama_client,
            key="embedding-first",
            label="Embedding call",
            model=active_embedding_model,
            is_embedding=True,
            num_gpu=embed_num_gpu_option,
        ),
        await performance_step(
            ollama_client,
            key="chat",
            label="Chat call",
            model=active_chat_model,
            is_embedding=False,
            num_ctx=active_context_length,
        ),
        await performance_step(
            ollama_client,
            key="embedding-second",
            label="Second embedding call",
            model=active_embedding_model,
            is_embedding=True,
            num_gpu=embed_num_gpu_option,
        ),
        await performance_step(
            ollama_client,
            key="chat-final",
            label="Final chat call",
            model=active_chat_model,
            is_embedding=False,
            num_ctx=active_context_length,
        ),
    ]
    diagnostics = performance_recommendations(
        chat_model=active_chat_model,
        embedding_model=active_embedding_model,
        initial=initial,
        steps=steps,
    )
    return {
        "runtime": {
            "chat_model": active_chat_model,
            "embedding_model": active_embedding_model,
            "max_context_length": active_context_length,
            "embedding_num_gpu": active_embedding_num_gpu,
        },
        "initial": initial,
        "steps": steps,
        "diagnostics": diagnostics,
    }
