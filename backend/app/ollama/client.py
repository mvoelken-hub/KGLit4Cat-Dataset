import httpx
import ollama

from app.core.config import Settings, settings
from logging import Logger
from app.core.logging import logger

Embedding = list[float]

class OllamaClientWrapper:
    def __init__(self, settings: Settings, logger: Logger):

        self.embed_model = settings.ollama_embed_model
        self.chat_model = settings.ollama_chat_model
        self.embed_num_gpu = getattr(settings, "ollama_embed_num_gpu", -1)
        self.flash_attention = getattr(settings, "ollama_flash_attention", False)
        self.kv_cache_type = getattr(settings, "ollama_kv_cache_type", "f16")
        timeout = (
            httpx.Timeout(timeout=settings.ollama_timeout_seconds, connect=30.0)
            if settings.ollama_timeout_seconds is not None
            else httpx.Timeout(None)
        )

        self.model_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=timeout)
        self.embedding_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=timeout)
        self.chat_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=timeout)
        self.logger = logger
        self.embed_dimensions = settings.ollama_embed_dimensions
        self.max_context_length = settings.max_context_length

        self._base_url = settings.ollama_base_url

    def update_runtime_config(
        self,
        *,
        chat_model: str | None = None,
        embed_model: str | None = None,
        max_context_length: int | None = None,
        embed_num_gpu: int | None = None,
    ) -> None:
        if chat_model is not None:
            self.chat_model = chat_model
        if embed_model is not None:
            self.embed_model = embed_model
        if max_context_length is not None:
            self.max_context_length = max_context_length
        if embed_num_gpu is not None:
            self.embed_num_gpu = embed_num_gpu

    async def stop_all_models(self) -> None:
        await self.stop_embedding_model()
        await self.stop_chat_model()

    async def stop_embedding_model(self) -> None:
        await self.embedding_client.embed(
            model=self.embed_model,
            input="",
            keep_alive=0,
        )
        self.logger.info(f"Stopped embedding model: {self.embed_model}")

    async def stop_chat_model(self) -> None:
        await self.chat_client.chat(
            model=self.chat_model,
            messages=[
                ollama.Message(role="user", content=""),
            ],
            keep_alive=0,
        )
        self.logger.info(f"Stopped chat model: {self.chat_model}")

    async def close(self) -> None:
        await self.model_client.close()
        await self.embedding_client.close()
        await self.chat_client.close()
        self.logger.info("Closed Ollama clients")

    async def verify_embedding(self) -> bool:
        await self.embedding_client.embed(
            model=self.embed_model,
            input=["Test1", "Test2"],
            keep_alive=-1,
        )
        return True
    
    async def verify_chat(self) -> bool:
        await self.chat_client.chat(
            model=self.chat_model,
            messages=[
                ollama.Message(role="user", content=""),
            ],
            keep_alive=-1,
        )
        return True
    
    async def pull_models(self, model_names: list[str]) -> None:
        for model_name in model_names:
            await self.model_client.pull(model=model_name)
            self.logger.info(f"Pulled Ollama model: {model_name}")

    async def delete_model(self, model_name: str) -> None:
        await self.model_client.delete(model=model_name)
        self.logger.info(f"Deleted Ollama model: {model_name}")

    async def list_models(self) -> ollama.ListResponse:
        return await self.model_client.list()

    async def show_model(self, model_name: str) -> ollama.ShowResponse:
        return await self.model_client.show(model=model_name)

    async def list_running_models(self) -> ollama.ProcessResponse:
        return await self.model_client.ps()

    async def ping_model(
        self,
        model_name: str,
        num_ctx: int | None = None,
        is_embedding: bool = False,
        num_gpu: int | None = None,
    ) -> dict[str, object]:
        """Load a model into GPU memory to check if it fits.

        Uses keep_alive=-1 to keep the model loaded so VRAM can be inspected.
        For embedding models, uses the embed() endpoint; for chat/generate models,
        uses the generate() endpoint.
        Returns a dict with 'success' (bool), 'model' (str), 'error' (str|None),
        'load_duration_ns' (int|None), and 'eval_count' (int|None).
        """
        options = (
            ollama.Options(num_ctx=num_ctx, num_gpu=num_gpu)
            if num_ctx is not None or num_gpu is not None
            else None
        )
        try:
            if is_embedding:
                response = await self.model_client.embed(
                    model=model_name,
                    input=["ping"],
                    keep_alive=-1,
                    options=options,
                )
                load_duration = response.load_duration
                eval_count = None  # embed responses don't have eval_count
            else:
                response = await self.model_client.generate(
                    model=model_name,
                    prompt="ping... just respond with 'ok' to confirm you're alive.",
                    keep_alive=-1,
                    options=options,
                )
                load_duration = response.load_duration
                eval_count = response.eval_count
            return {
                "success": True,
                "model": model_name,
                "error": None,
                "load_duration_ns": load_duration,
                "eval_count": eval_count,
            }
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "error": f"{type(e).__name__}: {e}",
                "load_duration_ns": None,
                "eval_count": None,
            }

    async def unload_model(self, model_name: str, is_embedding: bool = False) -> None:
        """Unload a model from GPU memory by setting keep_alive=0.

        For embedding models, uses the embed() endpoint; for chat/generate models,
        uses the generate() endpoint.
        """
        if is_embedding:
            await self.model_client.embed(
                model=model_name,
                input=[""],
                keep_alive=0,
            )
        else:
            await self.model_client.generate(
                model=model_name,
                prompt="",
                keep_alive=0,
            )

    async def change_embedding_model(self, model_name: str) -> ollama.ShowResponse:
        await self.model_client.pull(model=model_name)
        await self.stop_embedding_model()
        previous_model = self.embed_model
        try:
            self.embed_model = model_name
            await self.verify_embedding()
            return await self.embedding_client.show(model=model_name)
        except Exception as e:
            self.logger.error(f"Failed to change embedding model to {model_name}: {e}")
            self.embed_model = previous_model
            await self.verify_embedding()
            return await self.embedding_client.show(model=previous_model)

    async def change_chat_model(self, model_name: str) -> ollama.ShowResponse:
        await self.model_client.pull(model=model_name)
        await self.stop_chat_model()
        previous_model = self.chat_model
        try:
            self.chat_model = model_name
            await self.verify_chat()
            return await self.chat_client.show(model=model_name)
        except Exception as e:
            self.logger.error(f"Failed to change chat model to {model_name}: {e}")
            self.chat_model = previous_model
            await self.verify_chat()
            return await self.chat_client.show(model=previous_model)

    @property
    def ollama_client(self) -> ollama.AsyncClient:
        """The underlying ollama.AsyncClient for direct /api/generate calls.

        Used by generate_structured() in the custom completion module.
        """
        return self.chat_client

    async def get_embeddings(self, input: list[str]) -> list[Embedding]:
        options = (
            ollama.Options(num_gpu=self.embed_num_gpu)
            if self.embed_num_gpu != -1
            else None
        )
        if options is None:
            response = await self.embedding_client.embed(
                model=self.embed_model,
                input=input,
                dimensions=self.embed_dimensions,
                truncate=False,
            )
        else:
            response = await self.embedding_client.embed(
                model=self.embed_model,
                input=input,
                dimensions=self.embed_dimensions,
                truncate=False,
                options=options,
            )
        return [list(embedding) for embedding in response.embeddings]

ollama_client = OllamaClientWrapper(settings=settings, logger=logger)
