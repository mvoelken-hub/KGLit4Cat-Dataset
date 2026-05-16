from typing import Sequence

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

        self.model_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=httpx.Timeout(None))
        self.embedding_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=httpx.Timeout(None))
        self.chat_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=httpx.Timeout(None))
        self.logger = logger
        self.embed_dimensions = settings.ollama_embed_dimensions
        self.max_context_length = settings.max_context_length

        # self.agent_model = OllamaModel(
        #     settings.ollama_chat_model,
        #     provider=OllamaProvider(base_url=settings.ollama_base_url+"/v1"),
        # )

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

        await self.model_client._client.aclose()
        await self.embedding_client._client.aclose()
        await self.chat_client._client.aclose()
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

    async def get_embeddings(self, input: list[str]) -> list[Embedding]:
        response = await self.embedding_client.embed(
            model=self.embed_model,
            input=input,
            dimensions=self.embed_dimensions,
            truncate=False,
        )
        return response.embeddings # type: ignore

ollama_client = OllamaClientWrapper(settings=settings, logger=logger)
