import unittest
from logging import getLogger
from types import SimpleNamespace

from app.ollama.client import OllamaClientWrapper


class FakeSettings:
    ollama_base_url = "http://localhost:11433"
    ollama_embed_model = "embed-test"
    ollama_chat_model = "chat-test"
    ollama_embed_dimensions = 768
    max_context_length = 64000


class FakeCpuEmbeddingSettings(FakeSettings):
    ollama_embed_num_gpu = 0


class FakeAsyncOllamaApi:
    def __init__(self):
        self.embed_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.pull_calls: list[dict] = []
        self.show_calls: list[dict] = []
        self.closed = False
        self.fail_next_embed = False
        self.fail_next_chat = False
        self._client = self

    async def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        if self.fail_next_embed and isinstance(kwargs.get("input"), list):
            self.fail_next_embed = False
            raise RuntimeError("embedding unavailable")
        return SimpleNamespace(embeddings=[[0.1, 0.2], [0.3, 0.4]])

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.fail_next_chat and kwargs.get("keep_alive") == -1:
            self.fail_next_chat = False
            raise RuntimeError("chat unavailable")
        return SimpleNamespace()

    async def pull(self, **kwargs):
        self.pull_calls.append(kwargs)

    async def generate(self, **kwargs):
        self.chat_calls.append(kwargs)
        return SimpleNamespace(load_duration=123_000_000, eval_count=1)

    async def show(self, **kwargs):
        self.show_calls.append(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    async def close(self):
        self.closed = True


class FakeLogger:
    def __init__(self):
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, message: str):
        self.info_messages.append(message)

    def error(self, message: str):
        self.error_messages.append(message)


class OllamaClientWrapperTests(unittest.TestCase):
    def test_exposes_direct_ollama_client(self):
        client = OllamaClientWrapper(FakeSettings(), getLogger(__name__))  # type: ignore[arg-type]

        self.assertEqual(client.chat_model, "chat-test")
        self.assertIs(client.ollama_client, client.chat_client)

    def test_configures_context_length(self):
        client = OllamaClientWrapper(FakeSettings(), getLogger(__name__))  # type: ignore[arg-type]

        self.assertEqual(client.max_context_length, FakeSettings.max_context_length)


class OllamaClientWrapperAsyncTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self):
        logger = FakeLogger()
        client = OllamaClientWrapper(FakeSettings(), logger)  # type: ignore[arg-type]
        model_client = FakeAsyncOllamaApi()
        embedding_client = FakeAsyncOllamaApi()
        chat_client = FakeAsyncOllamaApi()
        client.model_client = model_client  # type: ignore[assignment]
        client.embedding_client = embedding_client  # type: ignore[assignment]
        client.chat_client = chat_client  # type: ignore[assignment]
        return client, logger, model_client, embedding_client, chat_client

    async def test_stop_all_models_stops_embedding_and_chat_models(self):
        client, logger, _, embedding_client, chat_client = self.make_client()

        await client.stop_all_models()

        self.assertEqual(
            embedding_client.embed_calls,
            [{"model": "embed-test", "input": "", "keep_alive": 0}],
        )
        self.assertEqual(chat_client.chat_calls[0]["model"], "chat-test")
        self.assertEqual(chat_client.chat_calls[0]["keep_alive"], 0)
        self.assertIn("Stopped embedding model: embed-test", logger.info_messages)
        self.assertIn("Stopped chat model: chat-test", logger.info_messages)

    async def test_verify_models_keep_them_alive(self):
        client, _, _, embedding_client, chat_client = self.make_client()

        self.assertTrue(await client.verify_embedding())
        self.assertTrue(await client.verify_chat())

        self.assertEqual(embedding_client.embed_calls[0]["keep_alive"], -1)
        self.assertEqual(chat_client.chat_calls[0]["keep_alive"], -1)

    async def test_pull_models_pulls_each_model(self):
        client, logger, model_client, _, _ = self.make_client()

        await client.pull_models(["embed-test", "chat-test"])

        self.assertEqual(
            model_client.pull_calls,
            [{"model": "embed-test"}, {"model": "chat-test"}],
        )
        self.assertIn("Pulled Ollama model: chat-test", logger.info_messages)

    async def test_ping_chat_model_uses_non_empty_prompt_for_load_timing(self):
        client, _, model_client, _, _ = self.make_client()

        result = await client.ping_model("chat-test", num_ctx=8192)

        self.assertTrue(result["success"])
        self.assertTrue(model_client.chat_calls[0]["prompt"].startswith("ping"))
        self.assertEqual(model_client.chat_calls[0]["keep_alive"], -1)
        self.assertEqual(model_client.chat_calls[0]["options"].model_dump(exclude_none=True), {"num_ctx": 8192})
        self.assertEqual(result["load_duration_ns"], 123_000_000)

    async def test_get_embeddings_uses_dimensions_and_no_truncation(self):
        client, _, _, embedding_client, _ = self.make_client()

        embeddings = await client.get_embeddings(["one", "two"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(
            embedding_client.embed_calls[0],
            {
                "model": "embed-test",
                "input": ["one", "two"],
                "dimensions": FakeSettings.ollama_embed_dimensions,
                "truncate": False,
            },
        )

    async def test_get_embeddings_can_force_cpu_embedding(self):
        logger = FakeLogger()
        client = OllamaClientWrapper(FakeCpuEmbeddingSettings(), logger)  # type: ignore[arg-type]
        embedding_client = FakeAsyncOllamaApi()
        client.embedding_client = embedding_client  # type: ignore[assignment]

        await client.get_embeddings(["one"])

        self.assertEqual(embedding_client.embed_calls[0]["options"].model_dump(exclude_none=True), {"num_gpu": 0})

    def test_update_runtime_config_updates_models_and_limits(self):
        client = OllamaClientWrapper(FakeSettings(), getLogger(__name__))  # type: ignore[arg-type]

        client.update_runtime_config(chat_model="runtime-chat", max_context_length=2048, embed_num_gpu=0)

        self.assertEqual(client.chat_model, "runtime-chat")
        self.assertEqual(client.max_context_length, 2048)
        self.assertEqual(client.embed_num_gpu, 0)

    async def test_change_embedding_model_returns_new_model_when_verification_succeeds(self):
        client, _, model_client, embedding_client, _ = self.make_client()

        response = await client.change_embedding_model("new-embed")

        self.assertEqual(client.embed_model, "new-embed")
        self.assertEqual(model_client.pull_calls, [{"model": "new-embed"}])
        self.assertEqual(embedding_client.show_calls[-1], {"model": "new-embed"})
        self.assertEqual(response.model, "new-embed")

    async def test_change_embedding_model_restores_previous_model_when_verification_fails(self):
        client, logger, _, embedding_client, _ = self.make_client()
        embedding_client.fail_next_embed = True

        response = await client.change_embedding_model("broken-embed")

        self.assertEqual(client.embed_model, "embed-test")
        self.assertEqual(embedding_client.show_calls[-1], {"model": "embed-test"})
        self.assertEqual(response.model, "embed-test")
        self.assertIn("Failed to change embedding model to broken-embed", logger.error_messages[0])

    async def test_change_chat_model_returns_new_model_when_verification_succeeds(self):
        client, _, model_client, _, chat_client = self.make_client()

        response = await client.change_chat_model("new-chat")

        self.assertEqual(client.chat_model, "new-chat")
        self.assertEqual(model_client.pull_calls, [{"model": "new-chat"}])
        self.assertEqual(chat_client.show_calls[-1], {"model": "new-chat"})
        self.assertEqual(response.model, "new-chat")

    async def test_change_chat_model_restores_previous_model_when_verification_fails(self):
        client, logger, _, _, chat_client = self.make_client()
        chat_client.fail_next_chat = True

        response = await client.change_chat_model("broken-chat")

        self.assertEqual(client.chat_model, "chat-test")
        self.assertEqual(chat_client.show_calls[-1], {"model": "chat-test"})
        self.assertEqual(response.model, "chat-test")
        self.assertIn("Failed to change chat model to broken-chat", logger.error_messages[0])

    async def test_close_closes_all_underlying_clients(self):
        client, logger, model_client, embedding_client, chat_client = self.make_client()

        await client.close()

        self.assertTrue(model_client.closed)
        self.assertTrue(embedding_client.closed)
        self.assertTrue(chat_client.closed)
        self.assertIn("Closed Ollama clients", logger.info_messages)


if __name__ == "__main__":
    unittest.main()
