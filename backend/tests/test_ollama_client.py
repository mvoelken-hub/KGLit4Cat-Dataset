import unittest
from logging import getLogger

from pydantic_ai.models.ollama import OllamaModel

from app.ollama.client import OllamaClientWrapper


class FakeSettings:
    ollama_base_url = "http://localhost:11433"
    ollama_embed_model = "embed-test"
    ollama_chat_model = "chat-test"
    ollama_embed_dimensions = 768
    max_context_length = 64000


class OllamaClientWrapperTests(unittest.TestCase):
    def test_exposes_pydantic_ai_agent_model(self):
        client = OllamaClientWrapper(FakeSettings(), getLogger(__name__))  # type: ignore[arg-type]

        self.assertIsInstance(client.agent_model, OllamaModel)
        self.assertEqual(client.chat_model, "chat-test")
        self.assertIn(
            "http://localhost:11433/v1",
            str(client.agent_model.__dict__["_provider"]),
        )


if __name__ == "__main__":
    unittest.main()
