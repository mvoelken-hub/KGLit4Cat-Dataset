from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.system import router
from app.dependencies import get_ollama_client, get_settings


class FakeSettings:
    ollama_base_url = "http://localhost:11433"
    ollama_hostname = None
    ollama_embed_model = "embed-test"
    ollama_chat_model = "chat-test"
    ollama_flash_attention = True
    ollama_kv_cache_type = "q8_0"
    ollama_embed_num_gpu = -1
    embedding_batch_size = 32
    max_context_length = 8192


class FakeRunningModel:
    def __init__(self, model: str, size: int = 1024, size_vram: int = 512):
        self.model = model
        self.size = size
        self.size_vram = size_vram
        self.expires_at = None


class FakeRunningModels:
    models = [
        FakeRunningModel("chat-runtime"),
        FakeRunningModel("embed-runtime"),
    ]


class FakeOllamaClient:
    chat_model = "chat-runtime"
    embed_model = "embed-runtime"
    max_context_length = 4096
    embed_num_gpu = 0

    def __init__(self):
        self.updates = []

    async def list_running_models(self):
        return FakeRunningModels()

    def update_runtime_config(self, **kwargs):
        self.updates.append(kwargs)
        if kwargs.get("chat_model") is not None:
            self.chat_model = kwargs["chat_model"]
        if kwargs.get("embed_model") is not None:
            self.embed_model = kwargs["embed_model"]
        if kwargs.get("max_context_length") is not None:
            self.max_context_length = kwargs["max_context_length"]
        if kwargs.get("embed_num_gpu") is not None:
            self.embed_num_gpu = kwargs["embed_num_gpu"]


class TestSystemApi:
    def make_client(self, fake_ollama: FakeOllamaClient | None = None) -> TestClient:
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_settings] = lambda: FakeSettings()
        app.dependency_overrides[get_ollama_client] = lambda: fake_ollama or FakeOllamaClient()
        return TestClient(app)

    def test_llm_budget_endpoint_returns_safe_budget_metadata(self):
        client = self.make_client()

        response = client.get("/api/v1/llm-budget")

        assert response.status_code == 200
        assert response.json() == {
            "chat_model": "chat-runtime",
            "max_context_length": 4096,
            "input_token_budget": 3072,
            "input_target_ratio": 0.75,
            "warning_threshold": 0.8,
            "danger_threshold": 1.0,
        }

    def test_ollama_config_returns_runtime_and_host_metadata(self):
        client = self.make_client()

        response = client.get("/api/v1/ollama-config")

        assert response.status_code == 200
        payload = response.json()
        assert payload["host"]["base_url"] == "http://localhost:11433"
        assert payload["host"]["is_local"] is True
        assert payload["runtime"]["chat_model"] == "chat-runtime"
        assert payload["runtime"]["embedding_model"] == "embed-runtime"
        assert payload["runtime"]["max_context_length"] == 4096
        assert payload["runtime"]["embedding_num_gpu"] == 0
        assert payload["running"]["chat_model_loaded"] is True
        assert payload["running"]["embedding_model_loaded"] is True

    def test_ollama_runtime_patch_updates_in_process_values(self):
        fake_ollama = FakeOllamaClient()
        client = self.make_client(fake_ollama)

        response = client.patch(
            "/api/v1/ollama-config/runtime",
            json={
                "chat_model": "next-chat",
                "embedding_model": "next-embed",
                "max_context_length": 16384,
                "embedding_batch_size": 8,
                "embedding_num_gpu": -1,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["runtime"]["chat_model"] == "next-chat"
        assert payload["runtime"]["embedding_model"] == "next-embed"
        assert payload["runtime"]["max_context_length"] == 16384
        assert payload["runtime"]["embedding_batch_size"] == 8
        assert payload["runtime"]["embedding_num_gpu"] == -1
        assert fake_ollama.updates[-1] == {
            "chat_model": "next-chat",
            "embed_model": "next-embed",
            "max_context_length": 16384,
            "embed_num_gpu": -1,
        }

    def test_ollama_runtime_patch_validates_bounds(self):
        client = self.make_client()

        response = client.patch("/api/v1/ollama-config/runtime", json={"max_context_length": 128})

        assert response.status_code == 422
