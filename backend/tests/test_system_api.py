from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.system import router
from app.core.config import Settings
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
    ollama_generation_temperature = 0.0
    ollama_generation_seed = 42
    ollama_enforce_output_token_limit = True


class FakeRemoteSettings(FakeSettings):
    ollama_base_url = "http://christina-desktop:11433"
    ollama_hostname = "christina-desktop"


class FakeRunningModel:
    def __init__(self, model: str, size: int = 1024, size_vram: int = 1024, context_length: int = 4096):
        self.name = model
        self.model = model
        self.size = size
        self.size_vram = size_vram
        self.context_length = context_length
        self.expires_at = None


class FakeAvailableModel:
    def __init__(self, model: str):
        self.name = model
        self.model = model
        self.size = 1024
        self.digest = "abc"
        self.modified_at = None
        self.details = {"parameter_size": "1B", "quantization_level": "Q4_0"}


class FakeOllamaClient:
    chat_model = "chat-runtime"
    embed_model = "embed-runtime"
    max_context_length = 4096
    embed_num_gpu = 0
    generation_temperature = 0.0
    generation_seed = 42
    enforce_output_token_limit = True

    def __init__(self):
        self.updates = []
        self.pull_calls = []
        self.delete_calls = []
        self.snapshots = [
            [FakeRunningModel("chat-runtime"), FakeRunningModel("embed-runtime")],
        ]
        self.ping_calls = []

    async def list_running_models(self):
        models = self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]
        return type("FakeRunningModels", (), {"models": models})()

    async def list_models(self):
        return type(
            "FakeModels",
            (),
            {"models": [FakeAvailableModel("chat-runtime"), FakeAvailableModel("embed-runtime")]},
        )()

    async def show_model(self, model_name):
        capabilities = ["embedding"] if model_name == "embed-runtime" else ["completion"]
        return type("FakeShow", (), {"capabilities": capabilities})()

    async def pull_models(self, model_names):
        self.pull_calls.append(model_names)

    async def delete_model(self, model_name):
        self.delete_calls.append(model_name)

    async def ping_model(self, model_name, num_ctx=None, is_embedding=False, num_gpu=None):
        self.ping_calls.append(
            {"model": model_name, "num_ctx": num_ctx, "is_embedding": is_embedding, "num_gpu": num_gpu}
        )
        return {
            "success": True,
            "model": model_name,
            "error": None,
            "load_duration_ns": 100_000_000,
            "eval_count": None,
        }

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
        if kwargs.get("generation_temperature") is not None:
            self.generation_temperature = kwargs["generation_temperature"]
        if kwargs.get("generation_seed") is not None:
            self.generation_seed = kwargs["generation_seed"]
        if kwargs.get("enforce_output_token_limit") is not None:
            self.enforce_output_token_limit = kwargs["enforce_output_token_limit"]


class TestSystemApi:
    def make_client(self, fake_ollama: FakeOllamaClient | None = None, settings=None) -> TestClient:
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_settings] = lambda: settings or FakeSettings()
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
        assert payload["runtime"]["generation_temperature"] == 0.0
        assert payload["runtime"]["generation_seed"] == 42
        assert payload["runtime"]["enforce_output_token_limit"] is True
        assert payload["running"]["chat_model_loaded"] is True
        assert payload["running"]["embedding_model_loaded"] is True
        assert payload["running"]["both_configured_models_loaded"] is True
        assert payload["models"]["models"][0]["model"] == "chat-runtime"
        assert payload["diagnostics"]["status"] == "ok"
        assert payload["host"]["flash_attention"] is True
        assert payload["running"]["models"][0]["processor"] == "100% GPU"
        assert payload["models"]["models"][0]["kind"] == "chat"
        assert payload["models"]["models"][1]["kind"] == "embedding"
        assert payload["models"]["models"][0]["is_cloud"] is False

    def test_ollama_models_classify_embedding_by_name_when_show_has_no_capabilities(self):
        class NoCapabilitiesOllama(FakeOllamaClient):
            async def show_model(self, model_name):
                return type("FakeShow", (), {"capabilities": []})()

        client = self.make_client(NoCapabilitiesOllama())

        response = client.get("/api/v1/ollama-models")

        assert response.status_code == 200
        payload = response.json()
        assert payload["models"][0]["kind"] == "unknown"
        assert payload["models"][1]["kind"] == "embedding"

    def test_ollama_models_mark_cloud_models(self):
        class CloudOllama(FakeOllamaClient):
            async def list_models(self):
                return type(
                    "FakeModels",
                    (),
                    {"models": [FakeAvailableModel("gemma4:31b-cloud")]},
                )()

            async def show_model(self, model_name):
                return type("FakeShow", (), {"capabilities": ["completion"]})()

        client = self.make_client(CloudOllama())

        response = client.get("/api/v1/ollama-models")

        assert response.status_code == 200
        payload = response.json()
        assert payload["models"][0]["kind"] == "chat"
        assert payload["models"][0]["is_cloud"] is True

    def test_ollama_config_reports_partial_chat_gpu_residency(self):
        fake_ollama = FakeOllamaClient()
        fake_ollama.snapshots = [
            [FakeRunningModel("chat-runtime", size=1000, size_vram=710), FakeRunningModel("embed-runtime")],
        ]
        client = self.make_client(fake_ollama)

        response = client.get("/api/v1/ollama-config")

        assert response.status_code == 200
        payload = response.json()
        assert payload["running"]["models"][0]["processor"] == "29%/71% CPU/GPU"
        assert payload["diagnostics"]["status"] == "chat-pressure"
        assert "does not fit completely on the GPU" in payload["diagnostics"]["summary"]

    def test_remote_ollama_config_hides_server_memory_settings(self):
        client = self.make_client(settings=FakeRemoteSettings())

        response = client.get("/api/v1/ollama-config")

        assert response.status_code == 200
        payload = response.json()
        assert payload["host"]["is_local"] is False
        assert "flash_attention" not in payload["host"]
        assert "kv_cache_type" not in payload["host"]

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
                "generation_temperature": 0.2,
                "generation_seed": 123,
                "enforce_output_token_limit": False,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["runtime"]["chat_model"] == "next-chat"
        assert payload["runtime"]["embedding_model"] == "next-embed"
        assert payload["runtime"]["max_context_length"] == 16384
        assert payload["runtime"]["embedding_batch_size"] == 8
        assert payload["runtime"]["embedding_num_gpu"] == -1
        assert payload["runtime"]["generation_temperature"] == 0.2
        assert payload["runtime"]["generation_seed"] == 123
        assert payload["runtime"]["enforce_output_token_limit"] is False
        assert fake_ollama.updates[-1] == {
            "chat_model": "next-chat",
            "embed_model": "next-embed",
            "max_context_length": 16384,
            "embed_num_gpu": -1,
            "generation_temperature": 0.2,
            "generation_seed": 123,
            "enforce_output_token_limit": False,
        }

    def test_ollama_runtime_patch_validates_bounds(self):
        client = self.make_client()

        response = client.patch("/api/v1/ollama-config/runtime", json={"max_context_length": 128})

        assert response.status_code == 422

    def test_generation_runtime_settings_load_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_GENERATION_TEMPERATURE", "0.25")
        monkeypatch.setenv("OLLAMA_ENFORCE_OUTPUT_TOKEN_LIMIT", "false")

        settings = Settings(_env_file=None)

        assert settings.ollama_generation_temperature == 0.25
        assert settings.ollama_enforce_output_token_limit is False

    def test_ollama_models_can_be_listed_pulled_and_removed_for_remote_host(self):
        fake_ollama = FakeOllamaClient()
        client = self.make_client(fake_ollama, settings=FakeRemoteSettings())

        list_response = client.get("/api/v1/ollama-models")
        pull_response = client.post("/api/v1/ollama-models/pull", json={"model": "new-model:latest"})
        delete_response = client.delete("/api/v1/ollama-models/new-model:latest")

        assert list_response.status_code == 200
        assert list_response.json()["models"][0]["model"] == "chat-runtime"
        assert pull_response.status_code == 200
        assert delete_response.status_code == 200
        assert fake_ollama.pull_calls == [["new-model:latest"]]
        assert fake_ollama.delete_calls == ["new-model:latest"]

    def test_performance_test_reports_models_stay_resident(self):
        fake_ollama = FakeOllamaClient()
        fake_ollama.snapshots = [
            [],
            [FakeRunningModel("embed-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
        ]
        client = self.make_client(fake_ollama)

        response = client.post("/api/v1/ollama-config/runtime/performance-test", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["diagnostics"]["status"] == "ok"
        assert payload["diagnostics"]["both_resident_after_final"] is True
        assert payload["steps"][-1]["key"] == "chat-final"
        assert fake_ollama.ping_calls[1]["num_ctx"] == 4096
        assert fake_ollama.ping_calls[0]["num_gpu"] == 0
        assert fake_ollama.ping_calls[-1]["is_embedding"] is False

    def test_performance_test_reports_embedding_displaces_chat(self):
        fake_ollama = FakeOllamaClient()
        fake_ollama.snapshots = [
            [FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime")],
            [FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime")],
            [FakeRunningModel("chat-runtime")],
        ]
        client = self.make_client(fake_ollama)

        response = client.post("/api/v1/ollama-config/runtime/performance-test", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["diagnostics"]["status"] == "reload-pressure"
        assert payload["diagnostics"]["chat_displaced_by_embedding"] is True
        assert "embedding_num_gpu=0" in payload["diagnostics"]["recommendations"][0]

    def test_performance_test_ignores_initial_warmup_when_final_switch_is_stable(self):
        class WarmupOllama(FakeOllamaClient):
            async def ping_model(self, model_name, num_ctx=None, is_embedding=False, num_gpu=None):
                result = await super().ping_model(model_name, num_ctx, is_embedding, num_gpu)
                if len(self.ping_calls) == 2:
                    result["load_duration_ns"] = 23_800_000_000
                elif len(self.ping_calls) == 3:
                    result["load_duration_ns"] = 4_400_000_000
                elif len(self.ping_calls) == 4:
                    result["load_duration_ns"] = 753_000_000
                return result

        fake_ollama = WarmupOllama()
        fake_ollama.snapshots = [
            [FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
        ]
        client = self.make_client(fake_ollama)

        response = client.post("/api/v1/ollama-config/runtime/performance-test", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["steps"][1]["load_duration_ms"] == 23800
        assert payload["steps"][2]["load_duration_ms"] == 4400
        assert payload["steps"][3]["load_duration_ms"] == 753
        assert payload["diagnostics"]["status"] == "ok"
        assert payload["diagnostics"]["recommendations"] == []

    def test_performance_test_reports_slow_chat_load(self):
        class SlowChatOllama(FakeOllamaClient):
            async def ping_model(self, model_name, num_ctx=None, is_embedding=False, num_gpu=None):
                result = await super().ping_model(model_name, num_ctx, is_embedding, num_gpu)
                if not is_embedding:
                    result["load_duration_ns"] = 2_500_000_000
                return result

        fake_ollama = SlowChatOllama()
        fake_ollama.snapshots = [
            [],
            [FakeRunningModel("embed-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
            [FakeRunningModel("embed-runtime"), FakeRunningModel("chat-runtime")],
        ]
        client = self.make_client(fake_ollama)

        response = client.post("/api/v1/ollama-config/runtime/performance-test", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["diagnostics"]["status"] == "chat-pressure"
        assert "Reduce runtime max context length" in payload["diagnostics"]["recommendations"][0]

    def test_performance_test_reports_unknown_when_ps_unavailable(self):
        class BrokenPsOllama(FakeOllamaClient):
            async def list_running_models(self):
                raise RuntimeError("ps unavailable")

        client = self.make_client(BrokenPsOllama())

        response = client.post("/api/v1/ollama-config/runtime/performance-test", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["diagnostics"]["status"] == "unknown"
