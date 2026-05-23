from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.system import router
from app.dependencies import get_settings


class FakeSettings:
    ollama_chat_model = "chat-test"
    max_context_length = 8192


class SystemApiTests:
    def test_llm_budget_endpoint_returns_safe_budget_metadata(self):
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_settings] = lambda: FakeSettings()
        client = TestClient(app)

        response = client.get("/api/v1/llm-budget")

        assert response.status_code == 200
        assert response.json() == {
            "chat_model": "chat-test",
            "max_context_length": 8192,
            "warning_threshold": 0.8,
            "danger_threshold": 1.0,
        }
