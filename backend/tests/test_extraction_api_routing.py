import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.extraction import router
from app.core.task_registry import TaskStatus
from app.dependencies import get_workflow_service
from app.domain.extraction import (
    ChatModelUnavailableError,
    CompleteWorkflowProgress,
    InitialContextFailedError,
)


class FakeWorkflowService:
    async def get_complete_workflow_progress(self, data_package_id: str):
        return TaskStatus.UNKNOWN, CompleteWorkflowProgress(data_package_id=data_package_id)


class UnavailableChatWorkflowService(FakeWorkflowService):
    async def run_initial_context(self, **_kwargs):
        raise ChatModelUnavailableError("Chat model 'chat' is unavailable (status code 402)")


class FailedInitialContextWorkflowService(FakeWorkflowService):
    async def run_extraction(self, **_kwargs):
        raise InitialContextFailedError("Initial file understanding failed for all 2 file(s)")


def make_test_client(fake_service: FakeWorkflowService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_workflow_service] = lambda: fake_service
    return TestClient(app)


class ExtractionApiRoutingTests(unittest.TestCase):
    def test_new_workflow_progress_route_is_available(self):
        client = make_test_client(FakeWorkflowService())

        response = client.get("/api/v1/extraction/workflows/package-id/progress")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["progress"]["data_package_id"], "package-id")

    def test_old_extraction_routes_are_gone(self):
        client = make_test_client(FakeWorkflowService())

        self.assertEqual(client.post("/api/v1/extraction/run", json={}).status_code, 404)
        self.assertEqual(client.get("/api/v1/extraction/result/package-id").status_code, 404)

    def test_unavailable_chat_model_maps_to_service_unavailable(self):
        client = make_test_client(UnavailableChatWorkflowService())

        response = client.post("/api/v1/extraction/stages/orientation/package-id")

        self.assertEqual(response.status_code, 503)
        self.assertIn("402", response.json()["detail"])

    def test_failed_initial_context_maps_to_conflict(self):
        client = make_test_client(FailedInitialContextWorkflowService())

        response = client.post(
            "/api/v1/extraction/stages/evidence",
            json={"data_package_id": "package-id"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("failed for all 2", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
