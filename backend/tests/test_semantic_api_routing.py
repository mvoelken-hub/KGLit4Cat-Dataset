import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.semantic import router
from app.core.task_registry import TaskStatus
from app.dependencies import get_semantic_service
from app.domain.semantics import (
    VocabQueryResult,
    VocabSchemeInfo,
    VocabTermScheme,
)


VOCAB_IDENTIFIER = "https://w3id.org/nfdi4cat/voc4cat"
ENCODED_VOCAB_IDENTIFIER = "https%3A%2F%2Fw3id.org%2Fnfdi4cat%2Fvoc4cat"


class FakeSemanticService:
    def __init__(self):
        self.embedding_identifier: str | None = None
        self.vocabulary_identifier: str | None = None
        self.query_identifier: str | None = None
        self.query = None

    async def generate_embeddings_for_vocabulary(self, identifier: str):
        self.embedding_identifier = identifier
        return 3, TaskStatus.RUNNING

    async def get_vocabulary(self, identifier: str):
        self.vocabulary_identifier = identifier
        return VocabSchemeInfo(
            identifier=identifier,
            source="https://example.org/vocab.ttl",
            rdf_format="text/turtle",
            num_triples=42,
            vocab_term_schemes=[
                VocabTermScheme(
                    rdf_type="skos__Concept",
                    properties=["skos__prefLabel"],
                    applicable_relationships=["skos__broader"],
                    count=2,
                )
            ],
        )

    async def query_vocabulary(self, identifier: str, query):
        self.query_identifier = identifier
        self.query = query
        return VocabQueryResult(
            identifier=identifier,
            rdf_type=query.rdf_type,
        )


def make_test_client(fake_service: FakeSemanticService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_semantic_service] = lambda: fake_service
    return TestClient(app)


class SemanticApiRoutingTests(unittest.TestCase):
    def test_embeddings_route_is_matched_before_vocabulary_identifier_catchall(self):
        fake_service = FakeSemanticService()
        client = make_test_client(fake_service)

        response = client.post(
            f"/api/v1/semantic/vocabularies/embeddings/{ENCODED_VOCAB_IDENTIFIER}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"pending_updates": 3, "task_status": "running"},
        )
        self.assertEqual(fake_service.embedding_identifier, VOCAB_IDENTIFIER)
        self.assertIsNone(fake_service.vocabulary_identifier)

    def test_vocabulary_route_still_accepts_uri_identifier(self):
        fake_service = FakeSemanticService()
        client = make_test_client(fake_service)

        response = client.get(
            f"/api/v1/semantic/vocabularies/{ENCODED_VOCAB_IDENTIFIER}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["identifier"], VOCAB_IDENTIFIER)
        self.assertEqual(response.json()["vocab_term_schemes"][0]["rdf_type"], "skos__Concept")
        self.assertEqual(fake_service.vocabulary_identifier, VOCAB_IDENTIFIER)
        self.assertIsNone(fake_service.embedding_identifier)

    def test_query_route_is_matched_before_vocabulary_identifier_catchall(self):
        fake_service = FakeSemanticService()
        client = make_test_client(fake_service)

        response = client.post(
            f"/api/v1/semantic/vocabularies/query/{ENCODED_VOCAB_IDENTIFIER}",
            json={
                "rdf_type": "skos__Concept",
                "fulltext_query": "temperature",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["identifier"], VOCAB_IDENTIFIER)
        self.assertEqual(response.json()["rdf_type"], "skos__Concept")
        self.assertEqual(fake_service.query_identifier, VOCAB_IDENTIFIER)
        self.assertEqual(fake_service.query.fulltext_query, "temperature")
        self.assertIsNone(fake_service.vocabulary_identifier)

    def test_query_route_rejects_missing_query_text_as_bad_request(self):
        fake_service = FakeSemanticService()
        client = make_test_client(fake_service)

        response = client.post(
            f"/api/v1/semantic/vocabularies/query/{ENCODED_VOCAB_IDENTIFIER}",
            json={"rdf_type": "skos__Concept"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(fake_service.query_identifier)


if __name__ == "__main__":
    unittest.main()
