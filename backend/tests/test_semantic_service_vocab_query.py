import unittest
from app.domain.semantics import (
    VocabGraphStatement,
    VocabNotFoundError,
    VocabQuery,
    VocabResource,
    VocabSchemeInfo,
    VocabSearchCandidate,
    VocabTermScheme,
)
from app.services.semantic_service import SemanticService


class FakeSettings:
    embedding_batch_size = 10


class FakeOllamaClient:
    def __init__(self):
        self.embedding_inputs: list[str] | None = None

    async def get_embeddings(self, input: list[str]):
        self.embedding_inputs = input
        return [[0.1, 0.2, 0.3] for _ in input]


class FakeSemanticGraphRepository:
    def __init__(self, vocab: VocabSchemeInfo | None):
        self.vocab = vocab
        self.vector_call: dict | None = None
        self.fulltext_call: dict | None = None
        self.expand_call: dict | None = None
        self.resource_uris: set[str] | None = None

    async def get_vocabulary(self, identifier: str):
        return self.vocab

    async def query_vocab_vector_candidates(
        self,
        identifier: str,
        rdf_type: str,
        embedding: list[float],
        top_k: int,
    ):
        self.vector_call = {
            "identifier": identifier,
            "rdf_type": rdf_type,
            "embedding": embedding,
            "top_k": top_k,
        }
        return [
            VocabSearchCandidate(uri="urn:seed", score=0.9, rank=1, source="vector")
        ]

    async def query_vocab_fulltext_candidates(
        self,
        identifier: str,
        rdf_type: str,
        query_text: str,
        top_k: int,
    ):
        self.fulltext_call = {
            "identifier": identifier,
            "rdf_type": rdf_type,
            "query_text": query_text,
            "top_k": top_k,
        }
        return [
            VocabSearchCandidate(uri="urn:seed", score=12.0, rank=1, source="fulltext")
        ]

    async def expand_vocab_graph(
        self,
        identifier: str,
        seed_uris: list[str],
        allowed_rel_types: list[str],
        traversal_direction: str,
        max_hops: int,
        max_statements_per_seed: int,
    ):
        self.expand_call = {
            "identifier": identifier,
            "seed_uris": seed_uris,
            "allowed_rel_types": allowed_rel_types,
            "traversal_direction": traversal_direction,
            "max_hops": max_hops,
            "max_statements_per_seed": max_statements_per_seed,
        }
        return [
            VocabGraphStatement(
                subject_uri="urn:seed",
                predicate="skos__broader",
                object_uri="urn:parent",
            )
        ]

    async def get_vocab_resources(self, uris: set[str]):
        self.resource_uris = uris
        return [
            VocabResource(
                uri=uri,
                rdf_types=["skos__Concept"],
                properties={"skos__prefLabel": uri},
            )
            for uri in sorted(uris)
        ]


def make_vocab() -> VocabSchemeInfo:
    return VocabSchemeInfo(
        identifier="urn:vocab",
        source="urn:vocab.ttl",
        rdf_format="text/turtle",
        num_triples=3,
        vocab_term_schemes=[
            VocabTermScheme(
                rdf_type="skos__Concept",
                properties=["skos__prefLabel"],
                applicable_relationships=["skos__broader"],
                count=2,
            )
        ],
    )


def make_service(repository: FakeSemanticGraphRepository, ollama: FakeOllamaClient):
    return SemanticService(
        semantic_graph_repository=repository,
        settings=FakeSettings(),
        ollama_client=ollama,
        task_registry=None,
    )


class SemanticServiceVocabQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_vector_query_generates_embedding_and_uses_default_relationships(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        ollama = FakeOllamaClient()
        service = make_service(repository, ollama)

        result = await service.query_vocabulary(
            "urn:vocab",
            VocabQuery(rdf_type="skos__Concept", vector_query="temperature"),
        )

        self.assertEqual(ollama.embedding_inputs, ["temperature"])
        self.assertEqual(repository.vector_call["rdf_type"], "skos__Concept")
        self.assertIsNone(repository.fulltext_call)
        self.assertEqual(repository.expand_call["allowed_rel_types"], ["skos__broader"])
        self.assertEqual(repository.resource_uris, {"urn:seed", "urn:parent"})
        self.assertEqual(result.seeds[0].vector_rank, 1)

    async def test_fulltext_query_does_not_generate_embedding(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        ollama = FakeOllamaClient()
        service = make_service(repository, ollama)

        result = await service.query_vocabulary(
            "urn:vocab",
            VocabQuery(rdf_type="skos__Concept", fulltext_query="temperature"),
        )

        self.assertIsNone(ollama.embedding_inputs)
        self.assertIsNone(repository.vector_call)
        self.assertEqual(repository.fulltext_call["query_text"], "temperature")
        self.assertEqual(result.seeds[0].fulltext_rank, 1)

    async def test_combined_query_calls_both_retrievers(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        ollama = FakeOllamaClient()
        service = make_service(repository, ollama)

        result = await service.query_vocabulary(
            "urn:vocab",
            VocabQuery(
                rdf_type="skos__Concept",
                vector_query="temp",
                fulltext_query="temperature",
            ),
        )

        self.assertIsNotNone(repository.vector_call)
        self.assertIsNotNone(repository.fulltext_call)
        self.assertEqual(result.seeds[0].vector_rank, 1)
        self.assertEqual(result.seeds[0].fulltext_rank, 1)

    async def test_invalid_rdf_type_is_rejected(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = make_service(repository, FakeOllamaClient())

        with self.assertRaises(ValueError):
            await service.query_vocabulary(
                "urn:vocab",
                VocabQuery(rdf_type="qudt__Unit", fulltext_query="kelvin"),
            )

    async def test_missing_vocabulary_is_not_found(self):
        repository = FakeSemanticGraphRepository(None)
        service = make_service(repository, FakeOllamaClient())

        with self.assertRaises(VocabNotFoundError):
            await service.query_vocabulary(
                "urn:missing",
                VocabQuery(rdf_type="skos__Concept", fulltext_query="temperature"),
            )


if __name__ == "__main__":
    unittest.main()
