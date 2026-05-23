import asyncio
from types import SimpleNamespace
import unittest
from app.domain.semantics import (
    VocabAlreadyExistsError,
    VocabGraphStatement,
    VocabNotFoundError,
    VocabQuery,
    VocabResource,
    VocabSchemeInfo,
    VocabSearchCandidate,
    VocabTermScheme,
)
from app.core.task_registry import TaskInfo, TaskStatus, TaskType
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
        self.identifiers = [vocab.identifier] if vocab else []
        self.pending_updates: list[VocabResource] = []
        self.updated_embedding_batches: list[list[VocabResource]] = []
        self.created_index_identifier: str | None = None
        self.deleted_index_identifier: str | None = None
        self.deleted_vocabulary_identifier: str | None = None
        self.cleaned_up = False
        self.vector_call: dict | None = None
        self.fulltext_call: dict | None = None
        self.expand_call: dict | None = None
        self.resource_uris: set[str] | None = None

    async def get_vocabulary(self, identifier: str):
        return self.vocab

    async def list_vocabulary_identifiers(self):
        return self.identifiers

    async def cleanup_untyped_resources(self):
        self.cleaned_up = True

    async def create_vocab_indexes(self, identifier: str):
        self.created_index_identifier = identifier

    async def delete_vocab_indexes(self, identifier: str):
        self.deleted_index_identifier = identifier

    async def delete_vocabulary(self, identifier: str):
        self.deleted_vocabulary_identifier = identifier

    async def check_pending_embedding_updates(self, identifier: str):
        return self.pending_updates

    async def update_resource_embeddings(self, resources: list[VocabResource]):
        self.updated_embedding_batches.append(resources)

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

    async def test_empty_allowed_relationships_keeps_query_to_seed_resources(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = make_service(repository, FakeOllamaClient())

        result = await service.query_vocabulary(
            "urn:vocab",
            VocabQuery(
                rdf_type="skos__Concept",
                fulltext_query="temperature",
                allowed_rel_types=[],
            ),
        )

        self.assertIsNone(repository.expand_call)
        self.assertEqual(repository.resource_uris, {"urn:seed"})
        self.assertEqual(result.graph_statements, [])

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

    async def test_list_and_cleanup_delegate_to_repository(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = make_service(repository, FakeOllamaClient())

        self.assertEqual(await service.list_vocabularies(), ["urn:vocab"])
        await service.cleanup_untyped_resources()

        self.assertTrue(repository.cleaned_up)

    async def test_delete_vocabulary_removes_indexes_before_vocabulary(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = make_service(repository, FakeOllamaClient())

        await service.delete_vocabulary("urn:vocab")

        self.assertEqual(repository.deleted_index_identifier, "urn:vocab")
        self.assertEqual(repository.deleted_vocabulary_identifier, "urn:vocab")

    async def test_delete_vocabulary_rejects_missing_identifier(self):
        repository = FakeSemanticGraphRepository(None)
        service = make_service(repository, FakeOllamaClient())

        with self.assertRaises(ValueError):
            await service.delete_vocabulary("urn:missing")

    async def test_create_and_delete_vocab_indexes_delegate_to_repository(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = make_service(repository, FakeOllamaClient())

        await service._create_vocab_indexes("urn:vocab")
        await service._delete_vocab_indexes("urn:vocab")

        self.assertEqual(repository.created_index_identifier, "urn:vocab")
        self.assertEqual(repository.deleted_index_identifier, "urn:vocab")


class FakeTaskRegistry:
    def __init__(self, task_info: TaskInfo | None = None):
        self.task_info = task_info
        self.created: list[dict] = []

    def get_task_info(self, name: str):
        self.requested_name = name
        return self.task_info

    async def create_task(self, coro, type, name: str):
        self.created.append({"type": type, "name": name})
        coro.close()
        return SimpleNamespace()


class SemanticServiceEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def make_service(
        self,
        repository: FakeSemanticGraphRepository,
        task_registry: FakeTaskRegistry,
        ollama: FakeOllamaClient | None = None,
    ):
        return SemanticService(
            semantic_graph_repository=repository,
            settings=FakeSettings(),
            ollama_client=ollama or FakeOllamaClient(),
            task_registry=task_registry,
        )

    def make_resource(self, uri: str) -> VocabResource:
        return VocabResource(
            uri=uri,
            rdf_types=["skos__Concept"],
            properties={"skos__prefLabel": uri},
        )

    async def test_generate_embeddings_rejects_missing_vocabulary(self):
        repository = FakeSemanticGraphRepository(None)
        service = self.make_service(repository, FakeTaskRegistry())

        with self.assertRaises(ValueError):
            await service.generate_embeddings_for_vocabulary("urn:missing")

    async def test_generate_embeddings_completes_when_no_pending_updates(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        service = self.make_service(repository, FakeTaskRegistry())

        pending_count, status = await service.generate_embeddings_for_vocabulary("urn:vocab")

        self.assertEqual(pending_count, 0)
        self.assertEqual(status, TaskStatus.COMPLETED)

    async def test_generate_embeddings_reports_running_task_without_starting_duplicate(self):
        task = asyncio.create_task(asyncio.sleep(0))
        task_info = TaskInfo(task=task, status=TaskStatus.RUNNING, type=TaskType.EMBEDDING)
        repository = FakeSemanticGraphRepository(make_vocab())
        repository.pending_updates = [self.make_resource("urn:seed")]
        task_registry = FakeTaskRegistry(task_info)
        service = self.make_service(repository, task_registry)

        pending_count, status = await service.generate_embeddings_for_vocabulary("urn:vocab")

        self.assertEqual(pending_count, 1)
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(task_registry.created, [])
        await task

    async def test_generate_embeddings_raises_crashed_task_exception(self):
        async def fail():
            raise RuntimeError("embedding failed")

        task = asyncio.create_task(fail())
        await asyncio.gather(task, return_exceptions=True)
        task_info = TaskInfo(task=task, status=TaskStatus.CRASHED, type=TaskType.EMBEDDING)
        repository = FakeSemanticGraphRepository(make_vocab())
        repository.pending_updates = [self.make_resource("urn:seed")]
        service = self.make_service(repository, FakeTaskRegistry(task_info))

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            await service.generate_embeddings_for_vocabulary("urn:vocab")

    async def test_generate_embeddings_starts_new_task_for_pending_updates(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        repository.pending_updates = [self.make_resource("urn:seed")]
        task_registry = FakeTaskRegistry()
        service = self.make_service(repository, task_registry)

        pending_count, status = await service.generate_embeddings_for_vocabulary("urn:vocab")

        self.assertEqual(pending_count, 1)
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(
            task_registry.created,
            [{"type": TaskType.EMBEDDING, "name": "embedding:vocab:urn:vocab"}],
        )

    async def test_run_embedding_generation_batches_and_persists_embeddings(self):
        repository = FakeSemanticGraphRepository(make_vocab())
        resources = [self.make_resource("urn:one"), self.make_resource("urn:two")]
        ollama = FakeOllamaClient()
        service = self.make_service(repository, FakeTaskRegistry(), ollama=ollama)

        await service._run_embedding_generation(resources)

        self.assertEqual(len(repository.updated_embedding_batches), 1)
        self.assertEqual([resource.embedding for resource in resources], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        self.assertEqual(
            ollama.embedding_inputs,
            [resource.to_embedding_str() for resource in resources],
        )


if __name__ == "__main__":
    unittest.main()
