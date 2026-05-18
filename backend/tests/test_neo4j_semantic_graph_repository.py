import unittest

from neo4j.time import Date as Neo4jDate

from infra.neo4j_semantic_graph_repository import Neo4jSemanticGraphRepository


class FakeNode(dict):
    def __init__(self, properties: dict):
        super().__init__(properties)
        self.properties = properties

    def get(self, key: str, default=None):
        return self.properties.get(key, default)


class FakeNeo4jDriver:
    def __init__(self):
        self.queries: list[tuple[str, dict | None]] = []

    async def query(self, query: str, parameters: dict | None = None, db_name: str | None = None) -> list[dict]:
        self.queries.append((query, parameters))
        if len(self.queries) == 1:
            return [{
                "v": FakeNode({
                    "identifier": "https://w3id.org/nfdi4cat/voc4cat",
                    "source": "http://nfdi4cat.github.io/voc4cat/v2025-10-14/voc4cat.ttl",
                    "rdf_format": "text/turtle",
                    "num_triples": 5028,
                }),
                "resources": ["https://w3id.org/nfdi4cat/voc4cat_0000001"],
            }]
        return [{
            "RdfType": "skos__Concept",
            "Properties": ["uri", "skos__prefLabel"],
            "ApplicableRelationships": ["skos__broader"],
            "Count": 532,
        }]


class FakeOllamaClient:
    pass


class QueryReturningNeo4jDriver:
    def __init__(self, result: list[dict]):
        self.result = result
        self.queries: list[tuple[str, dict | None]] = []

    async def query(self, query: str, parameters: dict | None = None, db_name: str | None = None) -> list[dict]:
        self.queries.append((query, parameters))
        return self.result


class Neo4jSemanticGraphRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_vocabulary_returns_single_rdf_type(self):
        repository = Neo4jSemanticGraphRepository(FakeNeo4jDriver(), FakeOllamaClient())

        vocab = await repository.get_vocabulary("https://w3id.org/nfdi4cat/voc4cat")

        self.assertIsNotNone(vocab)
        self.assertEqual(vocab.vocab_term_schemes[0].rdf_type, "skos__Concept")
        self.assertEqual(vocab.vocab_term_schemes[0].properties, ["skos__prefLabel"])
        self.assertEqual(vocab.vocab_term_schemes[0].applicable_relationships, ["skos__broader"])
        self.assertEqual(vocab.vocab_term_schemes[0].count, 532)

    async def test_vector_candidate_query_uses_rdf_type_index_and_vocabulary_filter(self):
        driver = QueryReturningNeo4jDriver([{"uri": "urn:term", "score": 0.91}])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        candidates = await repository.query_vocab_vector_candidates(
            identifier="urn:vocab",
            rdf_type="skos__Concept",
            embedding=[0.1, 0.2],
            top_k=3,
        )

        query, parameters = driver.queries[0]
        self.assertIn("db.index.vector.queryNodes", query)
        self.assertIn("HAS_RESOURCE", query)
        self.assertEqual(parameters["indexName"], "vocab_skos_concept_VECTOR")
        self.assertEqual(parameters["identifier"], "urn:vocab")
        self.assertEqual(candidates[0].uri, "urn:term")
        self.assertEqual(candidates[0].rank, 1)
        self.assertEqual(candidates[0].source, "vector")

    async def test_fulltext_candidate_query_uses_rdf_type_index_and_vocabulary_filter(self):
        driver = QueryReturningNeo4jDriver([{"uri": "urn:term", "score": 12.0}])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        candidates = await repository.query_vocab_fulltext_candidates(
            identifier="urn:vocab",
            rdf_type="skos__Concept",
            query_text="temperature",
            top_k=5,
        )

        query, parameters = driver.queries[0]
        self.assertIn("db.index.fulltext.queryNodes", query)
        self.assertIn("HAS_RESOURCE", query)
        self.assertEqual(parameters["indexName"], "vocab_skos_concept_FULLTEXT")
        self.assertEqual(parameters["queryText"], "temperature")
        self.assertEqual(candidates[0].source, "fulltext")

    async def test_graph_expansion_honors_direction_allowed_relationships_and_internal_exclusion(self):
        driver = QueryReturningNeo4jDriver([
            {
                "subjectUri": "urn:seed",
                "predicate": "skos__broader",
                "objectUri": "urn:parent",
            }
        ])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        statements = await repository.expand_vocab_graph(
            identifier="urn:vocab",
            seed_uris=["urn:seed"],
            allowed_rel_types=["skos__broader"],
            traversal_direction="outgoing",
            max_hops=2,
            max_statements_per_seed=7,
        )

        query, parameters = driver.queries[0]
        self.assertIn("(seed)-[*1..2]->(other)", query)
        self.assertIn("CALL (seed) {", query)
        self.assertNotIn("CALL {\n                WITH seed", query)
        self.assertIn("type(rel) <> 'HAS_RESOURCE'", query)
        self.assertIn("type(rel) IN $allowedRelTypes", query)
        self.assertEqual(parameters["allowedRelTypes"], ["skos__broader"])
        self.assertEqual(parameters["maxStatementsPerSeed"], 7)
        self.assertEqual(statements[0].subject_uri, "urn:seed")
        self.assertEqual(statements[0].predicate, "skos__broader")
        self.assertEqual(statements[0].object_uri, "urn:parent")

    async def test_get_vocab_resources_normalizes_neo4j_temporal_properties(self):
        driver = QueryReturningNeo4jDriver([
            {
                "uri": "urn:term",
                "rdfTypes": ["Resource", "skos__Concept"],
                "props": {
                    "skos__prefLabel": "Term",
                    "dcterms__created": Neo4jDate(2026, 5, 18),
                    "nested": {"date": Neo4jDate(2026, 5, 19)},
                },
            }
        ])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        resources = await repository.get_vocab_resources({"urn:term"})

        self.assertEqual(resources[0].properties["dcterms__created"], "2026-05-18")
        self.assertEqual(resources[0].properties["nested"]["date"], "2026-05-19")


if __name__ == "__main__":
    unittest.main()
