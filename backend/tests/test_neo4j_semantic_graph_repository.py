import unittest

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


class Neo4jSemanticGraphRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_vocabulary_wraps_single_rdf_type_label_as_list(self):
        repository = Neo4jSemanticGraphRepository(FakeNeo4jDriver(), FakeOllamaClient())

        vocab = await repository.get_vocabulary("https://w3id.org/nfdi4cat/voc4cat")

        self.assertIsNotNone(vocab)
        self.assertEqual(vocab.vocab_term_schemes[0].rdf_types, ["skos__Concept"])
        self.assertEqual(vocab.vocab_term_schemes[0].properties, ["uri", "skos__prefLabel"])
        self.assertEqual(vocab.vocab_term_schemes[0].applicable_relationships, ["skos__broader"])
        self.assertEqual(vocab.vocab_term_schemes[0].count, 532)


if __name__ == "__main__":
    unittest.main()
