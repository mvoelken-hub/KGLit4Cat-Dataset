import unittest

from neo4j.time import Date as Neo4jDate
from rdflib import Graph

from app.domain.semantics import VocabSchemeInfo
from infra.neo4j_semantic_graph_repository import Neo4jSemanticGraphRepository, escape_lucene_query


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


class ImportNeo4jDriver:
    def __init__(self):
        self.added_graphs: list[Graph] = []
        self.constraints: list[tuple[str, str]] = []
        self.queries: list[tuple[str, dict | None]] = []

    def add_graph(self, graph: Graph) -> None:
        self.added_graphs.append(graph)

    async def create_uniqueness_constraint(self, label: str, property_key: str) -> None:
        self.constraints.append((label, property_key))

    async def query(self, query: str, parameters: dict | None = None, db_name: str | None = None) -> list[dict]:
        self.queries.append((query, parameters))
        return []


class Neo4jSemanticGraphRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_vocabulary_returns_single_rdf_type(self):
        repository = Neo4jSemanticGraphRepository(FakeNeo4jDriver(), FakeOllamaClient())

        vocab = await repository.get_vocabulary("https://w3id.org/nfdi4cat/voc4cat")

        self.assertIsNotNone(vocab)
        self.assertEqual(vocab.vocab_term_schemes[0].rdf_type, "skos__Concept")
        self.assertEqual(vocab.vocab_term_schemes[0].properties, ["skos__prefLabel"])
        self.assertEqual(vocab.vocab_term_schemes[0].applicable_relationships, ["skos__broader"])
        self.assertEqual(vocab.vocab_term_schemes[0].count, 532)

    async def test_cleanup_untyped_resources_deletes_resource_only_nodes(self):
        driver = QueryReturningNeo4jDriver([])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        await repository.cleanup_untyped_resources()

        query, _ = driver.queries[0]
        self.assertIn("MATCH (r:Resource)", query)
        self.assertIn("labels(r)", query)
        self.assertIn("label <> 'Resource'", query)
        self.assertIn("DETACH DELETE r", query)

    async def test_import_vocabulary_cleans_resource_only_nodes_after_graph_import(self):
        driver = ImportNeo4jDriver()
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())
        vocab_info = VocabSchemeInfo(
            identifier="urn:vocab",
            source="vocab.ttl",
            rdf_format="turtle",
            num_triples=0,
            resources=["urn:typed"],
        )

        await repository.import_vocabulary(vocab_info, Graph())

        self.assertEqual(len(driver.added_graphs), 1)
        cleanup_query, _ = driver.queries[0]
        self.assertIn("MATCH (r:Resource)", cleanup_query)
        self.assertIn("labels(r)", cleanup_query)
        self.assertIn("label <> 'Resource'", cleanup_query)
        self.assertIn("DETACH DELETE r", cleanup_query)
        link_query, link_parameters = driver.queries[1]
        self.assertIn("HAS_RESOURCE", link_query)
        self.assertEqual(link_parameters["resources"], ["urn:typed"])

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

    async def test_fulltext_candidate_query_escapes_lucene_special_characters(self):
        driver = QueryReturningNeo4jDriver([{"uri": "urn:term", "score": 8.0}])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        candidates = await repository.query_vocab_fulltext_candidates(
            identifier="urn:vocab",
            rdf_type="skos__Concept",
            query_text="nucleus: ^1H",
            top_k=5,
        )

        _, parameters = driver.queries[0]
        self.assertEqual(parameters["queryText"], r"nucleus\: \^1H")

    async def test_fulltext_candidate_query_escapes_forward_slash(self):
        driver = QueryReturningNeo4jDriver([{"uri": "urn:term", "score": 8.0}])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        candidates = await repository.query_vocab_fulltext_candidates(
            identifier="urn:vocab",
            rdf_type="skos__Concept",
            query_text="Compression Mode: diff/dup",
            top_k=5,
        )

        _, parameters = driver.queries[0]
        self.assertEqual(parameters["queryText"], r"Compression Mode\: diff\/dup")

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

    async def test_graph_expansion_skips_relationship_query_when_max_hops_is_zero(self):
        driver = QueryReturningNeo4jDriver([])
        repository = Neo4jSemanticGraphRepository(driver, FakeOllamaClient())

        statements = await repository.expand_vocab_graph(
            identifier="urn:vocab",
            seed_uris=["urn:seed"],
            allowed_rel_types=["skos__broader"],
            traversal_direction="outgoing",
            max_hops=0,
            max_statements_per_seed=7,
        )

        self.assertEqual(statements, [])
        self.assertEqual(driver.queries, [])

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


class EscapeLuceneQueryTests(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(escape_lucene_query("temperature"), "temperature")

    def test_escapes_caret(self):
        self.assertEqual(escape_lucene_query("^1H"), r"\^1H")

    def test_escapes_colon(self):
        self.assertEqual(escape_lucene_query("nucleus: ^1H"), r"nucleus\: \^1H")

    def test_escapes_forward_slash(self):
        self.assertEqual(escape_lucene_query("diff/dup"), r"diff\/dup")

    def test_escapes_plus_minus(self):
        self.assertEqual(escape_lucene_query("a+b-c"), r"a\+b\-c")

    def test_escapes_parentheses_and_brackets(self):
        self.assertEqual(escape_lucene_query("(test){val}[x]"), r"\(test\)\{val\}\[x\]")

    def test_escapes_wildcards(self):
        self.assertEqual(escape_lucene_query("test*test?test"), r"test\*test\?test")

    def test_escapes_tilde_and_quotes(self):
        self.assertEqual(escape_lucene_query('test~"val"'), r"test\~\"val\"")

    def test_escapes_double_ampersand_and_pipe(self):
        self.assertEqual(escape_lucene_query("a&&b||c"), r"a\&\&b\|\|c")

    def test_escapes_exclamation_mark(self):
        self.assertEqual(escape_lucene_query("!test"), r"\!test")

    def test_escapes_backslash(self):
        self.assertEqual(escape_lucene_query(r"path\to\file"), r"path\\to\\file")

    def test_empty_string(self):
        self.assertEqual(escape_lucene_query(""), "")


if __name__ == "__main__":
    unittest.main()
