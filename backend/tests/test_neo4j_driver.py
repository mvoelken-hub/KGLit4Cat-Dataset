import unittest

from app.neo4j.driver import Neo4jDriver


class QueryCapturingNeo4jDriver(Neo4jDriver):
    def __init__(self):
        self.queries: list[tuple[str, str | None]] = []

    async def query(self, query: str, parameters: dict | None = None, db_name: str | None = None) -> list[dict]:
        self.queries.append((query, db_name))
        return []


class Neo4jDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_uniqueness_constraint_uses_current_cypher_syntax(self):
        driver = QueryCapturingNeo4jDriver()

        await driver.create_uniqueness_constraint(label="VocabScheme", property_key="identifier")

        self.assertEqual(
            driver.queries,
            [(
                "CREATE CONSTRAINT vocabscheme_identifier_unique IF NOT EXISTS\n"
                "FOR (n:VocabScheme) REQUIRE n.identifier IS UNIQUE",
                None,
            )],
        )


if __name__ == "__main__":
    unittest.main()
