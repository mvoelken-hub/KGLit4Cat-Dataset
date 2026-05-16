from typing import Any

from rdflib import Graph, URIRef

from app.neo4j.driver import Neo4jDriver

from app.domain.semantics import (
    VocabSchemeInfo,
)
from app.domain.semantics.controlled_vocabularies import VocabTermScheme

class Neo4jSemanticGraphRepository:
    def __init__(self, neo4j_driver: Neo4jDriver):
        self._neo4j_driver = neo4j_driver

    async def import_vocabulary(self, vocab_scheme_info: VocabSchemeInfo, rdf_graph: Graph) -> None:
        self._neo4j_driver.add_graph(rdf_graph)

        await self._neo4j_driver.create_uniqueness_constraint(label="VocabScheme", property_key="identifier")

        await self._neo4j_driver.query(
            """
            MERGE (v:VocabScheme { identifier: $identifier })
            SET v.source = $source, v.rdfFormat = $rdfFormat, v.numTriples = $numTriples
            WITH v
            UNWIND $resources AS resUri
            MATCH (r:Resource { uri: resUri })
            MERGE (v)-[:HAS_RESOURCE]->(r)
            """,
            parameters={
                "identifier": vocab_scheme_info.identifier,
                "source": vocab_scheme_info.source,
                "rdfFormat": vocab_scheme_info.rdf_format,
                "numTriples": vocab_scheme_info.num_triples,
                "resources": vocab_scheme_info.resources,
            },
        )

    async def get_vocabulary(self, identifier: str) -> VocabSchemeInfo | None:
        result = await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme { identifier: $identifier })-[:HAS_RESOURCE]->(r:Resource)
            RETURN v, collect(r.uri) AS resources
            """,
            parameters={"identifier": identifier},
        )

        if not result:
            return None

        record = result[0]
        vocab_info = VocabSchemeInfo(
            identifier=record["v"].get("identifier"),   
            source=record["v"].get("source"),
            rdf_format=record["v"].get("rdfFormat"),
            num_triples=record["v"].get("numTriples"),
            resources=record["resources"],
        )

        result_2 = await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme { identifier: $identifier })-[:HAS_RESOURCE]->(r:Resource)
            UNWIND labels(r) AS rdfTypes
            WITH v, r, rdfTypes
            WHERE rdfTypes<> 'Resource'
            OPTIONAL MATCH (r)-[rel]-()
            WHERE rel IS NULL OR type(rel) <> 'HAS_RESOURCE'
            WITH
                rdfTypes,
                [relType IN collect(DISTINCT type(rel)) WHERE relType IS NOT NULL] AS relTypes,
                apoc.coll.toSet(apoc.coll.flatten(collect(keys(r)))) AS props,
                count(DISTINCT r) AS resourceCount
            RETURN rdfTypes AS RdfTypes,
                   relTypes AS ApplicableRelationships,
                   props AS Properties,
                   resourceCount AS Count
            """,
            parameters={"identifier": identifier},
        )

        vocab_info.vocab_term_schemes = [
            VocabTermScheme(
                rdf_types=record["RdfTypes"],
                properties=record["Properties"],
                applicable_relationships=record["ApplicableRelationships"],
                count=record["Count"]
            )
            for record in result_2
        ]

        return vocab_info
    
    async def list_vocabulary_identifiers(self) -> list[str]:
        result = await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme)
            RETURN v.identifier AS identifier
            """
        )
        return [record["identifier"] for record in result]