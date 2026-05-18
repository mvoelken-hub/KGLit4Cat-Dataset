import re
from typing import Any, Literal

from rdflib import Graph, URIRef

from app.neo4j import (
    Neo4jDriver,
    VectorIndexInfo,
    FullTextIndexInfo,
    CreateIndexRequest,
    normalize_index_token,
    SemanticIndexType,
    SEMANTIC_INDEX_TYPES,
)

from app.ollama import OllamaClientWrapper

from app.domain.semantics import (
    VocabSchemeInfo,
    VocabResource,
    META_ONTOLOGY_TYPES,
    META_PROPERTIES,
    RELEVANT_QUDT_TYPES,
)
from app.domain.semantics.controlled_vocabularies import VocabTermScheme

VOCAB_INDEX_PREFIX = "vocab"

class Neo4jSemanticGraphRepository:
    def __init__(self, neo4j_driver: Neo4jDriver, ollama_client: OllamaClientWrapper):
        self._neo4j_driver = neo4j_driver
        self._ollama_client = ollama_client

    async def import_vocabulary(self, vocab_scheme_info: VocabSchemeInfo, rdf_graph: Graph) -> None:
        self._neo4j_driver.add_graph(rdf_graph)

        await self._neo4j_driver.create_uniqueness_constraint(label="VocabScheme", property_key="identifier")

        await self._neo4j_driver.query(
            """
            MERGE (v:VocabScheme { identifier: $identifier })
            SET v += $props
            WITH v
            UNWIND $resources AS resUri
            MATCH (r:Resource { uri: resUri })
            MERGE (v)-[:HAS_RESOURCE]->(r)
            """,
            parameters={
                "identifier": vocab_scheme_info.identifier,
                "props": vocab_scheme_info.model_dump(exclude={"resources", "vocab_term_schemes"}),
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
        vocab_info = VocabSchemeInfo.model_validate(record["v"])
        vocab_info.resources = record["resources"]

        result_2 = await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme { identifier: $identifier })-[:HAS_RESOURCE]->(r:Resource)
            UNWIND labels(r) AS rdfType
            WITH v, r, rdfType
            WHERE rdfType <> 'Resource'
            OPTIONAL MATCH (r)-[rel]-()
            WHERE rel IS NULL OR type(rel) <> 'HAS_RESOURCE'
            WITH
                rdfType,
                [relType IN collect(DISTINCT type(rel)) WHERE relType IS NOT NULL] AS relTypes,
                apoc.coll.toSet(apoc.coll.flatten(collect(keys(r)))) AS props,
                count(DISTINCT r) AS resourceCount
            RETURN rdfType AS RdfType,
                   relTypes AS ApplicableRelationships,
                   props AS Properties,
                   resourceCount AS Count
            """,
            parameters={"identifier": identifier},
        )

        for record in result_2:
            # Skip meta-ontology types
            rdf_type: str = str(record["RdfType"])
            if any(rdf_type.startswith(prefix) for prefix in META_ONTOLOGY_TYPES) and rdf_type not in RELEVANT_QUDT_TYPES:
                continue
            # Filter out meta-ontology properties
            properties = [
                p for p in record["Properties"]
                if not any(substring.lower() in p.lower() for substring in META_PROPERTIES)
            ]

            if not properties:
                continue

            vocab_term_scheme = VocabTermScheme(
                rdf_type=rdf_type,
                properties=properties,
                applicable_relationships=record["ApplicableRelationships"],
                count=record["Count"]
            )
            
            vocab_info.vocab_term_schemes.append(vocab_term_scheme)

        return vocab_info
    
    async def list_vocabulary_identifiers(self) -> list[str]:
        result = await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme)
            RETURN v.identifier AS identifier
            """
        )
        return [record["identifier"] for record in result]
    
    async def delete_vocabulary(self, identifier: str) -> None:
        await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme { identifier: $identifier })-[:HAS_RESOURCE]->(r:Resource)
            WHERE COUNT { (r)<-[:HAS_RESOURCE]-() } = 1
            DETACH DELETE r
            """,
            parameters={"identifier": identifier},
        )

        await self._neo4j_driver.query(
            """
            MATCH (v:VocabScheme { identifier: $identifier })
            DETACH DELETE v
            """,
            parameters={"identifier": identifier},
        )


    async def create_vocab_indexes(self, identifier: str) -> None:
        vocab_info = await self.get_vocabulary(identifier)
        if not vocab_info:
            raise ValueError(f"Vocabulary with identifier '{identifier}' not found")
        
        for term_scheme in vocab_info.vocab_term_schemes:
            for index_type in SEMANTIC_INDEX_TYPES:
                index_name = self.get_index_name(term_scheme.rdf_type, index_type)
                existing_index = await self._neo4j_driver.get_index_info(index_name)
                if existing_index:
                    await self._neo4j_driver.resample_index(index_name)
                    # We assume that the existing index covers all important properties for the FULLTEXT index (e.g. labels, descriptions, etc.).
                    # If the new term scheme had new props we would need to delete the old index and add the new propetties to the esisting ones.
                    continue
                index_request = CreateIndexRequest(
                    name=index_name,
                    type=index_type,
                    entity_type="NODE",
                    on_label_or_type=[term_scheme.rdf_type],
                    on_property=["embedding"] if index_type == "VECTOR" else term_scheme.properties,
                )
                await self._neo4j_driver.create_node_index(index_request)
                        

    async def get_vocab_indexes(self, identifier: str) -> list[VectorIndexInfo | FullTextIndexInfo]:
        vocab_info = await self.get_vocabulary(identifier)
        if not vocab_info:
            raise ValueError(f"Vocabulary with identifier '{identifier}' not found")
        
        all_indexes = await self._neo4j_driver.list_indexes()
        vocab_indexes = []
        for term in vocab_info.vocab_term_schemes:
            # First find all vector indexes
            vector_index_name = self.get_index_name(term.rdf_type, "VECTOR")
            vector_index = next((idx for idx in all_indexes if idx.name == vector_index_name), None)
            if vector_index:
                vocab_indexes.append(vector_index)
            # Then find all fulltext indexes
            fulltext_index_name = self.get_index_name(term.rdf_type, "FULLTEXT")
            fulltext_index = next((idx for idx in all_indexes if idx.name == fulltext_index_name), None)
            if fulltext_index:
                vocab_indexes.append(fulltext_index)
                
        return vocab_indexes
                            
    async def delete_vocab_indexes(self, identifier: str) -> None:
        vocab_info = await self.get_vocabulary(identifier)
        if not vocab_info:
            raise ValueError(f"Vocabulary with identifier '{identifier}' not found")
        
        for term in vocab_info.vocab_term_schemes:
            for index_type in SEMANTIC_INDEX_TYPES:
                index_name = self.get_index_name(term.rdf_type, index_type)
                await self._neo4j_driver.drop_index_by_name(index_name)



    async def get_vocab_resources(self, uris: set[str]) -> list[VocabResource]:
        query = f"""
        UNWIND $uris AS uri
        MATCH (r:Resource {{ uri: uri }})
        RETURN r.uri AS uri, labels(r) AS rdfTypes, properties(r) AS props
        """
        result = await self._neo4j_driver.query(query, parameters={"uris": list(uris)})
        resources = []
        for record in result:
            labels = [label for label in record["rdfTypes"] if label != "Resource"]
            if not labels:
                raise ValueError(f"Resource with URI '{record['uri']}' has no RDF types other than 'Resource'")
            
            labels = [label for label in labels if label not in META_ONTOLOGY_TYPES]
            if not labels:
                raise ValueError(f"Resource with URI '{record['uri']}' has no RDF types after filtering out meta-ontology types")
            
            resources.append(
                VocabResource(
                    uri=record["uri"],
                    rdf_types=labels,
                    properties=record["props"]
                )
            )

        return resources

    async def check_pending_embedding_updates(self, identifier: str) -> list[VocabResource]:
        vocab_indexes = await self.get_vocab_indexes(identifier)
        if not vocab_indexes:
            raise ValueError(f"No indexes found for vocabulary with identifier '{identifier}'")
        
        vector_indexes = [idx for idx in vocab_indexes if isinstance(idx, VectorIndexInfo)]
        if not vector_indexes:
            raise ValueError(f"No vector indexes found for vocabulary with identifier '{identifier}'")
        
        node_uris_without_embeddings: set[str] = set()
        for vector_index in vector_indexes:
            uris = await self.find_nodes_without_emebdding(vector_index)
            node_uris_without_embeddings.update(uris)
        
        return await self.get_vocab_resources(node_uris_without_embeddings)

    async def update_resource_embeddings(self, vocab_resources: list[VocabResource]) -> None:
        query = f"""
        UNWIND $resources AS res
        MATCH (r:Resource {{ uri: res.uri }})
        SET r.embedding = res.embedding
        """
        resources_data = [
            {"uri": resource.uri, "embedding": resource.embedding}
            for resource in vocab_resources
        ]
        await self._neo4j_driver.query(query, parameters={"resources": resources_data})

    # Helper

    @staticmethod
    def get_index_name(rdf_type: str, index_type: SemanticIndexType) -> str:
        safe_rdf_type = normalize_index_token(rdf_type)        
        return f"{VOCAB_INDEX_PREFIX}_{safe_rdf_type}_{index_type}"
    
    async def find_nodes_without_emebdding(self, vector_index: VectorIndexInfo) -> set[str]:
        query = f"""
        MATCH (n:`{vector_index.labels_or_types[0]}`)
        WHERE n.`{vector_index.properties[0]}` IS NULL
        RETURN n.uri AS uri
        """
        result = await self._neo4j_driver.query(query)
        return set([str(record["uri"]) for record in result])
    