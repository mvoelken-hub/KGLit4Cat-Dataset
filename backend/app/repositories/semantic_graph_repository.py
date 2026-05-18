from typing import Protocol
from rdflib import Graph

from app.domain.semantics import (
    VocabSchemeInfo,
    VocabResource
)

from app.neo4j import (
    VectorIndexInfo,
    FullTextIndexInfo,
)

class SemanticGraphRepository(Protocol):
    async def import_vocabulary(self, vocab_scheme_info: VocabSchemeInfo, rdf_graph: Graph) -> None:
        ...

    async def get_vocabulary(self, identifier: str) -> VocabSchemeInfo | None:
        ...

    async def list_vocabulary_identifiers(self) -> list[str]:
        ...

    async def delete_vocabulary(self, identifier: str) -> None:
        ...

    async def create_vocab_indexes(self, identifier: str) -> None:
        ...

    async def delete_vocab_indexes(self, identifier: str) -> None:
        ...

    async def get_vocab_resources(self, uris: set[str]) -> list[VocabResource]:
        ...

    async def check_pending_embedding_updates(self, identifier: str) -> list[VocabResource]:
        ...

    async def update_resource_embeddings(self, vocab_resources: list[VocabResource]) -> None:
        ...