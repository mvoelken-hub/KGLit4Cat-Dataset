from typing import Protocol
from rdflib import Graph

from app.domain.semantics import (
    VocabSchemeInfo,
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