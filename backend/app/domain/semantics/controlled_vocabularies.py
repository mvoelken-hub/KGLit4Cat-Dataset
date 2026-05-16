from typing import Any
from rdflib import Graph, URIRef

from pydantic import BaseModel, Field

from app.domain.semantics.ontologies import (
    META_PROPERTIES,
    VOCAB_DESC_TYPES
)

from app.domain.semantics.rdf import (
    LoadedRdfGraph,
)

class VocabAlreadyExistsError(Exception):
    pass


class VocabResource(BaseModel):
    """A vocabulary resource projection used for embedding generation."""

    uri: str
    rdf_types: list[str]
    properties: dict[str, Any] = Field(default_factory=dict)

    def to_embedding_str(self) -> str:
        lines = [
            "RDF Resource",
            self.uri,
            f"Applicable RDF Types: {', '.join(self.rdf_types)}",
            "Properties:",
        ]
        for key, value in self.properties.items():
            if key not in META_PROPERTIES:
                lines.append(f" - {key}: {value}")
        return "\n".join(lines)


class VocabTermScheme(BaseModel):
    """Domain model for a vocabulary term type and its graph shape."""

    rdf_types: list[str] = Field(...)
    properties: list[str] = Field(default_factory=list)
    applicable_relationships: list[str] = Field(default_factory=list)
    count: int = Field(...)


class VocabSchemeInfo(BaseModel):
    """Domain-level representation of a vocabulary scheme."""

    identifier: str
    source: str
    rdf_format: str
    num_triples: int
    description: str | None = None
    resources: list[str] = Field(default_factory=list)
    vocab_term_schemes: list[VocabTermScheme] = Field(default_factory=list)

    @classmethod
    def from_loaded_graph(cls, loaded_graph: LoadedRdfGraph):

        graph = loaded_graph.graph

        resource_uris = [
            str(node)
            for node in set(graph.subjects()) | set(graph.objects())
            if isinstance(node, URIRef)
        ]

        return cls(
            identifier=graph.identifier,
            source=loaded_graph.source,
            rdf_format=loaded_graph.rdf_format,
            num_triples=len(graph),
            description=loaded_graph.description,
            resources=resource_uris,
        )





