from dataclasses import dataclass
from hashlib import sha256
from pydantic import HttpUrl

from rdflib import (
    BNode,
    Graph,
    Literal as RDFLiteral,
    Node,
    URIRef,
    util as rdflib_util,
    RDF
)

from app.domain.semantics.ontologies import (
    VOCAB_DESC_TYPES,
)

SKOLEM_BASE_URI = "http://example.org/.well-known/genid/"
SKOLEM_PREFIX = "bnode"


@dataclass
class LoadedRdfGraph:
    graph: Graph
    source: str
    rdf_format: str
    description: str | None = None

@dataclass
class SerializedRdfGraph:
    data: bytes
    file_name: str | None = None


class RdfLoadError(Exception):
    pass

async def load_rdf_graph(rdf_source: HttpUrl | SerializedRdfGraph, identifier: str) -> LoadedRdfGraph:
    if isinstance(rdf_source, SerializedRdfGraph):
        loaded_rdf_graph = await load_rdf_graph_from_file(rdf_source, identifier)
    elif isinstance(rdf_source, HttpUrl):
        loaded_rdf_graph = await load_rdf_graph_from_url(rdf_source, identifier)
    else:
        raise RdfLoadError("Unsupported RDF source type.")

    loaded_rdf_graph.graph = skolemize_bnodes_deterministically(loaded_rdf_graph.graph)
    remove_non_en_literals(loaded_rdf_graph.graph)
    loaded_rdf_graph.description = extract_description_for_graph(loaded_rdf_graph.graph)

    return loaded_rdf_graph


async def load_rdf_graph_from_url(url: HttpUrl, identifier: str) -> LoadedRdfGraph:
    graph = Graph(identifier=identifier)

    source = str(url)
    rdf_format = rdflib_util.guess_format(source) or "turtle"
    try:
        graph.parse(source=source)
    except Exception as exc:
        raise RdfLoadError(f"Failed to parse RDF source '{source}'.") from exc

    return LoadedRdfGraph(
        graph=graph,
        source=source,
        rdf_format=rdf_format,
    )

async def load_rdf_graph_from_file(file: SerializedRdfGraph, identifier: str) -> LoadedRdfGraph:

    if not file.file_name:
        raise RdfLoadError("Missing uploaded RDF filename.")

    graph = Graph(identifier=identifier)

    source = file.file_name
    rdf_format = rdflib_util.guess_format(source) or "ttl"

    try:
        content = file.data
        graph.parse(data=content, format=rdf_format)
    except Exception as exc:
        raise RdfLoadError(
            f"Failed to parse uploaded RDF file '{source}' as '{rdf_format}'."
        ) from exc

    return LoadedRdfGraph(
        graph=graph,
        source=source,
        rdf_format=rdf_format,
    )

def extract_description_for_graph(graph: Graph) -> str:

    desc = ""
    for desc_type in VOCAB_DESC_TYPES:
        for subject in graph.subjects(RDF.type, desc_type):
            desc += f"Vocab contains {subject} as {desc_type}:\n"
            for predicate, object_ in graph.predicate_objects(subject):
                desc += f" - {predicate}: {object_}\n"
            desc += "\n"

    return desc.strip()



    
def remove_non_en_literals(graph: Graph) -> None:
    triples_to_remove = []

    for subject, predicate, object_ in graph:
        if isinstance(object_, RDFLiteral):
            if object_.language and object_.language.lower() != "en":
                triples_to_remove.append((subject, predicate, object_))

    for triple in triples_to_remove:
        graph.remove(triple)

def skolemize_bnodes_deterministically(
    graph: Graph,
    skolem_base: str = SKOLEM_BASE_URI,
    skolem_prefix: str = SKOLEM_PREFIX,
) -> Graph:
    bnode_map: dict[BNode, URIRef] = {}

    for subject, predicate, object_ in graph:
        for node in (subject, predicate, object_):
            if isinstance(node, BNode) and node not in bnode_map:
                digest = sha256(str(node).encode("utf-8")).hexdigest()[:8]
                bnode_map[node] = URIRef(f"{skolem_base}{digest}")

    if not bnode_map:
        graph.namespace_manager.bind(skolem_prefix, URIRef(skolem_base), override=True, replace=True)
        return graph

    skolemized_graph = Graph()
    for prefix, namespace in graph.namespace_manager.namespaces():
        skolemized_graph.namespace_manager.bind(prefix, namespace)
    skolemized_graph.namespace_manager.bind(skolem_prefix, URIRef(skolem_base), override=True, replace=True)

    def replace_node(node: Node) -> Node:
        return bnode_map.get(node, node) if isinstance(node, BNode) else node

    for subject, predicate, object_ in graph:
        skolemized_graph.add((replace_node(subject), replace_node(predicate), replace_node(object_)))

    return skolemized_graph