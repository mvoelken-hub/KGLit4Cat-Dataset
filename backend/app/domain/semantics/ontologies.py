from rdflib import (
    OWL,
    SKOS,
    DCAT,
    VOID,
)

VOCAB_DESC_TYPES = frozenset({
    OWL.Ontology,
    DCAT.Dataset,
    DCAT.Catalog,
    SKOS.ConceptScheme,
    VOID.Dataset,
})

META_ONTOLOGY_TYPES: frozenset[str] = frozenset({
    # OWL structural types
    "owl__Axiom",
    "owl__Restriction",
    "owl__AllDifferent",
    "owl__AllDisjointClasses",
    "owl__AnnotationProperty",
    "owl__ObjectProperty",
    "owl__DatatypeProperty",
    "owl__InverseFunctionalProperty",
    "owl__IrreflexiveProperty",
    "owl__SymmetricProperty",
    "owl__FunctionalProperty",
    "owl__TransitiveProperty",
    "owl__Nothing",
    "owl__NegativePropertyAssertion",
    "owl__DeprecatedProperty",
    "owl__DeprecatedClass",
    # RDFS meta-types that are schema definitions, not vocabulary content.
    "rdfs__Datatype",
    "rdfs__Container",
    "rdfs__ContainerMembershipProperty",
    # schema types from common vocabularies that are not useful for indexing or embedding.
    "skos__ConceptScheme",
    "schema__Organization",
    "schema__Person",
    # Any swrl types
    "swrl__",
    # Blank-node identifiers (internal RDF structure, not vocabulary content)
    "bnode__",

})

# Meta properties that are common in vocabularies but not useful for indexing or embedding.
META_PROPERTIES: frozenset[str] = frozenset({
    "date",
    "version",
    "note",
    "created",
    "modified",
    "issued",
    "license",
    "creator",
    "contributor",
    "publisher",
    "contact",
    "embedding",
    "embeddings",
    "uri",
})


def is_indexable_rdf_type(rdf_type: str) -> bool:
    if any(rdf_type.startswith(prefix) for prefix in META_ONTOLOGY_TYPES):
        return False
    return True