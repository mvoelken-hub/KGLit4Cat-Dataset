import unittest

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from app.domain.semantics import LoadedRdfGraph, VocabSchemeInfo


class VocabSchemeInfoTests(unittest.TestCase):
    def test_from_loaded_graph_includes_only_uri_nodes_with_explicit_rdf_type(self):
        graph = Graph(identifier=URIRef("urn:vocab"))

        typed_subject = URIRef("urn:typed-subject")
        typed_object = URIRef("urn:typed-object")
        untyped_object = URIRef("urn:untyped-object")
        predicate = URIRef("urn:predicate")
        source_predicate = URIRef("urn:source")
        literal_predicate = URIRef("urn:label")

        graph.add((typed_subject, RDF.type, URIRef("urn:Type")))
        graph.add((typed_object, RDF.type, URIRef("urn:OtherType")))
        graph.add((typed_subject, predicate, typed_object))
        graph.add((typed_subject, source_predicate, untyped_object))
        graph.add((typed_subject, literal_predicate, Literal("A literal value")))

        loaded_graph = LoadedRdfGraph(
            graph=graph,
            source="vocab.ttl",
            rdf_format="turtle",
        )

        vocab_info = VocabSchemeInfo.from_loaded_graph(loaded_graph)

        self.assertEqual(vocab_info.resources, ["urn:typed-object", "urn:typed-subject"])
        self.assertNotIn("urn:untyped-object", vocab_info.resources)
        self.assertNotIn("A literal value", vocab_info.resources)


if __name__ == "__main__":
    unittest.main()
