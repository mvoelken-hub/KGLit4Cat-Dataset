import unittest

from pydantic import ValidationError

from app.domain.semantics import (
    VocabGraphStatement,
    VocabQuery,
    VocabResource,
    VocabSearchCandidate,
    compact_vocab_query_result,
    fuse_vocab_candidates,
)


class VocabQueryDomainTests(unittest.TestCase):
    def test_query_requires_at_least_one_search_text(self):
        with self.assertRaises(ValidationError):
            VocabQuery(rdf_type="skos__Concept")

    def test_rrf_fuses_weighted_vector_and_fulltext_candidates(self):
        seeds = fuse_vocab_candidates(
            rdf_type="skos__Concept",
            vector_candidates=[
                VocabSearchCandidate(uri="urn:a", score=0.9, rank=1, source="vector"),
                VocabSearchCandidate(uri="urn:b", score=0.8, rank=2, source="vector"),
            ],
            fulltext_candidates=[
                VocabSearchCandidate(uri="urn:b", score=12.0, rank=1, source="fulltext"),
                VocabSearchCandidate(uri="urn:c", score=10.0, rank=2, source="fulltext"),
            ],
            seed_top_k=2,
            vector_weight=1.0,
            fulltext_weight=3.0,
            rrf_k=1,
        )

        self.assertEqual([seed.uri for seed in seeds], ["urn:b", "urn:c"])
        self.assertEqual(seeds[0].vector_rank, 2)
        self.assertEqual(seeds[0].fulltext_rank, 1)
        self.assertEqual(seeds[0].vector_score, 0.8)
        self.assertEqual(seeds[0].fulltext_score, 12.0)

    def test_rrf_keeps_vector_only_seed_metadata(self):
        seeds = fuse_vocab_candidates(
            rdf_type="skos__Concept",
            vector_candidates=[
                VocabSearchCandidate(uri="urn:a", score=0.9, rank=1, source="vector"),
            ],
            fulltext_candidates=[],
            seed_top_k=5,
            vector_weight=1.0,
            fulltext_weight=1.0,
            rrf_k=60,
        )

        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0].uri, "urn:a")
        self.assertEqual(seeds[0].vector_rank, 1)
        self.assertIsNone(seeds[0].fulltext_rank)

    def test_compaction_deduplicates_statements_and_strips_embeddings(self):
        seeds = fuse_vocab_candidates(
            rdf_type="skos__Concept",
            vector_candidates=[
                VocabSearchCandidate(uri="urn:a", score=0.9, rank=1, source="vector"),
            ],
            fulltext_candidates=[],
            seed_top_k=5,
            vector_weight=1.0,
            fulltext_weight=1.0,
            rrf_k=60,
        )

        result = compact_vocab_query_result(
            identifier="urn:vocab",
            rdf_type="skos__Concept",
            seeds=seeds,
            graph_statements=[
                VocabGraphStatement(
                    subject_uri="urn:a",
                    predicate="skos__broader",
                    object_uri="urn:b",
                ),
                VocabGraphStatement(
                    subject_uri="urn:a",
                    predicate="skos__broader",
                    object_uri="urn:b",
                ),
            ],
            resources=[
                VocabResource(
                    uri="urn:a",
                    rdf_types=["skos__Concept"],
                    properties={"skos__prefLabel": "A", "embedding": [1.0]},
                ),
                VocabResource(
                    uri="urn:b",
                    rdf_types=["skos__Concept"],
                    properties={"skos__prefLabel": "B"},
                ),
            ],
        )

        self.assertEqual(len(result.graph_statements), 1)
        self.assertEqual(set(result.resources), {"urn:a", "urn:b"})
        self.assertNotIn("embedding", result.resources["urn:a"].properties)


if __name__ == "__main__":
    unittest.main()
