import unittest

from app.domain.extraction import (
    QualitativeAttribute,
    VocabularyCandidateSelection,
    VocabularyFallbackQuery,
    build_candidate_selection_prompt,
    build_fallback_query_prompt,
    build_qualitative_vocab_query,
)


class VocabularyNormalizationDomainTests(unittest.TestCase):
    def test_candidate_selection_allows_no_match(self):
        selection = VocabularyCandidateSelection(
            selected_uri=None,
            confidence=0.0,
            reason="No candidate fits.",
        )

        self.assertIsNone(selection.selected_uri)

    def test_fallback_query_requires_text(self):
        with self.assertRaises(ValueError):
            VocabularyFallbackQuery.model_validate({})

    def test_qualitative_vocab_query_uses_attribute_text(self):
        query = build_qualitative_vocab_query(
            QualitativeAttribute(title="color", value="blue"),
            rdf_type="skos__Concept",
        )

        self.assertEqual(query.rdf_type, "skos__Concept")
        self.assertIn("color", query.vector_query or "")
        self.assertIn("blue", query.fulltext_query or "")

    def test_candidate_prompt_contains_only_supplied_candidates(self):
        prompt = build_candidate_selection_prompt(
            source_value="blue",
            source_context={"title": "color"},
            candidates=[{"uri": "urn:test:blue"}],
        )

        self.assertIn("urn:test:blue", prompt)
        self.assertIn("blue", prompt)

    def test_fallback_prompt_contains_failed_candidates(self):
        prompt = build_fallback_query_prompt(
            source_value="MHz",
            source_context={"unit": "MHz"},
            failed_candidates=[{"uri": "urn:test:hertz"}],
        )

        self.assertIn("urn:test:hertz", prompt)
        self.assertIn("MHz", prompt)


if __name__ == "__main__":
    unittest.main()

