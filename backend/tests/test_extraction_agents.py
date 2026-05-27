import unittest

from app.domain.extraction import (
    ChunkContext,
    ExtractionContext,
    FileContext,
    FileRankingResult,
    Quantity,
    RankedFile,
    build_extraction_context_prompt,
    build_file_ranking_prompt,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    fallback_file_ranking,
    merge_extraction_context_results,
)


class ExtractionDomainTests(unittest.TestCase):
    def test_file_ranking_result_is_pydantic_model(self):
        result = FileRankingResult(files=[RankedFile(rank=1, file_path="README.md")])

        self.assertEqual(result.files[0].file_path, "README.md")
        self.assertIn("README.md", build_file_ranking_prompt([FileContext(file_path="README.md")]))

    def test_fallback_file_ranking_prefers_metadata_files(self):
        result = fallback_file_ranking(
            [
                FileContext(file_path="raw/image.tif", byte_size=2_000_000),
                FileContext(file_path="README.md", byte_size=100),
            ]
        )

        self.assertEqual(result.files[0].file_path, "README.md")

    def test_chunk_prompt_contains_file_span_and_content(self):
        prompt = build_extraction_context_prompt(
            ChunkContext(
                content="temperature 20 C",
                start_idx=4,
                end_idx=6,
                file_path="metadata.txt",
                data_package_name="package",
            )
        )

        self.assertIn("metadata.txt", prompt)
        self.assertIn("4-6", prompt)
        self.assertIn("temperature 20 C", prompt)

    def test_merge_extraction_context_deduplicates_items(self):
        first = ExtractionContext.model_validate(
            {
                "methods": [
                    {
                        "identifier": "sample-method",
                        "description": "Method for sample A.",
                        "keywords": ["sample"],
                    }
                ]
            }
        )
        second = ExtractionContext.model_validate(
            {
                "methods": [
                    {
                        "identifier": "sample method",
                        "description": "Method for sample A with more detail.",
                        "keywords": ["experiment"],
                    }
                ]
            }
        )

        merged = merge_extraction_context_results([first, second])

        self.assertEqual(len(merged.methods), 1)
        self.assertEqual(merged.methods[0].keywords, ["experiment", "sample"])

    def test_qudt_query_builders_target_expected_types(self):
        quantity = Quantity(
            identifier="temperature",
            value="20",
            unit="C",
            quantity_kind="temperature",
        )

        kind_query = build_quantity_kind_vocab_query(quantity)
        unit_query = build_unit_vocab_query(quantity)

        self.assertEqual(kind_query.rdf_type, "qudt__QuantityKind")
        self.assertEqual(unit_query.rdf_type, "qudt__Unit")
        self.assertIn("temperature", kind_query.vector_query or "")
        self.assertIn("C", unit_query.fulltext_query or "")


if __name__ == "__main__":
    unittest.main()
