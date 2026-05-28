import unittest

from app.domain.extraction import (
    ChunkContext,
    ChunkMetadata,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    ExtractionContext,
    FileContext,
    FileRankingResult,
    QuantitativeAttribute,
    RankedFile,
    build_extraction_context_prompt,
    build_file_ranking_prompt,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    cap_extraction_context_for_prompt,
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
                metadata=ChunkMetadata(
                    start_idx=4,
                    end_idx=6,
                    file_path="metadata.txt",
                    data_package_name="package",
                ),
            )
        )

        self.assertIn("Chunk context metadata:", prompt)
        self.assertIn("Chunk content", prompt)
        self.assertIn("metadata.txt", prompt)
        self.assertIn('"start_idx":4', prompt)
        self.assertIn('"end_idx":6', prompt)
        self.assertIn("temperature 20 C", prompt)

    def test_extraction_system_prompt_limits_evaluated_entity_fallback(self):
        self.assertIn("Do not use evaluated_entity as a fallback class", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("Attach quantitative attributes", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("skip it", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("instead of producing one object per header or parameter line", EXTRACTION_CONTEXT_SYSTEM_PROMPT)

    def test_merge_extraction_context_deduplicates_items(self):
        first = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_type": "method",
                        "extracted_object": {
                            "identifier": "sample-method",
                            "description": "Method for sample A.",
                            "keywords": ["sample"],
                        },
                        "source_text": "Method for sample A.",
                    }
                ]
            }
        )
        second = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_type": "method",
                        "extracted_object": {
                            "identifier": "sample method",
                            "description": "Method for sample A with more detail.",
                            "keywords": ["experiment"],
                        },
                        "source_text": "Method for sample A with more detail.",
                    }
                ]
            }
        )

        merged = merge_extraction_context_results([first, second])

        self.assertEqual(len(merged.methods), 1)
        self.assertEqual(merged.methods[0].keywords, ["experiment", "sample"])

    def test_cap_extraction_context_for_prompt_keeps_recent_compact_objects(self):
        context = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_type": "resource",
                        "extracted_object": {
                            "identifier": f"resource-{index}",
                            "type": "dataset",
                            "description": "long description " * 20,
                        },
                        "source_text": "long source text " * 20,
                    }
                    for index in range(4)
                ]
            }
        )

        recent_two = ExtractionContext(
            extraction_objects=context.extraction_objects[-2:]
        )
        capped = cap_extraction_context_for_prompt(
            context,
            max_json_chars=len(recent_two.model_dump_json()),
        )

        self.assertIsNotNone(capped)
        self.assertEqual(
            [resource.identifier for resource in capped.resources],
            ["resource-2", "resource-3"],
        )
        self.assertLessEqual(
            len(capped.model_dump_json()),
            len(recent_two.model_dump_json()),
        )

    def test_qudt_query_builders_target_expected_types(self):
        quantity = QuantitativeAttribute(
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
