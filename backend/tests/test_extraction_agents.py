import unittest

from app.domain.extraction import (
    ChunkContext,
    ChunkMetadata,
    DefinedTerm,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    ExtractionContext,
    ExtractionNormalization,
    FileContext,
    FileRankingResult,
    GroundedExtractionObject,
    QualitativeAttribute,
    QuantitativeAttribute,
    RankedFile,
    DataGeneratingActivity,
    Resource,
    TracedExtractionObject,
    build_extraction_context_prompt,
    build_file_ranking_prompt,
    build_object_grounding_selection_prompt,
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
                        "object_kind": "Method",
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
                        "object_kind": "Method",
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

    def test_merge_extraction_context_deduplicates_by_shared_qualitative_attribute(self):
        first = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "Method",
                        "extracted_object": {
                            "identifier": "method-a",
                            "type": "NMR acquisition",
                            "description": "Acquisition method.",
                            "keywords": ["NMR"],
                            "has_qualitative_attributes": [
                                {"title": "pulse sequence", "value": "zg30"}
                            ],
                        },
                        "source_text": "pulse sequence zg30",
                    }
                ]
            }
        )
        second = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "Method",
                        "extracted_object": {
                            "identifier": "method-b",
                            "type": "nmr acquisition",
                            "description": "More detailed acquisition method.",
                            "keywords": ["spectrum"],
                            "has_qualitative_attributes": [
                                {"title": "Pulse Sequence", "value": "ZG30"}
                            ],
                        },
                        "source_text": "Pulse Sequence ZG30",
                    }
                ]
            }
        )

        merged = merge_extraction_context_results([first, second])

        self.assertEqual(len(merged.methods), 1)
        self.assertEqual(merged.methods[0].keywords, ["NMR", "spectrum"])
        self.assertEqual(len(merged.methods[0].has_qualitative_attributes), 1)

    def test_merge_extraction_context_deduplicates_by_shared_quantitative_attribute(self):
        first = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "DataGeneratingActivity",
                        "extracted_object": {
                            "identifier": "run-a",
                            "type": "temperature program",
                            "description": "Temperature ramp.",
                            "has_quantitative_attributes": [
                                {
                                    "identifier": "temperature",
                                    "value": "300",
                                    "unit": "K",
                                    "quantity_kind": "temperature",
                                }
                            ],
                        },
                        "source_text": "300 K",
                    }
                ]
            }
        )
        second = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "DataGeneratingActivity",
                        "extracted_object": {
                            "identifier": "run-b",
                            "type": "Temperature Program",
                            "description": "Temperature ramp with hold.",
                            "has_quantitative_attributes": [
                                {
                                    "identifier": "Temperature",
                                    "value": "300",
                                    "unit": "k",
                                    "quantity_kind": "Temperature",
                                }
                            ],
                        },
                        "source_text": "Temperature 300 k",
                    }
                ]
            }
        )

        merged = merge_extraction_context_results([first, second])

        self.assertEqual(len(merged.data_generating_activities), 1)
        self.assertEqual(
            len(merged.data_generating_activities[0].has_quantitative_attributes),
            1,
        )

    def test_merge_extraction_context_keeps_different_object_kinds_separate(self):
        first = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "Method",
                        "extracted_object": {
                            "identifier": "shared-id",
                            "description": "Shared description.",
                        },
                        "source_text": "method",
                    }
                ]
            }
        )
        second = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "Resource",
                        "extracted_object": {
                            "identifier": "shared-id",
                            "description": "Shared description.",
                        },
                        "source_text": "resource",
                    }
                ]
            }
        )

        merged = merge_extraction_context_results([first, second])

        self.assertEqual(len(merged.methods), 1)
        self.assertEqual(len(merged.resources), 1)

    def test_cap_extraction_context_for_prompt_keeps_recent_compact_objects(self):
        context = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "Resource",
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

    def test_extraction_object_query_text_splits_embedding_and_fulltext(self):
        resource = Resource(
            identifier="spectrum-file",
            description="NMR spectrum measured after catalyst activation.",
            keywords=["NMR", "spectrum"],
            type="dataset",
        )

        embedding_text = resource.to_embedding_text()
        fulltext_query = resource.to_fulltext_query()

        self.assertIn("NMR spectrum measured", embedding_text)
        self.assertIn("NMR", embedding_text)
        self.assertIn("dataset", embedding_text)
        self.assertNotIn("measured after catalyst activation", fulltext_query)
        self.assertIn("NMR", fulltext_query)
        self.assertIn("spectrum", fulltext_query)
        self.assertIn("dataset", fulltext_query)

    def test_defined_term_serializes_voc4cat_uri_title_and_source(self):
        term = DefinedTerm(
            id="https://w3id.org/nfdi4cat/voc4cat_42",
            title="IR spectrum",
            from_CV="https://w3id.org/nfdi4cat/voc4cat",
        )
        dump = term.model_dump()
        self.assertEqual(dump["id"], "https://w3id.org/nfdi4cat/voc4cat_42")
        self.assertEqual(dump["title"], "IR spectrum")
        self.assertEqual(dump["from_CV"], "https://w3id.org/nfdi4cat/voc4cat")
        # Round-trip via validate
        round_trip = DefinedTerm.model_validate(dump)
        self.assertEqual(round_trip, term)

    def test_defined_term_accepts_missing_title_and_source(self):
        term = DefinedTerm(id="https://example.org/x")
        self.assertIsNone(term.title)
        self.assertIsNone(term.from_CV)

    def test_grounded_extraction_object_wraps_original_object_and_optional_term(self):
        resource = Resource(
            identifier="ir-spectrum",
            description="IR spectrum measurement",
            type="spectrum",
        )
        grounded_with_term = GroundedExtractionObject(
            object_identifier=resource.identifier,
            object_kind="Resource",
            extracted_object=resource,
            source_value="spectrum",
            defined_term=DefinedTerm(
                id="https://w3id.org/nfdi4cat/voc4cat_42",
                title="IR spectrum",
                from_CV="https://w3id.org/nfdi4cat/voc4cat",
            ),
            confidence=0.92,
            reason="Direct match.",
        )
        self.assertIs(grounded_with_term.extracted_object, resource)
        self.assertEqual(grounded_with_term.defined_term.id, "https://w3id.org/nfdi4cat/voc4cat_42")
        self.assertEqual(grounded_with_term.source_value, "spectrum")

        grounded_without_term = GroundedExtractionObject(
            object_identifier=resource.identifier,
            object_kind="Resource",
            extracted_object=resource,
            source_value="spectrum",
        )
        self.assertIsNone(grounded_without_term.defined_term)

    def test_extraction_normalization_keeps_grounded_objects_and_object_groundings_in_sync(self):
        resource = Resource(
            identifier="ir-spectrum",
            description="IR spectrum measurement",
            type="spectrum",
        )
        grounded = GroundedExtractionObject(
            object_identifier=resource.identifier,
            object_kind="Resource",
            extracted_object=resource,
            source_value="spectrum",
            defined_term=DefinedTerm(
                id="https://w3id.org/nfdi4cat/voc4cat_42",
                title="IR spectrum",
                from_CV="https://w3id.org/nfdi4cat/voc4cat",
            ),
        )
        normalization = ExtractionNormalization(grounded_objects=[grounded])
        self.assertEqual(len(normalization.grounded_objects), 1)
        self.assertEqual(len(normalization.object_groundings), 1)
        self.assertEqual(normalization.grounded_objects[0], normalization.object_groundings[0])

        # Round-trip via the legacy field
        legacy = ExtractionNormalization(object_groundings=[grounded])
        self.assertEqual(len(legacy.grounded_objects), 1)
        self.assertEqual(legacy.grounded_objects[0], grounded)

    def test_base_extraction_model_type_stays_string_without_defined_term(self):
        resource = Resource(
            identifier="spectrum",
            description="IR spectrum",
            type="spectrum",
        )
        self.assertEqual(resource.type, "spectrum")
        dump = resource.model_dump()
        self.assertEqual(dump["type"], "spectrum")
        # The BaseExtractionModel must not expose a DefinedTerm-typed field.
        self.assertNotIn("enriched_type", dump)
        self.assertNotIn("defined_term", dump)

    def test_build_object_grounding_selection_prompt_includes_object_context(self):
        prompt = build_object_grounding_selection_prompt(
            object_identifier="ir-spectrum",
            object_kind="Resource",
            raw_type="spectrum",
            source_context={"identifier": "ir-spectrum", "type": "spectrum"},
            candidates=[{"uri": "https://w3id.org/nfdi4cat/voc4cat_42", "title": "IR spectrum"}],
        )
        self.assertIn("ir-spectrum", prompt)
        self.assertIn("spectrum", prompt)
        self.assertIn("Resource", prompt)
        self.assertIn("voc4cat_42", prompt)

    def test_traced_extraction_object_round_trips_around_base_extraction_model(self):
        trace = TracedExtractionObject(
            object_kind="Resource",
            extracted_object=Resource(
                identifier="spectrum",
                description="IR spectrum",
                type="spectrum",
            ),
            source_text="IR spectrum",
        )
        dump = trace.model_dump()
        round_trip = TracedExtractionObject.model_validate(dump)
        self.assertEqual(round_trip.extracted_object.identifier, "spectrum")
        self.assertEqual(round_trip.extracted_object.type, "spectrum")
        self.assertEqual(round_trip.source_text, "IR spectrum")




    def test_score_prefers_specific_identifiers(self):
        specific = TracedExtractionObject(
            object_kind="Resource",
            extracted_object=Resource(identifier="HMS-Q11-p", description="A sample"),
            source_text="HMS-Q11-p",
        )
        generic = TracedExtractionObject(
            object_kind="DataGeneratingActivity",
            extracted_object=DataGeneratingActivity(identifier="measurement", description="A run"),
            source_text="measurement",
        )
        from app.domain.extraction.extraction_context import _score_extraction_object
        self.assertGreater(_score_extraction_object(specific), _score_extraction_object(generic))

    def test_score_rewards_attribute_richness(self):
        sparse = TracedExtractionObject(
            object_kind="DataGeneratingActivity",
            extracted_object=DataGeneratingActivity(identifier="run-1", description="A run"),
            source_text="run-1",
        )
        rich = TracedExtractionObject(
            object_kind="DataGeneratingActivity",
            extracted_object=DataGeneratingActivity(
                identifier="run-1",
                description="A run with many attributes",
                has_quantitative_attributes=[QuantitativeAttribute(identifier="t", value="300", unit="K", quantity_kind="temperature")],
                has_qualitative_attributes=[QualitativeAttribute(title="mode", value="batch")],
                keywords=["batch", "reactor"],
            ),
            source_text="run-1 batch reactor",
        )
        from app.domain.extraction.extraction_context import _score_extraction_object
        self.assertGreater(_score_extraction_object(rich), _score_extraction_object(sparse))

    def test_build_system_prompt_returns_base_when_no_context(self):
        from app.domain.extraction.extraction_context import build_system_prompt_with_context, EXTRACTION_CONTEXT_SYSTEM_PROMPT
        result = build_system_prompt_with_context(
            base_prompt=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            global_context=None,
            num_ctx=8192,
        )
        self.assertEqual(result, EXTRACTION_CONTEXT_SYSTEM_PROMPT)

    def test_build_system_prompt_includes_all_when_space_available(self):
        from app.domain.extraction.extraction_context import build_system_prompt_with_context, EXTRACTION_CONTEXT_SYSTEM_PROMPT, ExtractionContext
        ctx = ExtractionContext(extraction_objects=[
            TracedExtractionObject(
                object_kind="Resource",
                extracted_object=Resource(identifier="file.txt", description="A file"),
                source_text="file.txt",
            ),
        ])
        result = build_system_prompt_with_context(
            base_prompt=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            global_context=ctx,
            num_ctx=8192,
        )
        self.assertIn("file.txt", result)
        self.assertIn("Previously extracted objects", result)

    def test_build_system_prompt_triages_when_space_limited(self):
        from app.domain.extraction.extraction_context import build_system_prompt_with_context, EXTRACTION_CONTEXT_SYSTEM_PROMPT, ExtractionContext
        objects = [
            TracedExtractionObject(
                object_kind="Resource",
                extracted_object=Resource(identifier=f"file-{i}.txt", description=f"File {i}"),
                source_text=f"file-{i}.txt",
            )
            for i in range(50)
        ]
        ctx = ExtractionContext(extraction_objects=objects)
        result = build_system_prompt_with_context(
            base_prompt=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            global_context=ctx,
            num_ctx=16384,
            schema_buffer_chars=500,
        )
        self.assertIn("Previously extracted objects", result)
        # With a small context, not all 50 objects should fit
        self.assertGreater(result.count("file-"), 0)

    def test_build_system_prompt_with_overview_uses_overview_and_same_file_memory(self):
        from app.domain.extraction.extraction_context import (
            EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            ExtractionContext,
            build_system_prompt_with_overview,
        )
        from app.domain.extraction.overview import ExtractionOverview

        same_file = ExtractionContext(extraction_objects=[
            TracedExtractionObject(
                object_kind="Resource",
                extracted_object=Resource(identifier="same-file.dx", description="Same file resource"),
                source_text="same-file.dx",
            ),
        ])
        overview = ExtractionOverview(
            dataset_theme="1H NMR package",
            summary="Attach NMR parameters to an acquisition run.",
            known_traps=["PLW1 is a pulse power parameter, not a sample."],
        )

        result = build_system_prompt_with_overview(
            base_prompt=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            overview=overview,
            overview_status="structured",
            same_file_context=same_file,
            num_ctx=8192,
        )

        self.assertIn("Initial extraction overview", result)
        self.assertIn("1H NMR package", result)
        self.assertIn("Earlier extracted objects from this same file", result)
        self.assertIn("same-file.dx", result)
        self.assertIn("orientation only", result)

    def test_extraction_prompt_warns_that_instrument_parameters_are_not_objects(self):
        from app.domain.extraction.extraction_context import EXTRACTION_CONTEXT_SYSTEM_PROMPT

        self.assertIn("PLW1", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("PULPROG", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("parameter keys, not standalone scientific objects", EXTRACTION_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("File names and generated artifacts", EXTRACTION_CONTEXT_SYSTEM_PROMPT)

    def test_global_context_includes_all_prior_completed_chunks(self):
        from app.domain.extraction.workflow import ExtractionRunState, ExtractionChunkResult
        from app.domain.extraction.extraction_context import ExtractionContext, merge_extraction_context_results
        from app.services.extraction_service import ExtractionService
        state = ExtractionRunState(chunk_results=[
            ExtractionChunkResult(
                chunk_index=0, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed", extraction_context=ExtractionContext(extraction_objects=[
                    TracedExtractionObject(object_kind="Resource", extracted_object=Resource(identifier="a", description="A"), source_text="a"),
                ]),
            ),
            ExtractionChunkResult(
                chunk_index=1, file_path="b.txt", start_idx=0, end_idx=1,
                status="completed", extraction_context=ExtractionContext(extraction_objects=[
                    TracedExtractionObject(object_kind="Resource", extracted_object=Resource(identifier="b", description="B"), source_text="b"),
                ]),
            ),
        ])
        ctx = ExtractionService._global_extraction_context_for_prompt(state, current_chunk_index=2)
        self.assertIsNotNone(ctx)
        ids = [t.extracted_object.identifier for t in ctx.extraction_objects]
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_global_context_respects_chunk_order(self):
        from app.domain.extraction.workflow import ExtractionRunState, ExtractionChunkResult
        from app.domain.extraction.extraction_context import ExtractionContext
        from app.services.extraction_service import ExtractionService
        state = ExtractionRunState(chunk_results=[
            ExtractionChunkResult(
                chunk_index=0, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed", extraction_context=ExtractionContext(extraction_objects=[]),
            ),
            ExtractionChunkResult(
                chunk_index=5, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed", extraction_context=ExtractionContext(extraction_objects=[
                    TracedExtractionObject(object_kind="Resource", extracted_object=Resource(identifier="later", description="Later"), source_text="later"),
                ]),
            ),
        ])
        ctx = ExtractionService._global_extraction_context_for_prompt(state, current_chunk_index=6)
        self.assertIsNotNone(ctx)
        ids = [t.extracted_object.identifier for t in ctx.extraction_objects]
        self.assertIn("later", ids)

        # For chunk 5, only chunks with index < 5 should be included
        ctx_5 = ExtractionService._global_extraction_context_for_prompt(state, current_chunk_index=5)
        self.assertIsNotNone(ctx_5)
        self.assertEqual(ctx_5.extraction_objects, [])
if __name__ == "__main__":
    unittest.main()
