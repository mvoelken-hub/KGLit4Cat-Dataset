import unittest

from jsonschema import Draft202012Validator

from app.domain.extraction import (
    ChunkContext,
    ChunkMetadata,
    DefinedTerm,
    EvidenceContext,
    EvidenceNote,
    EVIDENCE_CONTEXT_SYSTEM_PROMPT,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    ExtractionContext,
    ExtractionNormalization,
    FileContext,
    FileRankingResult,
    GroundedExtractionObject,
    ProfileObjectPatchResult,
    ProfileTargetWriteDocument,
    QualitativeAttribute,
    QuantitativeAttribute,
    RankedFile,
    DataGeneratingActivity,
    Resource,
    TracedExtractionObject,
    build_extraction_context_prompt,
    evidence_text_match_score,
    build_file_ranking_prompt,
    is_noisy_payload_chunk,
    build_object_grounding_selection_prompt,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    cap_extraction_context_for_prompt,
    fallback_file_ranking,
    merge_extraction_context_results,
    validate_evidence_context_for_chunk,
)
from app.services.extraction_service import ExtractionService


class ExtractionDomainTests(unittest.TestCase):
    INITIAL_DRAFT_SCHEMA = {
        "type": "object",
        "required": ["id", "title", "description", "was_generated_by"],
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "array", "items": {"type": "string"}},
            "identifier": {"type": "array", "items": {"type": "string"}},
            "keyword": {"type": "array", "items": {"type": "string"}},
            "modification_date": {"type": ["string", "null"]},
            "was_generated_by": {"type": "array", "items": {"$ref": "#/$defs/DataGeneratingActivity"}},
            "creator": {"type": ["array", "null"], "items": {"$ref": "#/$defs/Agent"}},
            "dataset_distribution": {"type": ["array", "null"], "items": {"$ref": "#/$defs/Distribution"}},
            "type": {"type": ["array", "null"], "items": {"$ref": "#/$defs/Concept"}},
            "is_about_entity": {"type": ["array", "null"], "items": {"$ref": "#/$defs/EvaluatedEntity"}},
            "is_about_activity": {"type": ["array", "null"], "items": {"$ref": "#/$defs/EvaluatedActivity"}},
        },
        "$defs": {
            "DataGeneratingActivity": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "array", "items": {"type": "string"}},
                    "has_qualitative_attribute": {"type": "array"},
                    "has_quantitative_attribute": {"type": "array"},
                    "evaluated_activity": {"type": "array"},
                    "evaluated_entity": {"type": "array"},
                },
            },
            "Agent": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "array", "items": {"type": "string"}}, "type": {"type": ["array", "null"]}},
            },
            "Resource": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}, "title": {"type": "array", "items": {"type": "string"}}},
            },
            "Distribution": {
                "type": "object",
                "required": ["access_URL"],
                "properties": {
                    "access_URL": {"type": "array", "items": {"$ref": "#/$defs/Resource"}},
                    "title": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "array", "items": {"type": "string"}},
                    "format": {"type": ["object", "null"]},
                    "media_type": {"type": ["object", "null"]},
                },
            },
            "Concept": {
                "type": "object",
                "required": ["preferred_label"],
                "properties": {"preferred_label": {"type": "array", "items": {"type": "string"}}},
            },
            "EvaluatedEntity": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "has_qualitative_attribute": {"type": "array"},
                    "has_quantitative_attribute": {"type": "array"},
                    "was_generated_by": {"type": "array"},
                },
            },
            "EvaluatedActivity": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "array", "items": {"type": "string"}},
                    "has_qualitative_attribute": {"type": "array"},
                    "has_quantitative_attribute": {"type": "array"},
                },
            },
        },
    }

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

    def test_evidence_validation_accepts_exact_and_whitespace_normalized_matches(self):
        chunk = "##TITLE= Sample A\n##OWNER= Lab Team\n##XUNITS= 1/CM"
        context = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="n1",
                    category="resource_signal",
                    observation="The chunk names Sample A.",
                    evidence_text="##TITLE= Sample A",
                ),
                EvidenceNote(
                    note_id="n2",
                    category="agent_signal",
                    observation="The owner is Lab Team.",
                    evidence_text="##OWNER=    Lab Team",
                ),
            ]
        )

        validated, dropped = validate_evidence_context_for_chunk(
            context,
            chunk_content=chunk,
            file_path="sample.dx",
            start_idx=0,
            end_idx=3,
        )

        self.assertEqual(len(validated.notes), 2)
        self.assertEqual(dropped, [])
        self.assertTrue(all(note.evidence_match_score >= 0.9 for note in validated.notes))

    def test_evidence_note_tracks_confidence_separately_from_match_score(self):
        chunk = "##$PULPROG= <zg30>\n##$SOLVENT= <CDCl3>"
        context = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="method_group",
                    category="method_signal",
                    observation="Acquisition uses zg30 with CDCl3.",
                    evidence_text="##$PULPROG= <zg30>\n##$SOLVENT= <CDCl3>",
                    interpretation_confidence="medium",
                    profile_worthiness="high",
                )
            ]
        )

        validated, dropped = validate_evidence_context_for_chunk(
            context,
            chunk_content=chunk,
            file_path="acqu",
            start_idx=1,
            end_idx=3,
        )

        self.assertEqual(dropped, [])
        self.assertEqual(validated.notes[0].evidence_match_score, 1.0)
        self.assertEqual(validated.notes[0].interpretation_confidence, "medium")
        self.assertEqual(validated.notes[0].profile_worthiness, "high")

    def test_evidence_prompt_discourages_boilerplate_and_requires_quality(self):
        self.assertIn("Suppress repeated boilerplate", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("profile_worthiness", EVIDENCE_CONTEXT_SYSTEM_PROMPT)

    def test_evidence_validation_rejects_synthetic_paraphrase(self):
        chunk = "##TITLE= Real JCAMP record\n##XUNITS= 1/CM"
        context = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="synthetic",
                    category="measurement_signal",
                    observation="Synthetic experiment for testing.",
                    evidence_text="experiment-1 synthetic data for testing",
                )
            ]
        )

        validated, dropped = validate_evidence_context_for_chunk(
            context,
            chunk_content=chunk,
            file_path="sample.dx",
            start_idx=0,
            end_idx=2,
        )

        self.assertEqual(validated.notes, [])
        self.assertEqual([note.note_id for note in dropped], ["synthetic"])
        self.assertLess(evidence_text_match_score(dropped[0].evidence_text, chunk), 0.9)

    def test_noisy_jcamp_payload_detection_skips_xydata_payload_but_not_headers(self):
        payload = "\n".join(
            [
                "##XYDATA=(X++(Y..Y))",
                "1000 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@@1234567890",
                "1001 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@@1234567890",
                "1002 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@@1234567890",
            ]
        )
        header = "\n".join(
            [
                "##TITLE= Aspirin sample",
                "##JCAMP-DX= 5.00",
                "##XUNITS= 1/CM",
                "##YUNITS= TRANSMITTANCE",
            ]
        )

        self.assertTrue(is_noisy_payload_chunk(payload))
        self.assertFalse(is_noisy_payload_chunk(header))

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
        from app.domain.extraction.overview import ExtractionFileSummary, ExtractionOverview

        same_file = ExtractionContext(extraction_objects=[
            TracedExtractionObject(
                object_kind="Resource",
                extracted_object=Resource(identifier="same-file.dx", description="Same file resource"),
                source_text="same-file.dx",
            ),
        ])
        overview = ExtractionOverview(
            observed_signals=["same-file.dx contains spectroscopy-like syntax."],
            suggested_interpretations=["Attach parameter labels to an acquisition context."],
            conflicts_or_uncertainties=["PLW1 is a pulse power parameter, not a sample."],
        )
        file_summary = ExtractionFileSummary(
            file_path="same-file.dx",
            rank=1,
            data_format="JCAMP-DX-like spectroscopy export",
            parameter_terms=["PULPROG", "PLW1"],
            known_traps=["Parameter labels should attach to the acquisition context."],
        )

        result = build_system_prompt_with_overview(
            base_prompt=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
            overview=overview,
            overview_status="structured",
            file_summary=file_summary,
            same_file_context=same_file,
            num_ctx=8192,
        )

        self.assertIn("Initial extraction overview", result)
        self.assertIn("same-file.dx contains spectroscopy-like syntax", result)
        self.assertIn("Attach parameter labels to an acquisition context", result)
        self.assertIn("Current file summary", result)
        self.assertIn("JCAMP-DX-like spectroscopy export", result)
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
        from app.services.extraction_service import ExtractionService
        state = ExtractionRunState(chunk_results=[
            ExtractionChunkResult(
                chunk_index=0, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed",
                evidence_context=EvidenceContext(notes=[
                    EvidenceNote(note_id="a", category="resource_signal", observation="A", evidence_text="a"),
                ]),
            ),
            ExtractionChunkResult(
                chunk_index=1, file_path="b.txt", start_idx=0, end_idx=1,
                status="completed",
                evidence_context=EvidenceContext(notes=[
                    EvidenceNote(note_id="b", category="resource_signal", observation="B", evidence_text="b"),
                ]),
            ),
        ])
        ctx = ExtractionService._global_evidence_context_for_prompt(state, current_chunk_index=2)
        self.assertIsNotNone(ctx)
        ids = [note.note_id for note in ctx.notes]
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_global_context_respects_chunk_order(self):
        from app.domain.extraction.workflow import ExtractionRunState, ExtractionChunkResult
        from app.services.extraction_service import ExtractionService
        state = ExtractionRunState(chunk_results=[
            ExtractionChunkResult(
                chunk_index=0, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed",
                evidence_context=EvidenceContext(notes=[]),
            ),
            ExtractionChunkResult(
                chunk_index=5, file_path="a.txt", start_idx=0, end_idx=1,
                status="completed",
                evidence_context=EvidenceContext(notes=[
                    EvidenceNote(note_id="later", category="resource_signal", observation="Later", evidence_text="later"),
                ]),
            ),
        ])
        ctx = ExtractionService._global_evidence_context_for_prompt(state, current_chunk_index=6)
        self.assertIsNotNone(ctx)
        ids = [note.note_id for note in ctx.notes]
        self.assertIn("later", ids)

        # For chunk 5, only chunks with index < 5 should be included
        ctx_5 = ExtractionService._global_evidence_context_for_prompt(state, current_chunk_index=5)
        self.assertIsNotNone(ctx_5)
        self.assertEqual(ctx_5.notes, [])

    def test_projection_identifier_disambiguates_chunk_local_note_ids(self):
        first = EvidenceNote(
            note_id="software_version",
            category="resource_signal",
            observation="TOPSPIN version",
            evidence_text="##TITLE= Audit trail, TOPSPIN Version 3.2",
            file_path="10.zip/10/audita.txt",
            start_idx=0,
            end_idx=25,
        )
        second = EvidenceNote(
            note_id="software_version",
            category="resource_signal",
            observation="TOPSPIN processing version",
            evidence_text="##TITLE= Parameter file, TOPSPIN Version 3.2",
            file_path="10.zip/10/pdata/1/outd",
            start_idx=0,
            end_idx=14,
        )

        self.assertNotEqual(
            ExtractionService._projection_identifier_for_evidence_note(first),
            ExtractionService._projection_identifier_for_evidence_note(second),
        )

    def test_initial_draft_includes_core_and_evidence_guided_scaffold(self):
        evidence = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="instrument",
                    category="agent_signal",
                    observation="Instrument owner is Bruker.",
                    evidence_text="##ORIGIN= Bruker BioSpin GmbH",
                ),
                EvidenceNote(
                    note_id="solvent",
                    category="entity_signal",
                    observation="Solvent is CDCl3.",
                    evidence_text="SOLVENT= <CDCl3>",
                ),
                EvidenceNote(
                    note_id="pulse",
                    category="method_signal",
                    observation="Pulse sequence is zg30.",
                    evidence_text="PULPROG= <zg30>",
                ),
            ]
        )

        document, scaffold = ExtractionService._initial_profile_document(
            data_package_id="package-id",
            evidence_context=evidence,
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
        )

        Draft202012Validator(self.INITIAL_DRAFT_SCHEMA).validate(document)
        self.assertEqual(document["id"], "package-id")
        self.assertIn("creator", document)
        self.assertIn("dataset_distribution", document)
        self.assertIn("is_about_entity", document)
        self.assertIn("is_about_activity", document)
        self.assertIn("was_generated_by", document)
        self.assertIn("description", document["was_generated_by"][0])
        self.assertIn("has_qualitative_attribute", document["was_generated_by"][0])
        self.assertIn("format", document["dataset_distribution"][0])
        self.assertIn("has_quantitative_attribute", document["is_about_activity"][0])
        paths = {entry["path"] for entry in scaffold["entries"]}
        self.assertIn("/creator/0", paths)
        self.assertIn("/dataset_distribution/0", paths)
        self.assertIn("/is_about_entity/0", paths)
        self.assertIn("/was_generated_by/0/description", paths)

    def test_initial_draft_prunes_only_untouched_optional_scaffold(self):
        document, scaffold = ExtractionService._initial_profile_document(
            data_package_id="package-id",
            evidence_context=EvidenceContext(notes=[]),
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
        )
        document["creator"][0]["name"] = ["Bruker BioSpin GmbH"]

        pruned = ExtractionService._prune_initial_draft_scaffold(document, scaffold)

        self.assertIn("creator", pruned)
        self.assertNotIn("dataset_distribution", pruned)
        self.assertNotIn("is_about_entity", pruned)
        self.assertIn("was_generated_by", pruned)
        self.assertNotIn("description", pruned["was_generated_by"][0])
        Draft202012Validator(self.INITIAL_DRAFT_SCHEMA).validate(pruned)

    def test_target_catalog_marks_description_as_last_resort_and_scaffold_status(self):
        document, scaffold = ExtractionService._initial_profile_document(
            data_package_id="package-id",
            evidence_context=EvidenceContext(notes=[]),
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
        )

        catalog = ExtractionService._target_catalog_from_document(
            document=document,
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
            scaffold=scaffold,
        )
        by_path = {item["path"]: item for item in catalog}

        self.assertTrue(by_path["/description"]["description_last_resort"])
        self.assertEqual(by_path["/dataset_distribution/0"]["scaffold_status"], "unfilled")
        self.assertIn("resource_signal", by_path["/dataset_distribution/0"]["category_affinities"])
        self.assertIn("current_value", by_path["/was_generated_by/0"])

    def test_target_object_rewrite_replaces_only_selected_target(self):
        document, _scaffold = ExtractionService._initial_profile_document(
            data_package_id="package-id",
            evidence_context=EvidenceContext(notes=[]),
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
        )
        replacement = {
            "access_URL": [{"id": "package-id:distribution:primary:access"}],
            "title": ["Primary JCAMP-DX distribution"],
            "description": ["JCAMP-DX spectral data files."],
            "format": None,
            "media_type": None,
        }

        updated = ExtractionService._replace_json_pointer(
            document,
            "/dataset_distribution/0",
            replacement,
        )

        self.assertEqual(updated["dataset_distribution"][0]["title"], ["Primary JCAMP-DX distribution"])
        self.assertEqual(document["dataset_distribution"][0]["title"], [])
        Draft202012Validator(self.INITIAL_DRAFT_SCHEMA).validate(updated)

    def test_profile_target_write_document_carries_complete_target_value(self):
        write = ProfileTargetWriteDocument(
            status="write",
            value={"name": ["Bruker BioSpin GmbH"]},
            reason="Creator evidence.",
        )

        self.assertEqual(write.status, "write")
        self.assertEqual(write.value["name"], ["Bruker BioSpin GmbH"])

    def test_target_writer_coerces_scalar_strings_to_existing_array_shape(self):
        current = {
            "access_URL": [{"id": "package-id:distribution:primary:access"}],
            "title": [],
            "description": [],
            "format": None,
            "media_type": None,
        }

        value = ExtractionService._coerce_profile_target_value(
            target_path="/dataset_distribution/0",
            current_value=current,
            proposed_value={
                "access_URL": [{"id": "package-id:distribution:primary:access"}],
                "title": "Primary JCAMP-DX distribution",
                "description": "JCAMP-DX spectral data files.",
                "format": "JCAMP-DX",
            },
        )

        self.assertEqual(value["title"], ["Primary JCAMP-DX distribution"])
        self.assertEqual(value["description"], ["JCAMP-DX spectral data files."])
        self.assertIsNone(value["format"])

    def test_deterministic_keyword_fallback_skips_raw_parameter_settings(self):
        notes = [
            EvidenceNote(
                note_id="td_setting",
                category="method_signal",
                observation="NMR acquisition parameter TD is 65536.",
                evidence_text="##$TD= 65536",
            )
        ]

        value = ExtractionService._fallback_profile_target_value(
            target_path="/keyword",
            current_value=[],
            notes=notes,
        )

        self.assertIsNone(value)

    def test_low_level_bruker_hardware_parameters_are_not_profile_targets(self):
        notes = [
            EvidenceNote(
                note_id="note_15",
                category="method_signal",
                observation="Bla01Eth parameter is set to '<149.236.99.254>'.",
                evidence_text="##$Bla01Eth= <149.236.99.254>",
                file_path="10.zip/10/uxnmr.par",
            ),
            EvidenceNote(
                note_id="note_16",
                category="method_signal",
                observation="Bla01Nam parameter is set to '<BLAXH300/100 E 200-600MHZ INR>'.",
                evidence_text="##$Bla01Nam= <BLAXH300/100 E 200-600MHZ INR>",
                file_path="10.zip/10/uxnmr.par",
            ),
        ]

        self.assertTrue(all(ExtractionService._is_low_level_parameter_note(note) for note in notes))
        reason = ExtractionService._profile_target_unsuitable_reason(
            target_path="/keyword",
            notes=notes,
        )

        self.assertIsNotNone(reason)
        self.assertIsNone(
            ExtractionService._fallback_profile_target_value(
                target_path="/keyword",
                current_value=[],
                notes=notes,
            )
        )

    def test_deterministic_fallback_keeps_useful_method_and_resource_targets(self):
        method_value = ExtractionService._fallback_profile_target_value(
            target_path="/was_generated_by/0",
            current_value={
                "id": "package-id:activity:metadata-extraction",
                "title": [],
                "description": [],
                "has_qualitative_attribute": [],
                "has_quantitative_attribute": [],
                "evaluated_activity": [],
                "evaluated_entity": [],
            },
            notes=[
                EvidenceNote(
                    note_id="pulprog_setting",
                    category="method_signal",
                    observation="Pulse program is zg30.",
                    evidence_text="##$PULPROG= <zg30>",
                )
            ],
        )
        distribution_value = ExtractionService._fallback_profile_target_value(
            target_path="/dataset_distribution/0",
            current_value={
                "access_URL": [{"id": "package-id:distribution:primary:access"}],
                "title": [],
                "description": [],
                "format": None,
                "media_type": None,
            },
            notes=[
                EvidenceNote(
                    note_id="jcamp_file",
                    category="resource_signal",
                    observation="The file is a JCAMP-DX NMR spectral export.",
                    evidence_text="##TITLE= JCAMP-DX NMR spectrum",
                    file_path="10.edit.jdx",
                )
            ],
        )

        self.assertIsNotNone(method_value)
        self.assertIn({"title": "pulprog", "value": "zg30"}, method_value["has_qualitative_attribute"])
        self.assertIsNotNone(distribution_value)
        self.assertIn("Primary NMR data distribution", distribution_value["title"])

    def test_final_profile_cleanup_removes_parameter_noise_but_keeps_nmr_signals(self):
        document = {
            "id": "package-id",
            "title": ["1H NMR"],
            "description": [
                "SIMONE metadata draft for 1H NMR.",
                "TD parameter set to 65536.",
                "Transmitter routing uses TCP/IP 149.236.99.254.",
            ],
            "keyword": [
                "NMR Spectroscopy",
                "TD parameter set to 65536",
                "Bla01Eth parameter is set to '<149.236.99.254>'",
                "JCAMP-DX",
            ],
            "dataset_distribution": [
                {
                    "access_URL": [{"id": "package-id:distribution:primary:access"}],
                    "title": ["Primary NMR data distribution"],
                    "description": [
                        "JCAMP-DX spectral data files.",
                        "NPOINTS parameter values from the Bruker file.",
                        "Nucleus FPNZ is no",
                        "This is a parameter file from TOPSPIN software version 3.2",
                    ],
                }
            ],
            "was_generated_by": [
                {
                    "id": "package-id:activity:metadata-extraction",
                    "has_qualitative_attribute": [
                        {
                            "value": "Pulse sequence is zg30",
                            "description": "Pulse sequence is zg30",
                        },
                        {
                            "value": "SOLVENT OFF setting",
                            "description": "##$SOLVOLD= <off>",
                        },
                        {
                            "value": "Instrument parameter structure",
                            "description": "NAME\tINSTRUM\n\tFORMAT\t\"\"",
                        },
                    ],
                }
            ],
        }

        curated = ExtractionService._curate_generated_profile_document(document)

        self.assertEqual(curated["description"], ["SIMONE metadata draft for 1H NMR."])
        self.assertEqual(curated["keyword"], ["NMR Spectroscopy", "JCAMP-DX"])
        self.assertEqual(
            curated["dataset_distribution"][0]["description"],
            ["JCAMP-DX spectral data files."],
        )
        self.assertEqual(
            curated["was_generated_by"][0]["has_qualitative_attribute"],
            [{"value": "Pulse sequence is zg30", "description": "Pulse sequence is zg30"}],
        )

    def test_fallback_title_prefers_explicit_dataset_name_over_audit_noise(self):
        evidence = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="audit_trail",
                    category="resource_signal",
                    observation="Audit trail records software version and user actions",
                    evidence_text="##AUDIT TRAIL= ... TOPSPIN 3.2",
                ),
                EvidenceNote(
                    note_id="dataset_name",
                    category="resource_signal",
                    observation="Dataset name is 1H NMR",
                    evidence_text="dataset name: 1H NMR",
                ),
            ]
        )

        title = ExtractionService._fallback_title("package-id", evidence)

        self.assertEqual(title, "1H NMR")

    def test_projection_groups_skip_non_curatable_parameter_notes(self):
        evidence = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="dataset_name",
                    category="resource_signal",
                    observation="Dataset name is 1H NMR",
                    evidence_text="dataset name: 1H NMR",
                    profile_worthiness="high",
                ),
                EvidenceNote(
                    note_id="bfreq_setting",
                    category="method_signal",
                    observation="BFREQ parameter is set to 500.13.",
                    evidence_text="##$BFREQ= 500.13",
                    profile_worthiness="high",
                ),
                EvidenceNote(
                    note_id="blocks",
                    category="resource_signal",
                    observation="single block structure",
                    evidence_text="##BLOCKS=1",
                    profile_worthiness="high",
                ),
            ]
        )

        groups = ExtractionService._projection_groups_for_evidence(evidence)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].target_hint, "/title")

    def test_title_cleanup_removes_spectrum_local_numeric_title(self):
        value = ExtractionService._curate_profile_target_value(
            target_path="/title",
            value=["Dataset name is 1H NMR", "10", "spectrum title"],
        )

        self.assertEqual(value, ["1H NMR"])

    def test_projection_grouping_collapses_repeated_note_families(self):
        evidence = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="jcamp_dx_version_1",
                    category="resource_signal",
                    observation="JCAMP-DX format version 5.00",
                    evidence_text="##JCAMP-DX=5.00",
                    file_path="10.edit.jdx",
                    start_idx=10,
                    end_idx=20,
                    profile_worthiness="high",
                ),
                EvidenceNote(
                    note_id="jcamp_dx_version_2",
                    category="resource_signal",
                    observation="JCAMP-DX format version 5.00",
                    evidence_text="##JCAMP-DX=5.00",
                    file_path="10.edit.jdx",
                    start_idx=10,
                    end_idx=20,
                    profile_worthiness="high",
                ),
            ]
        )

        groups = ExtractionService._projection_groups_for_evidence(evidence)

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].notes), 2)
        self.assertTrue(groups[0].group_id.startswith("group:dataset_distribution.0:"))

    def test_patch_path_constraint_rejects_description_sink_for_specific_target(self):
        self.assertTrue(
            ExtractionService._patch_path_allowed_for_target(
                "/dataset_distribution/0/title/-",
                "/dataset_distribution/0",
            )
        )
        self.assertFalse(
            ExtractionService._patch_path_allowed_for_target(
                "/description/-",
                "/dataset_distribution/0",
            )
        )

    def test_group_projection_ledger_preserves_target_and_note_ids(self):
        evidence = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="nucleus",
                    category="entity_signal",
                    observation="Primary nucleus is 1H.",
                    evidence_text="NUC1= <1H>",
                    file_path="10/acqus",
                    start_idx=1,
                    end_idx=2,
                )
            ]
        )
        group = ExtractionService._projection_groups_for_evidence(evidence)[0]
        record = ExtractionService._projection_record_from_group_patch_result(
            group=group,
            patch_result=ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="applied",
                target_path="/is_about_entity/0",
                target_class="EvaluatedEntity",
                planner_status="targeted",
                planner_reason="Entity evidence.",
                target_value={"id": "entity:primary", "title": "1H nucleus"},
            ),
        )

        self.assertEqual(record.status, "projected")
        self.assertEqual(record.target_path, "/is_about_entity/0")
        self.assertEqual(record.target_class, "EvaluatedEntity")
        self.assertEqual(record.planner_status, "targeted")
        self.assertEqual(record.projected_paths, ["/is_about_entity/0"])
        self.assertEqual(record.evidence_quality["note_count"], 1)
        self.assertEqual(len(record.evidence_note_identifiers), 1)
        self.assertIn("10/acqus#1-2#nucleus", record.evidence_note_identifiers)
if __name__ == "__main__":
    unittest.main()
