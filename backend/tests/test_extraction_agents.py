import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from app.domain.extraction import (
    DefinedTerm,
    EvidenceContext,
    EvidenceAssessment,
    EvidenceChunkContext,
    EvidenceChunkMetadata,
    EvidenceNote,
    EVIDENCE_CONTEXT_SYSTEM_PROMPT,
    ExtractionContext,
    ExtractionFileSummary,
    ExtractionNormalization,
    ExtractionOverview,
    FileContext,
    FileRankingResult,
    GroundedExtractionObject,
    ProfileObjectPatchResult,
    ProfileTargetWriteDocument,
    PromptTokenBudgeter,
    QuantitativeAttribute,
    RankedFile,
    Resource,
    TracedExtractionObject,
    build_evidence_context_prompt,
    build_evidence_system_prompt_with_overview,
    build_schema_branch_index,
    build_schema_search_query,
    dedupe_repeated_evidence_notes,
    evidence_text_match_score,
    route_evidence_candidates,
    build_file_ranking_prompt,
    is_noisy_payload_chunk,
    build_object_grounding_selection_prompt,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    fallback_file_ranking,
    normalize_chunk_text_for_evidence_prompt,
    ShallowDatasetProjection,
    ShallowDatasetLevelProjection,
    ShallowDistributionProjection,
    ShallowResourceProjection,
    build_dataset_level_projection_prompt_components,
    build_dataset_summary_prompt_components,
    build_overview_shallow_projection_prompt_components,
    compact_file_summaries_for_shallow_projection,
    compact_overview_for_shallow_projection,
    dataset_level_to_shallow_projection,
    deterministic_grouped_distributions,
    shallow_projection_to_dcat_document,
    shallow_required_skeleton,
    search_schema_branches,
    validate_evidence_context_for_chunk,
)
from app.services.extraction_service import ExtractionService


class ExtractionDomainTests(unittest.TestCase):
    SCHEMA_GUIDED_LINKML = """
id: https://example.org/test-profile
name: test_profile
prefixes:
  ex:
    prefix_prefix: ex
    prefix_reference: https://example.org/
default_prefix: ex
default_range: string
slots:
  id:
    range: uriorcurie
    required: true
  title:
    range: string
    multivalued: true
  description:
    range: string
    multivalued: true
  was_generated_by:
    range: DataGeneratingActivity
    multivalued: true
    inlined_as_list: true
  carried_out_by:
    description: AgenticEntity that played a part in carrying out the activity.
    range: AgenticEntity
    recommended: true
    multivalued: true
    inlined_as_list: true
  has_qualitative_attribute:
    range: QualitativeAttribute
    multivalued: true
    inlined_as_list: true
  has_quantitative_attribute:
    range: QuantitativeAttribute
    multivalued: true
    inlined_as_list: true
  type:
    range: DefinedTerm
  rdf_type:
    range: DefinedTerm
    recommended: true
  value:
    range: string
  exact:
    range: string
classes:
  Dataset:
    slots:
      - id
      - title
      - description
      - was_generated_by
  Activity:
    slots:
      - id
      - title
      - description
      - carried_out_by
  DataGeneratingActivity:
    is_a: Activity
    description: Activity that generates data.
  AgenticEntity:
    description: An entity responsible for an activity.
    slots:
      - id
      - title
      - description
      - has_qualitative_attribute
      - has_quantitative_attribute
      - type
      - rdf_type
  Device:
    is_a: AgenticEntity
    description: A material instrument that is designed to perform a function primarily by mechanical or electrical nature.
    aliases:
      - hardware instrument
    exact_mappings:
      - OBI:0000968
  DefinedTerm:
    slots:
      - id
      - title
  QualitativeAttribute:
    slots:
      - title
      - value
  QuantitativeAttribute:
    slots:
      - title
      - value
"""

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
                    "carried_out_by": {"type": "array", "items": {"$ref": "#/$defs/AgenticEntity"}},
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
            "AgenticEntity": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "rdf_type": {"type": ["object", "null"]},
                    "type": {"type": ["object", "null"]},
                    "has_qualitative_attribute": {"type": "array"},
                    "has_quantitative_attribute": {"type": "array"},
                    "has_part": {"type": "array"},
                    "part_of": {"type": "array"},
                    "other_identifier": {"type": "array"},
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

    def test_evidence_candidate_tracks_grounding_separately_from_quality(self):
        chunk = "title: Sample A\nmethod: calibration experiment"
        context = EvidenceContext(
            candidates=[
                EvidenceNote(
                    candidate_id="method_group",
                    category="method_signal",
                    claim="The resource states a calibration experiment method.",
                    evidence_text="method: calibration experiment",
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
        self.assertFalse(hasattr(validated.notes[0], "signal_level"))
        self.assertFalse(hasattr(validated.notes[0], "interpretation_confidence"))
        self.assertFalse(hasattr(validated.notes[0], "profile_worthiness"))

    def test_evidence_prompt_discourages_boilerplate_and_requires_quality(self):
        self.assertIn("high-recall", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("Do not decide whether a candidate is valuable", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("signal_level", EVIDENCE_CONTEXT_SYSTEM_PROMPT)

    def test_evidence_prompt_prevents_file_local_signals_from_becoming_dataset_identity(self):
        self.assertIn("scope to package, resource, section, execution_environment, or unknown", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("A later critic will route them", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("TOPSPIN", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("Bruker", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("NPOINTS", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("Counterexamples", EVIDENCE_CONTEXT_SYSTEM_PROMPT)

    def test_evidence_prompt_is_domain_agnostic(self):
        self.assertIn("Do not use domain-specific key names", EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertNotIn("nucleus", EVIDENCE_CONTEXT_SYSTEM_PROMPT.lower())
        self.assertNotIn("solvent", EVIDENCE_CONTEXT_SYSTEM_PROMPT.lower())
        self.assertNotIn("pulse", EVIDENCE_CONTEXT_SYSTEM_PROMPT.lower())

    def test_evidence_chunk_prompt_normalizes_control_characters(self):
        prompt = build_evidence_context_prompt(
            EvidenceChunkContext(
                content='title: local resource\t\tlabel\r\n<path "C:/Research Data">\x00',
                metadata=EvidenceChunkMetadata(
                    start_idx=0,
                    end_idx=2,
                    file_path="metadata.txt",
                    data_package_name="package",
                ),
            )
        )

        self.assertIn("title: local resource label", prompt)
        self.assertIn('<path "C:/Research Data">', prompt)
        self.assertNotIn("\r", prompt)
        self.assertNotIn("\t", prompt)
        self.assertNotIn("\x00", prompt)

    def test_prompt_token_budgeter_truncates_by_token_offsets(self):
        class FakeEncoding:
            def __init__(self, text: str):
                matches = list(re.finditer(r"\S+", text))
                self.ids = list(range(len(matches)))
                self.offsets = [(match.start(), match.end()) for match in matches]

        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return FakeEncoding(text)

        budgeter = PromptTokenBudgeter(tokenizer=FakeTokenizer())

        truncated = budgeter.truncate(
            "alpha beta gamma delta epsilon",
            max_tokens=4,
        )

        self.assertEqual(truncated, "alpha beta\n[orientation truncated]")
        self.assertLessEqual(budgeter.count(truncated), 4)

    def test_prompt_token_budgeter_falls_back_to_estimated_tokens(self):
        budgeter = PromptTokenBudgeter.from_tokenizer_source("")

        truncated = budgeter.truncate("x" * 200, max_tokens=20)

        self.assertTrue(budgeter.uses_fallback)
        self.assertIsNotNone(budgeter.fallback_reason)
        self.assertIn("[orientation truncated]", truncated)
        self.assertLessEqual(budgeter.count(truncated), 20)

    def test_prompt_token_budgeter_reuses_loaded_tokenizer(self):
        class FakeEncoding:
            ids = [1]

        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return FakeEncoding()

        from app.domain import token_budget

        token_budget._load_tokenizer.cache_clear()
        try:
            with patch("app.domain.token_budget.Tokenizer.from_pretrained", return_value=FakeTokenizer()) as loader:
                first = PromptTokenBudgeter.from_tokenizer_source(
                    "example/tokenizer",
                    hf_token="hf_test_token",
                )
                second = PromptTokenBudgeter.from_tokenizer_source(
                    "example/tokenizer",
                    hf_token="hf_test_token",
                )

            self.assertEqual(first.count("alpha"), 1)
            self.assertEqual(second.count("beta"), 1)
            loader.assert_called_once_with("example/tokenizer", token="hf_test_token")
        finally:
            token_budget._load_tokenizer.cache_clear()

    def test_evidence_system_prompt_uses_token_orientation_budget(self):
        class FakeEncoding:
            def __init__(self, text: str):
                matches = list(re.finditer(r"\S+", text))
                self.ids = list(range(len(matches)))
                self.offsets = [(match.start(), match.end()) for match in matches]

        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return FakeEncoding(text)

        budgeter = PromptTokenBudgeter(tokenizer=FakeTokenizer())
        overview = ExtractionOverview(
            source_file_paths=["metadata.txt", "other.txt"],
            nodes=[
                {
                    "node_id": "file:metadata.txt",
                    "label": "metadata.txt",
                    "kind": "file",
                    "file_path": "metadata.txt",
                    "summary": "resource metadata " + "x " * 80,
                },
                {
                    "node_id": "file:other.txt",
                    "label": "other.txt",
                    "kind": "file",
                    "file_path": "other.txt",
                    "summary": "should not dominate chunk prompt",
                },
                {
                    "node_id": "group:metadata",
                    "label": "resource metadata",
                    "kind": "group",
                    "summary": "contains file-local labels and timestamp metadata",
                },
            ],
            edges=[
                {
                    "source": "group:metadata",
                    "target": "file:metadata.txt",
                    "relation": "describes",
                    "note": "contains file-local labels and timestamp metadata",
                }
            ],
            uncertainties=[f"Uncertainty {index} " + "z " * 80 for index in range(12)],
        )
        summary = ExtractionFileSummary(
            file_path="metadata.txt",
            status="summarized",
            data_format="text",
            metadata_signals=[
                "metadata",
                "file-local labels",
                "local title",
                "generic owner",
                "timestamp",
            ],
        )

        prompt = build_evidence_system_prompt_with_overview(
            base_prompt=EVIDENCE_CONTEXT_SYSTEM_PROMPT,
            overview=overview,
            overview_status="structured",
            file_summary=summary,
            token_budgeter=budgeter,
            max_overview_tokens=60,
            max_file_summary_tokens=40,
        )

        self.assertIn("Current file graph neighborhood", prompt)
        self.assertIn("file:metadata.txt", prompt)
        self.assertNotIn("file:other.txt", prompt)
        orientation = prompt.removeprefix(EVIDENCE_CONTEXT_SYSTEM_PROMPT)
        self.assertIn("[orientation truncated]", orientation)
        self.assertLessEqual(
            budgeter.count(orientation),
            110,
        )

    def test_route_evidence_candidates_uses_generic_assessment(self):
        context = EvidenceContext(
            candidates=[
                EvidenceNote(candidate_id="portable", category="resource_signal", claim="Dataset title is Sample A.", evidence_text="Dataset title: Sample A"),
                EvidenceNote(candidate_id="contextual", category="method_signal", claim="A local runtime path is present.", evidence_text="Path: C:/tmp/run"),
            ]
        )

        routed = route_evidence_candidates(
            context,
            assessments=[
                EvidenceAssessment(
                    candidate_id="portable",
                    groundedness="yes",
                    self_containedness="yes",
                    scope_clarity="yes",
                    portability="yes",
                    semantic_interpretability="yes",
                    environment_dependence="low",
                    specificity="yes",
                    novelty="yes",
                    uncertainty="low",
                ),
                EvidenceAssessment(
                    candidate_id="contextual",
                    groundedness="yes",
                    self_containedness="partial",
                    scope_clarity="no",
                    portability="no",
                    semantic_interpretability="partial",
                    environment_dependence="high",
                    specificity="partial",
                    novelty="partial",
                    uncertainty="medium",
                ),
            ],
            chunk_index=2,
        )

        self.assertEqual([note.candidate_id for note in routed.portable_evidence], ["portable"])
        self.assertEqual([note.candidate_id for note in routed.contextual_evidence], ["contextual"])
        self.assertEqual(routed.rejected_evidence, [])

    def test_repeated_evidence_dedupe_keeps_best_representative(self):
        context = EvidenceContext(
            notes=[
                EvidenceNote(
                    note_id="secondary",
                    category="resource_signal",
                    observation="Format.",
                    evidence_text="##JCAMP-DX=5.00",
                    file_path="rank2.jdx",
                    signal_level="high",
                ),
                EvidenceNote(
                    note_id="primary",
                    category="resource_signal",
                    observation="JCAMP-DX file syntax format version 5.00 is declared.",
                    evidence_text="##JCAMP-DX=5.00",
                    file_path="rank1.jdx",
                    signal_level="high",
                ),
            ]
        )

        deduped, dropped = dedupe_repeated_evidence_notes(
            context,
            file_rank_by_path={"rank1.jdx": 1, "rank2.jdx": 2},
        )

        self.assertEqual([note.note_id for note in deduped.notes], ["primary"])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].reason, "duplicate_evidence")
        self.assertEqual(dropped[0].duplicate_representative_id, "primary")

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
        self.assertIn("/was_generated_by/0/carried_out_by/-", by_path)

    def test_schema_branch_index_exposes_device_candidate_for_carried_out_by(self):
        branches = build_schema_branch_index(
            self.SCHEMA_GUIDED_LINKML,
            target_class="Dataset",
            max_depth=3,
        )
        carried_out_by = next(
            branch
            for branch in branches
            if branch.path == "/was_generated_by/0/carried_out_by/-"
        )

        self.assertEqual(carried_out_by.range_class, "AgenticEntity")
        device = next(
            candidate
            for candidate in carried_out_by.subclass_candidates
            if candidate.class_name == "Device"
        )
        self.assertIn("hardware instrument", device.aliases)
        self.assertIn("OBI:0000968", device.exact_mappings)

    def test_schema_search_prefers_device_participant_branch(self):
        note = EvidenceNote(
            note_id="instrument",
            category="resource_signal",
            observation="Instrument used is Bruker Avance 500 MHz.",
            evidence_text="instrument: Bruker Avance 500 MHz",
        )
        branches = build_schema_branch_index(
            self.SCHEMA_GUIDED_LINKML,
            target_class="Dataset",
            max_depth=3,
        )
        result = search_schema_branches(
            branches,
            build_schema_search_query([note], max_depth=3),
            top_k=3,
        )

        self.assertEqual(result.candidates[0].path, "/was_generated_by/0/carried_out_by/-")

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

    def test_instrument_note_uses_free_text_observation_without_facets(self):
        note = EvidenceNote(
            note_id="instrument",
            category="agent_signal",
            observation="Instrument/device used is Bruker Avance 500 MHz.",
            evidence_text="instrument: Bruker Avance 500 MHz",
        )

        self.assertEqual(note.category, "agent_signal")
        self.assertIn("Instrument/device", note.observation)
        self.assertFalse(hasattr(note, "facets"))

    def test_low_level_parameters_only_get_parameter_schema_hint(self):
        note = EvidenceNote(
            note_id="td_setting",
            category="method_signal",
            observation="Low-level parameter TD is set to 65536.",
            evidence_text="##$TD= 65536",
        )
        query = build_schema_search_query([note], max_depth=3)

        self.assertEqual(query.semantic_hints, ["parameter_setting"])

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
                    note_id="method_setting",
                    category="method_signal",
                    observation="The workflow uses a calibration method.",
                    evidence_text="method = calibration",
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
                    note_id="format_file",
                    category="resource_signal",
                    observation="The file declares a structured data format.",
                    evidence_text="format = structured text",
                    file_path="data.txt",
                )
            ],
        )

        self.assertIsNotNone(method_value)
        self.assertIn("The workflow uses a calibration method.", method_value["description"])
        self.assertIsNotNone(distribution_value)
        self.assertIn("Primary dataset distribution", distribution_value["title"])

    def test_device_fallback_writes_agentic_entity_not_qualitative_attribute(self):
        note = EvidenceNote(
            note_id="instrument",
            category="resource_signal",
            observation="Instrument used is Bruker Avance 500 MHz.",
            evidence_text="instrument: Bruker Avance 500 MHz",
            signal_level="high",
        )
        value = ExtractionService._fallback_profile_target_value(
            target_path="/was_generated_by/0/carried_out_by/-",
            current_value=None,
            notes=[note],
        )

        self.assertIsNotNone(value)
        self.assertEqual(value["title"], "Bruker Avance 500 MHz")
        self.assertEqual(value["rdf_type"]["id"], "http://purl.obolibrary.org/obo/OBI_0000968")

    def test_schema_branch_merge_appends_and_dedupes_device_participants(self):
        document, _scaffold = ExtractionService._initial_profile_document(
            data_package_id="package-id",
            evidence_context=EvidenceContext(notes=[]),
            validation_schema=self.INITIAL_DRAFT_SCHEMA,
        )
        device = {
            "id": "device:bruker-avance-500-mhz",
            "title": "Bruker Avance 500 MHz",
            "description": "Instrument associated with the data-generating activity.",
            "rdf_type": {"id": "http://purl.obolibrary.org/obo/OBI_0000968", "title": "device"},
            "type": {"id": "http://purl.obolibrary.org/obo/OBI_0000968", "title": "device"},
            "has_qualitative_attribute": [],
            "has_quantitative_attribute": [],
            "has_part": [],
            "part_of": [],
            "other_identifier": [],
        }

        updated = ExtractionService._apply_profile_target_write(
            document,
            "/was_generated_by/0/carried_out_by/-",
            device,
        )
        updated_again = ExtractionService._apply_profile_target_write(
            updated,
            "/was_generated_by/0/carried_out_by/-",
            device,
        )

        self.assertEqual(len(updated_again["was_generated_by"][0]["carried_out_by"]), 1)
        self.assertEqual(updated_again["was_generated_by"][0]["has_qualitative_attribute"], [])
        Draft202012Validator(self.INITIAL_DRAFT_SCHEMA).validate(updated_again)

    def test_final_profile_cleanup_removes_parameter_noise_but_keeps_generic_metadata(self):
        document = {
            "id": "package-id",
            "title": ["Catalyst measurements"],
            "description": [
                "SIMONE metadata draft for catalyst measurements.",
                "TD parameter set to 65536.",
                "Transmitter routing uses TCP/IP 149.236.99.254.",
            ],
            "keyword": [
                "measurement",
                "TD parameter set to 65536",
                "Bla01Eth parameter is set to '<149.236.99.254>'",
                "dataset",
            ],
            "dataset_distribution": [
                {
                    "access_URL": [{"id": "package-id:distribution:primary:access"}],
                    "title": ["Primary dataset distribution"],
                    "description": [
                        "Structured measurement data files.",
                        "NPOINTS parameter values from the Bruker file.",
                        "Nucleus FPNZ is no",
                        "This is a parameter file from installed software version 3.2",
                    ],
                }
            ],
            "was_generated_by": [
                {
                    "id": "package-id:activity:metadata-extraction",
                    "has_qualitative_attribute": [
                        {
                            "value": "Calibration method",
                            "description": "Calibration method",
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

        self.assertEqual(curated["description"], ["SIMONE metadata draft for catalyst measurements."])
        self.assertEqual(curated["keyword"], ["measurement", "dataset"])
        self.assertEqual(
            curated["dataset_distribution"][0]["description"],
            ["Structured measurement data files."],
        )
        self.assertEqual(
            curated["was_generated_by"][0]["has_qualitative_attribute"],
            [{"value": "Calibration method", "description": "Calibration method"}],
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
                    signal_level="high",
                ),
                EvidenceNote(
                    note_id="bfreq_setting",
                    category="method_signal",
                    observation="BFREQ parameter is set to 500.13.",
                    evidence_text="##$BFREQ= 500.13",
                    signal_level="high",
                ),
                EvidenceNote(
                    note_id="blocks",
                    category="resource_signal",
                    observation="single block structure",
                    evidence_text="##BLOCKS=1",
                    signal_level="high",
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
                    note_id="format_version_1",
                    category="resource_signal",
                    observation="Structured data format version 5.00",
                    evidence_text="FORMAT=5.00",
                    file_path="data.txt",
                    start_idx=10,
                    end_idx=20,
                    signal_level="high",
                ),
                EvidenceNote(
                    note_id="format_version_2",
                    category="resource_signal",
                    observation="Structured data format version 5.00",
                    evidence_text="FORMAT=5.00",
                    file_path="data.txt",
                    start_idx=10,
                    end_idx=20,
                    signal_level="high",
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
                    note_id="sample",
                    category="entity_signal",
                    observation="Primary sample is catalyst batch A.",
                    evidence_text="sample = catalyst batch A",
                    file_path="metadata.txt",
                    start_idx=1,
                    end_idx=2,
                    signal_level="high",
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
                target_value={"id": "entity:primary", "title": "catalyst batch A"},
            ),
        )

        self.assertEqual(record.status, "projected")
        self.assertEqual(record.target_path, "/is_about_entity/0")
        self.assertEqual(record.target_class, "EvaluatedEntity")
        self.assertEqual(record.planner_status, "targeted")
        self.assertEqual(record.projected_paths, ["/is_about_entity/0"])
        self.assertEqual(record.evidence_quality["note_count"], 1)
        self.assertEqual(len(record.evidence_note_identifiers), 1)
        self.assertIn("metadata.txt#1-2#sample", record.evidence_note_identifiers)

    def test_shallow_projection_compact_serialization_uses_overview_and_summaries(self):
        overview = ExtractionOverview(
            source_file_paths=["README.md"],
            nodes=[
                {
                    "node_id": "file:README.md",
                    "label": "README.md",
                    "kind": "file",
                    "file_path": "README.md",
                    "summary": "A" * 500,
                }
            ],
            edges=[
                {
                    "source": "file:README.md",
                    "target": "group:metadata",
                    "relation": "documents",
                    "evidence": ["dataset description"],
                    "note": "README documents the dataset.",
                }
            ],
            uncertainties=["No raw evidence should be included."],
        )
        summaries = [
            ExtractionFileSummary(
                file_path="README.md",
                data_format="plain text",
                explicit_purpose="dataset description",
                purpose_evidence=["Dataset description"],
                metadata_signals=["1H NMR archive"],
                instrument_or_software_terms_and_settings=["Bruker Avance"],
                quantitative_signals=["500 MHz"],
            )
        ]

        compact_overview = compact_overview_for_shallow_projection(overview)
        compact_summaries = compact_file_summaries_for_shallow_projection(
            initial_file_summaries=summaries,
            ranked_files=[RankedFile(rank=1, file_path="README.md")],
        )
        summary_components = dict(
            build_dataset_summary_prompt_components(
                data_package_id="package-id",
                initial_file_summaries=summaries,
                ranked_files=[RankedFile(rank=1, file_path="README.md")],
            )
        )
        draft_components = dict(
            build_dataset_level_projection_prompt_components(
                data_package_id="package-id",
                dataset_summary="evaluated: dataset archive. generated_by: spectroscopy activity.",
                skeleton=shallow_required_skeleton("package-id", "1H NMR"),
            )
        )

        self.assertLessEqual(len(compact_overview["nodes"][0]["summary"]), 260)
        self.assertIn("r1|README.md", compact_summaries)
        self.assertIn("file_summaries", summary_components["dataset_summary_input_json"])
        self.assertNotIn('"overview"', summary_components["dataset_summary_input_json"])
        self.assertNotIn("raw_content", summary_components["dataset_summary_input_json"])
        self.assertIn("dataset_summary", draft_components["dataset_level_projection_input_json"])
        self.assertIn("required_skeleton", draft_components["dataset_level_projection_input_json"])
        self.assertNotIn("file_summaries", draft_components["dataset_level_projection_input_json"])

    def test_shallow_projection_model_accepts_broad_dataset_projection(self):
        projection = ShallowDatasetProjection.model_validate(
            {
                "title": ["1H NMR"],
                "description": ["A proton NMR dataset with acquisition and processing files."],
                "keyword": ["NMR", "spectroscopy"],
                "dataset_distribution": [
                    {
                        "access_URL": [{"id": "10.zip/10/acqu", "title": "acqu"}],
                        "title": ["Acquisition parameters"],
                    }
                ],
                "was_generated_by": [{"title": ["NMR acquisition"]}],
                "is_about_entity": [{"title": "sample"}],
            }
        )

        self.assertEqual(projection.title, ["1H NMR"])
        self.assertEqual(projection.dataset_distribution[0].access_URL[0].title, "acqu")

    def test_dataset_level_projection_excludes_llm_distributions_and_merges_backend_distributions(self):
        level_projection = ShallowDatasetLevelProjection.model_validate(
            {
                "id": "package-id",
                "title": ["1H NMR dataset"],
                "description": ["A compact dataset-level account of a proton NMR acquisition."],
                "keyword": ["NMR"],
            }
        )
        projection = dataset_level_to_shallow_projection(
            level_projection,
            distributions=[
                ShallowDistributionProjection(
                    access_URL=[ShallowResourceProjection(id="10.zip/10/fid", title="fid")],
                    title=["Raw data resources"],
                )
            ],
        )

        self.assertEqual(len(projection.dataset_distribution), 1)
        self.assertEqual(projection.dataset_distribution[0].access_URL[0].id, "10.zip/10/fid")

    def test_deterministic_grouped_distributions_map_scientific_file_summaries(self):
        summaries = [
            ExtractionFileSummary(file_path="dataset_description.txt", data_format="text", explicit_purpose="dataset description"),
            ExtractionFileSummary(file_path="10.zip/10/fid", data_format="binary", metadata_signals=["raw spectral data"]),
            ExtractionFileSummary(file_path="10.edit.jdx", data_format="JCAMP-DX", metadata_signals=["processed NMR spectrum"]),
            ExtractionFileSummary(file_path="10.zip/10/acqus", metadata_signals=["acquisition parameter file"]),
            ExtractionFileSummary(file_path="10.zip/10/pdata/1/proc", metadata_signals=["processing parameter file"]),
            ExtractionFileSummary(file_path="10.zip/10/uxnmr.info", explicit_purpose="configuration information"),
            ExtractionFileSummary(file_path="10.zip/10/audita.txt", explicit_purpose="audit trail"),
        ]
        ranked = [RankedFile(rank=index + 1, file_path=summary.file_path) for index, summary in enumerate(summaries)]

        distributions = deterministic_grouped_distributions(
            initial_file_summaries=summaries,
            ranked_files=ranked,
        )
        titles = [item.title[0] for item in distributions]

        self.assertIn("Documentation resources", titles)
        self.assertIn("Raw data resources", titles)
        self.assertIn("Processed data resources", titles)
        self.assertIn("Acquisition parameter resources", titles)
        self.assertIn("Processing parameter resources", titles)
        self.assertIn("Instrument and configuration resources", titles)
        self.assertIn("Audit and metadata resources", titles)

    def test_shallow_projection_fills_missing_ids_and_replaces_description_fallback(self):
        projection = ShallowDatasetProjection(
            title=["1H NMR"],
            description=["A proton NMR dataset with raw and processed Bruker files."],
            dataset_distribution=[
                ShallowDistributionProjection(
                    title=["Acquisition file"],
                    access_URL=[ShallowResourceProjection(title="acqu")],
                )
            ],
            was_generated_by=[],
        )

        document, ledger = shallow_projection_to_dcat_document(
            projection,
            data_package_id="package-id",
            fallback_title="fallback",
        )

        self.assertEqual(document["description"], ["A proton NMR dataset with raw and processed Bruker files."])
        self.assertEqual(document["title"], ["1H NMR"])
        self.assertIn("dataset_distribution", document)
        self.assertTrue(document["was_generated_by"][0]["id"].startswith("package-id:activity"))
        self.assertTrue(any(record.object_kind == "ScaffoldFact" for record in ledger))

    def test_shallow_projection_outputs_schema_valid_dcat_ap_plus_dataset(self):
        schema_path = Path(".runtime/profiles/dcat-ap-plus/json_schema.json")
        if not schema_path.exists():
            self.skipTest("dcat-ap-plus runtime profile is not registered")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        dataset_schema = {"$ref": "#/$defs/Dataset", "$defs": schema["$defs"]}
        projection = ShallowDatasetProjection.model_validate(
            {
                "title": ["1H NMR"],
                "description": ["A proton NMR dataset containing Bruker acquisition and processing resources."],
                "identifier": ["package-id"],
                "keyword": ["NMR"],
                "dataset_distribution": [
                    {
                        "access_URL": [{"id": "10.zip/10/acqu", "title": "acqu"}],
                        "title": ["Acquisition parameters"],
                        "description": ["Bruker acquisition parameter file."],
                    }
                ],
                "was_generated_by": [
                    {
                        "title": ["NMR acquisition"],
                        "description": ["NMR acquisition performed with a Bruker instrument."],
                        "carried_out_by": [
                            {
                                "title": "Bruker Avance 500 MHz",
                                "description": "NMR instrument.",
                            }
                        ],
                    }
                ],
            }
        )
        document, _ledger = shallow_projection_to_dcat_document(
            data_package_id="package-id",
            projection=projection,
            fallback_title="1H NMR",
        )

        errors = sorted(Draft202012Validator(dataset_schema).iter_errors(document), key=str)
        self.assertEqual(errors, [])
        self.assertEqual(document["title"], ["1H NMR"])
        self.assertIn("dataset_distribution", document)
        self.assertIn("carried_out_by", document["was_generated_by"][0])
if __name__ == "__main__":
    unittest.main()

