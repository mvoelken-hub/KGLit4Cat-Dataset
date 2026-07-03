from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from jsonschema import Draft201909Validator
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.extraction import (
    DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS,
    DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS,
    CurationLedgerRecord,
    DcatRequirement,
    EvidenceCandidate,
    RequirementEvaluation,
    RequirementAssessment,
    RequirementPatchAttempt,
    RequirementEvidenceItem,
    RequirementReportItem,
    REQUIREMENT_EVALUATOR_SYSTEM_PROMPT,
    SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT,
    SEMANTIC_SYNTHESIS_SYSTEM_PROMPT,
    SemanticReconstructionDefect,
    SemanticReconstructionDiagnosis,
    SemanticReconstructionRecord,
    SchemaConstrainedWrite,
    apply_schema_constrained_writes,
    build_schema_constrained_patch_schema,
    build_requirement_evaluation_prompt,
    build_requirement_report,
    build_semantic_diagnosis_prompt,
    build_semantic_reconstruction_prompt,
    compute_coverage_report,
    compute_source_trace_report,
    normalized_requirement_evaluation,
    RoutedEvidenceContext,
    report_items_from_evaluation,
    score_requirement_items,
    score_requirement_report,
    semantic_diagnosis_output_schema,
    select_requirement_evidence_packet,
    stable_evidence_id,
)
from app.domain.extraction.workflow import ExtractionRunProgress, ExtractionRunState
from app.domain.profiles import ProfileValidationIssue, ProfileValidationResult
from app.ollama.errors import CompletionError
from app.services.projection_service import (
    _CoreObjectIntent,
    _ProvenanceCoreIntentResponse,
    ProjectionService,
)
from app.services.workflow_service import WorkflowService


class RequirementScoringTests(unittest.TestCase):
    def test_semantic_requirement_set_uses_revised_scope(self):
        requirement_ids = [req.requirement_id for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS]

        self.assertEqual(len(requirement_ids), 8)
        self.assertIn("dataset_description_scope", requirement_ids)
        self.assertNotIn("dataset_description_identity", requirement_ids)
        self.assertNotIn("attribute_duplicate_coherence", requirement_ids)
        self.assertNotIn("attribute_label_quality", requirement_ids)
        self.assertNotIn("provenance_context_placement", requirement_ids)

    def test_requirement_prompt_defines_activity_target_semantics(self):
        self.assertIn("evaluated_entity/evaluated_activity answer what that specific", REQUIREMENT_EVALUATOR_SYSTEM_PROMPT)
        self.assertIn("A merely generated output is not an evaluated target", REQUIREMENT_EVALUATOR_SYSTEM_PROMPT)

    def test_semantic_score_excludes_not_applicable(self):
        items = [
            RequirementReportItem(
                requirement_id="fulfilled",
                label="Fulfilled",
                weight=2.0,
                status="fulfilled",
                applicable=True,
                quality=1.0,
                weighted_score=0.0,
            ),
            RequirementReportItem(
                requirement_id="partial",
                label="Partial",
                weight=1.0,
                status="partial",
                applicable=True,
                quality=0.5,
                weighted_score=0.0,
            ),
            RequirementReportItem(
                requirement_id="na",
                label="N/A",
                weight=10.0,
                status="not_applicable",
                applicable=False,
                quality=0.0,
                weighted_score=0.0,
            ),
        ]
        score = score_requirement_items(items)
        self.assertAlmostEqual(score, 2.5 / 3.0)
        self.assertEqual(items[2].weighted_score, 0.0)

    def test_coverage_report_scores_all_schema_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "array", "items": {"type": "string"}},
                "activity": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            },
        }
        report = compute_coverage_report({"title": ["Dataset"], "activity": {"title": "Run"}}, schema)
        self.assertEqual(report.score, 3.0)
        self.assertEqual(report.filled_fields, 3)
        self.assertEqual(report.total_fields, 5)

    def test_source_trace_averages_used_evidence_match_scores(self):
        used = EvidenceCandidate(
            candidate_id="m1",
            category="method_signal",
            role="descriptor",
            claim="Plan.",
            evidence_text="PULPROG=zg30",
            evidence_match_score=0.8,
        )
        unused = EvidenceCandidate(
            candidate_id="m2",
            category="method_signal",
            role="descriptor",
            claim="Other.",
            evidence_text="PULPROG=noesy",
            evidence_match_score=1.0,
        )
        context = RoutedEvidenceContext(portable_evidence=[used, unused])
        report = compute_source_trace_report(context, [stable_evidence_id(used)])
        self.assertEqual(report.used_evidence_count, 1)
        self.assertAlmostEqual(report.score, 0.8)

    def test_source_trace_excludes_unknown_description_evidence_ids(self):
        source = EvidenceCandidate(
            candidate_id="source",
            category="other",
            role="other_metadata",
            claim="Source fact.",
            evidence_text="Source fact.",
            evidence_match_score=1.0,
        )
        source_id = stable_evidence_id(source)
        report = compute_source_trace_report(
            RoutedEvidenceContext(portable_evidence=[source]),
            [source_id, "ev:description-derived"],
        )

        self.assertEqual(report.used_evidence_count, 1)
        self.assertEqual(report.evidence_ids, [source_id])

    def test_missing_evaluator_requirement_becomes_unanswered_report_item(self):
        req = DcatRequirement(
            requirement_id="method_plan",
            label="Method",
            description="Need method.",
            target_paths=["/was_generated_by/0/realized_plan"],
            evidence_hints=["pulse sequence"],
        )
        items = report_items_from_evaluation(
            requirements=[req],
            evaluation=RequirementEvaluation(assessments=[]),
        )
        self.assertEqual(items[0].status, "unanswered")
        self.assertEqual(items[0].target_paths, ["/was_generated_by/0/realized_plan"])
        self.assertEqual(items[0].evidence_search_hints, ["pulse sequence"])

    def test_normalized_evaluation_returns_every_requirement_once(self):
        requirements = list(DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS[:2])
        evaluation = normalized_requirement_evaluation(
            requirements=requirements,
            evaluation=RequirementEvaluation(
                assessments=[
                    RequirementAssessment(
                        requirement_id=requirements[0].requirement_id,
                        status="fulfilled",
                        quality=0.2,
                    ),
                    RequirementAssessment(
                        requirement_id=requirements[0].requirement_id,
                        status="missing",
                        quality=1.0,
                    ),
                ]
            ),
        )

        self.assertEqual([item.requirement_id for item in evaluation.assessments], [req.requirement_id for req in requirements])
        self.assertEqual(evaluation.assessments[0].quality, 1.0)
        self.assertEqual(evaluation.assessments[1].status, "unanswered")

    def test_core_requirement_not_applicable_is_treated_as_missing(self):
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "method_plan")

        evaluation = normalized_requirement_evaluation(
            requirements=[requirement],
            evaluation=RequirementEvaluation(
                assessments=[
                    RequirementAssessment(
                        requirement_id=requirement.requirement_id,
                        status="not_applicable",
                        quality=0.0,
                        applicable=False,
                    )
                ]
            ),
        )

        self.assertEqual(evaluation.assessments[0].status, "missing")
        self.assertTrue(evaluation.assessments[0].applicable)

    def test_build_report_exposes_three_scores(self):
        semantic = RequirementReportItem(
            requirement_id="method_plan",
            label="Method",
            weight=1.0,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=0.0,
            selected_evidence=[
                {
                    "candidate_id": "e1",
                    "category": "method_signal",
                    "role": "descriptor",
                    "claim": "Pulse sequence is zg30.",
                    "evidence_text": "PULPROG= zg30",
                }
            ],
        )

        source_trace = compute_source_trace_report(
            RoutedEvidenceContext(),
            [],
        )
        report = build_requirement_report(
            schema_valid=True,
            coverage=compute_coverage_report({"title": ["Dataset"]}, {"type": "object", "properties": {"title": {"type": "array"}}}),
            semantic_requirements=[semantic],
            source_trace=source_trace,
            coverage_patches=[],
        )

        self.assertEqual(report.semantic_requirements_score, 1.0)
        self.assertEqual(report.source_trace_score, 0.0)
        self.assertEqual(report.coverage_score, 1.0)

    def test_build_report_copies_semantic_reconstruction_status_to_item_patch(self):
        semantic = RequirementReportItem(
            requirement_id="instrument_settings_semantics",
            label="Instrument settings",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
        )

        report = build_requirement_report(
            schema_valid=True,
            coverage=compute_coverage_report({}, {}),
            semantic_requirements=[semantic],
            source_trace=compute_source_trace_report(RoutedEvidenceContext(), []),
            coverage_patches=[],
            semantic_reconstructions=[
                SemanticReconstructionRecord(
                    requirement_id="instrument_settings_semantics",
                    status="rolled_back",
                    target_paths=["/was_generated_by/0/has_quantitative_attribute"],
                    changed_paths=["/was_generated_by/0/has_quantitative_attribute/0"],
                    reason="Semantic reconstruction failed validation.",
                    validation_errors=["Additional properties are not allowed ('category' was unexpected)"],
                )
            ],
        )

        patch = report.semantic_requirements[0].patch
        self.assertTrue(patch.attempted)
        self.assertEqual(patch.status, "rolled_back")
        self.assertEqual(patch.target_path, "/was_generated_by/0/has_quantitative_attribute/0")
        self.assertEqual(patch.validation_errors, ["Additional properties are not allowed ('category' was unexpected)"])

    def test_schema_constrained_patch_schema_slices_quantitative_attribute_defs(self):
        schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$ref": "#/$defs/Dataset",
            "$defs": {
                "Dataset": {
                    "type": "object",
                    "properties": {
                        "was_generated_by": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/DataGeneratingActivity"},
                        },
                        "dataset_distribution": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Distribution"},
                        },
                    },
                },
                "DataGeneratingActivity": {
                    "type": "object",
                    "properties": {
                        "has_quantitative_attribute": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/QuantitativeAttribute"},
                        }
                    },
                },
                "QuantitativeAttribute": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "value": {"type": "number"},
                        "has_quantity_type": {"type": "string"},
                    },
                    "required": ["value", "has_quantity_type"],
                },
                "Distribution": {"type": "object"},
            },
        }

        output_schema = build_schema_constrained_patch_schema(
            validation_schema=schema,
            allowed_target_paths=["/was_generated_by/0/has_quantitative_attribute"],
        )

        self.assertIn("QuantitativeAttribute", output_schema["$defs"])
        self.assertNotIn("Distribution", output_schema["$defs"])
        write_schema = output_schema["properties"]["writes"]["items"]["oneOf"][0]
        append_schema = next(item for item in write_schema["oneOf"] if item["properties"]["mode"]["const"] == "append")
        self.assertEqual(append_schema["properties"]["target_path"]["const"], "/was_generated_by/0/has_quantitative_attribute")
        self.assertEqual(append_schema["properties"]["mode"]["const"], "append")

    def test_schema_constrained_patch_schema_rejects_invalid_quantitative_items(self):
        output_schema = build_schema_constrained_patch_schema(
            validation_schema=quantitative_schema("DataGeneratingActivity"),
            allowed_target_paths=["/was_generated_by/0/has_quantitative_attribute"],
        )
        validator = Draft201909Validator(output_schema)
        valid = {
            "writes": [
                {
                    "target_path": "/was_generated_by/0/has_quantitative_attribute",
                    "mode": "append",
                    "items": [
                        {"value": 373.96442, "has_quantity_type": "minimum wavenumber", "unit": "1/CM"},
                        {"value": 3997.453, "has_quantity_type": "maximum wavenumber", "unit": "1/CM"},
                    ],
                    "reason": "Represent wavenumber range as min/max attributes.",
                }
            ],
            "reason": "ok",
        }
        legacy_should_apply = {**valid, "should_apply": True}
        extra_field = {
            **valid,
            "writes": [
                {
                    **valid["writes"][0],
                    "items": [{**valid["writes"][0]["items"][0], "category": "measurement_condition"}],
                }
            ],
        }
        range_object = {
            **valid,
            "writes": [
                {
                    **valid["writes"][0],
                    "items": [
                        {
                            "has_quantity_type": "wavenumber range",
                            "minimum": 373.96442,
                            "maximum": 3997.453,
                            "unit": "1/CM",
                        }
                    ],
                }
            ],
        }

        self.assertEqual(list(validator.iter_errors(valid)), [])
        self.assertNotEqual(list(validator.iter_errors(legacy_should_apply)), [])
        self.assertNotEqual(list(validator.iter_errors(extra_field)), [])
        self.assertNotEqual(list(validator.iter_errors(range_object)), [])


class RequirementEvidencePacketTests(unittest.TestCase):
    def test_stable_evidence_id_disambiguates_repeated_candidate_ids(self):
        left = EvidenceCandidate(
            candidate_id="candidate-0",
            category="measurement_signal",
            role="parameter",
            claim="Observe frequency is 500 MHz.",
            evidence_text="##.OBSERVE FREQUENCY=500.133088507478",
            file_path="10.edit.jdx",
            start_idx=0,
            end_idx=58,
        )
        right = EvidenceCandidate(
            candidate_id="candidate-0",
            category="resource_signal",
            role="descriptor",
            claim="Parameter file.",
            evidence_text="##TITLE= Parameter file",
            file_path="10.zip/10/acqus",
            start_idx=0,
            end_idx=6,
        )

        self.assertNotEqual(stable_evidence_id(left), stable_evidence_id(right))

    def test_selects_requirement_packet_by_hints_and_class(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
            if req.requirement_id == "method_plan"
        )
        target = EvidenceCandidate(
            candidate_id="m1",
            category="method_signal",
            role="descriptor",
            claim="Pulse sequence is zg30.",
            evidence_text="PULPROG= zg30",
            file_path="acqus",
            start_idx=100,
            end_idx=120,
        )
        nearby = EvidenceCandidate(
            candidate_id="m2",
            category="measurement_signal",
            role="parameter",
            claim="Acquisition temperature is 298 K.",
            evidence_text="TE= 298",
            file_path="acqus",
            start_idx=160,
            end_idx=180,
        )
        unrelated = EvidenceCandidate(
            candidate_id="x1",
            category="instrument_signal",
            role="identity",
            claim="Owner is nmr.",
            evidence_text="OWNER=nmr",
            file_path="owner",
            start_idx=0,
            end_idx=10,
        )
        context = RoutedEvidenceContext(
            portable_evidence=[target, unrelated],
            contextual_evidence=[nearby],
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            evidence_search_hints=["pulse sequence"],
        )
        selected, window = select_requirement_evidence_packet(
            requirement=requirement,
            assessment=item,
            evidence_context=context,
        )
        self.assertEqual(selected[0].candidate_id, "m1")
        self.assertIn("m2", {entry.candidate_id for entry in window})

    def test_duplicate_attribute_cleanup_is_not_a_semantic_requirement(self):
        requirement_ids = {req.requirement_id for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS}

        self.assertNotIn("attribute_duplicate_coherence", requirement_ids)

    def test_method_plan_packet_accepts_explicit_procedure_cue_from_instrument_signal(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
            if req.requirement_id == "method_plan"
        )
        procedure = EvidenceCandidate(
            candidate_id="procedure",
            category="instrument_signal",
            role="descriptor",
            claim="The sampling procedure is Diamant ATR.",
            evidence_text="##SAMPLING PROCEDURE=Diamant ATR",
        )
        context = RoutedEvidenceContext(portable_evidence=[procedure])
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            evidence_search_hints=requirement.evidence_hints,
        )

        selected, _ = select_requirement_evidence_packet(
            requirement=requirement,
            assessment=item,
            evidence_context=context,
        )

        self.assertEqual([entry.candidate_id for entry in selected], ["procedure"])

    def test_generation_activity_reality_accepts_supporting_activity_evidence(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "generation_activity_reality"
        )
        method = EvidenceCandidate(
            candidate_id="method",
            category="method_signal",
            role="descriptor",
            claim="The data was acquired using Diamant ATR sampling procedure.",
            evidence_text="##SAMPLING PROCEDURE=Diamant ATR",
        )
        agent = EvidenceCandidate(
            candidate_id="agent",
            category="instrument_signal",
            role="identity",
            claim="The instrument used is Bruker ALPHA.",
            evidence_text="instrument: Bruker ALPHA",
        )
        instrument = EvidenceCandidate(
            candidate_id="setting",
            category="instrument_signal",
            role="identity",
            claim="The threshold for peak detection is set to 0.93.",
            evidence_text="##$CSTHRESHOLD=0.93",
        )
        context = RoutedEvidenceContext(portable_evidence=[method, agent, instrument])
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            evidence_search_hints=requirement.evidence_hints,
        )

        selected, _ = select_requirement_evidence_packet(
            requirement=requirement,
            assessment=item,
            evidence_context=context,
        )

        self.assertEqual(
            {entry.candidate_id for entry in selected},
            {"method", "agent", "setting"},
        )

    def test_attribute_range_prompt_requires_selected_settings(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "attribute_range_decomposition"
        )

        prompt = build_requirement_evaluation_prompt(
            document={"/was_generated_by/0/has_quantitative_attribute": []},
            requirements=[requirement],
            selected_evidence=[
                RequirementEvidenceItem(
                    evidence_id="ev:unit",
                    candidate_id="unit",
                    category="instrument_signal",
                    role="identity",
                    claim="X-axis units are 1/CM.",
                    evidence_text="##XUNITS=1/CM",
                )
            ],
            context_window=[],
        )

        self.assertIn(
            "Ranges are represented as separate minimum and maximum attributes",
            prompt,
        )

    def test_attribute_range_evidence_prioritizes_explicit_bounds(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "attribute_range_decomposition"
        )
        candidates = [
            EvidenceCandidate(
                candidate_id=f"max-{index}",
                category="measurement_condition",
                role="parameter",
                claim=f"Maximum Y value {index}",
                evidence_text=f"MAXY={index}",
            )
            for index in range(4)
        ]
        candidates.append(
            EvidenceCandidate(
                candidate_id="explicit-range",
                category="measurement_condition",
                role="parameter",
                claim="Wavenumber range from 373.96 to 3997.45 1/CM",
                evidence_text="spanning 373.96-3997.45 1/CM",
            )
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            evidence_search_hints=requirement.evidence_hints,
        )

        selected, _ = select_requirement_evidence_packet(
            requirement=requirement,
            assessment=item,
            evidence_context=RoutedEvidenceContext(portable_evidence=candidates),
        )

        self.assertIn("explicit-range", [entry.candidate_id for entry in selected])

    def test_semantic_diagnosis_prompt_preserves_specific_quantity(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "attribute_range_decomposition"
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.75,
            target_paths=list(requirement.target_paths),
        )

        prompt = build_semantic_diagnosis_prompt(
            requirement=requirement,
            item=item,
            draft_excerpt={},
        )

        self.assertIn("Do not write the profile", prompt)
        self.assertIn("Ranges are represented as separate minimum and maximum", prompt)

    def test_semantic_diagnosis_prompt_guards_generic_attribute_intent(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "attribute_parent_placement"
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=list(requirement.target_paths),
        )

        prompt = build_semantic_diagnosis_prompt(
            requirement=requirement,
            item=item,
            draft_excerpt={},
        )

        self.assertIn("characterizes the exact target parent", prompt)
        self.assertIn("source-record metadata", prompt)
        self.assertIn("characterizes the exact target parent", prompt)

    def test_semantic_synthesis_prompt_rejects_placeholder_attributes(self):
        self.assertIn("directly characterizes the target parent", SEMANTIC_SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("placeholder labels", SEMANTIC_SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("measured/unknown/present", SEMANTIC_SYNTHESIS_SYSTEM_PROMPT)


class FakeProfileService:
    def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
        return ProfileValidationResult(valid=True, errors=[])


def quantitative_schema(*owner_classes: str) -> dict:
    defs = {
        "QuantitativeAttribute": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "value": {"type": "number"},
                "has_quantity_type": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["value", "has_quantity_type"],
        }
    }
    for owner_class in owner_classes:
        defs[owner_class] = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "rdf_type": {"type": "object"},
                "carried_out_by": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"$ref": "#/$defs/AgenticEntity"},
                            {"$ref": "#/$defs/Device"},
                            {"$ref": "#/$defs/Software"},
                        ]
                    },
                },
                "has_quantitative_attribute": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/QuantitativeAttribute"},
                },
            },
        }
    return {
        "type": "object",
        "properties": {
            "description": {"type": "array", "items": {"type": "string"}},
            "was_generated_by": {
                "type": "array",
                "items": {"$ref": "#/$defs/DataGeneratingActivity"},
            },
            "is_about_activity": {
                "type": "array",
                "items": {"$ref": "#/$defs/EvaluatedActivity"},
            },
            "is_about_entity": {
                "type": "array",
                "items": {"$ref": "#/$defs/EvaluatedEntity"},
            },
        },
        "$defs": defs,
    }


class RequirementEnrichmentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        pass

    def test_provenance_core_intents_are_backend_shaped_and_capped(self):
        service = WorkflowService(profile_service=FakeProfileService(), settings=Settings())
        document = {
            "id": "pkg",
            "title": ["Dataset"],
            "description": ["Dataset description."],
            "was_generated_by": [{"id": "pkg:activity:dataset-generation"}],
        }
        response = _ProvenanceCoreIntentResponse(
            answer="Core provenance reconstructed.",
            activity_title="NMR acquisition and processing",
            activity_description="1H NMR acquisition and processing workflow.",
            plan=_CoreObjectIntent(title="zg30 pulse program", description="1D sequence pulse program."),
            agents=[
                _CoreObjectIntent(title="Avance III NMR spectrometer", description="NMR spectrometer."),
                _CoreObjectIntent(title="TopSpin 3.2", description="Processing software."),
            ],
            evaluated_entities=[
                _CoreObjectIntent(title="1H NMR measurement", description="The measured NMR signal."),
                _CoreObjectIntent(title="NMR sample", description="Sample measured by NMR."),
            ],
            evaluated_activities=[
                _CoreObjectIntent(title="Acquisition", description="NMR acquisition."),
                _CoreObjectIntent(title="Processing", description="NMR processing."),
            ],
            input_entities=[_CoreObjectIntent(title="FID", description="Raw NMR FID data.")],
            output_entities=[_CoreObjectIntent(title="Processed NMR spectrum", description="Processed spectrum data.")],
        )

        updated, changed_paths = service._apply_provenance_core_intents(
            data_package_id="pkg",
            document=document,
            response=response,
        )

        activity = updated["was_generated_by"][0]
        self.assertEqual(activity["title"], ["NMR acquisition and processing"])
        self.assertEqual(activity["realized_plan"]["title"], "zg30 pulse program")
        self.assertEqual(activity["carried_out_by"][0]["title"], "Avance III NMR spectrometer")
        self.assertEqual(activity["evaluated_entity"][0]["title"], "1H NMR measurement")
        self.assertEqual(len(activity["evaluated_entity"]) + len(activity["evaluated_activity"]), 3)
        self.assertEqual(activity["had_input_entity"][0]["title"], "FID")
        self.assertEqual(activity["had_output_entity"][0]["title"], "Processed NMR spectrum")
        self.assertIn("/was_generated_by/0/carried_out_by/0", changed_paths)
        self.assertIn("/was_generated_by/0/evaluated_entity/0", changed_paths)

    def test_provenance_core_prompt_uses_draft_description_and_file_orientation(self):
        prompt = WorkflowService._provenance_core_prompt(
            document={"id": "pkg", "title": ["Dataset"], "description": ["Dataset-level orientation."]},
            orientation_context={"file_summaries": "ranked file context"},
        )
        payload = json.loads(prompt)

        self.assertEqual(payload["current_draft"]["description"], ["Dataset-level orientation."])
        self.assertEqual(payload["orientation_context"], {"file_summaries": "ranked file context"})
        self.assertNotIn("dataset_summary", payload["orientation_context"])
        self.assertNotIn("selected_evidence", payload)
        self.assertNotIn("context_window", payload)
        rules = "\n".join(payload["rules"])
        self.assertIn("Do not put vendors or manufacturers in agents unless the context says they performed", rules)
        self.assertIn("Do not put methods, protocols, scripts, recipes, program definitions", rules)

    def test_initial_draft_keeps_dataset_fields_and_clears_provenance_core(self):
        initial = WorkflowService._dataset_only_initial_draft(
            {
                "id": "pkg",
                "title": ["NMR dataset"],
                "description": ["Dataset summary."],
                "keyword": ["NMR"],
                "type": [{"preferred_label": ["spectroscopy dataset"]}],
                "was_generated_by": [{"id": "activity"}],
                "is_about_entity": [{"id": "subject"}],
            }
        )

        self.assertEqual(initial["title"], ["NMR dataset"])
        self.assertEqual(initial["description"], ["Dataset summary."])
        self.assertEqual(initial["keyword"], ["NMR"])
        self.assertEqual(initial["type"][0]["preferred_label"], ["spectroscopy dataset"])
        self.assertEqual(initial["was_generated_by"], [])
        self.assertNotIn("is_about_entity", initial)

    async def test_requirement_enrichment_persists_report_and_initial_draft(self):
        repo = Mock()
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
            output_repository=repo,
        )
        published_stages: list[str] = []
        service._update_progress = lambda _data_package_id, current: published_stages.append(current.stage)  # type: ignore[method-assign]
        service._evaluate_dcat_requirements = AsyncMock(
            return_value=RequirementEvaluation(
                assessments=[
                    RequirementAssessment(
                        requirement_id="method_plan",
                        status="missing",
                        quality=0.0,
                        applicable=True,
                        rationale="No plan.",
                        target_paths=["/was_generated_by/0/realized_plan"],
                        evidence_search_hints=["pulse sequence"],
                        expected_target_class="Plan",
                    )
                ]
            )
        )
        service._patch_requirement_gap = AsyncMock(
            return_value=(
                {
                    "id": "pkg",
                    "title": ["Dataset"],
                    "description": ["Desc"],
                    "was_generated_by": [{"id": "act", "realized_plan": {"title": "zg30"}}],
                },
                RequirementPatchAttempt(
                    attempted=True,
                    status="applied",
                    target_path="/was_generated_by/0/realized_plan",
                    target_class="Plan",
                    reason="Applied.",
                ),
            )
        )
        service._evaluate_semantic_requirements = AsyncMock(return_value=[])
        service._apply_provenance_core_construction = AsyncMock(
            side_effect=lambda **kwargs: kwargs["document"]
        )
        state = ExtractionRunState(
            generated_final_draft={
                "id": "pkg",
                "title": ["Dataset"],
                "description": ["Desc"],
                "was_generated_by": [{"id": "act"}],
            },
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        evidence_context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="p1",
                    category="method_signal",
                    role="descriptor",
                    claim="Pulse sequence is zg30.",
                    evidence_text="PULPROG= zg30",
                )
            ]
        )

        document = await service._enrich_draft_with_requirements(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=evidence_context,
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )

        self.assertEqual(document["was_generated_by"][0]["realized_plan"]["title"], "zg30")
        self.assertIsNotNone(state.generated_initial_draft)
        self.assertIsNotNone(state.requirement_report)
        self.assertIsNotNone(state.document_quality_state)
        self.assertTrue(state.document_quality_state.schema_valid)
        self.assertIsNone(state.document_quality_state.semantic_valid)
        self.assertIsNone(state.document_quality_state.operational_access_score)
        self.assertTrue(
            any(
                issue.code == "semantic_vocabulary_validation_not_run"
                for issue in state.document_quality_state.warnings
            )
        )
        self.assertTrue(
            any(
                issue.code == "operational_fair_checks_not_run"
                for issue in state.document_quality_state.warnings
            )
        )
        item = next(req for req in state.requirement_report.coverage_patches if req.requirement_id == "method_plan")
        self.assertEqual(item.patch.status, "applied")
        self.assertEqual(state.field_completion_ledger[0].json_path, "/was_generated_by/0/realized_plan")
        self.assertEqual(state.field_completion_ledger[0].field_name, "realized_plan")
        self.assertEqual(state.field_completion_ledger[0].enrichment_status, "grounded")
        self.assertTrue(state.field_completion_ledger[0].source_evidence[0].startswith("ev:"))
        self.assertTrue(
            any(
                record.object_kind == "RequirementPatch"
                and record.target_path == "/was_generated_by/0/realized_plan"
                for record in state.projection_ledger
            )
        )
        self.assertTrue(state.evidence_query_ledger)
        method_query = next(
            record
            for record in state.evidence_query_ledger
            if record.requirement_id == "method_plan"
        )
        self.assertEqual(method_query.target_path, "/was_generated_by/0/realized_plan")
        self.assertTrue(method_query.selected_evidence_ids[0].startswith("ev:"))
        self.assertIn(
            method_query.selected_evidence_ids[0],
            method_query.result_evidence_ids,
        )
        repo.save_requirement_report.assert_called()
        repo.save_generated_initial_draft.assert_called()
        expected_stages = [
            "evidence_patching",
            "coverage_scoring",
            "semantic_evaluation",
            "semantic_reconstruction",
            "semantic_revalidation",
        ]
        self.assertEqual(
            list(dict.fromkeys(stage for stage in published_stages if stage in expected_stages)),
            expected_stages,
        )

    async def test_semantic_reconstruction_runs_between_semantic_evaluations(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        first_item = RequirementReportItem(
            requirement_id="dataset_identity_semantics",
            label="Identity",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
        )
        second_item = RequirementReportItem(
            requirement_id="dataset_identity_semantics",
            label="Identity",
            weight=1.0,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=1.0,
        )
        service._evaluate_semantic_requirements = AsyncMock(side_effect=[[first_item], [second_item]])
        service._apply_provenance_core_construction = AsyncMock(side_effect=lambda **kwargs: kwargs["document"])
        service._semantic_reconstruction_update = AsyncMock(
            return_value=(
                {
                    "id": "pkg",
                    "title": ["Dataset"],
                    "description": ["Dataset acquired with method."],
                },
                ["/description"],
                "LLM streamlined description.",
                [],
            )
        )
        state = ExtractionRunState(
            generated_final_draft={
                "id": "pkg",
                "title": ["Dataset"],
                "description": ["Dataset acquired with method. Raw data points 10."],
            },
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])

        document = await service._enrich_draft_with_requirements(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=RoutedEvidenceContext(),
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )

        self.assertEqual(service._evaluate_semantic_requirements.call_count, 2)
        self.assertEqual(document["description"], ["Dataset acquired with method."])
        self.assertEqual(state.requirement_report.semantic_requirements[0].status, "fulfilled")
        self.assertEqual(state.requirement_report.semantic_reconstructions[0].status, "applied")
        self.assertEqual(state.requirement_report.semantic_reconstructions[0].changed_paths, ["/description"])

    async def test_applied_patch_is_re_evaluated_before_scoring(self):
        repo = Mock()
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
            output_repository=repo,
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "method_plan")
        service._evaluate_dcat_requirements = AsyncMock(
            side_effect=[
                RequirementEvaluation(
                    assessments=[
                        RequirementAssessment(
                            requirement_id=req.requirement_id,
                            status="fulfilled" if req.requirement_id != requirement.requirement_id else "missing",
                            quality=1.0 if req.requirement_id != requirement.requirement_id else 0.0,
                            applicable=True,
                            target_paths=req.target_paths,
                            evidence_search_hints=["pulse sequence"] if req.requirement_id == requirement.requirement_id else [],
                            expected_target_class=req.expected_target_class,
                        )
                        for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
                    ]
                ),
                RequirementEvaluation(
                    assessments=[
                        RequirementAssessment(
                            requirement_id=requirement.requirement_id,
                            status="fulfilled",
                            quality=1.0,
                            applicable=True,
                            target_paths=requirement.target_paths,
                            evidence_search_hints=["pulse sequence"],
                            expected_target_class="Plan",
                        )
                    ]
                ),
            ]
        )
        service._patch_requirement_gap = AsyncMock(
            return_value=(
                {
                    "id": "pkg",
                    "title": ["Dataset"],
                    "description": ["Desc"],
                    "was_generated_by": [{"id": "act", "realized_plan": {"title": "zg30"}}],
                },
                RequirementPatchAttempt(
                    attempted=True,
                    status="applied",
                    target_path="/was_generated_by/0/realized_plan",
                    target_class="Plan",
                    reason="Applied.",
                ),
            )
        )
        service._evaluate_semantic_requirements = AsyncMock(return_value=[])
        service._apply_provenance_core_construction = AsyncMock(
            side_effect=lambda **kwargs: kwargs["document"]
        )
        state = ExtractionRunState(
            generated_final_draft={
                "id": "pkg",
                "title": ["Dataset"],
                "description": ["Desc"],
                "was_generated_by": [{"id": "act"}],
            },
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        evidence_context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="p1",
                    category="method_signal",
                    role="descriptor",
                    claim="Pulse sequence is zg30.",
                    evidence_text="PULPROG= zg30",
                )
            ]
        )

        await service._enrich_draft_with_requirements(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=evidence_context,
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )

        item = next(req for req in state.requirement_report.coverage_patches if req.requirement_id == "method_plan")
        self.assertEqual(item.status, "fulfilled")
        self.assertEqual(item.patch.status, "applied")

    async def test_fulfilled_requirement_with_missing_target_path_is_patched(self):
        repo = Mock()
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
            output_repository=repo,
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "method_plan")
        service._evaluate_dcat_requirements = AsyncMock(
            side_effect=[
                RequirementEvaluation(
                    assessments=[
                        RequirementAssessment(
                            requirement_id=req.requirement_id,
                            status="fulfilled",
                            quality=1.0,
                            applicable=True,
                            target_paths=req.target_paths,
                            evidence_search_hints=["pulse sequence"] if req.requirement_id == requirement.requirement_id else [],
                            expected_target_class=req.expected_target_class,
                        )
                        for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
                    ]
                ),
                RequirementEvaluation(
                    assessments=[
                        RequirementAssessment(
                            requirement_id=requirement.requirement_id,
                            status="fulfilled",
                            quality=1.0,
                            applicable=True,
                            target_paths=requirement.target_paths,
                            evidence_search_hints=["pulse sequence"],
                            expected_target_class="Plan",
                        )
                    ]
                ),
            ]
        )
        service._patch_requirement_gap = AsyncMock(
            return_value=(
                {
                    "id": "pkg",
                    "title": ["Dataset"],
                    "description": ["Desc"],
                    "was_generated_by": [{"id": "act", "realized_plan": {"title": "zg30"}}],
                    "is_about_activity": [{"title": ["zg30 pulse sequence"]}],
                },
                RequirementPatchAttempt(
                    attempted=True,
                    status="applied",
                    target_path="/was_generated_by/0/realized_plan",
                    target_class="Plan",
                    reason="Applied.",
                ),
            )
        )
        service._evaluate_semantic_requirements = AsyncMock(return_value=[])
        service._apply_provenance_core_construction = AsyncMock(
            side_effect=lambda **kwargs: kwargs["document"]
        )
        state = ExtractionRunState(
            generated_final_draft={
                "id": "pkg",
                "title": ["Dataset"],
                "description": ["Desc"],
                "was_generated_by": [{"id": "act"}],
                "is_about_activity": [{"title": ["zg30 pulse sequence"]}],
            },
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        evidence_context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="p1",
                    category="method_signal",
                    role="descriptor",
                    claim="Pulse sequence is zg30.",
                    evidence_text="PULPROG= zg30",
                )
            ]
        )

        document = await service._enrich_draft_with_requirements(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=evidence_context,
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )

        service._patch_requirement_gap.assert_awaited()
        self.assertEqual(document["was_generated_by"][0]["realized_plan"]["title"], "zg30")

    def test_duplicate_requirement_patch_is_rejected(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )

        reason = service._duplicate_requirement_patch_reason(
            document={"is_about_entity": [{"id": "entity:1", "title": "CDCl3 Solvent"}]},
            target_path="/is_about_entity/-",
            instance={"id": "entity:1", "title": "CDCl3 Solvent"},
        )

        self.assertIn("duplicates existing object", reason)

    def test_plan_patch_sanitizer_removes_recursive_fields(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )

        sanitized = service._sanitize_requirement_patch_instance(
            instance={
                "title": ["zg30"],
                "description": ["Pulse sequence"],
                "was_generated_by": [{"was_generated_by": []}],
                "id": "bad-id",
            },
            target_class="Plan",
            target_path="/was_generated_by/0/realized_plan",
            item=Mock(selected_evidence=[]),
        )

        self.assertEqual(sanitized, {"title": "zg30", "description": "Pulse sequence"})

    def test_quantitative_patch_sanitizer_repairs_required_shape(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )

        sanitized = service._sanitize_requirement_patch_instance(
            instance={
                "id": "bad-id",
                "title": "Observe frequency",
                "value": "500.13 MHz",
                "has_quantity_type": {"title": "frequency"},
                "source": "bad",
            },
            target_class="QuantitativeAttribute",
            target_path="/was_generated_by/0/has_quantitative_attribute/-",
            item=Mock(selected_evidence=[]),
        )

        self.assertEqual(sanitized["value"], 500.13)
        self.assertEqual(sanitized["has_quantity_type"], "frequency")
        self.assertEqual(sanitized["unit"], "MHz")
        self.assertNotIn("id", sanitized)
        self.assertNotIn("source", sanitized)
        self.assertNotIn("type", sanitized)
        self.assertNotIn("rdf_type", sanitized)

    async def test_semantic_reconstruction_uses_diagnosis_schema(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="dataset_description_scope",
            label="Description",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/description"],
        )
        llm_result = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_identity",
                        "target_path": "/description",
                        "entry_indices": [],
                        "recommended_action": "no_action",
                        "needs_synthesis": False,
                        "reason": "No safe repair.",
                    }
                ],
                "reason": "No safe repair.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            updated, paths, reason, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={
                    "id": "pkg",
                    "title": ["Dataset"],
                    "description": [
                        "Dataset acquired with a documented method. Raw measurement values are summarized."
                    ],
                },
                item=item,
                requirement=DcatRequirement(
                    requirement_id="dataset_description_scope",
                    label="Description",
                    description="Description identity",
                    target_paths=["/description"],
                ),
                validation_schema=quantitative_schema("DataGeneratingActivity"),
            )

        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertEqual(reason, "semantic_defect_unresolved_empty_diagnosis")
        self.assertEqual(updated["description"], ["Dataset acquired with a documented method. Raw measurement values are summarized."])
        call_kwargs = mocked.call_args.kwargs
        self.assertEqual(call_kwargs["agent_name"], "semantic_reconstruction")
        self.assertEqual(call_kwargs["system"], SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT)
        self.assertNotIn("write envelopes", call_kwargs["system"])
        self.assertIsInstance(call_kwargs["output_type"], dict)
        self.assertIn("defects", call_kwargs["output_type"]["properties"])
        self.assertIn("draft_excerpt", call_kwargs["prompt"])
        self.assertEqual(item.defect_type, "bad_identity")
        self.assertEqual(item.diagnosed_defects[0]["recommended_action"], "no_action")

    async def test_semantic_synthesis_uses_exact_array_target_schema(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="dataset_title_identity",
            label="Title identity",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/title"],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_identity",
                        "target_path": "/title",
                        "entry_indices": [],
                        "recommended_action": "replace",
                        "needs_synthesis": True,
                        "reason": "Use the source-backed dataset title.",
                    }
                ],
                "reason": "Repair title identity.",
            },
            usage=None,
        )
        synthesis = Mock(output=["SG-V4050"], usage=None)
        validation_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
        }

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(side_effect=[diagnosis, synthesis]),
        ) as mocked:
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"title": ["Generic dataset"]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="dataset_title_identity",
                    label="Title identity",
                    description="Dataset title",
                    target_paths=["/title"],
                ),
                validation_schema=validation_schema,
            )

        self.assertEqual(updated["title"], ["SG-V4050"])
        self.assertEqual(paths, ["/title"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 1)
        synthesis_call = mocked.call_args_list[1].kwargs
        self.assertEqual(synthesis_call["system"], SEMANTIC_SYNTHESIS_SYSTEM_PROMPT)
        self.assertEqual(synthesis_call["output_type"]["type"], "array")
        self.assertEqual(synthesis_call["output_type"]["items"]["type"], "string")
        self.assertEqual(item.compiled_actions[0]["value"], ["SG-V4050"])

    async def test_dataset_description_scope_does_not_remove_only_description(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="dataset_description_scope",
            label="Description scope",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/description"],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_identity",
                        "target_path": "/description",
                        "entry_indices": [0],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Description is overloaded.",
                    }
                ],
                "reason": "Remove overloaded description.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=diagnosis)):
            updated, paths, reason, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"description": ["Dataset-level summary with some repeated facts."]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="dataset_description_scope",
                    label="Description scope",
                    description="Description scope",
                    target_paths=["/description"],
                ),
                validation_schema={
                    "type": "object",
                    "properties": {
                        "description": {"type": "array", "items": {"type": "string"}}
                    },
                },
            )

        self.assertEqual(updated["description"], ["Dataset-level summary with some repeated facts."])
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertEqual(applied, 0)
        self.assertIn("must not remove the only description", rejected[0])

    async def test_semantic_synthesis_replaces_missing_object_when_model_requests_append(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="method_plan_presence",
            label="Method plan",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/was_generated_by/0/realized_plan"],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "missing_plan",
                        "target_path": "/was_generated_by/0/realized_plan",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "Create the source-backed plan.",
                    }
                ],
                "reason": "Create plan.",
            },
            usage=None,
        )
        plan = {"title": ["Diamant ATR"], "description": ["Diamant ATR sampling procedure"]}
        synthesis = Mock(output=plan, usage=None)
        validation_schema = {
            "type": "object",
            "properties": {
                "was_generated_by": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "realized_plan": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "title": {"type": "array", "items": {"type": "string"}},
                                    "description": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["title", "description"],
                            }
                        },
                    },
                }
            },
        }

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(side_effect=[diagnosis, synthesis]),
        ):
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"was_generated_by": [{}]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="method_plan_presence",
                    label="Method plan",
                    description="Method plan",
                    target_paths=["/was_generated_by/0/realized_plan"],
                ),
                validation_schema=validation_schema,
            )

        self.assertEqual(updated["was_generated_by"][0]["realized_plan"], plan)
        self.assertEqual(paths, ["/was_generated_by/0/realized_plan"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 1)
        self.assertEqual(item.compiled_actions[0]["mode"], "replace")

    async def test_semantic_synthesis_refuses_collection_wide_attribute_replacement(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_parent_placement",
            label="Attribute labels",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/is_about_entity/0/has_quantitative_attribute"],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_label",
                        "target_path": "/is_about_entity/0/has_quantitative_attribute",
                        "entry_indices": [0, 1],
                        "recommended_action": "replace",
                        "needs_synthesis": True,
                        "reason": "Relabel two entries.",
                    }
                ],
                "reason": "Repair labels.",
            },
            usage=None,
        )
        document = {
            "is_about_entity": [
                {
                    "has_quantitative_attribute": [
                        {"title": "First X", "value": 10.0, "has_quantity_type": "first X"},
                        {"title": "Last X", "value": 1.0, "has_quantity_type": "last X"},
                        {"title": "Resolution", "value": 4.0, "has_quantity_type": "resolution"},
                    ]
                }
            ]
        }

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(return_value=diagnosis),
        ) as mocked:
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document=document,
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_parent_placement",
                    label="Attribute labels",
                    description="Concise labels",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        self.assertEqual(updated, document)
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertEqual(applied, 0)
        self.assertIn("refusing collection-wide replacement", rejected[0])
        self.assertEqual(mocked.await_count, 1)

    async def test_semantic_synthesis_refuses_mechanical_merge(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_parent_placement",
            label="Attribute parents",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/is_about_entity/0/has_quantitative_attribute"],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "duplicate_attribute",
                        "target_path": "/is_about_entity/0/has_quantitative_attribute",
                        "entry_indices": [0, 1],
                        "recommended_action": "merge",
                        "needs_synthesis": True,
                        "reason": "Merge duplicates.",
                    }
                ],
                "reason": "Repair duplicates.",
            },
            usage=None,
        )
        document = {
            "is_about_entity": [
                {
                    "has_quantitative_attribute": [
                        {"value": 1.0, "has_quantity_type": "point count"},
                        {"value": 1.0, "has_quantity_type": "point count"},
                    ]
                }
            ]
        }

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(return_value=diagnosis),
        ) as mocked:
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document=document,
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_parent_placement",
                    label="Attribute parents",
                    description="Place attributes",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        self.assertEqual(updated, document)
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertEqual(applied, 0)
        self.assertIn("must not request synthesis", rejected[0])
        self.assertEqual(mocked.await_count, 1)

    async def test_semantic_synthesis_refuses_unsupported_attribute_append(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_duplicate_coherence",
            label="Attribute duplicate coherence",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/was_generated_by/0/has_quantitative_attribute"],
            selected_evidence=[],
            context_window=[],
        )
        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "missing_plan",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "The target path is missing.",
                    }
                ],
                "reason": "Append missing attribute.",
            },
            usage=None,
        )
        document = {"was_generated_by": [{}]}

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(return_value=diagnosis),
        ) as mocked:
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document=document,
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_duplicate_coherence",
                    label="Attribute duplicate coherence",
                    description="Duplicate attributes are merged or removed.",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        self.assertEqual(updated, document)
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertEqual(applied, 0)
        self.assertIn("Attribute synthesis requires selected evidence", rejected[0])
        self.assertEqual(mocked.await_count, 1)

    def test_sanitizer_preserves_nested_attribute_payload(self):
        write = SchemaConstrainedWrite(
            target_path="/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute",
            mode="append",
            items=[
                {
                    "title": "Maximum transmittance",
                    "value": 0.98,
                    "has_quantity_type": "maximum transmittance",
                }
            ],
        )

        sanitized = WorkflowService._sanitize_schema_constrained_writes(
            writes=[write],
            requirement_id="semantic_reconstruction",
        )

        self.assertEqual(sanitized[0].items, write.items)

    def test_fulfillment_guard_downgrades_empty_provenance_target(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "technical_agent_kind"
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=requirement.weight,
            target_paths=requirement.target_paths,
            selected_evidence=[
                RequirementEvidenceItem(
                    candidate_id="origin",
                    category="surrounding_signal",
                    role="context",
                    claim="The origin is Example Lab.",
                    evidence_text="ORIGIN=Example Lab",
                )
            ],
        )

        WorkflowService._guard_semantic_requirement_assessment(
            requirement=requirement,
            document={"title": ["Dataset"]},
            item=item,
        )

        self.assertEqual(item.status, "partial")
        self.assertEqual(item.quality, 0.5)
        self.assertIn("none of the requirement target paths", item.rationale)

    def test_semantic_revalidation_carries_reconstruction_trace(self):
        item = RequirementReportItem(
            requirement_id="method_plan_presence",
            label="Method plan",
            weight=1.0,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=1.0,
        )
        record = SemanticReconstructionRecord(
            requirement_id="method_plan_presence",
            status="applied",
            defect_type="missing_plan",
            diagnosed_defects_count=1,
            compiled_actions_count=1,
            synthesis_calls_count=1,
            diagnosed_defects=[{"defect_type": "missing_plan"}],
            compiled_actions=[{"mode": "replace", "target_path": "/was_generated_by/0/realized_plan"}],
        )

        WorkflowService._carry_semantic_reconstruction_trace(
            semantic_items=[item],
            records=[record],
        )

        self.assertEqual(item.defect_type, "missing_plan")
        self.assertEqual(item.diagnosed_defects_count, 1)
        self.assertEqual(item.compiled_actions_count, 1)
        self.assertEqual(item.synthesis_calls_count, 1)
        self.assertEqual(item.diagnosed_defects, record.diagnosed_defects)
        self.assertEqual(item.compiled_actions, record.compiled_actions)

    def test_technical_agent_guard_recognizes_software_type(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "technical_agent_kind"
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            target_paths=requirement.target_paths,
        )

        WorkflowService._guard_semantic_requirement_assessment(
            requirement=requirement,
            document={
                "was_generated_by": [
                    {
                        "carried_out_by": [
                            {
                                "id": "CHEMSPECTRA",
                                "type": {
                                    "from_CV": "AgenticEntity",
                                    "id": "software",
                                    "title": "Software",
                                },
                            }
                        ]
                    }
                ]
            },
            item=item,
        )

        self.assertEqual(item.status, "fulfilled")
        self.assertEqual(item.quality, 1.0)
        self.assertIn("software", item.rationale)

    async def test_requirement_gap_patcher_uses_schema_constrained_output(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="instrument_settings_semantics",
            label="Instrument settings",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/was_generated_by/0/has_quantitative_attribute"],
        )
        llm_result = Mock(
            output={
                "writes": [
                    {
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "mode": "append",
                        "items": [{"value": 0.93, "has_quantity_type": "threshold"}],
                        "reason": "Add threshold.",
                    }
                ],
                "reason": "Add threshold.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            updated, attempt = await service._patch_requirement_gap(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"was_generated_by": [{"has_quantitative_attribute": []}]},
                requirement=DcatRequirement(
                    requirement_id="instrument_settings_semantics",
                    label="Instrument settings",
                    description="Instrument settings semantics",
                    target_paths=["/was_generated_by/0/has_quantitative_attribute"],
                    expected_target_class="QuantitativeAttribute",
                ),
                item=item,
                validation_schema=quantitative_schema("DataGeneratingActivity"),
            )

        self.assertEqual(attempt.status, "applied")
        self.assertEqual(updated["was_generated_by"][0]["has_quantitative_attribute"][0]["value"], 0.93)
        self.assertIsInstance(mocked.call_args.kwargs["output_type"], dict)

    async def test_semantic_reconstruction_diagnosis_rejects_action_envelope(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="dataset_description_scope",
            label="Description",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/description"],
        )
        llm_result = Mock(output={"writes": [], "reason": "No safe write."}, usage=None)

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            updated, paths, reason, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"description": ["Original."], "dataset_distribution": [{"title": ["D"]}]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="dataset_description_scope",
                    label="Description",
                    description="Description identity",
                    target_paths=["/description"],
                ),
                validation_schema={
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "type": "object",
                    "properties": {
                        "description": {"type": "array", "items": {"type": "string"}},
                        "dataset_distribution": {"type": "array", "items": {"type": "object"}},
                    },
                },
            )

        self.assertEqual(updated["description"], ["Original."])
        self.assertEqual(paths, [])
        self.assertIn("patch/action syntax", reason)
        self.assertEqual(errors, [reason])
        output_schema = mocked.call_args.kwargs["output_type"]
        self.assertIn("defects", output_schema["properties"])

    def test_schema_constrained_apply_normalizes_legacy_append_path(self):
        document = {"was_generated_by": [{"has_quantitative_attribute": []}]}
        updated, paths = apply_schema_constrained_writes(
            document=document,
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute/-",
                    mode="append",
                    items=[{"value": 0.93, "has_quantity_type": "threshold"}],
                    reason="legacy append path",
                )
            ],
            data_package_id="pkg",
            validation_schema=quantitative_schema("DataGeneratingActivity"),
        )

        self.assertEqual(paths, ["/was_generated_by/0/has_quantitative_attribute/0"])
        self.assertEqual(updated["was_generated_by"][0]["has_quantitative_attribute"][0]["value"], 0.93)

    def test_schema_constrained_apply_does_not_add_id_for_union_object_without_id(self):
        schema = {
            "type": "object",
            "properties": {
                "was_generated_by": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "realized_plan": {
                                "anyOf": [
                                    {"$ref": "#/$defs/Plan"},
                                    {"type": "null"},
                                ]
                            }
                        },
                    },
                }
            },
            "$defs": {
                "Plan": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                    },
                }
            },
        }

        updated, _ = apply_schema_constrained_writes(
            document={"was_generated_by": [{}]},
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/realized_plan",
                    mode="replace",
                    value={"title": "Diamant ATR", "description": "ATR sampling procedure"},
                )
            ],
            data_package_id="pkg",
            validation_schema=schema,
        )

        plan = updated["was_generated_by"][0]["realized_plan"]
        self.assertNotIn("id", plan)
        Draft201909Validator(schema).validate(updated)

    def test_semantic_action_compiler_skips_duplicate_quantitative_write(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated, paths, reason, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {"value": 0.93, "has_quantity_type": "threshold"}
                        ]
                    }
                ]
            },
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute",
                    mode="append",
                    items=[
                        {
                            "title": "Threshold",
                            "description": "Threshold: 0.93",
                            "value": 0.93,
                            "has_quantity_type": "threshold",
                        }
                    ],
                    reason="duplicate",
                )
            ],
            reason="duplicate",
            validation_schema=quantitative_schema("DataGeneratingActivity"),
        )

        self.assertEqual(paths, [])
        self.assertEqual(errors, [])
        self.assertIn("duplicate_semantic_slot", reason)
        self.assertEqual(len(updated["was_generated_by"][0]["has_quantitative_attribute"]), 1)

    async def test_semantic_reconstruction_rejects_json_patch_syntax(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        llm_result = Mock(
            output={
                "writes": [
                    {
                        "op": "replace",
                        "path": "/description",
                        "value": ["Bad syntax."],
                    }
                ],
                "reason": "legacy",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)):
            updated, paths, reason, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"description": ["Original."]},
                item=RequirementReportItem(
                    requirement_id="dataset_identity_semantics",
                    label="Identity",
                    weight=1.0,
                    status="partial",
                    applicable=True,
                    quality=0.5,
                    weighted_score=0.5,
                    target_paths=["/description"],
                ),
                requirement=DcatRequirement(
                    requirement_id="dataset_identity_semantics",
                    label="Identity",
                    description="Identity semantics",
                    target_paths=["/description"],
                ),
                validation_schema={
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "type": "object",
                    "properties": {
                        "description": {"type": "array", "items": {"type": "string"}},
                        "dataset_distribution": {"type": "array", "items": {"type": "object"}},
                    },
                },
            )

        self.assertEqual(updated["description"], ["Original."])
        self.assertEqual(paths, [])
        self.assertEqual(applied, 0)
        self.assertIn("patch/action syntax", reason)
        self.assertEqual(errors, rejected)

    def test_semantic_action_compiler_remove_deletes_duplicate_attribute(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated, paths, reason, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {"value": 0.93, "has_quantity_type": "threshold"},
                            {"value": 0.93, "has_quantity_type": "threshold"},
                        ]
                    }
                ]
            },
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute/1",
                    mode="remove",
                    reason="Remove duplicate threshold.",
                )
            ],
            reason="Remove duplicate threshold.",
            validation_schema={},
        )

        self.assertEqual(len(updated["was_generated_by"][0]["has_quantitative_attribute"]), 1)
        self.assertEqual(paths, ["/was_generated_by/0/has_quantitative_attribute/1"])
        self.assertEqual(errors, [])
        self.assertEqual(applied, 1)
        self.assertEqual(rejected, [])
        self.assertEqual(reason, "Remove duplicate threshold.")

    async def test_attribute_construction_cleanup_merges_point_count_variants(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated = service._cleanup_parent_attributes_after_construction(
            data_package_id="pkg",
            profile_identifier="profile",
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {"title": "Number data points", "value": 2559, "has_quantity_type": "number data points"},
                            {"title": "Point count", "value": 2559, "has_quantity_type": "point count"},
                        ]
                    }
                ]
            },
            validation_schema={},
        )

        attributes = updated["was_generated_by"][0]["has_quantitative_attribute"]
        self.assertEqual(len(attributes), 1)
        self.assertEqual(attributes[0]["has_quantity_type"], "number data points")

    async def test_attribute_construction_cleanup_merges_max_transmittance_variants(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated = service._cleanup_parent_attributes_after_construction(
            data_package_id="pkg",
            profile_identifier="profile",
            document={
                "is_about_entity": [
                    {
                        "has_quantitative_attribute": [
                            {"title": "Maximum Y", "value": 0.98018688, "has_quantity_type": "maximum Y"},
                            {"title": "Maximum transmittance dataset", "value": 0.9801868804981199, "has_quantity_type": "maximum transmittance dataset"},
                        ]
                    }
                ]
            },
            validation_schema={},
        )

        attributes = updated["is_about_entity"][0]["has_quantitative_attribute"]
        self.assertEqual(len(attributes), 1)
        self.assertEqual(attributes[0]["value"], 0.9801868804981199)

    async def test_attribute_construction_cleanup_keeps_materially_different_values(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated = service._cleanup_parent_attributes_after_construction(
            data_package_id="pkg",
            profile_identifier="profile",
            document={
                "is_about_entity": [
                    {
                        "has_quantitative_attribute": [
                            {"value": 0.98, "has_quantity_type": "maximum Y"},
                            {"value": 0.91, "has_quantity_type": "maximum transmittance"},
                        ]
                    }
                ]
            },
            validation_schema={},
        )

        self.assertEqual(len(updated["is_about_entity"][0]["has_quantitative_attribute"]), 2)

    async def test_semantic_range_decomposition_splits_bad_range_attribute(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_range_decomposition",
            label="Range decomposition",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/was_generated_by/0/has_quantitative_attribute"],
            selected_evidence=[
                RequirementEvidenceItem(
                    candidate_id="range",
                    category="measurement_condition",
                    role="parameter",
                    claim="Wavenumber range from 373.96 to 3997.45 1/CM.",
                    evidence_text="spanning 373.96-3997.45 1/CM",
                )
            ],
        )

        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "Add minimum wavenumber.",
                    },
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "Add maximum wavenumber.",
                    },
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [0],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Remove collapsed range.",
                    },
                ],
                "reason": "Decompose range.",
            },
            usage=None,
        )
        minimum = Mock(output={"title": "Minimum wavenumber", "value": 373.96, "has_quantity_type": "minimum wavenumber"}, usage=None)
        maximum = Mock(output={"title": "Maximum wavenumber", "value": 3997.45, "has_quantity_type": "maximum wavenumber"}, usage=None)

        with patch("app.services.projection_service.generate_structured", AsyncMock(side_effect=[diagnosis, minimum, maximum])):
            updated, paths, reason, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={
                    "was_generated_by": [
                        {
                            "has_quantitative_attribute": [
                                {
                                    "title": "Wavenumber range",
                                    "description": "Wavenumber range: 373.96 to",
                                    "value": 373.96,
                                    "has_quantity_type": "Wavenumber range",
                                    "unit": "to",
                                }
                            ]
                        }
                    ]
                },
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_range_decomposition",
                    label="Range decomposition",
                    description="Range decomposition",
                    target_paths=["/was_generated_by/0/has_quantitative_attribute"],
                ),
                validation_schema={},
            )

        attributes = updated["was_generated_by"][0]["has_quantitative_attribute"]
        self.assertEqual([attribute["has_quantity_type"] for attribute in attributes], ["minimum wavenumber", "maximum wavenumber"])
        self.assertEqual([attribute["value"] for attribute in attributes], [373.96, 3997.45])
        self.assertIn("/was_generated_by/0/has_quantitative_attribute/0", paths)
        self.assertEqual(errors, [])
        self.assertEqual(applied, 3)

    async def test_semantic_range_decomposition_reuses_existing_precise_bounds(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_range_decomposition",
            label="Range decomposition",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/is_about_entity/0/has_quantitative_attribute"],
            selected_evidence=[
                RequirementEvidenceItem(
                    candidate_id="range",
                    category="measurement_condition",
                    role="parameter",
                    claim="Wavenumber range from 373.96 to 3997.45 1/CM.",
                    evidence_text="2559 points spanning 373.96-3997.45 1/CM",
                ),
                RequirementEvidenceItem(
                    candidate_id="step",
                    category="measurement_condition",
                    role="parameter",
                    claim="The X-axis step is -1.4165319 1/CM.",
                    evidence_text="DELTAX=-1.4165319",
                ),
            ],
        )

        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_range",
                        "target_path": "/is_about_entity/0/has_quantitative_attribute",
                        "entry_indices": [2],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Precise bounds already exist.",
                    }
                ],
                "reason": "Remove collapsed range.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=diagnosis)):
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={
                    "is_about_entity": [
                        {
                            "has_quantitative_attribute": [
                                {"title": "First X 1 CM", "value": 3997.453, "has_quantity_type": "first X 1 CM"},
                                {"title": "Last X 1 CM", "value": 373.96442, "has_quantity_type": "last X 1 CM"},
                                {
                                    "title": "Wavenumber range from 3997 45 1 CM",
                                    "description": "Wavenumber range from 3997 45 1 CM: 373.96 to",
                                    "value": 373.96,
                                    "has_quantity_type": "Wavenumber range from 3997 45 1 CM",
                                    "unit": "to",
                                },
                            ]
                        }
                    ]
                },
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_range_decomposition",
                    label="Range decomposition",
                    description="Range decomposition",
                    target_paths=["/is_about_entity/0/has_quantitative_attribute"],
                ),
                validation_schema={},
            )

        attributes = updated["is_about_entity"][0]["has_quantitative_attribute"]
        self.assertEqual(len(attributes), 2)
        self.assertEqual([attribute["value"] for attribute in attributes], [3997.453, 373.96442])
        self.assertEqual(paths, ["/is_about_entity/0/has_quantitative_attribute/2"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 1)

    async def test_semantic_range_decomposition_uses_sibling_bounds_without_range_evidence(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_range_decomposition",
            label="Range decomposition",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/is_about_entity/0/has_quantitative_attribute"],
            selected_evidence=[
                RequirementEvidenceItem(
                    candidate_id="max-y",
                    category="measurement_condition",
                    role="parameter",
                    claim="The maximum Y value is 0.98.",
                    evidence_text="MAXY=0.98",
                )
            ],
        )
        attributes = [
            {"title": "Resolution", "value": 4.0, "has_quantity_type": "resolution"},
            {"title": "First X 1 CM", "value": 3997.453, "has_quantity_type": "first X 1 CM"},
            {"title": "Last X 1 CM", "value": 373.96442, "has_quantity_type": "last X 1 CM"},
            {"title": "Point count", "value": 2559.0, "has_quantity_type": "point count"},
            {
                "title": "Wavenumber range from 3997 45 1 CM",
                "description": "Wavenumber range from 3997 45 1 CM: 373.96 to",
                "value": 373.96,
                "has_quantity_type": "Wavenumber range from 3997 45 1 CM",
                "unit": "to",
            },
        ]

        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_range",
                        "target_path": "/is_about_entity/0/has_quantitative_attribute",
                        "entry_indices": [4],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Sibling bounds already represent the range.",
                    }
                ],
                "reason": "Remove collapsed range.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=diagnosis)):
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"is_about_entity": [{"has_quantitative_attribute": attributes}]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_range_decomposition",
                    label="Range decomposition",
                    description="Range decomposition",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        repaired = updated["is_about_entity"][0]["has_quantitative_attribute"]
        self.assertEqual(len(repaired), 4)
        self.assertEqual([entry["value"] for entry in repaired], [4.0, 3997.453, 373.96442, 2559.0])
        self.assertEqual(paths, ["/is_about_entity/0/has_quantitative_attribute/4"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 1)

    async def test_semantic_range_decomposition_uses_dataset_description_bounds(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_range_decomposition",
            label="Range decomposition",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/was_generated_by/0/has_quantitative_attribute"],
        )
        document = {
            "description": [
                "Transmittance values recorded at 2559 wavenumber points spanning 373.96-3997.45 1/CM."
            ],
            "was_generated_by": [
                {
                    "has_quantitative_attribute": [
                        {
                            "title": "Wavenumber range",
                            "value": 373.96,
                            "has_quantity_type": "wavenumber_range",
                        }
                    ]
                }
            ],
        }

        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "Add minimum wavenumber.",
                    },
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [],
                        "recommended_action": "append",
                        "needs_synthesis": True,
                        "reason": "Add maximum wavenumber.",
                    },
                    {
                        "defect_type": "bad_range",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [0],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Remove collapsed range.",
                    },
                ],
                "reason": "Decompose range.",
            },
            usage=None,
        )
        minimum = Mock(output={"title": "Minimum wavenumber", "value": 373.96, "has_quantity_type": "minimum wavenumber", "unit": "1/cm"}, usage=None)
        maximum = Mock(output={"title": "Maximum wavenumber", "value": 3997.45, "has_quantity_type": "maximum wavenumber", "unit": "1/cm"}, usage=None)

        with patch("app.services.projection_service.generate_structured", AsyncMock(side_effect=[diagnosis, minimum, maximum])):
            updated, _, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document=document,
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_range_decomposition",
                    label="Range decomposition",
                    description="Range decomposition",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        repaired = updated["was_generated_by"][0]["has_quantitative_attribute"]
        self.assertEqual([entry["has_quantity_type"] for entry in repaired], ["minimum wavenumber", "maximum wavenumber"])
        self.assertEqual([entry["value"] for entry in repaired], [373.96, 3997.45])
        self.assertEqual([entry["unit"] for entry in repaired], ["1/cm", "1/cm"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 3)

    async def test_parent_placement_removes_cross_parent_measurement_duplicates(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        item = RequirementReportItem(
            requirement_id="attribute_parent_placement",
            label="Attribute parent placement",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=[
                "/was_generated_by/0/has_quantitative_attribute",
                "/is_about_entity/0/has_quantitative_attribute",
            ],
        )
        document = {
            "was_generated_by": [
                {
                    "has_quantitative_attribute": [
                        {"value": 2559.0, "has_quantity_type": "Transmittance values recorded wavenumber points"},
                        {"value": 0.98018688, "has_quantity_type": "maximum Y value", "unit": "transmittance"},
                    ]
                }
            ],
            "is_about_entity": [
                {
                    "has_quantitative_attribute": [
                        {"title": "Total number data points", "value": 2559.0, "has_quantity_type": "total number data points"},
                        {"title": "Maximum Y transmittance", "value": 0.9801868804, "has_quantity_type": "maximum Y transmittance"},
                        {"title": "Resolution", "value": 4.0, "has_quantity_type": "resolution"},
                    ]
                }
            ],
        }

        diagnosis = Mock(
            output={
                "defects": [
                    {
                        "defect_type": "wrong_parent",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "entry_indices": [0, 1],
                        "recommended_action": "remove",
                        "needs_synthesis": False,
                        "reason": "Measurement attributes are already represented on the evaluated entity.",
                    }
                ],
                "reason": "Remove wrongly placed activity attributes.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=diagnosis)) as mocked:
            updated, paths, _, errors, applied, rejected = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document=document,
                item=item,
                requirement=DcatRequirement(
                    requirement_id="attribute_parent_placement",
                    label="Attribute parent placement",
                    description="Place attributes on their semantic owner",
                    target_paths=item.target_paths,
                ),
                validation_schema={},
            )

        self.assertEqual(updated["was_generated_by"][0]["has_quantitative_attribute"], [])
        self.assertEqual(len(updated["is_about_entity"][0]["has_quantitative_attribute"]), 3)
        self.assertEqual(
            paths,
            [
                "/was_generated_by/0/has_quantitative_attribute/1",
                "/was_generated_by/0/has_quantitative_attribute/0",
            ],
        )
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 2)
        self.assertIn("attribute_parents", mocked.call_args.kwargs["prompt"])

    def test_semantic_action_compiler_invalid_merge_is_rejected(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        document = {
            "was_generated_by": [
                {
                    "has_quantitative_attribute": [
                        {"value": 2559, "has_quantity_type": "point count"},
                    ]
                }
            ]
        }

        updated, paths, reason, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document=document,
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute",
                    mode="merge",
                    survivor_index=0,
                    merged_indices=[3],
                    reason="Bad merge.",
                )
            ],
            reason="Bad merge.",
            validation_schema={},
        )

        self.assertEqual(updated, document)
        self.assertEqual(paths, [])
        self.assertEqual(applied, 0)
        self.assertIn("out of range", rejected[0])
        self.assertEqual(errors, rejected)

    def test_duplicate_requirement_patch_uses_semantic_slot(self):
        reason = WorkflowService._duplicate_requirement_patch_reason(
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {
                                "title": "Number data points",
                                "value": 2559,
                                "has_quantity_type": "number data points",
                            }
                        ]
                    }
                ]
            },
            target_path="/was_generated_by/0/has_quantitative_attribute/-",
            instance={
                "title": "Point count",
                "value": 2559,
                "has_quantity_type": "point count",
            },
        )

        self.assertIn("duplicate_semantic_slot", reason)

    def test_duplicate_requirement_patch_keeps_same_value_different_quantity(self):
        reason = WorkflowService._duplicate_requirement_patch_reason(
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {"value": 93, "has_quantity_type": "peak threshold", "unit": "%"}
                        ]
                    }
                ]
            },
            target_path="/was_generated_by/0/has_quantitative_attribute/-",
            instance={"value": 93, "has_quantity_type": "scan count"},
        )

        self.assertIsNone(reason)

    def test_duplicate_requirement_patch_keeps_same_slot_under_different_parent(self):
        reason = WorkflowService._duplicate_requirement_patch_reason(
            document={
                "was_generated_by": [
                    {
                        "has_quantitative_attribute": [
                            {"value": 2559, "has_quantity_type": "point count"}
                        ]
                    }
                ],
                "is_about_entity": [{"has_quantitative_attribute": []}],
            },
            target_path="/is_about_entity/0/has_quantitative_attribute/-",
            instance={"value": 2559, "has_quantity_type": "point count"},
        )

        self.assertIsNone(reason)

    def test_parent_candidate_paths_route_measurement_conditions_to_generation_activity(self):
        paths = WorkflowService._attribute_parent_candidate_paths(
            document={"is_about_entity": [{"title": "sample", "has_quantitative_attribute": []}]},
            selected_evidence=[
                RequirementEvidenceItem(
                    candidate_id="range",
                    category="measurement_condition",
                    role="parameter",
                    claim="Range is 1 to 2 units.",
                    evidence_text="range 1-2 units",
                )
            ],
            fallback_paths=[
                "/was_generated_by/0/has_quantitative_attribute",
                "/is_about_entity/0/has_quantitative_attribute",
            ],
        )

        self.assertEqual(paths[0], "/was_generated_by/0/has_quantitative_attribute")

    def test_semantic_action_compiler_creates_missing_generation_activity_parent(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated, paths, reason, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document={"id": "pkg"},
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute",
                    mode="append",
                    items=[{"value": 2559, "has_quantity_type": "data point count"}],
                    reason="Attach measurement condition to generation activity.",
                )
            ],
            reason="Attach measurement condition to generation activity.",
            validation_schema=quantitative_schema("DataGeneratingActivity"),
        )

        self.assertEqual(paths, ["/was_generated_by/0/has_quantitative_attribute/0"])
        self.assertEqual(errors, [])
        self.assertIn("was_generated_by", updated)
        self.assertEqual(updated["was_generated_by"][0]["has_quantitative_attribute"][0]["value"], 2559)
        self.assertEqual(reason, "Attach measurement condition to generation activity.")

    def test_semantic_action_compiler_salvages_valid_writes_when_one_fails_validation(self):
        class RejectBadAttributeProfileService(FakeProfileService):
            def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
                for index, attribute in enumerate(document["was_generated_by"][0]["has_quantitative_attribute"]):
                    if attribute.get("has_quantity_type") == "bad setting":
                        return ProfileValidationResult(
                            valid=False,
                            errors=[
                                ProfileValidationIssue(
                                    path=f"/was_generated_by/0/has_quantitative_attribute/{index}",
                                    message="Rejected bad setting",
                                    schema_path="",
                                )
                            ],
                        )
                return ProfileValidationResult(valid=True, errors=[])

        service = WorkflowService(
            profile_service=RejectBadAttributeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        updated, paths, reason, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document={"was_generated_by": [{"has_quantitative_attribute": []}]},
            writes=[
                SchemaConstrainedWrite(
                    target_path="/was_generated_by/0/has_quantitative_attribute",
                    mode="append",
                    items=[
                        {
                            "title": "Transmittance",
                            "description": "Transmittance value range maximum.",
                            "value": 0.98,
                            "has_quantity_type": "maximum transmittance",
                        },
                        {
                            "title": "Bad",
                            "description": "Rejected test setting.",
                            "value": 1.0,
                            "has_quantity_type": "bad setting",
                        },
                    ],
                    reason="Attach instrument evidence.",
                )
            ],
            reason="Attach instrument evidence.",
            validation_schema=quantitative_schema("DataGeneratingActivity"),
        )

        attributes = updated["was_generated_by"][0]["has_quantitative_attribute"]
        self.assertEqual(len(attributes), 1)
        self.assertEqual(attributes[0]["has_quantity_type"], "maximum transmittance")
        self.assertEqual(paths, ["/was_generated_by/0/has_quantitative_attribute/0"])
        self.assertIn("Applied valid operations", reason)
        self.assertEqual(errors, [])

    async def test_semantic_reconstruction_rolls_back_failed_validation(self):
        class InvalidProfileService(FakeProfileService):
            def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
                return ProfileValidationResult(
                    valid=False,
                    errors=[ProfileValidationIssue(path="/description", message="bad", schema_path="")],
                )

        service = WorkflowService(
            profile_service=InvalidProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        state = ExtractionRunState(generated_final_draft={"description": ["Original."]})
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="dataset_identity_semantics",
            label="Identity",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/description"],
        )
        service._semantic_reconstruction_update = AsyncMock(
            return_value=({"description": ["Updated."]}, ["/description"], "LLM patch.", [])
        )

        document, records = await service._reconstruct_semantic_defects(
            data_package_id="pkg",
            profile_identifier="profile",
            document={"description": ["Original. Raw data points 10."]},
            semantic_items=[item],
            validation_schema={},
            state=state,
            progress=progress,
        )

        self.assertEqual(document["description"], ["Original. Raw data points 10."])
        self.assertEqual(records[-1].status, "rolled_back")
        self.assertEqual(records[-1].validation_errors, ["bad"])

    async def test_semantic_reconstruction_empty_defective_action_is_unresolved(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        state = ExtractionRunState(generated_final_draft={"description": ["Original."]})
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="dataset_description_scope",
            label="Description",
            weight=1.0,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.5,
            target_paths=["/description"],
        )
        service._semantic_reconstruction_update = AsyncMock(
            return_value=({"description": ["Original."]}, [], "semantic_defect_unresolved_empty_diagnosis", [], 0, [])
        )

        document, records = await service._reconstruct_semantic_defects(
            data_package_id="pkg",
            profile_identifier="profile",
            document={"description": ["Original."]},
            semantic_items=[item],
            validation_schema={},
            state=state,
            progress=progress,
        )

        self.assertEqual(document["description"], ["Original."])
        self.assertEqual(records[-1].status, "unresolved")
        self.assertEqual(records[-1].reason, "semantic_defect_unresolved_empty_diagnosis")

    def test_requirement_patch_strip_removes_auto_ids_from_schema_forbidden_targets(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        document = {
            "was_generated_by": [
                {
                    "realized_plan": {"id": "auto-id", "title": "zg30", "was_generated_by": []},
                    "has_quantitative_attribute": [{"id": "auto-id", "value": 500.13, "has_quantity_type": "frequency"}],
                }
            ]
        }

        service._strip_requirement_patch_forbidden_fields(
            document=document,
            target_path="/was_generated_by/0/realized_plan",
            target_class="Plan",
            appended_index=None,
        )
        service._strip_requirement_patch_forbidden_fields(
            document=document,
            target_path="/was_generated_by/0/has_quantitative_attribute/-",
            target_class="QuantitativeAttribute",
            appended_index=0,
        )

        self.assertNotIn("id", document["was_generated_by"][0]["realized_plan"])
        self.assertNotIn("was_generated_by", document["was_generated_by"][0]["realized_plan"])
        self.assertNotIn("id", document["was_generated_by"][0]["has_quantitative_attribute"][0])

    def test_activity_evaluation_target_expands_paths_for_every_activity(self):
        configured = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        )

        requirement = WorkflowService._runtime_semantic_requirement(
            requirement=configured,
            document={"was_generated_by": [{"id": "a:0"}, {"id": "a:1"}]},
        )

        self.assertEqual(
            requirement.target_paths,
            [
                "/was_generated_by/0/evaluated_entity",
                "/was_generated_by/0/evaluated_activity",
                "/was_generated_by/1/evaluated_entity",
                "/was_generated_by/1/evaluated_activity",
            ],
        )

    def test_activity_evaluation_target_guard_checks_every_activity(self):
        configured = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        )
        document = {
            "was_generated_by": [
                {"id": "a:0", "evaluated_entity": [{"id": "sample:1"}]},
                {"id": "a:1"},
            ]
        }
        requirement = WorkflowService._runtime_semantic_requirement(
            requirement=configured,
            document=document,
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=requirement.weight,
            target_paths=requirement.target_paths,
        )

        WorkflowService._guard_semantic_requirement_assessment(
            requirement=requirement,
            document=document,
            item=item,
        )

        self.assertEqual(item.status, "partial")
        self.assertIn("without an evaluation target", item.rationale)

    def test_activity_evaluation_target_guard_rejects_self_reference(self):
        configured = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        )
        document = {
            "was_generated_by": [
                {"id": "activity:1", "evaluated_activity": [{"id": "activity:1"}]},
            ]
        }
        requirement = WorkflowService._runtime_semantic_requirement(
            requirement=configured,
            document=document,
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=requirement.weight,
            target_paths=requirement.target_paths,
        )

        WorkflowService._guard_semantic_requirement_assessment(
            requirement=requirement,
            document=document,
            item=item,
        )

        self.assertEqual(item.status, "missing")
        self.assertIn("no evaluation target", item.rationale)

    async def test_activity_evaluation_target_compiler_removes_self_reference(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        )
        document = {
            "was_generated_by": [
                {"id": "activity:1", "evaluated_activity": [{"id": "activity:1"}]},
            ]
        }
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=0.625,
            target_paths=[
                "/was_generated_by/0/evaluated_entity",
                "/was_generated_by/0/evaluated_activity",
            ],
        )

        compiled = await service._compile_deterministic_semantic_actions(
            data_package_id="pkg",
            document=document,
            item=item,
            requirement=requirement,
            validation_schema={},
        )

        self.assertEqual(len(compiled.writes), 1)
        self.assertEqual(compiled.writes[0].mode, "remove")
        self.assertEqual(
            compiled.writes[0].target_path,
            "/was_generated_by/0/evaluated_activity/0",
        )

    def test_unsupported_subject_edge_removal_preserves_supported_activity_target(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        document = {
            "is_about_entity": [{"id": "sample:1", "title": "Sample"}],
            "was_generated_by": [
                {
                    "id": "activity:1",
                    "evaluated_entity": [{"id": "sample:1", "title": "Sample"}],
                }
            ],
        }

        updated, paths, _, errors, applied, rejected = service._apply_semantic_reconstruction_writes(
            data_package_id="pkg",
            profile_identifier="profile",
            document=document,
            writes=[
                SchemaConstrainedWrite(
                    target_path="/is_about_entity/0",
                    mode="remove",
                    reason="Dataset-subject edge lacks independent support.",
                )
            ],
            reason="Remove unsupported mirrored edge.",
            validation_schema={},
        )

        self.assertEqual(updated["is_about_entity"], [])
        self.assertEqual(updated["was_generated_by"][0]["evaluated_entity"][0]["id"], "sample:1")
        self.assertEqual(paths, ["/is_about_entity/0"])
        self.assertEqual(errors, [])
        self.assertEqual(rejected, [])
        self.assertEqual(applied, 1)

    def test_semantic_write_rejects_evaluated_activity_self_reference(self):
        document = {"was_generated_by": [{"id": "activity:1"}]}
        rejected = WorkflowService._validate_semantic_reconstruction_action(
            document=document,
            write=SchemaConstrainedWrite(
                target_path="/was_generated_by/0/evaluated_activity",
                mode="append",
                items=[{"id": "activity:1", "title": ["Self"]}],
                reason="Invalid self target.",
            ),
        )

        self.assertEqual(len(rejected), 1)
        self.assertIn("must not self-reference", rejected[0])


    def test_profile_rebuild_clears_stale_curated_and_semantic_artifacts(self):
        state = ExtractionRunState(
            generated_initial_draft={"id": "old-initial"},
            generated_final_draft={"id": "old-final"},
            generated_core_draft={"id": "old-core"},
            generated_attribute_draft={"id": "old-attribute"},
            generated_reconstructed_draft={"id": "old-reconstructed"},
            curated_document={"id": "old-curated"},
            requirement_report=build_requirement_report(
                schema_valid=True,
                coverage=compute_coverage_report({}, {}),
                semantic_requirements=[],
                source_trace=compute_source_trace_report(RoutedEvidenceContext(), []),
                coverage_patches=[],
            ),
            curation_ledger=[CurationLedgerRecord(json_path="/title", field_name="title")],
        )
        progress = ExtractionRunProgress(
            generated_initial_draft={"id": "old-initial"},
            generated_final_draft={"id": "old-final"},
            generated_core_draft={"id": "old-core"},
            generated_attribute_draft={"id": "old-attribute"},
            generated_reconstructed_draft={"id": "old-reconstructed"},
            curated_document={"id": "old-curated"},
            requirement_report=state.requirement_report,
        )

        WorkflowService._clear_profile_projection_state(state)
        WorkflowService._clear_profile_projection_progress(progress)

        for artifact in (
            state.generated_initial_draft,
            state.generated_final_draft,
            state.generated_core_draft,
            state.generated_attribute_draft,
            state.generated_reconstructed_draft,
            state.curated_document,
            state.requirement_report,
            progress.generated_initial_draft,
            progress.generated_final_draft,
            progress.generated_core_draft,
            progress.generated_attribute_draft,
            progress.generated_reconstructed_draft,
            progress.curated_document,
            progress.requirement_report,
        ):
            self.assertIsNone(artifact)
        self.assertEqual(state.curation_ledger, [])

    def test_activity_target_guard_treats_either_relation_family_as_fulfilled(self):
        configured = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        )
        document = {
            "was_generated_by": [
                {"id": "activity:1", "evaluated_entity": [{"id": "sample:1", "title": "Sample"}]},
            ]
        }
        requirement = WorkflowService._runtime_semantic_requirement(
            requirement=configured,
            document=document,
        )
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="partial",
            applicable=True,
            quality=0.5,
            weighted_score=requirement.weight * 0.5,
            target_paths=requirement.target_paths,
        )

        WorkflowService._guard_semantic_requirement_assessment(
            requirement=requirement,
            document=document,
            item=item,
        )

        self.assertEqual(item.status, "fulfilled")
        self.assertEqual(item.quality, 1.0)

    async def test_activity_target_reconstruction_refuses_synthesized_relation(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(
            requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if requirement.requirement_id == "activity_evaluation_target"
        ).model_copy(update={"target_paths": ["/was_generated_by/0/evaluated_activity"]})
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            target_paths=requirement.target_paths,
        )
        diagnosis = SemanticReconstructionDiagnosis(
            defects=[
                SemanticReconstructionDefect(
                    defect_type="bad_provenance",
                    target_path="/was_generated_by/0/evaluated_activity",
                    recommended_action="append",
                    needs_synthesis=True,
                    reason="No evidence directly supports adding an activity.",
                )
            ]
        )

        compiled = await service._compile_semantic_diagnosis_actions(
            data_package_id="pkg",
            document={"was_generated_by": [{"id": "activity:1"}]},
            item=item,
            requirement=requirement,
            diagnosis=diagnosis,
            validation_schema={},
        )

        self.assertEqual(compiled.writes, [])
        rejected_reasons = compiled.rejected_reasons or []
        self.assertIn("refusing synthesized relation", rejected_reasons[0])

    def test_absence_placeholder_is_not_a_method_plan(self):
        self.assertTrue(
            WorkflowService._semantic_absence_placeholder(
                {"title": "No method evidence", "description": "No explicit method or procedure evidence available"}
            )
        )

    def test_delivered_document_revalidation_prevents_stale_fulfilled_target_status(self):
        item = RequirementReportItem(
            requirement_id="activity_evaluation_target",
            label="Activity evaluation target",
            weight=1.25,
            status="fulfilled",
            applicable=True,
            quality=1.0,
            weighted_score=1.25,
            target_paths=["/was_generated_by/0/evaluated_entity", "/was_generated_by/0/evaluated_activity"],
        )
        state = ExtractionRunState(
            requirement_report=build_requirement_report(
                schema_valid=True,
                coverage=compute_coverage_report({}, {}),
                semantic_requirements=[item],
                source_trace=compute_source_trace_report(RoutedEvidenceContext(), []),
                coverage_patches=[],
            )
        )

        WorkflowService._revalidate_requirement_report_against_delivered_document(
            state=state,
            document={"was_generated_by": [{"id": "activity:1"}]},
            validation_schema={},
            schema_valid=True,
        )

        self.assertIsNotNone(state.requirement_report)
        delivered_item = state.requirement_report.semantic_requirements[0]  # type: ignore[union-attr]
        self.assertEqual(delivered_item.status, "missing")
        self.assertEqual(delivered_item.quality, 0.0)


class DescriptionMiningIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def service(self, repo: Mock | None = None) -> WorkflowService:
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
            output_repository=repo or Mock(),
        )
        service._record_llm_call_result = Mock()
        service._record_llm_call_exception = Mock()
        service._update_progress = Mock()
        return service

    def test_parent_attribute_ledger_records_schema_skip_reason(self):
        service = self.service()
        state = ExtractionRunState(chat_model="test-model")
        progress = ExtractionRunProgress()
        document = {"was_generated_by": [{"title": ["Acquisition"]}]}

        updated = service._append_parent_attribute(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=document,
            parent={"path": "/was_generated_by/0", "class": "DataGeneratingActivity"},
            attribute_kind="quantitative",
            intent_index=0,
            instance={
                "title": "Temperature",
                "description": "Temperature: 298 K",
                "value": 298.0,
                "has_quantity_type": "Temperature",
                "unit": "K",
            },
            raw_intent={"title": "Temperature", "value": 298.0, "unit": "K"},
            evidence_ids=["ev:temp"],
            answer="Temperature is a useful attribute.",
            validation_schema={},
            state=state,
            progress=progress,
        )

        self.assertEqual(updated, document)
        self.assertEqual(len(state.parent_attribute_ledger), 1)
        self.assertEqual(state.parent_attribute_ledger[0].status, "skipped_schema_missing")
        self.assertEqual(state.parent_attribute_ledger[0].parent_path, "/was_generated_by/0")
        self.assertEqual(state.parent_attribute_ledger[0].evidence_note_identifiers, ["ev:temp"])

    def test_parent_attribute_ledger_records_successful_append(self):
        service = self.service()
        state = ExtractionRunState(chat_model="test-model")
        progress = ExtractionRunProgress()
        document = {"was_generated_by": [{"title": ["Acquisition"]}]}

        updated = service._append_parent_attribute(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=document,
            parent={"path": "/was_generated_by/0", "class": "DataGeneratingActivity"},
            attribute_kind="quantitative",
            intent_index=0,
            instance={
                "title": "Temperature",
                "description": "Temperature: 298 K",
                "value": 298.0,
                "has_quantity_type": "Temperature",
                "unit": "K",
            },
            raw_intent={"title": "Temperature", "value": 298.0, "unit": "K"},
            evidence_ids=["ev:temp"],
            answer="Temperature is a useful attribute.",
            validation_schema=quantitative_schema("DataGeneratingActivity"),
            state=state,
            progress=progress,
        )

        self.assertEqual(updated["was_generated_by"][0]["has_quantitative_attribute"][0]["title"], "Temperature")
        self.assertNotIn("id", updated["was_generated_by"][0]["has_quantitative_attribute"][0])
        self.assertEqual(len(state.parent_attribute_ledger), 1)
        self.assertEqual(state.parent_attribute_ledger[0].status, "applied")
        self.assertEqual(state.parent_attribute_ledger[0].actual_path, "/was_generated_by/0/has_quantitative_attribute/0")
        self.assertEqual(state.projection_ledger[-1].status, "projected")

    def test_parent_attribute_evidence_is_filtered_to_parent_scope(self):
        service = self.service()
        context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="frequency",
                    category="instrument_signal",
                    role="parameter",
                    claim="Spectrometer frequency was 500 MHz.",
                    evidence_text="SFO1=500 MHz",
                    source_context="Acquisition frequency was 500 MHz in CDCl3 using zg30.",
                ),
                EvidenceCandidate(
                    candidate_id="target",
                    category="activity_signal",
                    role="descriptor",
                    claim="The observed target was the proton nucleus 1H.",
                    evidence_text="NUC1=<1H>",
                    source_context="The activity target was the proton nucleus 1H.",
                ),
                EvidenceCandidate(
                    candidate_id="solvent",
                    category="measurement_condition",
                    role="qualitative_attribute",
                    claim="The solvent was CDCl3.",
                    evidence_text="SOLVENT=CDCl3",
                    source_context="The solvent was CDCl3.",
                ),
            ]
        )

        selected, context_window = service._parent_attribute_evidence_packet(
            parent={
                "path": "/was_generated_by/0/evaluated_entity/0",
                "class": "EvaluatedEntity",
                "value": {"title": "Proton Nucleus (1H)", "description": "The target nucleus being observed."},
            },
            evidence_context=context,
        )

        self.assertEqual([item.candidate_id for item in selected], ["target"])
        self.assertEqual(context_window, [])

    def test_parent_attribute_evidence_allows_parameter_file_settings(self):
        service = self.service()
        context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="sw",
                    category="measurement_condition",
                    role="parameter",
                    claim="The acquisition parameter SW was 19.9946778763279.",
                    evidence_text="##$SW= 19.9946778763279",
                    source_context="Parameter file acqus contains ##$SW= 19.9946778763279.",
                ),
                EvidenceCandidate(
                    candidate_id="solvent",
                    category="measurement_condition",
                    role="qualitative_attribute",
                    claim="The solvent was CDCl3.",
                    evidence_text="SOLVENT=CDCl3",
                    source_context="The solvent was CDCl3.",
                ),
            ]
        )

        selected, _ = service._parent_attribute_evidence_packet(
            parent={
                "path": "/was_generated_by/0/had_input_entity/0",
                "class": "EvaluatedEntity",
                "value": {"title": "Parameter File (r1)", "description": "Parameter file for TOPSPIN Version 3.2."},
            },
            evidence_context=context,
        )

        self.assertEqual([item.candidate_id for item in selected], ["sw"])

    def test_parent_attribute_prompt_guards_attribute_intent(self):
        core_draft = {
            "title": "dataset",
            "was_generated_by": [
                {
                    "title": "Acquisition",
                    "carried_out_by": [
                        {
                            "title": "Example Instrument",
                            "description": "Instrument mentioned by the source.",
                        }
                    ],
                }
            ],
        }
        prompt = ProjectionService._parent_attribute_prompt(
            parent={
                "path": "/was_generated_by/0/carried_out_by/0",
                "class": "AgenticEntity",
                "value": {
                    "title": "Example Instrument",
                    "description": "Instrument mentioned by the source.",
                },
            },
            selected_evidence=[
                RequirementEvidenceItem(
                    evidence_id="ev:origin",
                    candidate_id="origin",
                    category="surrounding_signal",
                    role="context",
                    claim="Origin field names an organization.",
                    evidence_text="ORIGIN=Example Org",
                )
            ],
            context_window=[],
            core_draft=core_draft,
        )

        self.assertIn("recorded characterization of the focused object itself", prompt)
        self.assertIn("core_draft", prompt)
        self.assertIn("target_path", prompt)
        self.assertIn("target_class", prompt)
        self.assertIn("task_instructions", prompt)
        self.assertNotIn("parent_role", prompt)
        self.assertIn("source-record metadata", prompt)
        self.assertIn("creator, owner, origin", prompt)

    def test_parent_attribute_append_rejects_nonlocal_attribute_family(self):
        service = self.service()
        state = ExtractionRunState(chat_model="test-model")
        progress = ExtractionRunProgress()
        document = {"was_generated_by": [{"evaluated_entity": [{"title": "Proton Nucleus (1H)"}]}]}
        evidence = [
            RequirementEvidenceItem(
                evidence_id="ev:freq",
                candidate_id="frequency",
                category="instrument_signal",
                role="parameter",
                claim="Spectrometer frequency was 500 MHz.",
                evidence_text="SFO1=500 MHz",
            )
        ]

        updated = service._append_parent_attribute(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=document,
            parent={
                "path": "/was_generated_by/0/evaluated_entity/0",
                "class": "EvaluatedEntity",
                "value": {"title": "Proton Nucleus (1H)"},
            },
            attribute_kind="quantitative",
            intent_index=0,
            instance={
                "title": "Spectrometer Frequency",
                "description": "Spectrometer frequency: 500 MHz",
                "value": 500.0,
                "has_quantity_type": "Spectrometer Frequency",
                "unit": "MHz",
            },
            raw_intent={"title": "Spectrometer Frequency", "value": 500.0, "unit": "MHz"},
            evidence_ids=["ev:freq"],
            answer="Frequency is present.",
            selected_evidence=evidence,
            validation_schema=quantitative_schema("EvaluatedEntity"),
            state=state,
            progress=progress,
        )

        self.assertEqual(updated, document)
        self.assertEqual(state.parent_attribute_ledger[0].status, "skipped_semantic_placement")
