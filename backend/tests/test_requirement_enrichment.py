from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from jsonschema import Draft201909Validator
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.extraction import (
    DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS,
    DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS,
    DcatRequirement,
    EvidenceCandidate,
    RequirementEvaluation,
    RequirementAssessment,
    RequirementPatchAttempt,
    RequirementEvidenceItem,
    RequirementReportItem,
    SemanticReconstructionRecord,
    build_schema_constrained_patch_schema,
    build_requirement_evaluation_prompt,
    build_requirement_report,
    build_semantic_reconstruction_prompt,
    compute_coverage_report,
    compute_source_trace_report,
    normalized_requirement_evaluation,
    RoutedEvidenceContext,
    report_items_from_evaluation,
    score_requirement_items,
    score_requirement_report,
    select_requirement_evidence_packet,
    stable_evidence_id,
)
from app.domain.extraction.description_mining import RawDescriptionFacts
from app.domain.extraction.workflow import ExtractionRunProgress, ExtractionRunState
from app.domain.profiles import ProfileValidationIssue, ProfileValidationResult
from app.ollama.errors import CompletionError
from app.services.workflow_service import WorkflowService


class RequirementScoringTests(unittest.TestCase):
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
            claim="Plan.",
            evidence_text="PULPROG=zg30",
            evidence_match_score=0.8,
        )
        unused = EvidenceCandidate(
            candidate_id="m2",
            category="method_signal",
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

    def test_missing_evaluator_requirement_becomes_missing_report_item(self):
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
        self.assertEqual(items[0].status, "missing")
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
        self.assertEqual(evaluation.assessments[1].status, "missing")

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
        self.assertEqual(write_schema["properties"]["target_path"]["const"], "/was_generated_by/0/has_quantitative_attribute")
        self.assertEqual(write_schema["properties"]["mode"]["const"], "append")

    def test_schema_constrained_patch_schema_rejects_invalid_quantitative_items(self):
        output_schema = build_schema_constrained_patch_schema(
            validation_schema=quantitative_schema("DataGeneratingActivity"),
            allowed_target_paths=["/was_generated_by/0/has_quantitative_attribute"],
        )
        validator = Draft201909Validator(output_schema)
        valid = {
            "should_apply": True,
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
        self.assertNotEqual(list(validator.iter_errors(extra_field)), [])
        self.assertNotEqual(list(validator.iter_errors(range_object)), [])


class RequirementEvidencePacketTests(unittest.TestCase):
    def test_stable_evidence_id_disambiguates_repeated_candidate_ids(self):
        left = EvidenceCandidate(
            candidate_id="candidate-0",
            category="measurement_signal",
            claim="Observe frequency is 500 MHz.",
            evidence_text="##.OBSERVE FREQUENCY=500.133088507478",
            file_path="10.edit.jdx",
            start_idx=0,
            end_idx=58,
        )
        right = EvidenceCandidate(
            candidate_id="candidate-0",
            category="resource_signal",
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
            claim="Pulse sequence is zg30.",
            evidence_text="PULPROG= zg30",
            file_path="acqus",
            start_idx=100,
            end_idx=120,
        )
        nearby = EvidenceCandidate(
            candidate_id="m2",
            category="measurement_signal",
            claim="Acquisition temperature is 298 K.",
            evidence_text="TE= 298",
            file_path="acqus",
            start_idx=160,
            end_idx=180,
        )
        unrelated = EvidenceCandidate(
            candidate_id="x1",
            category="agent_signal",
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
        self.assertNotIn("m2", {entry.candidate_id for entry in window})
        self.assertNotIn("x1", {entry.candidate_id for entry in selected})

    def test_quantitative_packet_prefers_observe_frequency(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
            if req.requirement_id == "instrument_settings_attributes"
        )
        observe = EvidenceCandidate(
            candidate_id="observe",
            category="instrument_signal",
            claim="The NMR spectrum was recorded at an observe frequency of 500.133088507478 MHz.",
            evidence_text="##.OBSERVE FREQUENCY=500.133088507478",
            file_path="10.edit.jdx",
            start_idx=0,
            end_idx=58,
        )
        max_y = EvidenceCandidate(
            candidate_id="max_y",
            category="measurement_signal",
            claim="The maximum y-value in the NMR peak table is 6786105183.528301 arbitrary units.",
            evidence_text="##MAXY=6786105183.528301",
            file_path="10.edit.jdx",
            start_idx=952,
            end_idx=1010,
        )
        context = RoutedEvidenceContext(portable_evidence=[max_y, observe])
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

        self.assertEqual(selected[0].candidate_id, "observe")

    def test_quantitative_packet_accepts_measurement_condition_but_not_raw_measurement(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
            if req.requirement_id == "instrument_settings_attributes"
        )
        points = EvidenceCandidate(
            candidate_id="points",
            category="measurement_condition",
            claim="The spectrum contains 2559 data points.",
            evidence_text="NPOINTS=2559",
        )
        raw = EvidenceCandidate(
            candidate_id="raw",
            category="measurement_signal",
            claim="A row-like transmittance value is 0.42.",
            evidence_text="1234.5 0.42",
        )
        context = RoutedEvidenceContext(portable_evidence=[raw, points])
        item = RequirementReportItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            evidence_search_hints=["points"],
        )

        selected, _ = select_requirement_evidence_packet(
            requirement=requirement,
            assessment=item,
            evidence_context=context,
        )

        self.assertEqual([entry.candidate_id for entry in selected], ["points"])

    def test_dataset_generation_semantics_accepts_supporting_activity_evidence(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "dataset_generation_semantics"
        )
        method = EvidenceCandidate(
            candidate_id="method",
            category="method_signal",
            claim="The data was acquired using Diamant ATR sampling procedure.",
            evidence_text="##SAMPLING PROCEDURE=Diamant ATR",
        )
        agent = EvidenceCandidate(
            candidate_id="agent",
            category="agent_signal",
            claim="The instrument used is Bruker ALPHA.",
            evidence_text="instrument: Bruker ALPHA",
        )
        instrument = EvidenceCandidate(
            candidate_id="setting",
            category="instrument_signal",
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

    def test_instrument_settings_prompt_requires_selected_settings(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "instrument_settings_semantics"
        )

        prompt = build_requirement_evaluation_prompt(
            document={"/was_generated_by/0/has_quantitative_attribute": []},
            requirements=[requirement],
            selected_evidence=[
                RequirementEvidenceItem(
                    evidence_id="ev:unit",
                    candidate_id="unit",
                    category="instrument_signal",
                    claim="X-axis units are 1/CM.",
                    evidence_text="##XUNITS=1/CM",
                )
            ],
            context_window=[],
        )

        self.assertIn(
            "selected as concrete evidence are represented as suitable attributes",
            prompt,
        )

    def test_instrument_reconstruction_prompt_preserves_specific_quantity(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
            if req.requirement_id == "instrument_settings_semantics"
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

        prompt = build_semantic_reconstruction_prompt(
            requirement=requirement,
            item=item,
            document={},
            draft_excerpt={},
            schema_branches={},
        )

        self.assertIn("add separate schema-valid attributes or set should_apply=false", prompt)
        self.assertIn("Represent numeric ranges as separate schema-valid minimum and maximum", prompt)


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
        async def passthrough_mining(*args, **kwargs):
            return kwargs["evidence_context"]

        self.description_mining_patcher = patch.object(
            WorkflowService,
            "_mine_dataset_description",
            side_effect=passthrough_mining,
        )
        self.description_mining_patcher.start()
        self.addAsyncCleanup(self.description_mining_patcher.stop)

    async def test_requirement_enrichment_persists_report_and_initial_draft(self):
        repo = Mock()
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
            output_repository=repo,
        )
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

    def test_quantitative_evidence_groups_multiple_numeric_labels(self):
        notes = [
            EvidenceCandidate(
                candidate_id="frequency",
                category="instrument_signal",
                claim="Observation frequency is 400.13 MHz.",
                evidence_text="OBSERVE FREQUENCY=400.13 MHz",
                file_path="acqus",
                start_idx=10,
                end_idx=40,
            ),
            EvidenceCandidate(
                candidate_id="scans",
                category="instrument_signal",
                claim="Number of scans is 16.",
                evidence_text="NS=16",
                file_path="acqus",
                start_idx=120,
                end_idx=130,
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1].value, 16.0)
        self.assertIsNone(groups[1].unit)

    def test_quantitative_duplicate_groups_merge_evidence_ids(self):
        notes = [
            EvidenceCandidate(
                candidate_id="a",
                category="instrument_signal",
                claim="Temperature is 298 K.",
                evidence_text="TEMP=298 K",
                file_path="acqus",
                start_idx=10,
                end_idx=20,
            ),
            EvidenceCandidate(
                candidate_id="b",
                category="instrument_signal",
                claim="Temperature is 298 K.",
                evidence_text="TEMP=298 K",
                file_path="acqus",
                start_idx=30,
                end_idx=40,
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].notes), 2)

    def test_quantitative_grouping_rejects_identifier_and_path_numbers(self):
        notes = [
            EvidenceCandidate(
                candidate_id="title",
                category="resource_signal",
                claim="Dataset name is 1H_NMR_clean.",
                evidence_text="##TITLE=1H_NMR_clean /opt/topspin3.5pl6/data",
            ),
            EvidenceCandidate(
                candidate_id="program",
                category="method_signal",
                claim="Pulse sequence used was zg30.",
                evidence_text="/opt/topspin3.5pl6/exp/stan/nmr/lists/pp/zg30",
            ),
            EvidenceCandidate(
                candidate_id="software",
                category="agent_signal",
                claim="TopSpin 3.5 pl 6 software version.",
                evidence_text="TopSpin 3.5 pl 6",
            ),
            EvidenceCandidate(
                candidate_id="good",
                category="instrument_signal",
                claim="Observation frequency is 400.13 MHz.",
                evidence_text="OBSERVE FREQUENCY=400.13 MHz",
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].value, 400.13)

    def test_quantitative_grouping_caps_repeated_row_like_values(self):
        notes = [
            EvidenceCandidate(
                candidate_id=f"peak-{index}",
                category="measurement_signal",
                claim=f"Peak observed spectrum at {7.0 + index / 1000} ppm.",
                evidence_text=f"{7.0 + index / 1000} ppm",
                file_path="peaks.txt",
                start_idx=index * 10,
                end_idx=index * 10 + 5,
            )
            for index in range(12)
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 0)

    def test_quantitative_grouping_rejects_primary_data_summaries_without_domain_keys(self):
        notes = [
            EvidenceCandidate(
                candidate_id="identifier",
                category="instrument_signal",
                claim="The run identifier is A-42.",
                evidence_text="RUN=A-42",
            ),
            EvidenceCandidate(
                candidate_id="maximum",
                category="instrument_signal",
                claim="The maximum observed signal value is 3997.453.",
                evidence_text="UPPER_BOUND=3997.453",
            ),
            EvidenceCandidate(
                candidate_id="rows",
                category="instrument_signal",
                claim="The number of data points in the result table is 23.",
                evidence_text="ROWS=23",
            ),
            EvidenceCandidate(
                candidate_id="threshold",
                category="instrument_signal",
                claim="The threshold for peak detection is set to 0.93.",
                evidence_text="THRESHOLD=0.93",
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual([(group.label, group.value) for group in groups], [("threshold for peak detection", 0.93)])

    def test_quantitative_grouping_accepts_measurement_condition_descriptors(self):
        notes = [
            EvidenceCandidate(
                candidate_id="points",
                category="measurement_condition",
                claim="The spectrum contains a data point count of 2559.",
                evidence_text="NPOINTS=2559",
            ),
            EvidenceCandidate(
                candidate_id="raw",
                category="measurement_signal",
                claim="A raw observed transmittance value is 0.42.",
                evidence_text="1234.5 0.42",
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].value, 2559)

    def test_quantitative_grouping_rejects_qualitative_placeholder_values(self):
        notes = [
            EvidenceCandidate(
                candidate_id="name",
                category="instrument_signal",
                claim="The solvent name is unspecified.",
                evidence_text="MATERIAL_NAME=- - -",
            ),
            EvidenceCandidate(
                candidate_id="value",
                category="instrument_signal",
                claim="The material value is set to 0.",
                evidence_text="MATERIAL_VALUE=0",
            ),
            EvidenceCandidate(
                candidate_id="position",
                category="instrument_signal",
                claim="The material x-position is set to 0.",
                evidence_text="MATERIAL_X=0",
            ),
        ]

        groups = WorkflowService._quantitative_evidence_groups(notes)

        self.assertEqual(groups, [])

    def test_quantitative_attribute_description_is_plain(self):
        note = EvidenceCandidate(
            candidate_id="threshold",
            category="instrument_signal",
            claim="The threshold is set to 0.93.",
            evidence_text="THRESHOLD=0.93",
        )
        group = WorkflowService._quantitative_evidence_groups([note])[0]

        instance = WorkflowService._quantitative_attribute_instance_from_group(group)

        self.assertEqual(instance["description"], "Threshold: 0.93")
        self.assertNotIn("Evidence-grounded quantitative attribute", instance["description"])

    async def test_semantic_reconstruction_applies_schema_constrained_write(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
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
        llm_result = Mock(
            output={
                "should_apply": True,
                "writes": [
                    {
                        "target_path": "/description",
                        "mode": "replace",
                        "value": ["Dataset acquired with a documented method."],
                        "reason": "Streamlined dataset identity.",
                    }
                ],
                "reason": "Streamlined dataset identity.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            updated, paths, reason, errors = await service._semantic_reconstruction_update(
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
                    requirement_id="dataset_identity_semantics",
                    label="Identity",
                    description="Identity semantics",
                    target_paths=["/description"],
                ),
                validation_schema=quantitative_schema("DataGeneratingActivity"),
            )

        self.assertEqual(paths, ["/description"])
        self.assertEqual(errors, [])
        self.assertEqual(reason, "Streamlined dataset identity.")
        self.assertEqual(updated["description"], ["Dataset acquired with a documented method."])
        call_kwargs = mocked.call_args.kwargs
        self.assertEqual(call_kwargs["agent_name"], "semantic_reconstruction")
        self.assertIsInstance(call_kwargs["output_type"], dict)
        self.assertIn("allowed_target_paths", call_kwargs["prompt"])

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
                "should_apply": True,
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

    async def test_semantic_reconstruction_schema_only_allows_target_paths(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
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
        llm_result = Mock(
            output={"should_apply": False, "writes": [], "reason": "No safe write."},
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            updated, paths, reason, errors = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={"description": ["Original."], "dataset_distribution": [{"title": ["D"]}]},
                item=item,
                requirement=DcatRequirement(
                    requirement_id="dataset_identity_semantics",
                    label="Identity",
                    description="Identity semantics",
                    target_paths=["/description"],
                ),
                validation_schema=quantitative_schema("DataGeneratingActivity"),
            )

        self.assertEqual(updated["description"], ["Original."])
        self.assertEqual(paths, [])
        self.assertEqual(reason, "No safe write.")
        self.assertEqual(errors, [])
        output_schema = mocked.call_args.kwargs["output_type"]
        branch = output_schema["properties"]["writes"]["items"]["oneOf"][0]
        self.assertEqual(branch["properties"]["target_path"]["const"], "/description")

    async def test_semantic_reconstruction_salvages_valid_writes_when_one_fails_validation(self):
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
                "should_apply": True,
                "writes": [
                    {
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "mode": "append",
                        "items": [
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
                        "reason": "Attach instrument evidence.",
                    }
                ],
                "reason": "Attach instrument evidence.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)):
            updated, paths, reason, errors = await service._semantic_reconstruction_update(
                data_package_id="pkg",
                profile_identifier="profile",
                document={
                    "was_generated_by": [
                        {
                            "has_quantitative_attribute": []
                        }
                    ]
                },
                item=item,
                requirement=DcatRequirement(
                    requirement_id="instrument_settings_semantics",
                    label="Instrument settings",
                    description="Instrument settings semantics",
                    target_paths=["/was_generated_by/0/has_quantitative_attribute"],
                ),
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

    def test_quantitative_groups_project_to_existing_owner_before_creating_owner(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "instrument_settings_attributes")
        state = ExtractionRunState(
            generated_final_draft={
                "id": "pkg",
                "title": ["Dataset"],
                "description": ["Desc"],
                "was_generated_by": [{"id": "act", "title": "Acquisition"}],
            },
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="instrument_settings_attributes",
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
            target_paths=requirement.target_paths,
            evidence_search_hints=requirement.evidence_hints,
        )
        context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="freq",
                    category="instrument_signal",
                    claim="Observation frequency is 400.13 MHz.",
                    evidence_text="OBSERVE FREQUENCY=400.13 MHz",
                    file_path="acqus",
                    start_idx=10,
                    end_idx=40,
                ),
                EvidenceCandidate(
                    candidate_id="scans",
                    category="instrument_signal",
                    claim="Number of scans is 16.",
                    evidence_text="NS=16",
                    file_path="acqus",
                    start_idx=120,
                    end_idx=130,
                ),
            ]
        )

        document = service._apply_quantitative_evidence_groups(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=state.generated_final_draft,
            evidence_context=context,
            validation_schema=quantitative_schema("DataGeneratingActivity"),
            state=state,
            progress=progress,
            requirement=requirement,
            item=item,
        )

        self.assertEqual(len(document["was_generated_by"]), 1)
        self.assertEqual(len(document["was_generated_by"][0]["has_quantitative_attribute"]), 2)
        self.assertEqual(len(state.field_completion_ledger), 2)
        self.assertTrue(all(record.source_evidence for record in state.field_completion_ledger))
        self.assertEqual(item.status, "fulfilled")

    def test_quantitative_group_creates_reachable_device_owner(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "instrument_settings_attributes")
        state = ExtractionRunState(
            generated_final_draft={"id": "pkg", "title": ["Dataset"], "description": ["Desc"]},
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="instrument_settings_attributes",
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
        )
        context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="device-temp",
                    category="instrument_signal",
                    claim="Device temperature is 298 K.",
                    evidence_text="temperature 298 K",
                    file_path="run.txt",
                    start_idx=0,
                    end_idx=20,
                )
            ]
        )

        document = service._apply_quantitative_evidence_groups(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=state.generated_final_draft,
            evidence_context=context,
            validation_schema=quantitative_schema("DataGeneratingActivity", "AgenticEntity", "Device"),
            state=state,
            progress=progress,
            requirement=requirement,
            item=item,
        )

        owner = document["was_generated_by"][0]["carried_out_by"][0]
        self.assertEqual(owner["rdf_type"]["title"], "Device")
        self.assertEqual(owner["has_quantitative_attribute"][0]["value"], 298.0)

    def test_quantitative_unreachable_owner_group_is_skipped(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "instrument_settings_attributes")
        state = ExtractionRunState(
            generated_final_draft={"id": "pkg", "title": ["Dataset"], "description": ["Desc"]},
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="instrument_settings_attributes",
            label=requirement.label,
            weight=requirement.weight,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
        )
        context = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="value",
                    category="measurement_signal",
                    claim="Measured value is 12.",
                    evidence_text="value 12",
                    file_path="run.txt",
                    start_idx=0,
                    end_idx=20,
                )
            ]
        )

        document = service._apply_quantitative_evidence_groups(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            document=state.generated_final_draft,
            evidence_context=context,
            validation_schema=quantitative_schema(),
            state=state,
            progress=progress,
            requirement=requirement,
            item=item,
        )

        self.assertNotIn("was_generated_by", document)
        self.assertEqual(item.status, "missing")
        self.assertFalse(state.projection_ledger)

    def test_quantitative_patch_fallback_keeps_first_generic_numeric_evidence(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )

        sanitized = service._sanitize_requirement_patch_instance(
            instance={"value": "125000", "has_quantity_type": "width"},
            target_class="QuantitativeAttribute",
            target_path="/was_generated_by/0/has_quantitative_attribute/-",
            item=Mock(
                selected_evidence=[
                    RequirementEvidenceItem(
                        evidence_id="ev:width",
                        candidate_id="candidate-1",
                        category="measurement_signal",
                        claim="Field width is 125000",
                        evidence_text="FW= 125000",
                    ),
                    RequirementEvidenceItem(
                        evidence_id="ev:frequency",
                        candidate_id="candidate-0",
                        category="measurement_signal",
                        claim="The NMR spectrum was recorded at an observe frequency of 500.133088507478 MHz.",
                        evidence_text="##.OBSERVE FREQUENCY=500.133088507478",
                    ),
                ]
            ),
        )

        self.assertEqual(sanitized["value"], 125000.0)
        self.assertIn("Field width", sanitized["has_quantity_type"])

    def test_single_frequency_quantitative_group_still_projects(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        groups = service._quantitative_evidence_groups(
            [
                EvidenceCandidate(
                    candidate_id="frequency",
                    category="instrument_signal",
                    claim="Observation frequency is 500.133088507478 MHz.",
                    evidence_text="##.OBSERVE FREQUENCY=500.133088507478 MHz",
                    file_path="acqus",
                    start_idx=10,
                    end_idx=40,
                )
            ]
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].value, 500.133088507478)
        self.assertEqual(groups[0].unit, "MHz")

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

    def test_file_like_about_entities_are_removed(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        document = {
            "is_about_entity": [
                {
                    "id": "pkg:entity:acqus",
                    "title": "acqus file",
                    "description": "NMR acquisition parameters file",
                },
                {
                    "id": "pkg:entity:sample",
                    "title": "CDCl3 solvent",
                    "description": "Solvent used in NMR experiment",
                },
            ]
        }

        cleaned = service._remove_file_like_about_entities(document)

        self.assertEqual(len(cleaned["is_about_entity"]), 1)
        self.assertEqual(cleaned["is_about_entity"][0]["title"], "CDCl3 solvent")


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

    async def test_miner_augments_portable_context_and_persists_validated_facts(self):
        repo = Mock()
        service = self.service(repo)
        result = Mock(
            output=RawDescriptionFacts.model_validate(
                {
                    "facts": [
                        {
                            "source_description_path": "/description/0",
                            "category": "instrument_signal",
                            "claim": "Acquisition frequency was 500 MHz.",
                            "evidence_text": "frequency was 500 MHz",
                        }
                    ]
                }
            )
        )
        original = RoutedEvidenceContext()
        state = ExtractionRunState(chat_model="test-model")
        progress = ExtractionRunProgress()

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(return_value=result),
        ):
            augmented = await service._mine_dataset_description(
                data_package_id="pkg",
                document={"description": ["Acquisition frequency was 500 MHz."]},
                evidence_context=original,
                state=state,
                progress=progress,
                warnings=[],
            )

        self.assertEqual(original.portable_evidence, [])
        self.assertEqual(len(augmented.portable_evidence), 1)
        self.assertEqual(augmented.portable_evidence[0].category, "instrument_signal")
        groups = service._quantitative_evidence_groups(augmented.portable_evidence)
        self.assertEqual(groups[0].value, 500.0)
        artifact = repo.save_description_facts.call_args.kwargs["artifact"]
        self.assertEqual(artifact.status, "completed")
        self.assertEqual(len(artifact.facts), 1)

    async def test_miner_failure_writes_artifact_warns_and_continues(self):
        repo = Mock()
        service = self.service(repo)
        original = RoutedEvidenceContext()
        warnings: list[str] = []

        with patch(
            "app.services.projection_service.generate_structured",
            AsyncMock(side_effect=CompletionError("bad structured output")),
        ):
            result = await service._mine_dataset_description(
                data_package_id="pkg",
                document={"description": ["Dataset description."]},
                evidence_context=original,
                state=ExtractionRunState(chat_model="test-model"),
                progress=ExtractionRunProgress(),
                warnings=warnings,
            )

        self.assertIs(result, original)
        self.assertIn("continuing with source evidence only", warnings[0])
        artifact = repo.save_description_facts.call_args.kwargs["artifact"]
        self.assertEqual(artifact.status, "failed")

    async def test_missing_description_writes_skipped_artifact(self):
        repo = Mock()
        service = self.service(repo)
        original = RoutedEvidenceContext()

        result = await service._mine_dataset_description(
            data_package_id="pkg",
            document={"title": ["Dataset"]},
            evidence_context=original,
            state=ExtractionRunState(chat_model="test-model"),
            progress=ExtractionRunProgress(),
            warnings=[],
        )

        self.assertIs(result, original)
        artifact = repo.save_description_facts.call_args.kwargs["artifact"]
        self.assertEqual(artifact.status, "skipped")

    async def test_mining_precedes_coverage_and_semantics_keep_original_context(self):
        service = self.service()
        original = RoutedEvidenceContext()
        augmented = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    candidate_id="description:1",
                    category="other",
                    claim="Description fact.",
                    evidence_text="Description fact.",
                    file_path="draft-description:/description/0",
                )
            ]
        )
        events: list[str] = []

        async def mine(**kwargs):
            events.append("mine")
            return augmented

        async def evaluate(**kwargs):
            events.append("semantic")
            self.assertIs(kwargs["evidence_context"], original)
            return []

        async def reconstruct(**kwargs):
            return kwargs["document"], []

        service._mine_dataset_description = AsyncMock(side_effect=mine)
        service._evaluate_semantic_requirements = AsyncMock(side_effect=evaluate)
        service._reconstruct_semantic_defects = AsyncMock(side_effect=reconstruct)
        initial = {"id": "pkg", "title": ["Dataset"], "description": ["Description fact."]}
        state = ExtractionRunState(generated_final_draft=initial, chat_model="test-model")

        await service._enrich_draft_with_requirements(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=original,
            validation_schema={},
            state=state,
            progress=ExtractionRunProgress(),
            warnings=[],
        )

        self.assertEqual(events, ["mine", "semantic", "semantic"])
        self.assertEqual(state.generated_initial_draft["description"], ["Description fact."])
        self.assertEqual(state.generated_patched_draft["description"], ["Description fact."])

