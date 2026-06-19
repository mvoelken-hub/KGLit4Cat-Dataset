from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from app.core.config import Settings
from app.domain.extraction import (
    DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS,
    DcatRequirement,
    EvidenceCandidate,
    RequirementEvaluation,
    RequirementAssessment,
    RequirementPatchAttempt,
    RequirementEvidenceItem,
    RequirementReportItem,
    normalized_requirement_evaluation,
    RoutedEvidenceContext,
    report_items_from_evaluation,
    score_requirement_report,
    select_requirement_evidence_packet,
    stable_evidence_id,
)
from app.domain.extraction.workflow import ExtractionRunProgress, ExtractionRunState
from app.domain.profiles import ProfileValidationResult
from app.services.extraction_service import ExtractionService


class RequirementScoringTests(unittest.TestCase):
    def test_weighted_score_excludes_not_applicable(self):
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
        report = score_requirement_report(items)
        self.assertEqual(report.applicable_weight, 3.0)
        self.assertEqual(report.earned_weight, 2.5)
        self.assertAlmostEqual(report.metadata_completeness_score, 2.5 / 3.0)
        self.assertEqual(report.requirements[2].weighted_score, 0.0)

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

    def test_source_trace_scores_from_fulfilled_requirement_evidence(self):
        trace = RequirementReportItem(
            requirement_id="source_trace",
            label="Trace",
            weight=1.0,
            status="missing",
            applicable=True,
            quality=0.0,
            weighted_score=0.0,
        )
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

        report = score_requirement_report([semantic, trace])

        self.assertEqual(report.requirements[1].status, "fulfilled")
        self.assertEqual(report.requirements[1].quality, 1.0)


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
        self.assertIn("m2", {entry.candidate_id for entry in window})
        self.assertNotIn("x1", {entry.candidate_id for entry in selected})

    def test_quantitative_packet_prefers_observe_frequency(self):
        requirement = next(
            req
            for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS
            if req.requirement_id == "semantic_attributes"
        )
        observe = EvidenceCandidate(
            candidate_id="observe",
            category="measurement_signal",
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


class FakeProfileService:
    def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
        return ProfileValidationResult(valid=True, errors=[])


def quantitative_schema(*owner_classes: str) -> dict:
    defs = {
        "QuantitativeAttribute": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "value": {"type": "number"},
                "has_quantity_type": {"type": "string"},
                "unit": {"type": "string"},
            },
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
    async def test_requirement_enrichment_persists_report_and_initial_draft(self):
        repo = Mock()
        service = ExtractionService(
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
        self.assertEqual(state.requirement_report.requirements[6].patch.status, "applied")
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

    async def test_applied_patch_is_re_evaluated_before_scoring(self):
        repo = Mock()
        service = ExtractionService(
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

        item = next(req for req in state.requirement_report.requirements if req.requirement_id == "method_plan")
        self.assertEqual(item.status, "fulfilled")
        self.assertEqual(item.patch.status, "applied")

    async def test_fulfilled_requirement_with_missing_target_path_is_patched(self):
        repo = Mock()
        service = ExtractionService(
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
        service = ExtractionService(
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
        service = ExtractionService(
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
        service = ExtractionService(
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
                category="measurement_signal",
                claim="Observation frequency is 400.13 MHz.",
                evidence_text="OBSERVE FREQUENCY=400.13 MHz",
                file_path="acqus",
                start_idx=10,
                end_idx=40,
            ),
            EvidenceCandidate(
                candidate_id="scans",
                category="measurement_signal",
                claim="Number of scans is 16.",
                evidence_text="NS=16",
                file_path="acqus",
                start_idx=120,
                end_idx=130,
            ),
        ]

        groups = ExtractionService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1].value, 16.0)
        self.assertIsNone(groups[1].unit)

    def test_quantitative_duplicate_groups_merge_evidence_ids(self):
        notes = [
            EvidenceCandidate(
                candidate_id="a",
                category="measurement_signal",
                claim="Temperature is 298 K.",
                evidence_text="TEMP=298 K",
                file_path="acqus",
                start_idx=10,
                end_idx=20,
            ),
            EvidenceCandidate(
                candidate_id="b",
                category="measurement_signal",
                claim="Temperature is 298 K.",
                evidence_text="TEMP=298 K",
                file_path="acqus",
                start_idx=30,
                end_idx=40,
            ),
        ]

        groups = ExtractionService._quantitative_evidence_groups(notes)

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
                category="data_quality_signal",
                claim="TopSpin 3.5 pl 6 software version.",
                evidence_text="TopSpin 3.5 pl 6",
            ),
            EvidenceCandidate(
                candidate_id="good",
                category="measurement_signal",
                claim="Observation frequency is 400.13 MHz.",
                evidence_text="OBSERVE FREQUENCY=400.13 MHz",
            ),
        ]

        groups = ExtractionService._quantitative_evidence_groups(notes)

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

        groups = ExtractionService._quantitative_evidence_groups(notes)

        self.assertEqual(len(groups), 5)

    def test_quantitative_groups_project_to_existing_owner_before_creating_owner(self):
        service = ExtractionService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "semantic_attributes")
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
            requirement_id="semantic_attributes",
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
                    category="measurement_signal",
                    claim="Observation frequency is 400.13 MHz.",
                    evidence_text="OBSERVE FREQUENCY=400.13 MHz",
                    file_path="acqus",
                    start_idx=10,
                    end_idx=40,
                ),
                EvidenceCandidate(
                    candidate_id="scans",
                    category="measurement_signal",
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
        service = ExtractionService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "semantic_attributes")
        state = ExtractionRunState(
            generated_final_draft={"id": "pkg", "title": ["Dataset"], "description": ["Desc"]},
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="semantic_attributes",
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
                    category="measurement_signal",
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
        service = ExtractionService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        requirement = next(req for req in DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS if req.requirement_id == "semantic_attributes")
        state = ExtractionRunState(
            generated_final_draft={"id": "pkg", "title": ["Dataset"], "description": ["Desc"]},
            chat_model="test-model",
        )
        progress = ExtractionRunProgress(warnings=[])
        item = RequirementReportItem(
            requirement_id="semantic_attributes",
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
        self.assertEqual(item.status, "fulfilled")
        self.assertEqual(state.projection_ledger[0].merge_status, "skipped")
        self.assertIn("target_unresolved", state.projection_ledger[0].reason)

    def test_quantitative_patch_fallback_keeps_first_generic_numeric_evidence(self):
        service = ExtractionService(
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
        service = ExtractionService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        groups = service._quantitative_evidence_groups(
            [
                EvidenceCandidate(
                    candidate_id="frequency",
                    category="measurement_signal",
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
        service = ExtractionService(
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
        service = ExtractionService(
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
