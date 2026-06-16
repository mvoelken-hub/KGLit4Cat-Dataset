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
    RequirementReportItem,
    normalized_requirement_evaluation,
    RoutedEvidenceContext,
    report_items_from_evaluation,
    score_requirement_report,
    select_requirement_evidence_packet,
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


class FakeProfileService:
    def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
        return ProfileValidationResult(valid=True, errors=[])


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
        self.assertEqual(state.requirement_report.requirements[6].patch.status, "applied")
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
