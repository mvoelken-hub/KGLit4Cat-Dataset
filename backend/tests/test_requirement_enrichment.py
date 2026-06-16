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
