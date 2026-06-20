from __future__ import annotations

import json
import unittest
from pathlib import Path

from unittest.mock import AsyncMock, Mock, patch

from app.core.config import Settings
from app.domain.extraction import (
    EvidenceCandidate,
    EvidenceNoveltyDecision,
    RoutedEvidenceContext,
    apply_evidence_instance,
    builder_output_model_for_target,
    build_context_window_for_note,
    route_evidence_note_to_target,
)


from app.core.config import Settings
from app.domain.extraction.workflow import ExtractionRunProgress, ExtractionRunState
from app.domain.profiles import ProfileValidationResult
from app.services.workflow_service import WorkflowService


class EvidenceEnrichmentRouterTests(unittest.TestCase):
    def _note(self, **kwargs) -> EvidenceCandidate:
        defaults = {
            "candidate_id": "c1",
            "category": "other",
            "claim": "claim",
            "evidence_text": "evidence",
            "file_path": "f.txt",
            "start_idx": 0,
            "end_idx": 10,
        }
        return EvidenceCandidate(**{**defaults, **kwargs})

    def test_device_signal_routes_to_activity_carried_out_by(self):
        note = self._note(
            category="agent_signal",
            claim="spectra acquired with Bruker Avance 500 MHz spectrometer",
            evidence_text="Bruker instrument",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertEqual(path, "/was_generated_by/0/carried_out_by/-")
        self.assertEqual(cls, "AgenticEntity")

    def test_surrounding_signal_routes_to_creator(self):
        note = self._note(
            category="surrounding_signal",
            claim="owner is nmr",
            evidence_text="OWNER= nmr",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertEqual(path, "/creator/-")
        self.assertEqual(cls, "Agent")

    def test_activity_signal_routes_to_was_generated_by(self):
        note = self._note(
            category="activity_signal",
            claim="file contains acquisition activity parameters",
            evidence_text="acquisition",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertEqual(path, "/was_generated_by/-")
        self.assertEqual(cls, "DataGeneratingActivity")

    def test_measurement_signal_has_no_downstream_target(self):
        note = self._note(
            category="measurement_signal",
            claim="field width is 125000",
            evidence_text="FW= 125000",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertIsNone(path)
        self.assertIsNone(cls)

    def test_method_signal_routes_to_realized_plan(self):
        note = self._note(
            category="method_signal",
            claim="NMR pulse sequence zg30",
            evidence_text="zg30",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertEqual(path, "/was_generated_by/0/realized_plan")
        self.assertEqual(cls, "Plan")

    def test_resource_signal_routes_to_distribution(self):
        note = self._note(
            category="resource_signal",
            claim="downloadable parameter file",
            evidence_text="download",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertEqual(path, "/dataset_distribution/-")
        self.assertEqual(cls, "Distribution")

    def test_measurement_signal_is_excluded_from_context_window(self):
        note = self._note(
            category="measurement_signal",
            claim="raw point count is 2559",
            evidence_text="NPOINTS=2559",
        )
        path, cls = route_evidence_note_to_target(note)
        self.assertIsNone(path)
        self.assertIsNone(cls)


class EvidenceEnrichmentContextWindowTests(unittest.TestCase):
    def _note(self, **kwargs) -> EvidenceCandidate:
        defaults = {
            "candidate_id": "c1",
            "category": "instrument_signal",
            "claim": "claim",
            "evidence_text": "evidence",
            "file_path": "f.txt",
            "start_idx": 100,
            "end_idx": 120,
        }
        return EvidenceCandidate(**{**defaults, **kwargs})

    def test_window_includes_overlap_and_nearby_notes(self):
        target = self._note(candidate_id="target", start_idx=1000, end_idx=1020)
        overlap = self._note(candidate_id="overlap", start_idx=1010, end_idx=1030)
        near = self._note(candidate_id="near", start_idx=1400, end_idx=1420)
        far = self._note(candidate_id="far", start_idx=3000, end_idx=3020)
        other_file = self._note(candidate_id="other_file", file_path="other.txt", start_idx=1010, end_idx=1030)
        context = RoutedEvidenceContext(
            portable_evidence=[target, far],
            contextual_evidence=[overlap, near, other_file],
        )
        window = build_context_window_for_note(target, context, window_chars=500)
        ids = {n.candidate_id for n in window}
        self.assertIn("overlap", ids)
        self.assertIn("near", ids)
        self.assertNotIn("target", ids)
        self.assertNotIn("far", ids)
        self.assertNotIn("other_file", ids)
        for note in window:
            self.assertNotIn("raw_chunk", note.model_dump())


class EvidenceEnrichmentApplyTests(unittest.TestCase):
    def test_appends_to_existing_array_and_fills_id(self):
        doc = {
            "id": "pkg",
            "is_about_entity": [
                {"id": "existing", "title": "existing entity"},
            ],
        }
        instance = {"title": "new entity", "description": "fresh"}
        updated = apply_evidence_instance(
            doc,
            "/is_about_entity/-",
            instance,
            data_package_id="pkg",
        )
        self.assertEqual(len(updated["is_about_entity"]), 2)
        self.assertEqual(updated["is_about_entity"][0]["title"], "existing entity")
        self.assertEqual(updated["is_about_entity"][1]["title"], "new entity")
        self.assertTrue(updated["is_about_entity"][1]["id"].startswith("pkg:entity:"))

    def test_creates_array_when_missing(self):
        doc = {"id": "pkg"}
        instance = {"title": "dist"}
        updated = apply_evidence_instance(
            doc,
            "/dataset_distribution/-",
            instance,
            data_package_id="pkg",
        )
        self.assertEqual(len(updated["dataset_distribution"]), 1)
        self.assertTrue(updated["dataset_distribution"][0]["id"].startswith("pkg:distribution:"))

    def test_fills_nested_id(self):
        doc = {"id": "pkg", "is_about_entity": []}
        instance = {"title": "ent", "type": {"title": "Sample"}}
        updated = apply_evidence_instance(
            doc,
            "/is_about_entity/-",
            instance,
            data_package_id="pkg",
        )
        created = updated["is_about_entity"][0]
        self.assertIn("id", created["type"])
        self.assertTrue(created["type"]["id"].startswith("pkg:defined-term:"))

    def test_does_not_add_root_id_when_schema_disallows_it(self):
        doc = {"id": "pkg", "creator": []}
        instance = {"name": ["NMR lab"]}
        updated = apply_evidence_instance(
            doc,
            "/creator/-",
            instance,
            data_package_id="pkg",
            target_schema={
                "properties": {
                    "name": {"type": "array", "items": {"type": "string"}},
                }
            },
        )
        self.assertEqual(updated["creator"][0], {"name": ["NMR lab"]})


class EvidenceEnrichmentBuilderModelTests(unittest.TestCase):
    def test_evaluated_entity_model_accepts_valid_instance(self):
        model = builder_output_model_for_target("EvaluatedEntity", {})
        data = {
            "id": "e1",
            "title": "Sample",
            "description": "A sample",
            "type": {"title": "Sample type"},
            "extra_field": "ignored",
        }
        obj = model.model_validate(data)
        dumped = obj.model_dump(mode="json")
        self.assertEqual(dumped["title"], "Sample")
        self.assertNotIn("extra_field", dumped)

    def test_dynamic_model_for_unknown_target(self):
        schema = {
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "children": {"type": "array"},
            }
        }
        model = builder_output_model_for_target("CustomThing", schema)
        obj = model.model_validate({"id": "x", "children": [{"a": 1}]})
        self.assertEqual(obj.id, "x")

    def test_schema_branch_overrides_builtin_model_shape(self):
        schema = {
            "properties": {
                "name": {"type": "array", "items": {"type": "string"}},
            }
        }
        model = builder_output_model_for_target("Agent", schema)
        obj = model.model_validate({"name": ["NMR lab"], "id": "not-allowed"})
        dumped = obj.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped, {"name": ["NMR lab"]})


class EvidenceNoveltyDecisionTests(unittest.TestCase):
    def test_round_trip(self):
        decision = EvidenceNoveltyDecision(
            is_novel=True,
            reason="New SW parameter.",
            corrected_target_path="/is_about_entity/-",
            corrected_target_class="EvaluatedEntity",
        )
        data = decision.model_dump(mode="json")
        restored = EvidenceNoveltyDecision.model_validate(data)
        self.assertTrue(restored.is_novel)
        self.assertEqual(restored.corrected_target_class, "EvaluatedEntity")

class FakeProfileService:
    def validate_document(self, *, identifier: str, document: dict) -> ProfileValidationResult:
        return ProfileValidationResult(valid=True, errors=[], warnings=[])


def evidence_activity_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$ref": "#/$defs/Dataset",
        "$defs": {
            "Dataset": {
                "type": "object",
                "properties": {
                    "was_generated_by": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/DataGeneratingActivity"},
                    }
                },
            },
            "DataGeneratingActivity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    }


class EvidenceEnrichmentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> WorkflowService:
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
        )
        service.ollama_client = object()
        return service

    async def test_enrichment_appends_novel_instances_and_ledgers_them(self):
        service = self._service()
        service._persist_state_artifacts = Mock()
        service._evaluate_evidence_novelty = AsyncMock(
            return_value=EvidenceNoveltyDecision(is_novel=True, reason="new")
        )
        service._build_evidence_instance = AsyncMock(
            side_effect=lambda **kwargs: {
                "title": kwargs["target_class"],
                "description": "enriched",
            }
        )
        state = ExtractionRunState(
            data_package_id="pkg",
            generated_final_draft={
                "id": "pkg",
                "was_generated_by": [],
            },
        )
        progress = ExtractionRunProgress(stage="profile_projection")
        note = EvidenceCandidate(
            candidate_id="n1",
            category="activity_signal",
            claim="new activity",
            evidence_text="ACTIVITY= x",
            file_path="f.txt",
            start_idx=0,
            end_idx=10,
        )
        context = RoutedEvidenceContext(portable_evidence=[note])
        result = await service._enrich_draft_with_evidence(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=context,
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )
        self.assertEqual(len(result["was_generated_by"]), 1)
        service._persist_state_artifacts.assert_called_once()
        self.assertTrue(
            any(
                record.planner_status == "evidence_enrichment" and record.status == "projected"
                for record in state.projection_ledger
            )
        )
        self.assertEqual(state.generated_initial_draft, {"id": "pkg", "was_generated_by": []})

    async def test_evidence_instance_builder_uses_schema_constrained_output(self):
        service = WorkflowService(
            profile_service=FakeProfileService(),
            settings=Settings(),
            ollama_client=Mock(chat_model="test-model", max_context_length=4096),
        )
        note = EvidenceCandidate(
            candidate_id="n1",
            category="activity_signal",
            claim="new activity",
            evidence_text="ACTIVITY= x",
        )
        llm_result = Mock(
            output={
                "should_apply": True,
                "writes": [
                    {
                        "target_path": "/was_generated_by/-",
                        "mode": "append",
                        "items": [{"title": "Data generating activity"}],
                        "reason": "Add activity.",
                    }
                ],
                "reason": "Add activity.",
            },
            usage=None,
        )

        with patch("app.services.projection_service.generate_structured", AsyncMock(return_value=llm_result)) as mocked:
            instance = await service._build_evidence_instance(
                data_package_id="pkg",
                note=note,
                contextual_notes=[],
                draft_excerpt=[],
                schema_branch={},
                validation_schema=evidence_activity_schema(),
                target_path="/was_generated_by/-",
                target_class="DataGeneratingActivity",
            )

        self.assertEqual(instance, {"title": "Data generating activity"})
        self.assertIsInstance(mocked.call_args.kwargs["output_type"], dict)
        branch = mocked.call_args.kwargs["output_type"]["properties"]["writes"]["items"]["oneOf"][0]
        self.assertEqual(branch["properties"]["target_path"]["const"], "/was_generated_by/-")

    async def test_enrichment_skips_non_novel_notes(self):
        service = self._service()
        service._evaluate_evidence_novelty = AsyncMock(
            return_value=EvidenceNoveltyDecision(is_novel=False, reason="already present")
        )
        state = ExtractionRunState(
            data_package_id="pkg",
            generated_final_draft={"id": "pkg", "was_generated_by": []},
        )
        progress = ExtractionRunProgress(stage="profile_projection")
        note = EvidenceCandidate(
            candidate_id="n1",
            category="activity_signal",
            claim="same activity",
            evidence_text="ACTIVITY= x",
            file_path="f.txt",
            start_idx=0,
            end_idx=10,
        )
        context = RoutedEvidenceContext(portable_evidence=[note])
        result = await service._enrich_draft_with_evidence(
            data_package_id="pkg",
            profile_identifier="dcat-ap-plus",
            evidence_context=context,
            validation_schema={},
            state=state,
            progress=progress,
            warnings=[],
        )
        self.assertEqual(result["was_generated_by"], [])
        self.assertTrue(
            any(
                record.status == "not_projected" and "Not novel" in (record.reason or "")
                for record in state.projection_ledger
            )
        )


