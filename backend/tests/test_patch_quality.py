import copy
import json
import unittest

from pydantic_ai.models.test import TestModel

from app.domain.datasources import DataPackage, FileEntry
from app.domain.datasources.chunking import ContentChunk
from app.domain.extraction import (
    DEFAULT_OUTPUT_RETRIES,
    InitialContext,
    PatchCandidate,
    PatchRecord,
    SchemaPatchResult,
    apply_merge_patch,
    patch_draft_from_content_chunks,
    validate_candidate_draft,
)
from app.domain.extraction.patch_quality import (
    CandidateQualityRating,
    PatchQualityDecision,
    PatchQualityIssue,
    PatchQualityReport,
    UnmappedFact,
    review_patch_semantic_quality,
)
from app.domain.profiles import ProfileManifest


PROFILE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "$defs": {
        "Dataset": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "was_generated_by": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "has_qualitative_attribute": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "value": {"type": "string"},
                                    },
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        }
    },
}


INITIAL_CONTEXT_OUTPUT = {
    "device_name": "Mass spectrometer",
    "device_model": "MS-1000",
    "entities_analyzed": ["sample-1"],
    "analytical_technique": "mass spectrometry",
    "file_relationships": [],
    "metadata_sources": [],
    "keywords": ["mass spectrometry", "sample-1"],
    "summary": "The package contains mass spectrometry data for sample-1.",
}


def make_profile_manifest() -> ProfileManifest:
    return ProfileManifest(
        identifier="test-profile",
        source="profile.yaml",
        source_type="upload",
        schema_file_name="profile.yaml",
        target_class="Dataset",
        checksum="sha256:test",
    )


def make_content_chunks() -> list[list[ContentChunk]]:
    return [
        [
            ContentChunk(
                content="Technique: GC-MS\nSample: sample-1",
                data_package_id="package-id",
                file_path="README.txt",
                start_idx=0,
                end_idx=1,
            )
        ]
    ]


def make_initial_context() -> InitialContext:
    return InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT)


# Field-level patch candidate output from the extraction agent.
FIELD_PATCH_OUTPUT = {
    "candidates": [
        {
            "field_path": "description",
            "patch": {"description": "Updated with chunk evidence."},
            "confidence": 0.9,
            "reasoning": "Chunk contains technique description.",
            "source_evidence": ["Technique: GC-MS"],
        },
        {
            "field_path": "keywords",
            "patch": {"keywords": ["chunk-keyword"]},
            "confidence": 0.8,
            "reasoning": "Chunk mentions relevant keyword.",
            "source_evidence": ["GC-MS"],
        },
    ]
}

LEAN_DESCRIPTION_PATCH_OUTPUT = {
    "information": "Technique information should update the description.",
    "evidence": ["Technique: GC-MS"],
    "location_picks": [
        {
            "path": "/description",
            "rationale": "The fact belongs in the dataset summary.",
            "confidence": 0.9,
        }
    ],
    "destination": "/description",
    "patch": {"description": "Updated with chunk evidence."},
    "reasoning": "Description is the best single destination.",
}

LEAN_ACTIVITY_PATCH_OUTPUT = {
    "information": "The pulse program is zgpg30.",
    "evidence": ["pulse sequence zg30"],
    "location_picks": [
        {
            "path": "/was_generated_by/0/has_qualitative_attribute",
            "rationale": "The fact describes the generating activity.",
            "confidence": 0.7,
        }
    ],
    "destination": "/was_generated_by/0/has_qualitative_attribute",
    "patch": {
        "was_generated_by": [
            {
                "id": "activity-1",
                "has_qualitative_attribute": [
                    {"title": "pulse program", "value": "zgpg30"}
                ],
            }
        ]
    },
    "reasoning": "Structured activity attribute is the best destination.",
}

NO_INFORMATION_OUTPUT = {
    "information": None,
    "evidence": [],
    "location_picks": [],
    "destination": "/description",
    "patch": {},
    "reasoning": "No useful information.",
}

INVALID_SCHEMA_PATCH_OUTPUT = {
    "information": "The chunk mentions an unsupported field.",
    "evidence": ["some text"],
    "location_picks": [
        {
            "path": "/description",
            "rationale": "The closest valid draft location is description.",
            "confidence": 0.5,
        }
    ],
    "destination": "/description",
    "patch": {"unknown_field": "not in schema"},
    "reasoning": "This intentionally violates the schema.",
}


class ValidateCandidateDraftTests(unittest.TestCase):
    def test_validate_candidate_draft_returns_empty_list_for_valid_document(self):
        candidate = {"title": "A valid dataset"}
        manifest = make_profile_manifest()
        errors = validate_candidate_draft(
            candidate=candidate,
            profile_json_schema=PROFILE_JSON_SCHEMA,
            profile_manifest=manifest,
        )
        self.assertEqual(errors, [])

    def test_validate_candidate_draft_returns_errors_for_invalid_document(self):
        candidate = {"description": "Missing required title"}
        manifest = make_profile_manifest()
        errors = validate_candidate_draft(
            candidate=candidate,
            profile_json_schema=PROFILE_JSON_SCHEMA,
            profile_manifest=manifest,
        )
        self.assertTrue(len(errors) > 0)
        self.assertIn("'title' is a required property", errors[0])


class PatchQualityReportModelTests(unittest.TestCase):
    def test_accept_decision_with_no_issues(self):
        report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                    summary="Patch is appropriate.",
                )
            ],
            summary="All candidates accepted.",
        )
        self.assertEqual(report.overall_decision, "accept")
        self.assertEqual(report.decision, "accept")
        self.assertEqual(len(report.candidate_ratings), 1)
        self.assertEqual(len(report.issues), 0)

    def test_revise_decision_with_revised_patch(self):
        report = PatchQualityReport(
            overall_decision="revise",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="was_generated_by",
                    decision="revise",
                    issues=[
                        PatchQualityIssue(
                            path="$.was_generated_by[0].description",
                            issue_type="description_abuse",
                            explanation="Description is used as a fallback container.",
                        )
                    ],
                    revised_patch={
                        "was_generated_by": [
                            {
                                "id": "activity-1",
                                "has_qualitative_attribute": [
                                    {"title": "pulse program", "value": "zgpg30"}
                                ],
                            }
                        ]
                    },
                )
            ],
            summary="Moved structured fact out of description.",
        )
        self.assertEqual(report.overall_decision, "revise")
        self.assertEqual(report.decision, "revise")
        self.assertEqual(len(report.issues), 1)

    def test_reject_decision_with_unmapped_facts(self):
        report = PatchQualityReport(
            overall_decision="reject",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="reject",
                    issues=[
                        PatchQualityIssue(
                            path="$",
                            issue_type="overly_granular",
                            severity="major",
                            explanation="Patch attempts to add raw low-level log information.",
                        )
                    ],
                    unmapped_facts=[
                        UnmappedFact(
                            fact="Some source-supported low-level detail",
                            reason="No suitable schema field.",
                        )
                    ],
                )
            ],
            summary="Patch rejected but facts stored separately.",
        )
        self.assertEqual(report.overall_decision, "reject")
        self.assertEqual(report.decision, "reject")
        self.assertEqual(len(report.unmapped_facts), 1)

    def test_legacy_properties_aggregate_from_candidates(self):
        """Test that .issues and .unmapped_facts aggregate across candidates."""
        report = PatchQualityReport(
            overall_decision="revise",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="was_generated_by",
                    decision="revise",
                    issues=[
                        PatchQualityIssue(
                            path="$.was_generated_by[0]",
                            issue_type="description_abuse",
                            explanation="Description abuse.",
                        )
                    ],
                    unmapped_facts=[
                        UnmappedFact(
                            fact="Low-level detail",
                            reason="No suitable field.",
                        )
                    ],
                ),
            ],
            summary="Mixed results.",
        )
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(len(report.unmapped_facts), 1)


class PatchDraftLoopWithQualityReviewTests(unittest.IsolatedAsyncioTestCase):
    """Test the lean schema-late patch loop."""

    async def test_accept_clean_patch(self):
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset", "keywords": ["existing"]}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(LEAN_DESCRIPTION_PATCH_OUTPUT),
        )

        result = await patch_draft_from_content_chunks(
            initial_context=initial_context,
            initial_draft=initial_draft,
            content_chunks_by_file=chunks,
            profile_manifest=manifest,
            profile_json_schema=PROFILE_JSON_SCHEMA,
            model=patch_model,
            num_chunks_per_turn=1,
        )

        self.assertEqual(result.draft["description"], "Updated with chunk evidence.")
        self.assertEqual(len(result.patches), 1)
        self.assertEqual(result.patches[0].accepted_fields, ["description"])
        self.assertEqual(result.patches[0].validation_errors, [])
        self.assertEqual(result.patches[0].quality_report.overall_decision, "accept")

    async def test_schema_writer_applies_structured_activity_patch(self):
        initial_context = make_initial_context()
        initial_draft = {
            "title": "Test Dataset",
            "was_generated_by": [
                {"id": "activity-1", "description": ["Original description."]}
            ],
        }
        manifest = make_profile_manifest()
        chunks = make_content_chunks()
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(LEAN_ACTIVITY_PATCH_OUTPUT),
        )

        result = await patch_draft_from_content_chunks(
            initial_context=initial_context,
            initial_draft=initial_draft,
            content_chunks_by_file=chunks,
            profile_manifest=manifest,
            profile_json_schema=PROFILE_JSON_SCHEMA,
            model=patch_model,
            num_chunks_per_turn=1,
        )

        self.assertEqual(len(result.patches), 1)
        self.assertIn("was_generated_by", result.patches[0].accepted_fields)
        activity = result.draft["was_generated_by"][0]
        self.assertIn("has_qualitative_attribute", activity)
        self.assertEqual(activity["description"], ["Original description."])

    async def test_discovery_with_no_information_skips_patch_writer(self):
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset", "keywords": ["existing"]}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(NO_INFORMATION_OUTPUT),
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.create_schema_bound_patch",
        ) as patch_writer:
            with mock.patch(
                "app.domain.extraction.patch_draft.repair_schema_bound_patch",
            ) as repair_writer:
                result = await patch_draft_from_content_chunks(
                    initial_context=initial_context,
                    initial_draft=initial_draft,
                    content_chunks_by_file=chunks,
                    profile_manifest=manifest,
                    profile_json_schema=PROFILE_JSON_SCHEMA,
                    model=patch_model,
                    num_chunks_per_turn=1,
                )

        self.assertEqual(result.draft, initial_draft)
        self.assertEqual(len(result.patches), 1)
        self.assertEqual(result.patches[0].candidates, [])
        self.assertEqual(result.patches[0].accepted_fields, [])
        patch_writer.assert_not_called()
        repair_writer.assert_not_called()

    async def test_repair_attempts_until_valid_patch(self):
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset"}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(LEAN_DESCRIPTION_PATCH_OUTPUT),
        )
        invalid_patch = SchemaPatchResult(
            destination="/description",
            patch={"unknown_field": "this is not allowed by the schema"},
            reasoning="Initial writer missed the schema.",
        )
        repaired_patch = SchemaPatchResult(
            destination="/description",
            patch={"description": "Updated with chunk evidence."},
            reasoning="Repair moved the value into description.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.create_schema_bound_patch",
            return_value=invalid_patch,
        ):
            with mock.patch(
                "app.domain.extraction.patch_draft.repair_schema_bound_patch",
                side_effect=[invalid_patch, repaired_patch],
            ) as repair_writer:
                result = await patch_draft_from_content_chunks(
                    initial_context=initial_context,
                    initial_draft=initial_draft,
                    content_chunks_by_file=chunks,
                    profile_manifest=manifest,
                    profile_json_schema=PROFILE_JSON_SCHEMA,
                    model=patch_model,
                    num_chunks_per_turn=1,
                )

        self.assertEqual(repair_writer.call_count, DEFAULT_OUTPUT_RETRIES)
        self.assertEqual(result.draft["description"], "Updated with chunk evidence.")
        self.assertEqual(result.patches[0].accepted_fields, ["description"])
        self.assertEqual(result.patches[0].validation_errors, [])

    async def test_deterministic_guard_rejects_unrepaired_schema_errors(self):
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset"}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(INVALID_SCHEMA_PATCH_OUTPUT),
        )

        result = await patch_draft_from_content_chunks(
            initial_context=initial_context,
            initial_draft=initial_draft,
            content_chunks_by_file=chunks,
            profile_manifest=manifest,
            profile_json_schema=PROFILE_JSON_SCHEMA,
            model=patch_model,
            num_chunks_per_turn=1,
        )

        self.assertEqual(result.draft, initial_draft)
        self.assertEqual(len(result.patches), 1)
        self.assertEqual(result.patches[0].accepted_fields, [])
        self.assertIsNotNone(result.patches[0].validation_errors)
        self.assertTrue(len(result.patches[0].validation_errors) > 0)
        self.assertEqual(result.patches[0].quality_report.overall_decision, "reject")


if __name__ == "__main__":
    unittest.main()
