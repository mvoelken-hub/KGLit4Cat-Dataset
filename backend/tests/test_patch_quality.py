import copy
import json
import unittest

from pydantic_ai.models.test import TestModel

from app.domain.datasources import DataPackage, FileEntry
from app.domain.datasources.chunking import ContentChunk
from app.domain.extraction import (
    InitialContext,
    PatchCandidate,
    PatchRecord,
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
    """Test the patch loop integration with quality review.

    These tests use TestModel to control the patch extraction agent output
    and mock the quality review agent to test the decision branching logic.
    """

    async def test_accept_clean_patch(self):
        """When the quality agent accepts all candidates, they are applied to the draft."""
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset", "keywords": ["existing"]}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()

        # The patch extraction agent returns field-level candidates.
        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(FIELD_PATCH_OUTPUT),
        )

        # The quality agent accepts all candidates.
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="accept",
                    issues=[],
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="All candidates accepted.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            result = await patch_draft_from_content_chunks(
                initial_context=initial_context,
                initial_draft=initial_draft,
                content_chunks_by_file=chunks,
                profile_manifest=manifest,
                profile_json_schema=PROFILE_JSON_SCHEMA,
                model=patch_model,
                num_chunks_per_turn=1,
            )

        # The draft should include the patch content.
        self.assertEqual(result.draft["description"], "Updated with chunk evidence.")
        self.assertIn("chunk-keyword", result.draft["keywords"])
        # The patch record should have accepted fields.
        self.assertEqual(len(result.patches), 1)
        self.assertIn("description", result.patches[0].accepted_fields)
        self.assertIn("keywords", result.patches[0].accepted_fields)
        self.assertEqual(result.patches[0].validation_errors, [])

    async def test_revise_description_abusive_patch(self):
        """When the quality agent revises a candidate, the revised patch is applied."""
        initial_context = make_initial_context()
        initial_draft = {
            "title": "Test Dataset",
            "was_generated_by": [
                {"id": "activity-1", "description": ["Original description."]}
            ],
        }
        manifest = make_profile_manifest()
        chunks = make_content_chunks()

        # The patch extraction agent returns a candidate that abuses description.
        abusive_candidate_output = {
            "candidates": [
                {
                    "field_path": "was_generated_by",
                    "patch": {
                        "was_generated_by": [
                            {
                                "id": "activity-1",
                                "description": [
                                    "Long list of operational details that should not be a description."
                                ],
                                "has_qualitative_attribute": [
                                    {"title": "pulse program", "value": "zgpg30"}
                                ],
                            }
                        ]
                    },
                    "confidence": 0.7,
                    "reasoning": "Chunk contains activity details.",
                    "source_evidence": ["pulse sequence zg30"],
                }
            ]
        }

        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(abusive_candidate_output),
        )

        # The quality agent revises the candidate to remove description abuse.
        revised_patch = {
            "was_generated_by": [
                {
                    "id": "activity-1",
                    "has_qualitative_attribute": [
                        {"title": "pulse program", "value": "zgpg30"}
                    ],
                }
            ]
        }
        revise_report = PatchQualityReport(
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
                    revised_patch=revised_patch,
                )
            ],
            summary="Moved structured fact out of description.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=revise_report,
        ):
            result = await patch_draft_from_content_chunks(
                initial_context=initial_context,
                initial_draft=initial_draft,
                content_chunks_by_file=chunks,
                profile_manifest=manifest,
                profile_json_schema=PROFILE_JSON_SCHEMA,
                model=patch_model,
                num_chunks_per_turn=1,
            )

        # The accepted patch should be the revised one (without description abuse).
        self.assertEqual(len(result.patches), 1)
        self.assertIn("was_generated_by", result.patches[0].accepted_fields)
        # The draft should have the qualitative attribute but not the description dump.
        activity = result.draft["was_generated_by"][0]
        self.assertIn("has_qualitative_attribute", activity)

    async def test_reject_noisy_patch(self):
        """When the quality agent rejects all candidates, the draft remains unchanged."""
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset", "keywords": ["existing"]}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()

        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(FIELD_PATCH_OUTPUT),
        )

        # The quality agent rejects all candidates.
        reject_report = PatchQualityReport(
            overall_decision="reject",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="reject",
                    issues=[
                        PatchQualityIssue(
                            path="$.description",
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
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="reject",
                    issues=[
                        PatchQualityIssue(
                            path="$.keywords",
                            issue_type="unsupported_fact",
                            explanation="Keyword not supported by evidence.",
                        )
                    ],
                ),
            ],
            summary="All candidates rejected.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=reject_report,
        ):
            result = await patch_draft_from_content_chunks(
                initial_context=initial_context,
                initial_draft=initial_draft,
                content_chunks_by_file=chunks,
                profile_manifest=manifest,
                profile_json_schema=PROFILE_JSON_SCHEMA,
                model=patch_model,
                num_chunks_per_turn=1,
            )

        # The draft should remain unchanged.
        self.assertEqual(result.draft, initial_draft)
        # The patch record should have no accepted fields.
        self.assertEqual(len(result.patches), 1)
        self.assertEqual(result.patches[0].accepted_fields, [])
        self.assertIsNotNone(result.patches[0].quality_report)
        self.assertEqual(result.patches[0].quality_report.overall_decision, "reject")

    async def test_revised_patch_must_still_validate(self):
        """When a revised candidate still fails schema validation, it is rejected."""
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset"}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()

        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(FIELD_PATCH_OUTPUT),
        )

        # The quality agent revises, but the revised patch introduces an
        # additionalProperties violation (unknown field "unknown_field").
        invalid_revised_patch = {
            "title": "Test Dataset",
            "unknown_field": "this is not allowed by the schema",
        }
        revise_report = PatchQualityReport(
            overall_decision="revise",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="description",
                    decision="revise",
                    issues=[
                        PatchQualityIssue(
                            path="$",
                            issue_type="schema_mismatch",
                            explanation="Revised patch introduces unknown fields.",
                        )
                    ],
                    revised_patch=invalid_revised_patch,
                ),
                CandidateQualityRating(
                    field_path="keywords",
                    decision="accept",
                    issues=[],
                ),
            ],
            summary="Attempted revision but still invalid.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=revise_report,
        ):
            result = await patch_draft_from_content_chunks(
                initial_context=initial_context,
                initial_draft=initial_draft,
                content_chunks_by_file=chunks,
                profile_manifest=manifest,
                profile_json_schema=PROFILE_JSON_SCHEMA,
                model=patch_model,
                num_chunks_per_turn=1,
            )

        # The revised description candidate is rejected because it fails
        # schema validation, but the keywords candidate is accepted.
        self.assertIn("keywords", result.patches[0].accepted_fields)
        self.assertNotIn("description", result.patches[0].accepted_fields)
        # The draft should have the keywords but not the invalid revised patch.
        self.assertIn("chunk-keyword", result.draft["keywords"])

    async def test_deterministic_guard_rejects_accept_with_schema_errors(self):
        """Even if the quality agent accepts, schema errors cause rejection."""
        initial_context = make_initial_context()
        initial_draft = {"title": "Test Dataset"}
        manifest = make_profile_manifest()
        chunks = make_content_chunks()

        # The patch extraction agent returns candidates that add an unknown field.
        invalid_candidate_output = {
            "candidates": [
                {
                    "field_path": "unknown_field",
                    "patch": {"unknown_field": "not in schema"},
                    "confidence": 0.5,
                    "reasoning": "Chunk mentions unknown concept.",
                    "source_evidence": ["some text"],
                }
            ]
        }

        patch_model = TestModel(
            call_tools=[],
            custom_output_text=json.dumps(invalid_candidate_output),
        )

        # The quality agent accepts the candidates despite schema errors.
        accept_report = PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path="unknown_field",
                    decision="accept",
                    issues=[],
                )
            ],
            summary="Patch looks semantically fine.",
        )

        import unittest.mock as mock

        with mock.patch(
            "app.domain.extraction.patch_draft.review_patch_semantic_quality",
            return_value=accept_report,
        ):
            result = await patch_draft_from_content_chunks(
                initial_context=initial_context,
                initial_draft=initial_draft,
                content_chunks_by_file=chunks,
                profile_manifest=manifest,
                profile_json_schema=PROFILE_JSON_SCHEMA,
                model=patch_model,
                num_chunks_per_turn=1,
            )

        # The draft should remain unchanged because the deterministic guard
        # rejects structurally invalid patches even when the quality agent accepts.
        self.assertEqual(result.draft, initial_draft)
        self.assertEqual(len(result.patches), 1)
        self.assertEqual(result.patches[0].accepted_fields, [])
        self.assertIsNotNone(result.patches[0].validation_errors)
        self.assertTrue(len(result.patches[0].validation_errors) > 0)


if __name__ == "__main__":
    unittest.main()
