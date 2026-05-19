import json
import unittest

from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import PromptedOutput

from app.domain.datasources import DataPackage, FileEntry
from app.domain.datasources.chunking import ContentChunk
from app.domain.extraction import (
    DEFAULT_OUTPUT_RETRIES,
    JSON_OUTPUT_TEMPLATE,
    InitialContext,
    InitialContextDeps,
    InitialDraftDeps,
    apply_merge_patch,
    create_initial_context_agent,
    create_initial_draft_agent,
    create_patch_draft_agent,
    extract_initial_context_from_data_package,
    initialize_draft_from_initial_context,
    list_initial_context_dataset_files,
    create_schema_validated_agent,
    prompted_json_output,
    read_initial_context_file_content,
    structured_profile_output,
    validate_json_output_against_schema,
    patch_draft_from_content_chunks,
)
from app.domain.profiles import ProfileManifest


OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title"],
    "additionalProperties": False,
}

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
            },
            "required": ["title"],
            "additionalProperties": False,
        }
    },
}

INITIAL_DRAFT_OUTPUT = {
    "title": "Mass spectrometry dataset for sample-1",
    "description": "Initial draft for a mass spectrometry package.",
    "keywords": ["mass spectrometry", "sample-1"],
}

MERGE_PATCH_OUTPUT = {
    "patch": {
        "description": "Updated with chunk evidence.",
        "keywords": ["chunk-keyword"],
    }
}


class TinyOutput(BaseModel):
    title: str


INITIAL_CONTEXT_OUTPUT = {
    "device_name": "Mass spectrometer",
    "device_model": "MS-1000",
    "entities_analyzed": ["sample-1"],
    "analytical_technique": "mass spectrometry",
    "file_relationships": [
        {
            "source_file": "README.txt",
            "related_file": "data/sample.csv",
            "relationship_type": "metadata_for",
            "description": "README describes the sample table.",
            "evidence": "README references sample.csv",
            "confidence": 0.9,
        }
    ],
    "metadata_sources": [
        {
            "file_path": "README.txt",
            "source_type": "README",
            "description": "Contains instrument and sample metadata.",
            "extracted_fields": ["device_name", "entities_analyzed"],
            "evidence": "Instrument: Mass spectrometer",
            "confidence": 0.95,
        }
    ],
    "keywords": ["mass spectrometry", "sample-1"],
    "summary": "The package contains mass spectrometry data for sample-1.",
}


def make_data_package() -> DataPackage:
    return DataPackage(
        file_name="test-package",
        files=[
            FileEntry(
                file_path="README.txt",
                file_name="README.txt",
                file_extension=".txt",
                raw_content=(
                    b"Instrument: Mass spectrometer MS-1000\n"
                    b"Sample: sample-1\n"
                    b"Related data: data/sample.csv\n"
                ),
            ),
            FileEntry(
                file_path="data/sample.csv",
                file_name="sample.csv",
                file_extension=".csv",
                raw_content=b"sample,intensity\nsample-1,42\n",
            ),
        ],
    )


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


class ExtractionAgentHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_prompted_json_output_uses_default_template_for_fixed_models(self):
        output = prompted_json_output(TinyOutput, name="TinyOutput")

        self.assertIsInstance(output, PromptedOutput)
        self.assertEqual(output.template, JSON_OUTPUT_TEMPLATE)
        self.assertEqual(output.name, "TinyOutput")

    def test_structured_profile_output_accepts_dynamic_json_schema(self):
        output = structured_profile_output(
            OUTPUT_SCHEMA,
            name="DatasetMetadata",
            description="Metadata extracted from a research dataset",
        )

        self.assertIsInstance(output, PromptedOutput)
        self.assertEqual(output.name, "DatasetMetadata")
        self.assertEqual(
            output.description,
            "Metadata extracted from a research dataset",
        )

    def test_validate_json_output_against_schema_returns_valid_output(self):
        document = {"title": "Catalyst dataset"}

        result = validate_json_output_against_schema(document, OUTPUT_SCHEMA)

        self.assertEqual(result, document)

    def test_validate_json_output_against_schema_raises_model_retry(self):
        with self.assertRaises(ModelRetry) as error:
            validate_json_output_against_schema(
                {"description": "Missing required title."},
                OUTPUT_SCHEMA,
            )

        self.assertIn("Output did not match JSON Schema", str(error.exception))
        self.assertIn("'title' is a required property", str(error.exception))

    def test_create_schema_validated_agent_uses_pydantic_ai_retry_budget(self):
        agent = create_schema_validated_agent(
            model="test",
            json_schema=OUTPUT_SCHEMA,
            output_name="DatasetMetadata",
            instructions="Extract dataset metadata.",
        )

        self.assertEqual(agent._max_output_retries, DEFAULT_OUTPUT_RETRIES)

    def test_initial_context_model_accepts_expected_shape(self):
        context = InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT)

        self.assertEqual(context.device_model, "MS-1000")
        self.assertEqual(context.file_relationships[0].relationship_type, "metadata_for")

    def test_initial_context_file_tools_use_data_package_content(self):
        deps = InitialContextDeps(
            data_package=make_data_package(),
            max_files_to_read=1,
            max_chars_per_file=12,
        )

        self.assertEqual(
            list_initial_context_dataset_files(deps),
            ["README.txt", "data/sample.csv"],
        )
        self.assertEqual(
            read_initial_context_file_content(
                deps,
                file_path="README.txt",
                max_chars=100,
            ),
            "Instrument: ...[truncated]",
        )
        self.assertEqual(
            read_initial_context_file_content(
                deps,
                file_path="data/sample.csv",
                max_chars=100,
            ),
            "[File read limit reached; use already inspected files or "
            "increase max_files_to_read.]",
        )

    def test_create_initial_context_agent_uses_retry_budget(self):
        agent = create_initial_context_agent(
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(INITIAL_CONTEXT_OUTPUT),
            ),
        )

        self.assertEqual(agent._max_output_retries, DEFAULT_OUTPUT_RETRIES)

    async def test_extract_initial_context_from_data_package_returns_model(self):
        result = await extract_initial_context_from_data_package(
            data_package=make_data_package(),
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(INITIAL_CONTEXT_OUTPUT),
            ),
        )

        self.assertIsInstance(result, InitialContext)
        self.assertEqual(result.device_name, "Mass spectrometer")

    def test_create_initial_draft_agent_uses_retry_budget(self):
        agent = create_initial_draft_agent(
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(INITIAL_DRAFT_OUTPUT),
            ),
            profile_json_schema=PROFILE_JSON_SCHEMA,
            target_class="Dataset",
        )

        self.assertEqual(agent._max_output_retries, DEFAULT_OUTPUT_RETRIES)

    async def test_initialize_draft_from_initial_context_returns_profile_dict(self):
        result = await initialize_draft_from_initial_context(
            initial_context=InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT),
            data_package=make_data_package(),
            profile_manifest=make_profile_manifest(),
            profile_json_schema=PROFILE_JSON_SCHEMA,
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(INITIAL_DRAFT_OUTPUT),
            ),
        )

        self.assertEqual(result["title"], "Mass spectrometry dataset for sample-1")
        self.assertEqual(result["keywords"], ["mass spectrometry", "sample-1"])

    def test_initial_draft_deps_carry_context_and_profile(self):
        deps = InitialDraftDeps(
            initial_context=InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT),
            data_package=make_data_package(),
            profile_manifest=make_profile_manifest(),
            profile_json_schema=PROFILE_JSON_SCHEMA,
        )

        self.assertEqual(deps.profile_manifest.identifier, "test-profile")
        self.assertEqual(deps.initial_context.device_model, "MS-1000")

    def test_apply_merge_patch_merges_nested_values_and_string_lists(self):
        draft = {
            "title": "Dataset",
            "keywords": ["existing"],
            "items": [{"id": "a", "label": "old"}],
        }
        patch = {
            "keywords": ["existing", "new"],
            "items": [{"id": "a", "label": "new"}, {"label": "added"}],
            "ignored": None,
        }

        result = apply_merge_patch(draft, patch)

        self.assertEqual(result["keywords"], ["existing", "new"])
        self.assertEqual(result["items"][0]["label"], "new")
        self.assertEqual(result["items"][1]["label"], "added")
        self.assertNotIn("ignored", result)
        self.assertNotIn("ignored", draft)

    def test_create_patch_draft_agent_uses_retry_budget(self):
        agent = create_patch_draft_agent(
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(MERGE_PATCH_OUTPUT),
            ),
        )

        self.assertEqual(agent._max_output_retries, DEFAULT_OUTPUT_RETRIES)

    async def test_patch_draft_from_content_chunks_returns_updated_draft_and_patches(self):
        result = await patch_draft_from_content_chunks(
            initial_context=InitialContext.model_validate(INITIAL_CONTEXT_OUTPUT),
            initial_draft=INITIAL_DRAFT_OUTPUT,
            content_chunks_by_file=make_content_chunks(),
            profile_manifest=make_profile_manifest(),
            profile_json_schema=PROFILE_JSON_SCHEMA,
            model=TestModel(
                call_tools=[],
                custom_output_text=json.dumps(MERGE_PATCH_OUTPUT),
            ),
            num_chunks_per_turn=1,
        )

        self.assertEqual(result.draft["description"], "Updated with chunk evidence.")
        self.assertIn("chunk-keyword", result.draft["keywords"])
        self.assertEqual(result.patches[0].patch, MERGE_PATCH_OUTPUT["patch"])


if __name__ == "__main__":
    unittest.main()
