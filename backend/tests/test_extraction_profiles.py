from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.profiles import router
from app.dependencies import get_profile_service
from app.domain.profiles import (
    InvalidProfileIdentifierError,
    ProfileCompatibilityError,
    detect_enrichable_fields,
    generate_profile_artifacts,
    validate_profile_identifier,
)
from app.services.profile_service import ProfileService
from infra.filesystem_profile_repository import FileSystemProfileRepository


TINY_PROFILE_SCHEMA = """
id: https://example.org/test-profile
name: test_profile
imports:
  - linkml:types
prefixes:
  linkml:
    prefix_prefix: linkml
    prefix_reference: https://w3id.org/linkml/
  ex:
    prefix_prefix: ex
    prefix_reference: https://example.org/
default_prefix: ex
default_range: string
slots:
  title:
    required: true
  description:
  type:
  unit:
classes:
  Dataset:
    slots:
      - title
      - description
      - type
      - unit
""".strip()


def make_artifacts(identifier: str = "test-profile"):
    return generate_profile_artifacts(
        identifier=identifier,
        source_schema=TINY_PROFILE_SCHEMA,
        source="profile.yaml",
        source_type="upload",
        schema_file_name="profile.yaml",
        target_class="Dataset",
        version="1.0.0",
    )


class ExtractionProfileDomainTests(unittest.TestCase):
    def test_profile_identifier_validation_rejects_path_like_values(self):
        with self.assertRaises(InvalidProfileIdentifierError):
            validate_profile_identifier("../profile")

    def test_missing_target_class_is_rejected(self):
        with self.assertRaises(ProfileCompatibilityError):
            generate_profile_artifacts(
                identifier="test-profile",
                source_schema=TINY_PROFILE_SCHEMA,
                source="profile.yaml",
                source_type="upload",
                schema_file_name="profile.yaml",
                target_class="Catalog",
            )

    def test_checksum_is_stable_for_same_schema(self):
        first = make_artifacts("test-profile")
        second = make_artifacts("test-profile")

        self.assertEqual(first.manifest.checksum, second.manifest.checksum)

    def test_enrichable_fields_are_detected_from_target_slots(self):
        self.assertEqual(
            detect_enrichable_fields(
                induced_slot_names=["title", "type", "unit", "ignored"],
            ),
            ["type", "unit"],
        )


class FileSystemProfileRepositoryTests(unittest.TestCase):
    def test_profile_artifacts_round_trip(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemProfileRepository(Path(temporary_directory))
            saved_manifest = repository.save_profile(make_artifacts())

            loaded_manifest = repository.get_profile_manifest(saved_manifest.identifier)
            json_schema = repository.load_json_schema(saved_manifest.identifier)
            context = repository.load_jsonld_context(saved_manifest.identifier)
            merged_schema = repository.load_merged_schema(saved_manifest.identifier)

        self.assertIsNotNone(loaded_manifest)
        self.assertEqual(loaded_manifest.identifier, "test-profile")  # type: ignore[union-attr]
        self.assertIn("Dataset", json_schema["$defs"])
        self.assertIn("@context", context)
        self.assertIn("classes:", merged_schema)


class ExtractionProfileApiTests(unittest.TestCase):
    def make_client(self, temporary_directory: str) -> TestClient:
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        service = ProfileService(FileSystemProfileRepository(Path(temporary_directory)))
        app.dependency_overrides[get_profile_service] = lambda: service
        return TestClient(app)

    def test_github_blob_schema_urls_are_normalized_to_raw_urls(self):
        self.assertEqual(
            ProfileService._normalize_schema_url(
                "https://github.com/nfdi-de/chem-dcat-ap/blob/main/src/chem_dcat_ap/schema/chem_dcat_ap.yaml"
            ),
            "https://raw.githubusercontent.com/nfdi-de/chem-dcat-ap/main/src/chem_dcat_ap/schema/chem_dcat_ap.yaml",
        )

    def test_blank_enrichable_fields_are_treated_as_omitted(self):
        self.assertIsNone(ProfileService._normalize_enrichable_fields([""]))

    def test_profile_registration_validation_and_jsonld_export(self):
        with TemporaryDirectory() as temporary_directory:
            client = self.make_client(temporary_directory)

            register_response = client.post(
                "/api/v1/profiles",
                data={
                    "identifier": "test-profile",
                    "target_class": "Dataset",
                    "version": "1.0.0",
                    "enrichable_fields": "",
                },
                files={
                    "schema_file": (
                        "profile.yaml",
                        TINY_PROFILE_SCHEMA,
                        "application/yaml",
                    )
                },
            )
            self.assertEqual(register_response.status_code, 201)
            registered = register_response.json()
            self.assertEqual(registered["identifier"], "test-profile")
            self.assertEqual(registered["enrichable_fields"], ["type", "unit"])

            list_response = client.get("/api/v1/profiles")
            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json()[0]["identifier"], "test-profile")

            schema_response = client.get(
                "/api/v1/profiles/test-profile/json-schema"
            )
            self.assertEqual(schema_response.status_code, 200)
            self.assertIn("Dataset", schema_response.json()["$defs"])

            context_response = client.get(
                "/api/v1/profiles/test-profile/jsonld-context"
            )
            self.assertEqual(context_response.status_code, 200)
            self.assertIn("@context", context_response.json())

            valid_response = client.post(
                "/api/v1/profiles/test-profile/validate",
                json={"document": {"title": "Catalyst dataset", "type": "Dataset"}},
            )
            self.assertEqual(valid_response.status_code, 200)
            self.assertTrue(valid_response.json()["valid"])

            invalid_response = client.post(
                "/api/v1/profiles/test-profile/validate",
                json={"document": {"description": "Missing title", "extra": "nope"}},
            )
            self.assertEqual(invalid_response.status_code, 200)
            self.assertFalse(invalid_response.json()["valid"])

            jsonld_response = client.post(
                "/api/v1/profiles/test-profile/jsonld",
                json={
                    "document": {
                        "title": "Catalyst dataset",
                        "description": None,
                    }
                },
            )
            self.assertEqual(jsonld_response.status_code, 200)
            jsonld = jsonld_response.json()
            self.assertGreater(jsonld["triple_count"], 0)
            self.assertIn("@context", jsonld["document"])
            self.assertEqual(jsonld["document"]["title"], "Catalyst dataset")
            self.assertNotIn("description", jsonld["document"])


if __name__ == "__main__":
    unittest.main()
