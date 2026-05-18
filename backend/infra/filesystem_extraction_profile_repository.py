import json
from pathlib import Path
from typing import Any

from app.domain.extraction import (
    GeneratedProfileArtifacts,
    ProfileManifest,
    ProfileNotFoundError,
    validate_profile_identifier,
)


PROFILE_MANIFEST_FILE = "profile_manifest.json"
SOURCE_SCHEMA_FILE = "source_schema.yaml"
MERGED_SCHEMA_FILE = "merged_schema.yaml"
JSON_SCHEMA_FILE = "json_schema.json"
JSONLD_CONTEXT_FILE = "jsonld_context.json"


class FileSystemExtractionProfileRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_profile(self, artifacts: GeneratedProfileArtifacts) -> ProfileManifest:
        profile_dir = self._profile_dir(artifacts.manifest.identifier)
        profile_dir.mkdir(parents=True, exist_ok=True)

        self._write_text(profile_dir / SOURCE_SCHEMA_FILE, artifacts.source_schema)
        self._write_text(profile_dir / MERGED_SCHEMA_FILE, artifacts.merged_schema)
        self._write_json(profile_dir / JSON_SCHEMA_FILE, artifacts.json_schema)
        self._write_json(profile_dir / JSONLD_CONTEXT_FILE, artifacts.jsonld_context)
        self._write_json(
            profile_dir / PROFILE_MANIFEST_FILE,
            artifacts.manifest.model_dump(mode="json"),
        )

        return artifacts.manifest

    def get_profile_manifest(self, identifier: str) -> ProfileManifest | None:
        manifest_path = self._profile_dir(identifier) / PROFILE_MANIFEST_FILE
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as file:
            return ProfileManifest.model_validate(json.load(file))

    def list_profile_manifests(self) -> list[ProfileManifest]:
        if not self.base_path.exists():
            return []

        manifests = []
        for manifest_path in self.base_path.glob(f"*/{PROFILE_MANIFEST_FILE}"):
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifests.append(ProfileManifest.model_validate(json.load(file)))

        return sorted(manifests, key=lambda manifest: manifest.identifier)

    def load_merged_schema(self, identifier: str) -> str:
        return self._read_required_text(identifier, MERGED_SCHEMA_FILE)

    def load_json_schema(self, identifier: str) -> dict[str, Any]:
        return self._read_required_json(identifier, JSON_SCHEMA_FILE)

    def load_jsonld_context(self, identifier: str) -> dict[str, Any]:
        return self._read_required_json(identifier, JSONLD_CONTEXT_FILE)

    def _read_required_text(self, identifier: str, file_name: str) -> str:
        path = self._profile_dir(identifier) / file_name
        if not path.exists():
            raise ProfileNotFoundError(f"Profile artifact '{file_name}' not found for '{identifier}'.")
        return path.read_text(encoding="utf-8")

    def _read_required_json(self, identifier: str, file_name: str) -> dict[str, Any]:
        path = self._profile_dir(identifier) / file_name
        if not path.exists():
            raise ProfileNotFoundError(f"Profile artifact '{file_name}' not found for '{identifier}'.")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _profile_dir(self, identifier: str) -> Path:
        validate_profile_identifier(identifier)
        base_path = self.base_path.resolve()
        profile_dir = (base_path / identifier).resolve()
        if not profile_dir.is_relative_to(base_path):
            raise ValueError(f"Profile path escapes profile storage directory: {identifier}")
        return profile_dir

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, content: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
