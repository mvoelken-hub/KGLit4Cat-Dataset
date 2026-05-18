from typing import Any, Protocol

from app.domain.extraction import GeneratedProfileArtifacts, ProfileManifest


class ExtractionProfileRepository(Protocol):
    def save_profile(self, artifacts: GeneratedProfileArtifacts) -> ProfileManifest:
        ...

    def get_profile_manifest(self, identifier: str) -> ProfileManifest | None:
        ...

    def list_profile_manifests(self) -> list[ProfileManifest]:
        ...

    def load_merged_schema(self, identifier: str) -> str:
        ...

    def load_json_schema(self, identifier: str) -> dict[str, Any]:
        ...

    def load_jsonld_context(self, identifier: str) -> dict[str, Any]:
        ...
