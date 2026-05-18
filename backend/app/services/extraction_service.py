from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import UploadFile

from app.core.config import Settings
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContextRequiredError,
    InitialContext,
    PatchDraftPrerequisiteError,
    JsonLdExportResult,
    ProfileAlreadyExistsError,
    ProfileManifest,
    ProfileNotFoundError,
    ProfileSourceError,
    ProfileValidationResult,
    export_document_to_jsonld,
    extract_initial_context_from_data_package,
    generate_profile_artifacts,
    initialize_draft_from_initial_context,
    patch_draft_from_content_chunks,
    validate_document_against_profile,
    validate_profile_identifier,
)
from app.repositories.extraction_profile_repository import ExtractionProfileRepository
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService


class ExtractionService:
    def __init__(
        self,
        profile_repository: ExtractionProfileRepository,
        settings: Settings,
        datasource_service: DataSourceService | None = None,
        ollama_client: OllamaClientWrapper | None = None,
        output_repository: ExtractionOutputRepository | None = None,
    ):
        self.profile_repository = profile_repository
        self.settings = settings
        self.datasource_service = datasource_service
        self.ollama_client = ollama_client
        self.output_repository = output_repository

    async def register_profile(
        self,
        *,
        identifier: str,
        target_class: str = "Dataset",
        schema_url: str | None = None,
        schema_file: UploadFile | None = None,
        version: str | None = None,
        enrichable_fields: list[str] | None = None,
    ) -> ProfileManifest:
        validate_profile_identifier(identifier)

        schema_file_present = self._has_uploaded_schema(schema_file)
        if bool(schema_url) == schema_file_present:
            raise ProfileSourceError("Exactly one of schema_url or schema_file is required.")

        if self.profile_repository.get_profile_manifest(identifier) is not None:
            raise ProfileAlreadyExistsError(
                f"A profile with identifier '{identifier}' already exists."
            )

        normalized_enrichable_fields = self._normalize_enrichable_fields(enrichable_fields)

        if schema_url:
            normalized_schema_url = self._normalize_schema_url(schema_url)
            source_schema = await self._load_schema_from_url(normalized_schema_url)
            artifacts = generate_profile_artifacts(
                identifier=identifier,
                source_schema=source_schema,
                source=normalized_schema_url,
                source_type="url",
                target_class=target_class,
                version=version,
                enrichable_fields=normalized_enrichable_fields,
                schema_url=normalized_schema_url,
                schema_location=normalized_schema_url,
                source_suffix=self._suffix_from_source(normalized_schema_url),
            )
        else:
            source_schema, file_name = await self._load_schema_from_upload(schema_file)
            artifacts = generate_profile_artifacts(
                identifier=identifier,
                source_schema=source_schema,
                source=file_name,
                source_type="upload",
                target_class=target_class,
                version=version,
                enrichable_fields=normalized_enrichable_fields,
                schema_file_name=file_name,
                source_suffix=self._suffix_from_source(file_name),
            )

        return self.profile_repository.save_profile(artifacts)

    def list_profiles(self) -> list[ProfileManifest]:
        return self.profile_repository.list_profile_manifests()

    def get_profile(self, identifier: str) -> ProfileManifest:
        manifest = self.profile_repository.get_profile_manifest(identifier)
        if manifest is None:
            raise ProfileNotFoundError(f"Profile with identifier '{identifier}' not found.")
        return manifest

    def get_json_schema(self, identifier: str) -> dict:
        self.get_profile(identifier)
        return self.profile_repository.load_json_schema(identifier)

    def get_jsonld_context(self, identifier: str) -> dict:
        self.get_profile(identifier)
        return self.profile_repository.load_jsonld_context(identifier)

    def validate_document(
        self,
        *,
        identifier: str,
        document: dict,
    ) -> ProfileValidationResult:
        manifest = self.get_profile(identifier)
        json_schema = self.profile_repository.load_json_schema(identifier)
        return validate_document_against_profile(
            document=document,
            json_schema=json_schema,
            target_class=manifest.target_class,
        )

    def export_jsonld(
        self,
        *,
        identifier: str,
        document: dict,
    ) -> JsonLdExportResult:
        manifest = self.get_profile(identifier)
        jsonld_context = self.profile_repository.load_jsonld_context(identifier)
        return export_document_to_jsonld(
            document=document,
            jsonld_context=jsonld_context,
            target_class=manifest.target_class,
        )

    async def extract_initial_context(
        self,
        *,
        data_package_id: str,
        max_files_to_read: int = 12,
        max_chars_per_file: int = 3000,
    ) -> InitialContext:
        if self.datasource_service is None or self.ollama_client is None:
            raise RuntimeError(
                "ExtractionService requires datasource_service and ollama_client "
                "to run extraction agents."
            )

        data_package = self.datasource_service.get_data_package(data_package_id)
        initial_context = await extract_initial_context_from_data_package(
            data_package=data_package,
            model=self.ollama_client.agent_model,
            max_files_to_read=max_files_to_read,
            max_chars_per_file=max_chars_per_file,
        )
        if self.output_repository is not None:
            self.output_repository.save_initial_context(
                workflow_id=data_package_id,
                initial_context=initial_context,
            )
        return initial_context

    async def extract_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
    ) -> dict[str, Any]:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "and output_repository to run extraction agents."
            )

        data_package = self.datasource_service.get_data_package(data_package_id)
        profile_manifest = self.get_profile(profile_identifier)
        profile_json_schema = self.profile_repository.load_json_schema(profile_identifier)
        try:
            initial_context = self.output_repository.load_initial_context(data_package_id)
        except FileNotFoundError as exc:
            raise InitialContextRequiredError(
                "Initial context output not found for workflow "
                f"'{data_package_id}'. Run /api/v1/extraction/initial-context "
                "before /api/v1/extraction/initial-draft."
            ) from exc

        initial_draft = await initialize_draft_from_initial_context(
            initial_context=initial_context,
            data_package=data_package,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            model=self.ollama_client.agent_model,
        )
        self.output_repository.save_initial_draft(
            workflow_id=data_package_id,
            initial_draft=initial_draft,
        )
        return initial_draft

    async def patch_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        num_chunks_per_turn: int | None = None,
    ) -> dict[str, Any]:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "and output_repository to run extraction agents."
            )

        data_package = self.datasource_service.get_data_package(data_package_id)
        profile_manifest = self.get_profile(profile_identifier)
        profile_json_schema = self.profile_repository.load_json_schema(profile_identifier)

        try:
            initial_context = self.output_repository.load_initial_context(data_package_id)
            initial_draft = self.output_repository.load_initial_draft(data_package_id)
        except FileNotFoundError as exc:
            raise PatchDraftPrerequisiteError(
                "Patch extraction requires existing initial_context.json and "
                "initial_draft.json workflow outputs. Run /api/v1/extraction/"
                "initial-context and /api/v1/extraction/initial-draft first."
            ) from exc

        content_chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        if not content_chunks_by_file:
            raise ChunkingRequiredError(
                "Patch extraction requires completed datasource chunking. "
                "Run /api/v1/datasources/chunk until it returns completed chunks first."
            )

        result = await patch_draft_from_content_chunks(
            initial_context=initial_context,
            initial_draft=initial_draft,
            content_chunks_by_file=content_chunks_by_file,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            model=self.ollama_client.agent_model,
            num_chunks_per_turn=(
                num_chunks_per_turn
                if num_chunks_per_turn is not None
                else self.settings.num_chunks_per_turn
            ),
        )

        self.output_repository.save_draft(
            workflow_id=data_package_id,
            draft=result.draft,
        )
        for patch_record in result.patches:
            self.output_repository.save_patch(
                workflow_id=data_package_id,
                patch_file_name=patch_record.file_name,
                patch=patch_record.patch,
            )

        return result.draft

    async def _load_schema_from_upload(self, schema_file: UploadFile | None) -> tuple[str, str]:
        if schema_file is None:
            raise ProfileSourceError("Missing uploaded schema file.")
        if not schema_file.filename:
            raise ProfileSourceError("Uploaded schema file must have a filename.")

        content = await schema_file.read()
        try:
            return content.decode("utf-8"), schema_file.filename
        except UnicodeDecodeError as exc:
            raise ProfileSourceError("Uploaded schema file must be UTF-8 text.") from exc

    async def _load_schema_from_url(self, schema_url: str) -> str:
        return await asyncio.to_thread(self._fetch_schema_url, schema_url)

    @staticmethod
    def _has_uploaded_schema(schema_file: UploadFile | None) -> bool:
        return schema_file is not None and bool(schema_file.filename)

    @staticmethod
    def _normalize_enrichable_fields(enrichable_fields: list[str] | None) -> list[str] | None:
        if enrichable_fields is None:
            return None

        normalized_fields = [
            field.strip()
            for field in enrichable_fields
            if field.strip()
        ]
        return normalized_fields or None

    @staticmethod
    def _normalize_schema_url(schema_url: str) -> str:
        parsed = urlparse(schema_url)
        path_parts = [
            part
            for part in parsed.path.strip("/").split("/")
            if part
        ]

        if (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.lower() == "github.com"
            and len(path_parts) >= 5
            and path_parts[2] in {"blob", "raw"}
        ):
            owner, repo, _, ref = path_parts[:4]
            file_path = "/".join(path_parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"

        return schema_url

    @staticmethod
    def _fetch_schema_url(schema_url: str) -> str:
        request = Request(
            schema_url,
            headers={
                "Accept": "application/yaml, application/x-yaml, application/json, text/yaml, text/plain",
                "User-Agent": "SIMONE-profile-registration",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset)
        except UnicodeDecodeError as exc:
            raise ProfileSourceError("Remote schema response must be text.") from exc
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ProfileSourceError(f"Failed to load remote schema '{schema_url}': {exc}") from exc

    @staticmethod
    def _suffix_from_source(source: str) -> str:
        path = urlparse(source).path or source
        suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ".yaml"
        return suffix
