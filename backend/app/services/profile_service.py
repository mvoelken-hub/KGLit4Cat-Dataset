from __future__ import annotations

import asyncio
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import UploadFile

from app.domain.profiles import (
    JsonLdExportResult,
    ProfileAlreadyExistsError,
    ProfileManifest,
    ProfileNotFoundError,
    ProfileSourceError,
    ProfileValidationResult,
    export_document_to_jsonld,
    generate_profile_artifacts,
    validate_document_against_profile,
    validate_profile_identifier,
)
from app.repositories.profile_repository import ProfileRepository


class ProfileService:
    def __init__(self, profile_repository: ProfileRepository):
        self.profile_repository = profile_repository

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

    def delete_profile(self, identifier: str) -> None:
        self.get_profile(identifier)
        self.profile_repository.delete_profile(identifier)

    def load_json_schema(self, identifier: str) -> dict:
        self.get_profile(identifier)
        return self.profile_repository.load_json_schema(identifier)

    def load_jsonld_context(self, identifier: str) -> dict:
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

    async def _load_schema_from_upload(
        self,
        schema_file: UploadFile | None,
    ) -> tuple[str, str]:
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
