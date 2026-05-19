from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContextRequiredError,
    InitialContext,
    PatchDraftPrerequisiteError,
    extract_initial_context_from_data_package,
    initialize_draft_from_initial_context,
    patch_draft_from_content_chunks,
)
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService


class ExtractionService:
    def __init__(
        self,
        profile_service: ProfileService,
        settings: Settings,
        datasource_service: DataSourceService | None = None,
        ollama_client: OllamaClientWrapper | None = None,
        output_repository: ExtractionOutputRepository | None = None,
    ):
        self.profile_service = profile_service
        self.settings = settings
        self.datasource_service = datasource_service
        self.ollama_client = ollama_client
        self.output_repository = output_repository

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
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
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
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)

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
