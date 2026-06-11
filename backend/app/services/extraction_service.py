from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import jsonpatch
from pydantic import ValidationError
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.core.logging import logger
from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk
from app.domain.extraction import (
    DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    PROFILE_PROJECTION_SYSTEM_PROMPT,
    PROFILE_PATCH_SYSTEM_PROMPT,
    QUDT_QUANTITY_KIND_VOCAB,
    QUDT_UNIT_VOCAB,
    VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
    VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
    VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
    ChunkContext,
    ChunkMetadata,
    ChunkingRequiredError,
    CompleteWorkflowProgress,
    CompleteWorkflowStepProgress,
    DefinedTerm,
    ExtractionChunkRef,
    ExtractionChunkResult,
    ExtractionContext,
    ExtractionNormalization,
    ExtractionResultNotFoundError,
    ExtractionRunProgress,
    ExtractionRunResult,
    ExtractionRunState,
    ExtractionValidationError,
    ExtractionVocabQueryConfig,
    ExtractionVocabQueryRecord,
    FileContext,
    FileRankingResult,
    GroundedExtractionObject,
    ProfileFieldNormalization,
    ProfileObjectPatchResult,
    ProfilePatchDocument,
    QualitativeAttribute,
    QualitativeAttributeNormalization,
    QuantitativeAttribute,
    QuantityNormalization,
    Resource,
    TracedExtractionObject,
    VocabularyCandidateSelection,
    VocabularyFallbackQuery,
    VocabularyTermMapping,
    build_candidate_selection_prompt,
    build_extraction_context_prompt,
    build_fallback_query_prompt,
    build_object_grounding_selection_prompt,
    build_profile_patch_prompt,
    build_profile_projection_prompt,
    build_qualitative_vocab_query,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    cap_extraction_context_for_prompt,
    fallback_file_ranking,
    merge_extraction_context_results,
)
from app.domain.profiles import remove_null_values, validation_schema_for_target_class
from app.domain.semantics import VocabQuery, VocabQueryResult
from app.ollama.completion import generate_structured, repair_structured_output
from app.ollama.errors import CompletionError, MaxRetriesExceeded
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService
    from app.services.semantic_service import SemanticService


@dataclass
class _QuantityCandidateDiscovery:
    quantity: QuantitativeAttribute
    quantity_kind_query_id: str
    unit_query_id: str


@dataclass
class _QualitativeCandidateDiscovery:
    attribute: QualitativeAttribute
    query_ids: list[str]


@dataclass
class _ObjectGroundingCandidateDiscovery:
    object_identifier: str
    object_kind: str
    raw_type: str
    source_context: dict[str, Any]
    query_ids: list[str]


@dataclass
class _ProfileFieldCandidateDiscovery:
    json_path: str
    field_name: str
    source_value: str
    vocabulary_identifier: str
    query_ids: list[str]

class ExtractionService:
    def __init__(
        self,
        profile_service: ProfileService,
        settings: Settings,
        datasource_service: DataSourceService | None = None,
        ollama_client: OllamaClientWrapper | None = None,
        output_repository: ExtractionOutputRepository | None = None,
        task_registry: TaskRegistry | None = None,
        semantic_service: SemanticService | None = None,
    ):
        self.profile_service = profile_service
        self.settings = settings
        self.datasource_service = datasource_service
        self.ollama_client = ollama_client
        self.output_repository = output_repository
        self.task_registry = task_registry
        self.semantic_service = semantic_service

    async def run_extraction(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        qualitative_vocab_identifiers: list[str] | None = None,
        resume: bool = False,
    ) -> tuple[ExtractionRunResult | None, TaskStatus]:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None
        assert self.task_registry is not None

        self.datasource_service.get_data_package(data_package_id)
        self.profile_service.get_profile(profile_identifier)
        self.profile_service.load_json_schema(profile_identifier)

        chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        if not chunks_by_file:
            raise ChunkingRequiredError(
                "Extraction requires completed datasource chunking. Run chunking first."
            )

        task_name = self._extraction_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status == TaskStatus.RUNNING:
            return self._load_result_or_none(data_package_id), TaskStatus.RUNNING
        if task_info is not None and task_info.status == TaskStatus.COMPLETED:
            result = self._load_result_or_none(data_package_id)
            if result is not None:
                return result, TaskStatus.COMPLETED

        if not resume:
            self.output_repository.clear_extraction_run(data_package_id)
        await self.task_registry.create_task(
            coro=self._run_extraction_task(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                qualitative_vocab_identifiers=qualitative_vocab_identifiers,
                resume=resume,
            ),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        return None, TaskStatus.RUNNING

    async def run_complete_workflow(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        qualitative_vocab_identifiers: list[str] | None = None,
        buffer_window_size: int = 1,
        semantic_chunking_threshold: float = 95.0,
        replace_existing_chunks: bool = False,
        resume: bool = False,
        force_rerun: bool = False,
    ) -> TaskStatus:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None
        assert self.task_registry is not None

        self.datasource_service.get_data_package(data_package_id)
        self.profile_service.get_profile(profile_identifier)
        self.profile_service.load_json_schema(profile_identifier)

        task_name = self._complete_workflow_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status == TaskStatus.RUNNING:
            if force_rerun:
                raise ValueError("Cannot force-rerun a complete workflow while it is already running.")
            return TaskStatus.RUNNING
        if (
            not force_rerun
            and task_info is not None
            and task_info.status == TaskStatus.COMPLETED
        ):
            if self._load_result_or_none(data_package_id) is not None:
                return TaskStatus.COMPLETED
        if (
            task_info is not None
            and task_info.status == TaskStatus.CRASHED
            and not force_rerun
        ):
            exception = task_info.task.exception()
            raise exception if exception else Exception("Complete workflow task crashed without an exception.")

        if force_rerun:
            self.output_repository.clear_extraction_run(data_package_id)

        await self.task_registry.create_task(
            coro=self._run_complete_workflow_task(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                qualitative_vocab_identifiers=qualitative_vocab_identifiers,
                buffer_window_size=buffer_window_size,
                semantic_chunking_threshold=semantic_chunking_threshold,
                replace_existing_chunks=replace_existing_chunks,
                resume=resume,
            ),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        return TaskStatus.RUNNING

    async def _run_complete_workflow_task(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        qualitative_vocab_identifiers: list[str] | None,
        buffer_window_size: int,
        semantic_chunking_threshold: float,
        replace_existing_chunks: bool,
        resume: bool,
    ) -> None:
        assert self.datasource_service is not None
        assert self.task_registry is not None

        self._update_complete_workflow_progress(
            data_package_id=data_package_id,
            progress=CompleteWorkflowProgress(
                stage="chunking",
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                chunking_status=TaskStatus.RUNNING,
                extraction_status=TaskStatus.UNKNOWN,
                result_url=self._result_url(data_package_id),
            ),
        )
        _, chunk_status = await self.datasource_service.chunk_file_entries_in_data_package(
            data_package_id=data_package_id,
            buffer_window_size=buffer_window_size,
            semantic_chunking_threshold=semantic_chunking_threshold,
            replace_existing_chunks=replace_existing_chunks,
        )
        if chunk_status == TaskStatus.RUNNING:
            self._update_complete_workflow_progress(
                data_package_id=data_package_id,
                progress=CompleteWorkflowProgress(
                    stage="chunking",
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    chunking_status=TaskStatus.RUNNING,
                    extraction_status=TaskStatus.UNKNOWN,
                    result_url=self._result_url(data_package_id),
                ),
            )
            await self.task_registry.wait_for_task(
                self.datasource_service.chunk_task_name(data_package_id)
            )

        chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        if not chunks_by_file:
            raise ChunkingRequiredError(
                "Complete workflow could not continue because chunking produced no completed chunks."
            )

        self._update_complete_workflow_progress(
            data_package_id=data_package_id,
            progress=CompleteWorkflowProgress(
                stage="extraction",
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                chunking_status=TaskStatus.COMPLETED,
                extraction_status=TaskStatus.RUNNING,
                result_url=self._result_url(data_package_id),
            ),
        )
        _, extraction_status = await self.run_extraction(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            qualitative_vocab_identifiers=qualitative_vocab_identifiers,
            resume=resume,
        )
        if extraction_status == TaskStatus.RUNNING:
            await self.task_registry.wait_for_task(
                self._extraction_task_name(data_package_id)
            )
        workflow_progress = self._derive_complete_workflow_progress(
            data_package_id=data_package_id,
            workflow_status=TaskStatus.RUNNING,
        )
        if workflow_progress is not None:
            self._update_complete_workflow_progress(
                data_package_id=data_package_id,
                progress=workflow_progress,
            )

    async def get_complete_workflow_progress(
        self,
        *,
        data_package_id: str,
    ) -> tuple[TaskStatus, CompleteWorkflowProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None

        task_info = self.task_registry.get_task_info(
            self._complete_workflow_task_name(data_package_id)
        )
        if task_info is not None:
            saved_progress = (
                CompleteWorkflowProgress.model_validate(task_info.progress)
                if task_info.progress
                else None
            )
            progress = self._derive_complete_workflow_progress(
                data_package_id=data_package_id,
                workflow_status=task_info.status,
            )
            if progress is not None and saved_progress is not None:
                update: dict[str, Any] = {
                    "profile_identifier": progress.profile_identifier
                    or saved_progress.profile_identifier,
                }
                if (
                    progress.extraction_status == TaskStatus.UNKNOWN
                    and saved_progress.extraction_status != TaskStatus.UNKNOWN
                ):
                    update["extraction_status"] = saved_progress.extraction_status
                    if progress.stage == "extraction_pending":
                        update["stage"] = saved_progress.stage
                if (
                    progress.extraction_progress is None
                    and saved_progress.extraction_progress is not None
                ):
                    update["extraction_progress"] = saved_progress.extraction_progress
                progress = progress.model_copy(update=update)
            progress = progress or saved_progress
            if progress is None:
                return task_info.status, None
            return task_info.status, self._complete_workflow_progress_with_steps(
                progress,
                workflow_status=task_info.status,
            )

        progress = self._derive_complete_workflow_progress(
            data_package_id=data_package_id,
            workflow_status=TaskStatus.UNKNOWN,
        )
        if progress is None:
            return TaskStatus.UNKNOWN, None
        status = (
            TaskStatus.COMPLETED
            if progress.stage == "completed"
            else TaskStatus.UNKNOWN
        )
        return status, self._complete_workflow_progress_with_steps(
            progress,
            workflow_status=status,
        )

    async def get_extraction_progress(
        self,
        *,
        data_package_id: str,
    ) -> tuple[TaskStatus, ExtractionRunProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None
        task_info = self.task_registry.get_task_info(
            self._extraction_task_name(data_package_id)
        )
        if task_info is None:
            state = self._load_run_state_or_none(data_package_id)
            result = self._load_result_or_none(data_package_id)
            if result is not None:
                return TaskStatus.COMPLETED, ExtractionRunProgress(
                    stage="completed",
                    processed_chunks=self._completed_chunk_count(state) if state else 0,
                    total_chunks=len(state.chunk_results) if state else 0,
                    interim_context=result.extraction_context,
                    vocab_query_config=state.vocab_query_config if state else self._default_vocab_query_config(None),
                    ranked_files=state.ranked_files if state else [],
                    chunk_results=state.chunk_results if state else [],
                    vocab_queries=state.vocab_queries if state else [],
                    interim_profile_document=state.interim_profile_document if state else None,
                    profile_patch_results=state.profile_patch_results if state else [],
                    warnings=list(result.warnings),
                )
            interim_context = self._load_context_or_none(data_package_id)
            if interim_context is not None or state is not None:
                return TaskStatus.UNKNOWN, ExtractionRunProgress(
                    stage="interim_context",
                    processed_chunks=self._completed_chunk_count(state) if state else 0,
                    total_chunks=len(state.chunk_results) if state else 0,
                    interim_context=interim_context or (
                        self._merged_completed_chunk_context_or_none(state)
                        if state
                        else None
                    ),
                    vocab_query_config=state.vocab_query_config if state else self._default_vocab_query_config(None),
                    ranked_files=state.ranked_files if state else [],
                    chunk_results=state.chunk_results if state else [],
                    vocab_queries=state.vocab_queries if state else [],
                    interim_profile_document=state.interim_profile_document if state else None,
                    profile_patch_results=state.profile_patch_results if state else [],
                    warnings=self._load_warnings_or_empty(data_package_id),
                )
            return TaskStatus.UNKNOWN, None
        try:
            progress = (
                ExtractionRunProgress.model_validate(task_info.progress)
                if task_info.progress
                else None
            )
        except ValidationError:
            progress = None
        if progress is None:
            state = self._load_run_state_or_none(data_package_id)
            if state is not None:
                progress = ExtractionRunProgress(
                    stage="interim_context",
                    processed_chunks=self._completed_chunk_count(state),
                    total_chunks=len(state.chunk_results),
                    interim_context=self._merged_completed_chunk_context_or_none(state),
                    vocab_query_config=state.vocab_query_config,
                    ranked_files=state.ranked_files,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    interim_profile_document=state.interim_profile_document,
                    profile_patch_results=state.profile_patch_results,
                )
        if progress is not None and progress.interim_context is None:
            progress.interim_context = self._load_context_or_none(data_package_id)
        if progress is not None:
            state = self._load_run_state_or_none(data_package_id)
            if state is not None:
                progress.vocab_query_config = state.vocab_query_config
                if not progress.chunk_results:
                    progress.ranked_files = state.ranked_files
                    progress.chunk_results = state.chunk_results
                    progress.processed_chunks = self._completed_chunk_count(state)
                    progress.total_chunks = len(state.chunk_results)
                progress.vocab_queries = state.vocab_queries
                progress.interim_profile_document = state.interim_profile_document
                progress.profile_patch_results = state.profile_patch_results
        return task_info.status, progress

    async def pause_extraction(
        self,
        *,
        data_package_id: str,
    ) -> tuple[TaskStatus, ExtractionRunProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None

        task_name = self._extraction_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is None or task_info.status != TaskStatus.RUNNING:
            return await self.get_extraction_progress(data_package_id=data_package_id)

        await self.task_registry.cancel_task(task_name)

        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            return await self.get_extraction_progress(data_package_id=data_package_id)

        for chunk_result in state.chunk_results:
            if chunk_result.status == "running":
                chunk_result.status = "pending"
                chunk_result.error = None
        self._save_run_state(data_package_id, state)

        progress = ExtractionRunProgress(
            stage="paused",
            processed_chunks=self._completed_chunk_count(state),
            total_chunks=len(state.chunk_results),
            interim_context=self._merged_completed_chunk_context_or_none(state)
            or self._load_context_or_none(data_package_id),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            interim_profile_document=state.interim_profile_document,
            profile_patch_results=state.profile_patch_results,
            current_chunk=None,
            warnings=self._load_warnings_or_empty(data_package_id),
        )
        self._update_progress(data_package_id, progress)
        next_task_info = self.task_registry.get_task_info(task_name)
        return (
            next_task_info.status if next_task_info else TaskStatus.CANCELLED,
            progress,
        )

    async def get_extraction_result(
        self,
        *,
        data_package_id: str,
    ) -> ExtractionRunResult:
        if self.output_repository is None:
            raise ExtractionResultNotFoundError("Extraction output repository is unavailable.")
        try:
            return self.output_repository.load_extraction_result(data_package_id)
        except FileNotFoundError as exc:
            raise ExtractionResultNotFoundError(
                f"Extraction result not found for workflow '{data_package_id}'."
            ) from exc

    async def get_token_usage(self, data_package_id: str) -> dict[str, Any]:
        if self.output_repository is None:
            return {"agents": {}}
        return self._token_usage_summary(
            self.output_repository.load_token_usage(data_package_id)
        )

    async def update_vocab_query_config(
        self,
        *,
        data_package_id: str,
        config: ExtractionVocabQueryConfig,
    ) -> ExtractionRunProgress:
        if self.output_repository is None:
            raise ExtractionResultNotFoundError("Extraction output repository is unavailable.")
        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            raise ValueError(
                "Cannot update vocabulary query configuration because the extraction run state is missing or unreadable."
            )
        state.vocab_query_config = config
        self._save_run_state(data_package_id, state)
        return ExtractionRunProgress(
            stage="vocabulary_config_updated",
            processed_chunks=self._completed_chunk_count(state),
            total_chunks=len(state.chunk_results),
            interim_context=self._merged_completed_chunk_context_or_none(state),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            interim_profile_document=state.interim_profile_document,
            profile_patch_results=state.profile_patch_results,
            warnings=self._load_warnings_or_empty(data_package_id),
        )

    async def rerun_vocab_queries(
        self,
        *,
        data_package_id: str,
        query_id: str | None = None,
    ) -> ExtractionRunResult:
        self._require_runtime_dependencies()
        assert self.output_repository is not None
        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            raise ValueError(
                "Cannot rerun vocabulary queries because the extraction run state is missing or unreadable."
            )
        profile_identifier = state.profile_identifier
        if not profile_identifier:
            raise ValueError("Cannot rerun vocabulary queries because this extraction run has no profile identifier.")
        warnings = self._load_warnings_or_empty(data_package_id)
        records = [
            record
            for record in state.vocab_queries
            if query_id is None or record.query_id == query_id
        ]
        if query_id is not None and not records:
            raise ValueError(f"Vocabulary query '{query_id}' was not found.")
        for record in records:
            configured_query = self._configured_vocab_query(
                record.query,
                state.vocab_query_config,
                group="quantitative" if record.kind in {"quantity_kind", "unit"} else "qualitative",
            )
            record.query = configured_query
            record.rdf_type = configured_query.rdf_type
            await self._run_vocab_query_record(
                data_package_id=data_package_id,
                record=record,
                vocabulary_identifier=record.vocabulary_identifier,
                query=configured_query,
                on_progress=lambda: self._save_run_state(data_package_id, state),
                warnings=warnings,
            )
        self._save_run_state(data_package_id, state)
        return await self._rerun_vocab_downstream(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            state=state,
            warnings=warnings,
        )

    async def _run_extraction_task(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        qualitative_vocab_identifiers: list[str] | None,
        resume: bool = False,
    ) -> ExtractionRunResult:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.ollama_client is not None
        assert self.output_repository is not None

        data_package = self.datasource_service.get_data_package(data_package_id)
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
        validation_schema = validation_schema_for_target_class(
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        if not chunks_by_file:
            raise ChunkingRequiredError(
                "Extraction requires completed datasource chunking. Run chunking first."
            )

        warnings: list[str] = []
        persisted_state = (
            self._load_run_state_or_none(data_package_id)
            if resume
            else None
        )
        progress = ExtractionRunProgress(
            stage="file_ranking",
            total_chunks=sum(len(chunks) for chunks in chunks_by_file),
            ranked_files=persisted_state.ranked_files if persisted_state else [],
            chunk_results=persisted_state.chunk_results if persisted_state else [],
            vocab_queries=persisted_state.vocab_queries if persisted_state else [],
            interim_profile_document=(
                persisted_state.interim_profile_document if persisted_state else None
            ),
            profile_patch_results=(
                persisted_state.profile_patch_results if persisted_state else []
            ),
            vocab_query_config=(
                persisted_state.vocab_query_config
                if persisted_state
                else self._default_vocab_query_config(qualitative_vocab_identifiers)
            ),
        )
        self._update_progress(data_package_id, progress)

        ranking = (
            FileRankingResult(files=persisted_state.ranked_files)
            if persisted_state and persisted_state.ranked_files
            else await self._rank_files(data_package_id, data_package, warnings)
        )
        ordered_chunks = self._ordered_chunks(chunks_by_file, ranking)
        state = self._prepare_run_state(
            ranking=ranking,
            ordered_chunks=ordered_chunks,
            persisted_state=persisted_state,
            profile_identifier=profile_identifier,
            vocab_query_config=progress.vocab_query_config,
        )
        self._save_run_state(data_package_id, state)

        progress.ranked_files = state.ranked_files
        progress.chunk_results = state.chunk_results
        progress.vocab_query_config = state.vocab_query_config
        progress.vocab_queries = state.vocab_queries
        progress.interim_profile_document = state.interim_profile_document
        progress.profile_patch_results = state.profile_patch_results
        progress.total_chunks = len(state.chunk_results)
        progress.processed_chunks = self._completed_chunk_count(state)
        progress.interim_context = self._merged_completed_chunk_context_or_none(state)
        self._update_progress(data_package_id, progress)

        progress.stage = "chunk_extraction"
        chunk_repairs: list[tuple[ExtractionChunkResult, MaxRetriesExceeded]] = []

        try:
            for chunk_result, chunk in zip(state.chunk_results, ordered_chunks):
                if chunk_result.status == "completed" and chunk_result.extraction_context is not None:
                    continue

                chunk_result.status = "running"
                chunk_result.error = None
                progress.current_chunk = self._chunk_ref(chunk_result)
                progress.chunk_results = state.chunk_results
                self._save_run_state(data_package_id, state)
                self._update_progress(data_package_id, progress)

                try:
                    result = await generate_structured(
                        self.ollama_client,
                        model=self.ollama_client.chat_model,
                        system=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
                        prompt=build_extraction_context_prompt(
                            ChunkContext(
                                content=chunk.content,
                                metadata=ChunkMetadata(
                                    start_idx=chunk.start_idx,
                                    end_idx=chunk.end_idx,
                                    file_path=chunk.file_path,
                                    data_package_name=data_package.file_name,
                                    initial_extraction_context=self._initial_extraction_context_for_prompt(
                                        state,
                                        file_path=chunk.file_path,
                                        current_chunk_index=chunk_result.chunk_index,
                                    ),
                                ),
                            )
                        ),
                        output_type=ExtractionContext,
                        retries=0,
                        num_ctx=self.ollama_client.max_context_length,
                    )
                except MaxRetriesExceeded as exc:
                    logger.exception(
                        "Chunk extraction structured output failed",
                        extra={
                            "data_package_id": data_package_id,
                            "file_path": chunk_result.file_path,
                            "chunk_index": chunk_result.chunk_index,
                            "error_type": type(exc).__name__,
                            "repair_queued": bool(exc.failed_response),
                        },
                    )
                    self._record_workflow_token_usage(
                        data_package_id=data_package_id,
                        agent_name="chunk_extraction",
                        usage=exc.usage,
                    )
                    chunk_result.status = "failed"
                    chunk_result.error = (
                        "Queued for repair after first-pass extraction"
                        if exc.failed_response
                        else str(exc)
                    )
                    chunk_result.response_duration_ms = self._usage_float(
                        exc.usage,
                        "response_duration_ms",
                    )
                    chunk_result.context_tokens = self._usage_int(exc.usage, "input_tokens")
                    progress.current_chunk = None
                    progress.chunk_results = state.chunk_results
                    self._save_run_state(data_package_id, state)
                    self._update_progress(data_package_id, progress)
                    if exc.failed_response:
                        chunk_repairs.append((chunk_result, exc))
                    else:
                        warnings.append(
                            "Chunk extraction failed for "
                            f"{chunk_result.file_path} chunk {chunk_result.chunk_index}: {exc}"
                        )
                        progress.warnings = list(warnings)
                        self._save_run_state(data_package_id, state)
                        self._update_progress(data_package_id, progress)
                    continue
                except CompletionError as exc:
                    logger.exception(
                        "Chunk extraction completion failed",
                        extra={
                            "data_package_id": data_package_id,
                            "file_path": chunk_result.file_path,
                            "chunk_index": chunk_result.chunk_index,
                            "error_type": type(exc).__name__,
                        },
                    )
                    chunk_result.status = "failed"
                    chunk_result.error = str(exc)
                    warnings.append(
                        "Chunk extraction failed for "
                        f"{chunk_result.file_path} chunk {chunk_result.chunk_index}: {exc}"
                    )
                    progress.current_chunk = None
                    progress.chunk_results = state.chunk_results
                    progress.warnings = list(warnings)
                    self._save_run_state(data_package_id, state)
                    self._update_progress(data_package_id, progress)
                    continue
                except Exception as exc:
                    chunk_result.status = "failed"
                    chunk_result.error = str(exc)
                    progress.current_chunk = None
                    progress.chunk_results = state.chunk_results
                    self._save_run_state(data_package_id, state)
                    self._update_progress(data_package_id, progress)
                    raise

                self._record_workflow_token_usage(
                    data_package_id=data_package_id,
                    agent_name="chunk_extraction",
                    usage=result.usage,
                )
                chunk_result.status = "completed"
                chunk_result.extraction_context = result.output
                chunk_result.response_duration_ms = self._usage_float(
                    result.usage,
                    "response_duration_ms",
                )
                chunk_result.context_tokens = self._usage_int(result.usage, "input_tokens")
                self._save_run_state(data_package_id, state)

                partial_context = self._merged_completed_chunk_context(state)
                self.output_repository.save_extraction_context(
                    workflow_id=data_package_id,
                    extraction_context=partial_context,
                )
                progress.processed_chunks = self._completed_chunk_count(state)
                progress.interim_context = partial_context
                progress.current_chunk = None
                progress.chunk_results = state.chunk_results
                self._update_progress(data_package_id, progress)

            if chunk_repairs:
                progress.stage = "chunk_repair"
                self._update_progress(data_package_id, progress)
        except asyncio.CancelledError:
            raise

        for chunk_result, failure in chunk_repairs:
            chunk_result.status = "running"
            chunk_result.error = None
            progress.current_chunk = self._chunk_ref(chunk_result)
            progress.chunk_results = state.chunk_results
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)

            try:
                repair = await repair_structured_output(
                    self.ollama_client,
                    model=self.ollama_client.chat_model,
                    failed_response=failure.failed_response or "",
                    error=failure.last_error or failure,
                    output_type=ExtractionContext,
                    num_ctx=self.ollama_client.max_context_length,
                )
            except CompletionError as exc:
                usage = getattr(exc, "usage", None)
                if usage is not None:
                    self._record_workflow_token_usage(
                        data_package_id=data_package_id,
                        agent_name="chunk_extraction_repair",
                        usage=usage,
                    )
                chunk_result.status = "failed"
                chunk_result.error = str(exc)
                warnings.append(
                    "Chunk extraction repair failed for "
                    f"{chunk_result.file_path} chunk {chunk_result.chunk_index}: {exc}"
                )
                progress.current_chunk = None
                progress.chunk_results = state.chunk_results
                progress.warnings = list(warnings)
                self._save_run_state(data_package_id, state)
                self._update_progress(data_package_id, progress)
                continue

            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="chunk_extraction_repair",
                usage=repair.usage,
            )
            chunk_result.status = "completed"
            chunk_result.extraction_context = repair.output
            chunk_result.response_duration_ms = self._usage_float(
                repair.usage,
                "response_duration_ms",
            )
            chunk_result.context_tokens = self._usage_int(repair.usage, "input_tokens")
            self._save_run_state(data_package_id, state)

            partial_context = self._merged_completed_chunk_context(state)
            self.output_repository.save_extraction_context(
                workflow_id=data_package_id,
                extraction_context=partial_context,
            )
            progress.processed_chunks = self._completed_chunk_count(state)
            progress.interim_context = partial_context
            progress.current_chunk = None
            progress.chunk_results = state.chunk_results
            self._update_progress(data_package_id, progress)

        failed_chunks = [
            chunk_result
            for chunk_result in state.chunk_results
            if chunk_result.status == "failed"
        ]
        if failed_chunks:
            warnings.append(
                "Chunk extraction completed with failed chunks: "
                + "; ".join(
                    f"{chunk.file_path} chunk {chunk.chunk_index}: {chunk.error or 'unknown error'}"
                    for chunk in failed_chunks
                )
            )
            progress.warnings = list(warnings)
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)

        extraction_context = self._context_with_resource_inventory(
            data_package=data_package,
            context=self._merged_completed_chunk_context(state),
        )
        self.output_repository.save_extraction_context(
            workflow_id=data_package_id,
            extraction_context=extraction_context,
        )

        progress.stage = "profile_projection"
        progress.interim_context = extraction_context
        self._update_progress(data_package_id, progress)

        profile_document = await self._build_profile_document_by_patching(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_manifest.target_class,
            extraction_context=extraction_context,
            validation_schema=validation_schema,
            state=state,
            progress=progress,
            warnings=warnings,
        )

        progress.stage = "vocabulary_normalization"
        progress.interim_context = extraction_context
        progress.interim_profile_document = profile_document
        progress.profile_patch_results = state.profile_patch_results
        progress.vocab_queries = state.vocab_queries
        self._update_progress(data_package_id, progress)
        vocab_query_semaphore = asyncio.Semaphore(self._vocab_query_concurrency())

        def persist_vocab_progress() -> None:
            progress.chunk_results = state.chunk_results
            progress.vocab_query_config = state.vocab_query_config
            progress.vocab_queries = state.vocab_queries
            progress.interim_profile_document = state.interim_profile_document
            progress.profile_patch_results = state.profile_patch_results
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)

        candidate_tasks = [
            asyncio.create_task(
                self._discover_profile_field_candidates(
                    json_path=json_path,
                    field_name=field_name,
                    source_value=source_value,
                    state=state,
                    data_package_id=data_package_id,
                    query_semaphore=vocab_query_semaphore,
                    on_progress=persist_vocab_progress,
                    warnings=warnings,
                )
            )
            for json_path, field_name, source_value in self._profile_vocab_sources(
                profile_document,
                enrichable_fields=getattr(profile_manifest, "enrichable_fields", []),
            )
        ]
        try:
            normalization = await self._normalize_profile_field_candidate_tasks(
                data_package_id=data_package_id,
                state=state,
                candidate_tasks=candidate_tasks,
                warnings=warnings,
            )
        except asyncio.CancelledError:
            await self._cancel_candidate_tasks(candidate_tasks)
            raise
        progress.normalized_quantities = len(
            [
                item
                for item in normalization.profile_fields
                if item.field_name in {"has_quantity_type", "unit"}
            ]
        )
        progress.normalized_qualitative_attributes = len(normalization.profile_fields)
        progress.warnings = list(warnings)
        progress.vocab_queries = state.vocab_queries
        self._update_progress(data_package_id, progress)

        result = await self._save_validated_profile_result(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            extraction_context=extraction_context,
            normalization=normalization,
            document=profile_document,
            warnings=warnings,
        )
        progress.stage = "completed"
        progress.interim_context = extraction_context
        progress.interim_profile_document = profile_document
        progress.profile_patch_results = state.profile_patch_results
        progress.warnings = warnings
        self._update_progress(data_package_id, progress)
        return result

    async def _rank_files(
        self,
        data_package_id: str,
        data_package: Any,
        warnings: list[str],
    ) -> FileRankingResult:
        _ = data_package_id, warnings
        files = [
            FileContext(file_path=file.file_path, byte_size=len(file.raw_content))
            for file in data_package.files
        ]
        return fallback_file_ranking(files)

    async def _rerun_vocab_downstream(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        state: ExtractionRunState,
        warnings: list[str],
    ) -> ExtractionRunResult:
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
        validation_schema = validation_schema_for_target_class(
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        extraction_context = self._merged_completed_chunk_context(state)
        profile_document = state.interim_profile_document or self._fallback_profile_document(
            data_package_id=data_package_id,
            extraction_context=extraction_context,
            validation_schema=validation_schema,
        )
        normalization = await self._normalize_profile_fields_from_state_vocab_queries(
            data_package_id=data_package_id,
            state=state,
            warnings=warnings,
        )
        result = await self._save_validated_profile_result(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            extraction_context=extraction_context,
            normalization=normalization,
            document=profile_document,
            warnings=warnings,
        )
        if self.task_registry is not None:
            self.task_registry.update_progress(
                self._extraction_task_name(data_package_id),
                ExtractionRunProgress(
                    stage="completed",
                    processed_chunks=self._completed_chunk_count(state),
                    total_chunks=len(state.chunk_results),
                    normalized_quantities=len(normalization.quantities),
                    normalized_qualitative_attributes=len(normalization.qualitative_attributes),
                    interim_context=extraction_context,
                    vocab_query_config=state.vocab_query_config,
                    ranked_files=state.ranked_files,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    interim_profile_document=profile_document,
                    profile_patch_results=state.profile_patch_results,
                    warnings=warnings,
                ).model_dump(mode="json"),
            )
        return result

    async def _build_profile_document_by_patching(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        extraction_context: ExtractionContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> dict[str, Any]:
        document = state.interim_profile_document or self._fallback_profile_document(
            data_package_id=data_package_id,
            extraction_context=extraction_context,
            validation_schema=validation_schema,
        )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        if not validation.valid:
            warnings.append(
                "Initial profile skeleton was not schema-valid: "
                + "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
            )

        schema_slice = self._profile_schema_slice(validation_schema, max_depth=2)
        patched_identifiers = {
            result.object_identifier
            for result in state.profile_patch_results
            if result.status == "applied"
        }
        for trace in extraction_context.extraction_objects:
            object_identifier = trace.extracted_object.identifier
            if object_identifier in patched_identifiers:
                continue
            patch_result = await self._patch_profile_with_extraction_object(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                profile_target_class=profile_target_class,
                current_document=document,
                extraction_object=trace,
                schema_slice=schema_slice,
                warnings=warnings,
            )
            state.profile_patch_results.append(patch_result)
            if patch_result.status == "applied":
                document = self._apply_profile_patch(
                    document,
                    patch_result.operations,
                )
            state.interim_profile_document = document
            progress.interim_profile_document = document
            progress.profile_patch_results = state.profile_patch_results
            progress.warnings = list(warnings)
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)
        return document

    async def _patch_profile_with_extraction_object(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        current_document: dict[str, Any],
        extraction_object: TracedExtractionObject,
        schema_slice: dict[str, Any],
        warnings: list[str],
    ) -> ProfileObjectPatchResult:
        assert self.ollama_client is not None
        object_identifier = extraction_object.extracted_object.identifier
        object_kind = extraction_object.object_kind
        try:
            patch = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=PROFILE_PATCH_SYSTEM_PROMPT,
                prompt=build_profile_patch_prompt(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    profile_target_class=profile_target_class,
                    current_document=current_document,
                    extraction_object=extraction_object,
                    schema_slice=schema_slice,
                ),
                output_type=ProfilePatchDocument,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="profile_patch",
                usage=patch.usage,
            )
        except CompletionError as exc:
            warnings.append(f"Profile patch failed for '{object_identifier}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=object_identifier,
                object_kind=object_kind,
                status="failed",
                error=str(exc),
            )

        patch_document = (
            patch.output
            if isinstance(patch.output, ProfilePatchDocument)
            else ProfilePatchDocument.model_validate(patch.output)
        )
        operations = patch_document.operations
        if not operations:
            return ProfileObjectPatchResult(
                object_identifier=object_identifier,
                object_kind=object_kind,
                status="skipped",
                reason=patch_document.reason,
            )
        try:
            candidate = self._apply_profile_patch(current_document, operations)
        except Exception as exc:
            warnings.append(f"Profile patch skipped for '{object_identifier}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=object_identifier,
                object_kind=object_kind,
                status="failed",
                operations=operations,
                error=str(exc),
                reason=patch_document.reason,
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=candidate,
        )
        if not validation.valid:
            error = "; ".join(
                f"{issue.path}: {issue.message}" for issue in validation.errors
            )
            warnings.append(
                f"Profile patch skipped for '{object_identifier}' because it broke schema validation: {error}"
            )
            return ProfileObjectPatchResult(
                object_identifier=object_identifier,
                object_kind=object_kind,
                status="failed",
                operations=operations,
                error=error,
                reason=patch_document.reason,
            )
        return ProfileObjectPatchResult(
            object_identifier=object_identifier,
            object_kind=object_kind,
            status="applied",
            operations=operations,
            reason=patch_document.reason,
        )

    @staticmethod
    def _apply_profile_patch(
        document: dict[str, Any],
        operations: list[Any],
    ) -> dict[str, Any]:
        patch_ops = [
            operation.model_dump(mode="json", exclude_none=True)
            if hasattr(operation, "model_dump")
            else operation
            for operation in operations
        ]
        return jsonpatch.JsonPatch(patch_ops).apply(document, in_place=False)

    async def _save_validated_profile_result(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        extraction_context: ExtractionContext,
        normalization: ExtractionNormalization,
        document: dict[str, Any],
        warnings: list[str],
    ) -> ExtractionRunResult:
        assert self.output_repository is not None
        clean_document = remove_null_values(document)
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=clean_document,
        )
        if not validation.valid:
            raise ExtractionValidationError(
                [f"{issue.path}: {issue.message}" for issue in validation.errors]
            )
        token_usage = await self.get_token_usage(data_package_id)
        result = ExtractionRunResult(
            document=clean_document,
            extraction_context=extraction_context,
            normalization=normalization,
            warnings=warnings,
            token_usage=token_usage,
        )
        self.output_repository.save_extraction_warnings(
            workflow_id=data_package_id,
            warnings=warnings,
        )
        self.output_repository.save_extraction_result(
            workflow_id=data_package_id,
            result=result,
        )
        return result

    @classmethod
    def _fallback_profile_document(
        cls,
        *,
        data_package_id: str,
        extraction_context: ExtractionContext,
        validation_schema: dict[str, Any],
    ) -> dict[str, Any]:
        properties = cls._target_schema_properties(validation_schema)
        title = cls._fallback_title(data_package_id, extraction_context)
        document: dict[str, Any] = {
            "title": cls._fallback_property_value(properties.get("title"), title),
        }
        if "description" in properties:
            document["description"] = cls._fallback_property_value(
                properties.get("description"),
                f"SIMONE metadata draft for {title}.",
            )
        if "identifier" in properties:
            document["identifier"] = cls._fallback_property_value(
                properties.get("identifier"),
                data_package_id,
            )
        if "id" in properties:
            document["id"] = data_package_id
        if "was_generated_by" in properties:
            document["was_generated_by"] = [
                {"id": f"{data_package_id}:activity:metadata-extraction"}
            ]
        return document

    @staticmethod
    def _fallback_property_value(schema: Any, value: str) -> Any:
        if isinstance(schema, dict):
            schema_type = schema.get("type")
            if schema_type == "array" or (
                isinstance(schema_type, list) and "array" in schema_type
            ):
                return [value]
        return value

    @classmethod
    def _profile_schema_slice(
        cls,
        validation_schema: dict[str, Any],
        *,
        max_depth: int,
    ) -> dict[str, Any]:
        target = cls._resolve_schema_node(validation_schema, validation_schema)
        return cls._compact_schema_node(
            target,
            validation_schema,
            depth=max_depth,
        )

    @classmethod
    def _compact_schema_node(
        cls,
        node: Any,
        root: dict[str, Any],
        *,
        depth: int,
    ) -> Any:
        node = cls._resolve_schema_node(node, root)
        if not isinstance(node, dict):
            return node
        compact: dict[str, Any] = {}
        for key in ("type", "required", "enum", "const", "description"):
            if key in node:
                compact[key] = node[key]
        if "anyOf" in node or "oneOf" in node:
            union_key = "anyOf" if "anyOf" in node else "oneOf"
            compact[union_key] = [
                cls._compact_schema_node(option, root, depth=max(0, depth - 1))
                for option in node.get(union_key, [])
            ]
        if "items" in node:
            compact["items"] = cls._compact_schema_node(
                node["items"],
                root,
                depth=max(0, depth - 1),
            )
        if depth > 0 and "properties" in node:
            compact["properties"] = {
                key: cls._compact_schema_node(value, root, depth=depth - 1)
                for key, value in node.get("properties", {}).items()
            }
        elif "properties" in node:
            compact["properties"] = sorted(node.get("properties", {}).keys())
        return compact

    @staticmethod
    def _resolve_schema_node(node: Any, root: dict[str, Any]) -> Any:
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
        return node

    @staticmethod
    def _target_schema_properties(validation_schema: dict[str, Any]) -> dict[str, Any]:
        ref = validation_schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = ref.removeprefix("#/$defs/")
            target_schema = validation_schema.get("$defs", {}).get(target, {})
            if isinstance(target_schema, dict):
                properties = target_schema.get("properties", {})
                return properties if isinstance(properties, dict) else {}
        properties = validation_schema.get("properties", {})
        return properties if isinstance(properties, dict) else {}

    @staticmethod
    def _fallback_title(
        data_package_id: str,
        extraction_context: ExtractionContext,
    ) -> str:
        for object_kind in ("EvaluatedEntity", "Resource", "DataGeneratingActivity", "Method"):
            for trace in extraction_context.extraction_objects:
                if trace.object_kind != object_kind:
                    continue
                identifier = getattr(trace.extracted_object, "identifier", None)
                if identifier:
                    return str(identifier)
                description = getattr(trace.extracted_object, "description", None)
                if description:
                    return str(description)
        return f"SIMONE extraction result for {data_package_id}"

    @classmethod
    def _profile_vocab_sources(
        cls,
        document: dict[str, Any],
        *,
        enrichable_fields: list[str],
    ) -> list[tuple[str, str, str]]:
        target_fields = {"has_quantity_type", "unit"} | set(enrichable_fields)
        sources: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        def collect_scalar_values(value: Any, path: str) -> list[tuple[str, str]]:
            if isinstance(value, str):
                stripped = value.strip()
                return [(path, stripped)] if stripped else []
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return [(path, str(value))]
            if isinstance(value, list):
                collected: list[tuple[str, str]] = []
                for index, item in enumerate(value):
                    collected.extend(collect_scalar_values(item, f"{path}/{index}"))
                return collected
            if isinstance(value, dict):
                collected = []
                for key, item in value.items():
                    collected.extend(
                        collect_scalar_values(item, f"{path}/{cls._json_pointer_escape(key)}")
                    )
                return collected
            return []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    item_path = f"{path}/{cls._json_pointer_escape(key)}"
                    if key in target_fields:
                        for scalar_path, scalar in collect_scalar_values(item, item_path):
                            record = (scalar_path, key, scalar)
                            if record not in seen:
                                seen.add(record)
                                sources.append(record)
                    walk(item, item_path)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}/{index}")

        walk(document, "")
        return sources

    @staticmethod
    def _json_pointer_escape(value: str) -> str:
        return value.replace("~", "~0").replace("/", "~1")

    @staticmethod
    def _ordered_chunks(
        chunks_by_file: list[list[ContentChunk]],
        ranking: FileRankingResult,
    ) -> list[ContentChunk]:
        rank_by_path = {
            item.file_path: item.rank
            for item in ranking.files
        }
        chunks = [chunk for group in chunks_by_file for chunk in group]
        return sorted(
            chunks,
            key=lambda chunk: (
                rank_by_path.get(chunk.file_path, 10_000),
                chunk.file_path,
                chunk.start_idx,
            ),
        )

    @classmethod
    def _prepare_run_state(
        cls,
        *,
        ranking: FileRankingResult,
        ordered_chunks: list[ContentChunk],
        persisted_state: ExtractionRunState | None,
        profile_identifier: str,
        vocab_query_config: ExtractionVocabQueryConfig,
    ) -> ExtractionRunState:
        persisted_by_key = {
            cls._chunk_result_key(result): result
            for result in (persisted_state.chunk_results if persisted_state else [])
        }
        chunk_results: list[ExtractionChunkResult] = []
        for index, chunk in enumerate(ordered_chunks):
            existing = persisted_by_key.get(cls._chunk_key(chunk))
            if existing and existing.status == "completed" and existing.extraction_context is not None:
                chunk_results.append(
                    existing.model_copy(
                        update={
                            "chunk_index": index,
                            "status": "completed",
                            "error": None,
                        }
                    )
                )
                continue

            chunk_results.append(
                ExtractionChunkResult(
                    chunk_index=index,
                    file_path=chunk.file_path,
                    start_idx=chunk.start_idx,
                    end_idx=chunk.end_idx,
                    status="pending",
                )
            )

        return ExtractionRunState(
            profile_identifier=profile_identifier,
            vocab_query_config=(
                persisted_state.vocab_query_config
                if persisted_state
                else vocab_query_config
            ),
            ranked_files=ranking.files,
            chunk_results=chunk_results,
            vocab_queries=persisted_state.vocab_queries if persisted_state else [],
            interim_profile_document=(
                persisted_state.interim_profile_document if persisted_state else None
            ),
            profile_patch_results=(
                persisted_state.profile_patch_results if persisted_state else []
            ),
        )

    @staticmethod
    def _chunk_key(chunk: ContentChunk) -> tuple[str, int, int]:
        return (chunk.file_path, chunk.start_idx, chunk.end_idx)

    @staticmethod
    def _chunk_result_key(result: ExtractionChunkResult) -> tuple[str, int, int]:
        return (result.file_path, result.start_idx, result.end_idx)

    @staticmethod
    def _chunk_ref(result: ExtractionChunkResult) -> ExtractionChunkRef:
        return ExtractionChunkRef(
            chunk_index=result.chunk_index,
            file_path=result.file_path,
            start_idx=result.start_idx,
            end_idx=result.end_idx,
        )

    @staticmethod
    def _completed_chunk_contexts(
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> list[ExtractionContext]:
        return [
            result.extraction_context
            for result in state.chunk_results
            if result.status == "completed"
            and result.extraction_context is not None
            and (file_path is None or result.file_path == file_path)
        ]

    @classmethod
    def _completed_chunk_count(cls, state: ExtractionRunState) -> int:
        return len(cls._completed_chunk_contexts(state))

    @classmethod
    def _merged_completed_chunk_context(
        cls,
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> ExtractionContext:
        return merge_extraction_context_results(
            cls._completed_chunk_contexts(state, file_path=file_path)
        )

    @staticmethod
    def _context_with_resource_inventory(
        *,
        data_package: Any,
        context: ExtractionContext,
    ) -> ExtractionContext:
        existing_resource_ids = {
            trace.extracted_object.identifier
            for trace in context.extraction_objects
            if trace.object_kind == "Resource"
            and isinstance(trace.extracted_object, Resource)
        }
        inventory_objects = []
        for file in data_package.files:
            if file.file_path in existing_resource_ids:
                continue
            inventory_objects.append(
                TracedExtractionObject(
                    object_kind="Resource",
                    extracted_object=Resource(
                        identifier=file.file_path,
                        type=getattr(getattr(file, "file_type", None), "value", "file"),
                        description=(
                            f"Package file '{file.file_path}' "
                            f"({len(file.raw_content)} bytes, extension {file.file_extension})."
                        ),
                    ),
                    source_text=file.file_path,
                )
            )
        return context.model_copy(
            update={
                "extraction_objects": [
                    *context.extraction_objects,
                    *inventory_objects,
                ]
            }
        )


    @classmethod
    def _merged_completed_chunk_context_or_none(
        cls,
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> ExtractionContext | None:
        contexts = cls._completed_chunk_contexts(state, file_path=file_path)
        if not contexts:
            return None
        return merge_extraction_context_results(contexts)

    def _initial_extraction_context_for_prompt(
        self,
        state: ExtractionRunState,
        *,
        file_path: str,
        current_chunk_index: int,
    ) -> ExtractionContext | None:
        context_results = self._completed_chunk_results(
            state,
            file_path=file_path,
            before_chunk_index=current_chunk_index,
        )
        if not context_results:
            return None
        context = merge_extraction_context_results(
            [
                result.extraction_context
                for result in context_results
                if result.extraction_context is not None
            ]
        )
        previous_result = self._latest_completed_chunk_result_with_tokens(
            state,
            file_path=file_path,
            before_chunk_index=current_chunk_index,
        )
        threshold = self._initial_extraction_context_token_threshold()
        if (
            previous_result is None
            or previous_result.context_tokens is None
            or previous_result.context_tokens <= threshold
        ):
            return context

        previous_context_results = self._completed_chunk_results(
            state,
            file_path=file_path,
            before_chunk_index=previous_result.chunk_index,
        )
        if not previous_context_results:
            return context
        previous_context = merge_extraction_context_results(
            [
                result.extraction_context
                for result in previous_context_results
                if result.extraction_context is not None
            ]
        )
        previous_context_chars = len(previous_context.model_dump_json())
        if previous_context_chars <= 0:
            return context

        target_chars = int(
            previous_context_chars
            * threshold
            / max(1, previous_result.context_tokens)
        )
        return cap_extraction_context_for_prompt(
            context,
            max_json_chars=target_chars,
        )

    @staticmethod
    def _completed_chunk_results(
        state: ExtractionRunState,
        *,
        file_path: str,
        before_chunk_index: int,
    ) -> list[ExtractionChunkResult]:
        return [
            result
            for result in state.chunk_results
            if result.status == "completed"
            and result.extraction_context is not None
            and result.file_path == file_path
            and result.chunk_index < before_chunk_index
        ]

    @classmethod
    def _latest_completed_chunk_result_with_tokens(
        cls,
        state: ExtractionRunState,
        *,
        file_path: str,
        before_chunk_index: int,
    ) -> ExtractionChunkResult | None:
        results = [
            result
            for result in cls._completed_chunk_results(
                state,
                file_path=file_path,
                before_chunk_index=before_chunk_index,
            )
            if result.context_tokens is not None
        ]
        return max(results, key=lambda result: result.chunk_index, default=None)

    def _initial_extraction_context_token_threshold(self) -> int:
        configured_threshold = getattr(
            self.settings,
            "initial_extraction_context_token_threshold",
            None,
        )
        if configured_threshold is not None:
            return max(0, int(configured_threshold))
        max_context_length = (
            getattr(self.ollama_client, "max_context_length", None)
            or getattr(self.settings, "max_context_length", 8192)
        )
        return max(1, int(max_context_length * 0.75))

    async def _normalize_from_candidate_tasks(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        candidate_tasks: list[asyncio.Task[_QuantityCandidateDiscovery | _QualitativeCandidateDiscovery | _ObjectGroundingCandidateDiscovery]],
        extraction_context: ExtractionContext,
        warnings: list[str],
    ) -> ExtractionNormalization:
        if not candidate_tasks:
            return ExtractionNormalization()
        discoveries = await asyncio.gather(*candidate_tasks)
        selection_semaphore = asyncio.Semaphore(self._vocab_selection_llm_concurrency())
        quantity_discoveries = [
            item for item in discoveries if isinstance(item, _QuantityCandidateDiscovery)
        ]
        qualitative_discoveries = [
            item for item in discoveries if isinstance(item, _QualitativeCandidateDiscovery)
        ]
        object_grounding_discoveries = [
            item for item in discoveries if isinstance(item, _ObjectGroundingCandidateDiscovery)
        ]
        trace_by_identifier: dict[str, TracedExtractionObject] = {
            trace.extracted_object.identifier: trace
            for trace in extraction_context.extraction_objects
        }

        if self._vocab_selection_parallel_enabled():
            quantity_results = await asyncio.gather(
                *[
                    self._normalize_quantity_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                    for discovery in quantity_discoveries
                ]
            )
            qualitative_results = await asyncio.gather(
                *[
                    self._normalize_qualitative_attribute_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                    for discovery in qualitative_discoveries
                ]
            )
            object_grounding_results = await asyncio.gather(
                *[
                    self._normalize_object_grounding_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        trace=trace_by_identifier.get(
                            discovery.object_identifier,
                            TracedExtractionObject(
                                object_kind=discovery.object_kind,
                                extracted_object=Resource(
                                    identifier=discovery.object_identifier,
                                    description=discovery.source_context.get("description", ""),
                                ),
                                source_text=discovery.source_context.get("description", ""),
                            ),
                        ),
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                    for discovery in object_grounding_discoveries
                ]
            )
        else:
            quantity_results = []
            for discovery in quantity_discoveries:
                quantity_results.append(
                    await self._normalize_quantity_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                )
            qualitative_results = []
            for discovery in qualitative_discoveries:
                qualitative_results.append(
                    await self._normalize_qualitative_attribute_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                )
            object_grounding_results = []
            for discovery in object_grounding_discoveries:
                object_grounding_results.append(
                    await self._normalize_object_grounding_from_candidates(
                        data_package_id=data_package_id,
                        state=state,
                        discovery=discovery,
                        trace=trace_by_identifier.get(
                            discovery.object_identifier,
                            TracedExtractionObject(
                                object_kind=discovery.object_kind,
                                extracted_object=Resource(
                                    identifier=discovery.object_identifier,
                                    description=discovery.source_context.get("description", ""),
                                ),
                                source_text=discovery.source_context.get("description", ""),
                            ),
                        ),
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
                )

        return ExtractionNormalization(
            quantities=quantity_results,
            qualitative_attributes=qualitative_results,
            object_groundings=object_grounding_results,
        )

    async def _normalize_from_state_vocab_queries(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        extraction_context: ExtractionContext,
        warnings: list[str],
    ) -> ExtractionNormalization:
        quantity_groups: dict[str, dict[str, Any]] = {}
        qualitative_discoveries: list[_QualitativeCandidateDiscovery] = []
        object_grounding_groups: dict[str, dict[str, Any]] = {}
        for record in state.vocab_queries:
            if record.kind in {"quantity_kind", "unit"}:
                key = repr(sorted(record.source_context.items()))
                group = quantity_groups.setdefault(
                    key,
                    {"source_context": record.source_context, "quantity_kind": None, "unit": None},
                )
                group[record.kind] = record.query_id
            elif record.kind == "qualitative_attribute":
                try:
                    attribute = QualitativeAttribute.model_validate(record.source_context)
                except Exception:
                    continue
                existing = next(
                    (
                        discovery
                        for discovery in qualitative_discoveries
                        if discovery.attribute == attribute
                    ),
                    None,
                )
                if existing is None:
                    qualitative_discoveries.append(
                        _QualitativeCandidateDiscovery(
                            attribute=attribute,
                            query_ids=[record.query_id],
                        )
                    )
                else:
                    existing.query_ids.append(record.query_id)
            elif record.kind == "object_grounding":
                source_context = dict(record.source_context)
                identifier = str(source_context.get("identifier", ""))
                if not identifier:
                    continue
                group = object_grounding_groups.setdefault(
                    identifier,
                    {
                        "object_kind": str(source_context.get("object_kind", "unknown")),
                        "raw_type": str(source_context.get("type", "")),
                        "source_context": source_context,
                        "query_ids": [],
                    },
                )
                group["query_ids"].append(record.query_id)

        quantity_discoveries: list[_QuantityCandidateDiscovery] = []
        for group in quantity_groups.values():
            try:
                quantity = QuantitativeAttribute.model_validate(group["source_context"])
            except Exception:
                continue
            if group.get("quantity_kind") and group.get("unit"):
                quantity_discoveries.append(
                    _QuantityCandidateDiscovery(
                        quantity=quantity,
                        quantity_kind_query_id=group["quantity_kind"],
                        unit_query_id=group["unit"],
                    )
                )

        object_grounding_discoveries: list[_ObjectGroundingCandidateDiscovery] = [
            _ObjectGroundingCandidateDiscovery(
                object_identifier=identifier,
                object_kind=group["object_kind"],
                raw_type=group["raw_type"],
                source_context=group["source_context"],
                query_ids=group["query_ids"],
            )
            for identifier, group in object_grounding_groups.items()
        ]
        trace_by_identifier: dict[str, TracedExtractionObject] = {
            trace.extracted_object.identifier: trace
            for trace in extraction_context.extraction_objects
        }

        selection_semaphore = asyncio.Semaphore(self._vocab_selection_llm_concurrency())
        quantity_results = [
            await self._normalize_quantity_from_candidates(
                data_package_id=data_package_id,
                state=state,
                discovery=discovery,
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
            for discovery in quantity_discoveries
        ]
        qualitative_results = [
            await self._normalize_qualitative_attribute_from_candidates(
                data_package_id=data_package_id,
                state=state,
                discovery=discovery,
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
            for discovery in qualitative_discoveries
        ]
        object_grounding_results = [
            await self._normalize_object_grounding_from_candidates(
                data_package_id=data_package_id,
                state=state,
                discovery=discovery,
                trace=trace_by_identifier.get(
                    discovery.object_identifier,
                    TracedExtractionObject(
                        object_kind=discovery.object_kind,
                        extracted_object=Resource(
                            identifier=discovery.object_identifier,
                            description=discovery.source_context.get("description", ""),
                        ),
                        source_text=discovery.source_context.get("description", ""),
                    ),
                ),
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
            for discovery in object_grounding_discoveries
        ]
        return ExtractionNormalization(
            quantities=quantity_results,
            qualitative_attributes=qualitative_results,
            object_groundings=object_grounding_results,
        )

    async def _discover_profile_field_candidates(
        self,
        *,
        json_path: str,
        field_name: str,
        source_value: str,
        state: ExtractionRunState,
        data_package_id: str,
        query_semaphore: asyncio.Semaphore,
        on_progress: Any,
        warnings: list[str],
    ) -> _ProfileFieldCandidateDiscovery:
        query_ids: list[str] = []
        source_context = {
            "json_path": json_path,
            "field_name": field_name,
            "source_value": source_value,
        }
        if field_name == "has_quantity_type":
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type="qudt__QuantityKind",
                    vector_query=source_value,
                    fulltext_query=source_value,
                    vector_top_k=12,
                    fulltext_top_k=12,
                    seed_top_k=6,
                    max_hops=0,
                ),
                state.vocab_query_config,
                group="quantitative",
            )
            record = self._ensure_run_vocab_query_record(
                state=state,
                kind="profile_has_quantity_type",
                source_value=source_value,
                source_context=source_context,
                vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
                query=query,
            )
            query_ids.append(record.query_id)
            on_progress()
            async with query_semaphore:
                await self._run_vocab_query_record(
                    data_package_id=data_package_id,
                    record=record,
                    vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
                    query=query,
                    on_progress=on_progress,
                    warnings=warnings,
                )
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
                query_ids=query_ids,
            )
        if field_name == "unit":
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type="qudt__Unit",
                    vector_query=source_value,
                    fulltext_query=source_value,
                    vector_top_k=12,
                    fulltext_top_k=12,
                    seed_top_k=6,
                    max_hops=0,
                ),
                state.vocab_query_config,
                group="quantitative",
            )
            record = self._ensure_run_vocab_query_record(
                state=state,
                kind="profile_unit",
                source_value=source_value,
                source_context=source_context,
                vocabulary_identifier=QUDT_UNIT_VOCAB,
                query=query,
            )
            query_ids.append(record.query_id)
            on_progress()
            async with query_semaphore:
                await self._run_vocab_query_record(
                    data_package_id=data_package_id,
                    record=record,
                    vocabulary_identifier=QUDT_UNIT_VOCAB,
                    query=query,
                    on_progress=on_progress,
                    warnings=warnings,
                )
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                vocabulary_identifier=QUDT_UNIT_VOCAB,
                query_ids=query_ids,
            )

        vocabulary_identifier = "https://w3id.org/nfdi4cat/voc4cat"
        if self.semantic_service is None:
            warnings.append(
                f"Profile field '{json_path}' was not grounded because semantic service is unavailable."
            )
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                vocabulary_identifier=vocabulary_identifier,
                query_ids=[],
            )
        try:
            vocab_info = await self.semantic_service.get_vocabulary(vocabulary_identifier)
        except Exception as exc:
            warnings.append(f"Vocabulary '{vocabulary_identifier}' unavailable: {exc}")
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                vocabulary_identifier=vocabulary_identifier,
                query_ids=[],
            )
        if vocab_info is None:
            warnings.append(f"Vocabulary '{vocabulary_identifier}' is not registered.")
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                vocabulary_identifier=vocabulary_identifier,
                query_ids=[],
            )
        for term_scheme in vocab_info.vocab_term_schemes:
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type=term_scheme.rdf_type,
                    vector_query=source_value,
                    fulltext_query=source_value,
                    vector_top_k=6,
                    fulltext_top_k=6,
                    seed_top_k=3,
                    max_hops=1,
                    max_statements_per_seed=20,
                ),
                state.vocab_query_config,
            )
            record = self._ensure_run_vocab_query_record(
                state=state,
                kind=f"profile_{field_name}",
                source_value=source_value,
                source_context=source_context,
                vocabulary_identifier=vocabulary_identifier,
                query=query,
            )
            query_ids.append(record.query_id)
            on_progress()
            async with query_semaphore:
                await self._run_vocab_query_record(
                    data_package_id=data_package_id,
                    record=record,
                    vocabulary_identifier=vocabulary_identifier,
                    query=query,
                    on_progress=on_progress,
                    warnings=warnings,
                )
        return _ProfileFieldCandidateDiscovery(
            json_path=json_path,
            field_name=field_name,
            source_value=source_value,
            vocabulary_identifier=vocabulary_identifier,
            query_ids=query_ids,
        )

    async def _normalize_profile_field_candidate_tasks(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        candidate_tasks: list[asyncio.Task[_ProfileFieldCandidateDiscovery]],
        warnings: list[str],
    ) -> ExtractionNormalization:
        if not candidate_tasks:
            return ExtractionNormalization()
        discoveries = await asyncio.gather(*candidate_tasks)
        return await self._normalize_profile_field_discoveries(
            data_package_id=data_package_id,
            state=state,
            discoveries=list(discoveries),
            warnings=warnings,
        )

    async def _normalize_profile_fields_from_state_vocab_queries(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        warnings: list[str],
    ) -> ExtractionNormalization:
        groups: dict[tuple[str, str, str, str], _ProfileFieldCandidateDiscovery] = {}
        for record in state.vocab_queries:
            if not record.kind.startswith("profile_"):
                continue
            json_path = str(record.source_context.get("json_path", ""))
            field_name = str(record.source_context.get("field_name", ""))
            source_value = str(record.source_context.get("source_value", record.source_value))
            key = (json_path, field_name, source_value, record.vocabulary_identifier)
            discovery = groups.setdefault(
                key,
                _ProfileFieldCandidateDiscovery(
                    json_path=json_path,
                    field_name=field_name,
                    source_value=source_value,
                    vocabulary_identifier=record.vocabulary_identifier,
                    query_ids=[],
                ),
            )
            discovery.query_ids.append(record.query_id)
        return await self._normalize_profile_field_discoveries(
            data_package_id=data_package_id,
            state=state,
            discoveries=list(groups.values()),
            warnings=warnings,
        )

    async def _normalize_profile_field_discoveries(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        discoveries: list[_ProfileFieldCandidateDiscovery],
        warnings: list[str],
    ) -> ExtractionNormalization:
        selection_semaphore = asyncio.Semaphore(self._vocab_selection_llm_concurrency())
        profile_fields: list[ProfileFieldNormalization] = []
        for discovery in discoveries:
            candidates: list[dict[str, Any]] = []
            for query_id in discovery.query_ids:
                record = self._find_vocab_query_record(state, query_id)
                if record and record.result:
                    candidates.extend(self._candidate_records(record.result))
            mapping = await self._select_from_candidates_with_semaphore(
                data_package_id=data_package_id,
                agent_name="profile_field_vocab_selection",
                source_value=discovery.source_value,
                source_context={
                    "json_path": discovery.json_path,
                    "field_name": discovery.field_name,
                },
                candidates=self._deduplicate_candidates(candidates),
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
            if mapping is None:
                warnings.append(
                    f"Profile field '{discovery.json_path}' kept raw value '{discovery.source_value}'."
                )
            profile_fields.append(
                ProfileFieldNormalization(
                    json_path=discovery.json_path,
                    field_name=discovery.field_name,
                    source_value=discovery.source_value,
                    term=mapping,
                )
            )
        return ExtractionNormalization(profile_fields=profile_fields)

    async def _discover_quantity_candidates(
        self,
        *,
        quantity: QuantitativeAttribute,
        state: ExtractionRunState,
        data_package_id: str,
        query_semaphore: asyncio.Semaphore,
        on_progress: Any,
        warnings: list[str],
    ) -> _QuantityCandidateDiscovery:
        kind_query = self._configured_vocab_query(
            build_quantity_kind_vocab_query(quantity),
            state.vocab_query_config,
            group="quantitative",
        )
        unit_query = self._configured_vocab_query(
            build_unit_vocab_query(quantity),
            state.vocab_query_config,
            group="quantitative",
        )
        kind_record = self._ensure_run_vocab_query_record(
            state=state,
            kind="quantity_kind",
            source_value=quantity.quantity_kind,
            source_context=quantity.model_dump(mode="json"),
            vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
            query=kind_query,
        )
        unit_record = self._ensure_run_vocab_query_record(
            state=state,
            kind="unit",
            source_value=quantity.unit,
            source_context=quantity.model_dump(mode="json"),
            vocabulary_identifier=QUDT_UNIT_VOCAB,
            query=unit_query,
        )
        on_progress()
        async with query_semaphore:
            await self._run_vocab_query_record(
                data_package_id=data_package_id,
                record=kind_record,
                vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
                query=kind_query,
                on_progress=on_progress,
                warnings=warnings,
            )
        async with query_semaphore:
            await self._run_vocab_query_record(
                data_package_id=data_package_id,
                record=unit_record,
                vocabulary_identifier=QUDT_UNIT_VOCAB,
                query=unit_query,
                on_progress=on_progress,
                warnings=warnings,
            )
        return _QuantityCandidateDiscovery(
            quantity=quantity,
            quantity_kind_query_id=kind_record.query_id,
            unit_query_id=unit_record.query_id,
        )

    async def _discover_object_grounding_candidates(
        self,
        *,
        trace: TracedExtractionObject,
        state: ExtractionRunState,
        data_package_id: str,
        query_semaphore: asyncio.Semaphore,
        on_progress: Any,
        warnings: list[str],
    ) -> _ObjectGroundingCandidateDiscovery:
        obj = trace.extracted_object
        if self.semantic_service is None:
            warnings.append(
                f"Object '{obj.identifier}' was not grounded because semantic service is unavailable."
            )
            return _ObjectGroundingCandidateDiscovery(
                object_identifier=obj.identifier,
                object_kind=trace.object_kind,
                raw_type="",
                source_context={},
                query_ids=[],
            )

        voc4cat_identifier = "https://w3id.org/nfdi4cat/voc4cat"
        query_ids: list[str] = []
        try:
            vocab_info = await self.semantic_service.get_vocabulary(voc4cat_identifier)
        except Exception as exc:
            warnings.append(f"Vocabulary '{voc4cat_identifier}' unavailable: {exc}")
            return _ObjectGroundingCandidateDiscovery(
                object_identifier=obj.identifier,
                object_kind=trace.object_kind,
                raw_type="",
                source_context={},
                query_ids=[],
            )
        if vocab_info is None:
            warnings.append(f"Vocabulary '{voc4cat_identifier}' is not registered.")
            return _ObjectGroundingCandidateDiscovery(
                object_identifier=obj.identifier,
                object_kind=trace.object_kind,
                raw_type="",
                source_context={},
                query_ids=[],
            )

        for term_scheme in vocab_info.vocab_term_schemes:
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type=term_scheme.rdf_type,
                    vector_query=obj.to_embedding_text(),
                    fulltext_query=obj.to_fulltext_query(),
                    vector_top_k=6,
                    fulltext_top_k=6,
                    seed_top_k=3,
                    max_hops=1,
                    max_statements_per_seed=20,
                ),
                state.vocab_query_config,
            )
            record = self._ensure_run_vocab_query_record(
                state=state,
                kind="object_grounding",
                source_value=obj.identifier,
                source_context=obj.model_dump(mode="json"),
                vocabulary_identifier=voc4cat_identifier,
                query=query,
            )
            query_ids.append(record.query_id)
            on_progress()
            async with query_semaphore:
                await self._run_vocab_query_record(
                    data_package_id=data_package_id,
                    record=record,
                    vocabulary_identifier=voc4cat_identifier,
                    query=query,
                    on_progress=on_progress,
                    warnings=warnings,
                )

        return _ObjectGroundingCandidateDiscovery(
            object_identifier=obj.identifier,
            object_kind=trace.object_kind,
            raw_type=obj.type,
            source_context=obj.model_dump(mode="json"),
            query_ids=query_ids,
        )
    async def _normalize_quantity_from_candidates(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        discovery: _QuantityCandidateDiscovery,
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> QuantityNormalization:
        quantity = discovery.quantity
        kind_record = self._find_vocab_query_record(state, discovery.quantity_kind_query_id)
        unit_record = self._find_vocab_query_record(state, discovery.unit_query_id)
        quantity_kind = await self._select_term_with_fallback_candidates(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
            source_value=quantity.quantity_kind,
            source_context=quantity.model_dump(mode="json"),
            query=kind_record.query if kind_record else build_quantity_kind_vocab_query(quantity),
            initial_candidates=self._candidate_records(kind_record.result) if kind_record and kind_record.result else [],
            agent_name="quantity_vocab_selection",
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        unit = await self._select_term_with_fallback_candidates(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_UNIT_VOCAB,
            source_value=quantity.unit,
            source_context=quantity.model_dump(mode="json"),
            query=unit_record.query if unit_record else build_unit_vocab_query(quantity),
            initial_candidates=self._candidate_records(unit_record.result) if unit_record and unit_record.result else [],
            agent_name="quantity_vocab_selection",
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        if quantity_kind is None and quantity.quantity_kind:
            warnings.append(
                f"Quantity '{quantity.identifier}' kept raw quantity kind '{quantity.quantity_kind}'."
            )
            quantity_kind = VocabularyTermMapping(
                source_value=quantity.quantity_kind,
                reason="No QUDT quantity-kind candidate selected; raw value retained.",
            )
        if unit is None and quantity.unit:
            warnings.append(
                f"Quantity '{quantity.identifier}' kept raw unit '{quantity.unit}'."
            )
            unit = VocabularyTermMapping(
                source_value=quantity.unit,
                reason="No QUDT unit candidate selected; raw value retained.",
            )
        return QuantityNormalization(
            quantity=quantity,
            quantity_kind=quantity_kind,
            unit=unit,
        )

    async def _normalize_qualitative_attribute_from_candidates(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        discovery: _QualitativeCandidateDiscovery,
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> QualitativeAttributeNormalization:
        attribute = discovery.attribute
        source_value = f"{attribute.title}: {attribute.value}".strip(": ")
        candidates: list[dict[str, Any]] = []
        for query_id in discovery.query_ids:
            record = self._find_vocab_query_record(state, query_id)
            if record and record.result:
                candidates.extend(self._candidate_records(record.result))
        mapping = await self._select_from_candidates_with_semaphore(
            data_package_id=data_package_id,
            agent_name="qualitative_vocab_selection",
            source_value=source_value,
            source_context=attribute.model_dump(mode="json"),
            candidates=self._deduplicate_candidates(candidates),
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        if mapping is None:
            warnings.append(
                f"Qualitative attribute '{attribute.title}' kept raw value '{attribute.value}'."
            )
        return QualitativeAttributeNormalization(attribute=attribute, term=mapping)

    async def _normalize_object_grounding_from_candidates(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        discovery: _ObjectGroundingCandidateDiscovery,
        trace: TracedExtractionObject,
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> GroundedExtractionObject:
        """Build a `GroundedExtractionObject` for a single voc4cat-grounded object.

        Skos collections are not candidate types; only skos:Concept records are
        presented to the LLM. When the LLM declines to select a term, the wrapper
        still carries the original `extracted_object` and the raw `type` string.
        """
        candidates: list[dict[str, Any]] = []
        for query_id in discovery.query_ids:
            record = self._find_vocab_query_record(state, query_id)
            if record is None or record.result is None:
                continue
            if record.query.rdf_type != "skos__Concept":
                continue
            for entry in self._candidate_records(record.result):
                candidates.append(entry)
        candidates = self._deduplicate_candidates(candidates)
        if not candidates:
            return GroundedExtractionObject(
                object_identifier=discovery.object_identifier,
                object_kind=discovery.object_kind,
                extracted_object=trace.extracted_object,
                source_value=discovery.raw_type,
            )
        try:
            selection = await self._select_object_grounding_term(
                data_package_id=data_package_id,
                object_identifier=discovery.object_identifier,
                object_kind=discovery.object_kind,
                raw_type=discovery.raw_type,
                source_context=discovery.source_context,
                candidates=candidates,
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
        except CompletionError as exc:
            warnings.append(
                f"Object grounding selection failed for '{discovery.object_identifier}': {exc}"
            )
            return GroundedExtractionObject(
                object_identifier=discovery.object_identifier,
                object_kind=discovery.object_kind,
                extracted_object=trace.extracted_object,
                source_value=discovery.raw_type,
            )
        if selection is None:
            return GroundedExtractionObject(
                object_identifier=discovery.object_identifier,
                object_kind=discovery.object_kind,
                extracted_object=trace.extracted_object,
                source_value=discovery.raw_type,
            )
        return GroundedExtractionObject(
            object_identifier=discovery.object_identifier,
            object_kind=discovery.object_kind,
            extracted_object=trace.extracted_object,
            source_value=discovery.raw_type,
            defined_term=DefinedTerm(
                id=selection.selected_uri or "",
                title=selection.selected_title,
                from_CV=selection.vocabulary_identifier,
            ),
            confidence=selection.confidence,
            reason=selection.reason,
        )

    async def _select_object_grounding_term(
        self,
        *,
        data_package_id: str,
        object_identifier: str,
        object_kind: str,
        raw_type: str,
        source_context: dict[str, Any],
        candidates: list[dict[str, Any]],
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        if not candidates:
            return None
        async with selection_semaphore:
            assert self.ollama_client is not None
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
                prompt=build_object_grounding_selection_prompt(
                    object_identifier=object_identifier,
                    object_kind=object_kind,
                    raw_type=raw_type,
                    source_context=source_context,
                    candidates=candidates,
                ),
                output_type=VocabularyCandidateSelection,
                num_ctx=self.ollama_client.max_context_length,
            )
        self._record_workflow_token_usage(
            data_package_id=data_package_id,
            agent_name="object_grounding_vocab_selection",
            usage=result.usage,
        )
        selection = (
            result.output
            if isinstance(result.output, VocabularyCandidateSelection)
            else VocabularyCandidateSelection.model_validate(result.output)
        )
        selected = selection.selected_uri
        if selected is None:
            return None
        candidate = next((item for item in candidates if item.get("uri") == selected), None)
        if candidate is None:
            warnings.append(
                f"Object grounding selector returned unknown URI '{selected}'."
            )
            return None
        return VocabularyTermMapping(
            source_value=raw_type,
            vocabulary_identifier=candidate.get("vocabulary_identifier"),
            rdf_type=candidate.get("rdf_type"),
            selected_uri=selected,
            selected_title=candidate.get("title"),
            confidence=selection.confidence,
            reason=selection.reason,
        )

    async def _select_term_with_fallback(
        self,
        *,
        data_package_id: str,
        vocabulary_identifier: str,
        source_value: str,
        source_context: dict[str, Any],
        query: VocabQuery,
        agent_name: str,
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        candidates = await self._query_candidate_records(
            vocabulary_identifier=vocabulary_identifier,
            query=query,
            warnings=warnings,
        )
        return await self._select_term_with_fallback_candidates(
            data_package_id=data_package_id,
            vocabulary_identifier=vocabulary_identifier,
            source_value=source_value,
            source_context=source_context,
            query=query,
            initial_candidates=candidates,
            agent_name=agent_name,
            selection_semaphore=asyncio.Semaphore(self._vocab_selection_llm_concurrency()),
            warnings=warnings,
        )

    async def _select_term_with_fallback_candidates(
        self,
        *,
        data_package_id: str,
        vocabulary_identifier: str,
        source_value: str,
        source_context: dict[str, Any],
        query: VocabQuery,
        initial_candidates: list[dict[str, Any]],
        agent_name: str,
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        mapping = await self._select_from_candidates(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            candidates=initial_candidates,
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        if mapping is not None:
            return mapping
        fallback = await self._build_fallback_query(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            failed_candidates=initial_candidates,
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        if fallback is None:
            return None
        fallback_query = query.model_copy(
            update={
                "vector_query": fallback.vector_query or query.vector_query,
                "fulltext_query": fallback.fulltext_query or query.fulltext_query,
            }
        )
        fallback_candidates = await self._query_candidate_records(
            vocabulary_identifier=vocabulary_identifier,
            query=fallback_query,
            warnings=warnings,
        )
        return await self._select_from_candidates(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            candidates=fallback_candidates,
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )

    async def _query_candidate_records(
        self,
        *,
        vocabulary_identifier: str,
        query: VocabQuery,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        if self.semantic_service is None:
            warnings.append("Vocabulary normalization skipped because semantic service is unavailable.")
            return []
        try:
            result = await self.semantic_service.query_vocabulary(
                vocabulary_identifier,
                query,
            )
        except Exception as exc:
            warnings.append(
                f"Vocabulary query failed for '{vocabulary_identifier}' ({query.rdf_type}): {exc}"
            )
            return []
        return self._candidate_records(result)

    async def _run_vocab_query_record(
        self,
        *,
        data_package_id: str,
        record: ExtractionVocabQueryRecord,
        vocabulary_identifier: str,
        query: VocabQuery,
        on_progress: Any | None,
        warnings: list[str],
    ) -> None:
        if self.semantic_service is None:
            record.status = "failed"
            record.error = "Vocabulary normalization skipped because semantic service is unavailable."
            warnings.append(record.error)
            if on_progress:
                on_progress()
            return
        record.status = "running"
        record.error = None
        record.result = None
        started = time.perf_counter()
        if on_progress:
            on_progress()
        try:
            record.result = await self.semantic_service.query_vocabulary(
                vocabulary_identifier,
                query,
            )
            record.status = "completed"
            record.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            warnings.append(
                f"Vocabulary query failed for '{vocabulary_identifier}' ({query.rdf_type}): {exc}"
            )
        if on_progress:
            on_progress()

    def _ensure_run_vocab_query_record(
        self,
        *,
        state: ExtractionRunState,
        kind: str,
        source_value: str,
        source_context: dict[str, Any],
        vocabulary_identifier: str,
        query: VocabQuery,
    ) -> ExtractionVocabQueryRecord:
        query_id = self._run_vocab_query_id(
            kind=kind,
            source_value=source_value,
            source_context=source_context,
            vocabulary_identifier=vocabulary_identifier,
            rdf_type=query.rdf_type,
        )
        existing = next(
            (record for record in state.vocab_queries if record.query_id == query_id),
            None,
        )
        if existing is not None:
            existing.source_context = source_context
            existing.query = query
            existing.vocabulary_identifier = vocabulary_identifier
            existing.rdf_type = query.rdf_type
            return existing
        record = ExtractionVocabQueryRecord(
            query_id=query_id,
            kind=kind,
            source_value=source_value,
            source_context=source_context,
            vocabulary_identifier=vocabulary_identifier,
            rdf_type=query.rdf_type,
            query=query,
        )
        state.vocab_queries.append(record)
        return record

    @staticmethod
    def _find_vocab_query_record(
        state: ExtractionRunState,
        query_id: str,
    ) -> ExtractionVocabQueryRecord | None:
        for record in state.vocab_queries:
            if record.query_id == query_id:
                return record
        return None

    @staticmethod
    def _run_vocab_query_id(
        *,
        kind: str,
        source_value: str,
        source_context: dict[str, Any],
        vocabulary_identifier: str,
        rdf_type: str,
    ) -> str:
        raw = "|".join(
            str(part)
            for part in (
                "run",
                kind,
                source_value,
                repr(sorted(source_context.items())),
                vocabulary_identifier,
                rdf_type,
            )
        )
        return sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _candidate_records(result: VocabQueryResult) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for uri, resource in result.resources.items():
            records.append(
                {
                    "uri": uri,
                    "vocabulary_identifier": result.identifier,
                    "rdf_type": result.rdf_type,
                    "title": _resource_title(resource.properties),
                    "rdf_types": resource.rdf_types,
                    "properties": resource.properties,
                }
            )
        return records

    async def _select_from_candidates(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        source_value: str,
        source_context: dict[str, Any],
        candidates: list[dict[str, Any]],
        selection_semaphore: asyncio.Semaphore | None = None,
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        if not candidates:
            return None
        if selection_semaphore is not None:
            async with selection_semaphore:
                return await self._select_from_candidates(
                    data_package_id=data_package_id,
                    agent_name=agent_name,
                    source_value=source_value,
                    source_context=source_context,
                    candidates=candidates,
                    selection_semaphore=None,
                    warnings=warnings,
                )
        assert self.ollama_client is not None
        result = await generate_structured(
            self.ollama_client,
            model=self.ollama_client.chat_model,
            system=VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
            prompt=build_candidate_selection_prompt(
                source_value=source_value,
                source_context=source_context,
                candidates=candidates,
            ),
            output_type=VocabularyCandidateSelection,
            num_ctx=self.ollama_client.max_context_length,
        )
        self._record_workflow_token_usage(
            data_package_id=data_package_id,
            agent_name=agent_name,
            usage=result.usage,
        )
        selected = result.output.selected_uri
        if selected is None:
            return None
        candidate = next((item for item in candidates if item.get("uri") == selected), None)
        if candidate is None:
            warnings.append(f"Vocabulary selector returned unknown URI '{selected}'.")
            return None
        return VocabularyTermMapping(
            source_value=source_value,
            vocabulary_identifier=candidate.get("vocabulary_identifier"),
            rdf_type=candidate.get("rdf_type"),
            selected_uri=selected,
            selected_title=candidate.get("title"),
            confidence=result.output.confidence,
            reason=result.output.reason,
        )

    async def _select_from_candidates_with_semaphore(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        source_value: str,
        source_context: dict[str, Any],
        candidates: list[dict[str, Any]],
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        return await self._select_from_candidates(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            candidates=candidates,
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )

    async def _build_fallback_query(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        source_value: str,
        source_context: dict[str, Any],
        failed_candidates: list[dict[str, Any]],
        selection_semaphore: asyncio.Semaphore | None = None,
        warnings: list[str],
    ) -> VocabularyFallbackQuery | None:
        if selection_semaphore is not None:
            async with selection_semaphore:
                return await self._build_fallback_query(
                    data_package_id=data_package_id,
                    agent_name=agent_name,
                    source_value=source_value,
                    source_context=source_context,
                    failed_candidates=failed_candidates,
                    selection_semaphore=None,
                    warnings=warnings,
                )
        assert self.ollama_client is not None
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
                prompt=build_fallback_query_prompt(
                    source_value=source_value,
                    source_context=source_context,
                    failed_candidates=failed_candidates,
                ),
                output_type=VocabularyFallbackQuery,
                num_ctx=self.ollama_client.max_context_length,
            )
        except CompletionError as exc:
            warnings.append(f"Fallback vocabulary query generation failed: {exc}")
            return None
        self._record_workflow_token_usage(
            data_package_id=data_package_id,
            agent_name=agent_name,
            usage=result.usage,
        )
        return result.output

    @staticmethod
    def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_uri: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            uri = candidate.get("uri")
            if isinstance(uri, str) and uri not in by_uri:
                by_uri[uri] = candidate
        return list(by_uri.values())

    @staticmethod
    def _default_vocab_query_config(
        qualitative_vocab_identifiers: list[str] | None,
    ) -> ExtractionVocabQueryConfig:
        return ExtractionVocabQueryConfig(
            qualitative_vocab_identifiers=(
                qualitative_vocab_identifiers or DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS
            )
        )

    @staticmethod
    def _configured_vocab_query(
        query: VocabQuery,
        config: ExtractionVocabQueryConfig,
        *,
        group: str = "qualitative",
    ) -> VocabQuery:
        if group == "quantitative":
            return query.model_copy(
                update={
                    "vector_top_k": config.quantitative_vector_top_k,
                    "fulltext_top_k": config.quantitative_fulltext_top_k,
                    "seed_top_k": config.quantitative_seed_top_k,
                    "max_hops": config.quantitative_max_hops,
                    "max_statements_per_seed": config.quantitative_max_statements_per_seed,
                    "traversal_direction": config.quantitative_traversal_direction,
                    "vector_weight": config.quantitative_vector_weight,
                    "fulltext_weight": config.quantitative_fulltext_weight,
                    "rrf_k": config.quantitative_rrf_k,
                }
            )
        return query.model_copy(
            update={
                "vector_top_k": config.vector_top_k,
                "fulltext_top_k": config.fulltext_top_k,
                "seed_top_k": config.seed_top_k,
                "max_hops": config.max_hops,
                "max_statements_per_seed": config.max_statements_per_seed,
                "traversal_direction": config.traversal_direction,
                "vector_weight": config.vector_weight,
                "fulltext_weight": config.fulltext_weight,
                "rrf_k": config.rrf_k,
            }
        )

    @staticmethod
    def _all_quantities(context: ExtractionContext) -> list[QuantitativeAttribute]:
        quantities: list[QuantitativeAttribute] = []
        for trace in context.extraction_objects:
            quantities.extend(trace.extracted_object.has_quantitative_attributes)
        return quantities

    @classmethod
    def _unique_quantities(cls, context: ExtractionContext) -> list[QuantitativeAttribute]:
        quantities: list[QuantitativeAttribute] = []
        seen: set[tuple[str, str, str, str]] = set()
        for quantity in cls._all_quantities(context):
            key = (
                quantity.identifier.strip().lower(),
                quantity.value.strip().lower(),
                quantity.unit.strip().lower(),
                quantity.quantity_kind.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            quantities.append(quantity)
        return quantities

    @staticmethod
    def _all_qualitative_attributes(context: ExtractionContext) -> list[QualitativeAttribute]:
        attributes: list[QualitativeAttribute] = []
        for trace in context.extraction_objects:
            attributes.extend(trace.extracted_object.has_qualitative_attributes)
        return attributes

    def _load_result_or_none(self, data_package_id: str) -> ExtractionRunResult | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_result(data_package_id)
        except (FileNotFoundError, ValidationError):
            return None

    def _load_context_or_none(self, data_package_id: str) -> ExtractionContext | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_context(data_package_id)
        except (FileNotFoundError, ValidationError):
            return None

    def _load_run_state_or_none(self, data_package_id: str) -> ExtractionRunState | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_run_state(data_package_id)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError):
            return None

    def _save_run_state(
        self,
        data_package_id: str,
        state: ExtractionRunState,
    ) -> None:
        if self.output_repository is None:
            return
        self.output_repository.save_extraction_run_state(
            workflow_id=data_package_id,
            state=state,
        )

    def _load_warnings_or_empty(self, data_package_id: str) -> list[str]:
        if self.output_repository is None:
            return []
        try:
            return self.output_repository.load_extraction_warnings(data_package_id)
        except FileNotFoundError:
            return []

    def _require_runtime_dependencies(self) -> None:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
            or self.task_registry is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "output_repository, and task_registry to run extraction."
            )

    def _update_progress(
        self,
        data_package_id: str,
        progress: ExtractionRunProgress,
    ) -> None:
        if self.task_registry is None:
            return
        self.task_registry.update_progress(
            self._extraction_task_name(data_package_id),
            progress.model_dump(mode="json"),
        )

    def _update_complete_workflow_progress(
        self,
        *,
        data_package_id: str,
        progress: CompleteWorkflowProgress,
    ) -> None:
        if self.task_registry is None:
            return
        self.task_registry.update_progress(
            self._complete_workflow_task_name(data_package_id),
            self._complete_workflow_progress_with_steps(
                progress,
                workflow_status=TaskStatus.RUNNING,
            ).model_dump(mode="json"),
        )

    def _derive_complete_workflow_progress(
        self,
        *,
        data_package_id: str,
        workflow_status: TaskStatus,
    ) -> CompleteWorkflowProgress | None:
        extraction_status, extraction_progress = self._current_extraction_progress(
            data_package_id
        )
        result = self._load_result_or_none(data_package_id)
        if result is not None:
            extraction_status = TaskStatus.COMPLETED
            state = self._load_run_state_or_none(data_package_id)
            extraction_progress = extraction_progress or ExtractionRunProgress(
                stage="completed",
                interim_context=result.extraction_context,
                vocab_query_config=state.vocab_query_config if state else None,
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                interim_profile_document=state.interim_profile_document if state else None,
                profile_patch_results=state.profile_patch_results if state else [],
                warnings=list(result.warnings),
            )

        chunking_status = self._current_chunking_status(data_package_id)
        if result is not None:
            chunking_status = TaskStatus.COMPLETED

        if (
            workflow_status == TaskStatus.UNKNOWN
            and chunking_status == TaskStatus.UNKNOWN
            and extraction_status == TaskStatus.UNKNOWN
            and result is None
        ):
            return None

        stage = self._complete_workflow_stage(
            workflow_status=workflow_status,
            chunking_status=chunking_status,
            extraction_status=extraction_status,
            extraction_progress=extraction_progress,
            result_exists=result is not None,
        )
        warnings = (
            list(extraction_progress.warnings)
            if extraction_progress is not None
            else self._load_warnings_or_empty(data_package_id)
        )
        return CompleteWorkflowProgress(
            stage=stage,
            data_package_id=data_package_id,
            profile_identifier=self._profile_identifier_from_state(data_package_id),
            chunking_status=chunking_status,
            extraction_status=extraction_status,
            extraction_progress=extraction_progress,
            warnings=warnings,
            result_url=self._result_url(data_package_id),
        )

    def _current_extraction_progress(
        self,
        data_package_id: str,
    ) -> tuple[TaskStatus, ExtractionRunProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None
        task_info = self.task_registry.get_task_info(
            self._extraction_task_name(data_package_id)
        )
        if task_info is not None:
            progress = (
                ExtractionRunProgress.model_validate(task_info.progress)
                if task_info.progress
                else None
            )
            return task_info.status, progress
        result = self._load_result_or_none(data_package_id)
        if result is not None:
            state = self._load_run_state_or_none(data_package_id)
            return TaskStatus.COMPLETED, ExtractionRunProgress(
                stage="completed",
                interim_context=result.extraction_context,
                vocab_query_config=state.vocab_query_config if state else None,
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                interim_profile_document=state.interim_profile_document if state else None,
                profile_patch_results=state.profile_patch_results if state else [],
                warnings=list(result.warnings),
            )
        state = self._load_run_state_or_none(data_package_id)
        if state is not None:
            return TaskStatus.UNKNOWN, ExtractionRunProgress(
                stage="interim_context",
                processed_chunks=self._completed_chunk_count(state),
                total_chunks=len(state.chunk_results),
                interim_context=self._merged_completed_chunk_context_or_none(state),
                vocab_query_config=state.vocab_query_config,
                ranked_files=state.ranked_files,
                chunk_results=state.chunk_results,
                vocab_queries=state.vocab_queries,
                interim_profile_document=state.interim_profile_document,
                profile_patch_results=state.profile_patch_results,
                warnings=self._load_warnings_or_empty(data_package_id),
            )
        return TaskStatus.UNKNOWN, None

    def _current_chunking_status(self, data_package_id: str) -> TaskStatus:
        if self.datasource_service is None:
            return TaskStatus.UNKNOWN
        get_status = getattr(self.datasource_service, "get_chunk_task_status", None)
        if callable(get_status):
            return get_status(data_package_id)
        chunks = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        return TaskStatus.COMPLETED if chunks else TaskStatus.UNKNOWN

    def _profile_identifier_from_state(self, data_package_id: str) -> str | None:
        state = self._load_run_state_or_none(data_package_id)
        return state.profile_identifier if state is not None else None

    @staticmethod
    def _complete_workflow_stage(
        *,
        workflow_status: TaskStatus,
        chunking_status: TaskStatus,
        extraction_status: TaskStatus,
        extraction_progress: ExtractionRunProgress | None,
        result_exists: bool,
    ) -> str:
        if result_exists or extraction_status == TaskStatus.COMPLETED:
            return "completed"
        if workflow_status == TaskStatus.CRASHED or extraction_status == TaskStatus.CRASHED:
            return "crashed"
        if workflow_status == TaskStatus.CANCELLED or extraction_status == TaskStatus.CANCELLED:
            return "cancelled"
        if extraction_progress is not None and extraction_progress.stage != "pending":
            return extraction_progress.stage
        if extraction_status == TaskStatus.RUNNING:
            return "extraction"
        if chunking_status == TaskStatus.RUNNING:
            return "chunking"
        if chunking_status == TaskStatus.COMPLETED:
            return "extraction_pending"
        return "pending"

    def _complete_workflow_progress_with_steps(
        self,
        progress: CompleteWorkflowProgress,
        *,
        workflow_status: TaskStatus,
    ) -> CompleteWorkflowProgress:
        if workflow_status == TaskStatus.CRASHED:
            progress = progress.model_copy(update={"stage": "crashed"})
        elif workflow_status == TaskStatus.CANCELLED:
            progress = progress.model_copy(update={"stage": "cancelled"})
        step_statuses = self._complete_workflow_step_statuses(
            stage=progress.stage,
            workflow_status=workflow_status,
            chunking_status=progress.chunking_status,
            extraction_status=progress.extraction_status,
            extraction_progress=progress.extraction_progress,
        )
        return progress.model_copy(
            update={
                "steps": [
                    CompleteWorkflowStepProgress(name=name, status=status)
                    for name, status in step_statuses.items()
                ]
            }
        )

    @staticmethod
    def _complete_workflow_step_statuses(
        *,
        stage: str,
        workflow_status: TaskStatus,
        chunking_status: TaskStatus,
        extraction_status: TaskStatus,
        extraction_progress: ExtractionRunProgress | None,
    ) -> dict[str, TaskStatus]:
        steps = {
            "upload": TaskStatus.COMPLETED,
            "chunking": chunking_status,
            "extraction": extraction_status,
            "normalization": TaskStatus.UNKNOWN,
            "profile_projection": TaskStatus.UNKNOWN,
            "validation": TaskStatus.UNKNOWN,
        }
        if extraction_progress is not None:
            if extraction_progress.stage in {
                "file_ranking",
                "chunk_extraction",
                "chunk_repair",
                "interim_context",
            }:
                steps["extraction"] = TaskStatus.RUNNING
            if extraction_progress.stage in {
                "vocabulary_normalization",
                "profile_projection",
                "completed",
            }:
                steps["extraction"] = TaskStatus.COMPLETED
                steps["normalization"] = (
                    TaskStatus.RUNNING
                    if extraction_progress.stage == "vocabulary_normalization"
                    else TaskStatus.COMPLETED
                )
            if extraction_progress.stage in {"profile_projection", "completed"}:
                steps["profile_projection"] = (
                    TaskStatus.RUNNING
                    if extraction_progress.stage == "profile_projection"
                    else TaskStatus.COMPLETED
                )
            if extraction_progress.stage == "completed":
                steps["validation"] = TaskStatus.COMPLETED
        if stage == "completed":
            for name in steps:
                steps[name] = TaskStatus.COMPLETED
        if workflow_status == TaskStatus.CRASHED:
            active_step = "extraction" if chunking_status == TaskStatus.COMPLETED else "chunking"
            steps[active_step] = TaskStatus.CRASHED
        return steps

    @staticmethod
    def _result_url(data_package_id: str) -> str:
        return f"/api/v1/extraction/result/{data_package_id}"

    def _record_workflow_token_usage(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        usage: Any,
    ) -> None:
        if self.output_repository is None:
            return
        totals = self.output_repository.load_token_usage(data_package_id)
        entry = totals.setdefault(
            agent_name,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0,
                "operation_count": 0,
                "prompt_eval_duration_ms": 0,
                "load_duration_ms": 0,
                "response_duration_ms": 0,
                "total_duration_ms": 0,
            },
        )
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "requests",
            "operation_count",
            "prompt_eval_duration_ms",
            "load_duration_ms",
            "response_duration_ms",
            "total_duration_ms",
        ):
            entry.setdefault(key, 0)
        input_tokens = self._usage_int(usage, "input_tokens")
        output_tokens = self._usage_int(usage, "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens") or (
            input_tokens + output_tokens
        )
        requests = self._usage_int(usage, "requests")
        prompt_eval_duration_ms = self._usage_int(usage, "prompt_eval_duration_ms")
        load_duration_ms = self._usage_int(usage, "load_duration_ms")
        response_duration_ms = self._usage_int(usage, "response_duration_ms")
        total_duration_ms = self._usage_int(usage, "total_duration_ms")
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["total_tokens"] += total_tokens
        entry["requests"] += requests
        entry["operation_count"] += 1
        entry["prompt_eval_duration_ms"] += prompt_eval_duration_ms
        entry["load_duration_ms"] += load_duration_ms
        entry["response_duration_ms"] += response_duration_ms
        entry["total_duration_ms"] += total_duration_ms
        self.output_repository.save_token_usage(
            workflow_id=data_package_id,
            token_usage=totals,
        )

    @staticmethod
    def _usage_int(usage: Any, field_name: str) -> int:
        try:
            return int(getattr(usage, field_name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _usage_float(usage: Any, field_name: str) -> float | None:
        try:
            value = float(getattr(usage, field_name, 0) or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @classmethod
    def _token_usage_summary(
        cls,
        totals: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        agents = {
            agent_name: cls._token_usage_entry_summary(values)
            for agent_name, values in totals.items()
            if any(
                values.get(key, 0)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "requests",
                    "prompt_eval_duration_ms",
                    "load_duration_ms",
                    "response_duration_ms",
                    "total_duration_ms",
                )
            )
        }
        if not agents:
            return {"agents": {}}
        combined = {
            "input_tokens": sum(item["input_tokens"] for item in agents.values()),
            "output_tokens": sum(item["output_tokens"] for item in agents.values()),
            "total_tokens": sum(item["total_tokens"] for item in agents.values()),
            "requests": sum(item["requests"] for item in agents.values()),
            "operation_count": sum(item["operation_count"] for item in agents.values()),
            "prompt_eval_duration_ms": sum(item["prompt_eval_duration_ms"] for item in agents.values()),
            "load_duration_ms": sum(item["load_duration_ms"] for item in agents.values()),
            "response_duration_ms": sum(item["response_duration_ms"] for item in agents.values()),
            "total_duration_ms": sum(item["total_duration_ms"] for item in agents.values()),
        }
        return {
            "agents": agents,
            "combined": cls._token_usage_entry_summary(combined),
        }

    @staticmethod
    def _token_usage_entry_summary(values: dict[str, int]) -> dict[str, int | float]:
        operation_count = max(1, int(values.get("operation_count", 0)))
        requests = max(1, int(values.get("requests", 0)))
        input_tokens = int(values.get("input_tokens", 0))
        output_tokens = int(values.get("output_tokens", 0))
        total_tokens = int(values.get("total_tokens", 0))
        prompt_eval_duration_ms = int(values.get("prompt_eval_duration_ms", 0))
        load_duration_ms = int(values.get("load_duration_ms", 0))
        response_duration_ms = int(values.get("response_duration_ms", 0))
        total_duration_ms = int(values.get("total_duration_ms", 0))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "requests": int(values.get("requests", 0)),
            "operation_count": operation_count,
            "prompt_eval_duration_ms": prompt_eval_duration_ms,
            "load_duration_ms": load_duration_ms,
            "response_duration_ms": response_duration_ms,
            "total_duration_ms": total_duration_ms,
            "prompt_eval_tokens_per_second": round(
                input_tokens / (prompt_eval_duration_ms / 1000),
                2,
            )
            if prompt_eval_duration_ms > 0
            else 0.0,
            "average_input_tokens_per_operation": round(input_tokens / operation_count, 2),
            "average_output_tokens_per_operation": round(output_tokens / operation_count, 2),
            "average_total_tokens_per_operation": round(total_tokens / operation_count, 2),
            "average_prompt_eval_duration_ms_per_operation": round(prompt_eval_duration_ms / operation_count, 2),
            "average_load_duration_ms_per_operation": round(load_duration_ms / operation_count, 2),
            "average_response_duration_ms_per_operation": round(response_duration_ms / operation_count, 2),
            "average_total_duration_ms_per_operation": round(total_duration_ms / operation_count, 2),
            "average_input_tokens_per_request": round(input_tokens / requests, 2),
            "average_output_tokens_per_request": round(output_tokens / requests, 2),
            "average_total_tokens_per_request": round(total_tokens / requests, 2),
            "average_prompt_eval_duration_ms_per_request": round(prompt_eval_duration_ms / requests, 2),
            "average_load_duration_ms_per_request": round(load_duration_ms / requests, 2),
            "average_response_duration_ms_per_request": round(response_duration_ms / requests, 2),
            "average_total_duration_ms_per_request": round(total_duration_ms / requests, 2),
        }

    def _vocab_query_concurrency(self) -> int:
        return max(1, int(getattr(self.settings, "extraction_vocab_query_concurrency", 4) or 4))

    def _vocab_selection_llm_concurrency(self) -> int:
        return max(1, int(getattr(self.settings, "vocab_selection_llm_concurrency", 1) or 1))

    def _vocab_selection_parallel_enabled(self) -> bool:
        return (
            str(getattr(self.settings, "vocab_selection_parallel_mode", "conservative"))
            .strip()
            .lower()
            == "parallel"
        )

    @staticmethod
    async def _cancel_candidate_tasks(
        candidate_tasks: list[
            asyncio.Task[
                _QuantityCandidateDiscovery
                | _QualitativeCandidateDiscovery
                | _ObjectGroundingCandidateDiscovery
                | _ProfileFieldCandidateDiscovery
            ]
        ],
    ) -> None:
        pending = [task for task in candidate_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _extraction_task_name(data_package_id: str) -> str:
        return f"extraction:run:{data_package_id}"

    @staticmethod
    def _complete_workflow_task_name(data_package_id: str) -> str:
        return f"workflow:complete:{data_package_id}"


def _resource_title(properties: dict[str, Any]) -> str | None:
    label_keys = (
        "label",
        "prefLabel",
        "skos__prefLabel",
        "preferred_label",
        "title",
        "skos__definition",
        "definition",
        "name",
        "symbol",
        "ucumCode",
    )
    for key in label_keys:
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return None
