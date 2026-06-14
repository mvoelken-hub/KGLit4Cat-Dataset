from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import jsonpatch
from pydantic import ValidationError
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import Settings
from app.core.logging import logger
from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk, FileType
from app.domain.extraction import (
    DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS,
    EVIDENCE_CONTEXT_SYSTEM_PROMPT,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT,
    EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT,
    EXTRACTION_OVERVIEW_SYSTEM_PROMPT,
    PROFILE_PROJECTION_SYSTEM_PROMPT,
    PROFILE_PATCH_SYSTEM_PROMPT,
    QUDT_QUANTITY_KIND_VOCAB,
    QUDT_UNIT_VOCAB,
    PROFILE_TARGET_PLANNER_SYSTEM_PROMPT,
    PROFILE_TARGET_WRITER_SYSTEM_PROMPT,
    VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
    VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
    VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
    ChunkContext,
    ChunkMetadata,
    ChunkingRequiredError,
    CompleteWorkflowProgress,
    CompleteWorkflowStepProgress,
    CurationLedgerRecord,
    DefinedTerm,
    DraftValidationResult,
    ChunkRepairMode,
    EvidenceChunkContext,
    EvidenceChunkMetadata,
    EvidenceContext,
    EvidenceNote,
    FilteredEvidenceNote,
    FileInventoryItem,
    ExtractionChunkRef,
    ExtractionChunkResult,
    ExtractionContext,
    ExtractionFileContentWindow,
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewFilePreview,
    ExtractionOverviewInspectedFile,
    ExtractionOverviewStatus,
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
    FieldCompletionLedgerRecord,
    GroundedExtractionObject,
    ProfileFieldNormalization,
    ProfileObjectPatchResult,
    ProfilePatchDocument,
    SchemaBranch,
    ProfileTargetWriteDocument,
    ProfileTargetDecision,
    ProjectionLedgerRecord,
    QualitativeAttribute,
    QualitativeAttributeNormalization,
    QuantitativeAttribute,
    QuantityNormalization,
    Resource,
    TracedExtractionObject,
    VocabularyCandidateSelection,
    VocabularyFallbackQuery,
    VocabularyTermMapping,
    build_evidence_context_prompt,
    build_evidence_system_prompt_with_overview,
    dedupe_repeated_evidence_notes,
    build_candidate_selection_prompt,
    build_extraction_context_prompt,
    build_extraction_file_summary_prompt,
    build_extraction_overview_fallback_prompt,
    build_extraction_overview_prompt,
    filtered_evidence_ledger,
    filter_evidence_context_by_signal_level,
    is_noisy_payload_chunk,
    build_system_prompt_with_overview,
    build_fallback_query_prompt,
    build_object_grounding_selection_prompt,
    build_profile_patch_prompt,
    build_profile_projection_prompt,
    build_profile_target_planner_prompt,
    build_profile_target_write_prompt,
    build_schema_branch_index,
    build_schema_search_query,
    build_qualitative_vocab_query,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    cap_extraction_context_for_prompt,
    fallback_file_ranking,
    merge_evidence_contexts,
    merge_extraction_context_results,
    normalize_chunk_text_for_evidence_prompt,
    schema_branches_to_catalog,
    search_schema_branches,
    validate_evidence_context_for_chunk,
)
from app.domain.profiles import (
    ProfileValidationIssue,
    remove_null_values,
    validation_schema_for_target_class,
)
from app.domain.semantics import VocabQuery, VocabQueryResult
from app.ollama.completion import generate_structured, repair_structured_output
from app.ollama.errors import CompletionError, MaxRetriesExceeded
from app.ollama.usage import RunUsage
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService
    from app.services.semantic_service import SemanticService


ExtractionTargetStage = Literal["context", "profile", "grounding", "complete"]
INITIAL_OVERVIEW_TOP_FILE_LIMIT = 16
INITIAL_OVERVIEW_PREVIEW_LINE_LIMIT = 80
INITIAL_OVERVIEW_MAX_LINE_CHARS = 500
INITIAL_FILE_SUMMARY_CONTEXT_RATIO = 0.35
ESTIMATED_CHARS_PER_TOKEN = 4


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


@dataclass
class _EvidenceProjectionGroup:
    group_id: str
    object_kind: str
    notes: list[EvidenceNote]
    target_hint: str
    target_class_hint: str | None

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
        profile_identifier: str | None = None,
        qualitative_vocab_identifiers: list[str] | None = None,
        resume: bool = False,
        target_stage: ExtractionTargetStage = "complete",
        chunk_repair_mode: ChunkRepairMode = "deferred",
    ) -> tuple[ExtractionRunResult | None, TaskStatus]:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None
        assert self.task_registry is not None

        self.datasource_service.get_data_package(data_package_id)
        profile_identifier_for_stage = self._profile_identifier_for_stage(
            profile_identifier=profile_identifier,
            target_stage=target_stage,
        )
        if profile_identifier_for_stage is not None:
            self.profile_service.get_profile(profile_identifier_for_stage)
            self.profile_service.load_json_schema(profile_identifier_for_stage)

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
            if result is not None and resume:
                return result, TaskStatus.COMPLETED

        if not resume:
            self._clear_downstream_extraction_outputs(data_package_id)
        await self.task_registry.create_task(
            coro=self._run_extraction_task(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier_for_stage,
                qualitative_vocab_identifiers=qualitative_vocab_identifiers,
                resume=resume,
                target_stage=target_stage,
                chunk_repair_mode=chunk_repair_mode,
            ),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        return None, TaskStatus.RUNNING

    async def run_initial_context(
        self,
        *,
        data_package_id: str,
        force_rerun: bool = False,
    ) -> TaskStatus:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None
        assert self.task_registry is not None

        self.datasource_service.get_data_package(data_package_id)

        task_name = self._initial_context_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status == TaskStatus.RUNNING:
            if force_rerun:
                raise ValueError("Cannot force-rerun initial context while it is already running.")
            return TaskStatus.RUNNING
        if (
            not force_rerun
            and task_info is not None
            and task_info.status == TaskStatus.COMPLETED
            and self._load_run_state_or_none(data_package_id) is not None
        ):
            return TaskStatus.COMPLETED

        persisted_state = self._load_run_state_or_none(data_package_id)
        if (
            not force_rerun
            and persisted_state is not None
            and persisted_state.initial_file_summary_status is not None
            and persisted_state.initial_extraction_overview_status is not None
        ):
            return TaskStatus.COMPLETED

        if force_rerun:
            self.output_repository.clear_extraction_run(data_package_id)

        await self.task_registry.create_task(
            coro=self._run_initial_context_task(data_package_id=data_package_id),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        return TaskStatus.RUNNING

    async def get_initial_context_progress(
        self,
        *,
        data_package_id: str,
    ) -> tuple[TaskStatus, ExtractionRunProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None
        task_info = self.task_registry.get_task_info(
            self._initial_context_task_name(data_package_id)
        )
        if task_info is not None:
            progress = (
                ExtractionRunProgress.model_validate(task_info.progress)
                if task_info.progress
                else None
            )
            return task_info.status, progress

        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            return TaskStatus.UNKNOWN, None
        status = (
            TaskStatus.COMPLETED
            if state.initial_file_summary_status is not None
            and state.initial_extraction_overview_status is not None
            else TaskStatus.UNKNOWN
        )
        return status, self._initial_context_progress_from_state(
            state,
            warnings=self._load_warnings_or_empty(data_package_id),
            stage=(
                "initial_context_completed"
                if status == TaskStatus.COMPLETED
                else "initial_context_pending"
            ),
        )

    async def _run_initial_context_task(self, *, data_package_id: str) -> None:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None

        data_package = self.datasource_service.get_data_package(data_package_id)
        warnings: list[str] = []
        progress = ExtractionRunProgress(stage="file_ranking")
        self._update_initial_context_progress(data_package_id, progress)

        ranking = await self._rank_files(data_package_id, data_package, warnings)
        state = ExtractionRunState(
            chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            ranked_files=ranking.files,
        )
        self._save_run_state(data_package_id, state)
        progress = self._initial_context_progress_from_state(
            state,
            warnings=warnings,
            stage="initial_file_summaries",
        )
        self._update_initial_context_progress(data_package_id, progress)

        await self._generate_initial_file_summaries(
            data_package_id=data_package_id,
            data_package=data_package,
            ranking=ranking,
            state=state,
            warnings=warnings,
        )
        progress = self._initial_context_progress_from_state(
            state,
            warnings=warnings,
            stage="initial_overview",
        )
        self._update_initial_context_progress(data_package_id, progress)

        await self._generate_initial_extraction_overview(
            data_package_id=data_package_id,
            data_package=data_package,
            ranking=ranking,
            state=state,
            warnings=warnings,
        )
        self.output_repository.save_extraction_warnings(
            workflow_id=data_package_id,
            warnings=warnings,
        )
        progress = self._initial_context_progress_from_state(
            state,
            warnings=warnings,
            stage="initial_context_completed",
        )
        self._update_initial_context_progress(data_package_id, progress)

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
                stage="initial_context",
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                chunking_status=TaskStatus.UNKNOWN,
                extraction_status=TaskStatus.UNKNOWN,
                result_url=self._result_url(data_package_id),
            ),
        )
        initial_status = await self.run_initial_context(
            data_package_id=data_package_id,
            force_rerun=False,
        )
        if initial_status == TaskStatus.RUNNING:
            await self.task_registry.wait_for_task(
                self._initial_context_task_name(data_package_id)
            )

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
            resume=True,
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
                    progress.stage == "pending"
                    and saved_progress.stage == "initial_context"
                ):
                    update["stage"] = saved_progress.stage
                if (
                    progress.stage == "initial_context_completed"
                    and saved_progress.stage in {"chunking", "extraction"}
                ):
                    update["stage"] = saved_progress.stage
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
                    interim_evidence_context=result.machine_evidence_context,
                    vocab_query_config=state.vocab_query_config if state else self._default_vocab_query_config(None),
                    ranked_files=state.ranked_files if state else [],
                    initial_file_summaries=result.initial_file_summaries,
                    initial_file_summary_status=result.initial_file_summary_status,
                    initial_extraction_overview=result.initial_extraction_overview,
                    initial_extraction_overview_status=result.initial_extraction_overview_status,
                    chunk_results=state.chunk_results if state else [],
                    vocab_queries=state.vocab_queries if state else [],
                    generated_final_draft=result.generated_final_draft,
                    curated_document=result.curated_document,
                    draft_quality_state=result.draft_quality_state,
                    validation=result.validation,
                    curated_validation=result.curated_validation,
                    initial_draft_scaffold=result.initial_draft_scaffold,
                    projection_ledger=result.projection_ledger,
                    field_completion_ledger=result.field_completion_ledger,
                    curation_ledger=result.curation_ledger,
                    warnings=list(result.warnings),
                )
            interim_evidence_context = self._load_evidence_context_or_none(data_package_id)
            if interim_evidence_context is not None or state is not None:
                stage = "interim_evidence_context"
                if interim_evidence_context is None and state and not state.chunk_results and (
                    state.initial_file_summary_status is not None
                    or state.initial_extraction_overview_status is not None
                ):
                    stage = "initial_context_completed"
                elif state and state.generated_final_draft:
                    stage = "profile_draft"
                return TaskStatus.UNKNOWN, ExtractionRunProgress(
                    stage=stage,
                    processed_chunks=self._completed_chunk_count(state) if state else 0,
                    total_chunks=len(state.chunk_results) if state else 0,
                    interim_evidence_context=interim_evidence_context or (
                        self._merged_completed_evidence_context_or_none(state)
                        if state
                        else None
                    ),
                    vocab_query_config=state.vocab_query_config if state else self._default_vocab_query_config(None),
                    ranked_files=state.ranked_files if state else [],
                    initial_file_summaries=state.initial_file_summaries if state else [],
                    initial_file_summary_status=state.initial_file_summary_status if state else None,
                    initial_extraction_overview=state.initial_extraction_overview if state else None,
                    initial_extraction_overview_status=state.initial_extraction_overview_status if state else None,
                    chunk_results=state.chunk_results if state else [],
                    vocab_queries=state.vocab_queries if state else [],
                    generated_final_draft=state.generated_final_draft if state else None,
                    curated_document=state.curated_document if state else None,
                    draft_quality_state=state.draft_quality_state if state else None,
                    validation=state.validation if state else DraftValidationResult(),
                    curated_validation=state.curated_validation if state else None,
                    initial_draft_scaffold=state.initial_draft_scaffold if state else {},
                    projection_ledger=state.projection_ledger if state else [],
                    field_completion_ledger=state.field_completion_ledger if state else [],
                    curation_ledger=state.curation_ledger if state else [],
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
                stage = "interim_evidence_context"
                if not state.chunk_results and (
                    state.initial_file_summary_status is not None
                    or state.initial_extraction_overview_status is not None
                ):
                    stage = "initial_context_completed"
                elif state.generated_final_draft:
                    stage = "profile_draft"
                progress = ExtractionRunProgress(
                    stage=stage,
                    processed_chunks=self._completed_chunk_count(state),
                    total_chunks=len(state.chunk_results),
                    interim_evidence_context=self._merged_completed_evidence_context_or_none(state),
                    vocab_query_config=state.vocab_query_config,
                    ranked_files=state.ranked_files,
                    initial_file_summaries=state.initial_file_summaries,
                    initial_file_summary_status=state.initial_file_summary_status,
                    initial_extraction_overview=state.initial_extraction_overview,
                    initial_extraction_overview_status=state.initial_extraction_overview_status,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    generated_final_draft=state.generated_final_draft,
                    curated_document=state.curated_document,
                    draft_quality_state=state.draft_quality_state,
                    validation=state.validation,
                    curated_validation=state.curated_validation,
                    projection_ledger=state.projection_ledger,
                    field_completion_ledger=state.field_completion_ledger,
                    curation_ledger=state.curation_ledger,
                )
        if progress is not None and progress.interim_evidence_context is None:
            progress.interim_evidence_context = self._load_evidence_context_or_none(data_package_id)
        if progress is not None:
            state = self._load_run_state_or_none(data_package_id)
            if state is not None:
                progress.vocab_query_config = state.vocab_query_config
                if not progress.chunk_results:
                    progress.ranked_files = state.ranked_files
                    progress.chunk_results = state.chunk_results
                    progress.processed_chunks = self._completed_chunk_count(state)
                    progress.total_chunks = len(state.chunk_results)
                progress.initial_extraction_overview = state.initial_extraction_overview
                progress.initial_extraction_overview_status = state.initial_extraction_overview_status
                progress.initial_file_summaries = state.initial_file_summaries
                progress.initial_file_summary_status = state.initial_file_summary_status
                progress.vocab_queries = state.vocab_queries
                progress.generated_final_draft = state.generated_final_draft
                progress.curated_document = state.curated_document
                progress.draft_quality_state = state.draft_quality_state
                progress.validation = state.validation
                progress.curated_validation = state.curated_validation
                progress.projection_ledger = state.projection_ledger
                progress.field_completion_ledger = state.field_completion_ledger
                progress.curation_ledger = state.curation_ledger
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
            interim_evidence_context=self._merged_completed_evidence_context_or_none(state)
            or self._load_evidence_context_or_none(data_package_id),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            generated_final_draft=state.generated_final_draft,
            curated_document=state.curated_document,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
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
            return self.output_repository.load_extraction_result(
                data_package_id,
                chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            )
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
            interim_evidence_context=self._merged_completed_evidence_context_or_none(state),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            generated_final_draft=state.generated_final_draft,
            curated_document=state.curated_document,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
            warnings=self._load_warnings_or_empty(data_package_id),
        )

    async def update_curated_document(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
    ) -> ExtractionRunProgress:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None

        self.datasource_service.get_data_package(data_package_id)
        self.profile_service.get_profile(profile_identifier)
        clean_document = remove_null_values(document)
        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            raise ExtractionResultNotFoundError(
                f"Extraction run state not found for '{data_package_id}'. Run context extraction first."
            )
        state.curated_document = clean_document
        state.curated_validation = self._validate_profile_document(
            profile_identifier=profile_identifier,
            document=clean_document,
        )
        state.curation_ledger = self._build_curation_ledger(
            generated_document=state.generated_final_draft or {},
            curated_document=clean_document,
            existing_field_ledger=state.field_completion_ledger,
        )
        state.field_completion_ledger = self._field_ledger_with_curated_values(
            state.field_completion_ledger,
            clean_document,
        )
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)

        progress = ExtractionRunProgress(
            stage="curated_document",
            processed_chunks=self._completed_chunk_count(state),
            total_chunks=len(state.chunk_results),
            interim_evidence_context=self._load_evidence_context_or_none(data_package_id)
            or self._merged_completed_evidence_context_or_none(state),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            generated_final_draft=state.generated_final_draft,
            curated_document=state.curated_document,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
            warnings=self._load_warnings_or_empty(data_package_id),
        )
        self._update_progress(data_package_id, progress)
        return progress

    async def apply_curation_field_action(
        self,
        *,
        data_package_id: str,
        action: Literal["select_vocab_term", "mark_unresolved"],
        json_path: str,
        selected_uri: str | None = None,
        selected_title: str | None = None,
        vocabulary_identifier: str | None = None,
    ) -> ExtractionRunProgress:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.output_repository is not None

        self.datasource_service.get_data_package(data_package_id)
        state = self._load_run_state_or_none(data_package_id)
        if state is None:
            raise ExtractionResultNotFoundError(
                f"Extraction run state not found for '{data_package_id}'."
            )
        profile_identifier = state.profile_identifier
        if not profile_identifier:
            raise ValueError("Cannot curate a field before a profile is selected.")

        curated_document = self._clone_json_object(
            state.curated_document or state.generated_final_draft or {}
        )
        if action == "select_vocab_term":
            if not selected_uri:
                raise ValueError("select_vocab_term requires selected_uri.")
            profile_manifest = self.profile_service.get_profile(profile_identifier)
            profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
            validation_schema = validation_schema_for_target_class(
                json_schema=profile_json_schema,
                target_class=profile_manifest.target_class,
            )
            field_schema = self._schema_for_json_pointer(validation_schema, json_path)
            selected_value = self._selected_vocab_value_for_schema(
                field_schema=field_schema,
                root_schema=validation_schema,
                selected_uri=selected_uri,
                selected_title=selected_title,
                vocabulary_identifier=vocabulary_identifier,
                existing_value=self._json_pointer_value(curated_document, json_path)[1],
            )
            curated_document = self._set_json_pointer_value(
                curated_document,
                json_path,
                selected_value,
            )

        state.curated_document = curated_document
        state.curated_validation = self._validate_profile_document(
            profile_identifier=profile_identifier,
            document=curated_document,
        )
        state.field_completion_ledger = self._mark_field_curation_status(
            ledger=self._field_ledger_with_curated_values(
                state.field_completion_ledger,
                curated_document,
            ),
            json_path=json_path,
            status=(
                "user_selected_vocab_term"
                if action == "select_vocab_term"
                else "intentionally_unresolved"
            ),
        )
        state.curation_ledger = self._mark_curation_ledger_status(
            ledger=self._build_curation_ledger(
                generated_document=state.generated_final_draft or {},
                curated_document=curated_document,
                existing_field_ledger=state.field_completion_ledger,
            ),
            json_path=json_path,
            status=(
                "user_selected_vocab_term"
                if action == "select_vocab_term"
                else "intentionally_unresolved"
            ),
        )
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)

        progress = ExtractionRunProgress(
            stage="curated_document",
            processed_chunks=self._completed_chunk_count(state),
            total_chunks=len(state.chunk_results),
            interim_evidence_context=self._load_evidence_context_or_none(data_package_id)
            or self._merged_completed_evidence_context_or_none(state),
            vocab_query_config=state.vocab_query_config,
            ranked_files=state.ranked_files,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            generated_final_draft=state.generated_final_draft,
            curated_document=state.curated_document,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
            warnings=self._load_warnings_or_empty(data_package_id),
        )
        self._update_progress(data_package_id, progress)
        return progress

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
        profile_identifier: str | None,
        qualitative_vocab_identifiers: list[str] | None,
        resume: bool = False,
        target_stage: ExtractionTargetStage = "complete",
        chunk_repair_mode: ChunkRepairMode = "deferred",
    ) -> ExtractionRunResult | None:
        self._require_runtime_dependencies()
        assert self.datasource_service is not None
        assert self.ollama_client is not None
        assert self.output_repository is not None

        data_package = self.datasource_service.get_data_package(data_package_id)
        profile_manifest = None
        validation_schema: dict[str, Any] | None = None
        if profile_identifier is not None:
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
        persisted_state = self._load_run_state_or_none(data_package_id)
        progress = ExtractionRunProgress(
            stage="file_ranking",
            chunk_repair_mode=chunk_repair_mode,
            total_chunks=sum(len(chunks) for chunks in chunks_by_file),
            ranked_files=persisted_state.ranked_files if persisted_state else [],
            initial_file_summaries=(
                persisted_state.initial_file_summaries if persisted_state else []
            ),
            initial_file_summary_status=(
                persisted_state.initial_file_summary_status if persisted_state else None
            ),
            initial_extraction_overview=(
                persisted_state.initial_extraction_overview if persisted_state else None
            ),
            initial_extraction_overview_status=(
                persisted_state.initial_extraction_overview_status if persisted_state else None
            ),
            chunk_results=persisted_state.chunk_results if persisted_state else [],
            vocab_queries=persisted_state.vocab_queries if persisted_state else [],
            generated_final_draft=(
                persisted_state.generated_final_draft if persisted_state else None
            ),
            curated_document=(
                persisted_state.curated_document if persisted_state else None
            ),
            draft_quality_state=(
                persisted_state.draft_quality_state if persisted_state else None
            ),
            validation=(
                persisted_state.validation if persisted_state else DraftValidationResult()
            ),
            curated_validation=(
                persisted_state.curated_validation if persisted_state else None
            ),
            projection_ledger=(
                persisted_state.projection_ledger if persisted_state else []
            ),
            initial_draft_scaffold=(
                persisted_state.initial_draft_scaffold if persisted_state else {}
            ),
            field_completion_ledger=(
                persisted_state.field_completion_ledger if persisted_state else []
            ),
            curation_ledger=(
                persisted_state.curation_ledger if persisted_state else []
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
            chunk_repair_mode=chunk_repair_mode,
        )
        self._save_run_state(data_package_id, state)

        progress.chunk_repair_mode = state.chunk_repair_mode
        progress.ranked_files = state.ranked_files
        progress.initial_file_summaries = state.initial_file_summaries
        progress.initial_file_summary_status = state.initial_file_summary_status
        progress.initial_extraction_overview = state.initial_extraction_overview
        progress.initial_extraction_overview_status = state.initial_extraction_overview_status
        progress.chunk_results = state.chunk_results
        progress.vocab_query_config = state.vocab_query_config
        progress.vocab_queries = state.vocab_queries
        progress.generated_final_draft = state.generated_final_draft
        progress.curated_document = state.curated_document
        progress.draft_quality_state = state.draft_quality_state
        progress.validation = state.validation
        progress.curated_validation = state.curated_validation
        progress.projection_ledger = state.projection_ledger
        progress.field_completion_ledger = state.field_completion_ledger
        progress.curation_ledger = state.curation_ledger
        progress.total_chunks = len(state.chunk_results)
        progress.processed_chunks = self._completed_chunk_count(state)
        progress.interim_evidence_context = self._merged_completed_evidence_context_or_none(state)
        self._update_progress(data_package_id, progress)

        if (
            state.initial_file_summary_status is None
            or state.initial_extraction_overview_status is None
        ):
            progress.stage = "initial_context_required"
            progress.warnings = [
                *warnings,
                "Run initial file understanding before chunk extraction.",
            ]
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)
            raise ValueError(
                "Run initial file understanding before chunk extraction."
            )

        progress.stage = "chunk_extraction"
        chunk_repairs: list[tuple[ExtractionChunkResult, MaxRetriesExceeded]] = []

        try:
            for chunk_result, chunk in zip(state.chunk_results, ordered_chunks):
                if chunk_result.status in {"completed", "skipped"} and chunk_result.evidence_context is not None:
                    continue

                if is_noisy_payload_chunk(chunk.content):
                    chunk_result.status = "skipped"
                    chunk_result.error = None
                    chunk_result.skip_reason = "encoded_or_payload_dominated_chunk"
                    chunk_result.evidence_context = EvidenceContext()
                    progress.processed_chunks = self._completed_chunk_count(state)
                    progress.current_chunk = None
                    progress.chunk_results = state.chunk_results
                    self._save_run_state(data_package_id, state)
                    self._update_progress(data_package_id, progress)
                    continue

                chunk_result.status = "running"
                chunk_result.error = None
                chunk_result.skip_reason = None
                normalized_chunk_content = normalize_chunk_text_for_evidence_prompt(chunk.content)
                progress.current_chunk = self._chunk_ref(chunk_result)
                progress.chunk_results = state.chunk_results
                self._save_run_state(data_package_id, state)
                self._update_progress(data_package_id, progress)

                try:
                    result = await generate_structured(
                        self.ollama_client,
                        model=self.ollama_client.chat_model,
                        system=build_evidence_system_prompt_with_overview(
                            base_prompt=EVIDENCE_CONTEXT_SYSTEM_PROMPT,
                            overview=state.initial_extraction_overview,
                            overview_status=state.initial_extraction_overview_status,
                            file_summary=self._initial_file_summary_for_prompt(
                                state,
                                file_path=chunk.file_path,
                            ),
                        ),
                        prompt=build_evidence_context_prompt(
                            EvidenceChunkContext(
                                content=normalized_chunk_content,
                                metadata=EvidenceChunkMetadata(
                                    start_idx=chunk.start_idx,
                                    end_idx=chunk.end_idx,
                                    file_path=chunk.file_path,
                                    data_package_name=data_package.file_name,
                                ),
                            )
                        ),
                        output_type=EvidenceContext,
                        retries=2,
                        temperature=0.1,
                        think=None,
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
                    repairable = bool(exc.failed_response)
                    chunk_result.status = (
                        "repair_pending"
                        if repairable and chunk_repair_mode == "deferred"
                        else "running"
                        if repairable and chunk_repair_mode == "immediate"
                        else "failed"
                    )
                    chunk_result.error = (
                        "Queued for repair after first-pass extraction"
                        if repairable and chunk_repair_mode == "deferred"
                        else "Repairing first-pass structured output"
                        if repairable and chunk_repair_mode == "immediate"
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
                    if repairable and chunk_repair_mode == "deferred":
                        chunk_repairs.append((chunk_result, exc))
                    elif repairable and chunk_repair_mode == "immediate":
                        await self._repair_chunk_evidence_context(
                            data_package_id=data_package_id,
                            chunk_result=chunk_result,
                            failure=exc,
                            chunk_content=normalized_chunk_content,
                            state=state,
                            progress=progress,
                            warnings=warnings,
                        )
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
                validated_context = self._validate_and_filter_evidence_context_for_chunk(
                    result.output,
                    chunk_content=normalized_chunk_content,
                    chunk_result=chunk_result,
                    state=state,
                )
                chunk_result.status = "completed"
                chunk_result.evidence_context = validated_context
                chunk_result.response_duration_ms = self._usage_float(
                    result.usage,
                    "response_duration_ms",
                )
                chunk_result.context_tokens = self._usage_int(result.usage, "input_tokens")
                self._save_run_state(data_package_id, state)

                partial_context = self._save_current_evidence_artifacts(
                    data_package_id=data_package_id,
                    state=state,
                )
                progress.processed_chunks = self._completed_chunk_count(state)
                progress.interim_evidence_context = partial_context
                progress.current_chunk = None
                progress.chunk_results = state.chunk_results
                progress.warnings = list(warnings)
                self._update_progress(data_package_id, progress)

            if chunk_repairs:
                progress.stage = "chunk_repair"
                self._update_progress(data_package_id, progress)
        except asyncio.CancelledError:
            raise

        chunk_by_key = {self._chunk_key(chunk): chunk for chunk in ordered_chunks}
        for chunk_result, failure in chunk_repairs:
            repaired_chunk = chunk_by_key.get(self._chunk_result_key(chunk_result))
            repaired_chunk_content = (
                normalize_chunk_text_for_evidence_prompt(repaired_chunk.content)
                if repaired_chunk is not None
                else ""
            )
            await self._repair_chunk_evidence_context(
                data_package_id=data_package_id,
                chunk_result=chunk_result,
                failure=failure,
                chunk_content=repaired_chunk_content,
                state=state,
                progress=progress,
                warnings=warnings,
            )

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

        merged_context, duplicate_records = self._filtered_completed_evidence_context(state)
        evidence_context = self._evidence_context_with_file_inventory(
            data_package=data_package,
            context=merged_context,
            state=state,
        )
        self.output_repository.save_evidence_context(
            workflow_id=data_package_id,
            evidence_context=evidence_context,
        )
        self._save_filtered_evidence_notes(
            data_package_id=data_package_id,
            state=state,
            duplicate_records=duplicate_records,
        )
        warnings.extend(self._filtered_evidence_summary_warnings(state, duplicate_records))

        if target_stage == "context":
            progress.stage = "interim_evidence_context"
            progress.interim_evidence_context = evidence_context
            progress.warnings = list(warnings)
            self._save_run_state(data_package_id, state)
            self.output_repository.save_extraction_warnings(
                workflow_id=data_package_id,
                warnings=warnings,
            )
            self._update_progress(data_package_id, progress)
            return None

        progress.stage = "profile_projection"
        progress.interim_evidence_context = evidence_context
        self._update_progress(data_package_id, progress)
        if profile_identifier is None or profile_manifest is None or validation_schema is None:
            raise ValueError(
                "A profile identifier is required before building the generated profile draft."
            )

        profile_document = await self._build_profile_document_by_patching(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_manifest.target_class,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
            state=state,
            progress=progress,
            warnings=warnings,
        )

        if target_stage == "profile":
            progress.stage = "profile_draft"
            progress.interim_evidence_context = evidence_context
            progress.generated_final_draft = profile_document
            progress.curated_document = state.curated_document
            progress.draft_quality_state = state.draft_quality_state
            progress.validation = state.validation
            progress.curated_validation = state.curated_validation
            progress.initial_draft_scaffold = state.initial_draft_scaffold
            progress.projection_ledger = state.projection_ledger
            progress.field_completion_ledger = state.field_completion_ledger
            progress.curation_ledger = state.curation_ledger
            progress.vocab_queries = state.vocab_queries
            progress.warnings = list(warnings)
            self._save_run_state(data_package_id, state)
            self._persist_state_artifacts(data_package_id, state)
            self.output_repository.save_extraction_warnings(
                workflow_id=data_package_id,
                warnings=warnings,
            )
            self._update_progress(data_package_id, progress)
            return None

        progress.stage = "vocabulary_normalization"
        progress.interim_evidence_context = evidence_context
        progress.generated_final_draft = profile_document
        progress.curated_document = state.curated_document
        progress.draft_quality_state = state.draft_quality_state
        progress.validation = state.validation
        progress.curated_validation = state.curated_validation
        progress.initial_draft_scaffold = state.initial_draft_scaffold
        progress.projection_ledger = state.projection_ledger
        progress.field_completion_ledger = state.field_completion_ledger
        progress.curation_ledger = state.curation_ledger
        progress.vocab_queries = state.vocab_queries
        self._update_progress(data_package_id, progress)
        vocab_query_semaphore = asyncio.Semaphore(self._vocab_query_concurrency())

        def persist_vocab_progress() -> None:
            progress.chunk_results = state.chunk_results
            progress.vocab_query_config = state.vocab_query_config
            progress.vocab_queries = state.vocab_queries
            progress.generated_final_draft = state.generated_final_draft
            progress.curated_document = state.curated_document
            progress.draft_quality_state = state.draft_quality_state
            progress.validation = state.validation
            progress.curated_validation = state.curated_validation
            progress.initial_draft_scaffold = state.initial_draft_scaffold
            progress.projection_ledger = state.projection_ledger
            progress.field_completion_ledger = state.field_completion_ledger
            progress.curation_ledger = state.curation_ledger
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

        result = await self._save_profile_result(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            evidence_context=evidence_context,
            normalization=normalization,
            document=profile_document,
            profile_manifest=profile_manifest,
            validation_schema=validation_schema,
            state=state,
            warnings=warnings,
        )
        progress.stage = "completed"
        progress.interim_evidence_context = evidence_context
        progress.generated_final_draft = result.generated_final_draft
        progress.curated_document = result.curated_document
        progress.draft_quality_state = result.draft_quality_state
        progress.validation = result.validation
        progress.curated_validation = result.curated_validation
        progress.initial_draft_scaffold = result.initial_draft_scaffold
        progress.projection_ledger = result.projection_ledger
        progress.field_completion_ledger = result.field_completion_ledger
        progress.curation_ledger = result.curation_ledger
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

    async def _generate_initial_file_summaries(
        self,
        *,
        data_package_id: str,
        data_package: Any,
        ranking: FileRankingResult,
        state: ExtractionRunState,
        warnings: list[str],
    ) -> None:
        if not hasattr(self.ollama_client, "ollama_client"):
            state.initial_file_summaries = [
                self._failed_initial_file_summary(
                    ranked_file=ranked_file,
                    reason="No Ollama client is available for file summary generation.",
                )
                for ranked_file in self._top_initial_context_ranked_files(ranking)
            ]
            state.initial_file_summary_status = "failed"
            self._save_run_state(data_package_id, state)
            self._persist_initial_file_summaries(data_package_id, state)
            return

        files_by_path = {file.file_path: file for file in data_package.files}
        summaries: list[ExtractionFileSummary] = []
        skipped_count = 0
        for ranked_file in self._top_initial_context_ranked_files(ranking):
            file_entry = files_by_path.get(ranked_file.file_path)
            if file_entry is None:
                summaries.append(
                    self._failed_initial_file_summary(
                        ranked_file=ranked_file,
                        reason="Ranked file is not present in the data package.",
                    )
                )
                continue
            if self._should_skip_initial_file_summary(file_entry):
                skipped_count += 1
                warnings.append(
                    f"Initial file summary skipped for {ranked_file.file_path}: image files are not text-extractable."
                )
                continue
            try:
                extracted_content = file_entry.get_extracted_content()
                content_windows = self._initial_file_summary_content_windows(
                    extracted_content,
                    num_ctx=self.ollama_client.max_context_length,
                )
                source_fingerprint = self._initial_file_summary_source_fingerprint(
                    data_package=data_package,
                    ranked_file=ranked_file,
                    byte_size=len(file_entry.raw_content),
                    extracted_char_count=len(extracted_content),
                    content_windows=content_windows,
                )
                result = await generate_structured(
                    self.ollama_client,
                    model=self.ollama_client.chat_model,
                    system=EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT,
                    prompt=build_extraction_file_summary_prompt(
                        data_package_name=data_package.file_name,
                        rank=ranked_file.rank,
                        file_path=ranked_file.file_path,
                        byte_size=len(file_entry.raw_content),
                        extracted_char_count=len(extracted_content),
                        content_windows=content_windows,
                    ),
                    output_type=ExtractionFileSummary,
                    retries=1,
                    temperature=0.0,
                    think=None,
                    num_ctx=self.ollama_client.max_context_length,
                )
                self._record_workflow_token_usage(
                    data_package_id=data_package_id,
                    agent_name="initial_file_summary",
                    usage=result.usage,
                )
                summaries.append(
                    self._validated_initial_file_summary(
                        result.output,
                        ranked_file=ranked_file,
                        source_fingerprint=source_fingerprint,
                        sampled_text="\n".join(
                            window.text for window in content_windows
                        ),
                        warnings=warnings,
                    )
                )
            except CompletionError as exc:
                usage = getattr(exc, "usage", None)
                if usage is not None:
                    self._record_workflow_token_usage(
                        data_package_id=data_package_id,
                        agent_name="initial_file_summary",
                        usage=usage,
                    )
                warnings.append(
                    f"Initial file summary failed for {ranked_file.file_path}: {exc}"
                )
                summaries.append(
                    self._failed_initial_file_summary(
                        ranked_file=ranked_file,
                        reason=str(exc),
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"Initial file summary failed for {ranked_file.file_path}: {exc}"
                )
                logger.warning(
                    "Initial file summary failed",
                    extra={
                        "data_package_id": data_package_id,
                        "file_path": ranked_file.file_path,
                        "error_type": type(exc).__name__,
                    },
                )
                summaries.append(
                    self._failed_initial_file_summary(
                        ranked_file=ranked_file,
                        reason=str(exc),
                    )
                )

        state.initial_file_summaries = summaries
        summarized_count = sum(1 for summary in summaries if summary.status == "summarized")
        if summarized_count == len(summaries) and summaries:
            state.initial_file_summary_status = "completed"
        elif summarized_count > 0:
            state.initial_file_summary_status = "partial"
        elif skipped_count > 0:
            state.initial_file_summary_status = "completed"
        else:
            state.initial_file_summary_status = "failed"
        self._save_run_state(data_package_id, state)
        self._persist_initial_file_summaries(data_package_id, state)

    async def _generate_initial_extraction_overview(
        self,
        *,
        data_package_id: str,
        data_package: Any,
        ranking: FileRankingResult,
        state: ExtractionRunState,
        warnings: list[str],
    ) -> None:
        if not hasattr(self.ollama_client, "ollama_client"):
            state.initial_extraction_overview = None
            state.initial_extraction_overview_status = "failed"
            self._save_run_state(data_package_id, state)
            self._persist_initial_extraction_overview(data_package_id, state)
            return
        previews = self._initial_overview_file_previews(
            data_package=data_package,
            ranking=ranking,
        )
        preview_file_paths = {preview.file_path for preview in previews}
        overview_ranked_files = [
            file
            for file in ranking.files
            if file.file_path in preview_file_paths
        ]
        source_fingerprint = self._initial_overview_source_fingerprint(
            data_package=data_package,
            ranking=FileRankingResult(files=overview_ranked_files),
            previews=previews,
        )
        summarized_file_summaries = self._summarized_initial_file_summaries(state)
        fallback_previews = [] if summarized_file_summaries else previews
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EXTRACTION_OVERVIEW_SYSTEM_PROMPT,
                prompt=build_extraction_overview_prompt(
                    data_package_name=data_package.file_name,
                    ranked_files=overview_ranked_files,
                    file_summaries=summarized_file_summaries,
                    file_previews=fallback_previews,
                ),
                output_type=ExtractionOverview,
                retries=1,
                temperature=0.1,
                think=None,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="initial_extraction_overview",
                usage=result.usage,
            )
            sanitized_overview = self._sanitize_initial_overview_file_roles(
                result.output,
                allowed_file_paths=preview_file_paths,
                warnings=warnings,
            )
            state.initial_extraction_overview = self._with_initial_overview_provenance(
                sanitized_overview,
                source_fingerprint=source_fingerprint,
                previews=previews,
            )
            state.initial_extraction_overview_status = "structured"
            self._save_run_state(data_package_id, state)
            self._persist_initial_extraction_overview(data_package_id, state)
            return
        except CompletionError as exc:
            usage = getattr(exc, "usage", None)
            if usage is not None:
                self._record_workflow_token_usage(
                    data_package_id=data_package_id,
                    agent_name="initial_extraction_overview",
                    usage=usage,
                )
            warnings.append(
                "Initial extraction overview structured generation failed; using free-text fallback."
            )
            logger.warning(
                "Initial extraction overview structured generation failed",
                extra={
                    "data_package_id": data_package_id,
                    "error_type": type(exc).__name__,
                },
            )

        try:
            response = await self.ollama_client.ollama_client.generate(
                model=self.ollama_client.chat_model,
                system=EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT,
                prompt=build_extraction_overview_fallback_prompt(
                    data_package_name=data_package.file_name,
                    ranked_files=overview_ranked_files,
                    file_previews=fallback_previews,
                ),
                options={
                    "temperature": 0.1,
                    "seed": 42,
                    "num_ctx": self.ollama_client.max_context_length,
                },
                think=None,
                keep_alive=-1,
            )
            usage = RunUsage.from_ollama_response(response)
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="initial_extraction_overview_fallback",
                usage=usage,
            )
            text = (response.response or "").strip()
            if not text:
                raise CompletionError("Initial extraction overview fallback returned an empty response.")
            state.initial_extraction_overview = self._with_initial_overview_provenance(
                ExtractionOverview(
                    observed_signals=[text],
                    conflicts_or_uncertainties=[
                        "This overview is an unstructured fallback and is orientation only."
                    ],
                ),
                source_fingerprint=source_fingerprint,
                previews=previews,
            )
            state.initial_extraction_overview_status = "unstructured_fallback"
        except Exception as exc:
            warnings.append(
                f"Initial extraction overview fallback failed; continuing without overview: {exc}"
            )
            logger.warning(
                "Initial extraction overview fallback failed",
                extra={
                    "data_package_id": data_package_id,
                    "error_type": type(exc).__name__,
                },
            )
            state.initial_extraction_overview = None
            state.initial_extraction_overview_status = "failed"

        self._save_run_state(data_package_id, state)
        self._persist_initial_extraction_overview(data_package_id, state)

    def _initial_overview_file_previews(
        self,
        *,
        data_package: Any,
        ranking: FileRankingResult,
    ) -> list[ExtractionOverviewFilePreview]:
        files_by_path = {file.file_path: file for file in data_package.files}
        previews: list[ExtractionOverviewFilePreview] = []
        for ranked_file in sorted(ranking.files, key=lambda item: item.rank)[
            :INITIAL_OVERVIEW_TOP_FILE_LIMIT
        ]:
            file_entry = files_by_path.get(ranked_file.file_path)
            if file_entry is None:
                continue
            if self._should_skip_initial_file_summary(file_entry):
                continue
            try:
                lines = file_entry.get_extracted_content().splitlines()
            except Exception as exc:
                lines = [f"[Text extraction failed: {exc}]"]
            previews.append(
                ExtractionOverviewFilePreview(
                    rank=ranked_file.rank,
                    file_path=ranked_file.file_path,
                    byte_size=len(file_entry.raw_content),
                    first_lines=[
                        self._truncate_overview_preview_line(line)
                        for line in lines[:INITIAL_OVERVIEW_PREVIEW_LINE_LIMIT]
                    ],
                )
            )
        return previews

    @staticmethod
    def _truncate_overview_preview_line(line: str) -> str:
        if len(line) <= INITIAL_OVERVIEW_MAX_LINE_CHARS:
            return line
        return line[: INITIAL_OVERVIEW_MAX_LINE_CHARS - 3].rstrip() + "..."

    @staticmethod
    def _initial_overview_source_fingerprint(
        *,
        data_package: Any,
        ranking: FileRankingResult,
        previews: list[ExtractionOverviewFilePreview],
    ) -> str:
        payload = {
            "data_package_name": getattr(data_package, "file_name", ""),
            "ranked_files": [file.model_dump(mode="json") for file in ranking.files],
            "previews": [preview.model_dump(mode="json") for preview in previews],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha1(encoded).hexdigest()

    @staticmethod
    def _top_initial_context_ranked_files(ranking: FileRankingResult) -> list[Any]:
        return sorted(ranking.files, key=lambda item: item.rank)[
            :INITIAL_OVERVIEW_TOP_FILE_LIMIT
        ]

    @staticmethod
    def _should_skip_initial_file_summary(file_entry: Any) -> bool:
        return getattr(file_entry, "file_type", None) == FileType.IMAGE

    @staticmethod
    def _initial_file_summary_content_windows(
        content: str,
        *,
        num_ctx: int | None,
    ) -> list[ExtractionFileContentWindow]:
        budget = max(
            1200,
            int(
                (num_ctx or 8192)
                * ESTIMATED_CHARS_PER_TOKEN
                * INITIAL_FILE_SUMMARY_CONTEXT_RATIO
            ),
        )
        if len(content) <= budget:
            return [
                ExtractionFileContentWindow(
                    label="full",
                    start_char_idx=0,
                    end_char_idx=len(content),
                    text=content,
                )
            ]

        window_size = max(400, budget // 3)
        middle_start = max(0, (len(content) - window_size) // 2)
        ranges = [
            ("beginning", 0, min(window_size, len(content))),
            ("middle", middle_start, min(middle_start + window_size, len(content))),
            ("end", max(0, len(content) - window_size), len(content)),
        ]
        return [
            ExtractionFileContentWindow(
                label=label,
                start_char_idx=start,
                end_char_idx=end,
                omitted_before_chars=start,
                omitted_after_chars=max(0, len(content) - end),
                text=content[start:end],
            )
            for label, start, end in ranges
        ]

    @staticmethod
    def _initial_file_summary_source_fingerprint(
        *,
        data_package: Any,
        ranked_file: Any,
        byte_size: int | None,
        extracted_char_count: int,
        content_windows: list[ExtractionFileContentWindow],
    ) -> str:
        payload = {
            "data_package_name": getattr(data_package, "file_name", ""),
            "rank": ranked_file.rank,
            "file_path": ranked_file.file_path,
            "byte_size": byte_size,
            "extracted_char_count": extracted_char_count,
            "content_windows": [
                window.model_dump(mode="json") for window in content_windows
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha1(encoded).hexdigest()

    @staticmethod
    def _validated_initial_file_summary(
        summary: ExtractionFileSummary,
        *,
        ranked_file: Any,
        source_fingerprint: str,
        sampled_text: str,
        warnings: list[str],
    ) -> ExtractionFileSummary:
        update: dict[str, Any] = {
            "source_fingerprint": source_fingerprint,
            "file_path": ranked_file.file_path,
            "rank": ranked_file.rank,
            "status": "summarized",
        }
        if summary.file_path != ranked_file.file_path:
            warnings.append(
                "Initial file summary returned a mismatched file_path; "
                f"expected {ranked_file.file_path}, got {summary.file_path}."
            )
        if summary.rank != ranked_file.rank:
            warnings.append(
                "Initial file summary returned a mismatched rank; "
                f"expected {ranked_file.rank}, got {summary.rank}."
            )
        verified_purpose_evidence = [
            evidence
            for evidence in summary.purpose_evidence
            if evidence and evidence in sampled_text
        ]
        if len(verified_purpose_evidence) != len(summary.purpose_evidence):
            warnings.append(
                f"Initial file summary for {ranked_file.file_path} included purpose evidence not found in the sampled file text; dropping unsupported evidence."
            )
            update["purpose_evidence"] = verified_purpose_evidence
        if summary.explicit_purpose and not verified_purpose_evidence:
            warnings.append(
                f"Initial file summary for {ranked_file.file_path} included an explicit purpose without evidence; clearing it."
            )
            update["explicit_purpose"] = ""
            update["uncertainty_notes"] = [
                *summary.uncertainty_notes,
                "No direct purpose evidence was provided by the summary output.",
            ]
        return summary.model_copy(update=update)

    @staticmethod
    def _failed_initial_file_summary(
        *,
        ranked_file: Any,
        reason: str,
    ) -> ExtractionFileSummary:
        return ExtractionFileSummary(
            file_path=ranked_file.file_path,
            rank=ranked_file.rank,
            status="failed",
            uncertainty_notes=[reason],
            known_traps=[
                "No reliable file summary is available; use only current chunk evidence."
            ],
        )

    @staticmethod
    def _summarized_initial_file_summaries(
        state: ExtractionRunState,
    ) -> list[ExtractionFileSummary]:
        return [
            summary
            for summary in state.initial_file_summaries
            if summary.status == "summarized"
        ]

    @staticmethod
    def _sanitize_initial_overview_file_roles(
        overview: ExtractionOverview,
        *,
        allowed_file_paths: set[str],
        warnings: list[str],
    ) -> ExtractionOverview:
        filtered_roles = [
            role for role in overview.file_roles if role.file_path in allowed_file_paths
        ]
        dropped = len(overview.file_roles) - len(filtered_roles)
        if dropped:
            warnings.append(
                f"Initial extraction overview referenced {dropped} unknown file role(s); they were removed."
            )
        return overview.model_copy(update={"file_roles": filtered_roles})

    @staticmethod
    def _with_initial_overview_provenance(
        overview: ExtractionOverview,
        *,
        source_fingerprint: str,
        previews: list[ExtractionOverviewFilePreview],
    ) -> ExtractionOverview:
        source_file_paths = [preview.file_path for preview in previews]
        inspected_by_path = {
            inspected.file_path: inspected
            for inspected in overview.inspected_files
            if inspected.file_path in source_file_paths
        }
        inspected_files = [
            ExtractionOverviewInspectedFile(
                file_path=preview.file_path,
                byte_size=preview.byte_size,
                chars_read=sum(len(line) for line in preview.first_lines),
                reason=inspected_by_path.get(preview.file_path, ExtractionOverviewInspectedFile(file_path=preview.file_path)).reason,
            )
            for preview in previews
        ]
        return overview.model_copy(
            update={
                "source_fingerprint": source_fingerprint,
                "source_file_paths": source_file_paths,
                "inspected_files": inspected_files,
            }
        )

    @staticmethod
    def _initial_overview_matches_current_run(
        *,
        overview: ExtractionOverview | None,
        status: ExtractionOverviewStatus | None,
        ranking: FileRankingResult,
    ) -> bool:
        if status is None or overview is None:
            return False
        source_file_paths = overview.source_file_paths
        if not source_file_paths:
            return False
        ranked_paths = [file.file_path for file in sorted(ranking.files, key=lambda item: item.rank)]
        ranked_path_set = set(ranked_paths)
        if any(path not in ranked_path_set for path in source_file_paths):
            return False
        source_index = 0
        for ranked_path in ranked_paths:
            if (
                source_index < len(source_file_paths)
                and source_file_paths[source_index] == ranked_path
            ):
                source_index += 1
        if source_index != len(source_file_paths):
            return False
        overview_role_paths = {role.file_path for role in overview.file_roles}
        if any(path not in ranked_paths for path in overview_role_paths):
            return False
        return True

    @staticmethod
    def _initial_file_summaries_match_current_run(
        *,
        summaries: list[ExtractionFileSummary],
        status: str | None,
        ranking: FileRankingResult,
    ) -> bool:
        if status is None or not summaries:
            return False
        ranked_paths = [
            file.file_path
            for file in sorted(ranking.files, key=lambda item: item.rank)[
                :INITIAL_OVERVIEW_TOP_FILE_LIMIT
            ]
        ]
        summary_paths = [summary.file_path for summary in summaries]
        if summary_paths != ranked_paths[: len(summary_paths)]:
            return False
        if any(summary.rank < 1 for summary in summaries):
            return False
        return True

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
        evidence_context = self._merged_completed_evidence_context(state)
        profile_document = state.generated_final_draft or self._fallback_profile_document(
            data_package_id=data_package_id,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
        )
        normalization = await self._normalize_profile_fields_from_state_vocab_queries(
            data_package_id=data_package_id,
            state=state,
            warnings=warnings,
        )
        result = await self._save_profile_result(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            evidence_context=evidence_context,
            normalization=normalization,
            document=profile_document,
            profile_manifest=profile_manifest,
            validation_schema=validation_schema,
            state=state,
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
                    interim_evidence_context=evidence_context,
                    vocab_query_config=state.vocab_query_config,
                    ranked_files=state.ranked_files,
                    initial_file_summaries=state.initial_file_summaries,
                    initial_file_summary_status=state.initial_file_summary_status,
                    initial_extraction_overview=state.initial_extraction_overview,
                    initial_extraction_overview_status=state.initial_extraction_overview_status,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    generated_final_draft=result.generated_final_draft,
                    curated_document=result.curated_document,
                    draft_quality_state=result.draft_quality_state,
                    validation=result.validation,
                    curated_validation=result.curated_validation,
                    initial_draft_scaffold=state.initial_draft_scaffold,
                    projection_ledger=result.projection_ledger,
                    field_completion_ledger=result.field_completion_ledger,
                    curation_ledger=result.curation_ledger,
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
        evidence_context: EvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> dict[str, Any]:
        if state.generated_final_draft is None:
            document, scaffold = self._initial_profile_document(
                data_package_id=data_package_id,
                evidence_context=evidence_context,
                validation_schema=validation_schema,
            )
            state.initial_draft_scaffold = scaffold
            state.generated_final_draft = document
            progress.generated_final_draft = document
            progress.initial_draft_scaffold = scaffold
        else:
            document = state.generated_final_draft
            progress.initial_draft_scaffold = state.initial_draft_scaffold
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        if not validation.valid:
            warnings.append(
                "Initial profile skeleton was not schema-valid: "
                + "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
            )

        target_catalog = self._target_catalog_from_document(
            document=document,
            validation_schema=validation_schema,
            scaffold=state.initial_draft_scaffold,
        )
        schema_branches = self._schema_branches_for_profile(
            profile_identifier=profile_identifier,
            profile_target_class=profile_target_class,
            warnings=warnings,
        )
        patched_identifiers = {
            record.object_identifier
            for record in state.projection_ledger
        }
        for group in self._projection_groups_for_evidence(evidence_context):
            if group.group_id in patched_identifiers:
                continue
            schema_searches, schema_candidate_branches = self._schema_candidates_for_group(
                group=group,
                schema_branches=schema_branches,
            )
            group_target_catalog = (
                schema_branches_to_catalog(schema_candidate_branches)
                + target_catalog
            )
            target_decision = await self._plan_profile_target_for_evidence_group(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                profile_target_class=profile_target_class,
                group=group,
                file_inventory=evidence_context.file_inventory,
                warnings=warnings,
                target_catalog=group_target_catalog,
            )
            selected_schema_branch = self._selected_schema_branch(
                target_decision.target_path,
                schema_candidate_branches,
            )
            if target_decision.status == "skip" or not target_decision.target_path:
                patch_result = ProfileObjectPatchResult(
                    object_identifier=group.group_id,
                    object_kind=group.object_kind,
                    status="skipped",
                    target_path=target_decision.target_path,
                    target_class=target_decision.target_class,
                    planner_status=target_decision.status,
                    planner_reason=target_decision.reason,
                    reason=target_decision.reason,
                    schema_queries=[search.model_dump(mode="json") for search in schema_searches],
                    candidate_paths=[branch.path for branch in schema_candidate_branches],
                    selected_schema_branch=(
                        selected_schema_branch.model_dump(mode="json")
                        if selected_schema_branch
                        else None
                    ),
                )
            else:
                patch_result = await self._write_profile_target_with_evidence_group(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    profile_target_class=profile_target_class,
                    current_document=document,
                    group=group,
                    target_decision=target_decision,
                    validation_schema=validation_schema,
                    file_inventory=evidence_context.file_inventory,
                    warnings=warnings,
                )
                patch_result.schema_queries = [
                    search.model_dump(mode="json") for search in schema_searches
                ]
                patch_result.candidate_paths = [
                    branch.path for branch in schema_candidate_branches
                ]
                patch_result.selected_schema_branch = (
                    selected_schema_branch.model_dump(mode="json")
                    if selected_schema_branch
                    else None
                )
            if patch_result.status == "applied":
                document = self._apply_profile_target_write(
                    document,
                    patch_result.target_path or group.target_hint,
                    patch_result.target_value,
                )
                target_catalog = self._target_catalog_from_document(
                    document=document,
                    validation_schema=validation_schema,
                    scaffold=state.initial_draft_scaffold,
                )
            projection_record = self._projection_record_from_group_patch_result(
                group=group,
                patch_result=patch_result,
            )
            state.projection_ledger = [
                record
                for record in state.projection_ledger
                if record.object_identifier != group.group_id
            ]
            state.projection_ledger.append(projection_record)
            state.generated_final_draft = document
            progress.generated_final_draft = document
            progress.initial_draft_scaffold = state.initial_draft_scaffold
            progress.projection_ledger = state.projection_ledger
            progress.warnings = list(warnings)
            self._save_run_state(data_package_id, state)
            self._persist_state_artifacts(data_package_id, state)
            self._update_progress(data_package_id, progress)
        return document

    def _schema_branches_for_profile(
        self,
        *,
        profile_identifier: str,
        profile_target_class: str,
        warnings: list[str],
    ) -> list[SchemaBranch]:
        try:
            merged_schema = self.profile_service.load_merged_schema(profile_identifier)
            return build_schema_branch_index(
                merged_schema,
                target_class=profile_target_class,
                max_depth=3,
            )
        except Exception as exc:
            warnings.append(
                f"Schema-guided projection disabled for '{profile_identifier}': {exc}"
            )
            return []

    @staticmethod
    def _schema_candidates_for_group(
        *,
        group: _EvidenceProjectionGroup,
        schema_branches: list[SchemaBranch],
    ):
        if not schema_branches:
            return [], []
        first_query = build_schema_search_query(group.notes, max_depth=3)
        first_result = search_schema_branches(schema_branches, first_query, top_k=8)
        searches = [first_result]
        candidates = first_result.candidates
        if not candidates:
            relaxed_query = first_query.model_copy(update={"max_depth": 4})
            relaxed_result = search_schema_branches(
                schema_branches,
                relaxed_query,
                top_k=8,
            )
            searches.append(relaxed_result)
            candidates = relaxed_result.candidates
        return searches, candidates

    @staticmethod
    def _selected_schema_branch(
        target_path: str | None,
        branches: list[SchemaBranch],
    ) -> SchemaBranch | None:
        if not target_path:
            return None
        return next((branch for branch in branches if branch.path == target_path), None)

    @staticmethod
    def _projection_identifier_for_evidence_note(note: EvidenceNote) -> str:
        file_path = note.file_path or "unknown-file"
        line_span = f"{note.start_idx}-{note.end_idx}"
        note_id = note.note_id or "unnamed-note"
        return f"{file_path}#{line_span}#{note_id}"

    @staticmethod
    def _projection_record_from_patch_result(
        *,
        note: EvidenceNote,
        patch_result: ProfileObjectPatchResult,
    ) -> ProjectionLedgerRecord:
        status = "not_projected"
        if patch_result.status == "applied":
            status = "projected"
        elif patch_result.status == "failed":
            status = "user_edit_required"
        return ProjectionLedgerRecord(
            object_identifier=patch_result.object_identifier,
            object_kind=patch_result.object_kind,
            source_evidence=note.evidence_text,
            status=status,
            projected_paths=[
                operation.path
                for operation in patch_result.operations
                if getattr(operation, "path", None)
            ],
            reason=patch_result.reason,
            error=patch_result.error,
        )

    @classmethod
    def _projection_record_from_group_patch_result(
        cls,
        *,
        group: _EvidenceProjectionGroup,
        patch_result: ProfileObjectPatchResult,
    ) -> ProjectionLedgerRecord:
        status = "not_projected"
        if patch_result.status == "applied":
            status = "projected"
        elif patch_result.status == "failed":
            status = "user_edit_required"
        projected_paths = [
            operation.path
            for operation in patch_result.operations
            if getattr(operation, "path", None)
        ]
        if patch_result.status == "applied" and not projected_paths and patch_result.target_path:
            projected_paths = [patch_result.target_path]
        return ProjectionLedgerRecord(
            object_identifier=patch_result.object_identifier,
            object_kind=patch_result.object_kind,
            source_evidence="\n".join(note.evidence_text for note in group.notes if note.evidence_text),
            evidence_note_identifiers=[
                cls._projection_identifier_for_evidence_note(note)
                for note in group.notes
            ],
            status=status,
            projected_paths=projected_paths,
            target_path=patch_result.target_path,
            target_class=patch_result.target_class,
            planner_status=patch_result.planner_status,
            planner_reason=patch_result.planner_reason,
            evidence_quality=cls._evidence_quality_summary(group.notes),
            schema_queries=patch_result.schema_queries,
            candidate_paths=patch_result.candidate_paths,
            selected_schema_branch=patch_result.selected_schema_branch,
            merge_status=patch_result.merge_status,
            reason=patch_result.reason,
            error=patch_result.error,
        )

    @staticmethod
    def _evidence_quality_summary(notes: list[EvidenceNote]) -> dict[str, Any]:
        signal_level: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for note in notes:
            signal_level[note.signal_level] = signal_level.get(note.signal_level, 0) + 1
        return {
            "note_count": len(notes),
            "signal_level": signal_level,
        }

    @classmethod
    def _projection_groups_for_evidence(
        cls,
        evidence_context: EvidenceContext,
        *,
        max_group_size: int = 6,
    ) -> list[_EvidenceProjectionGroup]:
        buckets: dict[tuple[str, str, str, str], list[EvidenceNote]] = {}
        for note in evidence_context.notes:
            if not cls._note_has_curatable_profile_signal(note):
                continue
            target_hint, target_class_hint = cls._target_hint_for_evidence_note(note)
            family = cls._evidence_note_family(note.note_id)
            key = (target_hint, target_class_hint or "", note.category, f"{note.file_path}:{family}")
            buckets.setdefault(key, []).append(note)

        groups: list[_EvidenceProjectionGroup] = []
        for (target_hint, target_class_hint, category, _family_key), notes in buckets.items():
            for index in range(0, len(notes), max_group_size):
                chunk = notes[index : index + max_group_size]
                note_ids = [cls._projection_identifier_for_evidence_note(note) for note in chunk]
                digest = sha1("|".join(note_ids).encode("utf-8")).hexdigest()[:12]
                groups.append(
                    _EvidenceProjectionGroup(
                        group_id=f"group:{target_hint.strip('/').replace('/', '.') or 'root'}:{digest}",
                        object_kind=category,
                        notes=chunk,
                        target_hint=target_hint,
                        target_class_hint=target_class_hint or None,
                    )
                )
        return groups

    @staticmethod
    def _evidence_note_family(note_id: str) -> str:
        family = note_id or "unnamed"
        while family and (family[-1].isdigit() or family[-1] in {"_", "-", "."}):
            family = family[:-1]
        return family or note_id or "unnamed"

    @classmethod
    def _target_hint_for_evidence_note(cls, note: EvidenceNote) -> tuple[str, str | None]:
        text = f"{note.note_id} {note.category} {note.observation} {note.evidence_text}".lower()
        if cls._note_has_device_signal(note):
            return "/was_generated_by/0/carried_out_by/-", "AgenticEntity"
        if note.category == "agent_signal" or any(term in text for term in ("origin", "owner", "creator", "author")):
            return "/creator/0", "Agent"
        if note.category in {"entity_signal", "measurement_signal"}:
            return "/is_about_entity/0", "EvaluatedEntity"
        if any(term in text for term in ("format", "jcamp", "file", "distribution", "download", "access")):
            return "/dataset_distribution/0", "Distribution"
        if note.category == "method_signal" or any(term in text for term in ("pulse", "method", "experiment", "acquisition")):
            return "/was_generated_by/0", "DataGeneratingActivity"
        if any(term in text for term in ("dataset name", "title", "spectrum title")):
            return "/title", None
        if any(term in text for term in ("date", "timestamp", "modified", "modification")):
            return "/modification_date", None
        if any(term in text for term in ("type", "category", "class")):
            return "/type/0", "Concept"
        if note.category == "resource_signal":
            return "/dataset_distribution/0", "Distribution"
        return "/description", None

    @classmethod
    def _note_has_device_signal(cls, note: EvidenceNote) -> bool:
        text = cls._note_search_text(note)
        return any(
            term in text
            for term in (
                "instrument",
                "device",
                "spectrometer",
                "probehead",
                "bruker avance",
            )
        )

    async def _plan_profile_target_for_evidence_group(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        group: _EvidenceProjectionGroup,
        file_inventory: list[Any],
        warnings: list[str],
        target_catalog: list[dict[str, Any]],
    ) -> ProfileTargetDecision:
        assert self.ollama_client is not None
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=PROFILE_TARGET_PLANNER_SYSTEM_PROMPT,
                prompt=build_profile_target_planner_prompt(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    profile_target_class=profile_target_class,
                    evidence_notes=group.notes,
                    target_catalog=target_catalog,
                    file_inventory=file_inventory,
                ),
                output_type=ProfileTargetDecision,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="profile_target_planner",
                usage=result.usage,
            )
            decision = (
                result.output
                if isinstance(result.output, ProfileTargetDecision)
                else ProfileTargetDecision.model_validate(result.output)
            )
        except CompletionError as exc:
            warnings.append(f"Profile target planning fell back for '{group.group_id}': {exc}")
            decision = self._fallback_target_decision_for_group(group, target_catalog)
        catalog_paths = {
            item.get("path")
            for item in target_catalog
            if isinstance(item.get("path"), str)
        }
        if decision.target_path not in catalog_paths:
            fallback = self._fallback_target_decision_for_group(group, target_catalog)
            fallback.reason = (
                f"Planner selected unavailable target {decision.target_path!r}; "
                f"using deterministic hint {fallback.target_path!r}."
            )
            return fallback
        return decision

    @staticmethod
    def _fallback_target_decision_for_group(
        group: _EvidenceProjectionGroup,
        target_catalog: list[dict[str, Any]],
    ) -> ProfileTargetDecision:
        paths = {
            item.get("path"): item
            for item in target_catalog
            if isinstance(item.get("path"), str)
        }
        target = paths.get(group.target_hint) or paths.get("/description")
        if target is None:
            return ProfileTargetDecision(
                status="skip",
                reason="No usable projection target is available.",
            )
        return ProfileTargetDecision(
            status="targeted",
            target_path=target["path"],
            target_class=target.get("target_class") or group.target_class_hint,
            target_label=target.get("label", ""),
            reason="Deterministic evidence-category target hint.",
        )

    async def _patch_profile_with_extraction_object(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        current_document: dict[str, Any],
        evidence_note: EvidenceNote,
        object_identifier: str,
        file_inventory: list[Any],
        schema_slice: dict[str, Any],
        warnings: list[str],
    ) -> ProfileObjectPatchResult:
        assert self.ollama_client is not None
        object_kind = evidence_note.category
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
                    evidence_notes=[evidence_note],
                    file_inventory=file_inventory,
                    schema_slice=schema_slice,
                    target_path="/description",
                    target_class=None,
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

    async def _write_profile_target_with_evidence_group(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        current_document: dict[str, Any],
        group: _EvidenceProjectionGroup,
        target_decision: ProfileTargetDecision,
        validation_schema: dict[str, Any],
        file_inventory: list[Any],
        warnings: list[str],
    ) -> ProfileObjectPatchResult:
        assert self.ollama_client is not None
        target_path = target_decision.target_path or group.target_hint
        target_schema = self._schema_slice_for_json_path(
            validation_schema,
            target_path,
            max_depth=3,
        )
        current_target_value = self._value_at_json_pointer(current_document, target_path)
        unsuitable_reason = self._profile_target_unsuitable_reason(
            target_path=target_path,
            notes=group.notes,
        )
        if unsuitable_reason:
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                reason=unsuitable_reason,
            )
        if target_path == "/description" and not self._description_target_worthy(group.notes):
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                reason="Description is reserved for dataset-level prose; grouped evidence is better handled by structured targets or skipped.",
            )
        try:
            write = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=PROFILE_TARGET_WRITER_SYSTEM_PROMPT,
                prompt=build_profile_target_write_prompt(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    profile_target_class=profile_target_class,
                    target_path=target_path,
                    target_class=target_decision.target_class,
                    target_label=target_decision.target_label,
                    current_target_value=current_target_value,
                    evidence_notes=group.notes,
                    file_inventory=file_inventory,
                    schema_slice=target_schema,
                ),
                output_type=ProfileTargetWriteDocument,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="profile_target_writer",
                usage=write.usage,
            )
        except CompletionError as exc:
            fallback = self._deterministic_profile_target_write_result(
                profile_identifier=profile_identifier,
                current_document=current_document,
                group=group,
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                current_target_value=current_target_value,
                reason_prefix="Deterministic schema-safe target fallback after model writer failure",
            )
            if fallback is not None:
                warnings.append(
                    f"Profile target write used deterministic fallback for '{group.group_id}' after model failure: {exc}"
                )
                return fallback
            warnings.append(f"Profile target write failed for '{group.group_id}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                error=str(exc),
            )

        write_document = (
            write.output
            if isinstance(write.output, ProfileTargetWriteDocument)
            else ProfileTargetWriteDocument.model_validate(write.output)
        )
        if write_document.status == "skip":
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                reason=write_document.reason or target_decision.reason,
            )

        target_value = self._coerce_profile_target_value(
            target_path=target_path,
            current_value=current_target_value,
            proposed_value=write_document.value,
        )
        target_value = self._curate_profile_target_value(
            target_path=target_path,
            value=target_value,
        )
        if self._target_write_is_empty(current_target_value, target_value):
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                reason="Target write only contained low-level instrument configuration or duplicate values.",
            )
        try:
            candidate = self._apply_profile_target_write(
                current_document,
                target_path,
                target_value,
            )
        except Exception as exc:
            warnings.append(f"Profile target write skipped for '{group.group_id}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                target_value=target_value,
                error=str(exc),
                reason=write_document.reason,
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=candidate,
        )
        if not validation.valid:
            error = "; ".join(
                f"{issue.path}: {issue.message}" for issue in validation.errors
            )
            fallback = self._deterministic_profile_target_write_result(
                profile_identifier=profile_identifier,
                current_document=current_document,
                group=group,
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                current_target_value=current_target_value,
                reason_prefix=f"Deterministic schema-safe target fallback after invalid writer value ({error})",
            )
            if fallback is not None:
                warnings.append(
                    f"Profile target write repaired for '{group.group_id}' with deterministic fallback after validation failed: {error}"
                )
                return fallback
            warnings.append(
                f"Profile target write skipped for '{group.group_id}' because it broke schema validation: {error}"
            )
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                target_value=target_value,
                error=error,
                reason=write_document.reason,
            )
        return ProfileObjectPatchResult(
            object_identifier=group.group_id,
            object_kind=group.object_kind,
            status="applied",
            target_path=target_path,
            target_class=target_decision.target_class,
            planner_status=target_decision.status,
            planner_reason=target_decision.reason,
            target_value=target_value,
            merge_status="merged",
            reason=write_document.reason or target_decision.reason,
        )

    def _deterministic_profile_target_write_result(
        self,
        *,
        profile_identifier: str,
        current_document: dict[str, Any],
        group: _EvidenceProjectionGroup,
        target_path: str,
        target_class: str | None,
        planner_status: str | None,
        planner_reason: str | None,
        current_target_value: Any,
        reason_prefix: str,
    ) -> ProfileObjectPatchResult | None:
        fallback_value = self._fallback_profile_target_value(
            target_path=target_path,
            current_value=current_target_value,
            notes=group.notes,
        )
        if fallback_value is None:
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_class,
                planner_status=planner_status,
                planner_reason=planner_reason,
                reason=f"{reason_prefix}; no profile-worthy schema-safe value could be derived.",
        )
        try:
            candidate = self._apply_profile_target_write(
                current_document,
                target_path,
                fallback_value,
            )
        except Exception as exc:
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                target_path=target_path,
                target_class=target_class,
                planner_status=planner_status,
                planner_reason=planner_reason,
                target_value=fallback_value,
                error=str(exc),
                reason=reason_prefix,
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=candidate,
        )
        if not validation.valid:
            return None
        return ProfileObjectPatchResult(
            object_identifier=group.group_id,
            object_kind=group.object_kind,
            status="applied",
            target_path=target_path,
            target_class=target_class,
            planner_status=planner_status,
            planner_reason=planner_reason,
            target_value=fallback_value,
            merge_status="merged",
            reason=reason_prefix,
        )

    @classmethod
    def _apply_profile_target_write(
        cls,
        document: dict[str, Any],
        target_path: str,
        target_value: Any,
    ) -> dict[str, Any]:
        if target_path.endswith("/-"):
            return cls._append_profile_sub_object(
                document,
                target_path.removesuffix("/-"),
                target_value,
            )
        return cls._replace_json_pointer(document, target_path, target_value)

    @classmethod
    def _append_profile_sub_object(
        cls,
        document: dict[str, Any],
        array_path: str,
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Schema branch append requires an object value.")
        updated = cls._clone_json_object(document)
        array_value = cls._value_at_json_pointer(updated, array_path)
        if array_value is None:
            cls._set_json_pointer(updated, array_path, [])
            array_value = cls._value_at_json_pointer(updated, array_path)
        if not isinstance(array_value, list):
            raise ValueError(f"Schema branch target is not an array: {array_path}")
        key = cls._profile_sub_object_key(value)
        for index, existing in enumerate(array_value):
            if isinstance(existing, dict) and cls._profile_sub_object_key(existing) == key:
                array_value[index] = cls._merge_profile_sub_object(existing, value)
                return updated
        array_value.append(cls._clone_json_object(value))
        return updated

    @classmethod
    def _set_json_pointer(cls, document: dict[str, Any], path: str, value: Any) -> None:
        tokens = cls._json_pointer_tokens(path)
        if not tokens:
            raise ValueError("Cannot assign root document through pointer helper.")
        parent: Any = document
        for token in tokens[:-1]:
            if isinstance(parent, list) and token.isdigit():
                parent = parent[int(token)]
            elif isinstance(parent, dict):
                parent = parent.setdefault(token, {})
            else:
                raise ValueError(f"Cannot create JSON Pointer path: {path}")
        last = tokens[-1]
        if isinstance(parent, dict):
            parent[last] = cls._clone_json_object(value)
        elif isinstance(parent, list) and last.isdigit():
            index = int(last)
            while len(parent) <= index:
                parent.append({})
            parent[index] = cls._clone_json_object(value)
        else:
            raise ValueError(f"Cannot assign JSON Pointer path: {path}")

    @staticmethod
    def _profile_sub_object_key(value: dict[str, Any]) -> str:
        for key in ("id", "title", "name"):
            item = value.get(key)
            if isinstance(item, list) and item:
                return f"{key}:{str(item[0]).strip().lower()}"
            if isinstance(item, str) and item.strip():
                return f"{key}:{item.strip().lower()}"
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _merge_profile_sub_object(
        cls,
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        merged = cls._clone_json_object(existing)
        for key, value in incoming.items():
            if cls._is_missing_value(value):
                continue
            if key not in merged or cls._is_missing_value(merged[key]):
                merged[key] = cls._clone_json_object(value)
            elif isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = cls._merge_unique_dicts(merged[key], value)
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._merge_profile_sub_object(merged[key], value)
        return merged

    @classmethod
    def _coerce_profile_target_value(
        cls,
        *,
        target_path: str,
        current_value: Any,
        proposed_value: Any,
    ) -> Any:
        if target_path in {"/title", "/description", "/identifier", "/keyword"}:
            return cls._merge_unique_strings(
                current_value if isinstance(current_value, list) else [],
                cls._string_list(proposed_value),
            )
        if target_path == "/modification_date":
            if isinstance(proposed_value, str):
                return proposed_value.strip()
            return current_value
        return cls._coerce_to_reference_shape(current_value, proposed_value)

    @classmethod
    def _coerce_to_reference_shape(cls, reference: Any, proposed: Any) -> Any:
        if isinstance(reference, list):
            proposed_items = proposed if isinstance(proposed, list) else [proposed]
            return [
                cls._coerce_to_reference_shape(reference[0], item)
                if reference
                else cls._clone_json_object(item)
                for item in proposed_items
                if not cls._is_missing_value(item)
            ]
        if isinstance(reference, dict):
            proposed_dict = proposed if isinstance(proposed, dict) else {}
            merged = cls._clone_json_object(reference)
            for key, current_item in reference.items():
                if key in proposed_dict:
                    merged[key] = cls._coerce_to_reference_shape(
                        current_item,
                        proposed_dict[key],
                    )
            return merged
        if isinstance(reference, str):
            if isinstance(proposed, list):
                return "; ".join(str(item).strip() for item in proposed if str(item).strip())
            if proposed is None:
                return reference
            return str(proposed).strip()
        if reference is None:
            return cls._clone_json_object(proposed) if isinstance(proposed, dict) else None
        return cls._clone_json_object(proposed)

    @classmethod
    def _fallback_profile_target_value(
        cls,
        *,
        target_path: str,
        current_value: Any,
        notes: list[EvidenceNote],
    ) -> Any | None:
        if not cls._evidence_group_has_profile_signal(notes):
            return None

        if target_path == "/keyword":
            keywords = cls._profile_keywords_for_notes(notes)
            if not keywords:
                return None
            return cls._merge_unique_strings(
                current_value if isinstance(current_value, list) else [],
                keywords,
            )
        if target_path == "/description":
            if not cls._description_target_worthy(notes):
                return None
            descriptions = cls._profile_observation_sentences(notes, max_count=3)
            if not descriptions:
                return None
            return cls._merge_unique_strings(
                current_value if isinstance(current_value, list) else [],
                descriptions,
            )
        if target_path == "/title":
            title = cls._fallback_target_title(notes)
            if not title:
                return None
            return cls._merge_unique_strings(
                current_value if isinstance(current_value, list) else [],
                [title],
            )
        if target_path == "/identifier":
            identifiers = cls._identifier_values_for_notes(notes)
            if not identifiers:
                return None
            return cls._merge_unique_strings(
                current_value if isinstance(current_value, list) else [],
                identifiers,
            )
        if target_path == "/modification_date":
            return cls._date_value_for_notes(notes)
        if target_path == "/creator/0":
            return cls._fallback_creator_value(current_value, notes)
        if target_path == "/dataset_distribution/0":
            return cls._fallback_distribution_value(current_value, notes)
        if target_path == "/was_generated_by/0":
            return cls._fallback_activity_value(
                current_value,
                notes,
                default_title="NMR data generation activity",
            )
        if target_path == "/was_generated_by/0/carried_out_by/-":
            return cls._fallback_agentic_entity_value(notes)
        if target_path == "/is_about_activity/0":
            return cls._fallback_activity_value(
                current_value,
                notes,
                default_title="NMR acquisition activity",
            )
        if target_path == "/is_about_entity/0":
            return cls._fallback_entity_value(current_value, notes)
        if target_path == "/type/0":
            return cls._fallback_concept_value(current_value, notes)
        return None

    @classmethod
    def _fallback_agentic_entity_value(cls, notes: list[EvidenceNote]) -> dict[str, Any] | None:
        device_notes = [note for note in notes if cls._note_has_device_signal(note)]
        if not device_notes:
            return None
        title = cls._device_title_for_notes(device_notes)
        if not title:
            return None
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "device"
        return {
            "id": f"device:{slug}",
            "title": title,
            "description": "Instrument associated with the data-generating activity.",
            "rdf_type": {
                "id": "http://purl.obolibrary.org/obo/OBI_0000968",
                "title": "device",
            },
            "type": {
                "id": "http://purl.obolibrary.org/obo/OBI_0000968",
                "title": "device",
            },
            "has_qualitative_attribute": [],
            "has_quantitative_attribute": [],
            "has_part": [],
            "part_of": [],
            "other_identifier": [],
        }

    @staticmethod
    def _device_title_for_notes(notes: list[EvidenceNote]) -> str | None:
        for note in notes:
            text = f"{note.evidence_text}\n{note.observation}"
            match = re.search(
                r"(?:instrument|spectrometer|probehead)\s*(?:used\s*)?(?:is|:|=)\s*<?([^>\r\n;]+)>?",
                text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
            match = re.search(r"\b(Bruker\s+Avance[^\r\n;]*)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def _profile_target_unsuitable_reason(
        cls,
        *,
        target_path: str,
        notes: list[EvidenceNote],
    ) -> str | None:
        if not notes:
            return "No evidence notes were available for profile projection."
        if all(cls._is_low_level_parameter_note(note) for note in notes):
            return (
                "Evidence contains only low-level instrument configuration; "
                "leaving it out of the dataset-level DCAT-AP-plus object."
            )
        if target_path in {"/description", "/keyword"}:
            useful_keywords = cls._profile_keywords_for_notes(notes)
            if not useful_keywords and all(
                note.category in {"method_signal", "measurement_signal", "resource_signal"}
                for note in notes
            ):
                return (
                    "Dataset description and keywords are reserved for curation-level facts, "
                    "not raw acquisition or processing parameters."
                )
        return None

    @classmethod
    def _fallback_creator_value(
        cls,
        current_value: Any,
        notes: list[EvidenceNote],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        names: list[str] = []
        for note in notes:
            key, value = cls._assignment_from_note(note)
            text = cls._note_search_text(note)
            if key and key.lower() in {"origin", "owner", "author", "creator"} and value:
                names.append(value)
            elif "bruker" in text:
                names.append("Bruker BioSpin GmbH")
        if not names:
            return None
        value = cls._clone_json_object(current_value)
        value["name"] = cls._merge_unique_strings(value.get("name", []), names)
        return value

    @classmethod
    def _fallback_distribution_value(
        cls,
        current_value: Any,
        notes: list[EvidenceNote],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        keywords = cls._profile_keywords_for_notes(notes)
        file_like = any(
            term in cls._note_search_text(note)
            for note in notes
            for term in ("jcamp", ".jdx", ".dx", "topspin", "zip", "file", "format")
        )
        if not file_like and not keywords:
            return None
        value = cls._clone_json_object(current_value)
        title = "Primary NMR data distribution" if "NMR spectroscopy" in keywords or "1H NMR" in keywords else "Primary dataset distribution"
        value["title"] = cls._merge_unique_strings(value.get("title", []), [title])
        description_parts = cls._profile_observation_sentences(notes, max_count=2)
        if not description_parts:
            description_parts = ["Dataset files contain NMR data and associated metadata."]
        value["description"] = cls._merge_unique_strings(value.get("description", []), description_parts)
        if not isinstance(value.get("access_URL"), list) or not value.get("access_URL"):
            value["access_URL"] = current_value.get("access_URL", [])
        if not isinstance(value.get("format"), dict):
            value["format"] = None
        if not isinstance(value.get("media_type"), dict):
            value["media_type"] = None
        return value

    @classmethod
    def _fallback_activity_value(
        cls,
        current_value: Any,
        notes: list[EvidenceNote],
        *,
        default_title: str,
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        if not any(note.category in {"method_signal", "measurement_signal"} for note in notes):
            return None
        value = cls._clone_json_object(current_value)
        value["title"] = cls._merge_unique_strings(value.get("title", []), [default_title])
        descriptions = cls._profile_observation_sentences(notes, max_count=3)
        if descriptions:
            value["description"] = cls._merge_unique_strings(value.get("description", []), descriptions)
        qualitative = cls._qualitative_attributes_for_notes(notes)
        if qualitative:
            value["has_qualitative_attribute"] = cls._merge_unique_dicts(
                value.get("has_qualitative_attribute", []),
                qualitative,
            )
        return value

    @classmethod
    def _fallback_entity_value(
        cls,
        current_value: Any,
        notes: list[EvidenceNote],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        if not any(note.category in {"entity_signal", "measurement_signal"} for note in notes):
            return None
        value = cls._clone_json_object(current_value)
        title = cls._entity_title_for_notes(notes)
        if title:
            value["title"] = title
        descriptions = cls._profile_observation_sentences(notes, max_count=2)
        if descriptions:
            value["description"] = "; ".join(descriptions)
        qualitative = cls._qualitative_attributes_for_notes(notes)
        if qualitative:
            value["has_qualitative_attribute"] = cls._merge_unique_dicts(
                value.get("has_qualitative_attribute", []),
                qualitative,
            )
        return value

    @classmethod
    def _fallback_concept_value(
        cls,
        current_value: Any,
        notes: list[EvidenceNote],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        labels = cls._profile_keywords_for_notes(notes)
        if "1H NMR" in labels:
            labels.insert(0, "1H NMR dataset")
        elif "NMR spectroscopy" in labels:
            labels.insert(0, "NMR spectroscopy dataset")
        labels = cls._dedupe_strings(labels)
        if not labels:
            return None
        value = cls._clone_json_object(current_value)
        value["preferred_label"] = cls._merge_unique_strings(value.get("preferred_label", []), labels[:3])
        return value

    @classmethod
    def _description_target_worthy(cls, notes: list[EvidenceNote]) -> bool:
        if not notes:
            return False
        if all(cls._is_low_level_parameter_note(note) for note in notes):
            return False
        category_set = {note.category for note in notes}
        if category_set <= {"method_signal", "measurement_signal", "resource_signal"}:
            return False
        return any(
            term in cls._note_search_text(note)
            for note in notes
            for term in ("dataset", "sample", "contains", "study", "experiment")
        )

    @classmethod
    def _evidence_group_has_profile_signal(cls, notes: list[EvidenceNote]) -> bool:
        if not notes:
            return False
        if not any(note.signal_level == "high" for note in notes):
            return False
        return any(not cls._is_low_level_parameter_note(note) for note in notes)

    @classmethod
    def _is_low_level_parameter_note(cls, note: EvidenceNote) -> bool:
        text = cls._note_search_text(note)
        note_id = (note.note_id or "").lower()
        low_level_keys = {
            "acb",
            "aq",
            "aq_mod",
            "aqmod",
            "autopos",
            "bacdel",
            "bacsair",
            "bacscap",
            "bfreq",
            "bf1",
            "bgaeth1",
            "birds",
            "bla01eth",
            "bla01nam",
            "bla01pn",
            "bla01sn",
            "bsms",
            "bsmseth",
            "cnst",
            "comdig2",
            "decim",
            "digi140",
            "dspfirm",
            "dspfvs",
            "ds",
            "dw",
            "fw",
            "ns",
            "nus",
            "o1",
            "phcor",
            "pl",
            "plstrt",
            "pqphase",
            "rg",
            "ro",
            "sfo1",
            "sw",
            "td",
            "te",
            "vtueth",
            "xfac",
        }
        key, _value = cls._assignment_from_note(note)
        high_level_keys = {"solvent", "nuc1", "pulprog", "origin", "owner", "title"}
        if key and key.lower().strip("$") in high_level_keys:
            return False
        normalized_key = key.lower().strip("$") if key else ""
        low_level_prefixes = (
            "bacs",
            "bga",
            "bla",
            "bsms",
            "cfa",
            "cnf",
            "com",
            "cpdb",
            "crc",
            "csw",
            "ctb",
            "digi",
            "dru",
            "rxf",
            "samchg",
            "sgu",
            "shim",
            "tfx",
            "triplo",
            "vtu",
            "wb",
            "user",
        )
        key_is_low = bool(
            normalized_key
            and (
                normalized_key in low_level_keys
                or normalized_key.startswith(low_level_prefixes)
            )
        )
        if key_is_low or note_id.endswith("_setting") or note_id in low_level_keys:
            return True
        low_level_observation_terms = (
            "parameter is set",
            "parameter set",
            "configuration settings",
            "network configuration",
            "ethernet",
            "tcp/ip",
            "switchbox routing",
            "shim settings",
            "digital signal processing configuration",
        )
        if any(term in text for term in low_level_observation_terms):
            return True
        high_level_terms = {
            "solvent",
            "nuc1",
            "nucleus",
            "pulprog",
            "pulse program",
            "origin",
            "owner",
            "author",
            "creator",
            "title",
            "jcamp",
            "topspin",
            "1h",
            "nmr",
        }
        if any(term in text for term in high_level_terms):
            return False
        return False

    @classmethod
    def _curate_generated_profile_document(cls, document: dict[str, Any]) -> dict[str, Any]:
        curated = cls._clone_json_object(document)
        if "title" in curated:
            curated["title"] = cls._curate_profile_target_value(
                target_path="/title",
                value=curated.get("title"),
            )
        if "description" in curated:
            curated["description"] = cls._curate_profile_target_value(
                target_path="/description",
                value=curated.get("description"),
            )
        if "keyword" in curated:
            curated["keyword"] = cls._curate_profile_target_value(
                target_path="/keyword",
                value=curated.get("keyword"),
            )
        distributions = curated.get("dataset_distribution")
        if isinstance(distributions, list):
            for distribution in distributions:
                if not isinstance(distribution, dict):
                    continue
                if "title" in distribution:
                    distribution["title"] = cls._curate_profile_target_value(
                        target_path="/dataset_distribution/0/title",
                        value=distribution.get("title"),
                    )
                if "description" in distribution:
                    distribution["description"] = cls._curate_profile_target_value(
                        target_path="/dataset_distribution/0/description",
                        value=distribution.get("description"),
                    )
        return cls._curate_nested_profile_values(curated)

    @classmethod
    def _curate_nested_profile_values(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                cls._curate_nested_profile_values(item)
                for item in value
                if not cls._is_low_level_profile_attribute(item)
            ]
        if isinstance(value, dict):
            return {
                key: cls._curate_nested_profile_values(item)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _is_low_level_profile_attribute(cls, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if not ({"value", "description"} & set(value)):
            return False
        text = " ".join(
            str(value.get(key) or "")
            for key in ("value", "description")
        )
        lower = text.lower()
        return cls._is_low_level_profile_text(text) or any(
            term in lower
            for term in (
                "solvent off setting",
                "instrument parameter structure",
                "parameter structure",
                "name\tinstrum",
            )
        )

    @classmethod
    def _curate_profile_target_value(
        cls,
        *,
        target_path: str,
        value: Any,
    ) -> Any:
        if target_path == "/title":
            return cls._dedupe_strings(
                [
                    cleaned
                    for item in cls._string_list(value)
                    if (cleaned := cls._clean_profile_title_text(item)) is not None
                ]
            )
        if target_path == "/keyword":
            return [
                item
                for item in cls._string_list(value)
                if cls._is_profile_keyword_text(item)
            ]
        if target_path == "/description":
            return [
                item
                for item in cls._string_list(value)
                if cls._is_profile_description_text(item)
            ]
        if target_path.endswith("/title") or target_path.endswith("/description"):
            if isinstance(value, list):
                filtered = [
                    item
                    for item in cls._string_list(value)
                    if not cls._is_low_level_profile_text(item)
                ]
                return filtered
            if isinstance(value, str):
                return "" if cls._is_low_level_profile_text(value) else value
        return value

    @classmethod
    def _is_profile_keyword_text(cls, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized or cls._is_low_level_profile_text(normalized):
            return False
        allowed_exact = {
            "1h nmr",
            "nmr spectroscopy",
            "jcamp-dx",
            "bruker",
            "bruker topspin",
            "bruker avance 500 mhz",
            "cdcl3",
            "nmr pulse program",
            "data analysis",
        }
        lower = normalized.lower()
        if lower in allowed_exact:
            return True
        return any(
            term in lower
            for term in (
                "nmr",
                "jcamp",
                "bruker avance",
                "spectrum",
                "spectroscopy",
                "dataset",
            )
        )

    @classmethod
    def _is_profile_description_text(cls, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized or cls._is_low_level_profile_text(normalized):
            return False
        lower = normalized.lower()
        return any(
            term in lower
            for term in (
                "dataset",
                "1h nmr",
                "nmr",
                "spectrum",
                "spectroscopy",
                "instrument",
                "sample",
                "jcamp",
                "file",
            )
        )

    @staticmethod
    def _is_low_level_profile_text(text: str) -> bool:
        lower = text.lower()
        if "##$" in text or re.search(r"\b[A-Z][A-Z0-9_]{1,16}\s+parameter\b", text):
            return True
        if re.search(r"\b[A-Z][A-Z0-9_]{1,16}\s+(?:set to|is set to)\b", text):
            return True
        if re.search(r"\b[A-Z][A-Z0-9_]{1,16}\s+is\s+(?:no|yes|[0-9])\b", text):
            return True
        noisy_terms = (
            "tcp/ip",
            "ethernet",
            "switchbox",
            "shim_setting",
            "shim settings",
            "routing",
            "blanking",
            "configuration settings",
            "digital signal processing configuration",
            "powerlevels",
            "npoints",
            "parameter values",
            "parameter file from topspin",
        )
        return any(term in lower for term in noisy_terms)

    @classmethod
    def _clean_profile_title_text(cls, text: str) -> str | None:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized or cls._is_low_level_profile_text(normalized):
            return None
        match = re.search(r"\bdataset name\s*(?:is|:)\s*(.+)$", normalized, re.IGNORECASE)
        if match:
            normalized = match.group(1).strip()
        if normalized.lower() in {"title", "spectrum title"}:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
            return None
        if len(normalized) > 120:
            return None
        return normalized

    @classmethod
    def _title_value_from_note(cls, note: EvidenceNote) -> str | None:
        key, value = cls._assignment_from_note(note)
        if key and key.lower().strip("$") == "title":
            return cls._clean_profile_title_text(value or "")
        text = f"{note.observation or ''}\n{note.evidence_text or ''}"
        match = re.search(r"\bdataset name\s*(?:is|:)\s*([^\r\n.;]+)", text, re.IGNORECASE)
        if match:
            return cls._clean_profile_title_text(match.group(1))
        return None

    @staticmethod
    def _target_write_is_empty(current_value: Any, target_value: Any) -> bool:
        if isinstance(target_value, list):
            return not target_value or target_value == current_value
        return target_value in (None, "", {}, []) or target_value == current_value

    @classmethod
    def _profile_keywords_for_notes(cls, notes: list[EvidenceNote]) -> list[str]:
        keywords: list[str] = []
        for note in notes:
            if cls._is_low_level_parameter_note(note):
                continue
            text = cls._note_search_text(note)
            if "1h" in text or "proton" in text:
                keywords.append("1H NMR")
            if "nmr" in text:
                keywords.append("NMR spectroscopy")
            if "jcamp" in text or ".jdx" in text or ".dx" in text:
                keywords.append("JCAMP-DX")
            if "topspin" in text:
                keywords.append("Bruker TopSpin")
            if "bruker" in text:
                keywords.append("Bruker")
            if "cdcl3" in text or "chloroform-d" in text:
                keywords.append("CDCl3")
            if "pulse" in text or "pulprog" in text:
                keywords.append("NMR pulse program")
        return cls._dedupe_strings(keywords)

    @classmethod
    def _profile_observation_sentences(
        cls,
        notes: list[EvidenceNote],
        *,
        max_count: int,
    ) -> list[str]:
        sentences: list[str] = []
        for note in notes:
            if cls._is_low_level_parameter_note(note):
                continue
            text = (note.observation or "").strip()
            if not text:
                continue
            text = re.sub(r"\s+", " ", text)
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            sentences.append(text)
            if len(sentences) >= max_count:
                break
        return cls._dedupe_strings(sentences)

    @classmethod
    def _qualitative_attributes_for_notes(cls, notes: list[EvidenceNote]) -> list[dict[str, str]]:
        attributes: list[dict[str, str]] = []
        allowed_keys = {"solvent", "nuc1", "pulprog", "origin", "owner", "probehead", "instrument"}
        for note in notes:
            key, value = cls._assignment_from_note(note)
            if not key or not value:
                continue
            normalized_key = key.lower().strip("$")
            if normalized_key not in allowed_keys:
                continue
            attributes.append({"title": normalized_key, "value": value})
        return cls._merge_unique_dicts([], attributes)

    @classmethod
    def _assignment_from_note(cls, note: EvidenceNote) -> tuple[str | None, str | None]:
        text = f"{note.evidence_text or ''}\n{note.observation or ''}"
        match = re.search(
            r"(?:##\$?|^|\s)([A-Za-z][A-Za-z0-9_]{1,32})\s*=\s*<?([^>\r\n;]{1,120})>?",
            text,
        )
        if not match:
            return None, None
        key = match.group(1).strip()
        value = match.group(2).strip().strip("<>").strip()
        if not value:
            return key, None
        return key, value

    @classmethod
    def _identifier_values_for_notes(cls, notes: list[EvidenceNote]) -> list[str]:
        values: list[str] = []
        for note in notes:
            key, value = cls._assignment_from_note(note)
            if key and key.lower() in {"id", "identifier", "sample_id"} and value:
                values.append(value)
        return cls._dedupe_strings(values)

    @classmethod
    def _date_value_for_notes(cls, notes: list[EvidenceNote]) -> str | None:
        for note in notes:
            text = f"{note.evidence_text or ''} {note.observation or ''}"
            match = re.search(r"\b(20\d{2}-\d{2}-\d{2})(?:[T ][0-2]\d:[0-5]\d(?::[0-5]\d)?)?\b", text)
            if match:
                return match.group(1)
        return None

    @classmethod
    def _fallback_target_title(cls, notes: list[EvidenceNote]) -> str | None:
        for note in notes:
            title = cls._title_value_from_note(note)
            if title:
                return title
        keywords = cls._profile_keywords_for_notes(notes)
        if "1H NMR" in keywords:
            return "1H NMR dataset"
        if "NMR spectroscopy" in keywords:
            return "NMR spectroscopy dataset"
        observations = cls._profile_observation_sentences(notes, max_count=1)
        return observations[0] if observations else None

    @classmethod
    def _note_has_curatable_profile_signal(cls, note: EvidenceNote) -> bool:
        if cls._is_low_level_parameter_note(note):
            return False
        if note.signal_level != "high":
            return False
        key, value = cls._assignment_from_note(note)
        normalized_key = key.lower().strip("$") if key else ""
        if normalized_key in {"origin", "owner", "author", "creator"}:
            return bool(value)
        if normalized_key in {"solvent", "nuc1", "pulprog"}:
            return bool(value)
        if normalized_key == "title":
            return cls._title_value_from_note(note) is not None
        text = cls._note_search_text(note)
        curatable_terms = (
            "dataset name",
            "dataset contains",
            "instrument",
            "bruker avance",
            "1h nmr",
            "nmr spectrum",
            "nmr spectroscopy",
            "jcamp",
            ".jdx",
            ".dx",
            "topspin",
            "pulse program",
            "solvent",
            "nucleus",
            "sample",
            "modification date",
            "timestamp",
        )
        if any(term in text for term in curatable_terms):
            if "spectrum title" in text and cls._title_value_from_note(note) is None:
                return False
            return True
        return note.category in {"agent_signal", "entity_signal"} and bool(note.observation.strip())

    @classmethod
    def _entity_title_for_notes(cls, notes: list[EvidenceNote]) -> str | None:
        text = " ".join(cls._note_search_text(note) for note in notes)
        if "1h" in text or "proton" in text:
            return "1H NMR measurement target"
        if "cdcl3" in text or "solvent" in text:
            return "NMR sample environment"
        observations = cls._profile_observation_sentences(notes, max_count=1)
        return observations[0] if observations else None

    @staticmethod
    def _note_search_text(note: EvidenceNote) -> str:
        return " ".join(
            part
            for part in (
                note.note_id,
                note.category,
                note.observation,
                note.evidence_text,
                note.file_path,
            )
            if part
        ).lower()

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, dict):
            return []
        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def _merge_unique_strings(cls, current: Any, additions: list[str]) -> list[str]:
        values = cls._string_list(current) + cls._string_list(additions)
        return cls._dedupe_strings(values)

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value).strip())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _merge_unique_dicts(current: Any, additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = list(current) if isinstance(current, list) else []
        seen = {
            json.dumps(item, sort_keys=True)
            for item in merged
            if isinstance(item, dict)
        }
        for item in additions:
            key = json.dumps(item, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    async def _patch_profile_with_extraction_group(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        profile_target_class: str,
        current_document: dict[str, Any],
        group: _EvidenceProjectionGroup,
        target_decision: ProfileTargetDecision,
        validation_schema: dict[str, Any],
        file_inventory: list[Any],
        warnings: list[str],
    ) -> ProfileObjectPatchResult:
        assert self.ollama_client is not None
        target_path = target_decision.target_path or group.target_hint
        target_schema = self._schema_slice_for_json_path(
            validation_schema,
            target_path,
            max_depth=2,
        )
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
                    evidence_notes=group.notes,
                    file_inventory=file_inventory,
                    schema_slice=target_schema,
                    target_path=target_path,
                    target_class=target_decision.target_class,
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
            warnings.append(f"Profile patch failed for '{group.group_id}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
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
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="skipped",
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                reason=patch_document.reason or target_decision.reason,
            )
        invalid_paths = [
            operation.path
            for operation in operations
            if not self._patch_path_allowed_for_target(operation.path, target_path)
        ]
        if invalid_paths:
            error = f"Patch wrote outside selected target {target_path}: {', '.join(invalid_paths)}"
            warnings.append(f"Profile patch skipped for '{group.group_id}': {error}")
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                operations=operations,
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                error=error,
                reason=patch_document.reason,
            )
        try:
            candidate = self._apply_profile_patch(current_document, operations)
        except Exception as exc:
            warnings.append(f"Profile patch skipped for '{group.group_id}': {exc}")
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                operations=operations,
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
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
                f"Profile patch skipped for '{group.group_id}' because it broke schema validation: {error}"
            )
            return ProfileObjectPatchResult(
                object_identifier=group.group_id,
                object_kind=group.object_kind,
                status="failed",
                operations=operations,
                target_path=target_path,
                target_class=target_decision.target_class,
                planner_status=target_decision.status,
                planner_reason=target_decision.reason,
                error=error,
                reason=patch_document.reason,
            )
        return ProfileObjectPatchResult(
            object_identifier=group.group_id,
            object_kind=group.object_kind,
            status="applied",
            operations=operations,
            target_path=target_path,
            target_class=target_decision.target_class,
            planner_status=target_decision.status,
            planner_reason=target_decision.reason,
            reason=patch_document.reason or target_decision.reason,
        )

    @staticmethod
    def _patch_path_allowed_for_target(path: str, target_path: str) -> bool:
        if not path.startswith("/"):
            return False
        allowed_parent_paths = {"/id"}
        if path in allowed_parent_paths:
            return True
        target = target_path.rstrip("/") or "/"
        return path == target or path.startswith(f"{target}/")

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

    async def _save_profile_result(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        evidence_context: EvidenceContext,
        normalization: ExtractionNormalization,
        document: dict[str, Any],
        profile_manifest: Any,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        warnings: list[str],
    ) -> ExtractionRunResult:
        assert self.output_repository is not None
        pruned_document = self._prune_initial_draft_scaffold(
            document,
            state.initial_draft_scaffold,
        )
        curated_document = self._curate_generated_profile_document(pruned_document)
        clean_document = remove_null_values(curated_document)
        validation = self._validate_profile_document(
            profile_identifier=profile_identifier,
            document=clean_document,
        )
        state.generated_final_draft = clean_document
        state.validation = validation
        if state.curated_document is None:
            state.curated_document = self._clone_json_object(clean_document)
            state.curated_validation = validation
        elif state.curated_validation is None:
            state.curated_validation = self._validate_profile_document(
                profile_identifier=profile_identifier,
                document=state.curated_document,
            )
        state.field_completion_ledger = self._build_field_completion_ledger(
            generated_document=clean_document,
            curated_document=state.curated_document,
            validation=validation,
            normalization=normalization,
            validation_schema=validation_schema,
            enrichable_fields=getattr(profile_manifest, "enrichable_fields", []),
            projection_ledger=state.projection_ledger,
        )
        state.curation_ledger = self._build_curation_ledger(
            generated_document=clean_document,
            curated_document=state.curated_document or clean_document,
            existing_field_ledger=state.field_completion_ledger,
        )
        state.draft_quality_state = self._classify_draft_quality(
            validation=validation,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
        )
        token_usage = await self.get_token_usage(data_package_id)
        result = ExtractionRunResult(
            generated_final_draft=clean_document,
            machine_evidence_context=evidence_context,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            curated_document=state.curated_document,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
            chat_model=self.ollama_client.chat_model if self.ollama_client else None,
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
        self._save_run_state(data_package_id, state)
        return result

    def _validate_profile_document(
        self,
        *,
        profile_identifier: str,
        document: dict[str, Any],
        warnings: list[str] | None = None,
    ) -> DraftValidationResult:
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        errors = [
            error
            if isinstance(error, ProfileValidationIssue)
            else ProfileValidationIssue(
                path=str(getattr(error, "path", "")),
                message=str(getattr(error, "message", error)),
                schema_path=str(getattr(error, "schema_path", "")),
            )
            for error in validation.errors
        ]
        return DraftValidationResult(
            status="valid" if validation.valid else "invalid",
            errors=errors,
            warnings=warnings or [],
        )

    @staticmethod
    def _classify_draft_quality(
        *,
        validation: DraftValidationResult,
        projection_ledger: list[ProjectionLedgerRecord],
        field_completion_ledger: list[FieldCompletionLedgerRecord],
    ) -> str:
        if not projection_ledger or not any(
            record.status == "projected" for record in projection_ledger
        ):
            return "empty_profile_shell"
        has_projection_issue = any(
            record.status != "projected" for record in projection_ledger
        )
        has_field_issue = any(
            record.issue_categories or record.edit_needed_reason
            for record in field_completion_ledger
        )
        if validation.status == "valid" and not has_projection_issue and not has_field_issue:
            return "complete_final_draft"
        return "imperfect_final_draft"

    def _build_field_completion_ledger(
        self,
        *,
        generated_document: dict[str, Any],
        curated_document: dict[str, Any] | None,
        validation: DraftValidationResult,
        normalization: ExtractionNormalization,
        validation_schema: dict[str, Any],
        enrichable_fields: list[str],
        projection_ledger: list[ProjectionLedgerRecord],
    ) -> list[FieldCompletionLedgerRecord]:
        paths = set(self._required_profile_field_paths(validation_schema))
        profile_sources = self._profile_vocab_sources(
            generated_document,
            enrichable_fields=enrichable_fields,
        )
        paths.update(path for path, _field_name, _value in profile_sources)
        paths.update(
            self._profile_issue_path_to_pointer(issue.path)
            for issue in validation.errors
        )
        normalization_by_path = {
            item.json_path: item
            for item in normalization.profile_fields
        }
        source_evidence_by_path: dict[str, list[str]] = {}
        for record in projection_ledger:
            if not record.source_evidence:
                continue
            for path in record.projected_paths:
                source_evidence_by_path.setdefault(path, []).append(record.source_evidence)

        ledgers: list[FieldCompletionLedgerRecord] = []
        for path in sorted(path for path in paths if path):
            generated_exists, generated_value = self._json_pointer_value(
                generated_document,
                path,
            )
            curated_exists, curated_value = self._json_pointer_value(
                curated_document or {},
                path,
            )
            matching_errors = [
                issue
                for issue in validation.errors
                if self._profile_issue_path_to_pointer(issue.path) == path
            ]
            issue_categories: list[str] = []
            if not generated_exists or self._is_missing_value(generated_value):
                issue_categories.append("missing")
                validation_status = "missing"
            elif matching_errors:
                issue_categories.append("invalid")
                validation_status = "invalid"
            else:
                validation_status = "valid"

            normalized = normalization_by_path.get(path)
            enrichment_status = "not_grounded"
            if normalized is not None:
                if normalized.term is None:
                    enrichment_status = "no_candidate"
                elif normalized.term.selected_uri:
                    enrichment_status = "grounded"
                else:
                    enrichment_status = "not_grounded"
            elif self._field_name_from_pointer(path) not in (
                {"has_quantity_type", "unit"} | set(enrichable_fields)
            ):
                enrichment_status = "not_grounded"

            if normalized is not None and enrichment_status != "grounded":
                issue_categories.append("non_enriched")

            edit_needed_reason = "; ".join(
                issue.message for issue in matching_errors
            )
            if not edit_needed_reason and issue_categories:
                edit_needed_reason = ", ".join(issue_categories)

            ledgers.append(
                FieldCompletionLedgerRecord(
                    json_path=path,
                    field_name=self._field_name_from_pointer(path),
                    generated_value=generated_value if generated_exists else None,
                    curated_value=curated_value if curated_exists else None,
                    source_evidence=source_evidence_by_path.get(path, []),
                    validation_status=validation_status,
                    enrichment_status=enrichment_status,
                    issue_categories=issue_categories,
                    edit_needed_reason=edit_needed_reason,
                )
            )
        return ledgers

    def _build_curation_ledger(
        self,
        *,
        generated_document: dict[str, Any],
        curated_document: dict[str, Any],
        existing_field_ledger: list[FieldCompletionLedgerRecord],
    ) -> list[CurationLedgerRecord]:
        paths = (
            self._leaf_json_pointer_paths(generated_document)
            | self._leaf_json_pointer_paths(curated_document)
            | {record.json_path for record in existing_field_ledger}
        )
        evidence_by_path = {
            record.json_path: record.source_evidence
            for record in existing_field_ledger
        }
        ledger: list[CurationLedgerRecord] = []
        for path in sorted(path for path in paths if path):
            generated_exists, generated_value = self._json_pointer_value(
                generated_document,
                path,
            )
            curated_exists, curated_value = self._json_pointer_value(
                curated_document,
                path,
            )
            if not curated_exists:
                status = "user_removed"
            elif not generated_exists or generated_value != curated_value:
                status = "user_modified"
            else:
                status = "unchanged"
            ledger.append(
                CurationLedgerRecord(
                    json_path=path,
                    field_name=self._field_name_from_pointer(path),
                    generated_value=generated_value if generated_exists else None,
                    curated_value=curated_value if curated_exists else None,
                    source_evidence=evidence_by_path.get(path, []),
                    status=status,
                )
            )
        return ledger

    def _field_ledger_with_curated_values(
        self,
        ledger: list[FieldCompletionLedgerRecord],
        curated_document: dict[str, Any],
    ) -> list[FieldCompletionLedgerRecord]:
        updated: list[FieldCompletionLedgerRecord] = []
        for record in ledger:
            exists, value = self._json_pointer_value(curated_document, record.json_path)
            updated.append(
                record.model_copy(
                    update={"curated_value": value if exists else None}
                )
            )
        return updated

    def _persist_state_artifacts(
        self,
        data_package_id: str,
        state: ExtractionRunState,
    ) -> None:
        if self.output_repository is None:
            return
        chat_model = state.chat_model
        self._persist_initial_file_summaries(data_package_id, state)
        self._persist_initial_extraction_overview(data_package_id, state)
        if state.generated_final_draft is not None:
            self.output_repository.save_generated_final_draft(
                workflow_id=data_package_id,
                document=state.generated_final_draft,
                chat_model=chat_model,
            )
        if state.curated_document is not None:
            self.output_repository.save_curated_document(
                workflow_id=data_package_id,
                document=state.curated_document,
                chat_model=chat_model,
            )
        self.output_repository.save_projection_ledger(
            workflow_id=data_package_id,
            ledger=state.projection_ledger,
            chat_model=chat_model,
        )
        self.output_repository.save_field_completion_ledger(
            workflow_id=data_package_id,
            ledger=state.field_completion_ledger,
            chat_model=chat_model,
        )
        self.output_repository.save_curation_ledger(
            workflow_id=data_package_id,
            ledger=state.curation_ledger,
            chat_model=chat_model,
        )
        self.output_repository.save_validation(
            workflow_id=data_package_id,
            validation=state.validation,
            curated_validation=state.curated_validation,
            chat_model=chat_model,
        )

    def _clear_downstream_extraction_outputs(self, data_package_id: str) -> None:
        if self.output_repository is None:
            return
        state = self._load_run_state_or_none(data_package_id)
        self.output_repository.clear_extraction_downstream(data_package_id)
        if state is None:
            return
        cleared = state.model_copy(
            update={
                "chunk_results": [],
                "vocab_queries": [],
                "generated_final_draft": None,
                "curated_document": None,
                "draft_quality_state": None,
                "validation": DraftValidationResult(),
                "curated_validation": None,
                "initial_draft_scaffold": {},
                "projection_ledger": [],
                "field_completion_ledger": [],
                "curation_ledger": [],
            }
        )
        self._save_run_state(data_package_id, cleared)

    def _persist_initial_extraction_overview(
        self,
        data_package_id: str,
        state: ExtractionRunState,
    ) -> None:
        if self.output_repository is None:
            return
        if state.initial_extraction_overview_status is None:
            return
        self.output_repository.save_initial_extraction_overview(
            workflow_id=data_package_id,
            overview=state.initial_extraction_overview,
            status=state.initial_extraction_overview_status,
            chat_model=state.chat_model,
        )

    def _persist_initial_file_summaries(
        self,
        data_package_id: str,
        state: ExtractionRunState,
    ) -> None:
        if self.output_repository is None:
            return
        if state.initial_file_summary_status is None:
            return
        self.output_repository.save_initial_file_summaries(
            workflow_id=data_package_id,
            summaries=state.initial_file_summaries,
            status=state.initial_file_summary_status,
            chat_model=state.chat_model,
        )

    @classmethod
    def _required_profile_field_paths(
        cls,
        validation_schema: dict[str, Any],
    ) -> list[str]:
        target = cls._resolve_schema_node(validation_schema, validation_schema)
        if not isinstance(target, dict):
            return []
        required = target.get("required", [])
        if not isinstance(required, list):
            return []
        return [
            "/" + cls._json_pointer_escape(str(field))
            for field in required
            if isinstance(field, str)
        ]

    @staticmethod
    def _profile_issue_path_to_pointer(path: str) -> str:
        if path == "$":
            return ""
        pointer = ""
        remainder = path[2:] if path.startswith("$.") else path
        token = ""
        index_mode = False
        for char in remainder:
            if char == "." and not index_mode:
                if token:
                    pointer += "/" + ExtractionService._json_pointer_escape(token)
                    token = ""
                continue
            if char == "[":
                if token:
                    pointer += "/" + ExtractionService._json_pointer_escape(token)
                    token = ""
                index_mode = True
                continue
            if char == "]":
                if token:
                    pointer += "/" + token
                    token = ""
                index_mode = False
                continue
            token += char
        if token:
            pointer += "/" + ExtractionService._json_pointer_escape(token)
        return pointer

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        return value in (None, "", [], {})

    @classmethod
    def _field_name_from_pointer(cls, path: str) -> str:
        if not path:
            return "root"
        return cls._json_pointer_unescape(path.rsplit("/", 1)[-1])

    @classmethod
    def _json_pointer_value(
        cls,
        document: dict[str, Any],
        path: str,
    ) -> tuple[bool, Any]:
        if path in ("", "/"):
            return True, document
        current: Any = document
        for raw_part in path.strip("/").split("/"):
            part = cls._json_pointer_unescape(raw_part)
            if isinstance(current, dict):
                if part not in current:
                    return False, None
                current = current[part]
                continue
            if isinstance(current, list):
                try:
                    index = int(part)
                except ValueError:
                    return False, None
                if index < 0 or index >= len(current):
                    return False, None
                current = current[index]
                continue
            return False, None
        return True, current

    @classmethod
    def _set_json_pointer_value(
        cls,
        document: dict[str, Any],
        path: str,
        value: Any,
    ) -> dict[str, Any]:
        if path in ("", "/"):
            if not isinstance(value, dict):
                raise ValueError("Root curated document value must be a JSON object.")
            return value
        result = cls._clone_json_object(document)
        parts = [cls._json_pointer_unescape(part) for part in path.strip("/").split("/")]
        current: Any = result
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if isinstance(current, dict):
                if part not in current or current[part] is None:
                    current[part] = [] if next_part.isdigit() else {}
                current = current[part]
                continue
            if isinstance(current, list):
                item_index = int(part)
                while len(current) <= item_index:
                    current.append({} if not next_part.isdigit() else [])
                current = current[item_index]
                continue
            raise ValueError(f"Cannot set JSON Pointer path '{path}'.")

        final_part = parts[-1]
        if isinstance(current, dict):
            current[final_part] = value
        elif isinstance(current, list):
            item_index = int(final_part)
            while len(current) <= item_index:
                current.append(None)
            current[item_index] = value
        else:
            raise ValueError(f"Cannot set JSON Pointer path '{path}'.")
        return result

    @classmethod
    def _schema_for_json_pointer(
        cls,
        validation_schema: dict[str, Any],
        path: str,
    ) -> dict[str, Any]:
        current = cls._resolve_schema_node(validation_schema, validation_schema)
        for raw_part in path.strip("/").split("/") if path.strip("/") else []:
            part = cls._json_pointer_unescape(raw_part)
            current = cls._resolve_schema_node(current, validation_schema)
            if not isinstance(current, dict):
                return {}
            if part.isdigit():
                current = current.get("items", {})
                continue
            properties = current.get("properties", {})
            if not isinstance(properties, dict):
                return {}
            current = properties.get(part, {})
        current = cls._resolve_schema_node(current, validation_schema)
        return current if isinstance(current, dict) else {}

    @classmethod
    def _selected_vocab_value_for_schema(
        cls,
        *,
        field_schema: dict[str, Any],
        root_schema: dict[str, Any],
        selected_uri: str,
        selected_title: str | None,
        vocabulary_identifier: str | None,
        existing_value: Any,
    ) -> Any:
        field_schema = cls._resolve_schema_node(field_schema, root_schema)
        if cls._schema_is_array(field_schema, root_schema):
            item_schema = cls._resolve_schema_node(
                field_schema.get("items", {}),
                root_schema,
            )
            item_value = (
                cls._term_object_for_schema(
                    schema=item_schema,
                    root_schema=root_schema,
                    selected_uri=selected_uri,
                    selected_title=selected_title,
                    vocabulary_identifier=vocabulary_identifier,
                )
                if cls._schema_accepts_term_object(item_schema, root_schema)
                else selected_uri
            )
            current = list(existing_value) if isinstance(existing_value, list) else []
            if item_value not in current:
                current.append(item_value)
            return current
        if cls._schema_accepts_term_object(field_schema, root_schema):
            return cls._term_object_for_schema(
                schema=field_schema,
                root_schema=root_schema,
                selected_uri=selected_uri,
                selected_title=selected_title,
                vocabulary_identifier=vocabulary_identifier,
            )
        return selected_uri

    @classmethod
    def _schema_is_array(
        cls,
        schema: dict[str, Any],
        root_schema: dict[str, Any],
    ) -> bool:
        schema = cls._resolve_schema_node(schema, root_schema)
        schema_type = schema.get("type") if isinstance(schema, dict) else None
        return schema_type == "array" or (
            isinstance(schema_type, list) and "array" in schema_type
        )

    @classmethod
    def _schema_accepts_term_object(
        cls,
        schema: dict[str, Any],
        root_schema: dict[str, Any],
    ) -> bool:
        schema = cls._resolve_schema_node(schema, root_schema)
        if not isinstance(schema, dict):
            return False
        for union_key in ("anyOf", "oneOf"):
            options = schema.get(union_key)
            if isinstance(options, list):
                return any(
                    cls._schema_accepts_term_object(option, root_schema)
                    for option in options
                    if isinstance(option, dict)
                )
        properties = schema.get("properties", {})
        return isinstance(properties, dict) and "id" in properties

    @classmethod
    def _term_object_for_schema(
        cls,
        *,
        schema: dict[str, Any],
        root_schema: dict[str, Any],
        selected_uri: str,
        selected_title: str | None,
        vocabulary_identifier: str | None,
    ) -> dict[str, Any]:
        schema = cls._resolve_schema_node(schema, root_schema)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not isinstance(properties, dict):
            return {"id": selected_uri}
        value: dict[str, Any] = {"id": selected_uri}
        if "title" in properties and selected_title is not None:
            value["title"] = selected_title
        if "from_CV" in properties and vocabulary_identifier is not None:
            value["from_CV"] = vocabulary_identifier
        return value

    @staticmethod
    def _mark_field_curation_status(
        *,
        ledger: list[FieldCompletionLedgerRecord],
        json_path: str,
        status: str,
    ) -> list[FieldCompletionLedgerRecord]:
        updated: list[FieldCompletionLedgerRecord] = []
        found = False
        for record in ledger:
            if record.json_path != json_path:
                updated.append(record)
                continue
            found = True
            issue_categories = [
                category
                for category in record.issue_categories
                if category != "non_enriched"
            ]
            updated.append(
                record.model_copy(
                    update={
                        "enrichment_status": status,
                        "issue_categories": issue_categories,
                        "edit_needed_reason": "",
                    }
                )
            )
        if not found:
            updated.append(
                FieldCompletionLedgerRecord(
                    json_path=json_path,
                    field_name=ExtractionService._field_name_from_pointer(json_path),
                    enrichment_status=status,
                )
            )
        return updated

    @staticmethod
    def _mark_curation_ledger_status(
        *,
        ledger: list[CurationLedgerRecord],
        json_path: str,
        status: str,
    ) -> list[CurationLedgerRecord]:
        updated: list[CurationLedgerRecord] = []
        found = False
        for record in ledger:
            if record.json_path != json_path:
                updated.append(record)
                continue
            found = True
            updated.append(record.model_copy(update={"status": status}))
        if not found:
            updated.append(
                CurationLedgerRecord(
                    json_path=json_path,
                    field_name=ExtractionService._field_name_from_pointer(json_path),
                    status=status,
                )
            )
        return updated

    @classmethod
    def _leaf_json_pointer_paths(
        cls,
        value: Any,
        path: str = "",
    ) -> set[str]:
        if isinstance(value, dict):
            if not value and path:
                return {path}
            paths: set[str] = set()
            for key, item in value.items():
                paths.update(
                    cls._leaf_json_pointer_paths(
                        item,
                        f"{path}/{cls._json_pointer_escape(str(key))}",
                    )
                )
            return paths
        if isinstance(value, list):
            if not value and path:
                return {path}
            paths = set()
            for index, item in enumerate(value):
                paths.update(cls._leaf_json_pointer_paths(item, f"{path}/{index}"))
            return paths
        return {path} if path else set()

    @staticmethod
    def _clone_json_object(document: Any) -> Any:
        return json.loads(json.dumps(document))

    @classmethod
    def _fallback_profile_document(
        cls,
        *,
        data_package_id: str,
        evidence_context: EvidenceContext,
        validation_schema: dict[str, Any],
    ) -> dict[str, Any]:
        document, _scaffold = cls._initial_profile_document(
            data_package_id=data_package_id,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
        )
        return document

    @classmethod
    def _initial_profile_document(
        cls,
        *,
        data_package_id: str,
        evidence_context: EvidenceContext,
        validation_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        target_schema = cls._resolve_schema_node(validation_schema, validation_schema)
        document = cls._schema_shell_value(
            target_schema,
            validation_schema,
            depth=0,
            required_only=True,
        )
        if not isinstance(document, dict):
            document = {}
        properties = cls._target_schema_properties(validation_schema)
        title = cls._fallback_title(data_package_id, evidence_context)
        scaffold_entries: list[dict[str, Any]] = []
        if "title" in properties or "title" in document:
            document["title"] = cls._fallback_property_value(properties.get("title"), title)
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
                {
                    "id": f"{data_package_id}:activity:metadata-extraction",
                    "title": [],
                    "description": [],
                    "has_qualitative_attribute": [],
                    "has_quantitative_attribute": [],
                    "evaluated_activity": [],
                    "evaluated_entity": [],
                    "carried_out_by": [],
                }
            ]
            scaffold_entries.append(
                cls._scaffold_entry(
                    path="/was_generated_by/0",
                    value={"id": document["was_generated_by"][0]["id"]},
                    label="metadata extraction activity",
                    target_class="DataGeneratingActivity",
                    kind="required",
                    prune_if_unchanged=False,
                )
            )
            for child in (
                "title",
                "description",
                "has_qualitative_attribute",
                "has_quantitative_attribute",
                "evaluated_activity",
                "evaluated_entity",
                "carried_out_by",
            ):
                scaffold_entries.append(
                    cls._scaffold_entry(
                        path=f"/was_generated_by/0/{child}",
                        value=document["was_generated_by"][0][child],
                        label=f"metadata extraction activity {child.replace('_', ' ')}",
                        target_class=None,
                        kind="required-child",
                        prune_if_unchanged=True,
                    )
                )

        evidence_categories = {note.category for note in evidence_context.notes}
        core_slots = {
            "creator",
            "dataset_distribution",
            "keyword",
            "type",
            "modification_date",
            "is_about_entity",
            "is_about_activity",
        }
        evidence_slots: set[str] = set()
        if "agent_signal" in evidence_categories:
            evidence_slots.add("creator")
        if evidence_categories & {"entity_signal", "measurement_signal"}:
            evidence_slots.add("is_about_entity")
        if "method_signal" in evidence_categories:
            evidence_slots.add("is_about_activity")
        if "resource_signal" in evidence_categories:
            evidence_slots.add("dataset_distribution")

        for slot in sorted(core_slots | evidence_slots):
            if slot not in properties or slot in document:
                continue
            placeholder = cls._initial_placeholder_for_slot(
                data_package_id=data_package_id,
                slot=slot,
            )
            if placeholder is None:
                continue
            document[slot] = cls._fallback_property_value(properties.get(slot), placeholder)
            path = f"/{slot}"
            target_class = cls._placeholder_target_class(slot)
            value_for_compare = document[slot]
            if isinstance(document[slot], list) and document[slot]:
                path = f"/{slot}/0"
                value_for_compare = document[slot][0]
            scaffold_entries.append(
                cls._scaffold_entry(
                    path=path,
                    value=value_for_compare,
                    label=slot.replace("_", " "),
                    target_class=target_class,
                    kind="core" if slot in core_slots else "evidence",
                    prune_if_unchanged=True,
                )
            )

        scaffold = {
            "version": 1,
            "evidence_categories": sorted(evidence_categories),
            "entries": scaffold_entries,
        }
        return document, scaffold

    @staticmethod
    def _initial_placeholder_for_slot(
        *,
        data_package_id: str,
        slot: str,
    ) -> Any:
        placeholders: dict[str, Any] = {
            "creator": {"name": []},
            "dataset_distribution": {
                "access_URL": [
                    {"id": f"{data_package_id}:distribution:primary:access"}
                ],
                "title": [],
                "description": [],
                "format": None,
                "media_type": None,
            },
            "keyword": [],
            "type": [{"preferred_label": []}],
            "modification_date": "",
            "is_about_entity": [
                {
                    "id": f"{data_package_id}:entity:primary",
                    "title": "",
                    "description": "",
                    "has_qualitative_attribute": [],
                    "has_quantitative_attribute": [],
                    "was_generated_by": [],
                }
            ],
            "is_about_activity": [
                {
                    "id": f"{data_package_id}:activity:primary",
                    "title": [],
                    "description": [],
                    "has_qualitative_attribute": [],
                    "has_quantitative_attribute": [],
                }
            ],
        }
        return placeholders.get(slot)

    @staticmethod
    def _placeholder_target_class(slot: str) -> str | None:
        return {
            "creator": "Agent",
            "dataset_distribution": "Distribution",
            "type": "Concept",
            "is_about_entity": "EvaluatedEntity",
            "is_about_activity": "EvaluatedActivity",
        }.get(slot)

    @staticmethod
    def _scaffold_entry(
        *,
        path: str,
        value: Any,
        label: str,
        target_class: str | None,
        kind: str,
        prune_if_unchanged: bool,
    ) -> dict[str, Any]:
        return {
            "path": path,
            "value": json.loads(json.dumps(value)),
            "label": label,
            "target_class": target_class,
            "kind": kind,
            "prune_if_unchanged": prune_if_unchanged,
        }

    @classmethod
    def _schema_shell_value(
        cls,
        schema: Any,
        root: dict[str, Any],
        *,
        depth: int,
        required_only: bool,
    ) -> Any:
        schema = cls._resolve_schema_node(schema, root)
        if not isinstance(schema, dict):
            return None
        if "const" in schema:
            return schema["const"]
        if isinstance(schema.get("enum"), list) and schema["enum"]:
            return schema["enum"][0]
        for union_key in ("anyOf", "oneOf"):
            options = schema.get(union_key)
            if isinstance(options, list):
                non_null_options = [
                    option
                    for option in options
                    if not (isinstance(option, dict) and option.get("type") == "null")
                ]
                if non_null_options:
                    return cls._schema_shell_value(
                        non_null_options[0],
                        root,
                        depth=depth,
                        required_only=required_only,
                    )

        schema_type = schema.get("type")
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        if "object" in types or "properties" in schema:
            if depth >= 3:
                return {}
            properties = schema.get("properties", {})
            required = {
                item
                for item in schema.get("required", [])
                if isinstance(item, str)
            }
            if not isinstance(properties, dict):
                return {}
            keys = required if required_only else set(properties)
            return {
                key: cls._schema_shell_value(
                    properties[key],
                    root,
                    depth=depth + 1,
                    required_only=required_only,
                )
                for key in sorted(keys)
                if key in properties
            }
        if "array" in types:
            return []
        if "string" in types:
            return ""
        if "integer" in types or "number" in types:
            return 0
        if "boolean" in types:
            return False
        return None

    @staticmethod
    def _fallback_property_value(schema: Any, value: str) -> Any:
        if isinstance(schema, dict):
            schema_type = schema.get("type")
            if schema_type == "array" or (
                isinstance(schema_type, list) and "array" in schema_type
            ):
                if isinstance(value, list):
                    return value
                return [value]
        return value

    @classmethod
    def _target_catalog_from_document(
        cls,
        *,
        document: dict[str, Any],
        validation_schema: dict[str, Any],
        scaffold: dict[str, Any],
    ) -> list[dict[str, Any]]:
        properties = cls._target_schema_properties(validation_schema)
        catalog_by_path: dict[str, dict[str, Any]] = {}
        scaffold_by_path = {
            entry.get("path"): entry
            for entry in scaffold.get("entries", [])
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }

        def add(path: str, label: str, target_class: str | None = None) -> None:
            field_name = path.strip("/").split("/")[0] if path.strip("/") else ""
            prop = properties.get(field_name, {})
            description = prop.get("description", "") if isinstance(prop, dict) else ""
            scaffold_entry = scaffold_by_path.get(path)
            current_value = cls._value_at_json_pointer(document, path)
            scaffold_status = "not_scaffolded"
            if scaffold_entry:
                scaffold_status = (
                    "unfilled"
                    if current_value == scaffold_entry.get("value")
                    else "filled"
                )
            catalog_by_path[path] = {
                "path": path,
                "label": label,
                "field_name": field_name,
                "target_class": target_class,
                "description": description,
                "current_value": cls._catalog_value_preview(current_value),
                "scaffold_status": scaffold_status,
                "category_affinities": cls._target_category_affinities(path, target_class),
                "description_last_resort": path == "/description",
            }

        for field in ("title", "description", "identifier", "keyword", "modification_date"):
            if field in properties and field in document:
                add(f"/{field}", field.replace("_", " "))
        for entry in scaffold.get("entries", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            add(
                entry["path"],
                str(entry.get("label") or entry["path"]),
                entry.get("target_class"),
            )
        for path, target_class, label in (
            ("/creator/0", "Agent", "creator"),
            ("/dataset_distribution/0", "Distribution", "dataset distribution"),
            ("/type/0", "Concept", "dataset type"),
            ("/is_about_entity/0", "EvaluatedEntity", "evaluated entity"),
            ("/is_about_activity/0", "EvaluatedActivity", "evaluated activity"),
            ("/was_generated_by/0", "DataGeneratingActivity", "generating activity"),
            ("/was_generated_by/0/carried_out_by/-", "AgenticEntity", "generating activity participant"),
        ):
            field = path.strip("/").split("/")[0]
            if field in document:
                add(path, label, target_class)
        return list(catalog_by_path.values())

    @staticmethod
    def _target_category_affinities(path: str, target_class: str | None) -> list[str]:
        if "carried_out_by" in path or target_class == "AgenticEntity":
            return ["agent_signal"]
        if path.startswith("/creator"):
            return ["agent_signal"]
        if path.startswith("/dataset_distribution"):
            return ["resource_signal"]
        if path.startswith("/was_generated_by"):
            return ["method_signal", "measurement_signal"]
        if path.startswith("/is_about_activity"):
            return ["method_signal"]
        if path.startswith("/is_about_entity"):
            return ["entity_signal", "measurement_signal"]
        if path.startswith("/type"):
            return ["resource_signal", "data_quality_signal"]
        if path == "/modification_date":
            return ["resource_signal", "data_quality_signal"]
        if path == "/keyword":
            return ["resource_signal", "entity_signal", "method_signal"]
        if path == "/description":
            return ["data_quality_signal", "uncertainty", "other"]
        if target_class:
            return [target_class]
        return []

    @staticmethod
    def _catalog_value_preview(value: Any) -> Any:
        if isinstance(value, str):
            return value[:300]
        if isinstance(value, list):
            return value[:3]
        if isinstance(value, dict):
            return {
                key: value[key]
                for key in list(value.keys())[:10]
            }
        return value

    @classmethod
    def _schema_slice_for_json_path(
        cls,
        validation_schema: dict[str, Any],
        path: str,
        *,
        max_depth: int,
    ) -> dict[str, Any]:
        node: Any = cls._resolve_schema_node(validation_schema, validation_schema)
        for token in cls._json_pointer_tokens(path):
            node = cls._resolve_schema_node(node, validation_schema)
            if not isinstance(node, dict):
                break
            if token.isdigit() or token == "-":
                node = node.get("items", node)
                continue
            properties = node.get("properties", {})
            if isinstance(properties, dict) and token in properties:
                node = properties[token]
                continue
            break
        return cls._compact_schema_node(node, validation_schema, depth=max_depth)

    @staticmethod
    def _json_pointer_tokens(path: str) -> list[str]:
        if not path or path == "/":
            return []
        return [
            token.replace("~1", "/").replace("~0", "~")
            for token in path.lstrip("/").split("/")
        ]

    @classmethod
    def _value_at_json_pointer(cls, document: Any, path: str) -> Any:
        value = document
        for token in cls._json_pointer_tokens(path):
            if isinstance(value, list) and token.isdigit():
                index = int(token)
                if index >= len(value):
                    return None
                value = value[index]
            elif isinstance(value, dict):
                if token not in value:
                    return None
                value = value[token]
            else:
                return None
        return value

    @classmethod
    def _remove_json_pointer(cls, document: Any, path: str) -> None:
        tokens = cls._json_pointer_tokens(path)
        if not tokens:
            return
        parent = document
        for token in tokens[:-1]:
            if isinstance(parent, list) and token.isdigit():
                index = int(token)
                if index >= len(parent):
                    return
                parent = parent[index]
            elif isinstance(parent, dict):
                parent = parent.get(token)
            else:
                return
        last = tokens[-1]
        if isinstance(parent, list) and last.isdigit():
            index = int(last)
            if index < len(parent):
                parent.pop(index)
        elif isinstance(parent, dict):
            parent.pop(last, None)

    @classmethod
    def _replace_json_pointer(cls, document: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
        replaced = cls._clone_json_object(document)
        tokens = cls._json_pointer_tokens(path)
        if not tokens:
            if not isinstance(value, dict):
                raise ValueError("Root profile document replacement must be an object.")
            return cls._clone_json_object(value)
        parent: Any = replaced
        for token in tokens[:-1]:
            if isinstance(parent, list) and token.isdigit():
                index = int(token)
                if index >= len(parent):
                    raise IndexError(f"JSON Pointer parent index is out of range: {path}")
                parent = parent[index]
            elif isinstance(parent, dict):
                if token not in parent:
                    raise KeyError(f"JSON Pointer parent path does not exist: {path}")
                parent = parent[token]
            else:
                raise TypeError(f"JSON Pointer parent is not replaceable: {path}")
        last = tokens[-1]
        if isinstance(parent, list) and last.isdigit():
            index = int(last)
            if index >= len(parent):
                raise IndexError(f"JSON Pointer target index is out of range: {path}")
            parent[index] = cls._clone_json_object(value)
        elif isinstance(parent, dict):
            if last not in parent:
                raise KeyError(f"JSON Pointer target path does not exist: {path}")
            parent[last] = cls._clone_json_object(value)
        else:
            raise TypeError(f"JSON Pointer target is not replaceable: {path}")
        return replaced

    @classmethod
    def _prune_initial_draft_scaffold(
        cls,
        document: dict[str, Any],
        scaffold: dict[str, Any],
    ) -> dict[str, Any]:
        pruned = cls._clone_json_object(document)
        entries = [
            entry
            for entry in scaffold.get("entries", [])
            if isinstance(entry, dict) and entry.get("prune_if_unchanged")
        ]
        for entry in sorted(entries, key=lambda item: len(str(item.get("path", "")).split("/")), reverse=True):
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            current = cls._value_at_json_pointer(pruned, path)
            if current == entry.get("value"):
                cls._remove_json_pointer(pruned, path)
                tokens = cls._json_pointer_tokens(path)
                if len(tokens) == 2 and tokens[1].isdigit():
                    parent_path = f"/{tokens[0]}"
                    parent = cls._value_at_json_pointer(pruned, parent_path)
                    if parent == []:
                        cls._remove_json_pointer(pruned, parent_path)
        return pruned

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
        evidence_context: EvidenceContext,
    ) -> str:
        for note in evidence_context.notes:
            title = ExtractionService._title_value_from_note(note)
            if title:
                return title
        for category in ("entity_signal", "measurement_signal", "method_signal", "resource_signal"):
            for note in evidence_context.notes:
                if (
                    note.category == category
                    and note.observation.strip()
                    and not ExtractionService._is_low_level_parameter_note(note)
                ):
                    return note.observation.strip()
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
    def _json_pointer_unescape(value: str) -> str:
        return value.replace("~1", "/").replace("~0", "~")

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

    def _prepare_run_state(
        self,
        *,
        ranking: FileRankingResult,
        ordered_chunks: list[ContentChunk],
        persisted_state: ExtractionRunState | None,
        profile_identifier: str | None,
        vocab_query_config: ExtractionVocabQueryConfig,
        chunk_repair_mode: ChunkRepairMode = "deferred",
    ) -> ExtractionRunState:
        preserve_initial_context = (
            persisted_state is not None
            and self._initial_file_summaries_match_current_run(
                summaries=persisted_state.initial_file_summaries,
                status=persisted_state.initial_file_summary_status,
                ranking=ranking,
            )
            and self._initial_overview_matches_current_run(
                overview=persisted_state.initial_extraction_overview,
                status=persisted_state.initial_extraction_overview_status,
                ranking=ranking,
            )
        )
        preserve_completed_chunks = preserve_initial_context
        persisted_by_key = {
            self._chunk_result_key(result): result
            for result in (
                persisted_state.chunk_results
                if preserve_completed_chunks and persisted_state
                else []
            )
        }
        chunk_results: list[ExtractionChunkResult] = []
        for index, chunk in enumerate(ordered_chunks):
            existing = persisted_by_key.get(self._chunk_key(chunk))
            if (
                existing
                and existing.status in {"completed", "skipped"}
                and existing.evidence_context is not None
            ):
                chunk_results.append(
                    existing.model_copy(
                        update={
                            "chunk_index": index,
                            "status": existing.status,
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
            chunk_repair_mode=chunk_repair_mode,
            chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            vocab_query_config=(
                persisted_state.vocab_query_config
                if persisted_state
                else vocab_query_config
            ),
            ranked_files=ranking.files,
            initial_file_summaries=(
                persisted_state.initial_file_summaries
                if preserve_initial_context and persisted_state
                else []
            ),
            initial_file_summary_status=(
                persisted_state.initial_file_summary_status
                if preserve_initial_context and persisted_state
                else None
            ),
            initial_extraction_overview=(
                persisted_state.initial_extraction_overview
                if preserve_initial_context and persisted_state
                else None
            ),
            initial_extraction_overview_status=(
                persisted_state.initial_extraction_overview_status
                if preserve_initial_context and persisted_state
                else None
            ),
            chunk_results=chunk_results,
            vocab_queries=persisted_state.vocab_queries if persisted_state else [],
            generated_final_draft=(
                persisted_state.generated_final_draft if persisted_state else None
            ),
            curated_document=(
                persisted_state.curated_document if persisted_state else None
            ),
            draft_quality_state=(
                persisted_state.draft_quality_state if persisted_state else None
            ),
            validation=(
                persisted_state.validation if persisted_state else DraftValidationResult()
            ),
            curated_validation=(
                persisted_state.curated_validation if persisted_state else None
            ),
            projection_ledger=(
                persisted_state.projection_ledger if persisted_state else []
            ),
            initial_draft_scaffold=(
                persisted_state.initial_draft_scaffold if persisted_state else {}
            ),
            field_completion_ledger=(
                persisted_state.field_completion_ledger if persisted_state else []
            ),
            curation_ledger=(
                persisted_state.curation_ledger if persisted_state else []
            ),
            filtered_evidence_notes=(
                persisted_state.filtered_evidence_notes
                if persisted_state
                else []
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
    def _completed_chunk_evidence_contexts(
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> list[EvidenceContext]:
        return [
            result.evidence_context
            for result in state.chunk_results
            if result.status in {"completed", "skipped"}
            and result.evidence_context is not None
            and (file_path is None or result.file_path == file_path)
        ]

    @classmethod
    def _completed_chunk_count(cls, state: ExtractionRunState) -> int:
        return len(cls._completed_chunk_evidence_contexts(state))

    @classmethod
    def _merged_completed_evidence_context(
        cls,
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> EvidenceContext:
        return merge_evidence_contexts(
            cls._completed_chunk_evidence_contexts(state, file_path=file_path)
        )

    def _filtered_completed_evidence_context(
        self,
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> tuple[EvidenceContext, list[FilteredEvidenceNote]]:
        context = self._merged_completed_evidence_context(state, file_path=file_path)
        rank_by_path = {ranked.file_path: ranked.rank for ranked in state.ranked_files}
        return dedupe_repeated_evidence_notes(
            context,
            file_rank_by_path=rank_by_path,
        )

    def _save_filtered_evidence_notes(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        duplicate_records: list[FilteredEvidenceNote] | None = None,
    ) -> None:
        assert self.output_repository is not None
        records = [*state.filtered_evidence_notes, *(duplicate_records or [])]
        self.output_repository.save_filtered_evidence_notes(
            workflow_id=data_package_id,
            ledger=filtered_evidence_ledger(records),
        )

    def _validate_and_filter_evidence_context_for_chunk(
        self,
        context: EvidenceContext,
        *,
        chunk_content: str,
        chunk_result: ExtractionChunkResult,
        state: ExtractionRunState,
    ) -> EvidenceContext:
        validated_context, dropped_notes = validate_evidence_context_for_chunk(
            context,
            chunk_content=chunk_content,
            file_path=chunk_result.file_path,
            start_idx=chunk_result.start_idx,
            end_idx=chunk_result.end_idx,
        )
        state.filtered_evidence_notes.extend(
            FilteredEvidenceNote(
                reason="evidence_text_unsupported",
                note=dropped,
                file_path=dropped.file_path,
                start_idx=dropped.start_idx,
                end_idx=dropped.end_idx,
                chunk_index=chunk_result.chunk_index,
            )
            for dropped in dropped_notes
        )
        high_signal_context, signal_filtered = filter_evidence_context_by_signal_level(
            validated_context,
            chunk_index=chunk_result.chunk_index,
        )
        state.filtered_evidence_notes.extend(signal_filtered)
        return high_signal_context

    async def _repair_chunk_evidence_context(
        self,
        *,
        data_package_id: str,
        chunk_result: ExtractionChunkResult,
        failure: MaxRetriesExceeded,
        chunk_content: str,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> None:
        assert self.ollama_client is not None
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
                output_type=EvidenceContext,
                temperature=0.1,
                think=None,
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
            return

        self._record_workflow_token_usage(
            data_package_id=data_package_id,
            agent_name="chunk_extraction_repair",
            usage=repair.usage,
        )
        validated_context = self._validate_and_filter_evidence_context_for_chunk(
            repair.output,
            chunk_content=chunk_content,
            chunk_result=chunk_result,
            state=state,
        )
        chunk_result.status = "completed"
        chunk_result.evidence_context = validated_context
        chunk_result.response_duration_ms = self._usage_float(
            repair.usage,
            "response_duration_ms",
        )
        chunk_result.context_tokens = self._usage_int(repair.usage, "input_tokens")
        self._save_run_state(data_package_id, state)

        partial_context = self._save_current_evidence_artifacts(
            data_package_id=data_package_id,
            state=state,
        )
        progress.processed_chunks = self._completed_chunk_count(state)
        progress.interim_evidence_context = partial_context
        progress.current_chunk = None
        progress.chunk_results = state.chunk_results
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)

    def _save_current_evidence_artifacts(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        evidence_context: EvidenceContext | None = None,
    ) -> EvidenceContext:
        assert self.output_repository is not None
        if evidence_context is None:
            evidence_context, duplicate_records = self._filtered_completed_evidence_context(state)
        else:
            rank_by_path = {ranked.file_path: ranked.rank for ranked in state.ranked_files}
            evidence_context, duplicate_records = dedupe_repeated_evidence_notes(
                evidence_context,
                file_rank_by_path=rank_by_path,
            )
        self.output_repository.save_evidence_context(
            workflow_id=data_package_id,
            evidence_context=evidence_context,
        )
        self._save_filtered_evidence_notes(
            data_package_id=data_package_id,
            state=state,
            duplicate_records=duplicate_records,
        )
        return evidence_context

    @staticmethod
    def _filtered_evidence_summary_warnings(
        state: ExtractionRunState,
        duplicate_records: list[FilteredEvidenceNote],
    ) -> list[str]:
        records = [*state.filtered_evidence_notes, *duplicate_records]
        if not records:
            return []
        summary: dict[str, int] = {}
        for record in records:
            summary[record.reason] = summary.get(record.reason, 0) + 1
        labels = {
            "evidence_text_unsupported": "unsupported evidence-text notes",
            "signal_level_filtered": "medium/low signal notes",
            "duplicate_evidence": "duplicate evidence notes",
        }
        return [
            "Filtered evidence notes: "
            + ", ".join(
                f"{count} {labels.get(reason, reason)}"
                for reason, count in sorted(summary.items())
            )
            + "."
        ]

    @staticmethod
    def _evidence_context_with_file_inventory(
        *,
        data_package: Any,
        context: EvidenceContext,
        state: ExtractionRunState,
    ) -> EvidenceContext:
        rank_by_path = {ranked.file_path: ranked.rank for ranked in state.ranked_files}
        summary_by_path = {
            summary.file_path: summary
            for summary in state.initial_file_summaries
            if summary.status == "summarized"
        }
        inventory_by_path = {item.file_path: item for item in context.file_inventory}
        for file in data_package.files:
            summary = summary_by_path.get(file.file_path)
            summary_text = None
            if summary is not None:
                summary_parts = [
                    part
                    for part in (
                        summary.data_format,
                        summary.explicit_purpose,
                        "; ".join(summary.data_characteristics),
                    )
                    if part
                ]
                summary_text = " | ".join(summary_parts) or None
            inventory_by_path[file.file_path] = FileInventoryItem(
                file_path=file.file_path,
                byte_size=len(file.raw_content),
                file_type=getattr(getattr(file, "file_type", None), "value", None),
                rank=rank_by_path.get(file.file_path),
                summary=summary_text,
            )
        return context.model_copy(
            update={
                "file_inventory": sorted(
                    inventory_by_path.values(),
                    key=lambda item: (
                        item.rank if item.rank is not None else 10_000,
                        item.file_path,
                    ),
                )
            }
        )


    @classmethod
    def _merged_completed_evidence_context_or_none(
        cls,
        state: ExtractionRunState,
        *,
        file_path: str | None = None,
    ) -> EvidenceContext | None:
        contexts = cls._completed_chunk_evidence_contexts(state, file_path=file_path)
        if not contexts:
            return None
        return merge_evidence_contexts(contexts)

    @staticmethod
    def _initial_file_summary_for_prompt(
        state: ExtractionRunState,
        *,
        file_path: str,
    ) -> ExtractionFileSummary | None:
        for summary in state.initial_file_summaries:
            if summary.file_path == file_path and summary.status == "summarized":
                return summary
        return None

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
            if result.status in {"completed", "skipped"}
            and result.evidence_context is not None
            and result.file_path == file_path
            and result.chunk_index < before_chunk_index
        ]

    @classmethod
    def _global_evidence_context_for_prompt(
        cls,
        state: ExtractionRunState,
        *,
        current_chunk_index: int,
    ) -> EvidenceContext | None:
        contexts = [
            result.evidence_context
            for result in state.chunk_results
            if result.status in {"completed", "skipped"}
            and result.evidence_context is not None
            and result.chunk_index < current_chunk_index
        ]
        if not contexts:
            return None
        return merge_evidence_contexts(contexts)

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
        try:
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
        except CompletionError as exc:
            usage = getattr(exc, "usage", None)
            if usage is not None:
                self._record_workflow_token_usage(
                    data_package_id=data_package_id,
                    agent_name=agent_name,
                    usage=usage,
                )
            warnings.append(
                f"Vocabulary selector left '{source_value}' unresolved after model failure: {exc}"
            )
            return None
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
            return self.output_repository.load_extraction_result(
                data_package_id,
                chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            )
        except (FileNotFoundError, ValidationError):
            return None

    def _load_evidence_context_or_none(self, data_package_id: str) -> EvidenceContext | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_evidence_context(data_package_id)
        except (FileNotFoundError, ValidationError):
            return None

    def _load_run_state_or_none(self, data_package_id: str) -> ExtractionRunState | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_run_state(
                data_package_id,
                chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            )
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

    def _update_initial_context_progress(
        self,
        data_package_id: str,
        progress: ExtractionRunProgress,
    ) -> None:
        if self.task_registry is None:
            return
        self.task_registry.update_progress(
            self._initial_context_task_name(data_package_id),
            progress.model_dump(mode="json"),
        )

    @staticmethod
    def _initial_context_progress_from_state(
        state: ExtractionRunState,
        *,
        warnings: list[str],
        stage: str,
    ) -> ExtractionRunProgress:
        return ExtractionRunProgress(
            stage=stage,
            ranked_files=state.ranked_files,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            warnings=list(warnings),
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
                chunk_repair_mode=state.chunk_repair_mode if state else "deferred",
                interim_evidence_context=result.machine_evidence_context,
                vocab_query_config=state.vocab_query_config if state else None,
                initial_file_summaries=result.initial_file_summaries,
                initial_file_summary_status=result.initial_file_summary_status,
                initial_extraction_overview=result.initial_extraction_overview,
                initial_extraction_overview_status=result.initial_extraction_overview_status,
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                generated_final_draft=result.generated_final_draft,
                curated_document=result.curated_document,
                draft_quality_state=result.draft_quality_state,
                validation=result.validation,
                curated_validation=result.curated_validation,
                initial_draft_scaffold=result.initial_draft_scaffold,
                projection_ledger=result.projection_ledger,
                field_completion_ledger=result.field_completion_ledger,
                curation_ledger=result.curation_ledger,
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
                chunk_repair_mode=state.chunk_repair_mode if state else "deferred",
                interim_evidence_context=result.machine_evidence_context,
                vocab_query_config=state.vocab_query_config if state else None,
                initial_file_summaries=result.initial_file_summaries,
                initial_file_summary_status=result.initial_file_summary_status,
                initial_extraction_overview=result.initial_extraction_overview,
                initial_extraction_overview_status=result.initial_extraction_overview_status,
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                generated_final_draft=result.generated_final_draft,
                curated_document=result.curated_document,
                draft_quality_state=result.draft_quality_state,
                validation=result.validation,
                curated_validation=result.curated_validation,
                initial_draft_scaffold=result.initial_draft_scaffold,
                projection_ledger=result.projection_ledger,
                field_completion_ledger=result.field_completion_ledger,
                curation_ledger=result.curation_ledger,
                warnings=list(result.warnings),
            )
        state = self._load_run_state_or_none(data_package_id)
        if state is not None:
            if not state.chunk_results and (
                state.initial_file_summary_status is not None
                or state.initial_extraction_overview_status is not None
            ):
                stage = "initial_context_completed"
            else:
                stage = (
                    "profile_draft"
                    if state.generated_final_draft
                    else "interim_evidence_context"
                )
            return TaskStatus.UNKNOWN, ExtractionRunProgress(
                stage=stage,
                chunk_repair_mode=state.chunk_repair_mode,
                processed_chunks=self._completed_chunk_count(state),
                total_chunks=len(state.chunk_results),
                interim_evidence_context=self._merged_completed_evidence_context_or_none(state),
                vocab_query_config=state.vocab_query_config,
                ranked_files=state.ranked_files,
                initial_file_summaries=state.initial_file_summaries,
                initial_file_summary_status=state.initial_file_summary_status,
                initial_extraction_overview=state.initial_extraction_overview,
                initial_extraction_overview_status=state.initial_extraction_overview_status,
                chunk_results=state.chunk_results,
                vocab_queries=state.vocab_queries,
                generated_final_draft=state.generated_final_draft,
                curated_document=state.curated_document,
                draft_quality_state=state.draft_quality_state,
                validation=state.validation,
                curated_validation=state.curated_validation,
                initial_draft_scaffold=state.initial_draft_scaffold,
                projection_ledger=state.projection_ledger,
                field_completion_ledger=state.field_completion_ledger,
                curation_ledger=state.curation_ledger,
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
            "initial_context": TaskStatus.UNKNOWN,
            "chunking": chunking_status,
            "extraction": extraction_status,
            "normalization": TaskStatus.UNKNOWN,
            "profile_projection": TaskStatus.UNKNOWN,
            "validation": TaskStatus.UNKNOWN,
        }
        if extraction_progress is not None:
            if extraction_progress.stage in {
                "initial_context_required",
                "file_ranking",
                "initial_file_summaries",
                "initial_overview",
                "chunk_extraction",
                "chunk_repair",
                "interim_evidence_context",
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
            if extraction_progress.stage == "profile_draft":
                steps["extraction"] = TaskStatus.COMPLETED
            if extraction_progress.stage in {"profile_projection", "profile_draft", "completed"}:
                steps["profile_projection"] = (
                    TaskStatus.RUNNING
                    if extraction_progress.stage == "profile_projection"
                    else TaskStatus.COMPLETED
                )
            if extraction_progress.stage == "completed":
                steps["validation"] = TaskStatus.COMPLETED
        if stage == "initial_context":
            steps["initial_context"] = TaskStatus.RUNNING
        elif stage in {
            "initial_context_completed",
            "chunking",
            "extraction",
            "chunk_extraction",
            "chunk_repair",
            "interim_evidence_context",
            "vocabulary_normalization",
            "profile_projection",
            "profile_draft",
            "completed",
        }:
            steps["initial_context"] = TaskStatus.COMPLETED
        if stage == "completed":
            for name in steps:
                steps[name] = TaskStatus.COMPLETED
        if workflow_status == TaskStatus.CRASHED:
            active_step = "extraction" if chunking_status == TaskStatus.COMPLETED else "chunking"
            steps[active_step] = TaskStatus.CRASHED
        return steps

    @staticmethod
    def _profile_identifier_for_stage(
        *,
        profile_identifier: str | None,
        target_stage: ExtractionTargetStage,
    ) -> str | None:
        if target_stage == "context":
            return None
        normalized_identifier = (profile_identifier or "").strip()
        if not normalized_identifier:
            raise ValueError(
                "A profile identifier is required for profile, grounding, and complete extraction stages."
            )
        return normalized_identifier

    @staticmethod
    def _result_url(data_package_id: str) -> str:
        return f"/api/v1/extraction/result/{data_package_id}"

    @staticmethod
    def _initial_context_task_name(data_package_id: str) -> str:
        return f"initial-context:{data_package_id}"

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
