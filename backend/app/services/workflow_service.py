from __future__ import annotations

from app.services.extraction_shared import *
from app.services.orientation_service import OrientationService
from app.services.evidence_service import EvidenceService
from app.services.grounding_service import GroundingService
from app.services.projection_service import ProjectionService
from app.services.curation_service import CurationService


class WorkflowService(
    OrientationService,
    EvidenceService,
    GroundingService,
    ProjectionService,
    CurationService,
):
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
        force_profile_rebuild: bool = False,
        target_stage: ExtractionTargetStage = "complete",
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
        chunk_repair_mode: ChunkRepairMode = "deferred",
        evidence_critic_granularity: EvidenceCriticGranularity = "per_chunk",
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
            data_package_id,
            chunking_strategy,
        )
        if not chunks_by_file:
            raise ChunkingRequiredError(
                "Extraction requires completed datasource chunking. Run chunking first."
            )

        chat_model = chat_model or self._current_chat_model()
        task_name = self._extraction_task_name(data_package_id, chunking_strategy, chat_model)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is not None and task_info.status == TaskStatus.RUNNING:
            return self._load_result_or_none(data_package_id, chunking_strategy=chunking_strategy, chat_model=chat_model), TaskStatus.RUNNING
        if task_info is not None and task_info.status == TaskStatus.COMPLETED:
            result = self._load_result_or_none(data_package_id, chunking_strategy=chunking_strategy, chat_model=chat_model)
            if result is not None and resume and not force_profile_rebuild:
                return result, TaskStatus.COMPLETED

        if not resume:
            self._clear_downstream_extraction_outputs(data_package_id)
            self._clear_prompt_diagnostics(data_package_id)
        await self.task_registry.create_task(
            coro=self._run_extraction_task(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier_for_stage,
                qualitative_vocab_identifiers=qualitative_vocab_identifiers,
                resume=resume,
                force_profile_rebuild=force_profile_rebuild,
                target_stage=target_stage,
                chunking_strategy=chunking_strategy,
                chat_model=chat_model,
                chunk_repair_mode=chunk_repair_mode,
                evidence_critic_granularity=evidence_critic_granularity,
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
            self.output_repository.clear_initial_context(data_package_id)
        else:
            self._clear_prompt_diagnostics(data_package_id)

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
        state = (self._load_run_state_or_none(data_package_id) or ExtractionRunState()).model_copy(
            update={
                "ranked_files": [],
                "initial_file_summaries": [],
                "initial_file_summary_progress": None,
                "initial_file_summary_status": None,
                "initial_extraction_overview": None,
                "initial_extraction_overview_status": None,
                "initial_extraction_overview_diagnostic": None,
                "dataset_summary": "",
                "chat_model": self.ollama_client.chat_model if self.ollama_client else None,
            }
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
            state=state,
            warnings=warnings,
        )
        progress = self._initial_context_progress_from_state(
            state,
            warnings=warnings,
            stage="file_ranking",
        )
        self._update_initial_context_progress(data_package_id, progress)

        ranking = self._rank_files_from_summaries(data_package=data_package, state=state)
        state.ranked_files = ranking.files
        self._save_run_state(data_package_id, state)
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
        progress = self._initial_context_progress_from_state(
            state,
            warnings=warnings,
            stage="dataset_summary",
        )
        self._update_initial_context_progress(data_package_id, progress)

        await self._generate_initial_dataset_summary(
            data_package_id=data_package_id,
            state=state,
            warnings=warnings,
            skip_without_runtime=True,
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
        chunking_strategy: str = "semantic",
        fixed_tokens_per_chunk: int = 1024,
        min_tokens_per_chunk: int = 128,
        max_tokens_per_chunk: int = 1024,
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
        if min_tokens_per_chunk > max_tokens_per_chunk:
            raise ValueError("min_tokens_per_chunk must be less than or equal to max_tokens_per_chunk.")

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
                chunking_strategy=chunking_strategy,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
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
        chunking_strategy: str,
        fixed_tokens_per_chunk: int,
        min_tokens_per_chunk: int,
        max_tokens_per_chunk: int,
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
            chunking_strategy=chunking_strategy,
            fixed_tokens_per_chunk=fixed_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
            max_tokens_per_chunk=max_tokens_per_chunk,
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
            data_package_id,
            chunking_strategy,
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
            chunking_strategy=chunking_strategy,
        )
        if extraction_status == TaskStatus.RUNNING:
            await self.task_registry.wait_for_task(
                self._extraction_task_name(data_package_id, chunking_strategy, self._current_chat_model())
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
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> tuple[TaskStatus, ExtractionRunProgress | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None
        branch_strategy = chunking_strategy or "semantic"
        branch_model = chat_model or self._current_chat_model()
        task_info = self.task_registry.get_task_info(
            self._extraction_task_name(data_package_id, branch_strategy, branch_model)
        )
        if task_info is not None and chunking_strategy:
            state = self._load_run_state_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
            if state is not None and state.chunking_strategy != chunking_strategy:
                task_info = None
        if task_info is None:
            state = self._load_run_state_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
            state_for_branch = state if not chunking_strategy or state is None or state.chunking_strategy == chunking_strategy else None
            result = self._load_result_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
            if result is not None:
                return TaskStatus.COMPLETED, ExtractionRunProgress(
                    stage="completed",
                    processed_chunks=self._completed_chunk_count(state_for_branch) if state_for_branch else 0,
                    total_chunks=len(state_for_branch.chunk_results) if state_for_branch else 0,
                    interim_evidence_context=result.machine_evidence_context,
                    vocab_query_config=state_for_branch.vocab_query_config if state_for_branch else self._default_vocab_query_config(None),
                    ranked_files=state_for_branch.ranked_files if state_for_branch else [],
                    initial_file_summaries=result.initial_file_summaries,
                    initial_file_summary_progress=(
                        state_for_branch.initial_file_summary_progress if state_for_branch else None
                    ),
                    initial_file_summary_status=result.initial_file_summary_status,
                    initial_extraction_overview=result.initial_extraction_overview,
                    initial_extraction_overview_status=result.initial_extraction_overview_status,
                    initial_extraction_overview_diagnostic=(
                        state_for_branch.initial_extraction_overview_diagnostic if state_for_branch else None
                    ),
                    chunk_results=state_for_branch.chunk_results if state_for_branch else [],
                    vocab_queries=state_for_branch.vocab_queries if state_for_branch else [],
                    generated_final_draft=result.generated_final_draft,
                    curated_document=result.curated_document,
                    document_quality_state=result.document_quality_state,
                    draft_quality_state=result.draft_quality_state,
                    validation=result.validation,
                    curated_validation=result.curated_validation,
                    initial_draft_scaffold=result.initial_draft_scaffold,
                    projection_ledger=result.projection_ledger,
                    field_completion_ledger=result.field_completion_ledger,
                    curation_ledger=result.curation_ledger,
                    warnings=list(result.warnings),
                )
            interim_evidence_context = self._load_evidence_context_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
            mismatched_state = None
            if chunking_strategy and state is not None and state.chunking_strategy != chunking_strategy:
                mismatched_state = state
                state = None
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
                    initial_file_summary_progress=(
                        state.initial_file_summary_progress if state else None
                    ),
                    initial_file_summary_status=state.initial_file_summary_status if state else None,
                    initial_extraction_overview=state.initial_extraction_overview if state else None,
                    initial_extraction_overview_status=state.initial_extraction_overview_status if state else None,
                    initial_extraction_overview_diagnostic=(
                        state.initial_extraction_overview_diagnostic if state else None
                    ),
                    chunk_results=state.chunk_results if state else [],
                    vocab_queries=state.vocab_queries if state else [],
                    generated_final_draft=state.generated_final_draft if state else None,
                    curated_document=state.curated_document if state else None,
                    document_quality_state=state.document_quality_state if state else None,
                    draft_quality_state=state.draft_quality_state if state else None,
                    validation=state.validation if state else DraftValidationResult(),
                    curated_validation=state.curated_validation if state else None,
                    initial_draft_scaffold=state.initial_draft_scaffold if state else {},
                    projection_ledger=state.projection_ledger if state else [],
                    field_completion_ledger=state.field_completion_ledger if state else [],
                    curation_ledger=state.curation_ledger if state else [],
                    warnings=self._load_warnings_or_empty(data_package_id),
                )
            if mismatched_state is not None:
                return TaskStatus.UNKNOWN, self._initial_context_progress_from_state(
                    mismatched_state,
                    warnings=self._load_warnings_or_empty(data_package_id),
                    stage="initial_context_completed",
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
            state = self._load_run_state_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
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
                    initial_file_summary_progress=state.initial_file_summary_progress,
                    initial_file_summary_status=state.initial_file_summary_status,
                    initial_extraction_overview=state.initial_extraction_overview,
                    initial_extraction_overview_status=state.initial_extraction_overview_status,
                    initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    generated_final_draft=state.generated_final_draft,
                    curated_document=state.curated_document,
                    document_quality_state=state.document_quality_state,
                    draft_quality_state=state.draft_quality_state,
                    validation=state.validation,
                    curated_validation=state.curated_validation,
                    projection_ledger=state.projection_ledger,
                    field_completion_ledger=state.field_completion_ledger,
                    curation_ledger=state.curation_ledger,
                )
        if progress is not None and progress.interim_evidence_context is None:
            progress.interim_evidence_context = self._load_evidence_context_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
        if progress is not None:
            state = self._load_run_state_or_none(data_package_id, chunking_strategy=branch_strategy, chat_model=branch_model)
            if state is not None and (chunking_strategy is None or state.chunking_strategy == chunking_strategy):
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
                progress.document_quality_state = state.document_quality_state
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

        state = self._load_run_state_or_none(data_package_id)
        task_name = self._extraction_task_name(
            data_package_id,
            state.chunking_strategy if state else "semantic",
            state.chat_model if state else self._current_chat_model(),
        )
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is None or task_info.status != TaskStatus.RUNNING:
            return await self.get_extraction_progress(data_package_id=data_package_id)

        await self.task_registry.cancel_task(task_name)

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
            initial_file_summary_progress=state.initial_file_summary_progress,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
            chunk_results=state.chunk_results,
            vocab_queries=state.vocab_queries,
            generated_final_draft=state.generated_final_draft,
            curated_document=state.curated_document,
            document_quality_state=state.document_quality_state,
            draft_quality_state=state.draft_quality_state,
            validation=state.validation,
            curated_validation=state.curated_validation,
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            evidence_query_ledger=state.evidence_query_ledger,
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
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> ExtractionRunResult:
        if self.output_repository is None:
            raise ExtractionResultNotFoundError("Extraction output repository is unavailable.")
        try:
            branch_model = chat_model or self._current_chat_model()
            state = self._load_run_state_or_none(
                data_package_id,
                chunking_strategy=chunking_strategy or "semantic",
                chat_model=branch_model,
            )
            return self.output_repository.load_extraction_result(
                data_package_id,
                chat_model=state.chat_model if state else branch_model,
                chunking_strategy=chunking_strategy or (state.chunking_strategy if state else "semantic"),
            )
        except FileNotFoundError as exc:
            raise ExtractionResultNotFoundError(
                f"Extraction result not found for workflow '{data_package_id}'."
            ) from exc

    async def get_token_usage(
        self,
        data_package_id: str,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> dict[str, Any]:
        if self.output_repository is None:
            return {"agents": {}}
        branch_model = chat_model or self._current_chat_model()
        state = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy=chunking_strategy or "semantic",
            chat_model=branch_model,
        )
        return self._token_usage_summary(
            self.output_repository.load_token_usage(
                data_package_id,
                chat_model=state.chat_model if state else branch_model,
                chunking_strategy=chunking_strategy or (state.chunking_strategy if state else "semantic"),
            )
        )

    async def _run_extraction_task(
        self,
        *,
        data_package_id: str,
        profile_identifier: str | None,
        qualitative_vocab_identifiers: list[str] | None,
        resume: bool = False,
        force_profile_rebuild: bool = False,
        target_stage: ExtractionTargetStage = "complete",
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
        chunk_repair_mode: ChunkRepairMode = "deferred",
        evidence_critic_granularity: EvidenceCriticGranularity = "per_chunk",
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
            data_package_id,
            chunking_strategy,
        )
        if not chunks_by_file:
            raise ChunkingRequiredError(
                "Extraction requires completed datasource chunking. Run chunking first."
            )

        warnings: list[str] = []
        chat_model = chat_model or self._current_chat_model()
        persisted_state = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy=chunking_strategy,
            chat_model=chat_model,
        )
        persisted_state = self._with_initial_context_fallback(
            data_package_id=data_package_id,
            state=persisted_state,
            chunking_strategy=chunking_strategy,
            chat_model=chat_model,
        )
        progress = ExtractionRunProgress(
            stage="file_ranking",
            chunk_repair_mode=chunk_repair_mode,
            evidence_critic_granularity=evidence_critic_granularity,
            total_chunks=sum(len(chunks) for chunks in chunks_by_file),
            ranked_files=persisted_state.ranked_files if persisted_state else [],
            initial_file_summaries=(
                persisted_state.initial_file_summaries if persisted_state else []
            ),
            initial_file_summary_progress=(
                persisted_state.initial_file_summary_progress if persisted_state else None
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
            initial_extraction_overview_diagnostic=(
                persisted_state.initial_extraction_overview_diagnostic
                if persisted_state
                else None
            ),
            dataset_summary=persisted_state.dataset_summary if persisted_state else "",
            chunk_results=persisted_state.chunk_results if persisted_state else [],
            vocab_queries=persisted_state.vocab_queries if persisted_state else [],
            generated_final_draft=(
                persisted_state.generated_final_draft if persisted_state else None
            ),
            curated_document=(
                persisted_state.curated_document if persisted_state else None
            ),
            document_quality_state=(
                persisted_state.document_quality_state if persisted_state else None
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
        if force_profile_rebuild and target_stage in {"profile", "grounding", "complete"}:
            self._clear_profile_projection_progress(progress)
            self._clear_profile_projection_token_usage(data_package_id)
        self._update_progress(data_package_id, progress)

        if persisted_state and persisted_state.ranked_files:
            ranking = FileRankingResult(files=persisted_state.ranked_files)
        elif persisted_state and persisted_state.initial_file_summaries:
            ranking = self._rank_files_from_summaries(
                data_package=data_package,
                state=persisted_state,
            )
        else:
            ranking = await self._rank_files(data_package_id, data_package, warnings)
        ordered_chunks = self._ordered_chunks(chunks_by_file, ranking)
        state = self._prepare_run_state(
            ranking=ranking,
            ordered_chunks=ordered_chunks,
            persisted_state=persisted_state,
            profile_identifier=profile_identifier,
            vocab_query_config=progress.vocab_query_config,
            chunking_strategy=chunking_strategy,
            chunk_repair_mode=chunk_repair_mode,
            evidence_critic_granularity=evidence_critic_granularity,
        )
        if force_profile_rebuild and target_stage in {"profile", "grounding", "complete"}:
            self._clear_profile_projection_state(state)
        self._save_run_state(data_package_id, state)

        progress.chunk_repair_mode = state.chunk_repair_mode
        progress.evidence_critic_granularity = state.evidence_critic_granularity
        progress.ranked_files = state.ranked_files
        progress.initial_file_summaries = state.initial_file_summaries
        progress.initial_file_summary_status = state.initial_file_summary_status
        progress.initial_extraction_overview = state.initial_extraction_overview
        progress.initial_extraction_overview_status = state.initial_extraction_overview_status
        progress.dataset_summary = state.dataset_summary
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
        evidence_prompt_budgeter = self._prompt_token_budgeter()
        evidence_prompt_budget_warning = (
            (
                "Evidence prompt token budgeting is using conservative estimates: "
                + evidence_prompt_budgeter.fallback_reason
            )
            if evidence_prompt_budgeter.fallback_reason
            else None
        )

        try:
            for chunk_result, chunk in zip(state.chunk_results, ordered_chunks):
                if chunk_result.status in {"completed", "skipped"} and chunk_result.evidence_context is not None:
                    continue

                if is_noisy_payload_chunk(chunk.content):
                    chunk_result.status = "skipped"
                    chunk_result.error = None
                    chunk_result.skip_reason = "encoded_or_payload_dominated_chunk"
                    chunk_result.evidence_context = RoutedEvidenceContext()
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
                    current_file_summary = self._initial_file_summary_for_prompt(
                        state,
                        file_path=chunk.file_path,
                    )
                    evidence_system_components = build_evidence_system_prompt_components_with_overview(
                        EVIDENCE_CONTEXT_SYSTEM_PROMPT,
                        overview=state.initial_extraction_overview,
                        overview_status=state.initial_extraction_overview_status,
                        file_summary=current_file_summary,
                        token_budgeter=evidence_prompt_budgeter,
                    )
                    evidence_chunk_context = EvidenceChunkContext(
                        content=normalized_chunk_content,
                        metadata=EvidenceChunkMetadata(
                            start_idx=chunk.start_idx,
                            end_idx=chunk.end_idx,
                            file_path=chunk.file_path,
                            data_package_name=data_package.file_name,
                        ),
                    )
                    evidence_prompt_components = build_evidence_context_prompt_components(
                        evidence_chunk_context
                    )
                    result = await generate_structured(
                        self.ollama_client,
                        model=self.ollama_client.chat_model,
                        system="".join(text for _, text in evidence_system_components),
                        prompt="".join(text for _, text in evidence_prompt_components),
                        system_components=evidence_system_components,
                        prompt_components=evidence_prompt_components,
                        token_budgeter=evidence_prompt_budgeter,
                        operation_id=self._prompt_operation_id(
                            "chunk_extraction",
                            chunk.file_path,
                            chunk_result.chunk_index,
                        ),
                        agent_name="chunk_extraction",
                        diagnostic_metadata={
                            "file_path": chunk.file_path,
                            "chunk_index": chunk_result.chunk_index,
                            "start_idx": chunk.start_idx,
                            "end_idx": chunk.end_idx,
                        },
                        output_type=EvidenceContext,
                        omitted_fields={
                            "EvidenceCandidate": [
                                "candidate_id",
                                "file_path",
                                "start_idx",
                                "end_idx",
                                "evidence_match_score",
                            ]
                        },
                        retries=2,
                        temperature=0.1,
                        think=None,
                        num_ctx=self.ollama_client.max_context_length,
                    )
                except MaxRetriesExceeded as exc:
                    self._record_llm_call_exception(
                        data_package_id=data_package_id,
                        exc=exc,
                        agent_name="chunk_extraction",
                    )
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
                    chunk_result.context_tokens = self._single_attempt_input_tokens(exc)
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
                    self._record_llm_call_exception(
                        data_package_id=data_package_id,
                        exc=exc,
                        agent_name="chunk_extraction",
                    )
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

                self._record_llm_call_result(
                    data_package_id=data_package_id,
                    result=result,
                    agent_name="chunk_extraction",
                )
                validated_context = await self._validate_assess_and_route_evidence_context_for_chunk(
                    result.output,
                    chunk_content=normalized_chunk_content,
                    chunk_context=evidence_chunk_context,
                    chunk_result=chunk_result,
                    state=state,
                    critic_granularity=evidence_critic_granularity,
                    data_package_id=data_package_id,
                )
                chunk_result.status = "completed"
                chunk_result.evidence_context = validated_context
                chunk_result.response_duration_ms = self._usage_float(
                    result.usage,
                    "response_duration_ms",
                )
                chunk_result.context_tokens = self._single_attempt_input_tokens(result)
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
            chunking_strategy=state.chunking_strategy,
            chat_model=state.chat_model,
        )
        self._save_filtered_evidence_notes(
            data_package_id=data_package_id,
            state=state,
            duplicate_records=duplicate_records,
        )
        if (
            evidence_prompt_budget_warning
            and evidence_prompt_budget_warning not in warnings
        ):
            warnings.append(evidence_prompt_budget_warning)
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

        if state.validation.status == "valid":
            profile_document = await self._enrich_draft_with_requirements(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
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
            progress.requirement_report = state.requirement_report
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

    def _load_result_or_none(
        self,
        data_package_id: str,
        *,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> ExtractionRunResult | None:
        if self.output_repository is None:
            return None
        try:
            branch_model = chat_model or self._current_chat_model()
            state = self._load_run_state_or_none(
                data_package_id,
                chunking_strategy=chunking_strategy or "semantic",
                chat_model=branch_model,
            )
            return self.output_repository.load_extraction_result(
                data_package_id,
                chat_model=state.chat_model if state else branch_model,
                chunking_strategy=chunking_strategy or (state.chunking_strategy if state else "semantic"),
            )
        except (FileNotFoundError, ValidationError):
            return None

    def _load_evidence_context_or_none(
        self,
        data_package_id: str,
        *,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> EvidenceContext | None:
        if self.output_repository is None:
            return None
        try:
            branch_model = chat_model or self._current_chat_model()
            state = self._load_run_state_or_none(
                data_package_id,
                chunking_strategy=chunking_strategy or "semantic",
                chat_model=branch_model,
            )
            return self.output_repository.load_evidence_context(
                data_package_id,
                chunking_strategy=chunking_strategy or (state.chunking_strategy if state else "semantic"),
                chat_model=state.chat_model if state else branch_model,
            )
        except (FileNotFoundError, ValidationError):
            return None

    def _load_run_state_or_none(
        self,
        data_package_id: str,
        *,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> ExtractionRunState | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_run_state(
                data_package_id,
                chat_model=chat_model or self._current_chat_model(),
                chunking_strategy=chunking_strategy,
            )
        except (FileNotFoundError, json.JSONDecodeError, ValidationError):
            return None

    def _with_initial_context_fallback(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState | None,
        chunking_strategy: str,
        chat_model: str | None,
    ) -> ExtractionRunState | None:
        if state is not None and self._has_initial_context(state):
            return state
        if chunking_strategy == "semantic":
            return state
        source = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy="semantic",
            chat_model=chat_model,
        )
        if source is None or not self._has_initial_context(source):
            return state
        target = (
            state.model_copy(deep=True)
            if state is not None
            else ExtractionRunState(
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        )
        target.ranked_files = source.ranked_files
        target.initial_file_summaries = source.initial_file_summaries
        target.initial_file_summary_progress = source.initial_file_summary_progress
        target.initial_file_summary_status = source.initial_file_summary_status
        target.initial_extraction_overview = source.initial_extraction_overview
        target.initial_extraction_overview_status = source.initial_extraction_overview_status
        target.initial_extraction_overview_diagnostic = source.initial_extraction_overview_diagnostic
        target.dataset_summary = source.dataset_summary
        return target

    @staticmethod
    def _has_initial_context(state: ExtractionRunState) -> bool:
        return (
            state.initial_file_summary_status is not None
            and state.initial_extraction_overview_status is not None
        )

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
            chat_model=state.chat_model,
            chunking_strategy=state.chunking_strategy,
        )

    def _current_chat_model(self) -> str | None:
        return self.ollama_client.chat_model if self.ollama_client else None

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
                "WorkflowService requires datasource_service, ollama_client, "
                "output_repository, and task_registry to run extraction."
            )

    def _update_progress(
        self,
        data_package_id: str,
        progress: ExtractionRunProgress,
    ) -> None:
        if self.task_registry is None:
            return
        state = self._load_run_state_or_none(data_package_id)
        self.task_registry.update_progress(
            self._extraction_task_name(
                data_package_id,
                state.chunking_strategy if state else "semantic",
                state.chat_model if state else self._current_chat_model(),
            ),
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
            initial_file_summary_progress=state.initial_file_summary_progress,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
            dataset_summary=state.dataset_summary,
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
                initial_file_summary_progress=(
                    state.initial_file_summary_progress if state else None
                ),
                initial_file_summary_status=result.initial_file_summary_status,
                initial_extraction_overview=result.initial_extraction_overview,
                initial_extraction_overview_status=result.initial_extraction_overview_status,
                dataset_summary=result.dataset_summary,
                initial_extraction_overview_diagnostic=(
                    state.initial_extraction_overview_diagnostic if state else None
                ),
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                generated_final_draft=result.generated_final_draft,
                curated_document=result.curated_document,
                document_quality_state=result.document_quality_state,
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
        state = self._load_run_state_or_none(data_package_id)
        task_info = self.task_registry.get_task_info(
            self._extraction_task_name(
                data_package_id,
                state.chunking_strategy if state else "semantic",
                state.chat_model if state else self._current_chat_model(),
            )
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
                initial_file_summary_progress=(
                    state.initial_file_summary_progress if state else None
                ),
                initial_file_summary_status=result.initial_file_summary_status,
                initial_extraction_overview=result.initial_extraction_overview,
                initial_extraction_overview_status=result.initial_extraction_overview_status,
                dataset_summary=result.dataset_summary,
                initial_extraction_overview_diagnostic=(
                    state.initial_extraction_overview_diagnostic if state else None
                ),
                chunk_results=state.chunk_results if state else [],
                vocab_queries=state.vocab_queries if state else [],
                generated_final_draft=result.generated_final_draft,
                curated_document=result.curated_document,
                document_quality_state=result.document_quality_state,
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
                initial_file_summary_progress=state.initial_file_summary_progress,
                initial_file_summary_status=state.initial_file_summary_status,
                initial_extraction_overview=state.initial_extraction_overview,
                initial_extraction_overview_status=state.initial_extraction_overview_status,
                dataset_summary=state.dataset_summary,
                initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
                chunk_results=state.chunk_results,
                vocab_queries=state.vocab_queries,
                generated_final_draft=state.generated_final_draft,
                curated_document=state.curated_document,
                document_quality_state=state.document_quality_state,
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
        return f"/api/v1/extraction/results/{data_package_id}"

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
        task_branch = self._current_extraction_task_branch(data_package_id)
        state = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy=task_branch[0] if task_branch else "semantic",
            chat_model=task_branch[1] if task_branch else None,
        )
        chat_model = state.chat_model if state else (task_branch[1] if task_branch else (self.ollama_client.chat_model if self.ollama_client else None))
        chunking_strategy = state.chunking_strategy if state else (task_branch[0] if task_branch else "semantic")
        totals = self.output_repository.load_token_usage(
            data_package_id,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
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
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )

    def _record_llm_call_result(
        self,
        *,
        data_package_id: str,
        result: Any,
        agent_name: str | None = None,
    ) -> None:
        resolved_agent_name = self._llm_call_agent_name(result, fallback=agent_name)
        if resolved_agent_name:
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name=resolved_agent_name,
                usage=result.usage,
            )
        self._persist_result_prompt_diagnostics(
            data_package_id=data_package_id,
            result=result,
        )

    @staticmethod
    def _current_extraction_task_branch(data_package_id: str) -> tuple[str, str | None] | None:
        task = asyncio.current_task()
        if task is None:
            return None
        prefix = f"extraction:run:{data_package_id}:"
        name = task.get_name()
        if not name.startswith(prefix):
            return None
        remainder = name[len(prefix):]
        strategy, separator, chat_model = remainder.partition(":")
        if strategy not in {"semantic", "fixed_tokens"}:
            return None
        return strategy, chat_model if separator and chat_model != "default-model" else None

    def _record_llm_call_exception(
        self,
        *,
        data_package_id: str,
        exc: BaseException,
        agent_name: str | None = None,
    ) -> None:
        usage = getattr(exc, "usage", None)
        resolved_agent_name = self._llm_call_agent_name(exc, fallback=agent_name)
        if usage is not None and resolved_agent_name:
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name=resolved_agent_name,
                usage=usage,
            )
        self._persist_exception_prompt_diagnostics(
            data_package_id=data_package_id,
            exc=exc,
        )

    @staticmethod
    def _llm_call_agent_name(source: Any, *, fallback: str | None = None) -> str | None:
        diagnostics = getattr(source, "prompt_diagnostics", None)
        if diagnostics is not None and getattr(diagnostics, "agent_name", None):
            return diagnostics.agent_name
        return fallback

    def _single_attempt_input_tokens(self, source: Any) -> int | None:
        diagnostics = getattr(source, "prompt_diagnostics", None)
        attempt_inputs: list[int] = []
        for attempt in getattr(diagnostics, "attempts", []) or []:
            usage = getattr(attempt, "usage", None) or {}
            tokens = self._usage_int(usage, "input_tokens")
            if tokens:
                attempt_inputs.append(tokens)
        if attempt_inputs:
            return max(attempt_inputs)
        usage = getattr(source, "usage", None)
        if usage is None:
            return None
        return self._usage_int(usage, "input_tokens")

    def _persist_prompt_diagnostics(
        self,
        *,
        data_package_id: str,
        diagnostics: PromptCompletionDiagnostics | None,
        status: str | None = None,
    ) -> None:
        if self.output_repository is None or diagnostics is None:
            return
        if status is not None:
            diagnostics.status = status
        state = self._load_run_state_or_none(data_package_id)
        self.output_repository.append_prompt_diagnostic(
            workflow_id=data_package_id,
            chat_model=self.ollama_client.chat_model if self.ollama_client else None,
            chunking_strategy=state.chunking_strategy if state else "semantic",
            diagnostic=diagnostics.model_dump(mode="json"),
        )

    def _persist_result_prompt_diagnostics(
        self,
        *,
        data_package_id: str,
        result: Any,
    ) -> None:
        self._persist_prompt_diagnostics(
            data_package_id=data_package_id,
            diagnostics=getattr(result, "prompt_diagnostics", None),
            status="completed",
        )

    def _persist_exception_prompt_diagnostics(
        self,
        *,
        data_package_id: str,
        exc: BaseException,
    ) -> None:
        self._persist_prompt_diagnostics(
            data_package_id=data_package_id,
            diagnostics=getattr(exc, "prompt_diagnostics", None),
            status="failed",
        )

    def _clear_prompt_diagnostics(self, data_package_id: str) -> None:
        if self.output_repository is None:
            return
        self.output_repository.clear_prompt_diagnostics(
            data_package_id,
            chat_model=self.ollama_client.chat_model if self.ollama_client else None,
        )

    def _prompt_token_budgeter(self) -> PromptTokenBudgeter:
        return PromptTokenBudgeter.from_tokenizer_source(
            getattr(self.settings, "ollama_chat_tokenizer", ""),
            hf_token=getattr(self.settings, "hf_token", ""),
        )

    @staticmethod
    def _prompt_operation_id(
        agent_name: str,
        *parts: Any,
    ) -> str:
        normalized = [
            re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)).strip("_")
            for part in parts
            if part is not None and str(part) != ""
        ]
        return "__".join([agent_name, *normalized]) if normalized else agent_name

    @staticmethod
    def _usage_int(usage: Any, field_name: str) -> int:
        try:
            if isinstance(usage, dict):
                return int(usage.get(field_name, 0) or 0)
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
    def _extraction_task_name(
        data_package_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> str:
        return f"extraction:run:{data_package_id}:{chunking_strategy}:{chat_model or 'default-model'}"

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


    RoutedEvidenceContext,
