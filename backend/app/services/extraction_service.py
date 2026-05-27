from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk
from app.domain.extraction import (
    DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS,
    EXTRACTION_CONTEXT_SYSTEM_PROMPT,
    FILE_RANKING_SYSTEM_PROMPT,
    PROFILE_PROJECTION_SYSTEM_PROMPT,
    QUDT_QUANTITY_KIND_VOCAB,
    QUDT_UNIT_VOCAB,
    VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
    VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
    ChunkContext,
    ChunkingRequiredError,
    ExtractionContext,
    ExtractionNormalization,
    ExtractionResultNotFoundError,
    ExtractionRunProgress,
    ExtractionRunResult,
    ExtractionValidationError,
    FileContext,
    FileRankingResult,
    QualitativeAttribute,
    QualitativeAttributeNormalization,
    Quantity,
    QuantityNormalization,
    VocabularyCandidateSelection,
    VocabularyFallbackQuery,
    VocabularyTermMapping,
    build_candidate_selection_prompt,
    build_extraction_context_prompt,
    build_fallback_query_prompt,
    build_file_ranking_prompt,
    build_profile_projection_prompt,
    build_qualitative_vocab_query,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    fallback_file_ranking,
    merge_extraction_context_results,
)
from app.domain.profiles import remove_null_values, validation_schema_for_target_class
from app.domain.semantics import VocabQuery, VocabQueryResult
from app.ollama.completion import generate_structured
from app.ollama.errors import CompletionError
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService
    from app.services.semantic_service import SemanticService


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

        self.output_repository.clear_extraction_run(data_package_id)
        await self.task_registry.create_task(
            coro=self._run_extraction_task(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                qualitative_vocab_identifiers=qualitative_vocab_identifiers,
            ),
            type=TaskType.WORKFLOW,
            name=task_name,
        )
        return None, TaskStatus.RUNNING

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
            result = self._load_result_or_none(data_package_id)
            if result is not None:
                return TaskStatus.COMPLETED, ExtractionRunProgress(
                    stage="completed",
                    interim_context=result.extraction_context,
                    warnings=list(result.warnings),
                )
            interim_context = self._load_context_or_none(data_package_id)
            if interim_context is not None:
                return TaskStatus.UNKNOWN, ExtractionRunProgress(
                    stage="interim_context",
                    interim_context=interim_context,
                    warnings=self._load_warnings_or_empty(data_package_id),
                )
            return TaskStatus.UNKNOWN, None
        progress = (
            ExtractionRunProgress.model_validate(task_info.progress)
            if task_info.progress
            else None
        )
        if progress is not None and progress.interim_context is None:
            progress.interim_context = self._load_context_or_none(data_package_id)
        return task_info.status, progress

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

    async def _run_extraction_task(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        qualitative_vocab_identifiers: list[str] | None,
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
        progress = ExtractionRunProgress(
            stage="file_ranking",
            total_chunks=sum(len(chunks) for chunks in chunks_by_file),
        )
        self._update_progress(data_package_id, progress)

        ranking = await self._rank_files(data_package_id, data_package, warnings)
        ordered_chunks = self._ordered_chunks(chunks_by_file, ranking)

        contexts: list[ExtractionContext] = []
        progress.stage = "chunk_extraction"
        for chunk in ordered_chunks:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EXTRACTION_CONTEXT_SYSTEM_PROMPT,
                prompt=build_extraction_context_prompt(
                    ChunkContext(
                        content=chunk.content,
                        start_idx=chunk.start_idx,
                        end_idx=chunk.end_idx,
                        file_path=chunk.file_path,
                        data_package_name=data_package.file_name,
                    )
                ),
                output_type=ExtractionContext,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="chunk_extraction",
                usage=result.usage,
            )
            contexts.append(result.output)
            partial_context = merge_extraction_context_results(contexts)
            self.output_repository.save_extraction_context(
                workflow_id=data_package_id,
                extraction_context=partial_context,
            )
            progress.processed_chunks += 1
            progress.interim_context = partial_context
            self._update_progress(data_package_id, progress)

        extraction_context = merge_extraction_context_results(contexts)
        self.output_repository.save_extraction_context(
            workflow_id=data_package_id,
            extraction_context=extraction_context,
        )

        progress.stage = "vocabulary_normalization"
        progress.interim_context = extraction_context
        self._update_progress(data_package_id, progress)
        normalization = await self._normalize_context(
            data_package_id=data_package_id,
            extraction_context=extraction_context,
            qualitative_vocab_identifiers=qualitative_vocab_identifiers,
            warnings=warnings,
        )
        progress.normalized_quantities = len(normalization.quantities)
        progress.normalized_qualitative_attributes = len(
            normalization.qualitative_attributes
        )
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)

        progress.stage = "profile_projection"
        progress.interim_context = extraction_context
        self._update_progress(data_package_id, progress)
        projection = await generate_structured(
            self.ollama_client,
            model=self.ollama_client.chat_model,
            system=PROFILE_PROJECTION_SYSTEM_PROMPT,
            prompt=build_profile_projection_prompt(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                profile_target_class=profile_manifest.target_class,
                extraction_context=extraction_context,
                normalization=normalization,
                warnings=warnings,
                profile_schema=validation_schema,
            ),
            output_type=validation_schema,
            num_ctx=self.ollama_client.max_context_length,
        )
        self._record_workflow_token_usage(
            data_package_id=data_package_id,
            agent_name="profile_projection",
            usage=projection.usage,
        )

        clean_document = remove_null_values(projection.output)
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=clean_document,
        )
        if not validation.valid:
            errors = [
                f"{issue.path}: {issue.message}"
                for issue in validation.errors
            ]
            raise ExtractionValidationError(errors)

        token_usage = await self.get_token_usage(data_package_id)
        result = ExtractionRunResult(
            document=clean_document,
            extraction_context=extraction_context,
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
        progress.stage = "completed"
        progress.interim_context = extraction_context
        progress.warnings = warnings
        self._update_progress(data_package_id, progress)
        return result

    async def _rank_files(
        self,
        data_package_id: str,
        data_package: Any,
        warnings: list[str],
    ) -> FileRankingResult:
        assert self.ollama_client is not None
        files = [
            FileContext(file_path=file.file_path, byte_size=len(file.raw_content))
            for file in data_package.files
        ]
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=FILE_RANKING_SYSTEM_PROMPT,
                prompt=build_file_ranking_prompt(files, data_package.file_name),
                output_type=FileRankingResult,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="file_ranking",
                usage=result.usage,
            )
            allowed_paths = {file.file_path for file in files}
            ranked = [
                item
                for item in result.output.files
                if item.file_path in allowed_paths
            ]
            missing = [
                file
                for file in files
                if file.file_path not in {item.file_path for item in ranked}
            ]
            fallback_tail = fallback_file_ranking(missing).files
            return FileRankingResult(files=ranked + fallback_tail)
        except CompletionError as exc:
            warnings.append(f"File ranking fell back to heuristics: {exc}")
            return fallback_file_ranking(files)

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

    async def _normalize_context(
        self,
        *,
        data_package_id: str,
        extraction_context: ExtractionContext,
        qualitative_vocab_identifiers: list[str] | None,
        warnings: list[str],
    ) -> ExtractionNormalization:
        quantities = self._all_quantities(extraction_context)
        qualitative_attributes = self._all_qualitative_attributes(extraction_context)

        quantity_results = await asyncio.gather(
            *[
                self._normalize_quantity(
                    data_package_id=data_package_id,
                    quantity=quantity,
                    warnings=warnings,
                )
                for quantity in quantities
            ]
        )
        qualitative_results = await asyncio.gather(
            *[
                self._normalize_qualitative_attribute(
                    data_package_id=data_package_id,
                    attribute=attribute,
                    vocab_identifiers=(
                        qualitative_vocab_identifiers
                        or DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS
                    ),
                    warnings=warnings,
                )
                for attribute in qualitative_attributes
            ]
        )
        return ExtractionNormalization(
            quantities=quantity_results,
            qualitative_attributes=qualitative_results,
        )

    async def _normalize_quantity(
        self,
        *,
        data_package_id: str,
        quantity: Quantity,
        warnings: list[str],
    ) -> QuantityNormalization:
        quantity_kind = await self._select_term_with_fallback(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
            source_value=quantity.quantity_kind,
            source_context=quantity.model_dump(mode="json"),
            query=build_quantity_kind_vocab_query(quantity),
            agent_name="quantity_vocab_selection",
            warnings=warnings,
        )
        unit = await self._select_term_with_fallback(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_UNIT_VOCAB,
            source_value=quantity.unit,
            source_context=quantity.model_dump(mode="json"),
            query=build_unit_vocab_query(quantity),
            agent_name="quantity_vocab_selection",
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

    async def _normalize_qualitative_attribute(
        self,
        *,
        data_package_id: str,
        attribute: QualitativeAttribute,
        vocab_identifiers: list[str],
        warnings: list[str],
    ) -> QualitativeAttributeNormalization:
        if self.semantic_service is None:
            warnings.append(
                f"Qualitative attribute '{attribute.title}' was not normalized because semantic service is unavailable."
            )
            return QualitativeAttributeNormalization(attribute=attribute)

        source_value = f"{attribute.title}: {attribute.value}".strip(": ")
        candidates: list[dict[str, Any]] = []
        for vocab_identifier in vocab_identifiers:
            try:
                vocab_info = await self.semantic_service.get_vocabulary(vocab_identifier)
            except Exception as exc:
                warnings.append(f"Vocabulary '{vocab_identifier}' unavailable: {exc}")
                continue
            if vocab_info is None:
                warnings.append(f"Vocabulary '{vocab_identifier}' is not registered.")
                continue
            for term_scheme in vocab_info.vocab_term_schemes:
                query = build_qualitative_vocab_query(
                    attribute,
                    rdf_type=term_scheme.rdf_type,
                )
                candidates.extend(
                    await self._query_candidate_records(
                        vocabulary_identifier=vocab_identifier,
                        query=query,
                        warnings=warnings,
                    )
                )

        mapping = await self._select_from_candidates(
            data_package_id=data_package_id,
            agent_name="qualitative_vocab_selection",
            source_value=source_value,
            source_context=attribute.model_dump(mode="json"),
            candidates=self._deduplicate_candidates(candidates),
            warnings=warnings,
        )
        if mapping is None:
            warnings.append(
                f"Qualitative attribute '{attribute.title}' kept raw value '{attribute.value}'."
            )
        return QualitativeAttributeNormalization(attribute=attribute, term=mapping)

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
        mapping = await self._select_from_candidates(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            candidates=candidates,
            warnings=warnings,
        )
        if mapping is not None:
            return mapping
        fallback = await self._build_fallback_query(
            data_package_id=data_package_id,
            agent_name=agent_name,
            source_value=source_value,
            source_context=source_context,
            failed_candidates=candidates,
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
        warnings: list[str],
    ) -> VocabularyTermMapping | None:
        if not candidates:
            return None
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

    async def _build_fallback_query(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        source_value: str,
        source_context: dict[str, Any],
        failed_candidates: list[dict[str, Any]],
        warnings: list[str],
    ) -> VocabularyFallbackQuery | None:
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
    def _all_quantities(context: ExtractionContext) -> list[Quantity]:
        quantities: list[Quantity] = []
        for item in (
            context.data_generating_activities
            + context.evaluated_entities
            + context.agentic_entities
            + context.datasets
        ):
            quantities.extend(item.has_quantitative_attributes)
        return quantities

    @staticmethod
    def _all_qualitative_attributes(context: ExtractionContext) -> list[QualitativeAttribute]:
        attributes: list[QualitativeAttribute] = []
        for item in (
            context.data_generating_activities
            + context.evaluated_entities
            + context.agentic_entities
            + context.datasets
        ):
            attributes.extend(item.has_qualitative_attributes)
        return attributes

    def _load_result_or_none(self, data_package_id: str) -> ExtractionRunResult | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_result(data_package_id)
        except FileNotFoundError:
            return None

    def _load_context_or_none(self, data_package_id: str) -> ExtractionContext | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_extraction_context(data_package_id)
        except FileNotFoundError:
            return None

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
            },
        )
        input_tokens = self._usage_int(usage, "input_tokens")
        output_tokens = self._usage_int(usage, "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens") or (
            input_tokens + output_tokens
        )
        requests = self._usage_int(usage, "requests")
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["total_tokens"] += total_tokens
        entry["requests"] += requests
        entry["operation_count"] += 1
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
                for key in ("input_tokens", "output_tokens", "total_tokens", "requests")
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
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "requests": int(values.get("requests", 0)),
            "operation_count": operation_count,
            "average_input_tokens_per_operation": round(input_tokens / operation_count, 2),
            "average_output_tokens_per_operation": round(output_tokens / operation_count, 2),
            "average_total_tokens_per_operation": round(total_tokens / operation_count, 2),
            "average_input_tokens_per_request": round(input_tokens / requests, 2),
            "average_output_tokens_per_request": round(output_tokens / requests, 2),
            "average_total_tokens_per_request": round(total_tokens / requests, 2),
        }

    @staticmethod
    def _extraction_task_name(data_package_id: str) -> str:
        return f"extraction:run:{data_package_id}"


def _resource_title(properties: dict[str, Any]) -> str | None:
    for key in (
        "label",
        "prefLabel",
        "preferred_label",
        "title",
        "name",
        "symbol",
        "ucumCode",
    ):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return None
