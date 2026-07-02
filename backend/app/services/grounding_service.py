from __future__ import annotations

from app.services.extraction_shared import *


async def generate_structured(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.generate_structured(*args, **kwargs)


class GroundingService:
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
        state.vocab_queries = []
        state.generated_final_draft = None
        self._save_run_state(data_package_id, state)
        return ExtractionRunProgress(
            stage="vocabulary_config_updated",
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
            initial_draft_scaffold=state.initial_draft_scaffold,
            projection_ledger=state.projection_ledger,
            field_completion_ledger=state.field_completion_ledger,
            curation_ledger=state.curation_ledger,
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
                group=(
                    "quantitative"
                    if record.kind in {"quantity_kind", "unit", "profile_has_quantity_type", "profile_unit"}
                    else "qualitative"
                ),
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

    async def run_grounding_stage(
        self,
        *,
        data_package_id: str,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> ExtractionRunResult:
        """Run the vocabulary grounding stage as a standalone step.

        Grounds the persisted reconstructed profile draft without rebuilding the projection or
        requirement-enrichment stages. Discovers DefinedTerm fields, runs
        vocabulary queries, normalizes, and persists the grounded
        generated_final_draft.
        """
        self._require_runtime_dependencies()
        assert self.output_repository is not None
        # Auto-detect chunking strategy from existing state if not specified
        effective_chunking = chunking_strategy
        if effective_chunking is None:
            for strategy in ("fixed_tokens", "semantic"):
                probe_state = self._load_run_state_or_none(
                    data_package_id,
                    chunking_strategy=strategy,
                    chat_model=chat_model,
                )
                if probe_state is not None and probe_state.generated_reconstructed_draft is not None:
                    effective_chunking = strategy
                    break
            if effective_chunking is None:
                effective_chunking = "semantic"
        state = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy=effective_chunking,
            chat_model=chat_model,
        )
        if state is None:
            raise ValueError(
                "Cannot run the grounding stage because the extraction run state is missing or unreadable. Build the profile draft first."
            )
        profile_identifier = state.profile_identifier
        if not profile_identifier:
            raise ValueError(
                "Cannot run the grounding stage because this extraction run has no profile identifier."
            )
        grounding_document = state.generated_reconstructed_draft
        if grounding_document is None:
            raise ValueError(
                "Cannot run the grounding stage because no profile draft is available. Build the profile draft first."
            )
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
        validation_schema = validation_schema_for_target_class(
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        evidence_context = self._merged_completed_evidence_context(state)
        warnings = self._load_warnings_or_empty(data_package_id)
        vocab_query_semaphore = asyncio.Semaphore(self._vocab_query_concurrency())

        def persist_vocab_progress() -> None:
            self._save_run_state(data_package_id, state)

        sources = self._profile_vocab_sources(
            grounding_document,
            enrichable_fields=getattr(profile_manifest, "enrichable_fields", []),
            validation_schema=validation_schema,
        )
        candidate_tasks = [
            asyncio.create_task(
                self._discover_profile_field_candidates(
                    json_path=json_path,
                    field_name=field_name,
                    source_value=source_value,
                    document=grounding_document,
                    state=state,
                    data_package_id=data_package_id,
                    query_semaphore=vocab_query_semaphore,
                    on_progress=persist_vocab_progress,
                    warnings=warnings,
                )
            )
            for json_path, field_name, source_value in sources
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
        self._save_run_state(data_package_id, state)
        result = await self._save_profile_result(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            evidence_context=evidence_context,
            normalization=normalization,
            document=grounding_document,
            profile_manifest=profile_manifest,
            validation_schema=validation_schema,
            state=state,
            warnings=warnings,
        )
        if self.task_registry is not None:
            self.task_registry.update_progress(
                self._extraction_task_name(data_package_id, state.chunking_strategy, state.chat_model),
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
                    initial_file_summary_progress=state.initial_file_summary_progress,
                    initial_file_summary_status=state.initial_file_summary_status,
                    initial_extraction_overview=state.initial_extraction_overview,
                    initial_extraction_overview_status=state.initial_extraction_overview_status,
                    initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    generated_final_draft=result.generated_final_draft,
                    curated_document=result.curated_document,
                    document_quality_state=result.document_quality_state,
                    draft_quality_state=result.draft_quality_state,
                    validation=result.validation,
                    curated_validation=result.curated_validation,
                    initial_draft_scaffold=state.initial_draft_scaffold,
                    projection_ledger=result.projection_ledger,
                    field_completion_ledger=result.field_completion_ledger,
                    evidence_query_ledger=state.evidence_query_ledger,
                    curation_ledger=result.curation_ledger,
                    warnings=warnings,
                ).model_dump(mode="json"),
            )
        return result

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
        profile_document = state.generated_reconstructed_draft or self._fallback_profile_document(
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
                self._extraction_task_name(data_package_id, state.chunking_strategy, state.chat_model),
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
                    initial_file_summary_progress=state.initial_file_summary_progress,
                    initial_file_summary_status=state.initial_file_summary_status,
                    initial_extraction_overview=state.initial_extraction_overview,
                    initial_extraction_overview_status=state.initial_extraction_overview_status,
                    initial_extraction_overview_diagnostic=state.initial_extraction_overview_diagnostic,
                    chunk_results=state.chunk_results,
                    vocab_queries=state.vocab_queries,
                    generated_final_draft=result.generated_final_draft,
                    curated_document=result.curated_document,
                    document_quality_state=result.document_quality_state,
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

    @staticmethod
    def _grounding_context_label_text(node: Any) -> str:
        if not isinstance(node, dict):
            return ""
        parts: list[str] = []
        for key in ("title", "description", "preferred_label", "label", "name"):
            value = node.get(key)
            if isinstance(value, list):
                value = next((entry for entry in value if isinstance(entry, str) and entry.strip()), None)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return " ".join(parts)

    @staticmethod
    def _grounding_resolve_json_pointer(document: Any, json_path: str) -> Any:
        if json_path in ("", "/"):
            return document
        node: Any = document
        for raw_seg in json_path.split("/"):
            if raw_seg == "":
                continue
            seg = raw_seg.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                try:
                    node = node[int(seg)]
                except (ValueError, IndexError):
                    return None
            elif isinstance(node, dict):
                node = node.get(seg)
            else:
                return None
            if node is None:
                return None
        return node

    @classmethod
    def _grounding_semantic_context(cls, document: dict[str, Any], json_path: str, *, include_root: bool = True, max_chars: int = 500) -> str:
        if not isinstance(document, dict) or not json_path:
            return ""
        segments = [seg for seg in json_path.split("/") if seg]
        if not segments:
            return ""
        prefixes = [""] if include_root else []
        current = ""
        for seg in segments[:-1]:
            current = current + "/" + seg
            prefixes.append(current)
        parts: list[str] = []
        for prefix in prefixes:
            text = cls._grounding_context_label_text(cls._grounding_resolve_json_pointer(document, prefix))
            if text and text not in parts:
                parts.append(text)
        context = " | ".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars].rstrip()
        return context

    async def _discover_profile_field_candidates(
        self,
        *,
        json_path: str,
        field_name: str,
        source_value: str,
        document: dict[str, Any],
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
        role = self._grounding_role_for_field(field_name)
        source_context["role"] = role
        semantic_context = self._grounding_semantic_context(document, json_path)
        brief_context = self._grounding_semantic_context(document, json_path, include_root=False, max_chars=160)
        source_context["semantic_context"] = semantic_context
        # Enrich source_context with full sibling quantitative attribute fields
        # so the LLM query formulation can see has_quantity_type, unit, value, identifier
        # when grounding any single field of a QuantitativeAttribute.
        if field_name in ("has_quantity_type", "unit"):
            parent_path = "/".join(json_path.split("/")[:-1])
            parent_node = self._grounding_resolve_json_pointer(document, parent_path)
            if isinstance(parent_node, dict):
                for sibling_key in ("has_quantity_type", "unit", "value", "identifier"):
                    if sibling_key != field_name:
                        sib_val = parent_node.get(sibling_key)
                        if sib_val is not None and str(sib_val).strip() and str(sib_val).strip() != "?":
                            source_context[f"sibling_{sibling_key}"] = str(sib_val)
        formulated_query = await self._formulate_vocab_query(
            data_package_id=data_package_id,
            agent_name="vocab_query_formulation",
            source_value=source_value,
            source_context=source_context,
            query_semaphore=query_semaphore,
            warnings=warnings,
        )
        if formulated_query:
            vector_query_text = formulated_query
            fulltext_query_text = formulated_query
        else:
            vector_query_text = " ".join(part for part in (semantic_context, source_value) if part)
            fulltext_query_text = " ".join(part for part in (brief_context, source_value) if part)
        if field_name == "has_quantity_type":
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type="qudt__QuantityKind",
                    vector_query=vector_query_text,
                    fulltext_query=fulltext_query_text,
                    vector_top_k=12,
                    fulltext_top_k=12,
                    seed_top_k=6,
                    max_hops=1,
                    traversal_direction="undirected",
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
                formulated_query=formulated_query,
                vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
                query_ids=query_ids,
                role=role,
                source_context=source_context,
            )
        if field_name == "unit":
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type="qudt__Unit",
                    vector_query=vector_query_text,
                    fulltext_query=fulltext_query_text,
                    vector_top_k=12,
                    fulltext_top_k=12,
                    seed_top_k=6,
                    max_hops=1,
                    traversal_direction="undirected",
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
                formulated_query=formulated_query,
                vocabulary_identifier=QUDT_UNIT_VOCAB,
                query_ids=query_ids,
                role=role,
                source_context=source_context,
            )

        policy = self._grounding_policy_for_role(state.vocab_query_config, role)
        vocabulary_identifiers = self._policy_vocabularies(policy, state.vocab_query_config)
        if policy is None or not policy.enabled or not policy.rdf_type or not vocabulary_identifiers:
            if field_name == "rdf_type":
                warnings.append(
                    f"Profile field '{json_path}' skipped because rdf_type grounding policy is incomplete."
                )
            return _ProfileFieldCandidateDiscovery(
                json_path=json_path,
                field_name=field_name,
                source_value=source_value,
                formulated_query=formulated_query,
                vocabulary_identifier="",
                query_ids=[],
                role=role,
                source_context=source_context,
            )
        for vocabulary_identifier in vocabulary_identifiers:
            query = self._configured_vocab_query(
                VocabQuery(
                    rdf_type=policy.rdf_type,
                    vector_query=vector_query_text,
                    fulltext_query=fulltext_query_text,
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
            formulated_query=formulated_query,
            vocabulary_identifier=",".join(vocabulary_identifiers),
            query_ids=query_ids,
            role=role,
            source_context=source_context,
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
            # Reconstruct source_context from the vocab query record so
            # the selection LLM gets sibling + semantic context on reruns too.
            record_context = dict(record.source_context) if record.source_context else {}
            discovery = groups.setdefault(
                key,
                _ProfileFieldCandidateDiscovery(
                    json_path=json_path,
                    field_name=field_name,
                    source_value=source_value,
                    vocabulary_identifier=record.vocabulary_identifier,
                    query_ids=[],
                    role=str(record.source_context.get("role", field_name)),
                    source_context=record_context,
                ),
            )
            discovery.query_ids.append(record.query_id)
        return await self._normalize_profile_field_discoveries(
            data_package_id=data_package_id,
            state=state,
            discoveries=list(groups.values()),
            warnings=warnings,
        )

    @staticmethod
    def _parent_attribute_path(json_path: str) -> str:
        """Return the parent QuantitativeAttribute path for a field.

        For ``/.../has_quantitative_attribute/3/has_quantity_type`` this
        returns ``/.../has_quantitative_attribute/3``.
        Returns empty string if the path doesn't match the expected pattern.
        """
        parts = json_path.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1] in ("has_quantity_type", "unit"):
            return "/".join(parts[:-1])
        return ""

    @staticmethod
    def _compatible_unit_uris_from_quantity_kind(
        state: ExtractionRunState,
        quantity_kind_uri: str,
    ) -> set[str] | None:
        """Find unit URIs linked to *quantity_kind_uri* via hasQuantityKind.

        Scans the graph_statements of all completed quantityKind vocab queries
        in *state* for ``qudt__hasQuantityKind`` predicates whose object is
        *quantity_kind_uri*.  Returns the set of subject URIs (units), or
        ``None`` if no statements were found (caller should skip filtering).
        """
        compatible: set[str] = set()
        for record in state.vocab_queries:
            if record.kind != "profile_has_quantity_type":
                continue
            if not record.result or not record.result.graph_statements:
                continue
            for stmt in record.result.graph_statements:
                if stmt.predicate == "qudt__hasQuantityKind" and stmt.object_uri == quantity_kind_uri:
                    compatible.add(stmt.subject_uri)
        return compatible if compatible else None

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

        candidates_by_path: dict[str, list[dict[str, Any]]] = {}

        def _sort_key(d: _ProfileFieldCandidateDiscovery) -> tuple[int, str]:
            if d.field_name == "has_quantity_type":
                return (0, d.json_path)
            if d.field_name == "unit":
                return (1, d.json_path)
            return (2, d.json_path)

        ordered = sorted(discoveries, key=_sort_key)

        for discovery in ordered:
            candidates: list[dict[str, Any]] = []
            for query_id in discovery.query_ids:
                record = self._find_vocab_query_record(state, query_id)
                if record and record.result:
                    candidates.extend(self._candidate_records(record.result))
            candidates = self._deduplicate_candidates(candidates)
            candidates_by_path[discovery.json_path] = candidates

            selection_source_value = (
                discovery.formulated_query or discovery.source_value
            )
            selection_context = dict(discovery.source_context)
            selection_context.setdefault("json_path", discovery.json_path)
            selection_context.setdefault("field_name", discovery.field_name)
            mapping = await self._select_from_candidates_with_semaphore(
                data_package_id=data_package_id,
                agent_name="profile_field_vocab_selection",
                source_value=selection_source_value,
                source_context=selection_context,
                candidates=candidates,
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
        profile_fields = await self._resolve_profile_quantity_field_pairs(
            data_package_id=data_package_id,
            state=state,
            profile_fields=profile_fields,
            candidates_by_path=candidates_by_path,
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        return ExtractionNormalization(profile_fields=profile_fields)

    async def _resolve_profile_quantity_field_pairs(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        profile_fields: list[ProfileFieldNormalization],
        candidates_by_path: dict[str, list[dict[str, Any]]],
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> list[ProfileFieldNormalization]:
        by_parent: dict[str, dict[str, int]] = {}
        for index, item in enumerate(profile_fields):
            if item.field_name not in {"has_quantity_type", "unit"}:
                continue
            parent = self._parent_attribute_path(item.json_path)
            if not parent:
                continue
            by_parent.setdefault(parent, {})[item.field_name] = index

        resolved = list(profile_fields)
        for parent_path, indices in by_parent.items():
            kind_index = indices.get("has_quantity_type")
            unit_index = indices.get("unit")
            if kind_index is None or unit_index is None:
                continue
            kind_field = resolved[kind_index]
            unit_field = resolved[unit_index]
            kind_uri = kind_field.term.selected_uri if kind_field.term else None
            unit_uri = unit_field.term.selected_uri if unit_field.term else None
            if not kind_uri or not unit_uri:
                continue
            if self._qudt_quantity_pair_status(
                state=state,
                quantity_kind_uri=kind_uri,
                unit_uri=unit_uri,
            ) == "compatible":
                continue

            kind_candidates = self._qudt_candidates_for_role(
                candidates_by_path.get(kind_field.json_path, []),
                "quantity_kind",
            )
            unit_candidates = self._qudt_candidates_for_role(
                candidates_by_path.get(unit_field.json_path, []),
                "unit",
            )
            selection = await self._select_profile_quantity_pair(
                data_package_id=data_package_id,
                state=state,
                parent_path=parent_path,
                quantity_kind_field=kind_field,
                unit_field=unit_field,
                quantity_kind_candidates=kind_candidates,
                unit_candidates=unit_candidates,
                selection_semaphore=selection_semaphore,
                warnings=warnings,
            )
            reason = (
                f"QUDT quantity-kind/unit pair for '{parent_path}' was not cross-validated "
                f"via qudt:hasQuantityKind: {kind_uri} with {unit_uri}."
            )
            if selection is None:
                resolved[kind_index] = self._unselected_profile_field(kind_field, reason)
                resolved[unit_index] = self._unselected_profile_field(unit_field, reason)
                warnings.append(f"{reason} Rejected both fields.")
                continue

            selected_kind = selection.selected_quantity_kind_uri
            selected_unit = selection.selected_unit_uri
            if not selected_kind or not selected_unit:
                resolved[kind_index] = self._unselected_profile_field(kind_field, selection.reason or reason)
                resolved[unit_index] = self._unselected_profile_field(unit_field, selection.reason or reason)
                warnings.append(f"{reason} Pair resolver rejected both fields.")
                continue
            if self._qudt_quantity_pair_status(
                state=state,
                quantity_kind_uri=selected_kind,
                unit_uri=selected_unit,
            ) != "compatible":
                resolved[kind_index] = self._unselected_profile_field(kind_field, selection.reason or reason)
                resolved[unit_index] = self._unselected_profile_field(unit_field, selection.reason or reason)
                warnings.append(f"{reason} Pair resolver returned a non-compatible pair; rejected both fields.")
                continue

            kind_candidate = self._candidate_by_uri(kind_candidates, selected_kind)
            unit_candidate = self._candidate_by_uri(unit_candidates, selected_unit)
            if kind_candidate is None or unit_candidate is None:
                resolved[kind_index] = self._unselected_profile_field(kind_field, selection.reason or reason)
                resolved[unit_index] = self._unselected_profile_field(unit_field, selection.reason or reason)
                warnings.append(f"{reason} Pair resolver returned a URI outside the candidate sets; rejected both fields.")
                continue

            resolved[kind_index] = kind_field.model_copy(
                update={
                    "term": self._mapping_from_profile_candidate(
                        field=kind_field,
                        candidate=kind_candidate,
                        confidence=selection.confidence,
                        reason=selection.reason,
                    )
                }
            )
            resolved[unit_index] = unit_field.model_copy(
                update={
                    "term": self._mapping_from_profile_candidate(
                        field=unit_field,
                        candidate=unit_candidate,
                        confidence=selection.confidence,
                        reason=selection.reason,
                    )
                }
            )
            warnings.append(f"{reason} Pair resolver selected a compatible replacement pair.")
        return resolved

    async def _select_profile_quantity_pair(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        parent_path: str,
        quantity_kind_field: ProfileFieldNormalization,
        unit_field: ProfileFieldNormalization,
        quantity_kind_candidates: list[dict[str, Any]],
        unit_candidates: list[dict[str, Any]],
        selection_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> VocabularyQuantityPairSelection | None:
        if not quantity_kind_candidates or not unit_candidates:
            return None
        async with selection_semaphore:
            assert self.ollama_client is not None
            compatible_pairs = self._compatible_qudt_candidate_pairs(
                state=state,
                quantity_kind_candidates=quantity_kind_candidates,
                unit_candidates=unit_candidates,
            )
            prompt_components = [
                (
                    "pair_selection_instruction",
                    "Select a compatible QUDT QuantityKind and Unit pair for one quantitative attribute.\n"
                    "Use only the candidate URIs in the prompt.\n"
                    "The selected pair must appear in compatible_pairs.\n"
                    "If no compatible pair fits the source attribute, return null for both URIs.\n",
                ),
                (
                    "source_fields",
                    "Source fields JSON:\n"
                    + json.dumps(
                        {
                            "parent_path": parent_path,
                            "quantity_kind": quantity_kind_field.model_dump(mode="json"),
                            "unit": unit_field.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n",
                ),
                (
                    "quantity_kind_candidates",
                    "QuantityKind candidates JSON:\n"
                    + json.dumps(quantity_kind_candidates, ensure_ascii=False)
                    + "\n\n",
                ),
                (
                    "unit_candidates",
                    "Unit candidates JSON:\n"
                    + json.dumps(unit_candidates, ensure_ascii=False)
                    + "\n\n",
                ),
                (
                    "compatible_pairs",
                    "Compatible candidate pairs JSON:\n"
                    + json.dumps(compatible_pairs, ensure_ascii=False)
                    + "\n\n",
                ),
            ]
            try:
                result = await generate_structured(
                    self.ollama_client,
                    model=self.ollama_client.chat_model,
                    system=(
                        "You repair incompatible QUDT quantity grounding pairs. "
                        "Return one compatible QuantityKind and Unit pair, or nulls for both."
                    ),
                    prompt="".join(text for _, text in prompt_components),
                    system_components=[
                        (
                            "profile_quantity_pair_selection_system_prompt",
                            "You repair incompatible QUDT quantity grounding pairs. Return only JSON.",
                        ),
                    ],
                    prompt_components=prompt_components,
                    token_budgeter=self._prompt_token_budgeter(),
                    operation_id=self._prompt_operation_id(
                        "profile_quantity_pair_selection",
                        parent_path,
                    ),
                    agent_name="profile_quantity_pair_selection",
                    diagnostic_metadata={"parent_path": parent_path},
                    output_type=VocabularyQuantityPairSelection,
                    num_ctx=self.ollama_client.max_context_length,
                )
            except CompletionError as exc:
                self._record_llm_call_exception(
                    data_package_id=data_package_id,
                    exc=exc,
                    agent_name="profile_quantity_pair_selection",
                )
                warnings.append(f"QUDT pair resolver failed for '{parent_path}': {exc}")
                return None
        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=result,
            agent_name="profile_quantity_pair_selection",
        )
        return (
            result.output
            if isinstance(result.output, VocabularyQuantityPairSelection)
            else VocabularyQuantityPairSelection.model_validate(result.output)
        )

    @staticmethod
    def _qudt_candidates_for_role(
        candidates: list[dict[str, Any]],
        role: Literal["quantity_kind", "unit"],
    ) -> list[dict[str, Any]]:
        prefix = (
            "http://qudt.org/vocab/quantitykind/"
            if role == "quantity_kind"
            else "http://qudt.org/vocab/unit/"
        )
        return [candidate for candidate in candidates if str(candidate.get("uri", "")).startswith(prefix)]

    @staticmethod
    def _candidate_by_uri(
        candidates: list[dict[str, Any]],
        uri: str,
    ) -> dict[str, Any] | None:
        return next((candidate for candidate in candidates if candidate.get("uri") == uri), None)

    @staticmethod
    def _mapping_from_profile_candidate(
        *,
        field: ProfileFieldNormalization,
        candidate: dict[str, Any],
        confidence: float,
        reason: str,
    ) -> VocabularyTermMapping:
        return VocabularyTermMapping(
            source_value=field.source_value,
            vocabulary_identifier=candidate.get("vocabulary_identifier"),
            rdf_type=candidate.get("rdf_type"),
            selected_uri=candidate.get("uri"),
            selected_title=candidate.get("title"),
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _unselected_profile_field(
        field: ProfileFieldNormalization,
        reason: str,
    ) -> ProfileFieldNormalization:
        term = field.term or VocabularyTermMapping(source_value=field.source_value)
        return field.model_copy(
            update={
                "term": term.model_copy(
                    update={
                        "selected_uri": None,
                        "selected_title": None,
                        "reason": reason,
                    }
                )
            }
        )

    @staticmethod
    def _qudt_quantity_pair_status(
        *,
        state: ExtractionRunState,
        quantity_kind_uri: str,
        unit_uri: str,
    ) -> Literal["compatible", "incompatible", "unknown"]:
        compatible_kinds_for_unit: set[str] = set()
        compatible_units_for_kind: set[str] = set()
        for record in state.vocab_queries:
            if not record.result or not record.result.graph_statements:
                continue
            for stmt in record.result.graph_statements:
                if stmt.predicate != "qudt__hasQuantityKind":
                    continue
                if stmt.subject_uri == unit_uri:
                    compatible_kinds_for_unit.add(stmt.object_uri)
                if stmt.object_uri == quantity_kind_uri:
                    compatible_units_for_kind.add(stmt.subject_uri)
        if quantity_kind_uri in compatible_kinds_for_unit or unit_uri in compatible_units_for_kind:
            return "compatible"
        if compatible_kinds_for_unit or compatible_units_for_kind:
            return "incompatible"
        return "unknown"

    @classmethod
    def _compatible_qudt_candidate_pairs(
        cls,
        *,
        state: ExtractionRunState,
        quantity_kind_candidates: list[dict[str, Any]],
        unit_candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        quantity_kind_uris = {str(candidate.get("uri", "")) for candidate in quantity_kind_candidates}
        unit_uris = {str(candidate.get("uri", "")) for candidate in unit_candidates}
        pairs: list[dict[str, str]] = []
        for record in state.vocab_queries:
            if not record.result or not record.result.graph_statements:
                continue
            for stmt in record.result.graph_statements:
                if stmt.predicate != "qudt__hasQuantityKind":
                    continue
                if stmt.subject_uri in unit_uris and stmt.object_uri in quantity_kind_uris:
                    pairs.append(
                        {
                            "quantity_kind_uri": stmt.object_uri,
                            "unit_uri": stmt.subject_uri,
                        }
                    )
        return cls._deduplicate_compatible_pairs(pairs)

    @staticmethod
    def _deduplicate_compatible_pairs(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, str]] = []
        for pair in pairs:
            key = (pair["quantity_kind_uri"], pair["unit_uri"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(pair)
        return deduped

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

        # Step 1: Select BOTH unit and quantity_kind independently
        all_unit_candidates = self._candidate_records(unit_record.result) if unit_record and unit_record.result else []
        all_kind_candidates = self._candidate_records(kind_record.result) if kind_record and kind_record.result else []
        unit_unconstrained = await self._select_term_with_fallback_candidates(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_UNIT_VOCAB,
            source_value=quantity.unit,
            source_context=quantity.model_dump(mode="json"),
            query=unit_record.query if unit_record else build_unit_vocab_query(quantity),
            initial_candidates=all_unit_candidates,
            agent_name="quantity_vocab_selection",
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        quantity_kind_unconstrained = await self._select_term_with_fallback_candidates(
            data_package_id=data_package_id,
            vocabulary_identifier=QUDT_QUANTITY_KIND_VOCAB,
            source_value=quantity.quantity_kind,
            source_context=quantity.model_dump(mode="json"),
            query=kind_record.query if kind_record else build_quantity_kind_vocab_query(quantity),
            initial_candidates=all_kind_candidates,
            agent_name="quantity_vocab_selection",
            selection_semaphore=selection_semaphore,
            warnings=warnings,
        )
        # Step 2: Cross-constrain both directions via qudt__hasQuantityKind
        compatible_kind_uris: set[str] = set()
        if unit_unconstrained is not None and unit_unconstrained.selected_uri:
            if unit_record and unit_record.result:
                for stmt in unit_record.result.graph_statements:
                    if stmt.subject_uri == unit_unconstrained.selected_uri and "hasQuantityKind" in stmt.predicate:
                        compatible_kind_uris.add(stmt.object_uri)
        compatible_unit_uris: set[str] = set()
        if quantity_kind_unconstrained is not None and quantity_kind_unconstrained.selected_uri:
            if kind_record and kind_record.result:
                for stmt in kind_record.result.graph_statements:
                    if stmt.object_uri == quantity_kind_unconstrained.selected_uri and "hasQuantityKind" in stmt.predicate:
                        compatible_unit_uris.add(stmt.subject_uri)
        # Step 3: Re-select with constraints if incompatible
        quantity_kind = quantity_kind_unconstrained
        unit = unit_unconstrained
        if compatible_kind_uris:
            filtered_kind_candidates = [c for c in all_kind_candidates if c.get("uri") in compatible_kind_uris]
            if filtered_kind_candidates:
                if not (quantity_kind_unconstrained and quantity_kind_unconstrained.selected_uri in compatible_kind_uris):
                    quantity_kind = await self._select_from_candidates(
                        data_package_id=data_package_id,
                        agent_name="quantity_vocab_selection",
                        source_value=quantity.quantity_kind,
                        source_context={**quantity.model_dump(mode="json"), "unit_constraint": "Unit restricts to compatible kinds only"},
                        candidates=filtered_kind_candidates,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
            else:
                quantity_kind = None
        if compatible_unit_uris:
            filtered_unit_candidates = [c for c in all_unit_candidates if c.get("uri") in compatible_unit_uris]
            if filtered_unit_candidates:
                if not (unit_unconstrained and unit_unconstrained.selected_uri in compatible_unit_uris):
                    unit = await self._select_from_candidates(
                        data_package_id=data_package_id,
                        agent_name="quantity_vocab_selection",
                        source_value=quantity.unit,
                        source_context={**quantity.model_dump(mode="json"), "kind_constraint": "QuantityKind restricts to compatible units only"},
                        candidates=filtered_unit_candidates,
                        selection_semaphore=selection_semaphore,
                        warnings=warnings,
                    )
        # Step 4: Cross-validation
        if quantity_kind is not None and quantity_kind.selected_uri and unit is not None and unit.selected_uri and compatible_kind_uris and quantity_kind.selected_uri not in compatible_kind_uris:
            warnings.append("Quantity " + quantity.identifier + ": CROSS-VALIDATION MISMATCH")
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
            prompt_components = build_object_grounding_selection_prompt_components(
                object_identifier=object_identifier,
                object_kind=object_kind,
                raw_type=raw_type,
                source_context=source_context,
                candidates=candidates,
            )
            prompt_budgeter = self._prompt_token_budgeter()
            try:
                result = await generate_structured(
                    self.ollama_client,
                    model=self.ollama_client.chat_model,
                    system=VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
                    prompt="".join(text for _, text in prompt_components),
                    system_components=[
                        (
                            "object_grounding_vocab_selection_system_prompt",
                            VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
                        ),
                    ],
                    prompt_components=prompt_components,
                    token_budgeter=prompt_budgeter,
                    operation_id=self._prompt_operation_id(
                        "object_grounding_vocab_selection",
                        object_identifier,
                    ),
                    agent_name="object_grounding_vocab_selection",
                    diagnostic_metadata={
                        "object_identifier": object_identifier,
                        "object_kind": object_kind,
                        "raw_type": raw_type,
                    },
                    output_type=VocabularyCandidateSelection,
                    num_ctx=self.ollama_client.max_context_length,
                )
            except CompletionError as exc:
                self._record_llm_call_exception(
                    data_package_id=data_package_id,
                    exc=exc,
                    agent_name="object_grounding_vocab_selection",
                )
                raise
        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=result,
            agent_name="object_grounding_vocab_selection",
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
                repr(sorted((key, value) for key, value in source_context.items() if key not in {"semantic_context", "formulated_query"})),
                vocabulary_identifier,
                rdf_type,
            )
        )
        return sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _candidate_records(result: VocabQueryResult) -> list[dict[str, Any]]:
        """Build slim candidate dicts from a VocabQueryResult.

        Filters out blank nodes (genid URIs) that appear in ``resources``
        as intermediate graph-traversal artifacts (e.g. ``qudt__FactorUnit``
        blank nodes from ``qudt__hasQuantityKind`` edge expansion).  These
        carry no label or useful properties and only pollute the candidate
        list presented to the selection LLM.
        Cross-vocabulary resources (e.g. QuantityKind URIs discovered while
        querying for Units) are kept — they may be valid matches.

        Only essential fields are included (uri, label, description, symbol)
        to keep the candidate payload small enough for the selection LLM to
        produce valid structured output.  Full property dicts (with
        conversionMultiplier, ucumCode, etc.) are omitted — the LLM only needs
        label + definition per the system prompt.
        """
        records: list[dict[str, Any]] = []
        for uri, resource in result.resources.items():
            # Skip blank nodes generated by graph traversal expansion.
            # They use synthetic URIs like http://example.org/.well-known/genid/<hex>
            # and have no label or meaningful properties.
            if "/.well-known/genid/" in uri:
                continue
            props = resource.properties
            slim: dict[str, Any] = {
                "uri": uri,
                "vocabulary_identifier": result.identifier,
                "rdf_type": result.rdf_type,
                # Use 'title' key for backward compat with selection code
                # (which reads candidate.get("title"))
                "title": _resource_title(props),
            }
            # Add description if available (dcterms__description or rdfs__comment)
            desc = props.get("dcterms__description") or props.get("rdfs__comment")
            if desc:
                slim["description"] = str(desc)[:300]
            # Add symbol if available (useful for unit matching)
            symbol = props.get("qudt__symbol") or props.get("qudt__abbreviation")
            if symbol:
                slim["symbol"] = str(symbol)
            records.append(slim)
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
            prompt_components = build_candidate_selection_prompt_components(
                source_value=source_value,
                source_context=source_context,
                candidates=candidates,
            )
            prompt_budgeter = self._prompt_token_budgeter()
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
                prompt="".join(text for _, text in prompt_components),
                system_components=[
                    ("vocab_candidate_selection_system_prompt", VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT),
                ],
                prompt_components=prompt_components,
                token_budgeter=prompt_budgeter,
                operation_id=self._prompt_operation_id(
                    agent_name,
                    source_value,
                ),
                agent_name=agent_name,
                diagnostic_metadata={
                    "source_value": source_value,
                    "candidate_count": len(candidates),
                },
                output_type=VocabularyCandidateSelection,
                num_ctx=self.ollama_client.max_context_length,
            )
        except CompletionError as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name=agent_name,
            )
            warnings.append(
                f"Vocabulary selector left '{source_value}' unresolved after model failure: {exc}"
            )
            return None
        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=result,
            agent_name=agent_name,
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

    async def _formulate_vocab_query(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        source_value: str,
        source_context: dict[str, Any],
        query_semaphore: asyncio.Semaphore,
        warnings: list[str],
    ) -> str:
        """Distill source_value + semantic context into a concise vocabulary search phrase.

        Returns the phrase, or an empty string when formulation fails so callers fall back
        to the context-based query text.
        """
        if self.ollama_client is None:
            return ""
        async with query_semaphore:
            try:
                prompt_components = build_query_formulation_prompt_components(
                    source_value=source_value,
                    source_context=source_context,
                )
                prompt_budgeter = self._prompt_token_budgeter()
                result = await generate_structured(
                    self.ollama_client,
                    model=self.ollama_client.chat_model,
                    system=VOCAB_QUERY_FORMULATION_SYSTEM_PROMPT,
                    prompt="".join(text for _, text in prompt_components),
                    system_components=[
                        ("vocab_query_formulation_system_prompt", VOCAB_QUERY_FORMULATION_SYSTEM_PROMPT),
                    ],
                    prompt_components=prompt_components,
                    token_budgeter=prompt_budgeter,
                    operation_id=self._prompt_operation_id(
                        agent_name,
                        "query_formulation",
                        source_value,
                    ),
                    agent_name=agent_name,
                    diagnostic_metadata={"source_value": source_value},
                    output_type=VocabularyQueryFormulation,
                    num_ctx=self.ollama_client.max_context_length,
                )
            except CompletionError as exc:
                self._record_llm_call_exception(
                    data_package_id=data_package_id,
                    exc=exc,
                    agent_name=agent_name,
                )
                warnings.append(f"Vocabulary query formulation failed for '{source_value}': {exc}")
                return ""
        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=result,
            agent_name=agent_name,
        )
        return (result.output.query or "").strip()


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
            prompt_components = build_fallback_query_prompt_components(
                source_value=source_value,
                source_context=source_context,
                failed_candidates=failed_candidates,
            )
            prompt_budgeter = self._prompt_token_budgeter()
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
                prompt="".join(text for _, text in prompt_components),
                system_components=[
                    ("vocab_fallback_query_system_prompt", VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT),
                ],
                prompt_components=prompt_components,
                token_budgeter=prompt_budgeter,
                operation_id=self._prompt_operation_id(
                    agent_name,
                    "fallback_query",
                    source_value,
                ),
                agent_name=agent_name,
                diagnostic_metadata={
                    "source_value": source_value,
                    "failed_candidate_count": len(failed_candidates),
                },
                output_type=VocabularyFallbackQuery,
                num_ctx=self.ollama_client.max_context_length,
            )
        except CompletionError as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name=agent_name,
            )
            warnings.append(f"Fallback vocabulary query generation failed: {exc}")
            return None
        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=result,
            agent_name=agent_name,
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
    def _grounding_role_for_field(field_name: str) -> str:
        if field_name in {"type", "rdf_type", "has_quantity_type", "unit"}:
            return field_name
        return field_name

    @staticmethod
    def _grounding_policy_for_role(
        config: ExtractionVocabQueryConfig,
        role: str,
    ) -> GroundingRolePolicy | None:
        if role == "type":
            return config.type_policy
        if role == "rdf_type":
            return config.rdf_type_policy
        return None

    @staticmethod
    def _policy_vocabularies(
        policy: GroundingRolePolicy | None,
        config: ExtractionVocabQueryConfig,
    ) -> list[str]:
        if policy is None:
            return []
        values = policy.vocabulary_identifiers or (
            config.qualitative_vocab_identifiers
            if policy is config.type_policy
            else []
        )
        if policy is config.type_policy and not values:
            values = DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS
        return [value for value in values if value.strip()]

    @staticmethod
    def _default_vocab_query_config(
        qualitative_vocab_identifiers: list[str] | None,
    ) -> ExtractionVocabQueryConfig:
        qualitative = qualitative_vocab_identifiers or DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS
        return ExtractionVocabQueryConfig(
            qualitative_vocab_identifiers=qualitative,
            type_policy=GroundingRolePolicy(
                vocabulary_identifiers=qualitative,
                rdf_type="skos__Concept",
                enabled=True,
            ),
            rdf_type_policy=GroundingRolePolicy(
                vocabulary_identifiers=[],
                rdf_type="",
                enabled=False,
            ),
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

