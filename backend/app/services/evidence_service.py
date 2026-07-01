from __future__ import annotations

from app.services.extraction_shared import *


async def generate_structured(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.generate_structured(*args, **kwargs)


async def repair_structured_output(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.repair_structured_output(*args, **kwargs)


class EvidenceService:
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
        chunking_strategy: str = "semantic",
        chunk_repair_mode: ChunkRepairMode = "deferred",
        evidence_critic_granularity: EvidenceCriticGranularity = "per_chunk",
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
            chunking_strategy=chunking_strategy,
            chunk_repair_mode=chunk_repair_mode,
            evidence_critic_granularity=evidence_critic_granularity,
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
            initial_file_summary_progress=(
                persisted_state.initial_file_summary_progress
                if preserve_initial_context and persisted_state
                else None
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
            initial_extraction_overview_diagnostic=(
                persisted_state.initial_extraction_overview_diagnostic
                if preserve_initial_context and persisted_state
                else None
            ),
            dataset_summary=(
                persisted_state.dataset_summary
                if preserve_initial_context and persisted_state
                else ""
            ),
            chunk_results=chunk_results,
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
            filtered_evidence_notes=(
                persisted_state.filtered_evidence_notes
                if persisted_state
                else []
            ),
        )

    @staticmethod
    def _clear_profile_projection_state(state: ExtractionRunState) -> None:
        state.generated_initial_draft = None
        state.generated_final_draft = None
        state.generated_core_draft = None
        state.generated_attribute_draft = None
        state.generated_reconstructed_draft = None
        state.requirement_report = None
        state.curated_document = None
        state.document_quality_state = None
        state.draft_quality_state = None
        state.validation = DraftValidationResult()
        state.curated_validation = None
        state.projection_ledger = []
        state.initial_draft_scaffold = {}
        state.field_completion_ledger = []
        state.evidence_query_ledger = []
        state.curation_ledger = []
        state.vocab_queries = []

    @staticmethod
    def _clear_profile_projection_progress(progress: ExtractionRunProgress) -> None:
        progress.generated_initial_draft = None
        progress.generated_final_draft = None
        progress.generated_core_draft = None
        progress.generated_attribute_draft = None
        progress.generated_reconstructed_draft = None
        progress.requirement_report = None
        progress.curated_document = None
        progress.document_quality_state = None
        progress.draft_quality_state = None
        progress.validation = DraftValidationResult()
        progress.curated_validation = None
        progress.projection_ledger = []
        progress.initial_draft_scaffold = {}
        progress.field_completion_ledger = []
        progress.evidence_query_ledger = []
        progress.curation_ledger = []
        progress.vocab_queries = []

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
    ) -> list[RoutedEvidenceContext]:
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
    ) -> RoutedEvidenceContext:
        return merge_evidence_contexts(
            cls._completed_chunk_evidence_contexts(state, file_path=file_path)
        )

    async def _filtered_completed_evidence_context(
        self,
        state: ExtractionRunState,
        *,
        data_package_id: str = "",
        file_path: str | None = None,
        similarity_dedup: bool = False,
    ) -> tuple[RoutedEvidenceContext, list[FilteredEvidenceNote]]:
        context = self._merged_completed_evidence_context(state, file_path=file_path)
        rank_by_path = {ranked.file_path: ranked.rank for ranked in state.ranked_files}
        # Phase 1: exact-text dedup (fast, no LLM)
        context, exact_dups = dedupe_repeated_evidence_notes(
            context,
            file_rank_by_path=rank_by_path,
        )
        # Phase 2: similarity-based dedup (embedding + LLM triage)
        # Only run when explicitly requested (end of evidence extraction,
        # not after every chunk).
        if similarity_dedup:
            context, sim_dups = await self._dedupe_by_similarity(
                context,
                file_rank_by_path=rank_by_path,
                data_package_id=data_package_id,
            )
        else:
            sim_dups = []
        return context, [*exact_dups, *sim_dups]

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
            chunking_strategy=state.chunking_strategy,
            chat_model=state.chat_model,
        )

    async def _validate_assess_and_route_evidence_context_for_chunk(
        self,
        context: EvidenceContext,
        *,
        chunk_content: str,
        chunk_context: EvidenceChunkContext,
        chunk_result: ExtractionChunkResult,
        state: ExtractionRunState,
        critic_granularity: EvidenceCriticGranularity,
        data_package_id: str,
    ) -> RoutedEvidenceContext:
        validated_context, dropped_candidates = validate_evidence_candidates(
            context,
            chunk_content=chunk_content,
            file_path=chunk_result.file_path,
            start_idx=chunk_result.start_idx,
            end_idx=chunk_result.end_idx,
        )
        state.filtered_evidence_notes.extend(
            FilteredEvidenceNote(
                reason="evidence_text_unsupported",
                note=dropped_candidate,
                file_path=dropped_candidate.file_path,
                start_idx=dropped_candidate.start_idx,
                end_idx=dropped_candidate.end_idx,
                chunk_index=chunk_result.chunk_index,
            )
            for dropped_candidate in dropped_candidates
        )
        assessments = await self._assess_evidence_candidates(
            data_package_id=data_package_id,
            candidates=validated_context.candidates,
            chunk_context=chunk_context,
            critic_granularity=critic_granularity,
            chunk_result=chunk_result,
        )
        routed_context = route_evidence_candidates(
            validated_context,
            assessments=assessments,
            rejected_candidates=dropped_candidates,
            chunk_index=chunk_result.chunk_index,
        )
        state.filtered_evidence_notes.extend(
            FilteredEvidenceNote(
                reason="candidate_rejected",
                note=record.candidate,
                file_path=record.candidate.file_path,
                start_idx=record.candidate.start_idx,
                end_idx=record.candidate.end_idx,
                chunk_index=chunk_result.chunk_index,
            )
            for record in routed_context.rejected_evidence
            if record.reason != "evidence_text_unsupported"
        )
        return routed_context

    async def _assess_evidence_candidates(
        self,
        *,
        data_package_id: str,
        candidates: list[EvidenceCandidate],
        chunk_context: EvidenceChunkContext,
        critic_granularity: EvidenceCriticGranularity,
        chunk_result: ExtractionChunkResult,
    ) -> list[EvidenceAssessment]:
        if not candidates:
            return []
        if critic_granularity == "disabled":
            return [
                EvidenceAssessment(
                    candidate_id=candidate.candidate_id,
                    rationale="Evidence critic disabled; routed conservatively.",
                )
                for candidate in candidates
            ]
        if critic_granularity == "per_candidate":
            assessments: list[EvidenceAssessment] = []
            for candidate in candidates:
                assessments.extend(
                    await self._run_evidence_critic(
                        data_package_id=data_package_id,
                        candidates=[candidate],
                        chunk_context=chunk_context,
                        chunk_result=chunk_result,
                    )
                )
            return assessments
        return await self._run_evidence_critic(
            data_package_id=data_package_id,
            candidates=candidates,
            chunk_context=chunk_context,
            chunk_result=chunk_result,
        )

    async def _run_evidence_critic(
        self,
        *,
        data_package_id: str,
        candidates: list[EvidenceCandidate],
        chunk_context: EvidenceChunkContext,
        chunk_result: ExtractionChunkResult,
    ) -> list[EvidenceAssessment]:
        assert self.ollama_client is not None
        prompt_components = build_evidence_critic_prompt_components(
            candidates=candidates,
            chunk_context=chunk_context,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EVIDENCE_CRITIC_SYSTEM_PROMPT,
                prompt="".join(text for _, text in prompt_components),
                system_components=[("evidence_critic_system_prompt", EVIDENCE_CRITIC_SYSTEM_PROMPT)],
                prompt_components=prompt_components,
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id(
                    "evidence_critic",
                    chunk_result.file_path,
                    chunk_result.chunk_index,
                    str(len(candidates)),
                ),
                agent_name="evidence_critic",
                diagnostic_metadata={
                    "file_path": chunk_result.file_path,
                    "chunk_index": chunk_result.chunk_index,
                    "candidate_count": len(candidates),
                },
                output_type=EvidenceAssessmentContext,
                retries=1,
                think=None,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="evidence_critic",
            )
            return result.output.assessments
        except Exception as exc:
            if isinstance(exc, CompletionError):
                self._record_llm_call_exception(
                    data_package_id=data_package_id,
                    exc=exc,
                    agent_name="evidence_critic",
                )
            return [
                EvidenceAssessment(
                    candidate_id=candidate.candidate_id,
                    rationale=f"Evidence critic failed; routed conservatively: {exc}",
                )
                for candidate in candidates
            ]

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

        repair_prompt_budgeter = self._prompt_token_budgeter()
        try:
            repair = await repair_structured_output(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                failed_response=failure.failed_response or "",
                error=failure.last_error or failure,
                output_type=EvidenceContext,
                token_budgeter=repair_prompt_budgeter,
                operation_id=self._prompt_operation_id(
                    "chunk_extraction_repair",
                    chunk_result.file_path,
                    chunk_result.chunk_index,
                ),
                agent_name="chunk_extraction_repair",
                diagnostic_metadata={
                    "file_path": chunk_result.file_path,
                    "chunk_index": chunk_result.chunk_index,
                    "start_idx": chunk_result.start_idx,
                    "end_idx": chunk_result.end_idx,
                },
                think=None,
                num_ctx=self.ollama_client.max_context_length,
            )
        except CompletionError as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="chunk_extraction_repair",
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

        self._record_llm_call_result(
            data_package_id=data_package_id,
            result=repair,
            agent_name="chunk_extraction_repair",
        )
        repair_chunk_context = EvidenceChunkContext(
            content=chunk_content,
            metadata=EvidenceChunkMetadata(
                start_idx=chunk_result.start_idx,
                end_idx=chunk_result.end_idx,
                file_path=chunk_result.file_path,
                data_package_name=data_package_id,
            ),
        )
        validated_context = await self._validate_assess_and_route_evidence_context_for_chunk(
            repair.output,
            chunk_content=chunk_content,
            chunk_context=repair_chunk_context,
            chunk_result=chunk_result,
            state=state,
            critic_granularity=state.evidence_critic_granularity,
            data_package_id=data_package_id,
        )
        chunk_result.status = "completed"
        chunk_result.evidence_context = validated_context
        chunk_result.response_duration_ms = self._usage_float(
            repair.usage,
            "response_duration_ms",
        )
        chunk_result.context_tokens = self._single_attempt_input_tokens(repair)
        self._save_run_state(data_package_id, state)

        partial_context = await self._save_current_evidence_artifacts(
            data_package_id=data_package_id,
            state=state,
        )
        progress.processed_chunks = self._completed_chunk_count(state)
        progress.interim_evidence_context = partial_context
        progress.current_chunk = None
        progress.chunk_results = state.chunk_results
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)

    async def _save_current_evidence_artifacts(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        evidence_context: RoutedEvidenceContext | None = None,
    ) -> RoutedEvidenceContext:
        """Save evidence artifacts after a chunk. Only exact-text dedup here."""
        assert self.output_repository is not None
        if evidence_context is None:
            evidence_context, duplicate_records = await self._filtered_completed_evidence_context(
                state, data_package_id=data_package_id,
                similarity_dedup=False,
            )
        else:
            rank_by_path = {ranked.file_path: ranked.rank for ranked in state.ranked_files}
            evidence_context, duplicate_records = dedupe_repeated_evidence_notes(
                evidence_context,
                file_rank_by_path=rank_by_path,
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
        return evidence_context

    async def _dedupe_by_similarity(
        self,
        context: RoutedEvidenceContext,
        *,
        file_rank_by_path: dict[str, int],
        data_package_id: str = "",
    ) -> tuple[RoutedEvidenceContext, list[FilteredEvidenceNote]]:
        """Phase-2 dedup: SequenceMatcher similarity grouping + LLM triage.

        Uses difflib.SequenceMatcher (lexical, deterministic, no embeddings)
        to compute pairwise similarity between evidence texts.
        Only groups notes with DIFFERENT (category, role) pairs —
        same-classification duplicates are already handled by exact-text dedup.
        Also skips very short evidence texts (< 12 chars) as they produce
        ambiguous ratios that group unrelated numbers together.
        """
        from difflib import SequenceMatcher

        notes = context.portable_evidence
        if len(notes) <= 1:
            return context, []

        # Allow short but still semantically stable labels like "Version 3.2"
        # or "BF1= 500.13" to participate in similarity grouping.
        MIN_TEXT_LEN = 10

        def _sim(a: str, b: str) -> float:
            """SequenceMatcher ratio on normalized text."""
            na = _normalize_evidence_text(a)
            nb = _normalize_evidence_text(b)
            if len(na) < MIN_TEXT_LEN or len(nb) < MIN_TEXT_LEN:
                return 0.0
            return SequenceMatcher(None, na, nb).ratio()

        def _class_key(note: EvidenceCandidate) -> str:
            return f"{note.category}|{note.role}"

        # Build valid list (skip notes with very short evidence text)
        valid = [(i, notes[i]) for i in range(len(notes))
                 if len(_normalize_evidence_text(notes[i].evidence_text)) >= MIN_TEXT_LEN]
        if len(valid) <= 1:
            return context, []

        n = len(valid)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Group by similarity. Same-class notes stay protected unless their
        # normalized evidence text is exactly identical, which is a safe
        # duplicate-collapse case for repeated labels like "Version 3.2".
        for i in range(n):
            for j in range(i + 1, n):
                same_class = _class_key(valid[i][1]) == _class_key(valid[j][1])
                sim = _sim(valid[i][1].evidence_text, valid[j][1].evidence_text)
                if same_class and sim < 1.0:
                    continue
                if sim >= SIMILARITY_GROUP_THRESHOLD:
                    union(i, j)

        groups_map: dict[int, list[tuple[int, EvidenceCandidate]]] = {}
        for idx in range(n):
            root = find(idx)
            orig_idx = valid[idx][0]
            note = valid[idx][1]
            groups_map.setdefault(root, []).append((orig_idx, note))

        groups = list(groups_map.values())

        kept_with_order: list[tuple[int, EvidenceCandidate]] = []
        # Add notes with short evidence text as singletons (not similarity-compared)
        for i in range(len(notes)):
            if len(_normalize_evidence_text(notes[i].evidence_text)) < MIN_TEXT_LEN:
                kept_with_order.append((i, notes[i]))

        dropped: list[FilteredEvidenceNote] = []

        for group in groups:
            if len(group) <= 1:
                if group:
                    kept_with_order.append(group[0])
                continue

            max_sim = 0.0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    s = _sim(group[i][1].evidence_text, group[j][1].evidence_text)
                    if s > max_sim:
                        max_sim = s

            if max_sim >= SIMILARITY_DETERMINISTIC_THRESHOLD:
                representative, group_dropped = deterministic_triage(
                    group, file_rank_by_path=file_rank_by_path,
                )
            else:
                representative, group_dropped = await self._llm_triage(
                    group, file_rank_by_path=file_rank_by_path,
                    data_package_id=data_package_id,
                )

            kept_with_order.append(representative)
            for _, note in group_dropped:
                dropped.append(
                    _filtered_record(
                        note,
                        reason="similarity_merged",
                        duplicate_representative_id=representative[1].candidate_id,
                    )
                )

        kept = [note for _, note in sorted(kept_with_order, key=lambda item: item[0])]
        return context.model_copy(update={"portable_evidence": kept}), dropped

    async def _llm_triage(
        self,
        group: list[tuple[int, EvidenceCandidate]],
        *,
        file_rank_by_path: dict[str, int],
        data_package_id: str = "",
    ) -> tuple[tuple[int, EvidenceCandidate], list[tuple[int, EvidenceCandidate]]]:
        """Use the LLM to pick the best representative from a similarity group."""
        assert self.ollama_client is not None
        prompt = build_triage_prompt(group, file_rank_by_path=file_rank_by_path)
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=_DEDUPE_TRIAGE_SYSTEM_PROMPT,
                prompt=prompt,
                system_components=[("dedupe_triage", _DEDUPE_TRIAGE_SYSTEM_PROMPT)],
                prompt_components=[("dedupe_triage_prompt", prompt)],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id(
                    "dedupe_triage",
                    "similarity",
                    str(len(group)),
                ),
                agent_name="dedupe_triage",
                diagnostic_metadata={"group_size": len(group)},
                output_type=TriageSelection,
                retries=1,
                think=None,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="dedupe_triage",
            )
            return triage_parse_response(result.output, group)
        except Exception:
            return deterministic_triage(group, file_rank_by_path=file_rank_by_path)

    @staticmethod
    def _clear_chunk_evidence_contexts(state: ExtractionRunState) -> None:

        for chunk_result in state.chunk_results:
            chunk_result.evidence_context = None

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
            "evidence_text_unsupported": "unsupported evidence candidates",
            "candidate_rejected": "rejected evidence candidates",
            "duplicate_evidence": "duplicate evidence candidates",
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
        context: RoutedEvidenceContext,
        state: ExtractionRunState,
    ) -> RoutedEvidenceContext:
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
                        "; ".join(summary.metadata_signals[:3]),
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
    ) -> RoutedEvidenceContext | None:
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

    @classmethod
    def _global_evidence_context_for_prompt(
        cls,
        state: ExtractionRunState,
        *,
        current_chunk_index: int,
    ) -> RoutedEvidenceContext | None:
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

