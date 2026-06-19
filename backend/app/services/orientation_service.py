from __future__ import annotations

from app.services.extraction_shared import *


async def generate_structured(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.generate_structured(*args, **kwargs)


async def repair_structured_output(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.repair_structured_output(*args, **kwargs)


class OrientationService:
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

    def _rank_files_from_summaries(
        self,
        *,
        data_package: Any,
        state: ExtractionRunState,
    ) -> FileRankingResult:
        file_contexts = {
            file.file_path: FileContext(
                file_path=file.file_path,
                byte_size=len(file.raw_content),
            )
            for file in data_package.files
        }
        return rank_summarized_files(
            state.initial_file_summaries,
            file_contexts=file_contexts,
        )

    async def _generate_initial_file_summaries(
        self,
        *,
        data_package_id: str,
        data_package: Any,
        state: ExtractionRunState,
        warnings: list[str],
    ) -> None:
        diagnostics = InitialFileSummaryDiagnostics()
        candidate_files = list(self._initial_file_summary_candidate_files(data_package))
        summary_progress = InitialFileSummaryProgress(total_files=len(candidate_files))
        state.initial_file_summary_progress = summary_progress
        self._save_run_state(data_package_id, state)
        self._update_initial_context_progress(
            data_package_id,
            self._initial_context_progress_from_state(
                state,
                warnings=warnings,
                stage="initial_file_summaries",
            ),
        )
        if not hasattr(self.ollama_client, "ollama_client"):
            for file_entry in candidate_files:
                if self._should_skip_initial_file_summary(file_entry):
                    diagnostics.records.append(
                        InitialFileSummaryDiagnosticRecord(
                            file_path=file_entry.file_path,
                            reason="skipped",
                            message="Image files are not text-extractable.",
                        )
                    )
            state.initial_file_summaries = [
                self._failed_initial_file_summary(
                    file_path=file_entry.file_path,
                    reason="No Ollama client is available for file summary generation.",
                    diagnostics=diagnostics,
                )
                for file_entry in candidate_files
                if not self._should_skip_initial_file_summary(file_entry)
            ]
            state.initial_file_summary_progress = InitialFileSummaryProgress(
                total_files=len(candidate_files),
                processed_files=len(candidate_files),
                summarized_files=0,
                skipped_files=sum(
                    1
                    for file_entry in candidate_files
                    if self._should_skip_initial_file_summary(file_entry)
                ),
                failed_files=len(state.initial_file_summaries),
            )
            state.initial_file_summary_status = "failed"
            self._save_run_state(data_package_id, state)
            self._update_initial_context_progress(
                data_package_id,
                self._initial_context_progress_from_state(
                    state,
                    warnings=warnings,
                    stage="initial_file_summaries",
                ),
            )
            self._persist_initial_file_summaries(data_package_id, state)
            self._persist_initial_file_summary_diagnostics(
                data_package_id,
                state,
                diagnostics=diagnostics,
            )
            return

        summaries: list[ExtractionFileSummary] = []
        skipped_count = 0
        failed_count = 0

        def publish_summary_progress(current_file_path: str | None = None) -> None:
            state.initial_file_summaries = list(summaries)
            state.initial_file_summary_progress = InitialFileSummaryProgress(
                total_files=len(candidate_files),
                processed_files=len(summaries) + skipped_count,
                summarized_files=sum(
                    1 for summary in summaries if summary.status == "summarized"
                ),
                skipped_files=skipped_count,
                failed_files=failed_count,
                current_file_path=current_file_path,
            )
            self._save_run_state(data_package_id, state)
            self._update_initial_context_progress(
                data_package_id,
                self._initial_context_progress_from_state(
                    state,
                    warnings=warnings,
                    stage="initial_file_summaries",
                ),
            )

        for file_entry in candidate_files:
            publish_summary_progress(current_file_path=file_entry.file_path)
            if self._should_skip_initial_file_summary(file_entry):
                skipped_count += 1
                warnings.append(
                    f"Initial file summary skipped for {file_entry.file_path}: image files are not text-extractable."
                )
                diagnostics.records.append(
                    InitialFileSummaryDiagnosticRecord(
                        file_path=file_entry.file_path,
                        reason="skipped",
                        message="Image files are not text-extractable.",
                    )
                )
                publish_summary_progress()
                continue
            try:
                extracted_content = file_entry.get_extracted_content()
                if not extracted_content.strip():
                    skipped_count += 1
                    warnings.append(
                        f"Initial file summary skipped for {file_entry.file_path}: no extractable text content."
                    )
                    diagnostics.records.append(
                        InitialFileSummaryDiagnosticRecord(
                            file_path=file_entry.file_path,
                            reason="skipped",
                            message="No extractable text content.",
                        )
                    )
                    publish_summary_progress()
                    continue
                content_windows = self._initial_file_summary_content_windows(
                    extracted_content,
                    num_ctx=self.ollama_client.max_context_length,
                )
                file_summary_prompt = build_extraction_file_summary_prompt(
                    data_package_name=data_package.file_name,
                    file_path=file_entry.file_path,
                    byte_size=len(file_entry.raw_content),
                    extracted_char_count=len(extracted_content),
                    content_windows=content_windows,
                )
                try:
                    result = await generate_structured(
                        self.ollama_client,
                        model=self.ollama_client.chat_model,
                        system=EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT,
                        prompt=file_summary_prompt,
                        system_components=[
                            ("file_summary_system_prompt", EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT),
                        ],
                        prompt_components=[
                            (
                                "task_intro",
                                "Summarize one file for later extraction orientation.\n\n",
                            ),
                            (
                                "file_metadata",
                                f"Data package name: {data_package.file_name}\n"
                                f"File path: {file_entry.file_path}\n"
                                f"Byte size: {len(file_entry.raw_content)}\n"
                                f"Extracted character count: {len(extracted_content)}\n\n",
                            ),
                            (
                                "content_windows_json",
                                "Sampled content windows JSON:\n"
                                + "["
                                + ",\n".join(window.model_dump_json() for window in content_windows)
                                + "]\n\n",
                            ),
                            (
                                "return_instruction",
                                "Return an ExtractionFileSummary for this file. "
                                "Keep the summary compact: prefer 3-6 high-level, non-repetitive signals per list. "
                                "Use common metadata categories as orientation only, such as instrument settings, "
                                "software settings, acquisition settings, processing settings, calibration or reference settings, "
                                "sample conditions, identifiers, units, and quantity labels. "
                                "These categories are examples only: do not copy them into the output and do not enumerate every parameter. "
                                "Use metadata_signals for concise file-local orientation, instrument_or_software_terms_and_settings for visible "
                                "instrument/software/method/setting terms, and quantitative_signals only for coarse quantitative orientation. "
                                "Do not repeat identical timestamps, labels, units, or values.",
                            ),
                        ],
                        token_budgeter=self._prompt_token_budgeter(),
                        operation_id=self._prompt_operation_id(
                            "initial_file_summary",
                            file_entry.file_path,
                        ),
                        agent_name="initial_file_summary",
                        diagnostic_metadata={
                            "file_path": file_entry.file_path,
                        },
                        output_type=ExtractionFileSummary,
                        omitted_fields={
                            "ExtractionFileSummary": ["file_path", "status"],
                        },
                        retries=1,
                        temperature=0.0,
                        think=None,
                        num_ctx=self.ollama_client.max_context_length,
                    )
                except CompletionError:
                    raise
                self._record_llm_call_result(
                    data_package_id=data_package_id,
                    result=result,
                    agent_name="initial_file_summary",
                )
                summaries.append(
                    self._validated_initial_file_summary(
                        result.output,
                        file_path=file_entry.file_path,
                        sampled_text="\n".join(
                            window.text for window in content_windows
                        ),
                        warnings=warnings,
                        diagnostics=diagnostics,
                    )
                )
                publish_summary_progress()
            except CompletionError as exc:
                self._record_llm_call_exception(
                    data_package_id=data_package_id,
                    exc=exc,
                    agent_name="initial_file_summary",
                )
                warnings.append(
                    f"Initial file summary failed for {file_entry.file_path}: {exc}"
                )
                failed_count += 1
                summaries.append(
                    self._failed_initial_file_summary(
                        file_path=file_entry.file_path,
                        reason=str(exc),
                        details=self._structured_completion_debug_details(exc),
                        diagnostics=diagnostics,
                    )
                )
                publish_summary_progress()
            except Exception as exc:
                warnings.append(
                    f"Initial file summary failed for {file_entry.file_path}: {exc}"
                )
                logger.warning(
                    "Initial file summary failed",
                    extra={
                        "data_package_id": data_package_id,
                        "file_path": file_entry.file_path,
                        "error_type": type(exc).__name__,
                    },
                )
                failed_count += 1
                summaries.append(
                    self._failed_initial_file_summary(
                        file_path=file_entry.file_path,
                        reason=str(exc),
                        diagnostics=diagnostics,
                    )
                )
                publish_summary_progress()

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
        state.initial_file_summary_progress = state.initial_file_summary_progress.model_copy(
            update={"current_file_path": None}
        ) if state.initial_file_summary_progress else None
        self._save_run_state(data_package_id, state)
        self._update_initial_context_progress(
            data_package_id,
            self._initial_context_progress_from_state(
                state,
                warnings=warnings,
                stage="initial_file_summaries",
            ),
        )
        self._persist_initial_file_summaries(data_package_id, state)
        self._persist_initial_file_summary_diagnostics(
            data_package_id,
            state,
            diagnostics=diagnostics,
        )

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
        ranked_file_summaries = self._rank_ordered_initial_file_summaries(
            summaries=summarized_file_summaries,
            ranking=ranking,
        )
        fallback_previews = [] if summarized_file_summaries else previews
        seeded_overview = self._seed_initial_overview_graph(
            data_package_name=data_package.file_name,
            ranked_files=overview_ranked_files,
            file_summaries=summarized_file_summaries,
            file_previews=previews,
        )
        overview_prompt_budgeter = self._prompt_token_budgeter()
        if overview_prompt_budgeter.fallback_reason:
            warning = (
                "Initial overview prompt budgeting used conservative estimates: "
                + overview_prompt_budgeter.fallback_reason
            )
            if warning not in warnings:
                warnings.append(warning)
        overview_input_budget = self._initial_overview_input_token_budget(
            self.ollama_client.max_context_length
        )
        overview_prompt_report: dict[str, Any] = {
            "max_input_tokens": overview_input_budget,
            "expected_output_token_reserve": INITIAL_OVERVIEW_EXPECTED_OUTPUT_TOKENS,
            "input_safety_margin_tokens": INITIAL_OVERVIEW_INPUT_SAFETY_MARGIN_TOKENS,
        }
        try:
            overview_prompt, overview_prompt_report = self._build_budgeted_initial_overview_prompt(
                data_package_name=data_package.file_name,
                ranked_files=overview_ranked_files,
                file_summaries=ranked_file_summaries,
                file_previews=fallback_previews,
                seeded_overview=seeded_overview,
                token_budgeter=overview_prompt_budgeter,
                max_input_tokens=overview_input_budget,
            )
            overview_prompt_components = [
                (str(component.get("name", "component")), str(component.get("text", "")))
                for component in overview_prompt_report.get("final_prompt_components", [])
            ]
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EXTRACTION_OVERVIEW_SYSTEM_PROMPT,
                prompt=overview_prompt,
                system_components=[
                    ("initial_overview_system_prompt", EXTRACTION_OVERVIEW_SYSTEM_PROMPT),
                ],
                prompt_components=overview_prompt_components,
                token_budgeter=overview_prompt_budgeter,
                operation_id=self._prompt_operation_id("initial_extraction_overview"),
                agent_name="initial_extraction_overview",
                diagnostic_metadata={
                    "data_package_id": data_package_id,
                    "data_package_name": data_package.file_name,
                },
                output_type=ExtractionOverviewModelOutput,
                retries=1,
                think=None,
                num_ctx=self.ollama_client.max_context_length,
                num_predict=INITIAL_OVERVIEW_MAX_OUTPUT_TOKENS,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="initial_extraction_overview",
            )
            sanitized_overview = self._sanitize_initial_overview_graph(
                result.output.to_extraction_overview(),
                allowed_file_paths=preview_file_paths,
                seed_overview=seeded_overview,
                summaries_available=bool(summarized_file_summaries),
                warnings=warnings,
            )
            state.initial_extraction_overview = self._with_initial_overview_provenance(
                sanitized_overview,
                source_fingerprint=source_fingerprint,
                previews=previews,
            )
            state.initial_extraction_overview_status = "structured"
            overview_diagnostic = InitialOverviewPromptDiagnostic(
                status="structured_success",
                prompt_budget=overview_prompt_report,
                included_summary_paths=list(
                    overview_prompt_report.get("included_summary_paths", [])
                ),
                dropped_summary_paths=list(
                    overview_prompt_report.get("dropped_summary_paths", [])
                ),
                included_ranked_paths=list(
                    overview_prompt_report.get("included_ranked_paths", [])
                ),
                dropped_ranked_paths=list(
                    overview_prompt_report.get("dropped_ranked_paths", [])
                ),
                included_preview_paths=list(
                    overview_prompt_report.get("included_preview_paths", [])
                ),
                dropped_preview_paths=list(
                    overview_prompt_report.get("dropped_preview_paths", [])
                ),
                hard_truncated=bool(
                    overview_prompt_report.get("hard_truncated", False)
                ),
                usage=run_usage_to_dict(result.usage),
            )
            state.initial_extraction_overview_diagnostic = overview_diagnostic
            self._save_run_state(data_package_id, state)
            self._persist_initial_extraction_overview(data_package_id, state)
            self._persist_initial_extraction_overview_diagnostic(
                data_package_id,
                state,
                diagnostic=overview_diagnostic,
            )
            return
        except (CompletionError, ValueError) as exc:
            if not isinstance(exc, CompletionError):
                exc = CompletionError(str(exc))
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="initial_extraction_overview",
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
            overview_diagnostic = self._initial_overview_failure_diagnostic(
                exc,
                prompt_budget=overview_prompt_report,
                token_budgeter=overview_prompt_budgeter,
            )
            state.initial_extraction_overview_diagnostic = overview_diagnostic
            self._save_run_state(data_package_id, state)
            self._persist_initial_extraction_overview_diagnostic(
                data_package_id,
                state,
                diagnostic=overview_diagnostic,
            )

        try:
            fallback_prompt = build_extraction_overview_fallback_prompt(
                data_package_name=data_package.file_name,
                ranked_files=overview_ranked_files,
                file_previews=fallback_previews,
            )
            fallback_prompt = overview_prompt_budgeter.truncate(
                fallback_prompt,
                max_tokens=max(
                    1,
                    overview_input_budget
                    - overview_prompt_budgeter.count(EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT),
                ),
            )
            fallback_result = await generate_text(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT,
                prompt=fallback_prompt,
                options={
                    "temperature": 0.1,
                    "seed": 42,
                    "num_ctx": self.ollama_client.max_context_length,
                },
                think=None,
                keep_alive=-1,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name="initial_extraction_overview_fallback",
                usage=fallback_result.usage,
            )
            text = fallback_result.output.strip()
            if not text:
                raise CompletionError("Initial extraction overview fallback returned an empty response.")
            state.initial_extraction_overview = self._with_initial_overview_provenance(
                seeded_overview.model_copy(
                    update={
                        "uncertainties": [
                            *seeded_overview.uncertainties,
                            text,
                            "This overview is an unstructured fallback and is orientation only.",
                        ]
                    }
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

    @classmethod
    def _initial_overview_input_token_budget(cls, num_ctx: int | None) -> int:
        context_window = num_ctx or 8192
        reserve_safe_budget = (
            context_window
            - INITIAL_OVERVIEW_EXPECTED_OUTPUT_TOKENS
            - INITIAL_OVERVIEW_INPUT_SAFETY_MARGIN_TOKENS
        )
        if reserve_safe_budget <= 0:
            return 1
        return reserve_safe_budget

    @classmethod
    def _initial_overview_prompt_component_breakdown(
        cls,
        *,
        data_package_name: str,
        ranked_files: list[RankedFile],
        file_summaries: list[ExtractionFileSummary],
        file_previews: list[ExtractionOverviewFilePreview],
        seeded_overview: ExtractionOverview,
        seeded_overview_prompt_text: str,
        token_budgeter: PromptTokenBudgeter,
        prompt: str,
    ) -> dict[str, Any]:
        components = build_extraction_overview_prompt_components(
            data_package_name=data_package_name,
            ranked_files=ranked_files,
            file_summaries=file_summaries,
            file_previews=file_previews,
            seeded_overview=seeded_overview,
            seeded_overview_prompt_text=seeded_overview_prompt_text,
        )
        rows: list[dict[str, Any]] = []
        system_prompt = EXTRACTION_OVERVIEW_SYSTEM_PROMPT
        cumulative_text = system_prompt
        cumulative_tokens = token_budgeter.count(cumulative_text)
        rows.append(
            {
                "component": "system_prompt",
                "delta_tokens": cumulative_tokens,
                "cumulative_tokens": cumulative_tokens,
                "chars": len(system_prompt),
                "included_chars": len(system_prompt),
            }
        )

        consumed_chars = 0
        prompt_chars = len(prompt)
        for component_name, component_text in components:
            remaining_chars = max(0, prompt_chars - consumed_chars)
            included_text = component_text[:remaining_chars]
            consumed_chars += len(component_text)
            if not included_text:
                rows.append(
                    {
                        "component": component_name,
                        "delta_tokens": 0,
                        "cumulative_tokens": cumulative_tokens,
                        "chars": len(component_text),
                        "included_chars": 0,
                    }
                )
                continue

            next_text = cumulative_text + included_text
            next_tokens = token_budgeter.count(next_text)
            rows.append(
                {
                    "component": component_name,
                    "delta_tokens": next_tokens - cumulative_tokens,
                    "cumulative_tokens": next_tokens,
                    "chars": len(component_text),
                    "included_chars": len(included_text),
                }
            )
            cumulative_text = next_text
            cumulative_tokens = next_tokens

        return {
            "message_total_tokens": token_budgeter.count(system_prompt)
            + token_budgeter.count(prompt),
            "concatenated_total_tokens": cumulative_tokens,
            "prompt_tokens": token_budgeter.count(prompt),
            "prompt_chars": len(prompt),
            "components": rows,
        }

    @staticmethod
    def _isolated_token_count(
        token_budgeter: PromptTokenBudgeter,
        value: str,
    ) -> int:
        return token_budgeter.count(value)

    @classmethod
    def _initial_overview_payload_token_breakdown(
        cls,
        *,
        ranked_files: list[RankedFile],
        used_ranked_files: list[RankedFile],
        compacted_summaries: list[ExtractionFileSummary],
        used_compacted_summaries: list[ExtractionFileSummary],
        used_file_previews: list[ExtractionOverviewFilePreview],
        file_previews: list[ExtractionOverviewFilePreview],
        seeded_overview: ExtractionOverview,
        seeded_overview_prompt_text: str,
        token_budgeter: PromptTokenBudgeter,
    ) -> dict[str, Any]:
        seed_payload = json.loads(seeded_overview.model_dump_json(exclude_defaults=True))
        seed_nodes = seed_payload.get("nodes", [])
        seed_edges = seed_payload.get("edges", [])
        seed_uncertainties = seed_payload.get("uncertainties", [])
        return {
            "ranked_files": {
                "original_count": len(ranked_files),
                "included_count": len(used_ranked_files),
                "original_json_tokens": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(file.model_dump_json() for file in ranked_files),
                ),
                "included_json_tokens": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(file.model_dump_json() for file in used_ranked_files),
                ),
            },
            "compact_summaries": {
                "original_count": len(compacted_summaries),
                "included_count": len(used_compacted_summaries),
                "json_tokens_total": sum(
                    cls._isolated_token_count(
                        token_budgeter,
                        summary.model_dump_json(exclude_defaults=True),
                    )
                    for summary in compacted_summaries
                ),
                "per_file": [
                    {
                        "file_path": summary.file_path,
                        "included": summary.file_path
                        in {used.file_path for used in used_compacted_summaries},
                        "json_tokens": cls._isolated_token_count(
                            token_budgeter,
                            summary.model_dump_json(exclude_defaults=True),
                        ),
                        "json_chars": len(summary.model_dump_json(exclude_defaults=True)),
                    }
                    for summary in compacted_summaries
                ],
            },
            "file_previews": {
                "original_count": len(file_previews),
                "included_count": len(used_file_previews),
                "original_json_tokens": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(preview.model_dump_json() for preview in file_previews),
                ),
                "included_json_tokens": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(
                        preview.model_dump_json() for preview in used_file_previews
                    ),
                ),
            },
            "seeded_graph": {
                "node_count": len(seed_nodes),
                "edge_count": len(seed_edges),
                "uncertainty_count": len(seed_uncertainties),
                "json_tokens": cls._isolated_token_count(
                    token_budgeter,
                    seeded_overview.model_dump_json(exclude_defaults=True),
                ),
                "nodes_json_tokens": sum(
                    cls._isolated_token_count(
                        token_budgeter,
                        json.dumps(node, separators=(",", ":")),
                    )
                    for node in seed_nodes
                ),
                "edges_json_tokens": sum(
                    cls._isolated_token_count(
                        token_budgeter,
                        json.dumps(edge, separators=(",", ":")),
                    )
                    for edge in seed_edges
                ),
                "uncertainties_json_tokens": sum(
                    cls._isolated_token_count(
                        token_budgeter,
                        json.dumps(uncertainty, separators=(",", ":")),
                    )
                    for uncertainty in seed_uncertainties
                ),
            },
            "seeded_graph_compact_text": {
                "tokens": cls._isolated_token_count(
                    token_budgeter,
                    seeded_overview_prompt_text,
                ),
                "chars": len(seeded_overview_prompt_text),
                "line_count": len(seeded_overview_prompt_text.splitlines()),
            },
        }

    @classmethod
    def _build_budgeted_initial_overview_prompt(
        cls,
        *,
        data_package_name: str,
        ranked_files: list[RankedFile],
        file_summaries: list[ExtractionFileSummary],
        file_previews: list[ExtractionOverviewFilePreview],
        seeded_overview: ExtractionOverview,
        token_budgeter: PromptTokenBudgeter,
        max_input_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        def build_prompt_for(
            *,
            ranked_files_to_include: list[RankedFile],
            summaries_to_include: list[ExtractionFileSummary],
            previews_to_include: list[ExtractionOverviewFilePreview],
            seeded_overview_prompt_text: str,
        ) -> str:
            return build_extraction_overview_prompt(
                data_package_name=data_package_name,
                ranked_files=ranked_files_to_include,
                file_summaries=summaries_to_include,
                file_previews=previews_to_include,
                seeded_overview=seeded_overview,
                seeded_overview_prompt_text=seeded_overview_prompt_text,
            )

        def component_text(
            *,
            ranked_files_to_include: list[RankedFile],
            summaries_to_include: list[ExtractionFileSummary],
            previews_to_include: list[ExtractionOverviewFilePreview],
            seeded_overview_prompt_text: str,
            component_names: set[str],
        ) -> str:
            components = build_extraction_overview_prompt_components(
                data_package_name=data_package_name,
                ranked_files=ranked_files_to_include,
                file_summaries=summaries_to_include,
                file_previews=previews_to_include,
                seeded_overview=seeded_overview,
                seeded_overview_prompt_text=seeded_overview_prompt_text,
            )
            return "".join(text for name, text in components if name in component_names)

        def take_ranked_files_within_budget(
            candidates: list[RankedFile],
            max_tokens: int,
        ) -> list[RankedFile]:
            selected: list[RankedFile] = []
            for candidate in candidates:
                trial = selected + [candidate]
                tokens = cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(file.model_dump_json() for file in trial),
                )
                if tokens > max_tokens and selected:
                    break
                if tokens > max_tokens:
                    continue
                selected = trial
            return selected

        def take_summaries_within_budget(
            candidates: list[ExtractionFileSummary],
            max_tokens: int,
        ) -> list[ExtractionFileSummary]:
            selected: list[ExtractionFileSummary] = []
            for candidate in candidates:
                trial = selected + [candidate]
                tokens = cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(
                        summary.model_dump_json(exclude_defaults=True)
                        for summary in trial
                    ),
                )
                if tokens > max_tokens and selected:
                    break
                if tokens > max_tokens:
                    continue
                selected = trial
            return selected

        def take_previews_within_budget(
            candidates: list[ExtractionOverviewFilePreview],
            max_tokens: int,
        ) -> list[ExtractionOverviewFilePreview]:
            selected: list[ExtractionOverviewFilePreview] = []
            for candidate in candidates:
                trial = selected + [candidate]
                tokens = cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(preview.model_dump_json() for preview in trial),
                )
                if tokens > max_tokens and selected:
                    break
                if tokens > max_tokens:
                    continue
                selected = trial
            return selected

        def truncate_lines_to_budget(text: str, max_tokens: int) -> tuple[str, bool]:
            if max_tokens <= 0:
                return "", bool(text)
            if token_budgeter.count(text) <= max_tokens:
                return text, False
            kept: list[str] = []
            for line in text.splitlines():
                trial = "\n".join([*kept, line])
                if token_budgeter.count(trial) > max_tokens:
                    break
                kept.append(line)
            if not kept:
                return token_budgeter.truncate(text, max_tokens=max_tokens), True
            return "\n".join([*kept, "... compact graph truncated ..."]), True

        compacted_summaries = [
            cls._compact_initial_file_summary_for_overview(
                summary,
                token_budgeter=token_budgeter,
            )
            for summary in file_summaries
        ]
        all_compacted_summaries = list(compacted_summaries)
        full_seeded_prompt_text = compact_seeded_overview_for_prompt(seeded_overview)
        original_prompt = build_prompt_for(
            ranked_files_to_include=ranked_files,
            summaries_to_include=file_summaries,
            previews_to_include=file_previews,
            seeded_overview_prompt_text=full_seeded_prompt_text,
        )
        original_prompt_tokens = token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + original_prompt
        )
        original_component_breakdown = cls._initial_overview_prompt_component_breakdown(
            data_package_name=data_package_name,
            ranked_files=ranked_files,
            file_summaries=file_summaries,
            file_previews=file_previews,
            seeded_overview=seeded_overview,
            seeded_overview_prompt_text=full_seeded_prompt_text,
            token_budgeter=token_budgeter,
            prompt=original_prompt,
        )

        protected_text = component_text(
            ranked_files_to_include=[],
            summaries_to_include=[],
            previews_to_include=[],
            seeded_overview_prompt_text="",
            component_names={"intro_and_counts", "final_task_instructions"},
        )
        protected_section_tokens = token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + protected_text
        )
        if protected_section_tokens > max_input_tokens:
            raise ValueError(
                "Initial overview prompt budget cannot fit protected prompt sections "
                f"({protected_section_tokens} tokens > {max_input_tokens})."
            )

        summary_target_tokens = int(
            max_input_tokens * INITIAL_OVERVIEW_SUMMARY_BUDGET_RATIO
        )
        graph_target_tokens = int(
            max_input_tokens * INITIAL_OVERVIEW_GRAPH_BUDGET_RATIO
        )
        ranked_target_tokens = int(
            max_input_tokens * INITIAL_OVERVIEW_RANKED_FILE_BUDGET_RATIO
        )
        preview_target_tokens = (
            int(max_input_tokens * INITIAL_OVERVIEW_PREVIEW_BUDGET_RATIO)
            if not compacted_summaries
            else 0
        )

        used_compacted_summaries = take_summaries_within_budget(
            compacted_summaries,
            summary_target_tokens,
        )
        used_ranked_files = take_ranked_files_within_budget(
            ranked_files,
            ranked_target_tokens,
        )
        used_file_previews = take_previews_within_budget(
            file_previews,
            preview_target_tokens,
        )
        seeded_prompt_text, graph_text_truncated = truncate_lines_to_budget(
            full_seeded_prompt_text,
            graph_target_tokens,
        )

        def build_prompt() -> str:
            return build_prompt_for(
                ranked_files_to_include=used_ranked_files,
                summaries_to_include=used_compacted_summaries,
                previews_to_include=used_file_previews,
                seeded_overview_prompt_text=seeded_prompt_text,
            )

        prompt = build_prompt()
        compacted_prompt_tokens = token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt
        )
        compacted_component_breakdown = cls._initial_overview_prompt_component_breakdown(
            data_package_name=data_package_name,
            ranked_files=used_ranked_files,
            file_summaries=used_compacted_summaries,
            file_previews=used_file_previews,
            seeded_overview=seeded_overview,
            seeded_overview_prompt_text=seeded_prompt_text,
            token_budgeter=token_budgeter,
            prompt=prompt,
        )
        dropped_summary_paths = [
            summary.file_path
            for summary in compacted_summaries
            if summary.file_path not in {used.file_path for used in used_compacted_summaries}
        ]
        dropped_ranked_paths = [
            file.file_path
            for file in ranked_files
            if file.file_path not in {used.file_path for used in used_ranked_files}
        ]
        dropped_preview_paths = [
            preview.file_path
            for preview in file_previews
            if preview.file_path not in {used.file_path for used in used_file_previews}
        ]

        while used_file_previews and token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt
        ) > max_input_tokens:
            dropped_preview_paths.append(used_file_previews.pop().file_path)
            prompt = build_prompt()

        while used_ranked_files and token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt
        ) > max_input_tokens:
            dropped_ranked_paths.append(used_ranked_files.pop().file_path)
            prompt = build_prompt()

        if token_budgeter.count(EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt) > max_input_tokens:
            remaining_for_graph = max(0, graph_target_tokens // 2)
            seeded_prompt_text, graph_text_truncated_again = truncate_lines_to_budget(
                seeded_prompt_text,
                remaining_for_graph,
            )
            graph_text_truncated = graph_text_truncated or graph_text_truncated_again
            prompt = build_prompt()

        while len(used_compacted_summaries) > 1 and token_budgeter.count(
            EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt
        ) > max_input_tokens:
            dropped_summary_paths.append(used_compacted_summaries.pop().file_path)
            prompt = build_prompt()

        system_tokens = token_budgeter.count(EXTRACTION_OVERVIEW_SYSTEM_PROMPT)
        prompt_tokens = token_budgeter.count(prompt)
        prompt_before_hard_truncation = prompt
        final_before_truncation_breakdown = (
            cls._initial_overview_prompt_component_breakdown(
                data_package_name=data_package_name,
                ranked_files=used_ranked_files,
                file_summaries=used_compacted_summaries,
                file_previews=used_file_previews,
                seeded_overview=seeded_overview,
                seeded_overview_prompt_text=seeded_prompt_text,
                token_budgeter=token_budgeter,
                prompt=prompt_before_hard_truncation,
            )
        )
        hard_truncated = False
        if system_tokens + prompt_tokens > max_input_tokens:
            raise ValueError(
                "Initial overview prompt budget could not fit protected sections "
                "after optional context reduction."
            )
        final_sent_breakdown = cls._initial_overview_prompt_component_breakdown(
            data_package_name=data_package_name,
            ranked_files=used_ranked_files,
            file_summaries=used_compacted_summaries,
            file_previews=used_file_previews,
            seeded_overview=seeded_overview,
            seeded_overview_prompt_text=seeded_prompt_text,
            token_budgeter=token_budgeter,
            prompt=prompt,
        )
        final_prompt_components = build_extraction_overview_prompt_components(
            data_package_name=data_package_name,
            ranked_files=used_ranked_files,
            file_summaries=used_compacted_summaries,
            file_previews=used_file_previews,
            seeded_overview=seeded_overview,
            seeded_overview_prompt_text=seeded_prompt_text,
        )
        prompt_tokens = token_budgeter.count(prompt)

        report = {
            "max_input_tokens": max_input_tokens,
            "expected_output_token_reserve": INITIAL_OVERVIEW_EXPECTED_OUTPUT_TOKENS,
            "input_safety_margin_tokens": INITIAL_OVERVIEW_INPUT_SAFETY_MARGIN_TOKENS,
            "protected_section_tokens": protected_section_tokens,
            "system_tokens": system_tokens,
            "prompt_tokens": prompt_tokens,
            "total_input_tokens": system_tokens + prompt_tokens,
            "original_total_input_tokens": original_prompt_tokens,
            "compacted_total_input_tokens_before_drop": compacted_prompt_tokens,
            "token_budget_targets": {
                "ranked_files": ranked_target_tokens,
                "compact_summaries": summary_target_tokens,
                "seeded_graph_compact_text": graph_target_tokens,
                "fallback_previews": preview_target_tokens,
            },
            "token_budget_actuals": {
                "ranked_files": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(file.model_dump_json() for file in used_ranked_files),
                ),
                "compact_summaries": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(
                        summary.model_dump_json(exclude_defaults=True)
                        for summary in used_compacted_summaries
                    ),
                ),
                "seeded_graph_compact_text": cls._isolated_token_count(
                    token_budgeter,
                    seeded_prompt_text,
                ),
                "fallback_previews": cls._isolated_token_count(
                    token_budgeter,
                    ",\n".join(preview.model_dump_json() for preview in used_file_previews),
                ),
            },
            "tokenizer_fallback": token_budgeter.uses_fallback,
            "compact_summary_count": len(used_compacted_summaries),
            "original_summary_count": len(file_summaries),
            "included_summary_paths": [
                summary.file_path for summary in used_compacted_summaries
            ],
            "dropped_summary_paths": dropped_summary_paths,
            "ranked_file_count": len(used_ranked_files),
            "original_ranked_file_count": len(ranked_files),
            "included_ranked_paths": [
                file.file_path for file in used_ranked_files
            ],
            "dropped_ranked_paths": dropped_ranked_paths,
            "preview_count": len(used_file_previews),
            "original_preview_count": len(file_previews),
            "included_preview_paths": [
                preview.file_path for preview in used_file_previews
            ],
            "dropped_preview_paths": dropped_preview_paths,
            "hard_truncated": hard_truncated,
            "compact_graph_truncated": graph_text_truncated,
            "token_component_breakdown": {
                "original": original_component_breakdown,
                "compacted_before_drop": compacted_component_breakdown,
                "final_before_truncation": final_before_truncation_breakdown,
                "final_sent": final_sent_breakdown,
                "payloads": cls._initial_overview_payload_token_breakdown(
                    ranked_files=ranked_files,
                    used_ranked_files=used_ranked_files,
                    compacted_summaries=all_compacted_summaries,
                    used_compacted_summaries=used_compacted_summaries,
                    used_file_previews=used_file_previews,
                    file_previews=file_previews,
                    seeded_overview=seeded_overview,
                    seeded_overview_prompt_text=seeded_prompt_text,
                    token_budgeter=token_budgeter,
                ),
            },
            "final_prompt_components": [
                {"name": name, "text": text}
                for name, text in final_prompt_components
            ],
        }
        return prompt, report

    @classmethod
    def _compact_initial_file_summary_for_overview(
        cls,
        summary: ExtractionFileSummary,
        *,
        token_budgeter: PromptTokenBudgeter,
    ) -> ExtractionFileSummary:
        updates: dict[str, Any] = {
            "data_format": cls._compact_overview_text(
                summary.data_format,
                token_budgeter=token_budgeter,
                max_tokens=18,
            ),
            "explicit_purpose": cls._compact_overview_text(
                summary.explicit_purpose,
                token_budgeter=token_budgeter,
                max_tokens=28,
            ),
        }
        for field_name, max_items in OVERVIEW_SUMMARY_LIST_LIMITS.items():
            updates[field_name] = cls._compact_overview_text_list(
                getattr(summary, field_name),
                token_budgeter=token_budgeter,
                max_items=max_items,
                max_item_tokens=22,
            )

        compact = summary.model_copy(update=updates)
        while token_budgeter.count(compact.model_dump_json()) > INITIAL_OVERVIEW_SUMMARY_TOKEN_BUDGET:
            changed = False
            for field_name in OVERVIEW_SUMMARY_REDUCTION_ORDER:
                values = list(getattr(compact, field_name))
                if not values:
                    continue
                if len(values) > 1:
                    values = values[: max(1, len(values) // 2)]
                else:
                    values = []
                compact = compact.model_copy(update={field_name: values})
                changed = True
                break
            if not changed:
                compact = compact.model_copy(
                    update={
                        "purpose_evidence": [],
                        "metadata_signals": [],
                        "instrument_or_software_terms_and_settings": [],
                        "quantitative_signals": [],
                    }
                )
                break
        return compact

    @staticmethod
    def _compact_overview_text(
        value: str,
        *,
        token_budgeter: PromptTokenBudgeter,
        max_tokens: int,
    ) -> str:
        value = " ".join((value or "").split())
        if not value:
            return ""
        return token_budgeter.truncate(value, max_tokens=max_tokens)

    @classmethod
    def _compact_overview_text_list(
        cls,
        values: list[str],
        *,
        token_budgeter: PromptTokenBudgeter,
        max_items: int,
        max_item_tokens: int,
    ) -> list[str]:
        compacted: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = cls._compact_overview_text(
                value,
                token_budgeter=token_budgeter,
                max_tokens=max_item_tokens,
            )
            key = item.casefold()
            if not item or key in seen:
                continue
            compacted.append(item)
            seen.add(key)
            if len(compacted) >= max_items:
                break
        return compacted

    @classmethod
    def _initial_overview_failure_diagnostic(
        cls,
        exc: CompletionError,
        *,
        prompt_budget: dict[str, Any],
        token_budgeter: PromptTokenBudgeter,
    ) -> InitialOverviewFailureDiagnostic:
        failed_response = getattr(exc, "failed_response", None) or exc.details.get(
            "failed_response",
            "",
        )
        first_response = getattr(exc, "first_response", None) or exc.details.get(
            "first_response",
            "",
        )
        usage = getattr(exc, "usage", RunUsage())
        return InitialOverviewFailureDiagnostic(
            error_type=type(exc).__name__,
            message=str(exc),
            last_error_type=str(exc.details.get("last_error_type", "")),
            last_error=cls._compact_overview_text(
                str(exc.details.get("last_error", "")),
                token_budgeter=token_budgeter,
                max_tokens=INITIAL_OVERVIEW_FAILURE_EXCERPT_TOKENS,
            ),
            failed_response_excerpt=cls._compact_overview_text(
                str(failed_response),
                token_budgeter=token_budgeter,
                max_tokens=INITIAL_OVERVIEW_FAILURE_EXCERPT_TOKENS,
            ),
            first_model_output=str(first_response),
            failed_model_output=str(failed_response),
            prompt_budget=prompt_budget,
            usage={
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "prompt_eval_duration_ms": usage.prompt_eval_duration_ms,
                "load_duration_ms": usage.load_duration_ms,
                "response_duration_ms": usage.response_duration_ms,
                "total_duration_ms": usage.total_duration_ms,
            },
        )

    def _initial_overview_file_previews(
        self,
        *,
        data_package: Any,
        ranking: FileRankingResult,
    ) -> list[ExtractionOverviewFilePreview]:
        rank_by_path = {file.file_path: file.rank for file in ranking.files}
        package_files = sorted(
            enumerate(data_package.files),
            key=lambda item: (
                rank_by_path.get(item[1].file_path, 10_000),
                item[0],
            ),
        )
        previews: list[ExtractionOverviewFilePreview] = []
        fallback_rank = len(ranking.files)
        for inventory_index, file_entry in package_files:
            if self._should_skip_initial_file_summary(file_entry):
                lines = []
            else:
                try:
                    lines = file_entry.get_extracted_content().splitlines()
                except Exception as exc:
                    lines = [f"[Text extraction failed: {exc}]"]
            preview_rank = rank_by_path.get(
                file_entry.file_path,
                fallback_rank + inventory_index + 1,
            )
            previews.append(
                ExtractionOverviewFilePreview(
                    rank=preview_rank,
                    file_path=file_entry.file_path,
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
    def _initial_file_summary_candidate_files(data_package: Any) -> list[Any]:
        return list(getattr(data_package, "files", []) or [])

    @staticmethod
    def _validated_initial_file_summary(
        summary: ExtractionFileSummary,
        *,
        file_path: str,
        sampled_text: str,
        warnings: list[str],
        diagnostics: InitialFileSummaryDiagnostics | None = None,
    ) -> ExtractionFileSummary:
        update: dict[str, Any] = {
            "file_path": file_path,
            "status": "summarized",
        }
        if summary.file_path and summary.file_path != file_path:
            warnings.append(
                "Initial file summary returned a mismatched file_path; "
                f"expected {file_path}, got {summary.file_path}."
            )
        verified_purpose_evidence = [
            evidence
            for evidence in summary.purpose_evidence
            if evidence and evidence in sampled_text
        ]
        if len(verified_purpose_evidence) != len(summary.purpose_evidence):
            warnings.append(
                f"Initial file summary for {file_path} included purpose evidence not found in the sampled file text; dropping unsupported evidence."
            )
            if diagnostics is not None:
                diagnostics.records.append(
                    InitialFileSummaryDiagnosticRecord(
                        file_path=file_path,
                        reason="unsupported_purpose_evidence",
                        message="Purpose evidence was not found in the sampled file text.",
                        details={
                            "dropped": [
                                evidence
                                for evidence in summary.purpose_evidence
                                if evidence and evidence not in sampled_text
                            ]
                        },
                    )
                )
            update["purpose_evidence"] = verified_purpose_evidence
        if summary.explicit_purpose and not verified_purpose_evidence:
            warnings.append(
                f"Initial file summary for {file_path} included an explicit purpose without evidence; clearing it."
            )
            update["explicit_purpose"] = ""
            if diagnostics is not None:
                diagnostics.records.append(
                    InitialFileSummaryDiagnosticRecord(
                        file_path=file_path,
                        reason="unsupported_explicit_purpose",
                        message="Explicit purpose was cleared because no direct purpose evidence was provided.",
                        details={"explicit_purpose": summary.explicit_purpose},
                    )
                )
        return summary.model_copy(update=update)

    @staticmethod
    def _failed_initial_file_summary(
        *,
        file_path: str,
        reason: str,
        details: dict[str, Any] | None = None,
        diagnostics: InitialFileSummaryDiagnostics | None = None,
    ) -> ExtractionFileSummary:
        if diagnostics is not None:
            diagnostics.records.append(
                InitialFileSummaryDiagnosticRecord(
                    file_path=file_path,
                    reason="failed",
                    message=reason,
                    details=details or {},
                )
            )
        return ExtractionFileSummary(
            file_path=file_path,
            status="failed",
        )

    @staticmethod
    def _structured_completion_debug_details(exc: CompletionError) -> dict[str, Any]:
        first_response = getattr(exc, "first_response", None) or exc.details.get(
            "first_response",
            "",
        )
        failed_response = getattr(exc, "failed_response", None) or exc.details.get(
            "failed_response",
            "",
        )
        details: dict[str, Any] = {}
        if first_response:
            details["first_model_output"] = str(first_response)
        if failed_response:
            details["failed_model_output"] = str(failed_response)
        if exc.details.get("last_error_type"):
            details["last_error_type"] = str(exc.details.get("last_error_type", ""))
        if exc.details.get("last_error"):
            details["last_error"] = str(exc.details.get("last_error", ""))
        if exc.details.get("error_type"):
            details["error_type"] = str(exc.details.get("error_type", ""))
        if exc.details.get("api_attempts") is not None:
            details["api_attempts"] = exc.details.get("api_attempts")
        if exc.details.get("model"):
            details["model"] = str(exc.details.get("model", ""))
        return details

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
    def _rank_ordered_initial_file_summaries(
        *,
        summaries: list[ExtractionFileSummary],
        ranking: FileRankingResult,
    ) -> list[ExtractionFileSummary]:
        rank_by_path = {ranked.file_path: ranked.rank for ranked in ranking.files}
        return sorted(
            summaries,
            key=lambda summary: (
                rank_by_path.get(summary.file_path, 10_000),
                summary.file_path,
            ),
        )

    @staticmethod
    def _seed_initial_overview_graph(
        *,
        data_package_name: str,
        ranked_files: list[RankedFile],
        file_summaries: list[ExtractionFileSummary],
        file_previews: list[ExtractionOverviewFilePreview],
    ) -> ExtractionOverview:
        summary_by_path = {summary.file_path: summary for summary in file_summaries}
        rank_by_path = {ranked.file_path: ranked.rank for ranked in ranked_files}
        preview_paths = [preview.file_path for preview in file_previews]
        nodes: list[ExtractionOverviewNode] = [
            ExtractionOverviewNode(
                node_id="package:root",
                label=data_package_name,
                kind="package",
                summary="Package root; directory and file containment is derived from archive paths.",
            )
        ]
        edges: list[ExtractionOverviewEdge] = []
        node_ids = {"package:root"}
        edge_keys: set[tuple[str, str, str]] = set()

        def add_directory(path: str) -> str:
            node_id = f"dir:{path}"
            if node_id not in node_ids:
                nodes.append(
                    ExtractionOverviewNode(
                        node_id=node_id,
                        label=path.rsplit("/", 1)[-1],
                        kind="directory",
                        summary="Directory inferred from package file paths.",
                    )
                )
                node_ids.add(node_id)
            return node_id

        def add_contains(source: str, target: str) -> None:
            key = (source, "contains", target)
            if key in edge_keys:
                return
            edge_keys.add(key)
            edges.append(
                ExtractionOverviewEdge(
                    edge_id=f"{source}-contains-{target}",
                    source=source,
                    target=target,
                    relation="contains",
                    evidence=["path structure"],
                    note="Deterministic package path containment.",
                )
            )

        def add_group(rule: _InitialOverviewGroupRule) -> str:
            if rule.node_id not in node_ids:
                nodes.append(
                    ExtractionOverviewNode(
                        node_id=rule.node_id,
                        label=rule.label,
                        kind="group",
                        summary=rule.summary,
                    )
                )
                node_ids.add(rule.node_id)
            return rule.node_id

        def add_semantic_edge(
            source: str,
            relation: str,
            target: str,
            *,
            evidence: list[str],
            note: str,
        ) -> None:
            key = (source, relation, target)
            if key in edge_keys:
                return
            edge_keys.add(key)
            edges.append(
                ExtractionOverviewEdge(
                    edge_id=f"{source}-{relation}-{target}",
                    source=source,
                    target=target,
                    relation=relation,
                    evidence=evidence,
                    note=note,
                )
            )

        for file_path in preview_paths:
            parts = [part for part in file_path.split("/") if part]
            if not parts:
                continue
            parent_id = "package:root"
            directory_parts: list[str] = []
            for part in parts[:-1]:
                directory_parts.append(part)
                directory_path = "/".join(directory_parts)
                directory_id = add_directory(directory_path)
                add_contains(parent_id, directory_id)
                parent_id = directory_id

            file_node_id = f"file:{file_path}"
            if file_node_id not in node_ids:
                nodes.append(
                    ExtractionOverviewNode(
                        node_id=file_node_id,
                        label=parts[-1],
                        kind="file",
                        file_path=file_path,
                        rank=rank_by_path.get(file_path),
                        summary=OrientationService._seed_file_overview_summary(
                            summary_by_path.get(file_path)
                        ),
                    )
                )
                node_ids.add(file_node_id)
            add_contains(parent_id, file_node_id)

        seeded_group_ids: set[str] = set()
        for file_path in preview_paths:
            summary = summary_by_path.get(file_path)
            if summary is None:
                continue
            file_node_id = f"file:{file_path}"
            matches = OrientationService._initial_overview_group_matches_for_summary(
                summary
            )
            for rule, evidence in matches:
                group_id = add_group(rule)
                seeded_group_ids.add(group_id)
                add_semantic_edge(
                    file_node_id,
                    rule.relation,
                    group_id,
                    evidence=[evidence],
                    note="Deterministic group assignment from validated file summary.",
                )

        inferred_role_edges = (
            (
                "group:method_program",
                "configures",
                "group:acquisition_settings",
                ["method/program logic", "acquisition settings"],
            ),
            (
                "group:instrument_settings",
                "parameterizes",
                "group:acquisition_settings",
                ["instrument settings", "acquisition settings"],
            ),
            (
                "group:acquisition_settings",
                "parameterizes",
                "group:raw_data",
                ["acquisition settings", "raw data"],
            ),
            (
                "group:processing_settings",
                "parameterizes",
                "group:processed_data",
                ["processing settings", "processed data"],
            ),
            (
                "group:processed_data",
                "derives_from",
                "group:raw_data",
                ["processed data", "raw data"],
            ),
            (
                "group:derived_results",
                "derives_from",
                "group:processed_data",
                ["derived results", "processed data"],
            ),
        )
        for source, relation, target, evidence in inferred_role_edges:
            if source in seeded_group_ids and target in seeded_group_ids:
                add_semantic_edge(
                    source,
                    relation,
                    target,
                    evidence=evidence,
                    note="Conservative relation inferred from seeded package roles.",
                )

        return ExtractionOverview(nodes=nodes, edges=edges)

    @staticmethod
    def _seed_file_overview_summary(summary: ExtractionFileSummary | None) -> str:
        if summary is None:
            return ""
        values: list[str] = []
        if summary.explicit_purpose:
            values.append(summary.explicit_purpose)
        if summary.data_format:
            values.append(summary.data_format)
        values.extend(summary.metadata_signals[:3])
        values.extend(summary.instrument_or_software_terms_and_settings[:2])
        values.extend(summary.quantitative_signals[:2])
        seen: set[str] = set()
        compacted: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            compacted.append(normalized)
        return "; ".join(compacted[:5])

    @staticmethod
    def _initial_overview_summary_values(summary: ExtractionFileSummary) -> list[str]:
        values: list[str] = [
            summary.data_format,
            summary.explicit_purpose,
        ]
        values.extend(summary.purpose_evidence)
        values.extend(summary.metadata_signals)
        values.extend(summary.instrument_or_software_terms_and_settings)
        values.extend(summary.quantitative_signals)
        return [" ".join(value.split()) for value in values if value and value.strip()]

    @classmethod
    def _initial_overview_group_matches_for_summary(
        cls,
        summary: ExtractionFileSummary,
    ) -> list[tuple[_InitialOverviewGroupRule, str]]:
        values = cls._initial_overview_summary_values(summary)
        matches: list[tuple[_InitialOverviewGroupRule, str]] = []
        matched_group_ids: set[str] = set()
        for rule in INITIAL_OVERVIEW_GROUP_RULES:
            evidence = cls._first_matching_overview_summary_value(
                values,
                keywords=rule.keywords,
            )
            if evidence is None:
                continue
            matches.append((rule, evidence))
            matched_group_ids.add(rule.node_id)

        if (
            "group:parameter_settings" in matched_group_ids
            and matched_group_ids.intersection(INITIAL_OVERVIEW_SPECIFIC_SETTING_GROUPS)
        ):
            matches = [
                (rule, evidence)
                for rule, evidence in matches
                if rule.node_id != "group:parameter_settings"
            ]
        return matches

    @staticmethod
    def _first_matching_overview_summary_value(
        values: list[str],
        *,
        keywords: tuple[str, ...],
    ) -> str | None:
        for value in values:
            normalized_value = value.casefold()
            if any(
                keyword in normalized_value
                and not OrientationService._overview_keyword_is_negated(
                    normalized_value,
                    keyword,
                )
                for keyword in keywords
            ):
                return value
        return None

    @staticmethod
    def _overview_keyword_is_negated(value: str, keyword: str) -> bool:
        index = value.find(keyword)
        if index < 0:
            return False
        prefix = value[max(0, index - 40) : index]
        return any(
            marker in prefix
            for marker in (
                "not ",
                "no ",
                "without ",
                "rather than ",
                "instead of ",
                "non-",
            )
        )

    @staticmethod
    def _overview_edge_has_meaningful_evidence(edge: ExtractionOverviewEdge) -> bool:
        return any(
            evidence.strip().lower() not in INITIAL_OVERVIEW_WEAK_EDGE_EVIDENCE
            for evidence in edge.evidence
            if evidence.strip()
        )

    @staticmethod
    def _sanitize_initial_overview_graph(
        overview: ExtractionOverview,
        *,
        allowed_file_paths: set[str],
        seed_overview: ExtractionOverview,
        summaries_available: bool = False,
        warnings: list[str],
    ) -> ExtractionOverview:
        filtered_nodes: list[ExtractionOverviewNode] = list(seed_overview.nodes)
        used_node_ids: set[str] = {node.node_id for node in filtered_nodes}
        node_id_map: dict[str, str] = {node.node_id: node.node_id for node in filtered_nodes}
        dropped_file_nodes = 0
        dropped_duplicate_nodes = 0
        for index, node in enumerate(overview.nodes):
            original_node_id = node.node_id
            if node.kind == "file":
                if not node.file_path or node.file_path not in allowed_file_paths:
                    dropped_file_nodes += 1
                    continue
                normalized_node = node.model_copy(
                    update={
                        "node_id": f"file:{node.file_path}",
                        "label": node.label or node.file_path,
                    }
                )
            else:
                normalized_node = node
                if not normalized_node.node_id:
                    normalized_node = normalized_node.model_copy(
                        update={"node_id": f"{normalized_node.kind}:{index}"}
                    )
            if normalized_node.node_id in used_node_ids:
                node_id_map[original_node_id] = normalized_node.node_id
                dropped_duplicate_nodes += 1
                continue
            used_node_ids.add(normalized_node.node_id)
            node_id_map[original_node_id] = normalized_node.node_id
            filtered_nodes.append(normalized_node)

        remapped_ids = {node.node_id for node in filtered_nodes}
        filtered_edges: list[ExtractionOverviewEdge] = list(seed_overview.edges)
        seed_edge_keys = {
            (edge.source, edge.relation, edge.target)
            for edge in seed_overview.edges
        }
        edge_keys = set(seed_edge_keys)
        dropped_dangling_edges = 0
        dropped_contains_edges = 0
        dropped_self_edges = 0
        dropped_weak_evidence_edges = 0
        dropped_duplicate_edges = 0
        for index, edge in enumerate(overview.edges):
            source = node_id_map.get(edge.source, edge.source)
            target = node_id_map.get(edge.target, edge.target)
            if source not in remapped_ids or target not in remapped_ids:
                dropped_dangling_edges += 1
                continue
            if source == target:
                dropped_self_edges += 1
                continue
            edge_key = (source, edge.relation, target)
            if edge.relation == "contains":
                if edge_key not in seed_edge_keys:
                    dropped_contains_edges += 1
                continue
            if not OrientationService._overview_edge_has_meaningful_evidence(edge):
                dropped_weak_evidence_edges += 1
                continue
            if edge_key in edge_keys:
                dropped_duplicate_edges += 1
                continue
            edge_keys.add(edge_key)
            filtered_edges.append(
                edge.model_copy(
                    update={
                        "source": source,
                        "target": target,
                        "edge_id": edge.edge_id
                        or f"{source}-{edge.relation}-{target}-{index}"
                    }
                )
            )
        if dropped_file_nodes:
            warnings.append(
                f"Initial extraction overview referenced {dropped_file_nodes} unknown file node(s); they were removed."
            )
        if dropped_duplicate_nodes:
            warnings.append(
                f"Initial extraction overview repeated {dropped_duplicate_nodes} seeded graph node(s); seeded nodes were kept."
            )
        if dropped_dangling_edges:
            warnings.append(
                f"Initial extraction overview referenced {dropped_dangling_edges} dangling graph edge(s); they were removed."
            )
        if dropped_contains_edges:
            warnings.append(
                f"Initial extraction overview proposed {dropped_contains_edges} non-path containment edge(s); seeded path containment was kept."
            )
        if dropped_self_edges:
            warnings.append(
                f"Initial extraction overview proposed {dropped_self_edges} self-loop edge(s); they were removed."
            )
        if dropped_weak_evidence_edges:
            warnings.append(
                f"Initial extraction overview proposed {dropped_weak_evidence_edges} semantic edge(s) with weak placeholder evidence; they were removed."
            )
        if dropped_duplicate_edges:
            warnings.append(
                f"Initial extraction overview repeated {dropped_duplicate_edges} graph edge(s); duplicates were removed."
            )
        filtered_uncertainties = list(overview.uncertainties)
        if summaries_available:
            before_uncertainty_count = len(filtered_uncertainties)
            filtered_uncertainties = [
                uncertainty
                for uncertainty in filtered_uncertainties
                if not OrientationService._overview_uncertainty_contradicts_summaries(
                    uncertainty
                )
            ]
            dropped_uncertainties = before_uncertainty_count - len(filtered_uncertainties)
            if dropped_uncertainties:
                warnings.append(
                    f"Initial extraction overview made {dropped_uncertainties} uncertainty claim(s) contradicted by available file summaries; they were removed."
                )
        return overview.model_copy(
            update={
                "nodes": filtered_nodes,
                "edges": filtered_edges,
                "uncertainties": filtered_uncertainties,
            }
        )

    @staticmethod
    def _overview_uncertainty_contradicts_summaries(uncertainty: str) -> bool:
        normalized = uncertainty.strip().lower()
        if not normalized:
            return False
        summary_terms = ("summary", "summaries", "per-file")
        unavailable_terms = (
            "no ",
            "not available",
            "unavailable",
            "absent",
            "missing",
            "without summaries",
        )
        return any(term in normalized for term in summary_terms) and any(
            term in normalized for term in unavailable_terms
        )

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
        source_ranked_paths = [
            path for path in source_file_paths if path in ranked_path_set
        ]
        if source_ranked_paths != ranked_paths[: len(source_ranked_paths)]:
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
        ranked_paths = {file.file_path for file in ranking.files}
        summary_paths = {
            summary.file_path
            for summary in summaries
            if summary.status == "summarized"
        }
        if summary_paths != ranked_paths:
            return False
        return True

    async def _generate_initial_dataset_summary(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        warnings: list[str],
        skip_without_runtime: bool = False,
    ) -> str:
        if state.dataset_summary:
            return state.dataset_summary
        if skip_without_runtime and not hasattr(self.ollama_client, "ollama_client"):
            return ""
        assert self.ollama_client is not None
        assert self.output_repository is not None
        try:
            prompt_components = build_dataset_summary_prompt_components(
                data_package_id=data_package_id,
                initial_file_summaries=state.initial_file_summaries,
                ranked_files=state.ranked_files,
            )
            summary_result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=DATASET_SUMMARY_SYSTEM_PROMPT,
                prompt="".join(text for _, text in prompt_components),
                system_components=[
                    ("dataset_summary_system_prompt", DATASET_SUMMARY_SYSTEM_PROMPT),
                ],
                prompt_components=prompt_components,
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("dataset_summary"),
                agent_name="dataset_summary",
                output_type=DatasetSummaryProjection,
                num_ctx=self.ollama_client.max_context_length,
                num_predict=700,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=summary_result,
                agent_name="dataset_summary",
            )
            summary_output = (
                summary_result.output
                if isinstance(summary_result.output, DatasetSummaryProjection)
                else DatasetSummaryProjection.model_validate(summary_result.output)
            )
            dataset_summary = summary_output.summary.strip()
            if not dataset_summary:
                raise ValueError("Dataset summary output was empty.")
            state.dataset_summary = dataset_summary
            self.output_repository.save_dataset_summary(
                workflow_id=data_package_id,
                summary=dataset_summary,
                chat_model=self.ollama_client.chat_model,
                chunking_strategy=state.chunking_strategy,
            )
            self._save_run_state(data_package_id, state)
            return dataset_summary
        except (CompletionError, ValidationError, ValueError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="dataset_summary",
            )
            warnings.append(f"Dataset summary projection failed: {exc}")
            self._save_run_state(data_package_id, state)
            return ""

    async def _build_overview_shallow_projection(
        self,
        *,
        data_package_id: str,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        warnings: list[str],
    ) -> tuple[dict[str, Any], list[ProjectionLedgerRecord]]:
        assert self.ollama_client is not None
        skeleton = shallow_required_skeleton(
            data_package_id=data_package_id,
            fallback_title=self._fallback_title(data_package_id, self._merged_completed_evidence_context(state)),
        )
        distributions = deterministic_grouped_distributions(
            initial_file_summaries=state.initial_file_summaries,
            ranked_files=state.ranked_files,
        )
        fallback_document, fallback_records = shallow_projection_to_dcat_document(
            dataset_level_to_shallow_projection(
                ShallowDatasetLevelProjection.model_validate(skeleton),
                distributions=distributions,
            ),
            data_package_id=data_package_id,
            fallback_title=(skeleton.get("title") or [data_package_id])[0],
        )
        dataset_summary = await self._generate_initial_dataset_summary(
            data_package_id=data_package_id,
            state=state,
            warnings=warnings,
        )
        if not dataset_summary:
            return (
                fallback_document,
                [
                    projection_stage_record(
                        stage="deterministic_distributions",
                        object_kind="DeterministicDistributions",
                        status="projected",
                        reason="Backend created deterministic grouped distributions.",
                        projected_paths=["/dataset_distribution"] if distributions else [],
                    ),
                    *fallback_records,
                ],
            )

        prompt_components = build_dataset_level_projection_prompt_components(
            data_package_id=data_package_id,
            dataset_summary=dataset_summary,
            skeleton=skeleton,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=DATASET_LEVEL_PROJECTION_SYSTEM_PROMPT,
                prompt="".join(text for _, text in prompt_components),
                system_components=[
                    ("dataset_level_projection_system_prompt", DATASET_LEVEL_PROJECTION_SYSTEM_PROMPT),
                ],
                prompt_components=prompt_components,
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("dataset_level_projection"),
                agent_name="dataset_level_projection",
                output_type=ShallowDatasetLevelProjection,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="dataset_level_projection",
            )
            level_projection = (
                result.output
                if isinstance(result.output, ShallowDatasetLevelProjection)
                else ShallowDatasetLevelProjection.model_validate(result.output)
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="dataset_level_projection",
            )
            warnings.append(f"Dataset-level projection failed: {exc}")
            fallback_document_with_summary, fallback_records_with_summary = shallow_projection_to_dcat_document(
                dataset_level_to_shallow_projection(
                    ShallowDatasetLevelProjection.model_validate(skeleton),
                    distributions=distributions,
                ),
                data_package_id=data_package_id,
                fallback_title=(skeleton.get("title") or [data_package_id])[0],
                fallback_description=dataset_summary,
            )
            return (
                fallback_document_with_summary,
                [
                    projection_stage_record(
                        stage="dataset_level_projection",
                        object_kind="DatasetLevelProjection",
                        status="user_edit_required",
                        reason="Dataset-level projection failed; persisted required fallback skeleton with deterministic distributions.",
                        error=str(exc),
                    ),
                    projection_stage_record(
                        stage="deterministic_distributions",
                        object_kind="DeterministicDistributions",
                        status="projected",
                        reason="Backend created deterministic grouped distributions.",
                        projected_paths=["/dataset_distribution"] if distributions else [],
                    ),
                    *fallback_records_with_summary,
                ],
            )

        projection = dataset_level_to_shallow_projection(
            level_projection,
            distributions=distributions,
        )

        document, scaffold_records = shallow_projection_to_dcat_document(
            projection,
            data_package_id=data_package_id,
            fallback_title=self._fallback_title(data_package_id, self._merged_completed_evidence_context(state)),
            fallback_description=dataset_summary,
        )
        validation = self.profile_service.validate_document(
            identifier=state.profile_identifier or "dcat-ap-plus",
            document=document,
        )
        if not validation.valid:
            error = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
            repaired = await self._repair_overview_shallow_projection(
                data_package_id=data_package_id,
                failed_value=projection.model_dump(mode="json"),
                error=ValueError(error),
                validation_schema=validation_schema,
                warnings=warnings,
                state=state,
                distributions=distributions,
                fallback_description=dataset_summary,
            )
            if repaired is not None:
                return repaired
            warnings.append(f"Dataset-level projection failed full profile validation: {error}")
            fallback_document_with_summary, fallback_records_with_summary = shallow_projection_to_dcat_document(
                dataset_level_to_shallow_projection(
                    ShallowDatasetLevelProjection.model_validate(skeleton),
                    distributions=distributions,
                ),
                data_package_id=data_package_id,
                fallback_title=(skeleton.get("title") or [data_package_id])[0],
                fallback_description=dataset_summary,
            )
            return (
                fallback_document_with_summary,
                [
                    projection_stage_record(
                        stage="dataset_level_projection",
                        object_kind="DatasetLevelProjection",
                        status="user_edit_required",
                        reason="Dataset-level projection failed full profile validation; persisted required fallback skeleton with deterministic distributions.",
                        error=error,
                    ),
                    projection_stage_record(
                        stage="deterministic_distributions",
                        object_kind="DeterministicDistributions",
                        status="projected",
                        reason="Backend created deterministic grouped distributions.",
                        projected_paths=["/dataset_distribution"] if distributions else [],
                    ),
                    *fallback_records_with_summary,
                ],
            )

        projected_paths = sorted(f"/{key}" for key in document)
        return (
            document,
            [
                projection_stage_record(
                    stage="dataset_level_projection",
                    object_kind="DatasetLevelProjection",
                    status="projected",
                    reason="Dataset-level projection produced full-profile-valid Dataset draft.",
                    projected_paths=projected_paths,
                ),
                projection_stage_record(
                    stage="deterministic_distributions",
                    object_kind="DeterministicDistributions",
                    status="projected",
                    reason="Backend created deterministic grouped distributions.",
                    projected_paths=["/dataset_distribution"] if distributions else [],
                ),
                *scaffold_records,
            ],
        )

    async def _repair_overview_shallow_projection(
        self,
        *,
        data_package_id: str,
        failed_value: dict[str, Any],
        error: Exception,
        validation_schema: dict[str, Any],
        warnings: list[str],
        state: ExtractionRunState,
        distributions: list[Any],
        fallback_description: str | None = None,
    ) -> tuple[dict[str, Any], list[ProjectionLedgerRecord]] | None:
        assert self.ollama_client is not None
        try:
            result = await repair_structured_output(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                failed_response=json.dumps(failed_value, ensure_ascii=False),
                error=error,
                output_type=ShallowDatasetLevelProjection,
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("dataset_level_projection_repair"),
                agent_name="dataset_level_projection_repair",
                retries=0,
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="dataset_level_projection_repair",
            )
            level_projection = (
                result.output
                if isinstance(result.output, ShallowDatasetLevelProjection)
                else ShallowDatasetLevelProjection.model_validate(result.output)
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="dataset_level_projection_repair",
            )
            warnings.append(f"Dataset-level projection repair failed: {exc}")
            return None

        projection = dataset_level_to_shallow_projection(
            level_projection,
            distributions=distributions,
        )

        document, scaffold_records = shallow_projection_to_dcat_document(
            projection,
            data_package_id=data_package_id,
            fallback_title=self._fallback_title(data_package_id, self._merged_completed_evidence_context(state)),
            fallback_description=fallback_description,
        )
        validation = self.profile_service.validate_document(
            identifier=state.profile_identifier or "dcat-ap-plus",
            document=document,
        )
        if not validation.valid:
            warnings.append(
                "Dataset-level projection repair failed full profile validation: "
                + "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
            )
            return None
        projected_paths = sorted(f"/{key}" for key in document)
        return (
            document,
            [
                overview_projection_record(
                    status="projected",
                    reason="Overview shallow projection produced full-profile-invalid draft; repair succeeded.",
                    projected_paths=projected_paths,
                ),
                overview_projection_repair_record(
                    status="projected",
                    reason="Dataset-level projection repair produced full-profile-valid Dataset draft.",
                    projected_paths=projected_paths,
                ),
                projection_stage_record(
                    stage="deterministic_distributions",
                    object_kind="DeterministicDistributions",
                    status="projected",
                    reason="Backend created deterministic grouped distributions.",
                    projected_paths=["/dataset_distribution"] if distributions else [],
                ),
                *scaffold_records,
            ],
        )

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

    def _persist_initial_extraction_overview_diagnostic(
        self,
        data_package_id: str,
        state: ExtractionRunState,
        *,
        diagnostic: InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None,
    ) -> None:
        if self.output_repository is None:
            return
        self.output_repository.save_initial_extraction_overview_diagnostic(
            workflow_id=data_package_id,
            diagnostic=diagnostic,
            chat_model=state.chat_model,
        )

    def _persist_initial_file_summary_diagnostics(
        self,
        data_package_id: str,
        state: ExtractionRunState,
        *,
        diagnostics: InitialFileSummaryDiagnostics,
    ) -> None:
        if self.output_repository is None:
            return
        self.output_repository.save_initial_file_summary_diagnostics(
            workflow_id=data_package_id,
            diagnostics=diagnostics if diagnostics.records else None,
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

