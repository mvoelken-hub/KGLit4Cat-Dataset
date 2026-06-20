from __future__ import annotations

from app.domain.extraction.description_mining import (
    DESCRIPTION_FACT_MINING_SYSTEM_PROMPT,
    DescriptionMiningArtifact,
    RawDescriptionFacts,
    augment_evidence_context_with_description_facts,
    build_description_mining_prompt,
    collect_dataset_description_sources,
    is_description_derived_path,
    validate_description_facts,
)
from app.services.extraction_shared import *


async def generate_structured(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.generate_structured(*args, **kwargs)


class ProjectionService:
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
        if state.generated_final_draft is not None and state.curated_document is not None:
            progress.generated_final_draft = state.generated_final_draft
            progress.initial_draft_scaffold = state.initial_draft_scaffold
            progress.projection_ledger = state.projection_ledger
            return state.generated_final_draft

        _base_document, scaffold = self._initial_profile_document(
            data_package_id=data_package_id,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
        )
        state.initial_draft_scaffold = scaffold
        progress.initial_draft_scaffold = scaffold

        live_projection_ledger = [
            overview_projection_record(
                status="pending",
                reason="Overview-level shallow projection queued.",
            )
        ]
        state.projection_ledger = list(live_projection_ledger)
        progress.projection_ledger = state.projection_ledger
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)

        live_projection_ledger[0] = overview_projection_record(
            status="running",
            reason="Building overview-level shallow projection.",
        )
        state.projection_ledger = list(live_projection_ledger)
        progress.projection_ledger = state.projection_ledger
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)

        projection, projection_records = await self._build_overview_shallow_projection(
            data_package_id=data_package_id,
            validation_schema=validation_schema,
            state=state,
            warnings=warnings,
        )
        document = remove_null_values(projection)
        state.projection_ledger = projection_records
        state.generated_final_draft = document
        progress.generated_final_draft = document
        progress.projection_ledger = state.projection_ledger
        progress.warnings = list(warnings)
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        state.validation = DraftValidationResult(
            status="valid" if validation.valid else "invalid",
            errors=validation.errors,
            warnings=[],
        )
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        return document

    async def _enrich_draft_with_evidence(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        evidence_context: RoutedEvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> dict[str, Any]:
        document = state.generated_final_draft
        if document is None or self.ollama_client is None:
            return state.generated_final_draft or {}
        if state.generated_initial_draft is None:
            state.generated_initial_draft = self._clone_json_object(document)
            progress.generated_initial_draft = state.generated_initial_draft
        for note in evidence_context.portable_evidence:
            target_path, target_class = route_evidence_note_to_target(note)
            if target_path is None or target_class is None:
                state.projection_ledger.append(
                    self._enrichment_ledger_record(
                        note=note,
                        status="not_projected",
                        reason=f"Evidence note blocked or unmapped (target={target_path}, class={target_class}).",
                        target_path=target_path,
                        target_class=target_class,
                    )
                )
                continue
            contextual_notes = build_context_window_for_note(note, evidence_context)
            excerpt_path = self._parent_excerpt_path(target_path)
            draft_excerpt = self._value_at_json_pointer(document, excerpt_path)
            schema_branch = self._compact_schema_branch_for_target(
                validation_schema=validation_schema,
                target_path=target_path,
            )
            novelty = await self._evaluate_evidence_novelty(
                data_package_id=data_package_id,
                note=note,
                contextual_notes=contextual_notes,
                draft_excerpt=draft_excerpt,
                schema_branch=schema_branch,
                target_path=target_path,
                target_class=target_class,
            )
            if not novelty.is_novel:
                state.projection_ledger.append(
                    self._enrichment_ledger_record(
                        note=note,
                        status="not_projected",
                        reason=f"Not novel: {novelty.reason}",
                        target_path=novelty.corrected_target_path or target_path,
                        target_class=novelty.corrected_target_class or target_class,
                    )
                )
                continue
            target_path = novelty.corrected_target_path or target_path
            target_class = novelty.corrected_target_class or target_class
            instance = await self._build_evidence_instance(
                data_package_id=data_package_id,
                note=note,
                contextual_notes=contextual_notes,
                draft_excerpt=draft_excerpt,
                schema_branch=schema_branch,
                target_path=target_path,
                target_class=target_class,
            )
            if not instance:
                state.projection_ledger.append(
                    self._enrichment_ledger_record(
                        note=note,
                        status="not_projected",
                        reason="Builder failed to produce a valid instance.",
                        target_path=target_path,
                        target_class=target_class,
                    )
                )
                continue
            schema_branch = self._compact_schema_branch_for_target(
                validation_schema=validation_schema,
                target_path=target_path,
            )
            document, record = await self._apply_and_validate_evidence_instance(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                note=note,
                instance=instance,
                target_path=target_path,
                target_class=target_class,
                schema_branch=schema_branch,
            )
            state.projection_ledger.append(record)
            state.generated_final_draft = document
            progress.generated_final_draft = document
            progress.projection_ledger = state.projection_ledger
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)
        state.validation = DraftValidationResult(
            status="valid",
            errors=[],
            warnings=[],
        )
        progress.validation = state.validation
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        return document

    async def _mine_dataset_description(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> RoutedEvidenceContext:
        sources = collect_dataset_description_sources(document)
        source_paths = [source.path for source in sources]
        if not sources:
            self.output_repository.save_description_facts(
                workflow_id=data_package_id,
                artifact=DescriptionMiningArtifact(
                    status="skipped",
                    source_description_paths=[],
                    reason="Dataset-level description is absent or empty.",
                ),
                chat_model=state.chat_model,
                chunking_strategy=state.chunking_strategy,
            )
            return evidence_context

        progress.stage = "description_mining"
        progress.warnings = list(warnings)
        self._update_progress(data_package_id, progress)
        prompt = build_description_mining_prompt(sources)
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=DESCRIPTION_FACT_MINING_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=RawDescriptionFacts,
                system_components=[
                    ("description_fact_mining_system_prompt", DESCRIPTION_FACT_MINING_SYSTEM_PROMPT),
                ],
                prompt_components=[("dataset_description_sources_json", prompt)],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("description_fact_miner"),
                agent_name="description_fact_miner",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="description_fact_miner",
            )
            raw_facts = (
                result.output
                if isinstance(result.output, RawDescriptionFacts)
                else RawDescriptionFacts.model_validate(result.output)
            )
            facts, rejections = validate_description_facts(raw_facts, sources)
            artifact = DescriptionMiningArtifact(
                status="completed",
                source_description_paths=source_paths,
                facts=facts,
                rejected_count=len(rejections),
                rejection_reasons=rejections,
                reason=f"Validated {len(facts)} description fact(s).",
            )
            self.output_repository.save_description_facts(
                workflow_id=data_package_id,
                artifact=artifact,
                chat_model=state.chat_model,
                chunking_strategy=state.chunking_strategy,
            )
            if rejections:
                warnings.append(
                    f"Description mining dropped {len(rejections)} invalid or duplicate fact(s)."
                )
            progress.warnings = list(warnings)
            return augment_evidence_context_with_description_facts(evidence_context, facts)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="description_fact_miner",
            )
            warning = f"Description mining failed; continuing with source evidence only: {exc}"
            warnings.append(warning)
            progress.warnings = list(warnings)
            self.output_repository.save_description_facts(
                workflow_id=data_package_id,
                artifact=DescriptionMiningArtifact(
                    status="failed",
                    source_description_paths=source_paths,
                    reason=str(exc),
                ),
                chat_model=state.chat_model,
                chunking_strategy=state.chunking_strategy,
            )
            return evidence_context

    async def _enrich_draft_with_requirements(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        evidence_context: RoutedEvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        warnings: list[str],
    ) -> dict[str, Any]:
        document = state.generated_final_draft
        if document is None or self.ollama_client is None:
            return state.generated_final_draft or {}
        if state.generated_initial_draft is None:
            state.generated_initial_draft = self._clone_json_object(document)
            progress.generated_initial_draft = state.generated_initial_draft
        document = self._remove_file_like_about_entities(document)
        state.generated_final_draft = document
        progress.generated_final_draft = document
        coverage_evidence_context = await self._mine_dataset_description(
            data_package_id=data_package_id,
            document=document,
            evidence_context=evidence_context,
            state=state,
            progress=progress,
            warnings=warnings,
        )

        progress.stage = "coverage_scoring"
        coverage = compute_coverage_report(document, validation_schema)
        coverage_requirements = list(DCAT_AP_PLUS_COVERAGE_REQUIREMENTS)
        coverage_items = self._coverage_items_from_document(
            requirements=coverage_requirements,
            document=document,
        )
        requirements_by_id = {item.requirement_id: item for item in coverage_requirements}

        for item in coverage_items:
            requirement = requirements_by_id[item.requirement_id]
            selected_evidence, context_window = select_requirement_evidence_packet(
                requirement=requirement,
                assessment=item,
                evidence_context=coverage_evidence_context,
            )
            item.selected_evidence = selected_evidence
            item.context_window = context_window
            self._record_requirement_evidence_query(
                state=state,
                progress=progress,
                requirement=requirement,
                item=item,
                selected_evidence=selected_evidence,
                context_window=context_window,
            )
            if item.requirement_id == "instrument_settings_attributes" and item.applicable:
                document = self._apply_quantitative_evidence_groups(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    document=document,
                    evidence_context=coverage_evidence_context,
                    validation_schema=validation_schema,
                    state=state,
                    progress=progress,
                    requirement=requirement,
                    item=item,
                )
            if self._requirement_target_path_exists(
                document=document,
                target_paths=item.target_paths or requirement.target_paths,
            ):
                item.status = "fulfilled"
                item.quality = 1.0
                item.weighted_score = item.weight
                item.rationale = "Coverage target path exists after deterministic projection."
            if item.status != "missing" or not item.applicable:
                continue
            if not selected_evidence:
                item.patch = RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    reason="No matching evidence packet found for missing requirement.",
                )
                continue
            document, patch_attempt = await self._patch_requirement_gap(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                requirement=requirement,
                item=item,
                validation_schema=validation_schema,
            )
            item.patch = patch_attempt
            state.generated_final_draft = document
            progress.generated_final_draft = document
            if patch_attempt.status == "applied":
                self._record_requirement_patch_ledgers(
                    state=state,
                    progress=progress,
                    requirement=requirement,
                    item=item,
                    patch_attempt=patch_attempt,
                    document=document,
                )
                if self._requirement_target_path_exists(
                    document=document,
                    target_paths=item.target_paths or requirement.target_paths,
                ):
                    item.status = "fulfilled"
                    item.quality = 1.0
                    item.weighted_score = item.weight
            coverage = compute_coverage_report(document, validation_schema)
            source_trace = compute_source_trace_report(
                evidence_context,
                self._used_evidence_ids(state=state),
            )
            report = build_requirement_report(
                schema_valid=state.validation.status == "valid",
                coverage=coverage,
                semantic_requirements=[],
                source_trace=source_trace,
                coverage_patches=coverage_items,
            )
            state.requirement_report = report
            progress.requirement_report = report
            progress.projection_ledger = state.projection_ledger
            progress.field_completion_ledger = state.field_completion_ledger
            progress.evidence_query_ledger = state.evidence_query_ledger
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)

        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        state.validation = DraftValidationResult(
            status="valid" if validation.valid else "invalid",
            errors=validation.errors,
            warnings=[],
        )
        state.generated_final_draft = document
        state.generated_patched_draft = self._clone_json_object(document)
        progress.generated_patched_draft = state.generated_patched_draft
        coverage = compute_coverage_report(document, validation_schema)
        semantic_items = await self._evaluate_semantic_requirements(
            data_package_id=data_package_id,
            document=document,
            evidence_context=evidence_context,
        )
        progress.stage = "semantic_reconstruction"
        document, semantic_reconstructions = await self._reconstruct_semantic_defects(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            semantic_items=semantic_items,
            validation_schema=validation_schema,
            state=state,
            progress=progress,
        )
        state.generated_final_draft = document
        state.generated_reconstructed_draft = self._clone_json_object(document)
        progress.generated_final_draft = document
        progress.generated_reconstructed_draft = state.generated_reconstructed_draft
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=document,
        )
        state.validation = DraftValidationResult(
            status="valid" if validation.valid else "invalid",
            errors=validation.errors,
            warnings=[],
        )
        progress.stage = "semantic_revalidation"
        semantic_items = await self._evaluate_semantic_requirements(
            data_package_id=data_package_id,
            document=document,
            evidence_context=evidence_context,
        )
        coverage = compute_coverage_report(document, validation_schema)
        source_trace = compute_source_trace_report(
            evidence_context,
            self._used_evidence_ids(state=state),
        )
        state.requirement_report = build_requirement_report(
            schema_valid=validation.valid,
            coverage=coverage,
            semantic_requirements=semantic_items,
            source_trace=source_trace,
            coverage_patches=coverage_items,
            semantic_reconstructions=semantic_reconstructions,
        )
        state.document_quality_state = self._build_document_quality_state(
            validation=state.validation,
            requirement_report=state.requirement_report,
            field_completion_ledger=state.field_completion_ledger,
        )
        progress.validation = state.validation
        progress.generated_final_draft = document
        progress.requirement_report = state.requirement_report
        progress.document_quality_state = state.document_quality_state
        progress.projection_ledger = state.projection_ledger
        progress.field_completion_ledger = state.field_completion_ledger
        progress.evidence_query_ledger = state.evidence_query_ledger
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        return document

    async def _reconstruct_semantic_defects(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        semantic_items: list[RequirementReportItem],
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
    ) -> tuple[dict[str, Any], list[SemanticReconstructionRecord]]:
        order = [
            "provenance_context_semantics",
            "technical_agents_semantics",
            "method_plan_semantics",
            "instrument_settings_semantics",
            "dataset_identity_semantics",
            "aboutness_semantics",
        ]
        by_id = {item.requirement_id: item for item in semantic_items}
        records: list[SemanticReconstructionRecord] = []
        current = document
        for requirement_id in order:
            item = by_id.get(requirement_id)
            if item is None:
                continue
            if item.status in {"fulfilled", "not_applicable"}:
                records.append(
                    SemanticReconstructionRecord(
                        requirement_id=requirement_id,
                        status="skipped",
                        target_paths=item.target_paths,
                        reason="Semantic requirement has no defect to reconstruct.",
                    )
                )
                continue
            original = self._clone_json_object(current)
            updated, changed_paths, reason, validation_errors = await self._semantic_reconstruction_update(
                data_package_id=data_package_id,
                document=current,
                item=item,
                requirement=next(req for req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS if req.requirement_id == requirement_id),
                validation_schema=validation_schema,
            )
            if not changed_paths:
                records.append(
                    SemanticReconstructionRecord(
                        requirement_id=requirement_id,
                        status="failed" if validation_errors else "skipped",
                        target_paths=item.target_paths,
                        reason=reason or "No semantic reconstruction was proposed.",
                        validation_errors=validation_errors,
                    )
                )
                continue
            validation = self.profile_service.validate_document(
                identifier=profile_identifier,
                document=updated,
            )
            if not validation.valid:
                records.append(
                    SemanticReconstructionRecord(
                        requirement_id=requirement_id,
                        status="rolled_back",
                        target_paths=item.target_paths,
                        changed_paths=changed_paths,
                        reason=reason or "Semantic reconstruction failed validation.",
                        validation_errors=[issue.message for issue in validation.errors],
                    )
                )
                current = original
                continue
            current = updated
            record = SemanticReconstructionRecord(
                requirement_id=requirement_id,
                status="applied",
                target_paths=item.target_paths,
                changed_paths=changed_paths,
                reason=reason,
            )
            records.append(record)
            self._record_semantic_reconstruction_ledgers(
                data_package_id=data_package_id,
                state=state,
                progress=progress,
                document=current,
                item=item,
                record=record,
            )
        progress.projection_ledger = state.projection_ledger
        progress.field_completion_ledger = state.field_completion_ledger
        return current, records

    async def _semantic_reconstruction_update(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        item: RequirementReportItem,
        requirement: DcatRequirement,
        validation_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], str, list[str]]:
        assert self.ollama_client is not None
        allowed_paths = item.target_paths or requirement.target_paths
        draft_excerpt = {
            path: self._value_at_json_pointer(document, path)
            for path in allowed_paths
        }
        schema_branches = {
            path: self._compact_schema_branch_for_target(
                validation_schema=validation_schema,
                target_path=path,
            )
            for path in allowed_paths
        }
        prompt = build_semantic_reconstruction_prompt(
            requirement=requirement,
            item=item,
            document=document,
            draft_excerpt=draft_excerpt,
            schema_branches=schema_branches,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=SEMANTIC_RECONSTRUCTION_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=SemanticReconstructionPatchResult,
                system_components=[
                    ("semantic_reconstruction_system_prompt", SEMANTIC_RECONSTRUCTION_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("semantic_reconstruction"),
                agent_name="semantic_reconstruction",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="semantic_reconstruction",
            )
            patch = (
                result.output
                if isinstance(result.output, SemanticReconstructionPatchResult)
                else SemanticReconstructionPatchResult.model_validate(result.output)
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="semantic_reconstruction",
            )
            return document, [], f"Semantic reconstruction generation failed: {exc}", [str(exc)]
        if not patch.should_apply or not patch.operations:
            return document, [], patch.reason or "LLM found no safe reconstruction.", []
        if not self._semantic_patch_paths_allowed(patch.operations, allowed_paths):
            return document, [], "LLM reconstruction patch touched a disallowed path.", ["disallowed_patch_path"]
        try:
            updated = self._apply_profile_patch(document, patch.operations)
        except Exception as exc:
            return document, [], f"Semantic reconstruction patch could not be applied: {exc}", [str(exc)]
        return updated, self._semantic_patch_changed_paths(patch.operations), patch.reason, []

    @classmethod
    def _semantic_patch_paths_allowed(cls, operations: list[dict[str, Any]], allowed_paths: list[str]) -> bool:
        return all(
            isinstance(operation, dict)
            and cls._semantic_patch_path_allowed(str(operation.get("path", "")), allowed_paths)
            and (
                "from" not in operation
                or cls._semantic_patch_path_allowed(str(operation.get("from", "")), allowed_paths)
            )
            for operation in operations
        )

    @staticmethod
    def _semantic_patch_path_allowed(path: str, allowed_paths: list[str]) -> bool:
        return path.startswith("/") and any(
            ProjectionService._patch_path_allowed_for_target(path, allowed_path)
            for allowed_path in allowed_paths
        )

    @staticmethod
    def _semantic_patch_changed_paths(operations: list[dict[str, Any]]) -> list[str]:
        return list(dict.fromkeys(str(operation.get("path", "")) for operation in operations if operation.get("path")))

    def _record_semantic_reconstruction_ledgers(
        self,
        *,
        data_package_id: str,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        document: dict[str, Any],
        item: RequirementReportItem,
        record: SemanticReconstructionRecord,
    ) -> None:
        evidence_ids = [
            evidence.evidence_id
            for evidence in list(item.selected_evidence) + list(item.context_window)
            if getattr(evidence, "evidence_id", "")
        ]
        for path in record.changed_paths:
            field_name = self._field_name_from_json_pointer(path)
            value = self._value_at_json_pointer(document, path)
            state.field_completion_ledger.append(
                FieldCompletionLedgerRecord(
                    json_path=path,
                    field_name=field_name,
                    generated_value=value,
                    source_evidence=list(dict.fromkeys(evidence_ids)),
                    validation_status="valid",
                    enrichment_status="grounded" if evidence_ids else "not_grounded",
                    issue_categories=[],
                    edit_needed_reason=f"{item.requirement_id}: {record.reason}",
                )
            )
        object_identifier = f"semantic_reconstruction:{item.requirement_id}:{sha1('|'.join(record.changed_paths).encode('utf-8')).hexdigest()[:12]}"
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier=object_identifier,
                object_kind="SemanticReconstruction",
                source_evidence=evidence_ids[0] if evidence_ids else None,
                evidence_note_identifiers=list(dict.fromkeys(evidence_ids)),
                status="projected",
                projected_paths=record.changed_paths,
                target_path=record.changed_paths[0] if record.changed_paths else None,
                target_class=None,
                planner_status=item.requirement_id,
                planner_reason=record.reason,
                evidence_quality={"requirement_id": item.requirement_id},
                merge_status="applied",
                reason=record.reason,
            )
        )
        progress.field_completion_ledger = state.field_completion_ledger
        progress.projection_ledger = state.projection_ledger

    @classmethod
    def _coverage_items_from_document(
        cls,
        *,
        requirements: list[DcatRequirement],
        document: dict[str, Any],
    ) -> list[RequirementReportItem]:
        items: list[RequirementReportItem] = []
        for requirement in requirements:
            fulfilled = cls._requirement_target_path_exists(
                document=document,
                target_paths=requirement.target_paths,
            )
            items.append(
                RequirementReportItem(
                    requirement_id=requirement.requirement_id,
                    label=requirement.label,
                    weight=requirement.weight,
                    status="fulfilled" if fulfilled else "missing",
                    applicable=True,
                    quality=1.0 if fulfilled else 0.0,
                    weighted_score=requirement.weight if fulfilled else 0.0,
                    rationale=(
                        "Coverage target path exists."
                        if fulfilled
                        else "Coverage target path is missing."
                    ),
                    target_paths=list(requirement.target_paths),
                    evidence_search_hints=list(requirement.evidence_hints),
                )
            )
        return items

    async def _evaluate_semantic_requirements(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
    ) -> list[RequirementReportItem]:
        items: list[RequirementReportItem] = []
        for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS:
            seed_item = RequirementReportItem(
                requirement_id=requirement.requirement_id,
                label=requirement.label,
                weight=requirement.weight,
                status="missing",
                applicable=True,
                quality=0.0,
                weighted_score=0.0,
                target_paths=list(requirement.target_paths),
                evidence_search_hints=list(requirement.evidence_hints),
            )
            selected_evidence, context_window = select_requirement_evidence_packet(
                requirement=requirement,
                assessment=seed_item,
                evidence_context=evidence_context,
            )
            draft_excerpt = {
                path: self._value_at_json_pointer(document, path)
                for path in requirement.target_paths
            }
            evaluation = await self._evaluate_dcat_requirements(
                data_package_id=data_package_id,
                document=draft_excerpt,
                requirements=[requirement],
                selected_evidence=selected_evidence,
                context_window=context_window,
            )
            evaluated = report_items_from_evaluation(
                requirements=[requirement],
                evaluation=evaluation,
            )[0]
            evaluated.selected_evidence = selected_evidence
            evaluated.context_window = context_window
            items.append(evaluated)
        score_requirement_items(items)
        return items

    @staticmethod
    def _used_evidence_ids(*, state: ExtractionRunState) -> list[str]:
        ids: list[str] = []
        for record in state.field_completion_ledger:
            ids.extend(record.source_evidence or [])
        for record in state.projection_ledger:
            if record.status == "projected":
                ids.extend(record.evidence_note_identifiers or [])
        return list(dict.fromkeys(item for item in ids if item))

    @staticmethod
    def _build_document_quality_state(
        *,
        validation: DraftValidationResult,
        requirement_report: Any,
        field_completion_ledger: list[FieldCompletionLedgerRecord],
    ) -> DocumentQualityState:
        blocking: list[QualityIssue] = []
        warnings: list[QualityIssue] = []
        schema_valid = validation.status == "valid"
        if not schema_valid:
            blocking.extend(
                QualityIssue(
                    code="schema_invalid",
                    severity="blocking",
                    message=issue.message,
                    path=issue.path,
                )
                for issue in validation.errors
            )
        profile_conformant = schema_valid
        coverage_score = None
        semantic_requirements_score = None
        source_trace_score = None
        evidence_grounded = None
        if requirement_report is not None:
            coverage_score = requirement_report.coverage_score
            semantic_requirements_score = requirement_report.semantic_requirements_score
            source_trace_score = requirement_report.source_trace_score
            unmet = [
                item
                for item in requirement_report.coverage_patches
                if item.applicable and item.status not in {"fulfilled", "not_applicable"}
            ]
            if unmet:
                profile_conformant = False
                blocking.extend(
                    QualityIssue(
                        code="requirement_unmet",
                        severity="blocking",
                        message=f"{item.label} is {item.status}.",
                        requirement_id=item.requirement_id,
                    )
                    for item in unmet
                )
            evidence_grounded = requirement_report.source_trace.used_evidence_count > 0
        else:
            profile_conformant = None
        warnings.append(
            QualityIssue(
                code="semantic_vocabulary_validation_not_run",
                severity="info",
                message=(
                    "Ontology term-role, quantity-kind, unit, and vocabulary validation "
                    "are not part of this profile-stage run."
                ),
            )
        )
        warnings.append(
            QualityIssue(
                code="operational_fair_checks_not_run",
                severity="info",
                message=(
                    "Publisher, license, contact, resolvable identifier, access URL, "
                    "download URL, and FAIR publication checks are out of scope for this stage."
                ),
            )
        )
        unsupported_field_issues = [
            record
            for record in field_completion_ledger
            if record.validation_status == "invalid" or record.issue_categories
        ]
        blocking.extend(
            QualityIssue(
                code="field_completion_issue",
                severity="blocking",
                message=record.edit_needed_reason or "Field completion has validation issues.",
                path=record.json_path,
            )
            for record in unsupported_field_issues
        )
        return DocumentQualityState(
            schema_valid=schema_valid,
            profile_conformant=profile_conformant,
            evidence_grounded=evidence_grounded,
            semantic_valid=None,
            coverage_score=coverage_score,
            semantic_requirements_score=semantic_requirements_score,
            source_trace_score=source_trace_score,
            operational_access_score=None,
            fair_assessment=None,
            blocking_issues=blocking,
            warnings=warnings,
        )

    @staticmethod
    def _record_requirement_evidence_query(
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        requirement: DcatRequirement,
        item: Any,
        selected_evidence: list[Any],
        context_window: list[Any],
    ) -> None:
        selected_ids = [
            evidence.evidence_id
            for evidence in selected_evidence
            if getattr(evidence, "evidence_id", "")
        ]
        result_ids: list[str] = []
        for evidence in list(selected_evidence) + list(context_window):
            evidence_id = getattr(evidence, "evidence_id", "")
            if evidence_id and evidence_id not in result_ids:
                result_ids.append(evidence_id)
        target_paths = getattr(item, "target_paths", None) or requirement.target_paths
        target_path = target_paths[0] if target_paths else ""
        hints = getattr(item, "evidence_search_hints", None) or requirement.evidence_hints
        query_payload = {
            "text": " ".join(hints),
            "target_paths": target_paths,
            "expected_target_class": (
                getattr(item, "expected_target_class", None)
                or requirement.expected_target_class
            ),
            "routes": ["portable", "contextual"],
            "limit": {"selected": 5, "context": 12},
        }
        query_fingerprint = sha1(
            json.dumps(
                {
                    "requirement_id": requirement.requirement_id,
                    "target_paths": target_paths,
                    "hints": hints,
                    "target_class": query_payload["expected_target_class"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        query_id = f"requirement_query:{requirement.requirement_id}:{query_fingerprint}"
        rejected_result_reasons = {
            evidence_id: "context_window_only"
            for evidence_id in result_ids
            if evidence_id not in selected_ids
        }
        ranking_explanation = [
            "Ranked portable and contextual evidence by requirement search hints.",
            "Boosted candidates matching expected DCAT-AP+ target class.",
        ]
        if requirement.expected_target_class == "QuantitativeAttribute":
            ranking_explanation.append(
                "Portable evidence with a numeric value and quantity label is grouped for quantitative projection."
            )
        state.evidence_query_ledger = [
            record
            for record in state.evidence_query_ledger
            if record.query_id != query_id
        ]
        state.evidence_query_ledger.append(
            EvidenceQueryLedgerEntry(
                query_id=query_id,
                requirement_id=requirement.requirement_id,
                target_path=target_path,
                query=query_payload,
                result_evidence_ids=result_ids,
                selected_evidence_ids=selected_ids,
                rejected_result_reasons=rejected_result_reasons,
                ranking_explanation=ranking_explanation,
            )
        )
        progress.evidence_query_ledger = state.evidence_query_ledger

    def _record_requirement_patch_ledgers(
        self,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        requirement: DcatRequirement,
        item: Any,
        patch_attempt: RequirementPatchAttempt,
        document: dict[str, Any],
    ) -> None:
        target_path = patch_attempt.target_path or ""
        actual_path = self._actual_requirement_patch_path(document=document, target_path=target_path)
        selected_evidence = list(getattr(item, "selected_evidence", []) or [])
        evidence_ids = [
            evidence.evidence_id
            for evidence in selected_evidence
            if getattr(evidence, "evidence_id", "")
        ]
        origins = self._evidence_origins(selected_evidence)
        generated_value = self._value_at_json_pointer(document, actual_path) if actual_path else None
        field_name = self._field_name_from_json_pointer(actual_path)
        ledger_key = (actual_path, field_name)
        state.field_completion_ledger = [
            record
            for record in state.field_completion_ledger
            if (record.json_path, record.field_name) != ledger_key
        ]
        state.field_completion_ledger.append(
            FieldCompletionLedgerRecord(
                json_path=actual_path,
                field_name=field_name,
                generated_value=generated_value,
                source_evidence=evidence_ids,
                validation_status="valid",
                enrichment_status="grounded",
                issue_categories=[],
                edit_needed_reason=(
                    f"{requirement.requirement_id}: {patch_attempt.reason} "
                    f"Evidence origin: {', '.join(origins)}."
                ),
            )
        )
        object_identifier = f"requirement_patch:{requirement.requirement_id}:{actual_path}"
        state.projection_ledger = [
            record
            for record in state.projection_ledger
            if record.object_identifier != object_identifier
        ]
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier=object_identifier,
                object_kind="RequirementPatch",
                source_evidence=evidence_ids[0] if evidence_ids else None,
                evidence_note_identifiers=evidence_ids,
                status="projected",
                projected_paths=[actual_path] if actual_path else [],
                target_path=actual_path or None,
                target_class=patch_attempt.target_class,
                planner_status=requirement.requirement_id,
                planner_reason=patch_attempt.reason,
                evidence_quality={
                    "requirement_id": requirement.requirement_id,
                    "selected_evidence_ids": evidence_ids,
                    "evidence_origins": origins,
                    "construction_strategy": self._requirement_patch_construction_strategy(requirement),
                },
                merge_status="applied",
                reason=patch_attempt.reason,
            )
        )
        progress.field_completion_ledger = state.field_completion_ledger
        progress.projection_ledger = state.projection_ledger

    def _apply_quantitative_evidence_groups(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        requirement: DcatRequirement,
        item: Any,
    ) -> dict[str, Any]:
        groups = self._quantitative_evidence_groups(evidence_context.portable_evidence)
        if not groups:
            return document
        projected = 0
        skipped = 0
        selected_notes: list[RequirementEvidenceItem] = []
        for group in groups:
            selected_notes.extend(self._requirement_evidence_items_for_group(group))
            target_path, target_class, document = self._quantitative_target_path_for_group(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                group=group,
                validation_schema=validation_schema,
            )
            instance = self._quantitative_attribute_instance_from_group(group)
            if target_path is None:
                skipped += 1
                self._record_quantitative_group_skip(
                    state=state,
                    progress=progress,
                    group=group,
                    reason="target_unresolved: no schema-valid owner path supports has_quantitative_attribute.",
                )
                continue
            duplicate_reason = self._duplicate_requirement_patch_reason(
                document=document,
                target_path=target_path,
                instance=instance,
            )
            if duplicate_reason:
                skipped += 1
                self._record_quantitative_group_skip(
                    state=state,
                    progress=progress,
                    group=group,
                    target_path=target_path,
                    reason=duplicate_reason,
                )
                continue
            original = self._clone_json_object(document)
            schema_branch = self._compact_schema_branch_for_target(
                validation_schema=validation_schema,
                target_path=target_path,
            )
            try:
                document = apply_evidence_instance(
                    document=document,
                    target_path=target_path,
                    instance=instance,
                    data_package_id=data_package_id,
                    target_schema=schema_branch,
                )
                actual_path = self._actual_requirement_patch_path(
                    document=document,
                    target_path=target_path,
                )
                appended_index = int(actual_path.rsplit("/", 1)[-1]) if actual_path else None
                self._strip_requirement_patch_forbidden_fields(
                    document=document,
                    target_path=target_path,
                    target_class="QuantitativeAttribute",
                    appended_index=appended_index,
                )
            except (ValueError, TypeError) as exc:
                skipped += 1
                document = original
                self._record_quantitative_group_skip(
                    state=state,
                    progress=progress,
                    group=group,
                    target_path=target_path,
                    reason=f"projection_failed: {exc}",
                )
                continue
            validation = self.profile_service.validate_document(
                identifier=profile_identifier,
                document=document,
            )
            if not validation.valid:
                skipped += 1
                document = original
                self._record_quantitative_group_skip(
                    state=state,
                    progress=progress,
                    group=group,
                    target_path=target_path,
                    reason="validation_failed: "
                    + "; ".join(issue.message for issue in validation.errors),
                )
                continue
            projected += 1
            self._record_quantitative_group_projection(
                state=state,
                progress=progress,
                group=group,
                document=document,
                target_path=target_path,
                target_class=target_class,
            )
        if selected_notes:
            item.selected_evidence = selected_notes
        item.status = "fulfilled"
        item.quality = 1.0
        item.weighted_score = item.weight
        item.rationale = (
            f"Checked {len(groups)} portable numeric evidence group(s): "
            f"{projected} projected, {skipped} skipped with ledger reasons."
        )
        item.patch = RequirementPatchAttempt(
            attempted=True,
            status="applied" if projected else "not_attempted",
            target_class="QuantitativeAttribute",
            reason=item.rationale,
        )
        state.generated_final_draft = document
        progress.generated_final_draft = document
        return document

    @classmethod
    def _quantitative_evidence_groups(
        cls,
        evidence_items: list[Any],
    ) -> list[_QuantitativeEvidenceGroup]:
        groups: dict[str, _QuantitativeEvidenceGroup] = {}
        for note in evidence_items:
            candidate = cls._quantitative_group_candidate(note)
            if candidate is None:
                continue
            group_id, label, value, unit = candidate
            existing = groups.get(group_id)
            if existing is None:
                groups[group_id] = _QuantitativeEvidenceGroup(
                    group_id=group_id,
                    label=label,
                    value=value,
                    unit=unit,
                    notes=[note],
                )
            else:
                existing.notes.append(note)
        return cls._cap_repeated_quantitative_groups(list(groups.values()))

    @classmethod
    def _cap_repeated_quantitative_groups(
        cls,
        groups: list[_QuantitativeEvidenceGroup],
        *,
        per_skeleton_limit: int = 5,
    ) -> list[_QuantitativeEvidenceGroup]:
        kept: list[_QuantitativeEvidenceGroup] = []
        counts: dict[str, int] = {}
        for group in groups:
            key = cls._quantitative_group_skeleton(group)
            count = counts.get(key, 0)
            if count >= per_skeleton_limit:
                continue
            counts[key] = count + 1
            kept.append(group)
        return kept

    @staticmethod
    def _quantitative_group_skeleton(group: _QuantitativeEvidenceGroup) -> str:
        file_path = str(getattr(group.notes[0], "file_path", "") or "") if group.notes else ""
        label = re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", " ", group.label.lower())
        label = re.sub(r"\b(intensity|value|observed|detected|contains)\b", " ", label)
        label = re.sub(r"[^a-z]+", " ", label).strip()
        return "|".join([file_path, label, (group.unit or "").lower()])

    @classmethod
    def _quantitative_group_candidate(
        cls,
        note: Any,
    ) -> tuple[str, str, float, str | None] | None:
        claim = str(getattr(note, "claim", "") or "")
        evidence_text = str(getattr(note, "evidence_text", "") or "")
        text = f"{claim} {evidence_text}".strip()
        if not cls._quantitative_note_category_allowed(note, claim):
            return None
        match = re.search(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_])", text)
        if not match:
            return None
        value = float(match.group(0))
        unit = cls._unit_after_number(text, match.end())
        label = cls._quantity_label_from_text(claim or evidence_text, match.group(0), unit)
        if not label:
            return None
        if cls._quantitative_label_is_noise(label, claim, evidence_text, unit):
            return None
        normalized_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        value_key = ("%f" % value).rstrip("0").rstrip(".")
        file_path = str(getattr(note, "file_path", "") or "")
        start_idx = int(getattr(note, "start_idx", 0) or 0)
        neighborhood = start_idx // 500
        key = "|".join([normalized_label, value_key, (unit or "").lower(), file_path, str(neighborhood)])
        return sha1(key.encode("utf-8")).hexdigest()[:12], label, value, unit

    @staticmethod
    def _quantitative_note_category_allowed(note: Any, claim: str) -> bool:
        category = str(getattr(note, "category", "") or "")
        if category == "instrument_signal":
            return True
        return False

    @staticmethod
    def _quantitative_label_is_noise(label: str, claim: str, evidence_text: str, unit: str | None = None) -> bool:
        lowered = f"{label} {claim} {evidence_text}".lower()
        quantity_like = re.search(
            r"\b(threshold|calibration|unit|scale|scan|average|frequency|temperature|duration|delay|gain|power|resolution|voltage|current|pressure|speed|rate|limit|offset|phase|width|height|depth|length|distance|angle|time|count|number|size|mass|weight|volume|concentration|dose|flow)\b",
            lowered,
        )
        configurable_like = re.search(
            r"\b(threshold|calibration|unit|scale|scan|average|frequency|temperature|duration|delay|gain|power|resolution|voltage|current|pressure|speed|rate|limit|offset|phase|width|height|depth|length|distance|angle|time|size|mass|weight|volume|concentration|dose|flow|setting|configured|configuration|parameter)\b",
            lowered,
        )
        setting_like = quantity_like or configurable_like
        primary_data_like = re.search(
            r"\b(observed|measured|recorded|row|table|minimum|maximum|range|bound|extremum|extrema|axis|data points?)\b",
            lowered,
        )
        if primary_data_like and not configurable_like:
            return True
        qualitative_like = re.search(r"\b(name|category|class|type|status|mode|flag|label|title|code|identifier|id)\b", lowered)
        placeholder_like = re.search(r"\b(unspecified|unknown|none|null|not set|unset|default|placeholder|n/?a)\b", lowered)
        if qualitative_like or placeholder_like:
            return True
        if not quantity_like and not unit:
            return True
        if re.search(r"\b(identifier|id|file|dataset|data package|data path|software version|parameter file|classified|recommended)\b", lowered):
            return True
        if len(re.findall(r"\d", label)) > 2 and not setting_like:
            return True
        if re.search(r"[\\/]|[A-Za-z0-9]+_[A-Za-z0-9]+", evidence_text):
            return True
        return False

    @staticmethod
    def _unit_after_number(text: str, number_end: int) -> str | None:
        match = re.match(r"\s*([A-Za-z%°µμ][A-Za-z0-9%°µμ/_-]{0,10})", text[number_end:])
        if not match:
            return None
        token = match.group(1).strip()
        if token.lower() in {"is", "and", "used", "data"}:
            return None
        if token.isalpha() and len(token) > 4 and token.lower() == token:
            return None
        return token

    @staticmethod
    def _quantity_label_from_text(text: str, number_text: str, unit: str | None) -> str | None:
        cleaned = re.sub(re.escape(number_text), " ", text, count=1)
        if unit:
            cleaned = re.sub(rf"\b{re.escape(unit)}\b", " ", cleaned, count=1)
        cleaned = re.sub(r"[_=#:/\\|]+", " ", cleaned)
        cleaned = re.sub(r"\b(the|a|an|is|are|was|were|equals?|value|set|to|of|in|at)\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"[^A-Za-z0-9%°µμ -]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        if len(cleaned) < 3:
            return None
        return cleaned[:120]

    def _quantitative_target_path_for_group(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        group: _QuantitativeEvidenceGroup,
        validation_schema: dict[str, Any],
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        owners = self._quantitative_owner_paths(document, validation_schema)
        if owners:
            preferred_owner = self._preferred_quantitative_owner_class(group)
            if preferred_owner:
                owners = sorted(owners, key=lambda owner: 0 if owner[1] == preferred_owner else 1)
            return f"{owners[0][0]}/has_quantitative_attribute/-", owners[0][1], document
        created = self._create_quantitative_owner_if_reachable(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            group=group,
            validation_schema=validation_schema,
        )
        if created is None:
            return None, None, document
        owner_path, owner_class, document = created
        return f"{owner_path}/has_quantitative_attribute/-", owner_class, document

    def _quantitative_owner_paths(
        self,
        document: dict[str, Any],
        validation_schema: dict[str, Any],
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for key, owner_class in (
            ("was_generated_by", "DataGeneratingActivity"),
            ("is_about_activity", "EvaluatedActivity"),
            ("is_about_entity", "EvaluatedEntity"),
        ):
            for index, value in enumerate(document.get(key) or []):
                if isinstance(value, dict) and self._class_supports_quantitative_attribute(validation_schema, owner_class):
                    candidates.append((f"/{key}/{index}", owner_class))
        for activity_index, activity in enumerate(document.get("was_generated_by") or []):
            if not isinstance(activity, dict):
                continue
            for agent_index, agent in enumerate(activity.get("carried_out_by") or []):
                owner_class = self._quantitative_owner_class_from_object(agent)
                if self._class_supports_quantitative_attribute(validation_schema, owner_class):
                    candidates.append((f"/was_generated_by/{activity_index}/carried_out_by/{agent_index}", owner_class))
        return candidates

    def _create_quantitative_owner_if_reachable(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        group: _QuantitativeEvidenceGroup,
        validation_schema: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]] | None:
        preferred_owner = self._preferred_quantitative_owner_class(group)
        if preferred_owner in {"Device", "Software"} and self._class_supports_quantitative_attribute(validation_schema, preferred_owner):
            document = self._ensure_generation_activity_owner(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                validation_schema=validation_schema,
            )
            if document.get("was_generated_by"):
                target_path = "/was_generated_by/0/carried_out_by/-"
                if self._schema_for_json_pointer(validation_schema, target_path[:-2]):
                    owner = {
                        "title": preferred_owner,
                        "description": f"{preferred_owner} inferred from quantitative evidence: {group.label}.",
                        "rdf_type": {"title": preferred_owner},
                    }
                    original = self._clone_json_object(document)
                    try:
                        updated = apply_evidence_instance(
                            document=document,
                            target_path=target_path,
                            instance=owner,
                            data_package_id=data_package_id,
                            target_schema=self._compact_schema_branch_for_target(
                                validation_schema=validation_schema,
                                target_path=target_path,
                            ),
                        )
                    except ValueError:
                        updated = None
                    if updated is not None:
                        validation = self.profile_service.validate_document(
                            identifier=profile_identifier,
                            document=updated,
                        )
                        if validation.valid:
                            index = len(updated["was_generated_by"][0].get("carried_out_by") or []) - 1
                            return f"/was_generated_by/0/carried_out_by/{index}", preferred_owner, updated
                    document.clear()
                    document.update(original)
        if self._class_supports_quantitative_attribute(validation_schema, "DataGeneratingActivity"):
            updated = self._ensure_generation_activity_owner(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                validation_schema=validation_schema,
            )
            if updated.get("was_generated_by"):
                return f"/was_generated_by/{len(updated.get('was_generated_by') or []) - 1}", "DataGeneratingActivity", updated
        return None

    def _ensure_generation_activity_owner(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        validation_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if document.get("was_generated_by"):
            return document
        if not self._class_supports_quantitative_attribute(validation_schema, "DataGeneratingActivity"):
            return document
        original = self._clone_json_object(document)
        try:
            updated = apply_evidence_instance(
                document=document,
                target_path="/was_generated_by/-",
                instance={
                    "title": "Data generating activity",
                    "description": "Activity inferred from quantitative evidence.",
                },
                data_package_id=data_package_id,
                target_schema=self._compact_schema_branch_for_target(
                    validation_schema=validation_schema,
                    target_path="/was_generated_by/-",
                ),
            )
        except ValueError:
            return original
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        return updated if validation.valid else original

    @staticmethod
    def _preferred_quantitative_owner_class(group: _QuantitativeEvidenceGroup) -> str | None:
        text = " ".join(
            [group.label]
            + [str(getattr(note, "claim", "") or "") for note in group.notes]
            + [str(getattr(note, "evidence_text", "") or "") for note in group.notes]
        ).lower()
        if "software" in text:
            return "Software"
        if "device" in text:
            return "Device"
        return None

    @staticmethod
    def _quantitative_owner_class_from_object(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False).lower() if isinstance(value, dict) else ""
        if "software" in text:
            return "Software"
        if "device" in text:
            return "Device"
        return "AgenticEntity"

    @classmethod
    def _class_supports_quantitative_attribute(
        cls,
        validation_schema: dict[str, Any],
        class_name: str,
    ) -> bool:
        defs = validation_schema.get("$defs") if isinstance(validation_schema, dict) else None
        if not isinstance(defs, dict):
            return False
        schema = cls._resolve_schema_node(defs.get(class_name, {}), validation_schema)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        return isinstance(properties, dict) and "has_quantitative_attribute" in properties

    @staticmethod
    def _quantitative_attribute_instance_from_group(
        group: _QuantitativeEvidenceGroup,
    ) -> dict[str, Any]:
        title = group.label[:1].upper() + group.label[1:]
        description = f"{title}: {group.value:g}"
        if group.unit:
            description = f"{description} {group.unit}"
        instance: dict[str, Any] = {
            "title": title,
            "description": description,
            "value": group.value,
            "has_quantity_type": group.label,
        }
        if group.unit:
            instance["unit"] = group.unit
        return instance

    def _record_quantitative_group_projection(
        self,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        group: _QuantitativeEvidenceGroup,
        document: dict[str, Any],
        target_path: str,
        target_class: str | None,
    ) -> None:
        actual_path = self._actual_requirement_patch_path(document=document, target_path=target_path)
        evidence_ids = self._evidence_ids_for_group(group)
        origins = self._evidence_origins(group.notes)
        generated_value = self._value_at_json_pointer(document, actual_path) if actual_path else None
        state.field_completion_ledger = [
            record
            for record in state.field_completion_ledger
            if record.json_path != actual_path
        ]
        state.field_completion_ledger.append(
            FieldCompletionLedgerRecord(
                json_path=actual_path,
                field_name="has_quantitative_attribute",
                generated_value=generated_value,
                source_evidence=evidence_ids,
                validation_status="valid",
                enrichment_status="grounded",
                issue_categories=[],
                edit_needed_reason=(
                    "instrument_settings_attributes: quantitative evidence group projected. "
                    f"Evidence origin: {', '.join(origins)}."
                ),
            )
        )
        object_identifier = f"quantitative_group:{group.group_id}:{actual_path}"
        state.projection_ledger = [
            record
            for record in state.projection_ledger
            if record.object_identifier != object_identifier
        ]
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier=object_identifier,
                object_kind="QuantitativeAttribute",
                source_evidence=evidence_ids[0] if evidence_ids else None,
                evidence_note_identifiers=evidence_ids,
                status="projected",
                projected_paths=[actual_path] if actual_path else [],
                target_path=actual_path or None,
                target_class=target_class or "QuantitativeAttribute",
                planner_status="instrument_settings_attributes",
                planner_reason="Portable numeric evidence group projected.",
                evidence_quality={
                    "quantity_label": group.label,
                    "value": group.value,
                    "unit": group.unit,
                    "evidence_ids": evidence_ids,
                    "evidence_origins": origins,
                },
                merge_status="applied",
                reason="Projected schema-valid quantitative attribute.",
            )
        )
        progress.field_completion_ledger = state.field_completion_ledger
        progress.projection_ledger = state.projection_ledger

    def _record_quantitative_group_skip(
        self,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        group: _QuantitativeEvidenceGroup,
        reason: str,
        target_path: str | None = None,
    ) -> None:
        evidence_ids = self._evidence_ids_for_group(group)
        origins = self._evidence_origins(group.notes)
        object_identifier = f"quantitative_group:{group.group_id}:skip"
        state.projection_ledger = [
            record
            for record in state.projection_ledger
            if record.object_identifier != object_identifier
        ]
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier=object_identifier,
                object_kind="QuantitativeAttribute",
                source_evidence=evidence_ids[0] if evidence_ids else None,
                evidence_note_identifiers=evidence_ids,
                status="not_projected",
                projected_paths=[],
                target_path=target_path,
                target_class="QuantitativeAttribute",
                planner_status="instrument_settings_attributes",
                planner_reason=reason,
                evidence_quality={
                    "quantity_label": group.label,
                    "value": group.value,
                    "unit": group.unit,
                    "evidence_ids": evidence_ids,
                    "evidence_origins": origins,
                },
                merge_status="skipped",
                reason=reason,
            )
        )
        progress.projection_ledger = state.projection_ledger

    @staticmethod
    def _evidence_origins(evidence_items: list[Any]) -> list[str]:
        origins = [
            "draft_description"
            if is_description_derived_path(str(getattr(item, "file_path", "") or ""))
            else "source_file"
            for item in evidence_items
        ]
        return list(dict.fromkeys(origins)) or ["unknown"]

    @staticmethod
    def _evidence_ids_for_group(group: _QuantitativeEvidenceGroup) -> list[str]:
        ids: list[str] = []
        for note in group.notes:
            evidence_id = getattr(note, "evidence_id", "") or stable_evidence_id(note)
            if evidence_id and evidence_id not in ids:
                ids.append(evidence_id)
        return ids

    @classmethod
    def _requirement_evidence_items_for_group(
        cls,
        group: _QuantitativeEvidenceGroup,
    ) -> list[RequirementEvidenceItem]:
        items: list[RequirementEvidenceItem] = []
        for note in group.notes:
            items.append(
                RequirementEvidenceItem(
                    evidence_id=getattr(note, "evidence_id", "") or stable_evidence_id(note),
                    candidate_id=str(getattr(note, "candidate_id", "") or ""),
                    category=str(getattr(note, "category", "") or ""),
                    claim=str(getattr(note, "claim", "") or ""),
                    evidence_text=str(getattr(note, "evidence_text", "") or ""),
                    file_path=str(getattr(note, "file_path", "") or ""),
                    start_idx=int(getattr(note, "start_idx", 0) or 0),
                    end_idx=int(getattr(note, "end_idx", 0) or 0),
                )
            )
        return items

    @classmethod
    def _actual_requirement_patch_path(cls, *, document: dict[str, Any], target_path: str) -> str:
        if not target_path.endswith("/-"):
            return target_path
        parent_path = target_path[:-2]
        value = cls._value_at_json_pointer(document, parent_path)
        if isinstance(value, list) and value:
            return f"{parent_path}/{len(value) - 1}"
        return parent_path

    @classmethod
    def _requirement_target_path_exists(cls, *, document: dict[str, Any], target_paths: list[str]) -> bool:
        concrete_paths = [path for path in target_paths if path and path != "/"]
        if not concrete_paths:
            return True
        for path in concrete_paths:
            if path.endswith("/-"):
                value = cls._value_at_json_pointer(document, path[:-2])
                if isinstance(value, list) and value:
                    return True
                continue
            value = cls._value_at_json_pointer(document, path)
            if not cls._is_missing_value(value):
                return True
        return False

    @staticmethod
    def _field_name_from_json_pointer(path: str) -> str:
        parts = [part for part in path.split("/") if part]
        if not parts:
            return ""
        if parts[-1].isdigit() and len(parts) > 1:
            return parts[-2]
        return parts[-1]

    @staticmethod
    def _requirement_patch_construction_strategy(requirement: DcatRequirement) -> str:
        if requirement.expected_target_class == "QuantitativeAttribute":
            return "deterministic_quantitative_constructor_or_schema_sanitized_patch"
        if requirement.expected_target_class == "Plan":
            return "schema_sanitized_plan_patch"
        return "schema_sanitized_requirement_patch"

    @classmethod
    def _remove_file_like_about_entities(cls, document: dict[str, Any]) -> dict[str, Any]:
        entities = document.get("is_about_entity")
        if not isinstance(entities, list):
            return document
        filtered = [
            entity
            for entity in entities
            if not (isinstance(entity, dict) and cls._is_file_like_about_entity(entity))
        ]
        if len(filtered) != len(entities):
            document = cls._clone_json_object(document)
            if filtered:
                document["is_about_entity"] = filtered
            else:
                document.pop("is_about_entity", None)
        return document

    @staticmethod
    def _is_file_like_about_entity(entity: dict[str, Any]) -> bool:
        text_parts: list[str] = []
        for key in ("id", "title", "description"):
            value = entity.get(key)
            if isinstance(value, list):
                text_parts.extend(str(part) for part in value)
            elif value is not None:
                text_parts.append(str(value))
        text = " ".join(text_parts).lower()
        file_terms = (
            " file",
            "parameter",
            "resource",
            "distribution",
            "configuration",
            "settings",
        )
        return any(term in text for term in file_terms)

    async def _evaluate_dcat_requirements(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        requirements: list[DcatRequirement],
        selected_evidence: list[RequirementEvidenceItem] | None = None,
        context_window: list[RequirementEvidenceItem] | None = None,
    ) -> RequirementEvaluation:
        assert self.ollama_client is not None
        prompt = build_requirement_evaluation_prompt(
            document=document,
            requirements=requirements,
            selected_evidence=selected_evidence,
            context_window=context_window,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=REQUIREMENT_EVALUATOR_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=RequirementEvaluation,
                system_components=[
                    ("requirement_evaluator_system_prompt", REQUIREMENT_EVALUATOR_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("metadata_completeness_evaluator"),
                agent_name="metadata_completeness_evaluator",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="metadata_completeness_evaluator",
            )
            if isinstance(result.output, RequirementEvaluation):
                evaluation = result.output
            else:
                evaluation = RequirementEvaluation.model_validate(result.output)
            return normalized_requirement_evaluation(
                requirements=requirements,
                evaluation=evaluation,
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="metadata_completeness_evaluator",
            )
            return RequirementEvaluation()

    async def _patch_requirement_gap(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        requirement: DcatRequirement,
        item: Any,
        validation_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], RequirementPatchAttempt]:
        assert self.ollama_client is not None
        target_path = (item.target_paths or requirement.target_paths or [""])[0]
        target_class = requirement.expected_target_class or item.requirement_id
        if not target_path:
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    reason="Requirement has no target path for patching.",
                ),
            )
        schema_branch = self._compact_schema_branch_for_target(
            validation_schema=validation_schema,
            target_path=target_path,
        )
        draft_excerpt = self._value_at_json_pointer(document, self._parent_excerpt_path(target_path))
        prompt = build_requirement_patch_prompt(
            requirement=requirement,
            assessment=assessment_for_report_item(item),
            selected_evidence=item.selected_evidence,
            context_window=item.context_window,
            draft_excerpt=draft_excerpt,
            schema_branch=schema_branch,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=REQUIREMENT_PATCH_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=RequirementPatchResult,
                system_components=[
                    ("requirement_patch_system_prompt", REQUIREMENT_PATCH_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("metadata_requirement_patcher"),
                agent_name="metadata_requirement_patcher",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="metadata_requirement_patcher",
            )
            patch = result.output if isinstance(result.output, RequirementPatchResult) else RequirementPatchResult.model_validate(result.output)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="metadata_requirement_patcher",
            )
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=target_path,
                    target_class=target_class,
                    reason=f"Requirement patch generation failed: {exc}",
                ),
            )
        if (not patch.should_patch or not patch.instance) and (
            target_class == "QuantitativeAttribute"
            or any("/has_quantitative_attribute/" in path for path in (item.target_paths or requirement.target_paths))
        ):
            deterministic_instance = self._quantitative_attribute_from_evidence(item)
            if deterministic_instance:
                patch = RequirementPatchResult(
                    should_patch=True,
                    target_path=(item.target_paths or requirement.target_paths)[0],
                    target_class="QuantitativeAttribute",
                    instance=deterministic_instance,
                    rationale=(
                        "Deterministic quantitative constructor used selected measurement evidence "
                        "after patch model declined to build an instance."
                    ),
                )
        if not patch.should_patch or not patch.instance:
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=patch.target_path or target_path,
                    target_class=patch.target_class or target_class,
                    reason=patch.rationale or "Patch model found insufficient evidence.",
                ),
            )
        allowed_target_paths = set(item.target_paths or requirement.target_paths)
        target_path = patch.target_path
        if target_path not in allowed_target_paths:
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=target_path,
                    target_class=patch.target_class or target_class,
                    reason=f"Patch target path is not allowed for requirement: {target_path}",
                ),
            )
        target_class = patch.target_class or target_class
        patch.instance = self._sanitize_requirement_patch_instance(
            instance=patch.instance,
            target_class=target_class,
            target_path=target_path,
            item=item,
        )
        duplicate_reason = self._duplicate_requirement_patch_reason(
            document=document,
            target_path=target_path,
            instance=patch.instance,
        )
        if duplicate_reason:
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=target_path,
                    target_class=target_class,
                    reason=duplicate_reason,
                ),
            )
        schema_branch = self._compact_schema_branch_for_target(
            validation_schema=validation_schema,
            target_path=target_path,
        )
        original = self._clone_json_object(document)
        try:
            appended_index = None
            if target_path.endswith("/-"):
                existing_array = self._value_at_json_pointer(document, target_path[:-2])
                appended_index = len(existing_array) if isinstance(existing_array, list) else 0
            updated = apply_evidence_instance(
                document=document,
                target_path=target_path,
                instance=patch.instance,
                data_package_id=data_package_id,
                target_schema=schema_branch,
            )
            self._strip_requirement_patch_forbidden_fields(
                document=updated,
                target_path=target_path,
                target_class=target_class,
                appended_index=appended_index,
            )
        except ValueError as exc:
            return (
                original,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=target_path,
                    target_class=target_class,
                    reason=f"Could not apply requirement patch: {exc}",
                ),
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        if validation.valid:
            return (
                updated,
                RequirementPatchAttempt(
                    attempted=True,
                    status="applied",
                    target_path=target_path,
                    target_class=target_class,
                    reason=patch.rationale or "Requirement patch applied and validated.",
                ),
            )
        return (
            original,
            RequirementPatchAttempt(
                attempted=True,
                status="rolled_back",
                target_path=target_path,
                target_class=target_class,
                validation_errors=[issue.message for issue in validation.errors],
                reason="Requirement patch failed profile validation; restored previous draft.",
            ),
        )

    @classmethod
    def _strip_requirement_patch_forbidden_fields(
        cls,
        *,
        document: dict[str, Any],
        target_path: str,
        target_class: str,
        appended_index: int | None,
    ) -> None:
        actual_path = target_path
        if target_path.endswith("/-") and appended_index is not None:
            actual_path = f"{target_path[:-2]}/{appended_index}"
        value = cls._value_at_json_pointer(document, actual_path)
        if not isinstance(value, dict):
            return
        if target_class == "Plan" or actual_path.endswith("/realized_plan"):
            value.pop("id", None)
            value.pop("identifier", None)
            value.pop("was_generated_by", None)
            return
        if target_class == "QuantitativeAttribute" or "/has_quantitative_attribute/" in actual_path:
            value.pop("id", None)
            value.pop("source", None)
            value.pop("type", None)
            value.pop("rdf_type", None)

    @classmethod
    def _sanitize_requirement_patch_instance(
        cls,
        *,
        instance: dict[str, Any],
        target_class: str,
        target_path: str,
        item: Any,
    ) -> dict[str, Any]:
        if target_class == "Plan" or target_path.endswith("/realized_plan"):
            return cls._sanitize_plan_patch_instance(instance)
        if target_class == "QuantitativeAttribute" or "/has_quantitative_attribute/" in target_path:
            return cls._sanitize_quantitative_attribute_patch_instance(instance, item)
        return instance

    @staticmethod
    def _sanitize_plan_patch_instance(instance: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        title = instance.get("title")
        if isinstance(title, list):
            title = " ".join(str(part) for part in title if str(part).strip())
        if title:
            sanitized["title"] = str(title)
        description = instance.get("description")
        if isinstance(description, list):
            description = " ".join(str(part) for part in description if str(part).strip())
        if description:
            sanitized["description"] = str(description)
        for field in ("type", "rdf_type"):
            value = instance.get(field)
            if isinstance(value, dict):
                sanitized[field] = {
                    key: value[key]
                    for key in ("id", "title")
                    if key in value and value[key]
                }
            elif isinstance(value, str) and value.strip():
                sanitized[field] = {"title": value.strip()}
        return sanitized

    @classmethod
    def _sanitize_quantitative_attribute_patch_instance(
        cls,
        instance: dict[str, Any],
        item: Any,
    ) -> dict[str, Any]:
        deterministic = cls._quantitative_attribute_from_evidence(item)
        if deterministic:
            return deterministic
        evidence_text = " ".join(
            str(getattr(entry, "claim", "")) + " " + str(getattr(entry, "evidence_text", ""))
            for entry in list(getattr(item, "selected_evidence", []) or [])
        )
        sanitized: dict[str, Any] = {}
        title = instance.get("title")
        if isinstance(title, list):
            title = " ".join(str(part) for part in title if str(part).strip())
        if title:
            sanitized["title"] = str(title)
        description = instance.get("description")
        if isinstance(description, list):
            description = " ".join(str(part) for part in description if str(part).strip())
        if description:
            sanitized["description"] = str(description)
        value = cls._first_number(instance.get("value"))
        if value is None:
            value = cls._first_number(evidence_text)
        if value is not None:
            sanitized["value"] = value
        quantity_type = instance.get("has_quantity_type")
        if isinstance(quantity_type, dict):
            quantity_type = quantity_type.get("title") or quantity_type.get("id")
        if not isinstance(quantity_type, str) or not quantity_type.strip():
            quantity_type = cls._infer_quantity_type(" ".join([str(title or ""), str(description or ""), evidence_text]))
        if quantity_type:
            sanitized["has_quantity_type"] = str(quantity_type).strip()
        unit = instance.get("unit")
        if isinstance(unit, dict):
            unit = unit.get("title") or unit.get("id")
        if not isinstance(unit, str) or not unit.strip():
            unit = cls._infer_quantity_unit(
                " ".join([str(title or ""), str(description or ""), str(instance.get("value") or ""), evidence_text])
            )
        if unit:
            sanitized["unit"] = str(unit).strip()
        return sanitized

    @classmethod
    def _quantitative_attribute_from_evidence(cls, item: Any) -> dict[str, Any] | None:
        evidence_items = list(getattr(item, "selected_evidence", []) or [])
        if not evidence_items:
            return None
        ranked = sorted(
            evidence_items,
            key=lambda entry: cls._quantitative_evidence_rank(entry),
            reverse=True,
        )
        for entry in ranked:
            text = f"{getattr(entry, 'claim', '')} {getattr(entry, 'evidence_text', '')}"
            value = cls._first_number(text)
            quantity_type = cls._infer_quantity_type(text)
            unit = cls._infer_quantity_unit(text)
            if value is None or not quantity_type:
                continue
            title = cls._quantitative_attribute_title(text, quantity_type)
            description = cls._quantitative_attribute_description(text, quantity_type, unit)
            attribute: dict[str, Any] = {
                "title": title,
                "description": description,
                "value": value,
                "has_quantity_type": quantity_type,
            }
            if unit:
                attribute["unit"] = unit
            return attribute
        return None

    @staticmethod
    def _quantitative_evidence_rank(entry: Any) -> tuple[int, int, int]:
        text = f"{getattr(entry, 'claim', '')} {getattr(entry, 'evidence_text', '')}".lower()
        has_number = int(bool(re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)))
        observed_value = int("=" in text)
        instrument_signal = int(str(getattr(entry, "category", "")) == "instrument_signal")
        return (has_number, observed_value, instrument_signal)

    @staticmethod
    def _quantitative_attribute_title(text: str, quantity_type: str) -> str:
        return quantity_type[:1].upper() + quantity_type[1:]

    @staticmethod
    def _quantitative_attribute_description(text: str, quantity_type: str, unit: str | None) -> str:
        unit_suffix = f" in {unit}" if unit else ""
        return f"Evidence-grounded {quantity_type}{unit_suffix}."

    @staticmethod
    def _first_number(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
            if match:
                return float(match.group(0))
        return None

    @staticmethod
    def _infer_quantity_type(text: str) -> str | None:
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if match:
            return ProjectionService._quantity_label_from_text(
                text,
                match.group(0),
                ProjectionService._unit_after_number(text, match.end()),
            )
        return "measured quantity"

    @staticmethod
    def _infer_quantity_unit(text: str) -> str | None:
        lowered = text.lower()
        unit_hints = [
            ("mhz", "MHz"),
            (" hz", "Hz"),
            ("kelvin", "K"),
            (" k", "K"),
            ("milliseconds", "ms"),
            (" ms", "ms"),
            ("ppm", "ppm"),
            ("degree", "degree"),
            ("points", "points"),
        ]
        for needle, unit in unit_hints:
            if needle in lowered:
                return unit
        return None

    @classmethod
    def _duplicate_requirement_patch_reason(
        cls,
        *,
        document: dict[str, Any],
        target_path: str,
        instance: dict[str, Any],
    ) -> str | None:
        if not target_path.endswith("/-"):
            existing = cls._value_at_json_pointer(document, target_path)
            if isinstance(existing, dict) and cls._instances_semantically_equal(existing, instance):
                return f"Requirement patch duplicates existing object at {target_path}."
            return None
        parent_path = target_path[:-2]
        existing_items = cls._value_at_json_pointer(document, parent_path)
        if not isinstance(existing_items, list):
            return None
        for existing in existing_items:
            if isinstance(existing, dict) and cls._instances_semantically_equal(existing, instance):
                return f"Requirement patch duplicates existing object at {parent_path}."
        return None

    @classmethod
    def _instances_semantically_equal(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_id = left.get("id")
        right_id = right.get("id")
        if left_id and right_id and str(left_id).strip() == str(right_id).strip():
            return True
        left_sig = cls._instance_semantic_signature(left)
        right_sig = cls._instance_semantic_signature(right)
        return bool(left_sig and left_sig == right_sig)

    @staticmethod
    def _instance_semantic_signature(instance: dict[str, Any]) -> tuple[str, str] | None:
        title = instance.get("title")
        if isinstance(title, list):
            title_value = " ".join(str(part) for part in title)
        else:
            title_value = str(title or "")
        type_value = instance.get("type") or instance.get("rdf_type")
        if isinstance(type_value, dict):
            type_text = str(type_value.get("id") or type_value.get("title") or "")
        else:
            type_text = str(type_value or "")
        normalized_title = re.sub(r"\s+", " ", title_value).strip().lower()
        normalized_type = re.sub(r"\s+", " ", type_text).strip().lower()
        if not normalized_title:
            return None
        return (normalized_title, normalized_type)

    @staticmethod
    def _parent_excerpt_path(target_path: str) -> str:
        if target_path.endswith("/-"):
            return target_path[:-2]
        return target_path

    @classmethod
    def _compact_schema_branch_for_target(
        cls,
        validation_schema: dict[str, Any],
        target_path: str,
    ) -> dict[str, Any]:
        if target_path.endswith("/-"):
            schema_path = target_path[:-2]
        else:
            schema_path = target_path
        schema = cls._schema_for_json_pointer(validation_schema, schema_path)
        if isinstance(schema.get("items"), dict):
            schema = cls._resolve_schema_node(schema["items"], validation_schema)
        if not isinstance(schema, dict):
            return {}
        return cls._schema_shell_value(schema, validation_schema, depth=0, required_only=False) or {}

    async def _evaluate_evidence_novelty(
        self,
        *,
        data_package_id: str,
        note: EvidenceCandidate,
        contextual_notes: list[EvidenceCandidate],
        draft_excerpt: Any,
        schema_branch: dict[str, Any],
        target_path: str,
        target_class: str,
    ) -> EvidenceNoveltyDecision:
        assert self.ollama_client is not None
        prompt = build_novelty_evaluator_prompt(
            note=note,
            contextual_notes=contextual_notes,
            draft_excerpt=draft_excerpt,
            schema_branch=schema_branch,
            target_path=target_path,
            target_class=target_class,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EVIDENCE_NOVELTY_EVALUATOR_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=EvidenceNoveltyDecision,
                system_components=[
                    ("evidence_novelty_evaluator_system_prompt", EVIDENCE_NOVELTY_EVALUATOR_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("evidence_novelty_evaluator"),
                agent_name="evidence_novelty_evaluator",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="evidence_novelty_evaluator",
            )
            if isinstance(result.output, EvidenceNoveltyDecision):
                return result.output
            return EvidenceNoveltyDecision.model_validate(result.output)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="evidence_novelty_evaluator",
            )
            return EvidenceNoveltyDecision(
                is_novel=False,
                reason=f"Novelty evaluation failed: {exc}",
            )

    async def _build_evidence_instance(
        self,
        *,
        data_package_id: str,
        note: EvidenceCandidate,
        contextual_notes: list[EvidenceCandidate],
        draft_excerpt: Any,
        schema_branch: dict[str, Any],
        target_path: str,
        target_class: str,
    ) -> dict[str, Any] | None:
        assert self.ollama_client is not None
        model = builder_output_model_for_target(target_class, schema_branch)
        prompt = build_instance_builder_prompt(
            target_path=target_path,
            target_class=target_class,
            schema_branch=schema_branch,
            draft_excerpt=draft_excerpt,
            note=note,
            contextual_notes=contextual_notes,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EVIDENCE_INSTANCE_BUILDER_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=model,
                system_components=[
                    ("evidence_instance_builder_system_prompt", EVIDENCE_INSTANCE_BUILDER_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("evidence_instance_builder"),
                agent_name="evidence_instance_builder",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="evidence_instance_builder",
            )
            output = result.output
            if hasattr(output, "model_dump"):
                return output.model_dump(mode="json", exclude_none=True)
            return dict(output)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="evidence_instance_builder",
            )
            return None

    async def _repair_evidence_instance(
        self,
        *,
        data_package_id: str,
        instance: dict[str, Any],
        validation_errors: list[ProfileValidationIssue],
        schema_branch: dict[str, Any],
        target_path: str,
        target_class: str,
    ) -> dict[str, Any] | None:
        assert self.ollama_client is not None
        model = builder_output_model_for_target(target_class, schema_branch)
        error_messages = [str(e.message) for e in validation_errors]
        prompt = build_instance_repair_prompt(
            target_path=target_path,
            target_class=target_class,
            schema_branch=schema_branch,
            instance=instance,
            validation_errors=error_messages,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=EVIDENCE_INSTANCE_REPAIR_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=model,
                system_components=[
                    ("evidence_instance_repair_system_prompt", EVIDENCE_INSTANCE_REPAIR_SYSTEM_PROMPT),
                ],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("evidence_instance_repair"),
                agent_name="evidence_instance_repair",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="evidence_instance_repair",
            )
            output = result.output
            if hasattr(output, "model_dump"):
                return output.model_dump(mode="json", exclude_none=True)
            return dict(output)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="evidence_instance_repair",
            )
            return None

    async def _apply_and_validate_evidence_instance(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        note: EvidenceCandidate,
        instance: dict[str, Any],
        target_path: str,
        target_class: str,
        schema_branch: dict[str, Any],
    ) -> tuple[dict[str, Any], ProjectionLedgerRecord]:
        original = self._clone_json_object(document)
        try:
            updated = apply_evidence_instance(
                document=document,
                target_path=target_path,
                instance=instance,
                data_package_id=data_package_id,
                target_schema=schema_branch,
            )
        except ValueError as exc:
            return (
                original,
                self._enrichment_ledger_record(
                    note=note,
                    status="not_projected",
                    reason=f"Could not apply instance to {target_path}: {exc}",
                    target_path=target_path,
                    target_class=target_class,
                ),
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        if validation.valid:
            return (
                updated,
                self._enrichment_ledger_record(
                    note=note,
                    status="projected",
                    reason=f"Enrichment instance applied and validated at {target_path}.",
                    target_path=target_path,
                    target_class=target_class,
                    projected_paths=[target_path],
                ),
            )
        repaired = await self._repair_evidence_instance(
            data_package_id=data_package_id,
            instance=instance,
            validation_errors=validation.errors,
            schema_branch=schema_branch,
            target_path=target_path,
            target_class=target_class,
        )
        if repaired is None:
            return (
                original,
                self._enrichment_ledger_record(
                    note=note,
                    status="not_projected",
                    reason=f"Instance invalid; repair attempt failed. Validation errors: {[e.message for e in validation.errors]}",
                    target_path=target_path,
                    target_class=target_class,
                ),
            )
        try:
            updated = apply_evidence_instance(
                document=original,
                target_path=target_path,
                instance=repaired,
                data_package_id=data_package_id,
                target_schema=schema_branch,
            )
        except ValueError as exc:
            return (
                original,
                self._enrichment_ledger_record(
                    note=note,
                    status="not_projected",
                    reason=f"Repaired instance could not be applied: {exc}",
                    target_path=target_path,
                    target_class=target_class,
                ),
            )
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        if validation.valid:
            return (
                updated,
                self._enrichment_ledger_record(
                    note=note,
                    status="projected",
                    reason="Enrichment instance repaired and validated.",
                    target_path=target_path,
                    target_class=target_class,
                    projected_paths=[target_path],
                ),
            )
        return (
            original,
            self._enrichment_ledger_record(
                note=note,
                status="not_projected",
                reason=f"Repaired instance still invalid; restored previous draft. Errors: {[e.message for e in validation.errors]}",
                target_path=target_path,
                target_class=target_class,
            ),
        )

    @staticmethod
    def _enrichment_ledger_record(
        note: EvidenceCandidate,
        status: ProjectionLedgerStatus,
        reason: str,
        target_path: str | None,
        target_class: str | None,
        projected_paths: list[str] | None = None,
    ) -> ProjectionLedgerRecord:
        return ProjectionLedgerRecord(
            object_identifier=note.candidate_id or "unknown-note",
            object_kind=note.category,
            source_evidence=note.evidence_text,
            evidence_note_identifiers=[note.candidate_id],
            status=status,
            projected_paths=projected_paths or [],
            target_path=target_path,
            target_class=target_class,
            planner_status="evidence_enrichment",
            reason=reason,
        )


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
    def _projection_identifier_for_evidence_note(note: EvidenceCandidate) -> str:
        file_path = note.file_path or "unknown-file"
        line_span = f"{note.start_idx}-{note.end_idx}"
        note_id = note.candidate_id or "unnamed-note"
        return f"{file_path}#{line_span}#{note_id}"

    @staticmethod
    def _projection_record_from_patch_result(
        *,
        note: EvidenceCandidate,
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
    def _evidence_quality_summary(notes: list[EvidenceCandidate]) -> dict[str, Any]:
        return {
            "note_count": len(notes),
            "routes": {"portable_evidence": len(notes)},
        }

    @classmethod
    def _projection_groups_for_evidence(
        cls,
        evidence_context: EvidenceContext,
        *,
        max_group_size: int = 6,
    ) -> list[_EvidenceProjectionGroup]:
        buckets: dict[tuple[str, str, str, str], list[EvidenceCandidate]] = {}
        for note in evidence_context.candidates:
            if not cls._note_has_curatable_profile_signal(note):
                continue
            target_hint, target_class_hint = cls._target_hint_for_evidence_note(note)
            if not target_hint:
                continue
            family = cls._evidence_note_family(note.candidate_id)
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
    def _target_hint_for_evidence_note(cls, note: EvidenceCandidate) -> tuple[str, str | None]:
        text = f"{note.candidate_id} {note.category} {note.claim} {note.evidence_text}".lower()
        if note.category == "measurement_signal":
            return "", None
        if note.category in {"agent_signal", "instrument_signal"} or cls._note_has_device_signal(note):
            return "/was_generated_by/0/carried_out_by/-", "AgenticEntity"
        if note.category == "surrounding_signal" and any(term in text for term in ("origin", "owner", "creator", "author", "team", "laboratory")):
            return "/creator/0", "Agent"
        if note.category == "activity_signal" or any(term in text for term in ("experiment", "acquisition", "generation", "workflow")):
            return "/was_generated_by/0", "DataGeneratingActivity"
        if note.category == "method_signal":
            return "/was_generated_by/0/realized_plan", "Plan"
        if note.category == "resource_signal" or any(term in text for term in ("format", "file", "distribution", "download", "access")):
            return "/dataset_distribution/0", "Distribution"
        if any(term in text for term in ("dataset name", "title", "name")):
            return "/title", None
        if any(term in text for term in ("date", "timestamp", "modified", "modification")):
            return "/modification_date", None
        if any(term in text for term in ("type", "category", "class")):
            return "/type/0", "Concept"
        if note.category == "resource_signal":
            return "/dataset_distribution/0", "Distribution"
        return "/description", None

    @classmethod
    def _note_has_device_signal(cls, note: EvidenceCandidate) -> bool:
        text = cls._note_search_text(note)
        return any(
            term in text
            for term in (
                "instrument",
                "device",
                "equipment",
                "sensor",
                "apparatus",
            )
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
        notes: list[EvidenceCandidate],
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
                default_title="Data generation activity",
            )
        if target_path == "/was_generated_by/0/carried_out_by/-":
            return cls._fallback_agentic_entity_value(notes)
        if target_path == "/is_about_activity/0":
            return cls._fallback_activity_value(
                current_value,
                notes,
                default_title="Data acquisition activity",
            )
        if target_path == "/is_about_entity/0":
            return cls._fallback_entity_value(current_value, notes)
        if target_path == "/type/0":
            return cls._fallback_concept_value(current_value, notes)
        return None

    @classmethod
    def _fallback_agentic_entity_value(cls, notes: list[EvidenceCandidate]) -> dict[str, Any] | None:
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
    def _device_title_for_notes(notes: list[EvidenceCandidate]) -> str | None:
        for note in notes:
            text = f"{note.evidence_text}\n{note.claim}"
            match = re.search(
                r"(?:instrument|device|equipment|sensor|apparatus)\s*(?:used\s*)?(?:is|:|=)\s*<?([^>\r\n;]+)>?",
                text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def _profile_target_unsuitable_reason(
        cls,
        *,
        target_path: str,
        notes: list[EvidenceCandidate],
    ) -> str | None:
        if not notes:
            return "No evidence notes were available for profile projection."
        if target_path in {"/description", "/keyword"}:
            useful_keywords = cls._profile_keywords_for_notes(notes)
            if not useful_keywords and all(
                note.category in {"method_signal", "measurement_signal", "resource_signal", "instrument_signal"}
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
        notes: list[EvidenceCandidate],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        names: list[str] = []
        for note in notes:
            key, value = cls._assignment_from_note(note)
            if key and key.lower() in {"origin", "owner", "author", "creator"} and value:
                names.append(value)
        if not names:
            return None
        value = cls._clone_json_object(current_value)
        value["name"] = cls._merge_unique_strings(value.get("name", []), names)
        return value

    @classmethod
    def _fallback_distribution_value(
        cls,
        current_value: Any,
        notes: list[EvidenceCandidate],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        keywords = cls._profile_keywords_for_notes(notes)
        file_like = any(
            term in cls._note_search_text(note)
            for note in notes
            for term in ("file", "format", "distribution", "download", "archive")
        )
        if not file_like and not keywords:
            return None
        value = cls._clone_json_object(current_value)
        title = "Primary dataset distribution"
        value["title"] = cls._merge_unique_strings(value.get("title", []), [title])
        description_parts = cls._profile_observation_sentences(notes, max_count=2)
        if not description_parts:
            description_parts = ["Dataset files contain data and associated metadata."]
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
        notes: list[EvidenceCandidate],
        *,
        default_title: str,
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        if not any(note.category in {"activity_signal", "instrument_signal"} for note in notes):
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
        notes: list[EvidenceCandidate],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        if not any(note.category == "activity_signal" for note in notes):
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
        notes: list[EvidenceCandidate],
    ) -> Any | None:
        if not isinstance(current_value, dict):
            return None
        labels = cls._profile_keywords_for_notes(notes)
        labels = cls._dedupe_strings(labels)
        if not labels:
            return None
        value = cls._clone_json_object(current_value)
        value["preferred_label"] = cls._merge_unique_strings(value.get("preferred_label", []), labels[:3])
        return value

    @classmethod
    def _description_target_worthy(cls, notes: list[EvidenceCandidate]) -> bool:
        if not notes:
            return False
        category_set = {note.category for note in notes}
        if category_set <= {"method_signal", "measurement_signal", "resource_signal", "instrument_signal"}:
            return False
        return any(
            term in cls._note_search_text(note)
            for note in notes
            for term in ("dataset", "sample", "contains", "study", "experiment")
        )

    @classmethod
    def _evidence_group_has_profile_signal(cls, notes: list[EvidenceCandidate]) -> bool:
        if not notes:
            return False
        return any(bool(note.claim.strip()) for note in notes)

    @classmethod
    def _is_low_level_parameter_note(cls, note: EvidenceCandidate) -> bool:
        text = cls._note_search_text(note)
        key, _value = cls._assignment_from_note(note)
        normalized_key = key.lower().strip("$") if key else ""
        if normalized_key and re.fullmatch(r"[a-z]{1,4}\d{1,4}[a-z0-9_]*", normalized_key):
            return True
        if re.search(r"\b[A-Z][A-Z0-9_]{1,16}\s+(?:parameter|setting)\b", note.claim):
            return True
        if re.search(r"\bparameter\s+[A-Z][A-Z0-9_]{1,16}\b", note.claim):
            return True
        low_level_observation_terms = (
            "parameter is set",
            "parameter set",
            "configuration settings",
            "network configuration",
            "ethernet",
            "tcp/ip",
            "routing",
            "checksum",
            "local path",
        )
        if any(term in text for term in low_level_observation_terms):
            return True
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
        return cls._is_low_level_profile_text(text) or "parameter structure" in lower

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
        lower = normalized.lower()
        return any(
            term in lower
            for term in (
                "dataset",
                "experiment",
                "method",
                "sample",
                "measurement",
                "analysis",
                "workflow",
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
                "instrument",
                "sample",
                "method",
                "metadata",
                "measurement",
                "experiment",
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
            "routing",
            "blanking",
            "configuration settings",
            "parameter values",
            "parameter file",
            "local path",
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
    def _title_value_from_note(cls, note: EvidenceCandidate) -> str | None:
        key, value = cls._assignment_from_note(note)
        if key and key.lower().strip("$") == "title":
            return cls._clean_profile_title_text(value or "")
        text = f"{note.claim or ''}\n{note.evidence_text or ''}"
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
    def _profile_keywords_for_notes(cls, notes: list[EvidenceCandidate]) -> list[str]:
        keywords: list[str] = []
        for note in notes:
            if cls._is_low_level_parameter_note(note):
                continue
            for term in ("dataset", "experiment", "method", "measurement", "sample", "workflow", "analysis"):
                if term in cls._note_search_text(note):
                    keywords.append(term)
        return cls._dedupe_strings(keywords)

    @classmethod
    def _profile_observation_sentences(
        cls,
        notes: list[EvidenceCandidate],
        *,
        max_count: int,
    ) -> list[str]:
        sentences: list[str] = []
        for note in notes:
            if cls._is_low_level_parameter_note(note):
                continue
            text = (note.claim or "").strip()
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
    def _qualitative_attributes_for_notes(cls, notes: list[EvidenceCandidate]) -> list[dict[str, str]]:
        attributes: list[dict[str, str]] = []
        allowed_keys = {"origin", "owner", "author", "creator", "instrument", "device", "sample", "method"}
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
    def _assignment_from_note(cls, note: EvidenceCandidate) -> tuple[str | None, str | None]:
        text = f"{note.evidence_text or ''}\n{note.claim or ''}"
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
    def _identifier_values_for_notes(cls, notes: list[EvidenceCandidate]) -> list[str]:
        values: list[str] = []
        for note in notes:
            key, value = cls._assignment_from_note(note)
            if key and key.lower() in {"id", "identifier", "sample_id"} and value:
                values.append(value)
        return cls._dedupe_strings(values)

    @classmethod
    def _date_value_for_notes(cls, notes: list[EvidenceCandidate]) -> str | None:
        for note in notes:
            text = f"{note.evidence_text or ''} {note.claim or ''}"
            match = re.search(r"\b(20\d{2}-\d{2}-\d{2})(?:[T ][0-2]\d:[0-5]\d(?::[0-5]\d)?)?\b", text)
            if match:
                return match.group(1)
        return None

    @classmethod
    def _fallback_target_title(cls, notes: list[EvidenceCandidate]) -> str | None:
        for note in notes:
            title = cls._title_value_from_note(note)
            if title:
                return title
        keywords = cls._profile_keywords_for_notes(notes)
        observations = cls._profile_observation_sentences(notes, max_count=1)
        return observations[0] if observations else None

    @classmethod
    def _note_has_curatable_profile_signal(cls, note: EvidenceCandidate) -> bool:
        if note.category == "measurement_signal":
            return False
        if cls._is_low_level_parameter_note(note):
            return False
        key, value = cls._assignment_from_note(note)
        normalized_key = key.lower().strip("$") if key else ""
        if normalized_key in {"origin", "owner", "author", "creator"}:
            return bool(value)
        if normalized_key == "title":
            return cls._title_value_from_note(note) is not None
        text = cls._note_search_text(note)
        curatable_terms = (
            "dataset name",
            "dataset contains",
            "instrument",
            "device",
            "equipment",
            "format",
            "file",
            "distribution",
            "method",
            "measurement",
            "experiment",
            "sample",
            "modification date",
            "timestamp",
        )
        if any(term in text for term in curatable_terms):
            if "spectrum title" in text and cls._title_value_from_note(note) is None:
                return False
            return True
        return note.category in {"agent_signal", "activity_signal", "instrument_signal", "surrounding_signal", "resource_signal"} and bool(note.claim.strip())

    @classmethod
    def _entity_title_for_notes(cls, notes: list[EvidenceCandidate]) -> str | None:
        observations = cls._profile_observation_sentences(notes, max_count=1)
        return observations[0] if observations else None

    @staticmethod
    def _note_search_text(note: EvidenceCandidate) -> str:
        return " ".join(
            part
            for part in (
                note.candidate_id,
                note.category,
                note.claim,
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
        state.generated_reconstructed_draft = clean_document
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
        state.document_quality_state = self._build_document_quality_state(
            validation=state.validation,
            requirement_report=state.requirement_report,
            field_completion_ledger=state.field_completion_ledger,
        )
        self.output_repository.save_grounding_artifacts(
            workflow_id=data_package_id,
            vocab_queries=state.vocab_queries,
            normalization=normalization,
            chat_model=state.chat_model,
            chunking_strategy=state.chunking_strategy,
        )
        token_usage = await self.get_token_usage(
            data_package_id,
            chunking_strategy=state.chunking_strategy,
            chat_model=state.chat_model,
        )
        result = ExtractionRunResult(
            generated_final_draft=clean_document,
            machine_evidence_context=evidence_context,
            generated_initial_draft=state.generated_initial_draft,
            generated_patched_draft=state.generated_patched_draft,
            generated_reconstructed_draft=state.generated_reconstructed_draft,
            requirement_report=state.requirement_report,
            initial_file_summaries=state.initial_file_summaries,
            initial_file_summary_status=state.initial_file_summary_status,
            initial_extraction_overview=state.initial_extraction_overview,
            initial_extraction_overview_status=state.initial_extraction_overview_status,
            dataset_summary=state.dataset_summary,
            curated_document=state.curated_document,
            document_quality_state=state.document_quality_state,
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
            chunking_strategy=state.chunking_strategy,
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
        chunking_strategy = state.chunking_strategy
        self._persist_initial_file_summaries(data_package_id, state)
        self._persist_initial_extraction_overview(data_package_id, state)
        if state.generated_reconstructed_draft is not None:
            self.output_repository.save_generated_final_draft(
                workflow_id=data_package_id,
                document=state.generated_reconstructed_draft,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        if state.generated_initial_draft is not None:
            self.output_repository.save_generated_initial_draft(
                workflow_id=data_package_id,
                document=state.generated_initial_draft,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        if state.generated_patched_draft is not None:
            self.output_repository.save_generated_patched_draft(
                workflow_id=data_package_id,
                document=state.generated_patched_draft,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        if state.requirement_report is not None:
            self.output_repository.save_requirement_report(
                workflow_id=data_package_id,
                report=state.requirement_report,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        if state.curated_document is not None:
            self.output_repository.save_curated_document(
                workflow_id=data_package_id,
                document=state.curated_document,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        self.output_repository.save_projection_ledger(
            workflow_id=data_package_id,
            ledger=state.projection_ledger,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
        self.output_repository.save_field_completion_ledger(
            workflow_id=data_package_id,
            ledger=state.field_completion_ledger,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
        self.output_repository.save_evidence_query_ledger(
            workflow_id=data_package_id,
            ledger=state.evidence_query_ledger,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
        self.output_repository.save_curation_ledger(
            workflow_id=data_package_id,
            ledger=state.curation_ledger,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
        self.output_repository.save_validation(
            workflow_id=data_package_id,
            validation=state.validation,
            curated_validation=state.curated_validation,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
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

    def _clear_profile_projection_token_usage(self, data_package_id: str) -> None:
        if self.output_repository is None:
            return
        state = self._load_run_state_or_none(data_package_id)
        chat_model = state.chat_model if state else (self.ollama_client.chat_model if self.ollama_client else None)
        chunking_strategy = state.chunking_strategy if state else "semantic"
        totals = self.output_repository.load_token_usage(
            data_package_id,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
        )
        projection_agents = {
            "dataset_summary",
            "dataset_level_projection",
            "dataset_level_projection_repair",
            "description_fact_miner",
            "profile_target_planner",
            "profile_target_writer",
            "profile_patch",
            "profile_projection",
            "evidence_novelty_evaluator",
            "evidence_instance_builder",
            "evidence_instance_repair",
            "metadata_completeness_evaluator",
            "metadata_requirement_patcher",
        }
        pruned = {
            agent_name: values
            for agent_name, values in totals.items()
            if agent_name not in projection_agents
        }
        self.output_repository.save_token_usage(
            workflow_id=data_package_id,
            token_usage=pruned,
            chat_model=chat_model,
            chunking_strategy=chunking_strategy,
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
                    pointer += "/" + ProjectionService._json_pointer_escape(token)
                    token = ""
                continue
            if char == "[":
                if token:
                    pointer += "/" + ProjectionService._json_pointer_escape(token)
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
            pointer += "/" + ProjectionService._json_pointer_escape(token)
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
                    field_name=ProjectionService._field_name_from_pointer(json_path),
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
                    field_name=ProjectionService._field_name_from_pointer(json_path),
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
        evidence_context: RoutedEvidenceContext | EvidenceContext,
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

        evidence_categories = {
            note.category
            for note in (
                evidence_context.portable_evidence
                if isinstance(evidence_context, RoutedEvidenceContext)
                else evidence_context.candidates
            )
        }
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
        if "surrounding_signal" in evidence_categories:
            evidence_slots.add("creator")
            evidence_slots.add("modification_date")
        if "activity_signal" in evidence_categories:
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
            return ["agent_signal", "instrument_signal"]
        if path.startswith("/creator"):
            return ["surrounding_signal"]
        if path.startswith("/dataset_distribution"):
            return ["resource_signal"]
        if path.startswith("/was_generated_by"):
            if "realized_plan" in path or target_class == "Plan":
                return ["method_signal"]
            return ["activity_signal", "instrument_signal"]
        if path.startswith("/is_about_activity"):
            return ["activity_signal"]
        if path.startswith("/is_about_entity"):
            return ["activity_signal"]
        if path.startswith("/type"):
            return ["resource_signal", "surrounding_signal"]
        if path == "/modification_date":
            return ["surrounding_signal"]
        if path == "/keyword":
            return ["resource_signal", "activity_signal", "method_signal", "instrument_signal"]
        if path == "/description":
            return ["surrounding_signal", "other"]
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
        evidence_context: RoutedEvidenceContext | EvidenceContext,
    ) -> str:
        candidates = (
            evidence_context.portable_evidence
            if isinstance(evidence_context, RoutedEvidenceContext)
            else evidence_context.candidates
        )
        for note in candidates:
            title = ProjectionService._title_value_from_note(note)
            if title:
                return title
        for category in ("activity_signal", "instrument_signal", "method_signal", "resource_signal", "surrounding_signal"):
            for note in candidates:
                if (
                    note.category == category
                    and note.claim.strip()
                    and not ProjectionService._is_low_level_parameter_note(note)
                ):
                    return note.claim.strip()
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

