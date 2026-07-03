from __future__ import annotations

from copy import deepcopy
import math

from pydantic import BaseModel, ConfigDict, Field

from app.domain.extraction import (
    compact_file_summaries_for_shallow_projection,
)
from app.services.extraction_shared import *


async def generate_structured(*args: Any, **kwargs: Any) -> Any:
    from app.services import workflow_service

    return await workflow_service.generate_structured(*args, **kwargs)


@dataclass
class _CompiledSemanticActions:
    writes: list[SchemaConstrainedWrite]
    reason: str
    diagnosed_defects_count: int = 0
    synthesis_calls_count: int = 0
    rejected_reasons: list[str] | None = None
    diagnosed_defects: list[dict[str, Any]] | None = None
    compiled_actions: list[dict[str, Any]] | None = None


class _ParentQuantitativeAttributeIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    value: float | int | None = None
    unit: str | None = None
    description: str = ""


class _ParentQualitativeAttributeIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    value: str = ""
    description: str = ""


class _ParentAttributeIntentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    quantitative_attributes: list[_ParentQuantitativeAttributeIntent] = Field(default_factory=list)
    qualitative_attributes: list[_ParentQualitativeAttributeIntent] = Field(default_factory=list)


class _CoreObjectIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    description: str = ""


class _ProvenanceCoreIntentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    activity_title: str = ""
    activity_description: str = ""
    plan: _CoreObjectIntent | None = None
    agents: list[_CoreObjectIntent] = Field(default_factory=list)
    evaluated_entities: list[_CoreObjectIntent] = Field(default_factory=list)
    evaluated_activities: list[_CoreObjectIntent] = Field(default_factory=list)
    input_entities: list[_CoreObjectIntent] = Field(default_factory=list)
    input_activities: list[_CoreObjectIntent] = Field(default_factory=list)
    output_entities: list[_CoreObjectIntent] = Field(default_factory=list)


PROVENANCE_CORE_SYSTEM_PROMPT = (
    "Answer one narrow generic provenance-core question for a DCAT-AP+ dataset draft. "
    "Focus on the scientific or technical DataGeneratingActivity that produced the dataset. "
    "Classify each candidate by its role in that activity before placing it in exactly one relation. "
    "Use only the supplied orientation context. Use concise labels and descriptions only; "
    "do not emit JSON paths, schema patches, provenance bookkeeping, or nested profile objects. "
    "Return empty arrays when the orientation context does not support a relation. "
    "When unsure between agent and plan/input, prefer plan/input or omit."
)


PARENT_ATTRIBUTE_SYSTEM_PROMPT = (
    "Answer one narrow DCAT-AP+ parent-attribute question. "
    "A DCAT-AP+ attribute is a recorded characterization of the parent object itself. "
    "First decide which facts, if any, are useful profile-level attributes for the given parent. "
    "Silently check that each attribute characterizes this exact parent, is directly supported by selected evidence, "
    "has a clear attribute intent, and is not merely source-record metadata, provenance bookkeeping, "
    "creator/owner/origin information, a file path, a process path, an identifier for another object, "
    "or a fact better represented by a dedicated relation. "
    "Then organize only those selected facts into quantitative_attributes and qualitative_attributes. "
    "Return empty arrays when no useful parent-level attributes are supported."
)


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

        base_document, scaffold = self._initial_profile_document(
            data_package_id=data_package_id,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
        )
        state.initial_draft_scaffold = scaffold
        progress.initial_draft_scaffold = scaffold
        del base_document

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
        document = self._dataset_only_initial_draft(remove_null_values(projection))
        state.projection_ledger = projection_records
        state.generated_initial_draft = self._clone_json_object(document)
        state.generated_final_draft = document
        state.generated_core_draft = self._clone_json_object(document)
        progress.generated_initial_draft = state.generated_initial_draft
        progress.generated_final_draft = document
        progress.generated_core_draft = state.generated_core_draft
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
        progress.stage = "evidence_patching"
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        document = await self._apply_provenance_core_construction(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
            state=state,
            progress=progress,
        )
        state.generated_final_draft = document
        progress.generated_final_draft = document

        progress.stage = "coverage_scoring"
        self._update_progress(data_package_id, progress)
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
                evidence_context=evidence_context,
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
            if self._requirement_target_path_exists(
                document=document,
                target_paths=item.target_paths or requirement.target_paths,
            ):
                item.status = "fulfilled"
                item.quality = 1.0
                item.weighted_score = item.weight
                item.rationale = "Coverage target path exists."
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

        state.generated_core_draft = self._clone_json_object(document)
        progress.generated_core_draft = state.generated_core_draft
        document = await self._apply_parent_scoped_attribute_construction(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            evidence_context=evidence_context,
            validation_schema=validation_schema,
            state=state,
            progress=progress,
        )
        document = self._cleanup_parent_attributes_after_construction(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            validation_schema=validation_schema,
        )
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
        state.generated_attribute_draft = self._clone_json_object(document)
        progress.generated_attribute_draft = state.generated_attribute_draft
        progress.validation = state.validation
        progress.stage = "semantic_evaluation"
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        coverage = compute_coverage_report(document, validation_schema)
        semantic_items = await self._evaluate_semantic_requirements(
            data_package_id=data_package_id,
            document=document,
            evidence_context=evidence_context,
        )
        progress.stage = "semantic_reconstruction"
        self._update_progress(data_package_id, progress)
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
        progress.validation = state.validation
        self._save_run_state(data_package_id, state)
        self._persist_state_artifacts(data_package_id, state)
        self._update_progress(data_package_id, progress)
        semantic_items = await self._evaluate_semantic_requirements(
            data_package_id=data_package_id,
            document=document,
            evidence_context=evidence_context,
        )
        self._carry_semantic_reconstruction_trace(
            semantic_items=semantic_items,
            records=semantic_reconstructions,
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

    @staticmethod
    def _carry_semantic_reconstruction_trace(
        *,
        semantic_items: list[RequirementReportItem],
        records: list[SemanticReconstructionRecord],
    ) -> None:
        records_by_id = {record.requirement_id: record for record in records}
        for item in semantic_items:
            record = records_by_id.get(item.requirement_id)
            if record is None:
                continue
            item.defect_type = record.defect_type
            item.diagnosed_defects_count = record.diagnosed_defects_count
            item.compiled_actions_count = record.compiled_actions_count
            item.synthesis_calls_count = record.synthesis_calls_count
            item.diagnosed_defects = list(record.diagnosed_defects)
            item.compiled_actions = list(record.compiled_actions)
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
            "attribute_range_decomposition",
            "attribute_parent_placement",
            "technical_agent_kind",
            "method_plan_presence",
            "generation_activity_reality",
            "dataset_title_identity",
            "dataset_description_scope",
            "activity_evaluation_target",
        ]
        by_id = {item.requirement_id: item for item in semantic_items}
        order = order + [requirement_id for requirement_id in by_id if requirement_id not in order]
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
            update_result = await self._semantic_reconstruction_update(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=current,
                item=item,
                requirement=self._semantic_requirement_for_item(item),
                validation_schema=validation_schema,
            )
            if len(update_result) == 4:
                updated, changed_paths, reason, validation_errors = update_result
                applied_actions_count = 1 if changed_paths else 0
                rejected_reasons = list(validation_errors)
                diagnosed_defects_count = 0
                synthesis_calls_count = 0
            else:
                (
                    updated,
                    changed_paths,
                    reason,
                    validation_errors,
                    applied_actions_count,
                    rejected_reasons,
                ) = update_result
                diagnosed_defects_count = item.diagnosed_defects_count
                synthesis_calls_count = item.synthesis_calls_count
            if not changed_paths:
                status = "failed" if validation_errors else "skipped"
                if status == "skipped" and item.status in {"partial", "missing", "unanswered", "unresolved"}:
                    status = "unresolved"
                records.append(
                    SemanticReconstructionRecord(
                        requirement_id=requirement_id,
                        status=status,
                        target_paths=item.target_paths,
                        reason=reason or "No semantic reconstruction was proposed.",
                        validation_errors=validation_errors,
                        applied_actions_count=applied_actions_count,
                        rejected_actions_count=len(rejected_reasons),
                        rejected_reasons=rejected_reasons,
                        defect_type=item.defect_type,
                        diagnosed_defects_count=diagnosed_defects_count,
                        compiled_actions_count=item.compiled_actions_count,
                        synthesis_calls_count=synthesis_calls_count,
                        diagnosed_defects=item.diagnosed_defects,
                        compiled_actions=item.compiled_actions,
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
                        applied_actions_count=applied_actions_count,
                        rejected_actions_count=len(rejected_reasons),
                        rejected_reasons=rejected_reasons,
                        defect_type=item.defect_type,
                        diagnosed_defects_count=diagnosed_defects_count,
                        compiled_actions_count=item.compiled_actions_count,
                        synthesis_calls_count=synthesis_calls_count,
                        diagnosed_defects=item.diagnosed_defects,
                        compiled_actions=item.compiled_actions,
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
                applied_actions_count=applied_actions_count,
                rejected_actions_count=len(rejected_reasons),
                rejected_reasons=rejected_reasons,
                defect_type=item.defect_type,
                diagnosed_defects_count=diagnosed_defects_count,
                compiled_actions_count=item.compiled_actions_count,
                synthesis_calls_count=synthesis_calls_count,
                diagnosed_defects=item.diagnosed_defects,
                compiled_actions=item.compiled_actions,
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
        profile_identifier: str,
        document: dict[str, Any],
        item: RequirementReportItem,
        requirement: DcatRequirement,
        validation_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], str, list[str], int, list[str]]:
        assert self.ollama_client is not None
        allowed_paths = self._semantic_reconstruction_allowed_paths(
            document=document,
            item=item,
            requirement=requirement,
        )
        deterministic = await self._compile_deterministic_semantic_actions(
            data_package_id=data_package_id,
            document=document,
            item=item,
            requirement=requirement,
            validation_schema=validation_schema,
        )
        if deterministic.writes:
            self._set_semantic_item_trace(item=item, compiled=deterministic)
            updated, changed_paths, reason, errors, applied, rejected = self._apply_semantic_reconstruction_writes(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=document,
                writes=deterministic.writes,
                reason=deterministic.reason,
                validation_schema=validation_schema,
                rejected_reasons=deterministic.rejected_reasons or [],
            )
            item.diagnosed_defects_count = deterministic.diagnosed_defects_count
            item.synthesis_calls_count = deterministic.synthesis_calls_count
            return updated, changed_paths, reason, errors, applied, rejected

        draft_excerpt = self._semantic_reconstruction_draft_excerpt(
            document=document,
            requirement=requirement,
            allowed_paths=allowed_paths,
        )
        prompt = build_semantic_diagnosis_prompt(
            requirement=requirement,
            item=item,
            draft_excerpt=draft_excerpt,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=semantic_diagnosis_output_schema(),
                system_components=[
                    ("semantic_diagnosis_system_prompt", SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT),
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
            legacy_reason = self._legacy_semantic_diagnosis_response_reason(result.output)
            if legacy_reason:
                return document, [], legacy_reason, [legacy_reason], 0, [legacy_reason]
            diagnosis = SemanticReconstructionDiagnosis.model_validate(result.output)
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="semantic_reconstruction",
            )
            return document, [], f"Semantic reconstruction diagnosis failed: {exc}", [str(exc)], 0, [str(exc)]
        compiled = await self._compile_semantic_diagnosis_actions(
            data_package_id=data_package_id,
            document=document,
            item=item,
            requirement=requirement,
            diagnosis=diagnosis,
            validation_schema=validation_schema,
        )
        self._set_semantic_item_trace(item=item, compiled=compiled)
        if not compiled.writes:
            reason = compiled.reason or diagnosis.reason or "semantic_defect_unresolved_empty_diagnosis"
            rejected = compiled.rejected_reasons or []
            item.diagnosed_defects_count = compiled.diagnosed_defects_count
            item.synthesis_calls_count = compiled.synthesis_calls_count
            return document, [], reason, [], 0, rejected
        updated, changed_paths, reason, errors, applied, rejected = self._apply_semantic_reconstruction_writes(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            writes=compiled.writes,
            reason=compiled.reason or diagnosis.reason,
            validation_schema=validation_schema,
            rejected_reasons=compiled.rejected_reasons or [],
        )
        item.diagnosed_defects_count = compiled.diagnosed_defects_count
        item.synthesis_calls_count = compiled.synthesis_calls_count
        return updated, changed_paths, reason, errors, applied, rejected

    @staticmethod
    def _set_semantic_item_trace(
        *,
        item: RequirementReportItem,
        compiled: _CompiledSemanticActions,
    ) -> None:
        item.diagnosed_defects = list(compiled.diagnosed_defects or [])
        item.compiled_actions = list(compiled.compiled_actions or [])
        item.compiled_actions_count = len(item.compiled_actions)
        defect_types = list(
            dict.fromkeys(
                str(defect.get("defect_type") or "")
                for defect in item.diagnosed_defects
                if defect.get("defect_type") and defect.get("defect_type") != "no_defect"
            )
        )
        item.defect_type = ",".join(defect_types)

    @staticmethod
    def _semantic_requirement_for_item(item: RequirementReportItem) -> DcatRequirement:
        for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS:
            if requirement.requirement_id == item.requirement_id:
                return requirement.model_copy(update={"target_paths": list(item.target_paths)})
        return DcatRequirement(
            requirement_id=item.requirement_id,
            label=item.label or item.requirement_id,
            description=item.rationale or item.requirement_id,
            weight=item.weight or 1.0,
            target_paths=list(item.target_paths),
            evidence_hints=list(item.evidence_search_hints),
        )

    def _apply_semantic_reconstruction_writes(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        writes: list[SchemaConstrainedWrite],
        reason: str,
        validation_schema: dict[str, Any],
        rejected_reasons: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], str, list[str], int, list[str]]:
        patch = SchemaConstrainedPatchResult(writes=writes, reason=reason)
        patch = patch.model_copy(
            update={
                "writes": self._sanitize_schema_constrained_writes(
                    writes=patch.writes,
                    requirement_id="semantic_reconstruction",
                )
            }
        )
        allowed_rejected = self._semantic_reconstruction_rejected_actions(
            writes=patch.writes,
            allowed_paths=list(dict.fromkeys(write.target_path for write in patch.writes)),
        )
        deduped_writes, duplicate_reasons = self._filter_duplicate_schema_writes(
            document=document,
            writes=patch.writes,
        )
        if not deduped_writes:
            reason = patch.reason or "Semantic reconstruction only proposed duplicate writes."
            rejected = list(dict.fromkeys((rejected_reasons or []) + allowed_rejected + duplicate_reasons))
            if rejected:
                reason = f"{reason} {'; '.join(rejected)}"
            return document, [], reason, [], 0, rejected
        updated = document
        changed_paths: list[str] = []
        applied = 0
        rejected = list((rejected_reasons or []) + allowed_rejected + duplicate_reasons)
        for write in self._semantic_reconstruction_candidate_writes(deduped_writes):
            write_rejections = self._validate_semantic_reconstruction_action(
                document=updated,
                write=write,
            )
            if write_rejections:
                rejected.extend(write_rejections)
                continue
            try:
                candidate, candidate_paths = apply_schema_constrained_writes(
                    document=updated,
                    writes=[write],
                    data_package_id=data_package_id,
                    validation_schema=validation_schema,
                )
            except Exception as exc:
                rejected.append(str(exc))
                continue
            candidate_validation = self.profile_service.validate_document(
                identifier=profile_identifier,
                document=candidate,
            )
            if not candidate_validation.valid:
                rejected.extend(issue.message for issue in candidate_validation.errors)
                continue
            updated = candidate
            changed_paths.extend(candidate_paths)
            applied += 1
        if changed_paths:
            reason = patch.reason
            if rejected:
                reason = f"{reason} Applied valid operations; rejected invalid operations."
            return updated, list(dict.fromkeys(changed_paths)), reason, [], applied, list(dict.fromkeys(rejected))
        return document, [], patch.reason or "Semantic reconstruction failed validation.", list(dict.fromkeys(rejected)), 0, list(dict.fromkeys(rejected))

    @classmethod
    def _semantic_reconstruction_allowed_paths(
        cls,
        *,
        document: dict[str, Any],
        item: RequirementReportItem,
        requirement: DcatRequirement,
    ) -> list[str]:
        paths = list(item.target_paths or requirement.target_paths)
        if requirement.requirement_id == "generation_activity_reality":
            activities = document.get("was_generated_by")
            indices = [
                index
                for index, activity in enumerate(activities if isinstance(activities, list) else [])
                if isinstance(activity, dict)
            ] or [0]
            return [
                f"/was_generated_by/{index}/{field}"
                for index in indices
                for field in ("title", "description", "type")
            ]
        if requirement.requirement_id not in {
            "attribute_range_decomposition",
            "attribute_parent_placement",
        }:
            return paths
        parent_paths = cls._attribute_parent_candidate_paths(
            document=document,
            selected_evidence=item.selected_evidence,
            fallback_paths=paths,
        )
        return parent_paths or paths

    @classmethod
    def _semantic_reconstruction_draft_excerpt(
        cls,
        *,
        document: dict[str, Any],
        requirement: DcatRequirement,
        allowed_paths: list[str],
    ) -> dict[str, Any]:
        if requirement.requirement_id == "attribute_parent_placement":
            return cls._attribute_parent_placement_excerpt(document)
        return {path: cls._value_at_json_pointer(document, path) for path in allowed_paths}

    @classmethod
    def _attribute_parent_candidate_paths(
        cls,
        *,
        document: dict[str, Any],
        selected_evidence: list[RequirementEvidenceItem],
        fallback_paths: list[str],
    ) -> list[str]:
        text = " ".join(
            " ".join((evidence.category, evidence.claim, evidence.evidence_text))
            for evidence in selected_evidence
        ).lower()
        agent_cue = bool(re.search(r"\b(device|software|instrument|machine|sensor|apparatus)\b", text))
        measurement_or_setting = any(
            evidence.category in {"instrument_signal", "measurement_condition"}
            for evidence in selected_evidence
        )

        candidates: list[str] = []
        if agent_cue:
            for activity_index, activity in enumerate(document.get("was_generated_by") or []):
                if not isinstance(activity, dict):
                    continue
                for agent_index, agent in enumerate(activity.get("carried_out_by") or []):
                    if isinstance(agent, dict):
                        candidates.append(f"/was_generated_by/{activity_index}/carried_out_by/{agent_index}/has_quantitative_attribute")
            if not candidates:
                candidates.append("/was_generated_by/0/carried_out_by/0/has_quantitative_attribute")
        if measurement_or_setting or not candidates:
            candidates.append("/was_generated_by/0/has_quantitative_attribute")
        candidates.extend(path for path in fallback_paths if path not in candidates)
        return list(dict.fromkeys(candidates))

    def _semantic_reconstruction_output_schema(
        self,
        *,
        validation_schema: dict[str, Any],
        allowed_target_paths: list[str],
        requirement_id: str,
    ) -> dict[str, Any]:
        return SchemaConstrainedPatchRoute.output_schema(
            validation_schema=validation_schema,
            allowed_target_paths=allowed_target_paths,
        )

    async def _compile_deterministic_semantic_actions(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        item: RequirementReportItem,
        requirement: DcatRequirement,
        validation_schema: dict[str, Any],
    ) -> _CompiledSemanticActions:
        if requirement.requirement_id == "activity_evaluation_target":
            paths = self._evaluated_activity_self_reference_paths(document)
            writes = [
                SchemaConstrainedWrite(
                    target_path=path,
                    mode="remove",
                    reason="Remove evaluated_activity self-reference from data-generating activity.",
                )
                for path in sorted(paths, reverse=True)
            ]
            if writes:
                return _CompiledSemanticActions(
                    writes=writes,
                    reason="Deterministic evaluated-activity self-reference cleanup.",
                    diagnosed_defects_count=len(writes),
                    diagnosed_defects=[
                        {
                            "defect_type": "self_referential_evaluated_activity",
                            "target_path": write.target_path,
                            "entry_indices": [],
                            "recommended_action": "remove",
                            "needs_synthesis": False,
                            "reason": write.reason,
                        }
                        for write in writes
                    ],
                    compiled_actions=[write.model_dump(mode="json") for write in writes],
                )
        return _CompiledSemanticActions(writes=[], reason="")

    async def _compile_semantic_diagnosis_actions(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        item: RequirementReportItem,
        requirement: DcatRequirement,
        diagnosis: SemanticReconstructionDiagnosis,
        validation_schema: dict[str, Any],
    ) -> _CompiledSemanticActions:
        writes: list[SchemaConstrainedWrite] = []
        rejected: list[str] = []
        synthesis_calls = 0
        allowed_paths = self._semantic_reconstruction_allowed_paths(
            document=document,
            item=item,
            requirement=requirement,
        )
        for defect in diagnosis.defects:
            if defect.defect_type == "no_defect" or defect.recommended_action == "no_action":
                continue
            if not self._semantic_path_allowed(defect.target_path, allowed_paths):
                rejected.append(f"Diagnosis target outside allowed semantic reconstruction paths: {defect.target_path}")
                continue
            if (
                requirement.requirement_id == "dataset_description_scope"
                and defect.recommended_action == "remove"
                and self._description_removal_would_empty_document(document, defect)
            ):
                rejected.append(
                    "Dataset description scope may streamline the description but must not remove the only description entry."
                )
                continue
            if defect.needs_synthesis:
                if defect.recommended_action in {"merge", "remove", "move", "no_action"}:
                    rejected.append(
                        f"Mechanical action {defect.recommended_action} must not request synthesis at "
                        f"{defect.target_path}."
                    )
                    continue
                current_target = self._value_at_json_pointer(document, defect.target_path)
                attribute_array_target = defect.target_path.rstrip("/").endswith(
                    ("/has_quantitative_attribute", "/has_qualitative_attribute")
                )
                if (
                    attribute_array_target
                    and defect.recommended_action in {"append", "replace"}
                    and current_target is None
                    and not item.selected_evidence
                    and not item.context_window
                ):
                    rejected.append(
                        "Attribute synthesis requires selected evidence, context evidence, or an existing target; "
                        f"refusing placeholder attribute at {defect.target_path}."
                    )
                    continue
                if requirement.requirement_id == "activity_evaluation_target":
                    rejected.append(
                        "Evaluation-target relations must come from explicit projection evidence; "
                        f"refusing synthesized relation at {defect.target_path}."
                    )
                    continue
                synthesis_defect = defect
                if (
                    defect.recommended_action == "replace"
                    and isinstance(current_target, list)
                    and attribute_array_target
                ):
                    indices = list(dict.fromkeys(defect.entry_indices))
                    if len(indices) != 1:
                        rejected.append(
                            "Array replacement diagnosis must identify exactly one entry; "
                            f"refusing collection-wide replacement at {defect.target_path}."
                        )
                        continue
                    index = indices[0]
                    if index < 0 or index >= len(current_target):
                        rejected.append(f"Replacement index out of range at {defect.target_path}: {index}.")
                        continue
                    synthesis_defect = defect.model_copy(
                        update={
                            "target_path": f"{defect.target_path.rstrip('/')}/{index}",
                            "entry_indices": [],
                        }
                    )
                write, synth_rejected = await self._synthesize_semantic_reconstruction_write(
                    data_package_id=data_package_id,
                    document=document,
                    requirement=requirement,
                    defect=synthesis_defect,
                    item=item,
                    validation_schema=validation_schema,
                )
                synthesis_calls += 1
                if write is None:
                    rejected.extend(synth_rejected)
                else:
                    writes.append(write)
                continue
            if defect.recommended_action == "merge":
                if len(defect.entry_indices) < 2:
                    rejected.append(f"Merge diagnosis needs at least two entry indices at {defect.target_path}.")
                    continue
                survivor, *merged = defect.entry_indices
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=defect.target_path,
                        mode="merge",
                        survivor_index=survivor,
                        merged_indices=merged,
                        reason=defect.reason,
                    )
                )
                continue
            if defect.recommended_action == "remove":
                if defect.entry_indices:
                    for index in sorted(set(defect.entry_indices), reverse=True):
                        writes.append(
                            SchemaConstrainedWrite(
                                target_path=f"{defect.target_path.rstrip('/')}/{index}",
                                mode="remove",
                                reason=defect.reason,
                            )
                        )
                else:
                    writes.append(
                        SchemaConstrainedWrite(
                            target_path=defect.target_path,
                            mode="remove",
                            reason=defect.reason,
                        )
                    )
                continue
            rejected.append(f"Diagnosis action requires synthesis or explicit compiler support: {defect.recommended_action}.")
        reason = diagnosis.reason or "Compiled semantic diagnosis actions."
        if not writes and not rejected and item.status in {"partial", "missing", "unanswered", "unresolved"}:
            reason = "semantic_defect_unresolved_empty_diagnosis"
        return _CompiledSemanticActions(
            writes=writes,
            reason=reason,
            diagnosed_defects_count=len(diagnosis.defects),
            synthesis_calls_count=synthesis_calls,
            rejected_reasons=list(dict.fromkeys(rejected)),
            diagnosed_defects=[defect.model_dump(mode="json") for defect in diagnosis.defects],
            compiled_actions=[write.model_dump(mode="json") for write in writes],
        )

    @classmethod
    def _description_removal_would_empty_document(
        cls,
        document: dict[str, Any],
        defect: SemanticReconstructionDefect,
    ) -> bool:
        descriptions = document.get("description")
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        if not isinstance(descriptions, list):
            return True
        present_indices = [
            index
            for index, value in enumerate(descriptions)
            if cls._semantic_value_present(value)
        ]
        if not present_indices:
            return True
        if defect.target_path.rstrip("/") == "/description":
            removal_indices = set(defect.entry_indices or present_indices)
        else:
            match = re.fullmatch(r"/description/(\d+)", defect.target_path.rstrip("/"))
            removal_indices = {int(match.group(1))} if match else set()
        return bool(present_indices) and set(present_indices).issubset(removal_indices)

    async def _synthesize_semantic_reconstruction_write(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        requirement: DcatRequirement,
        defect: SemanticReconstructionDefect,
        item: RequirementReportItem,
        validation_schema: dict[str, Any],
    ) -> tuple[SchemaConstrainedWrite | None, list[str]]:
        schema = self._semantic_synthesis_schema(
            validation_schema=validation_schema,
            requirement=requirement,
            defect=defect,
        )
        if not schema:
            return None, [f"No target schema found for semantic synthesis path: {defect.target_path}"]
        prompt = self._semantic_synthesis_prompt(
            requirement=requirement,
            defect=defect,
            item=item,
            current_target=self._value_at_json_pointer(document, defect.target_path),
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=SEMANTIC_SYNTHESIS_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=schema,
                system_components=[("semantic_synthesis_system_prompt", SEMANTIC_SYNTHESIS_SYSTEM_PROMPT)],
                prompt_components=[],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("semantic_reconstruction_synthesis"),
                agent_name="semantic_reconstruction_synthesis",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="semantic_reconstruction_synthesis",
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="semantic_reconstruction_synthesis",
            )
            return None, [f"Semantic reconstruction synthesis failed: {exc}"]
        value = result.output
        if self._semantic_absence_placeholder(value):
            return None, [
                f"Semantic synthesis returned an absence placeholder instead of evidence-backed content at {defect.target_path}."
            ]
        target_schema = self._resolve_schema_node(
            schema_for_json_pointer(validation_schema, defect.target_path),
            validation_schema,
        )
        current_target = self._value_at_json_pointer(document, defect.target_path)
        target_type = target_schema.get("type") if isinstance(target_schema, dict) else None
        target_is_array = target_type == "array" or (
            isinstance(target_type, list) and "array" in target_type
        ) or isinstance(current_target, list) or defect.target_path.rstrip("/").endswith(
            ("/has_quantitative_attribute", "/has_qualitative_attribute")
        )
        if defect.recommended_action == "append" and target_is_array:
            return (
                SchemaConstrainedWrite(
                    target_path=defect.target_path,
                    mode="append",
                    items=[value],
                    reason=defect.reason or "Synthesized semantic value.",
                ),
                [],
            )
        return (
            SchemaConstrainedWrite(
                target_path=defect.target_path,
                mode="replace",
                value=value,
                reason=defect.reason or "Synthesized semantic value.",
            ),
            [],
        )

    @classmethod
    def _semantic_synthesis_schema(
        cls,
        *,
        validation_schema: dict[str, Any],
        requirement: DcatRequirement,
        defect: SemanticReconstructionDefect,
    ) -> dict[str, Any]:
        target_path = defect.target_path[:-2] if defect.target_path.endswith("/-") else defect.target_path
        target_schema = schema_for_json_pointer(validation_schema, target_path)
        if defect.recommended_action == "append":
            resolved = cls._resolve_schema_node(target_schema, validation_schema)
            if isinstance(resolved, dict) and isinstance(resolved.get("items"), dict):
                target_schema = cls._resolve_schema_node(resolved["items"], validation_schema)
        if not target_schema:
            return cls._fallback_semantic_synthesis_schema(requirement=requirement, defect=defect)
        allowed_properties = {
            "method_plan_presence": {"id", "title", "description", "type"},
            "technical_agent_kind": {"id", "title", "description", "type"},
            "attribute_parent_placement": {"title", "description", "value", "has_quantity_type", "has_attribute_type", "unit"},
            "attribute_range_decomposition": {"title", "description", "value", "has_quantity_type", "unit"},
            "generation_activity_reality": {"title", "description", "type"},
        }.get(requirement.requirement_id)
        compact = cls._compact_synthesis_schema_node(
            target_schema,
            validation_schema,
            allowed_properties=allowed_properties,
        )
        if compact:
            compact["$schema"] = "https://json-schema.org/draft/2019-09/schema"
        return compact

    @classmethod
    def _compact_synthesis_schema_node(
        cls,
        node: Any,
        root: dict[str, Any],
        *,
        allowed_properties: set[str] | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        resolved = cls._resolve_schema_node(node, root)
        if not isinstance(resolved, dict):
            return {}
        for union_key in ("anyOf", "oneOf"):
            branches = resolved.get(union_key)
            if isinstance(branches, list):
                compact_branches = [
                    cls._compact_synthesis_schema_node(branch, root, depth=depth + 1)
                    for branch in branches
                ]
                compact_branches = [branch for branch in compact_branches if branch]
                return {union_key: compact_branches} if compact_branches else {}
        schema_type = resolved.get("type")
        result: dict[str, Any] = {}
        for key in ("type", "enum", "const", "format", "minimum", "maximum", "minItems", "maxItems"):
            if key in resolved:
                result[key] = deepcopy(resolved[key])
        if schema_type == "array" or (isinstance(schema_type, list) and "array" in schema_type):
            item_schema = cls._compact_synthesis_schema_node(
                resolved.get("items", {}),
                root,
                depth=depth + 1,
            )
            if item_schema:
                result["items"] = item_schema
            return result
        properties = resolved.get("properties")
        if not isinstance(properties, dict):
            return result
        required = {str(name) for name in resolved.get("required", [])}
        nested_defaults = {
            "id",
            "title",
            "description",
            "type",
            "from_CV",
            "value",
            "unit",
            "has_quantity_type",
            "has_attribute_type",
        }
        selected_names = required | (allowed_properties if allowed_properties is not None else nested_defaults)
        compact_properties: dict[str, Any] = {}
        for name in properties:
            if name not in selected_names:
                continue
            compact_child = cls._compact_synthesis_schema_node(
                properties[name],
                root,
                allowed_properties=None,
                depth=depth + 1,
            )
            if compact_child:
                compact_properties[name] = compact_child
        result["type"] = schema_type or "object"
        result["additionalProperties"] = False
        result["properties"] = compact_properties
        compact_required = [name for name in resolved.get("required", []) if name in compact_properties]
        if compact_required:
            result["required"] = compact_required
        return result

    @staticmethod
    def _fallback_semantic_synthesis_schema(
        *,
        requirement: DcatRequirement,
        defect: SemanticReconstructionDefect,
    ) -> dict[str, Any]:
        if defect.target_path.rstrip("/") in {"/title", "/description"}:
            if defect.recommended_action == "append":
                return {"$schema": "https://json-schema.org/draft/2019-09/schema", "type": "string"}
            return {
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        if requirement.requirement_id in {"attribute_parent_placement", "attribute_range_decomposition"}:
            return {
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "value": {"type": ["number", "string"]},
                    "has_quantity_type": {"type": ["string", "object"]},
                    "unit": {"type": ["string", "object"]},
                },
                "required": ["value", "has_quantity_type"],
            }
        return {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "object"},
            },
            "required": ["title", "description"],
        }

    @staticmethod
    def _semantic_synthesis_prompt(
        *,
        requirement: DcatRequirement,
        defect: SemanticReconstructionDefect,
        item: RequirementReportItem,
        current_target: Any,
    ) -> str:
        payload = {
            "requirement": requirement.model_dump(mode="json"),
            "defect": defect.model_dump(mode="json"),
            "current_target": current_target,
            "selected_evidence": compact_requirement_evidence(
                item.selected_evidence,
                include_source_context=True,
            ),
            "context_window": compact_requirement_evidence(
                item.context_window,
                include_source_context=False,
            ),
            "rules": [
                "Return one small schema-valid object or value for the defect target.",
                "Use only selected evidence and context window.",
                "Do not include profile patch operations.",
                "For generic attributes, the value must characterize the exact target parent and have clear parent-specific intent.",
                "Do not synthesize source-record metadata, creator/owner/origin fields, file paths, process paths, audit/hash details, provenance bookkeeping, identifiers for other objects, placeholders, or generic measured/unknown/present values as attributes.",
                "For creator/publisher/provenance targets, do not infer dataset-level responsibility from generic origin/owner/source metadata alone.",
            ],
        }
        return "Synthesize one semantic reconstruction value for backend compilation.\n\n" + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def _attribute_duplicate_coherence_writes(
        cls,
        document: dict[str, Any],
    ) -> tuple[list[SchemaConstrainedWrite], int]:
        writes: list[SchemaConstrainedWrite] = []
        for parent_path, items in cls._attribute_array_items(document):
            groups: list[list[int]] = []
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                group = next(
                    (
                        candidate
                        for candidate in groups
                        if cls._attributes_semantically_compatible(
                            parent_path=parent_path,
                            left=items[candidate[0]],
                            right=item,
                        )
                        or cls._attributes_value_equivalent(
                            left=items[candidate[0]],
                            right=item,
                        )
                        or cls._attributes_same_semantic_identity(
                            parent_path=parent_path,
                            left=items[candidate[0]],
                            right=item,
                        )
                    ),
                    None,
                )
                if group is None:
                    groups.append([index])
                else:
                    group.append(index)
            # Collect all non-survivor indices across all groups, then emit
            # individual remove writes in descending index order so that
            # removals do not shift earlier indices.
            removals: list[int] = []
            for indices in groups:
                if len(indices) < 2:
                    continue
                survivor_index = max(
                    indices,
                    key=lambda index: cls._attribute_survivor_score(items[index]),
                )
                removals.extend(index for index in indices if index != survivor_index)
            for index in sorted(removals, reverse=True):
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=f"{parent_path}/{index}",
                        mode="remove",
                        reason="Same-parent duplicate semantic attribute slot with equivalent value.",
                    )
                )
        return writes, len(writes)

    @classmethod
    def _attribute_range_decomposition_writes(
        cls,
        *,
        document: dict[str, Any],
        item: RequirementReportItem,
    ) -> tuple[list[SchemaConstrainedWrite], int]:
        writes: list[SchemaConstrainedWrite] = []
        for parent_path, items in cls._attribute_array_items(document):
            remove_indices: list[int] = []
            append_items: list[dict[str, Any]] = []
            for index, attribute in enumerate(items):
                if not isinstance(attribute, dict) or not cls._attribute_looks_like_bad_range(attribute):
                    continue
                bounds = cls._range_bounds_from_evidence(
                    attribute=attribute,
                    evidence=list(item.selected_evidence) + list(item.context_window),
                )
                if bounds is None:
                    bounds = cls._range_bounds_from_document_description(document)
                if bounds is None:
                    bounds = cls._range_bounds_from_sibling_attributes(
                        attribute=attribute,
                        siblings=items,
                    )
                if bounds is None:
                    continue
                low, high, supporting_text = bounds
                base = cls._range_base_label(attribute)
                unit = cls._range_unit_text(supporting_text)
                for bound, value in (("minimum", low), ("maximum", high)):
                    candidate = cls._range_attribute(base=base, bound=bound, value=value, unit=unit)
                    if not any(
                        isinstance(existing, dict)
                        and cls._range_bound_matches_existing(
                            candidate=candidate,
                            existing=existing,
                        )
                        for existing in items
                        if existing is not attribute
                    ):
                        append_items.append(candidate)
                remove_indices.append(index)
            if append_items:
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=parent_path,
                        mode="append",
                        items=append_items,
                        reason="Decompose range into minimum and maximum attributes.",
                    )
                )
            for index in sorted(remove_indices, reverse=True):
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=f"{parent_path}/{index}",
                        mode="remove",
                        reason="Remove invalid range fragment attribute.",
                    )
                )
        return writes, len(writes)

    @classmethod
    def _attribute_parent_placement_writes(
        cls,
        document: dict[str, Any],
    ) -> tuple[list[SchemaConstrainedWrite], int]:
        entries: list[tuple[str, int, dict[str, Any]]] = []
        for parent_path, items in cls._attribute_array_items(document):
            entries.extend(
                (parent_path, index, item)
                for index, item in enumerate(items)
                if isinstance(item, dict)
            )
        groups: list[list[tuple[str, int, dict[str, Any]]]] = []
        for entry in entries:
            matching = next(
                (
                    group
                    for group in groups
                    if any(
                        other_path != entry[0]
                        and cls._cross_parent_attributes_compatible(other, entry[2])
                        for other_path, _, other in group
                    )
                ),
                None,
            )
            if matching is None:
                groups.append([entry])
            else:
                matching.append(entry)

        removals: list[tuple[str, int]] = []
        for group in groups:
            if len({path for path, _, _ in group}) < 2:
                continue
            survivor = max(
                group,
                key=lambda entry: cls._attribute_parent_survivor_score(entry[0], entry[2]),
            )
            removals.extend((path, index) for path, index, _ in group if (path, index) != survivor[:2])
        removals.sort(key=lambda entry: (entry[0], -entry[1]))
        writes = [
            SchemaConstrainedWrite(
                target_path=f"{path}/{index}",
                mode="remove",
                reason="Remove cross-parent duplicate from the less suitable semantic owner.",
            )
            for path, index in removals
        ]

        # ------------------------------------------------------------------
        # Detect misplaced measurement attributes on activities that share
        # significant vocabulary with evaluated_entity attributes.  If an
        # activity attribute shares >=2 significant words with any evaluated
        # entity attribute, it is likely a measurement condition that belongs
        # on the evaluated entity, not the generating activity.
        # ------------------------------------------------------------------
        _stop = {"the", "of", "for", "in", "and", "a", "an", "is", "are", "was", "were",
                 "be", "been", "being", "have", "has", "had", "do", "does", "did"}
        for act_idx, activity in enumerate(document.get("was_generated_by") or []):
            if not isinstance(activity, dict):
                continue
            act_attrs = activity.get("has_quantitative_attribute") or []
            if not act_attrs:
                continue
            evaluated_entities = activity.get("evaluated_entity") or []
            if not evaluated_entities:
                continue
            # Collect significant words from all evaluated_entity quantitative attributes
            ee_words: set[str] = set()
            for ee in evaluated_entities:
                if not isinstance(ee, dict):
                    continue
                for attr in ee.get("has_quantitative_attribute") or []:
                    if not isinstance(attr, dict):
                        continue
                    label = cls._normalized_attribute_label(
                        " ".join(str(attr.get(k) or "") for k in ("has_quantity_type", "title"))
                    )
                    ee_words.update(w for w in label.split() if len(w) > 1 and w not in _stop)
            if not ee_words:
                continue
            # Check each activity quantitative attribute for vocabulary overlap
            move_indices: list[int] = []
            move_items: list[dict[str, Any]] = []
            for attr_idx, attr in enumerate(act_attrs):
                if not isinstance(attr, dict):
                    continue
                # Skip if this attribute is already targeted for cross-parent removal
                if any(path == f"/was_generated_by/{act_idx}/has_quantitative_attribute" and index == attr_idx
                       for path, index in removals):
                    continue
                label = cls._normalized_attribute_label(
                    " ".join(str(attr.get(k) or "") for k in ("has_quantity_type", "title"))
                )
                attr_words = {w for w in label.split() if len(w) > 1 and w not in _stop}
                shared = attr_words & ee_words
                if len(shared) >= 2:
                    move_indices.append(attr_idx)
                    move_items.append(dict(attr))
            if move_indices:
                # Remove from activity in descending index order
                for attr_idx in sorted(move_indices, reverse=True):
                    writes.append(
                        SchemaConstrainedWrite(
                            target_path=f"/was_generated_by/{act_idx}/has_quantitative_attribute/{attr_idx}",
                            mode="remove",
                            reason="Move misplaced measurement attribute from activity to evaluated_entity.",
                        )
                    )
                # Append to first evaluated_entity
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=f"/was_generated_by/{act_idx}/evaluated_entity/0/has_quantitative_attribute/-",
                        mode="append",
                        items=move_items,
                        reason="Move misplaced measurement attribute from activity to evaluated_entity.",
                    )
                )
        return writes, len(writes)

    @classmethod
    def _cross_parent_attributes_compatible(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_slot = cls._attribute_semantic_slot(parent_path="", instance=left)
        right_slot = cls._attribute_semantic_slot(parent_path="", instance=right)
        if not left_slot or not right_slot or left_slot[1:3] != right_slot[1:3]:
            # Secondary check: same numeric value and compatible units even with different labels
            if not cls._attributes_value_equivalent(left, right):
                return False
        else:
            left_unit = left_slot[4]
            right_unit = right_slot[4]
            pseudo_units = {"transmittance", "intensity", "count", "points", "point count"}
            if left_unit != right_unit and not (
                not left_unit
                or not right_unit
                or left_unit in pseudo_units
                or right_unit in pseudo_units
            ):
                return False
        left_number = cls._first_number(left.get("value"))
        right_number = cls._first_number(right.get("value"))
        if left_number is not None or right_number is not None:
            if left_number is None or right_number is None:
                return False
            if left_number == right_number:
                return True
            if left_number == 0 or right_number == 0:
                return False
            return math.isclose(left_number, right_number, rel_tol=1e-6, abs_tol=1e-9)
        return left_slot[3] == right_slot[3]

    # ------------------------------------------------------------------
    # Label-quality cleaning: strip parent entity title words and
    # trailing file-index suffixes from attribute labels.
    # ------------------------------------------------------------------

    _LABEL_CLEAN_STOP_WORDS: frozenset[str] = frozenset({
        "the", "of", "for", "in", "and", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "can",
        "shall", "this", "that", "these", "those", "it", "its", "they",
        "them", "their", "there", "here", "where", "when", "how", "why",
        "what", "which", "who", "whom", "whose", "data", "set", "dataset",
        "record", "file", "source", "entity", "activity",
    })

    @classmethod
    def _parent_entity_title_words(
        cls,
        document: dict[str, Any],
        parent_array_path: str,
    ) -> set[str]:
        """Extract significant words from the parent entity title for label cleaning."""
        parts = [p for p in parent_array_path.strip("/").split("/") if p]
        if len(parts) < 2:
            return set()
        entity_path = "/" + "/".join(parts[:-1])
        entity = cls._value_at_json_pointer(document, entity_path)
        if not isinstance(entity, dict):
            return set()
        title = entity.get("title", "")
        if not title:
            return set()
        normalized = cls._normalized_text(title)
        return {
            w for w in normalized.split()
            if len(w) > 2 and w not in cls._LABEL_CLEAN_STOP_WORDS
        }

    @classmethod
    def _clean_attribute_label(
        cls,
        label: str,
        entity_words: set[str],
    ) -> str | None:
        """Strip parent entity title words and trailing file-index suffix from *label*.

        Returns the cleaned label if it changed, or None when no change is needed.
        """
        if not label or not entity_words:
            return None
        cleaned = label
        for word in sorted(entity_words, key=len, reverse=True):
            cleaned = re.sub(r"\b" + re.escape(word) + r"\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*-\s*\d+\s*$", "", cleaned)
        cleaned = re.sub(r"\s*-\s*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or cleaned == label:
            return None
        return cleaned

    @classmethod
    def _attribute_label_quality_writes(
        cls,
        document: dict[str, Any],
    ) -> tuple[list[SchemaConstrainedWrite], int]:
        """Generate replace writes for attributes whose labels contain parent entity
        title fragments or trailing file-index suffixes."""
        writes: list[SchemaConstrainedWrite] = []
        for parent_path, items in cls._attribute_array_items(document):
            entity_words = cls._parent_entity_title_words(document, parent_path)
            if not entity_words:
                continue
            for index in range(len(items) - 1, -1, -1):
                item = items[index]
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "")
                qty_type = str(item.get("has_quantity_type") or "")
                cleaned_title = cls._clean_attribute_label(title, entity_words)
                cleaned_qty = cls._clean_attribute_label(qty_type, entity_words)
                if cleaned_title is None and cleaned_qty is None:
                    continue
                cleaned_item = dict(item)
                if cleaned_title is not None:
                    cleaned_item["title"] = cleaned_title
                if cleaned_qty is not None:
                    cleaned_item["has_quantity_type"] = cleaned_qty
                writes.append(
                    SchemaConstrainedWrite(
                        target_path=f"{parent_path}/{index}",
                        mode="replace",
                        value=cleaned_item,
                        reason="Cleaned attribute label: removed parent entity title fragment and file-index suffix.",
                    )
                )
        return writes, len(writes)

    @classmethod
    def _attribute_parent_survivor_score(
        cls,
        parent_path: str,
        instance: dict[str, Any],
    ) -> tuple[int, int, int, int]:
        family = cls._parent_attribute_family(instance)
        preferred = 0
        if family == "spectrometer_frequency":
            if re.fullmatch(r"/was_generated_by/\d+/has_quantitative_attribute", parent_path):
                preferred = 4
            elif "/carried_out_by/" in parent_path or "/had_input_entity/" in parent_path:
                preferred = 3
            elif "/evaluated_entity/" in parent_path or "/had_output_entity/" in parent_path:
                preferred = 0
        elif family in {"software_version", "acquisition_parameter", "instrument_parameter", "experiment_type"}:
            if "/had_input_entity/" in parent_path:
                preferred = 4
            elif "/carried_out_by/" in parent_path:
                preferred = 3
            elif re.fullmatch(r"/was_generated_by/\d+/has_quantitative_attribute", parent_path):
                preferred = 2
        elif family in {"solvent", "activity_target"}:
            if "/evaluated_entity/" in parent_path or "/is_about_entity/" in parent_path:
                preferred = 4
            elif re.fullmatch(r"/was_generated_by/\d+/(has_quantitative_attribute|has_qualitative_attribute)", parent_path):
                preferred = 2
        elif family == "pulse_program":
            if "/had_input_entity/" in parent_path or "/carried_out_by/" in parent_path:
                preferred = 4
            elif re.fullmatch(r"/was_generated_by/\d+/has_qualitative_attribute", parent_path):
                preferred = 3
        elif family == "file_format":
            if "/had_input_entity/" in parent_path or "/had_output_entity/" in parent_path:
                preferred = 4
        elif "/evaluated_entity/" in parent_path:
            preferred = 2
        elif "/is_about_entity/" in parent_path:
            preferred = 1
        precision, populated, label_score = cls._attribute_survivor_score(instance)
        return preferred, precision, populated, label_score

    @classmethod
    def _range_bounds_from_evidence(
        cls,
        *,
        attribute: dict[str, Any],
        evidence: list[RequirementEvidenceItem],
    ) -> tuple[float, float, str] | None:
        for evidence_item in evidence:
            for text in (evidence_item.claim, evidence_item.evidence_text):
                normalized = cls._normalized_text(text)
                if not (
                    "range" in normalized
                    or "spanning" in normalized
                    or re.search(r"\bfrom\b.+\bto\b", normalized)
                ):
                    continue
                numbers = list(dict.fromkeys(cls._numbers_from_text(text)))
                if len(numbers) == 2:
                    support = " ".join(
                        (evidence_item.claim, evidence_item.evidence_text, evidence_item.source_context)
                    )
                    return min(numbers), max(numbers), support
        attribute_text = " ".join(
            str(attribute.get(key) or "")
            for key in ("title", "description", "has_quantity_type", "unit")
        )
        numbers = list(dict.fromkeys(cls._numbers_from_text(attribute_text)))
        if len(numbers) != 2:
            return None
        return min(numbers), max(numbers), attribute_text

    @classmethod
    def _range_bounds_from_document_description(
        cls,
        document: dict[str, Any],
    ) -> tuple[float, float, str] | None:
        descriptions = document.get("description") or []
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        number = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
        pattern = re.compile(
            rf"\b(?:spanning|range(?:\s+from)?|from)\s+({number})\s*(?:-|\bto\b)\s*({number})",
            flags=re.IGNORECASE,
        )
        for description in descriptions:
            text = str(description or "")
            match = pattern.search(text)
            if not match:
                continue
            low, high = float(match.group(1)), float(match.group(2))
            return min(low, high), max(low, high), text
        return None

    @classmethod
    def _range_bounds_from_sibling_attributes(
        cls,
        *,
        attribute: dict[str, Any],
        siblings: list[Any],
    ) -> tuple[float, float, str] | None:
        base = cls._range_base_label(attribute)
        expected = {
            "wavenumber": {"min wavenumber", "max wavenumber"},
            "transmittance": {"min transmittance", "max transmittance"},
        }.get(base)
        if not expected:
            return None
        values: dict[str, float] = {}
        supporting_parts: list[str] = []
        for sibling in siblings:
            if sibling is attribute or not isinstance(sibling, dict):
                continue
            label = cls._normalized_attribute_label(
                " ".join(str(sibling.get(key) or "") for key in ("has_quantity_type", "title"))
            )
            value = cls._first_number(sibling.get("value"))
            if label not in expected or value is None:
                continue
            values[label] = value
            supporting_parts.extend(
                str(sibling.get(key) or "") for key in ("title", "has_quantity_type", "unit")
            )
        minimum = values.get(f"min {base}")
        maximum = values.get(f"max {base}")
        if minimum is None or maximum is None:
            return None
        return min(minimum, maximum), max(minimum, maximum), " ".join(supporting_parts)

    @classmethod
    def _range_bound_matches_existing(
        cls,
        *,
        candidate: dict[str, Any],
        existing: dict[str, Any],
    ) -> bool:
        candidate_label = cls._normalized_attribute_label(
            " ".join(str(candidate.get(key) or "") for key in ("has_quantity_type", "title"))
        )
        existing_label = cls._normalized_attribute_label(
            " ".join(str(existing.get(key) or "") for key in ("has_quantity_type", "title"))
        )
        if not candidate_label or candidate_label != existing_label:
            return False
        candidate_value = cls._first_number(candidate.get("value"))
        existing_value = cls._first_number(existing.get("value"))
        if candidate_value is None or existing_value is None:
            return False
        candidate_text = str(candidate.get("value"))
        decimal_places = len(candidate_text.partition(".")[2].split("e", maxsplit=1)[0])
        rounding_tolerance = 0.5 * (10 ** -decimal_places) if decimal_places else 0.0
        return math.isclose(
            candidate_value,
            existing_value,
            rel_tol=1e-9,
            abs_tol=max(1e-12, rounding_tolerance),
        )

    @classmethod
    def _attribute_array_items(cls, document: Any, *, path: str = "") -> list[tuple[str, list[Any]]]:
        result: list[tuple[str, list[Any]]] = []
        if isinstance(document, dict):
            for key, value in document.items():
                child_path = f"{path}/{key}" if path else f"/{key}"
                if key in {"has_quantitative_attribute", "has_qualitative_attribute"} and isinstance(value, list):
                    result.append((child_path, value))
                result.extend(cls._attribute_array_items(value, path=child_path))
        elif isinstance(document, list):
            for index, value in enumerate(document):
                child_path = f"{path}/{index}" if path else f"/{index}"
                result.extend(cls._attribute_array_items(value, path=child_path))
        return result

    @classmethod
    def _attribute_looks_like_bad_range(cls, attribute: dict[str, Any]) -> bool:
        text = " ".join(
            cls._normalized_text(attribute.get(key))
            for key in ("title", "description", "has_quantity_type", "unit")
        )
        return "range" in text or re.search(r"\bto\b", text) is not None or cls._normalized_unit(attribute.get("unit")) == "to"

    @staticmethod
    def _numbers_from_text(text: str) -> list[float]:
        text = re.sub(r"(?<=\d)-(?=\d)", " ", text)
        text = re.sub(r"\b1\s*/\s*cm\b|\bcm\s*-?\s*1\b", " ", text, flags=re.IGNORECASE)
        values: list[float] = []
        for match in re.finditer(r"(?<![a-zA-Z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.IGNORECASE):
            try:
                values.append(float(match.group(0)))
            except ValueError:
                continue
        return values

    @classmethod
    def _range_base_label(cls, attribute: dict[str, Any]) -> str:
        text = cls._normalized_attribute_label(
            " ".join(str(attribute.get(key) or "") for key in ("has_quantity_type", "title", "description"))
        )
        if "transmittance" in text or text in {"y", "max y", "min y"}:
            return "transmittance"
        if "wavenumber" in text or "wavelength" in text or text in {"x", "max x", "min x", "first x", "last x"}:
            return "wavenumber"
        return text or "range"

    @classmethod
    def _range_unit_text(cls, text: str) -> str:
        normalized = cls._normalized_text(text)
        for candidate in ("1/cm", "1 cm", "cm-1", "percent", "%"):
            if candidate in normalized:
                return cls._normalized_unit(candidate)
        return ""

    @staticmethod
    def _range_attribute(*, base: str, bound: str, value: float, unit: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "title": f"{bound.title()} {base}",
            "description": f"{bound.title()} {base}: {value:g}",
            "value": value,
            "has_quantity_type": f"{bound} {base}",
        }
        if unit:
            result["unit"] = unit
        return result

    @staticmethod
    def _legacy_json_patch_response_reason(output: Any) -> str:
        if not isinstance(output, dict):
            return ""
        writes = output.get("writes")
        if not isinstance(writes, list):
            return ""
        for write in writes:
            if isinstance(write, dict) and ("op" in write or ("path" in write and "target_path" not in write)):
                return "Semantic reconstruction returned JSON Patch syntax; expected schema action envelope."
        return ""

    @staticmethod
    def _legacy_semantic_diagnosis_response_reason(output: Any) -> str:
        if not isinstance(output, dict):
            return "Semantic reconstruction diagnosis returned non-object output."
        if "operations" in output or "writes" in output:
            return "Semantic reconstruction returned patch/action syntax; expected diagnosis defects only."
        if "defects" not in output:
            return "Semantic reconstruction diagnosis missing defects array."
        return ""

    @classmethod
    def _semantic_path_allowed(cls, target_path: str, allowed_paths: list[str]) -> bool:
        target = target_path.rstrip("/") or "/"
        for allowed_path in allowed_paths:
            allowed = allowed_path.rstrip("/") or "/"
            if target == allowed or target.startswith(f"{allowed}/"):
                return True
        return False

    @classmethod
    def _semantic_reconstruction_candidate_writes(
        cls,
        writes: list[SchemaConstrainedWrite],
    ) -> list[SchemaConstrainedWrite]:
        candidates: list[SchemaConstrainedWrite] = []
        for write in writes:
            if write.mode == "append" and len(write.items) > 1:
                candidates.extend(write.model_copy(update={"items": [item]}) for item in write.items)
            else:
                candidates.append(write)
        # Sort writes so that merges and removes targeting the same array path
        # are applied from highest index to lowest, preventing index-shift errors.
        def _sort_key(write: SchemaConstrainedWrite) -> tuple[str, int]:
            indices: list[int] = []
            if write.survivor_index is not None:
                indices.append(write.survivor_index)
            indices.extend(write.merged_indices)
            parts = [p for p in write.target_path.strip("/").split("/") if p]
            if parts:
                try:
                    indices.append(int(parts[-1]))
                except ValueError:
                    pass
            max_index = max(indices) if indices else 0
            parent = write.target_path.rsplit("/", 1)[0] if "/" in write.target_path else write.target_path
            return (parent, -max_index)
        candidates.sort(key=_sort_key)
        return candidates

    @classmethod
    def _semantic_reconstruction_rejected_actions(
        cls,
        *,
        writes: list[SchemaConstrainedWrite],
        allowed_paths: list[str],
    ) -> list[str]:
        return [
            f"Action target outside allowed semantic reconstruction paths: {write.target_path}"
            for write in writes
            if cls._semantic_reconstruction_action_rejected(write=write, allowed_paths=allowed_paths)
        ]

    @classmethod
    def _semantic_reconstruction_action_rejected(
        cls,
        *,
        write: SchemaConstrainedWrite,
        allowed_paths: list[str],
    ) -> bool:
        target = write.target_path.rstrip("/") or "/"
        allowed = [path.rstrip("/") or "/" for path in allowed_paths if path]
        for allowed_path in allowed:
            if target == allowed_path:
                return False
            if write.mode == "remove" and target.startswith(f"{allowed_path}/"):
                return False
        return True

    @classmethod
    def _validate_semantic_reconstruction_action(
        cls,
        *,
        document: dict[str, Any],
        write: SchemaConstrainedWrite,
    ) -> list[str]:
        target_parts = [part for part in write.target_path.strip("/").split("/") if part]
        relation_names = {"is_about_entity", "is_about_activity", "evaluated_entity", "evaluated_activity"}
        relation_target = next((part for part in target_parts if part in relation_names), "")
        if relation_target and write.mode in {"append", "replace"}:
            proposed = list(write.items) if write.mode == "append" else [write.value]
            invalid = [
                item
                for item in proposed
                if not isinstance(item, dict)
                or not (
                    cls._semantic_value_present(item.get("title"))
                    or cls._semantic_value_present(item.get("description"))
                )
                or cls._relation_item_looks_like_requirement_placeholder(item)
            ]
            if invalid:
                return [f"Relation item at {write.target_path} must be a concrete evidence-backed object, not a placeholder."]
        if (
            len(target_parts) >= 3
            and target_parts[0] == "was_generated_by"
            and target_parts[1].isdigit()
            and target_parts[2] == "evaluated_activity"
            and write.mode in {"append", "replace"}
        ):
            activity_index = int(target_parts[1])
            activities = document.get("was_generated_by") or []
            activity = activities[activity_index] if activity_index < len(activities) else None
            activity_id = str(activity.get("id") or "").strip() if isinstance(activity, dict) else ""
            proposed = list(write.items) if write.mode == "append" else [write.value]
            if activity_id and any(
                isinstance(item, dict) and str(item.get("id") or "").strip() == activity_id
                for item in proposed
            ):
                return [f"evaluated_activity must not self-reference {activity_id} at {write.target_path}."]
        if write.mode != "merge":
            return []
        array_value = cls._value_at_json_pointer(document, write.target_path)
        if not isinstance(array_value, list):
            return [f"Merge target path is not an array: {write.target_path}"]
        survivor_index = write.survivor_index
        if survivor_index is None:
            return ["Merge action missing survivor_index."]
        indices = list(dict.fromkeys(write.merged_indices))
        all_indices = [survivor_index] + indices
        invalid = [index for index in all_indices if index < 0 or index >= len(array_value)]
        if invalid:
            return [f"Merge index out of range at {write.target_path}: {invalid}"]
        survivor = array_value[survivor_index]
        if not isinstance(survivor, dict):
            return [f"Merge survivor is not an object at {write.target_path}/{survivor_index}."]
        if not cls._attribute_semantic_slot(parent_path=write.target_path, instance=survivor):
            return [f"Merge survivor has no semantic slot at {write.target_path}/{survivor_index}."]
        rejected: list[str] = []
        for index in indices:
            item = array_value[index]
            if not isinstance(item, dict) or not (
                cls._attributes_semantically_compatible(
                    parent_path=write.target_path,
                    left=survivor,
                    right=item,
                )
                or cls._attributes_value_equivalent(survivor, item)
            ):
                rejected.append(f"Merge item at {write.target_path}/{index} is not semantically compatible with survivor.")
        return rejected

    @staticmethod
    def _relation_item_looks_like_requirement_placeholder(item: dict[str, Any]) -> bool:
        identifier = str(item.get("id") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        description = str(item.get("description") or "").strip().lower()
        placeholder_terms = {
            "activity_evaluation_target",
        }
        if identifier in placeholder_terms or title in placeholder_terms:
            return True
        return "requirement" in description and "evidence" not in description

    @classmethod
    def _semantic_patch_paths_allowed(cls, operations: list[Any], allowed_paths: list[str]) -> bool:
        return all(
            isinstance(cls._patch_operation_dict(operation), dict)
            and cls._semantic_patch_path_allowed(str(cls._patch_operation_dict(operation).get("path", "")), allowed_paths)
            and (
                "from" not in cls._patch_operation_dict(operation)
                or cls._semantic_patch_path_allowed(str(cls._patch_operation_dict(operation).get("from", "")), allowed_paths)
            )
            for operation in operations
        )

    @staticmethod
    def _patch_operation_dict(operation: Any) -> dict[str, Any]:
        if hasattr(operation, "model_dump"):
            return operation.model_dump(mode="json", by_alias=True, exclude_none=True)
        return operation

    @staticmethod
    def _semantic_patch_path_allowed(path: str, allowed_paths: list[str]) -> bool:
        return path.startswith("/") and any(
            ProjectionService._patch_path_allowed_for_target(path, allowed_path)
            for allowed_path in allowed_paths
        )

    @staticmethod
    def _semantic_patch_changed_paths(operations: list[Any]) -> list[str]:
        return list(
            dict.fromkeys(
                str(operation.get("path", ""))
                for operation in (ProjectionService._patch_operation_dict(operation) for operation in operations)
                if operation.get("path")
            )
        )

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
                evidence_quality={
                    "requirement_id": item.requirement_id,
                    "iteration_kind": record.iteration_kind,
                    "defect_type": record.defect_type,
                    "diagnosed_defects_count": record.diagnosed_defects_count,
                    "compiled_actions_count": record.compiled_actions_count,
                    "synthesis_calls_count": record.synthesis_calls_count,
                    "applied_actions_count": record.applied_actions_count,
                    "rejected_actions_count": record.rejected_actions_count,
                    "rejected_reasons": record.rejected_reasons,
                    "diagnosed_defects": record.diagnosed_defects,
                    "compiled_actions": record.compiled_actions,
                },
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
        for configured_requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS:
            requirement = self._runtime_semantic_requirement(
                requirement=configured_requirement,
                document=document,
            )
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
                iteration_kind="semantic_diagnosis",
            )
            selected_evidence, context_window = select_requirement_evidence_packet(
                requirement=requirement,
                assessment=seed_item,
                evidence_context=evidence_context,
            )
            draft_excerpt = (
                self._attribute_parent_placement_excerpt(document)
                if requirement.requirement_id == "attribute_parent_placement"
                else {
                    path: self._value_at_json_pointer(document, path)
                    for path in requirement.target_paths
                }
            )
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
            evaluated.target_paths = list(requirement.target_paths)
            evaluated.selected_evidence = selected_evidence
            evaluated.context_window = context_window
            self._guard_semantic_requirement_assessment(
                requirement=requirement,
                document=document,
                item=evaluated,
            )
            evaluated.iteration_kind = "semantic_diagnosis"
            items.append(evaluated)
        score_requirement_items(items)
        return items

    @staticmethod
    def _runtime_semantic_requirement(
        *,
        requirement: DcatRequirement,
        document: dict[str, Any],
    ) -> DcatRequirement:
        attribute_requirement_ids = {
            "attribute_parent_placement",
            "attribute_range_decomposition",
        }
        if requirement.requirement_id not in (
            "activity_evaluation_target",
            *attribute_requirement_ids,
        ):
            return requirement
        activities = document.get("was_generated_by")
        activity_indices = [
            index
            for index, activity in enumerate(activities if isinstance(activities, list) else [])
            if isinstance(activity, dict)
        ] or [0]
        evaluated_attr_paths = [
            f"/was_generated_by/{index}/{relation}/{attr_index}/has_quantitative_attribute"
            for index in activity_indices
            for relation in ("evaluated_entity", "evaluated_activity")
            for attr_index in range(
                len(
                    next(
                        (
                            activity
                            for i, activity in enumerate(activities if isinstance(activities, list) else [])
                            if i == index and isinstance(activity, dict)
                        ),
                        {},
                    ).get(relation, [])
                )
                if isinstance(activities, list)
                else 0
            )
        ] or [
            f"/was_generated_by/{index}/{relation}/0/has_quantitative_attribute"
            for index in activity_indices
            for relation in ("evaluated_entity", "evaluated_activity")
        ]
        if requirement.requirement_id == "activity_evaluation_target":
            activity_paths = [
                f"/was_generated_by/{index}/{relation}"
                for index in activity_indices
                for relation in ("evaluated_entity", "evaluated_activity")
            ]
            return requirement.model_copy(update={"target_paths": activity_paths})
        # Attribute requirements: add evaluated_entity/evaluated_activity attribute paths
        existing_paths = list(requirement.target_paths)
        for index in activity_indices:
            for relation in ("evaluated_entity", "evaluated_activity"):
                existing_paths.append(
                    f"/was_generated_by/{index}/{relation}/0/has_quantitative_attribute"
                )
        return requirement.model_copy(update={"target_paths": existing_paths})

    @classmethod
    def _guard_semantic_requirement_assessment(
        cls,
        *,
        requirement: DcatRequirement,
        document: dict[str, Any],
        item: RequirementReportItem,
    ) -> None:
        target_present = any(
            cls._semantic_value_present(cls._value_at_json_pointer(document, path))
            for path in requirement.target_paths
        )
        if item.status == "fulfilled" and not target_present:
            item.status = "partial" if item.selected_evidence else "missing"
            item.quality = 0.5 if item.selected_evidence else 0.0
            item.applicable = True
            item.rationale = (
                "Evidence establishes applicability, but none of the requirement target paths "
                "contains a placed draft value."
            )
        if requirement.requirement_id == "attribute_range_decomposition":
            if cls._document_has_bad_range_attribute(document):
                item.status = "partial"
                item.quality = 0.5
                item.applicable = True
                item.rationale = "Backend inspection found a collapsed or malformed range attribute."
            else:
                item.status = "fulfilled"
                item.quality = 1.0
                item.applicable = True
                item.rationale = "Backend inspection found no collapsed or malformed range attributes."
        if requirement.requirement_id == "technical_agent_kind":
            agents = [
                agent
                for activity in document.get("was_generated_by", [])
                if isinstance(activity, dict)
                for agent in (activity.get("carried_out_by") or [])
                if isinstance(agent, dict)
            ]
            if agents and all(cls._agent_is_technical(agent) for agent in agents):
                item.status = "fulfilled"
                item.quality = 1.0
                item.applicable = True
                item.rationale = "Backend inspection found only instruments, software, or devices in carried_out_by."
        if requirement.requirement_id == "activity_evaluation_target":
            activities = [
                activity
                for activity in (document.get("was_generated_by") or [])
                if isinstance(activity, dict)
            ]
            valid_target_counts = [cls._concrete_activity_target_count(activity) for activity in activities]
            self_references = cls._evaluated_activity_self_reference_paths(document)
            if not activities or not any(valid_target_counts):
                item.status = "missing"
                item.quality = 0.0
                item.applicable = True
                item.rationale = "Backend inspection found no evaluation target on any data-generating activity."
            elif self_references or any(count == 0 for count in valid_target_counts):
                item.status = "partial"
                item.quality = 0.5
                item.applicable = True
                item.rationale = (
                    "Backend inspection found a self-referential evaluated activity."
                    if self_references
                    else "Backend inspection found at least one data-generating activity without an evaluation target."
                )
            else:
                item.status = "fulfilled"
                item.quality = 1.0
                item.applicable = True
                item.rationale = "Backend inspection found at least one concrete evaluation target on every data-generating activity."
        if requirement.requirement_id == "method_plan_presence":
            activities = [
                activity
                for activity in (document.get("was_generated_by") or [])
                if isinstance(activity, dict)
            ]
            meaningful = [
                activity.get("realized_plan") not in (None, "", [], {})
                and not cls._semantic_absence_placeholder(activity.get("realized_plan"))
                for activity in activities
            ]
            if not activities or not any(meaningful):
                item.status = "missing"
                item.quality = 0.0
                item.applicable = True
                item.rationale = "Backend inspection found no evidence-backed method, procedure, protocol, or plan."
            elif not all(meaningful):
                item.status = "partial"
                item.quality = 0.5
                item.applicable = True
                item.rationale = "Backend inspection found at least one data-generating activity without an evidence-backed plan."
        if not item.rationale.strip():
            item.rationale = f"Semantic evaluator returned {item.status} without an explanation."
        item.weighted_score = item.weight * item.quality if item.applicable else 0.0

    @staticmethod
    def _semantic_value_present(value: Any) -> bool:
        return value not in (None, "", [], {})

    @classmethod
    def _relation_item_is_concrete(cls, item: Any) -> bool:
        return isinstance(item, dict) and (
            cls._semantic_value_present(item.get("title"))
            or cls._semantic_value_present(item.get("description"))
        )

    @classmethod
    def _semantic_absence_placeholder(cls, value: Any) -> bool:
        if isinstance(value, dict):
            text = " ".join(str(part) for part in value.values() if not isinstance(part, (dict, list)))
            nested = any(cls._semantic_absence_placeholder(part) for part in value.values() if isinstance(part, (dict, list)))
        elif isinstance(value, list):
            text = " ".join(str(part) for part in value if not isinstance(part, (dict, list)))
            nested = any(cls._semantic_absence_placeholder(part) for part in value if isinstance(part, (dict, list)))
        else:
            text = str(value or "")
            nested = False
        normalized = cls._normalized_text(text)
        return nested or any(
            phrase in normalized
            for phrase in (
                "no explicit method",
                "no method evidence",
                "no evidence available",
                "not available",
                "not specified",
                "unknown method",
            )
        )

    @classmethod
    def _concrete_activity_target_count(cls, activity: dict[str, Any]) -> int:
        activity_id = str(activity.get("id") or "").strip()
        count = 0
        for relation in ("evaluated_entity", "evaluated_activity"):
            targets = activity.get(relation) or []
            if not isinstance(targets, list):
                continue
            for target in targets:
                if not isinstance(target, dict):
                    continue
                target_id = str(target.get("id") or "").strip()
                if relation == "evaluated_activity" and activity_id and target_id == activity_id:
                    continue
                if cls._relation_item_looks_like_requirement_placeholder(target):
                    continue
                if cls._semantic_value_present(target_id) or cls._relation_item_is_concrete(target):
                    count += 1
        return count

    @staticmethod
    def _evaluated_activity_self_reference_paths(document: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for activity_index, activity in enumerate(document.get("was_generated_by") or []):
            if not isinstance(activity, dict):
                continue
            activity_id = str(activity.get("id") or "").strip()
            if not activity_id:
                continue
            for target_index, target in enumerate(activity.get("evaluated_activity") or []):
                if isinstance(target, dict) and str(target.get("id") or "").strip() == activity_id:
                    paths.append(f"/was_generated_by/{activity_index}/evaluated_activity/{target_index}")
        return paths

    @classmethod
    def _document_has_bad_range_attribute(cls, document: dict[str, Any]) -> bool:
        return any(
            isinstance(attribute, dict) and cls._attribute_looks_like_bad_range(attribute)
            for _, attributes in cls._attribute_array_items(document)
            for attribute in attributes
        )

    @classmethod
    def _agent_is_technical(cls, agent: dict[str, Any]) -> bool:
        parts: list[str] = []
        for key in ("id", "title", "description", "type", "rdf_type"):
            value = agent.get(key)
            if isinstance(value, dict):
                parts.extend(str(value.get(field) or "") for field in ("id", "title", "from_CV"))
            else:
                parts.append(str(value or ""))
        text = cls._normalized_text(" ".join(parts))
        return bool(re.search(r"\b(software|instrument|device|equipment|sensor|application)\b", text))

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

    async def _apply_provenance_core_construction(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
    ) -> dict[str, Any]:
        del evidence_context
        if self.ollama_client is None:
            return document
        orientation_context = self._provenance_core_orientation_context(state)
        if not self._semantic_value_present(orientation_context) and not (
            self._semantic_value_present(document.get("description"))
            or self._semantic_value_present(document.get("title"))
        ):
            return document
        progress.stage = "core_construction"
        self._update_progress(data_package_id, progress)
        response = await self._generate_provenance_core_intents(
            data_package_id=data_package_id,
            document=document,
            orientation_context=orientation_context,
        )
        if response is None:
            return document
        updated, changed_paths = self._apply_provenance_core_intents(
            data_package_id=data_package_id,
            document=document,
            response=response,
        )
        if not changed_paths:
            return document
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        if not validation.valid:
            return document
        self._record_provenance_core_projection(
            state=state,
            progress=progress,
            changed_paths=changed_paths,
            response=response,
        )
        state.generated_final_draft = updated
        progress.generated_final_draft = updated
        self._save_run_state(data_package_id, state)
        self._update_progress(data_package_id, progress)
        return updated

    async def _generate_provenance_core_intents(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        orientation_context: dict[str, Any],
    ) -> _ProvenanceCoreIntentResponse | None:
        prompt = self._provenance_core_prompt(
            document=document,
            orientation_context=orientation_context,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=PROVENANCE_CORE_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=_ProvenanceCoreIntentResponse,
                system_components=[("provenance_core_system_prompt", PROVENANCE_CORE_SYSTEM_PROMPT)],
                prompt_components=[("provenance_core_prompt", prompt)],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("provenance_core_constructor"),
                agent_name="provenance_core_constructor",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="provenance_core_constructor",
            )
            return (
                result.output
                if isinstance(result.output, _ProvenanceCoreIntentResponse)
                else _ProvenanceCoreIntentResponse.model_validate(result.output)
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="provenance_core_constructor",
            )
            return None

    def _apply_provenance_core_intents(
        self,
        *,
        data_package_id: str,
        document: dict[str, Any],
        response: _ProvenanceCoreIntentResponse,
    ) -> tuple[dict[str, Any], list[str]]:
        current = self._clone_json_object(document)
        activities = current.get("was_generated_by")
        if not isinstance(activities, list) or not activities:
            activities = [{"id": f"{data_package_id}:activity:dataset-generation"}]
            current["was_generated_by"] = activities
        activity = activities[0]
        if not isinstance(activity, dict):
            activity = {"id": f"{data_package_id}:activity:dataset-generation"}
            activities[0] = activity

        changed_paths: list[str] = []
        title = response.activity_title.strip()
        if title and not self._semantic_value_present(activity.get("title")):
            activity["title"] = [title]
            changed_paths.append("/was_generated_by/0/title")
        description = response.activity_description.strip()
        if description and not self._semantic_value_present(activity.get("description")):
            activity["description"] = [description]
            changed_paths.append("/was_generated_by/0/description")

        if response.plan is not None and not self._semantic_value_present(activity.get("realized_plan")):
            plan = self._core_plan_instance(response.plan)
            if plan is not None:
                activity["realized_plan"] = plan
                changed_paths.append("/was_generated_by/0/realized_plan")

        relation_specs = [
            ("carried_out_by", "agentic-entity", response.agents, 3, False),
            ("evaluated_entity", "evaluated-entity", response.evaluated_entities, 3, False),
            ("evaluated_activity", "evaluated-activity", response.evaluated_activities, 3, True),
            ("had_input_entity", "input-entity", response.input_entities, 3, False),
            ("had_input_activity", "input-activity", response.input_activities, 3, True),
            ("had_output_entity", "output-entity", response.output_entities, 3, False),
        ]
        for field_name, kind, intents, limit, list_text in relation_specs:
            allowed = limit
            if field_name == "evaluated_activity":
                allowed = max(0, 3 - len(activity.get("evaluated_entity") or []))
            changed_paths.extend(
                self._append_core_relation_intents(
                    data_package_id=data_package_id,
                    activity=activity,
                    field_name=field_name,
                    kind=kind,
                    intents=intents,
                    limit=allowed,
                    list_text=list_text,
                )
            )
        return remove_null_values(current), changed_paths

    def _append_core_relation_intents(
        self,
        *,
        data_package_id: str,
        activity: dict[str, Any],
        field_name: str,
        kind: str,
        intents: list[_CoreObjectIntent],
        limit: int,
        list_text: bool,
    ) -> list[str]:
        if limit <= 0:
            return []
        values = activity.get(field_name)
        if not isinstance(values, list):
            values = []
            activity[field_name] = values
        changed_paths: list[str] = []
        seen = {
            self._normalized_text(item.get("id") or item.get("title") or item.get("description"))
            for item in values
            if isinstance(item, dict)
        }
        for intent in intents:
            if len(values) >= limit:
                break
            instance = self._core_relation_instance(
                data_package_id=data_package_id,
                kind=kind,
                intent=intent,
                index=len(values),
                list_text=list_text,
            )
            if instance is None:
                continue
            key = self._normalized_text(instance.get("id") or instance.get("title") or instance.get("description"))
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(instance)
            changed_paths.append(f"/was_generated_by/0/{field_name}/{len(values) - 1}")
        if not values:
            activity.pop(field_name, None)
        return changed_paths

    def _core_relation_instance(
        self,
        *,
        data_package_id: str,
        kind: str,
        intent: _CoreObjectIntent,
        index: int,
        list_text: bool,
    ) -> dict[str, Any] | None:
        title = intent.title.strip()
        description = intent.description.strip()
        label = title or description
        if not label:
            return None
        instance: dict[str, Any] = {
            "id": self._core_generated_id(data_package_id=data_package_id, kind=kind, label=label, index=index),
        }
        if title:
            instance["title"] = [title] if list_text else title
        if description:
            instance["description"] = [description] if list_text else description
        return instance

    @staticmethod
    def _core_plan_instance(intent: _CoreObjectIntent) -> dict[str, Any] | None:
        title = intent.title.strip()
        description = intent.description.strip()
        if not title and not description:
            return None
        plan: dict[str, Any] = {}
        if title:
            plan["title"] = title
        if description:
            plan["description"] = description
        return plan

    @staticmethod
    def _core_generated_id(
        *,
        data_package_id: str,
        kind: str,
        label: str,
        index: int,
    ) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        return f"{data_package_id}:{kind}:{slug or index + 1}"

    @classmethod
    def _record_provenance_core_projection(
        cls,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        changed_paths: list[str],
        response: _ProvenanceCoreIntentResponse,
    ) -> None:
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier="provenance_core:generic",
                object_kind="ProvenanceCore",
                source_evidence=None,
                evidence_note_identifiers=[],
                status="projected",
                projected_paths=changed_paths,
                target_path="/was_generated_by/0",
                target_class="DataGeneratingActivity",
                planner_status="generic_provenance_core",
                planner_reason=response.answer,
                evidence_quality={
                    "requirement_id": "generic_provenance_core",
                    "source_packet": "orientation_context",
                    "construction_strategy": "backend_shaped_core_intents",
                },
                merge_status="applied",
                reason=response.answer or "Projected generic provenance core.",
            )
        )
        progress.projection_ledger = state.projection_ledger

    @staticmethod
    def _provenance_core_orientation_context(state: ExtractionRunState) -> dict[str, Any]:
        context: dict[str, Any] = {}

        file_summaries = compact_file_summaries_for_shallow_projection(
            initial_file_summaries=state.initial_file_summaries,
            ranked_files=state.ranked_files,
        ).strip()
        if file_summaries:
            context["file_summaries"] = "\n".join(
                ProjectionService._provenance_core_shorten(line, 520)
                for line in file_summaries.splitlines()[:8]
            )

        return context

    @staticmethod
    def _provenance_core_shorten(text: str, limit: int) -> str:
        stripped = " ".join(text.split())
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 1].rstrip() + "..."

    @staticmethod
    def _provenance_core_prompt(
        *,
        document: dict[str, Any],
        orientation_context: dict[str, Any],
    ) -> str:
        payload = {
            "question": (
                "Construct the generic provenance core. Return one combined acquisition/processing "
                "DataGeneratingActivity with supported agents, evaluated targets, inputs, outputs, and plan. "
                "Put generated data products in output_entities, not evaluated_entities. Keep only profile-level "
                "objects, not low-level internal parameters. Silently classify every candidate by relation role "
                "before emitting it, and place each object in at most one relation."
            ),
            "limits": {
                "data_generating_activity": 1,
                "evaluated_entities_and_activities_total": 3,
                "agents": 3,
                "plan": 1,
                "input_entities": 3,
                "input_activities": 3,
                "output_entities": 3,
            },
            "current_draft": document,
            "orientation_context": orientation_context,
            "rules": [
                "Use only the orientation context in this prompt.",
                "Do not invent low-level file parameters as profile entities.",
                "Use output_entities for generated dataset products.",
                "Use evaluated_entities or evaluated_activities for the thing being measured, observed, or analyzed.",
                "Use agents only for participants that perform, control, execute, operate, or are responsible for the activity.",
                "Do not put vendors or manufacturers in agents unless the context says they performed, controlled, operated, or were responsible for the activity.",
                "Do not put methods, protocols, scripts, recipes, program definitions, parameter files, settings, or instruction sets in agents.",
                "Use plan for the method, protocol, workflow, recipe, program definition, or instruction set that specifies how the activity is done.",
                "Use input_entities for files, configurations, materials, or other entities consumed, read, transformed, or used by the activity.",
                "If an object could be either an agent or a plan/input, choose plan/input or omit it.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def _apply_parent_scoped_attribute_construction(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
    ) -> dict[str, Any]:
        if self.ollama_client is None:
            return document
        progress.stage = "attribute_construction"
        self._update_progress(data_package_id, progress)
        core_draft = self._clone_json_object(document)
        current = document
        for parent in self._attribute_parent_targets(core_draft):
            selected_evidence, context_window = self._parent_attribute_evidence_packet(
                parent=parent,
                evidence_context=evidence_context,
            )
            if not selected_evidence:
                self._record_parent_attribute_observation(
                    state=state,
                    progress=progress,
                    parent=parent,
                    status="skipped_no_parent_evidence",
                    reason="No evidence note was semantically local to this parent.",
                )
                continue
            response = await self._generate_parent_attribute_intents(
                data_package_id=data_package_id,
                parent=parent,
                selected_evidence=selected_evidence,
                context_window=context_window,
                core_draft=core_draft,
            )
            if response is None:
                self._record_parent_attribute_observation(
                    state=state,
                    progress=progress,
                    parent=parent,
                    status="skipped_llm_failed",
                    reason="Parent attribute constructor failed or returned invalid structured output.",
                    evidence_ids=[item.evidence_id for item in selected_evidence if item.evidence_id],
                )
                continue
            current = self._apply_parent_attribute_intents(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=current,
                parent=parent,
                response=response,
                selected_evidence=selected_evidence,
                validation_schema=validation_schema,
                state=state,
                progress=progress,
            )
            state.generated_final_draft = current
            progress.generated_final_draft = current
            self._save_run_state(data_package_id, state)
            self._update_progress(data_package_id, progress)
        return current

    def _cleanup_parent_attributes_after_construction(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        validation_schema: dict[str, Any],
    ) -> dict[str, Any]:
        writes, count = self._attribute_duplicate_coherence_writes(document)
        if not writes:
            return document
        updated, changed_paths, _, _, _, _ = self._apply_semantic_reconstruction_writes(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            document=document,
            writes=writes,
            reason=f"Post-attribute-construction duplicate cleanup: {count} duplicate attribute(s).",
            validation_schema=validation_schema,
            rejected_reasons=[],
        )
        return updated if changed_paths else document

    @classmethod
    def _attribute_parent_placement_excerpt(cls, document: dict[str, Any]) -> dict[str, Any]:
        parents: list[dict[str, Any]] = []
        for parent in cls._attribute_parent_targets(document):
            value = cls._value_at_json_pointer(document, parent["path"])
            if not isinstance(value, dict):
                continue
            parents.append(
                {
                    "path": parent["path"],
                    "class": parent.get("class", ""),
                    "title": value.get("title"),
                    "description": value.get("description"),
                    "has_quantitative_attribute": value.get("has_quantitative_attribute") or [],
                    "has_qualitative_attribute": value.get("has_qualitative_attribute") or [],
                }
            )
        return {"attribute_parents": parents}

    async def _generate_parent_attribute_intents(
        self,
        *,
        data_package_id: str,
        parent: dict[str, Any],
        selected_evidence: list[RequirementEvidenceItem],
        context_window: list[RequirementEvidenceItem],
        core_draft: dict[str, Any],
    ) -> _ParentAttributeIntentResponse | None:
        prompt = self._parent_attribute_prompt(
            parent=parent,
            selected_evidence=selected_evidence,
            context_window=context_window,
            core_draft=core_draft,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=PARENT_ATTRIBUTE_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=_ParentAttributeIntentResponse,
                system_components=[("parent_attribute_system_prompt", PARENT_ATTRIBUTE_SYSTEM_PROMPT)],
                prompt_components=[("parent_attribute_prompt", prompt)],
                token_budgeter=self._prompt_token_budgeter(),
                operation_id=self._prompt_operation_id("parent_attribute_constructor"),
                agent_name="parent_attribute_constructor",
                num_ctx=self.ollama_client.max_context_length,
            )
            self._record_llm_call_result(
                data_package_id=data_package_id,
                result=result,
                agent_name="parent_attribute_constructor",
            )
            return (
                result.output
                if isinstance(result.output, _ParentAttributeIntentResponse)
                else _ParentAttributeIntentResponse.model_validate(result.output)
            )
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="parent_attribute_constructor",
            )
            return None

    def _apply_parent_attribute_intents(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        parent: dict[str, Any],
        response: _ParentAttributeIntentResponse,
        selected_evidence: list[RequirementEvidenceItem],
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
    ) -> dict[str, Any]:
        current = document
        evidence_ids = [item.evidence_id for item in selected_evidence if item.evidence_id]
        if not response.quantitative_attributes and not response.qualitative_attributes:
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_empty_response",
                reason="Parent attribute constructor returned no parent-scoped attributes.",
                answer=response.answer,
                evidence_ids=evidence_ids,
            )
            return current
        for intent_index, (kind, intent) in enumerate(
            ("quantitative", item) for item in response.quantitative_attributes
        ):
            raw_intent = intent.model_dump(mode="json")
            instance = self._quantitative_attribute_instance_from_intent(intent)
            if instance is None:
                self._record_parent_attribute_observation(
                    state=state,
                    progress=progress,
                    parent=parent,
                    status="skipped_invalid_intent",
                    reason="Quantitative attribute intent was missing a required title or value.",
                    answer=response.answer,
                    attribute_kind=kind,
                    intent_index=intent_index,
                    evidence_ids=evidence_ids,
                    raw_intent=raw_intent,
                )
                continue
            current = self._append_parent_attribute(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=current,
                parent=parent,
                attribute_kind=kind,
                intent_index=intent_index,
                instance=instance,
                raw_intent=raw_intent,
                evidence_ids=evidence_ids,
                selected_evidence=selected_evidence,
                answer=response.answer,
                validation_schema=validation_schema,
                state=state,
                progress=progress,
            )
        for intent_index, (kind, intent) in enumerate(
            ("qualitative", item) for item in response.qualitative_attributes
        ):
            raw_intent = intent.model_dump(mode="json")
            instance = self._qualitative_attribute_instance_from_intent(intent)
            if instance is None:
                self._record_parent_attribute_observation(
                    state=state,
                    progress=progress,
                    parent=parent,
                    status="skipped_invalid_intent",
                    reason="Qualitative attribute intent was missing a required title or value.",
                    answer=response.answer,
                    attribute_kind=kind,
                    intent_index=intent_index,
                    evidence_ids=evidence_ids,
                    raw_intent=raw_intent,
                )
                continue
            current = self._append_parent_attribute(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                document=current,
                parent=parent,
                attribute_kind=kind,
                intent_index=intent_index,
                instance=instance,
                raw_intent=raw_intent,
                evidence_ids=evidence_ids,
                selected_evidence=selected_evidence,
                answer=response.answer,
                validation_schema=validation_schema,
                state=state,
                progress=progress,
            )
        return current

    def _append_parent_attribute(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        document: dict[str, Any],
        parent: dict[str, Any],
        attribute_kind: str,
        intent_index: int,
        instance: dict[str, Any],
        raw_intent: dict[str, Any],
        evidence_ids: list[str],
        answer: str,
        validation_schema: dict[str, Any],
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        selected_evidence: list[RequirementEvidenceItem] | None = None,
    ) -> dict[str, Any]:
        field_name = "has_quantitative_attribute" if attribute_kind == "quantitative" else "has_qualitative_attribute"
        target_path = f"{parent['path']}/{field_name}/-"
        placement_reason = self._parent_attribute_semantic_placement_reason(
            parent=parent,
            instance=instance,
            selected_evidence=selected_evidence or [],
        )
        if placement_reason:
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_semantic_placement",
                reason=placement_reason,
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=field_name,
                target_path=target_path,
                evidence_ids=evidence_ids,
                candidate_attribute=instance,
                raw_intent=raw_intent,
            )
            return document
        schema_path = target_path[:-2]
        schema_branch = self._schema_for_json_pointer(validation_schema, schema_path)
        if not schema_branch:
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_schema_missing",
                reason=f"No schema branch was available for {schema_path}.",
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=field_name,
                target_path=target_path,
                evidence_ids=evidence_ids,
                candidate_attribute=instance,
                raw_intent=raw_intent,
            )
            return document
        target_schema = (
            self._resolve_schema_node(schema_branch["items"], validation_schema)
            if isinstance(schema_branch.get("items"), dict)
            else schema_branch
        )
        duplicate_reason = self._duplicate_requirement_patch_reason(
            document=document,
            target_path=target_path,
            instance=instance,
        )
        if duplicate_reason:
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_duplicate",
                reason=duplicate_reason,
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=field_name,
                target_path=target_path,
                evidence_ids=evidence_ids,
                candidate_attribute=instance,
                raw_intent=raw_intent,
            )
            return document
        original = self._clone_json_object(document)
        try:
            updated = apply_evidence_instance(
                document=document,
                target_path=target_path,
                instance=instance,
                data_package_id=data_package_id,
                target_schema=target_schema,
            )
        except (ValueError, TypeError) as exc:
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_apply_failed",
                reason="apply_evidence_instance failed for the parent-scoped attribute candidate.",
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=field_name,
                target_path=target_path,
                evidence_ids=evidence_ids,
                candidate_attribute=instance,
                raw_intent=raw_intent,
                error=str(exc),
            )
            return original
        validation = self.profile_service.validate_document(
            identifier=profile_identifier,
            document=updated,
        )
        if not validation.valid:
            validation_reason = (
                validation.errors[0].message
                if getattr(validation, "errors", None)
                else "Profile validation failed after appending the parent-scoped attribute."
            )
            self._record_parent_attribute_observation(
                state=state,
                progress=progress,
                parent=parent,
                status="skipped_profile_invalid",
                reason=validation_reason,
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=field_name,
                target_path=target_path,
                evidence_ids=evidence_ids,
                candidate_attribute=instance,
                raw_intent=raw_intent,
            )
            return original
        actual_path = self._actual_requirement_patch_path(document=updated, target_path=target_path)
        self._record_parent_attribute_projection(
            state=state,
            progress=progress,
            parent=parent,
            attribute_kind=attribute_kind,
            intent_index=intent_index,
            actual_path=actual_path,
            instance=instance,
            raw_intent=raw_intent,
            evidence_ids=evidence_ids,
            answer=answer,
        )
        return updated

    @classmethod
    def _record_parent_attribute_projection(
        cls,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        parent: dict[str, Any],
        attribute_kind: str,
        intent_index: int,
        actual_path: str,
        instance: dict[str, Any],
        raw_intent: dict[str, Any],
        evidence_ids: list[str],
        answer: str,
    ) -> None:
        field_name = "has_quantitative_attribute" if attribute_kind == "quantitative" else "has_qualitative_attribute"
        state.field_completion_ledger.append(
            FieldCompletionLedgerRecord(
                json_path=actual_path,
                field_name=field_name,
                generated_value=instance,
                source_evidence=evidence_ids,
                validation_status="valid",
                enrichment_status="grounded",
                issue_categories=[],
                edit_needed_reason=f"parent_attribute:{parent['path']}: {answer}",
            )
        )
        object_identifier = f"parent_attribute:{parent['path']}:{actual_path}"
        state.projection_ledger.append(
            ProjectionLedgerRecord(
                object_identifier=object_identifier,
                object_kind="QuantitativeAttribute" if attribute_kind == "quantitative" else "QualitativeAttribute",
                source_evidence=evidence_ids[0] if evidence_ids else None,
                evidence_note_identifiers=evidence_ids,
                status="projected",
                projected_paths=[actual_path],
                target_path=actual_path,
                target_class=parent["class"],
                planner_status="parent_scoped_attribute",
                planner_reason=f"Parent-scoped attribute question for {parent['class']}.",
                evidence_quality={
                    "parent_path": parent["path"],
                    "attribute_kind": attribute_kind,
                    "answer": answer,
                },
                merge_status="applied",
                reason="Projected schema-valid parent-scoped attribute.",
            )
        )
        cls._record_parent_attribute_observation(
            state=state,
            progress=progress,
            parent=parent,
            status="applied",
            reason="Projected schema-valid parent-scoped attribute.",
            answer=answer,
            attribute_kind=attribute_kind,
            intent_index=intent_index,
            target_field=field_name,
            target_path=f"{parent['path']}/{field_name}/-",
            actual_path=actual_path,
            evidence_ids=evidence_ids,
            candidate_attribute=instance,
            raw_intent=raw_intent,
        )
        progress.field_completion_ledger = state.field_completion_ledger
        progress.projection_ledger = state.projection_ledger

    @classmethod
    def _record_parent_attribute_observation(
        cls,
        *,
        state: ExtractionRunState,
        progress: ExtractionRunProgress,
        parent: dict[str, Any],
        status: str,
        reason: str,
        answer: str = "",
        attribute_kind: str | None = None,
        intent_index: int | None = None,
        target_field: str | None = None,
        target_path: str | None = None,
        actual_path: str | None = None,
        evidence_ids: list[str] | None = None,
        candidate_attribute: dict[str, Any] | None = None,
        raw_intent: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        state.parent_attribute_ledger.append(
            ParentAttributeLedgerRecord(
                parent_path=parent["path"],
                parent_class=parent["class"],
                status=status,
                reason=reason,
                answer=answer,
                attribute_kind=attribute_kind,
                intent_index=intent_index,
                target_field=target_field,
                target_path=target_path,
                actual_path=actual_path,
                evidence_note_identifiers=list(evidence_ids or []),
                candidate_attribute=candidate_attribute,
                raw_intent=raw_intent,
                error=error,
            )
        )
        progress.parent_attribute_ledger = state.parent_attribute_ledger
        progress.field_completion_ledger = state.field_completion_ledger
        progress.projection_ledger = state.projection_ledger

    @staticmethod
    def _quantitative_attribute_instance_from_intent(
        intent: _ParentQuantitativeAttributeIntent,
    ) -> dict[str, Any] | None:
        title = intent.title.strip()
        if not title or intent.value is None:
            return None
        value = float(intent.value)
        description = intent.description.strip() or f"{title}: {value:g}"
        instance: dict[str, Any] = {
            "title": title,
            "description": description,
            "value": value,
            "has_quantity_type": title,
        }
        if intent.unit and intent.unit.strip():
            instance["unit"] = intent.unit.strip()
        return instance

    @staticmethod
    def _qualitative_attribute_instance_from_intent(
        intent: _ParentQualitativeAttributeIntent,
    ) -> dict[str, Any] | None:
        title = intent.title.strip()
        value = intent.value.strip()
        if not title or not value:
            return None
        return {
            "title": title,
            "description": intent.description.strip() or f"{title}: {value}",
            "value": value,
        }

    @classmethod
    def _attribute_parent_targets(cls, document: dict[str, Any]) -> list[dict[str, Any]]:
        parents: list[dict[str, Any]] = []
        for activity_index, activity in enumerate(document.get("was_generated_by") or []):
            if not isinstance(activity, dict):
                continue
            activity_path = f"/was_generated_by/{activity_index}"
            cls._add_attribute_parent(parents, path=activity_path, value=activity, target_class="DataGeneratingActivity")
            for field_name, target_class in (
                ("carried_out_by", "AgenticEntity"),
                ("evaluated_entity", "EvaluatedEntity"),
                ("evaluated_activity", "EvaluatedActivity"),
                ("had_input_entity", "EvaluatedEntity"),
                ("had_input_activity", "EvaluatedActivity"),
                ("had_output_entity", "EvaluatedEntity"),
            ):
                for index, value in enumerate(activity.get(field_name) or []):
                    if isinstance(value, dict):
                        cls._add_attribute_parent(
                            parents,
                            path=f"{activity_path}/{field_name}/{index}",
                            value=value,
                            target_class=target_class,
                        )
        return parents

    @classmethod
    def _add_attribute_parent(
        cls,
        parents: list[dict[str, Any]],
        *,
        path: str,
        value: dict[str, Any],
        target_class: str,
    ) -> None:
        if not (
            cls._semantic_value_present(value.get("title"))
            or cls._semantic_value_present(value.get("description"))
        ):
            return
        parents.append({"path": path, "class": target_class, "value": cls._clone_json_object(value)})

    @classmethod
    def _parent_attribute_role(cls, parent: dict[str, Any]) -> str:
        path = str(parent.get("path") or "")
        value = parent.get("value") if isinstance(parent.get("value"), dict) else {}
        text = cls._normalized_text(
            " ".join(str(value.get(key) or "") for key in ("title", "description", "name", "preferred_label"))
        )
        if path.endswith("/was_generated_by/0") or re.fullmatch(r"/was_generated_by/\d+", path):
            return "activity"
        if "/carried_out_by/" in path:
            if any(term in text for term in ("topspin", "software", "dirdata")):
                return "software"
            if any(term in text for term in ("pulse", "zg30", "program")):
                return "pulse_program"
            if any(term in text for term in ("bruker", "biospin", "manufacturer", "gmbh")):
                return "manufacturer"
            if any(term in text for term in ("spectrometer", "instrument", "avance")):
                return "instrument"
            return "agent"
        if "/evaluated_entity/" in path:
            if any(term in text for term in ("cdcl", "solvent", "chloroform")):
                return "solvent"
            if any(term in text for term in ("nucleus", "proton", "1h", "target")):
                return "target_nucleus"
            return "evaluated_entity"
        if "/had_input_entity/" in path:
            if any(term in text for term in ("pulse", "zg30", "program")):
                return "pulse_program"
            if any(term in text for term in ("parameter", "jcamp", "file", "acqu", "proc")):
                return "parameter_file"
            return "input_entity"
        if "/had_output_entity/" in path:
            if any(term in text for term in ("audit", "trail", "log")):
                return "audit_trail"
            if any(term in text for term in ("spectrum", "spectra", "processed")):
                return "output_spectrum"
            return "output_entity"
        return str(parent.get("class") or "unknown").lower()

    @classmethod
    def _parent_attribute_family(cls, instance: dict[str, Any]) -> str:
        text = cls._normalized_text(
            " ".join(
                str(instance.get(key) or "")
                for key in ("title", "description", "has_quantity_type", "has_attribute_type", "unit", "value")
            )
        )
        if "mhz" in text or "frequency" in text or "spectrometer" in text:
            return "spectrometer_frequency"
        if "software" in text or "topspin version" in text or re.search(r"\bversion\b", text):
            return "software_version"
        if "pulse" in text or "zg30" in text:
            return "pulse_program"
        if "cdcl" in text or "solvent" in text:
            return "solvent"
        if "proton" in text or "nucleus" in text or "activity target" in text:
            return "activity_target"
        if "experiment type" in text:
            return "experiment_type"
        if "format" in text or "proprietary" in text:
            return "file_format"
        if "hash" in text or "md5" in text:
            return "data_hash"
        tokens = set(text.split())
        if any(term in text for term in ("jcamp", "data span", "relaxation", "flip", "mcwrk")) or tokens & {"fov", "sw"}:
            return "acquisition_parameter"
        if any(term in text for term in ("amplifier", "blanking", "current limit", "gab")):
            return "instrument_parameter"
        return "generic"

    @staticmethod
    def _parent_attribute_family_allowed(parent_role: str, family: str) -> bool:
        allowed = {
            "activity": {
                "spectrometer_frequency",
                "software_version",
                "pulse_program",
                "solvent",
                "activity_target",
                "experiment_type",
                "acquisition_parameter",
                "instrument_parameter",
                "generic",
            },
            "instrument": {"spectrometer_frequency", "instrument_parameter", "generic"},
            "manufacturer": {"generic"},
            "software": {"software_version", "acquisition_parameter", "instrument_parameter", "generic"},
            "pulse_program": {"pulse_program", "activity_target", "acquisition_parameter", "generic"},
            "target_nucleus": {"activity_target", "generic"},
            "solvent": {"solvent", "generic"},
            "parameter_file": {"software_version", "experiment_type", "acquisition_parameter", "file_format", "generic"},
            "output_spectrum": {"file_format", "generic"},
            "audit_trail": {"file_format", "generic"},
        }
        return family in allowed.get(parent_role, {"generic"})

    @classmethod
    def _parent_attribute_significant_tokens(cls, text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "data", "file",
            "entity", "activity", "used", "using", "parameter", "processed",
        }
        return {token for token in cls._normalized_text(text).split() if len(token) >= 3 and token not in stop}

    @classmethod
    def _evidence_item_from_candidate(cls, candidate: EvidenceCandidate) -> RequirementEvidenceItem:
        return RequirementEvidenceItem(
            evidence_id=stable_evidence_id(candidate),
            candidate_id=candidate.candidate_id,
            category=str(candidate.category),
            role=str(candidate.role),
            claim=candidate.claim,
            evidence_text=candidate.evidence_text,
            source_context=candidate.source_context,
            file_path=candidate.file_path,
            start_idx=candidate.start_idx,
            end_idx=candidate.end_idx,
            evidence_match_score=float(getattr(candidate, "evidence_match_score", 0.0) or 0.0),
        )

    @classmethod
    def _parent_scoped_evidence_score(cls, *, parent: dict[str, Any], candidate: EvidenceCandidate) -> int:
        role = cls._parent_attribute_role(parent)
        value = parent.get("value") if isinstance(parent.get("value"), dict) else {}
        parent_title = cls._normalized_text(value.get("title") or value.get("name") or value.get("preferred_label"))
        parent_text = cls._normalized_text(
            " ".join(str(value.get(key) or "") for key in ("title", "description", "name", "preferred_label"))
        )
        candidate_text = cls._normalized_text(
            " ".join(
                str(part or "")
                for part in (
                    candidate.category,
                    candidate.role,
                    candidate.claim,
                    candidate.evidence_text,
                    candidate.source_context,
                    candidate.file_path,
                )
            )
        )
        title_tokens = cls._parent_attribute_significant_tokens(parent_title)
        parent_tokens = cls._parent_attribute_significant_tokens(parent_text)
        title_hits = len(title_tokens & set(candidate_text.split()))
        parent_hits = len(parent_tokens & set(candidate_text.split()))
        category = str(candidate.category)
        evidence_role = str(candidate.role)
        score = title_hits * 4 + min(parent_hits, 3)

        if role == "manufacturer":
            if not title_hits:
                return 0
            if any(term in candidate_text for term in ("frequency", "mhz", "version", "topspin", "pulse", "solvent")):
                return 0
            return score + 2
        if role == "software":
            if category == "software_signal" or "software" in candidate_text or "topspin" in candidate_text:
                score += 5
            if any(term in candidate_text for term in ("version", "parameter", "config", "topspin")):
                score += 2
        elif role == "pulse_program":
            if category == "method_signal" or any(term in candidate_text for term in ("pulse", "zg30", "program")):
                score += 5
            if any(term in candidate_text for term in ("software version", "solvent", "frequency", "mhz")):
                score -= 4
        elif role == "target_nucleus":
            if any(term in candidate_text for term in ("nucleus", "proton", "1h", "target")):
                score += 5
            if any(term in candidate_text for term in ("solvent", "cdcl", "pulse", "zg30", "frequency", "mhz")):
                score -= 4
        elif role == "solvent":
            if any(term in candidate_text for term in ("solvent", "cdcl", "chloroform")):
                score += 6
            if any(term in candidate_text for term in ("pulse", "frequency", "mhz")):
                score -= 3
        elif role == "parameter_file":
            if evidence_role == "parameter" or category in {"software_signal", "instrument_signal", "measurement_condition", "resource_signal"}:
                score += 4
            if any(term in candidate_text for term in ("jcamp", "parameter", "acqu", "proc", "sw", "fov", "experiment type")):
                score += 4
            if any(term in candidate_text for term in ("frequency", "mhz", "bf1", "bf2", "bf3")):
                score -= 8
            if any(term in candidate_text for term in ("solvent", "pulse program")):
                score -= 3
        elif role == "output_spectrum":
            if category == "resource_signal" and any(term in candidate_text for term in ("spectrum", "spectra", "output", "format")):
                score += 5
            if any(term in candidate_text for term in ("frequency", "mhz", "solvent", "pulse", "zg30")):
                score -= 5
        elif role == "audit_trail":
            if category in {"resource_signal", "surrounding_signal"} and any(term in candidate_text for term in ("audit", "trail", "log")):
                score += 5
            if any(term in candidate_text for term in ("frequency", "mhz", "solvent", "pulse", "zg30")):
                score -= 5
        elif role == "activity":
            if category in {"activity_signal", "method_signal", "instrument_signal", "software_signal", "measurement_condition"}:
                score += 4
            if any(term in candidate_text for term in ("acquisition", "processing", "experiment")):
                score += 2

        return score if score >= 5 else 0

    @classmethod
    def _parent_attribute_semantic_placement_reason(
        cls,
        *,
        parent: dict[str, Any],
        instance: dict[str, Any],
        selected_evidence: list[RequirementEvidenceItem],
    ) -> str:
        if not selected_evidence:
            return ""
        parent_role = cls._parent_attribute_role(parent)
        family = cls._parent_attribute_family(instance)
        quantitative = "has_quantity_type" in instance or cls._first_number(instance.get("value")) is not None
        if quantitative and parent_role in {"manufacturer", "output_spectrum", "audit_trail"} and family == "generic":
            return f"Generic numeric attributes are not semantically local to parent role '{parent_role}'."
        if not cls._parent_attribute_family_allowed(parent_role, family):
            return f"Attribute family '{family}' is not semantically local to parent role '{parent_role}'."
        evidence_text = cls._normalized_text(
            " ".join(
                " ".join(str(getattr(item, key, "") or "") for key in ("claim", "evidence_text"))
                for item in selected_evidence
            )
        )
        label_tokens = cls._parent_attribute_significant_tokens(
            " ".join(str(instance.get(key) or "") for key in ("title", "has_quantity_type", "has_attribute_type"))
        )
        value_text = cls._normalized_text(instance.get("value"))
        has_label_support = bool(label_tokens and label_tokens & set(evidence_text.split()))
        has_value_support = bool(value_text and value_text in evidence_text)
        if not (has_label_support or has_value_support):
            return "Attribute label/value is not supported by the selected parent-local evidence."
        return ""

    @staticmethod
    def _parent_attribute_prompt(
        *,
        parent: dict[str, Any],
        selected_evidence: list[RequirementEvidenceItem],
        context_window: list[RequirementEvidenceItem],
        core_draft: dict[str, Any],
    ) -> str:
        payload = {
            "question": (
                "Which profile-level quantitative or qualitative attributes, if any, should be added "
                f"to the focused {parent['class']} object at the supplied target path? "
                "Return empty arrays for low-level/internal parameters "
                "or facts better represented by title, description, method, agent, input, output, or evaluated target relations. "
                "Use the core draft to understand the focused object in its profile context. "
                "Use selected_evidence only as attribute support. "
                "Apply DCAT-AP+ generic attribute semantics: an attribute must be a recorded characterization "
                "of the focused object itself, not of the source record that mentioned it."
            ),
            "task_instructions": [
                "Focus only on the object at target_path with target_class.",
                "Use core_draft for structural context, not as evidence support.",
                "Use selected_evidence as the only support for emitted attributes.",
                "Name the focused object being characterized.",
                "Identify the selected evidence phrase that directly states the characterization.",
                "Decide whether the fact describes the focused object itself rather than source-record metadata or provenance context.",
                "Reject creator, owner, origin, vendor, manufacturer, file path, process path, audit/hash, or bookkeeping facts unless they directly characterize the focused object.",
                "Reject facts better represented by title, description, method, agent, input, output, evaluated target, creator, publisher, provenance, or source trace.",
                "Reject attributes with unclear intent, placeholder labels, or generic values such as measured/unknown/present without a parent-specific characterization.",
                "Do not include this checklist in the response; return only quantitative_attributes and qualitative_attributes.",
            ],
            "target_path": parent["path"],
            "target_class": parent["class"],
            "core_draft": core_draft,
            "selected_evidence": compact_requirement_evidence(
                selected_evidence,
                include_source_context=False,
            ),
            "context_window": compact_requirement_evidence(
                context_window,
                include_source_context=False,
            ),
            "output_schema": {
                "answer": "string",
                "quantitative_attributes": [{"title": "string", "value": "number", "unit": "string|null", "description": "string"}],
                "qualitative_attributes": [{"title": "string", "value": "string", "description": "string"}],
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _parent_attribute_evidence_packet(
        self,
        *,
        parent: dict[str, Any],
        evidence_context: RoutedEvidenceContext,
    ) -> tuple[list[RequirementEvidenceItem], list[RequirementEvidenceItem]]:
        candidates = list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence)
        scored = [
            (self._parent_scoped_evidence_score(parent=parent, candidate=candidate), candidate)
            for candidate in candidates
        ]
        selected = [
            candidate
            for score, candidate in sorted(
                scored,
                key=lambda item: (-item[0], item[1].file_path, item[1].start_idx, item[1].candidate_id),
            )
            if score > 0
        ][:4]
        return ([self._evidence_item_from_candidate(candidate) for candidate in selected], [])

    @staticmethod
    def _parent_attribute_allowed_categories(target_class: str) -> list[str]:
        if target_class == "AgenticEntity":
            return ["instrument_signal", "software_signal", "surrounding_signal"]
        if target_class == "DataGeneratingActivity":
            return ["activity_signal", "method_signal", "instrument_signal", "software_signal", "measurement_condition"]
        return ["measurement_signal", "measurement_condition", "resource_signal", "activity_signal", "method_signal"]

    @staticmethod
    def _evidence_origins(evidence_items: list[Any]) -> list[str]:
        return ["source_file"]

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
        allowed_target_paths = item.target_paths or requirement.target_paths
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
        output_schema = SchemaConstrainedPatchRoute.output_schema(
            validation_schema=validation_schema,
            allowed_target_paths=allowed_target_paths,
        )
        try:
            result = await generate_structured(
                self.ollama_client,
                model=self.ollama_client.chat_model,
                system=REQUIREMENT_PATCH_SYSTEM_PROMPT,
                prompt=prompt,
                output_type=output_schema,
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
            patch = parse_schema_constrained_patch_result(result.output)
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
        if not patch.writes and (
            target_class == "QuantitativeAttribute"
            or any("/has_quantitative_attribute/" in path for path in (item.target_paths or requirement.target_paths))
        ):
            deterministic_instance = self._quantitative_attribute_from_evidence(item)
            if deterministic_instance:
                fallback_target_path = allowed_target_paths[0]
                fallback_schema_path = (
                    fallback_target_path[:-2]
                    if fallback_target_path.endswith("/-")
                    else fallback_target_path
                )
                fallback_is_array = self._schema_is_array(
                    schema_for_json_pointer(validation_schema, fallback_schema_path),
                    validation_schema,
                )
                fallback_write = (
                    SchemaConstrainedWrite(
                        target_path=fallback_target_path,
                        mode="append",
                        items=[deterministic_instance],
                        reason="Deterministic quantitative constructor used selected measurement evidence.",
                    )
                    if fallback_is_array
                    else SchemaConstrainedWrite(
                        target_path=fallback_target_path,
                        mode="replace",
                        value=deterministic_instance,
                        reason="Deterministic quantitative constructor used selected measurement evidence.",
                    )
                )
                patch = SchemaConstrainedPatchResult(
                    writes=[fallback_write],
                    reason=(
                        "Deterministic quantitative constructor used selected measurement evidence "
                        "after patch model declined to build an instance."
                    ),
                )
        if not patch.writes:
            return (
                document,
                RequirementPatchAttempt(
                    attempted=True,
                    status="failed",
                    target_path=target_path,
                    target_class=target_class,
                    reason=patch.reason or "Patch model found insufficient evidence.",
                ),
            )
        for write in patch.writes:
            duplicate_target = write.target_path
            if write.mode == "append" and not duplicate_target.endswith("/-"):
                duplicate_target = f"{duplicate_target}/-"
            for instance in write.items if write.mode == "append" else [write.value]:
                if not isinstance(instance, dict):
                    continue
                duplicate_reason = self._duplicate_requirement_patch_reason(
                    document=document,
                    target_path=duplicate_target,
                    instance=instance,
                )
                if duplicate_reason:
                    return (
                        document,
                        RequirementPatchAttempt(
                            attempted=True,
                            status="failed",
                            target_path=write.target_path,
                            target_class=target_class,
                            reason=duplicate_reason,
                        ),
                    )
        original = self._clone_json_object(document)
        try:
            updated, changed_paths = apply_schema_constrained_writes(
                document=document,
                writes=patch.writes,
                data_package_id=data_package_id,
                validation_schema=validation_schema,
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
                    target_path=changed_paths[0] if changed_paths else target_path,
                    target_class=target_class,
                    reason=patch.reason or "Requirement patch applied and validated.",
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
            if isinstance(existing, dict) and (
                cls._instances_semantically_equal(existing, instance)
                or cls._attributes_semantically_compatible(
                    parent_path=target_path,
                    left=existing,
                    right=instance,
                )
            ):
                return f"duplicate_semantic_slot: requirement patch duplicates existing object at {target_path}."
            return None
        parent_path = target_path[:-2]
        existing_items = cls._value_at_json_pointer(document, parent_path)
        if not isinstance(existing_items, list):
            return None
        for existing in existing_items:
            if isinstance(existing, dict) and (
                cls._instances_semantically_equal(existing, instance)
                or cls._attributes_semantically_compatible(
                    parent_path=parent_path,
                    left=existing,
                    right=instance,
                )
            ):
                return f"duplicate_semantic_slot: requirement patch duplicates existing object at {parent_path}."
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

    @classmethod
    def _sanitize_schema_constrained_writes(
        cls,
        *,
        writes: list[SchemaConstrainedWrite],
        requirement_id: str,
    ) -> list[SchemaConstrainedWrite]:
        sanitized: list[SchemaConstrainedWrite] = []
        for write in writes:
            path_parts = [part for part in write.target_path.strip("/").split("/") if part]
            is_aboutness_object_path = (
                bool(path_parts)
                and path_parts[0] in {"is_about_entity", "is_about_activity"}
                and len(path_parts) <= 2
            )
            evaluation_relation_index = next(
                (
                    index
                    for index, part in enumerate(path_parts)
                    if part in {"evaluated_entity", "evaluated_activity"}
                ),
                None,
            )
            is_evaluation_target_path = (
                evaluation_relation_index is not None
                and len(path_parts) <= evaluation_relation_index + 2
            )
            if is_aboutness_object_path or is_evaluation_target_path:
                if write.mode == "append":
                    sanitized.append(
                        write.model_copy(
                            update={
                                "items": [
                                    cls._sanitize_relation_item(item)
                                    for item in write.items
                                ]
                            }
                        )
                    )
                    continue
                if isinstance(write.value, dict):
                    sanitized.append(write.model_copy(update={"value": cls._sanitize_relation_item(write.value)}))
                    continue
            sanitized.append(write)
        return sanitized

    @staticmethod
    def _sanitize_relation_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        return {
            key: item[key]
            for key in ("id", "title", "description", "type", "rdf_type")
            if key in item and item[key] not in (None, "", [])
        }

    @classmethod
    def _filter_duplicate_schema_writes(
        cls,
        *,
        document: dict[str, Any],
        writes: list[SchemaConstrainedWrite],
    ) -> tuple[list[SchemaConstrainedWrite], list[str]]:
        filtered: list[SchemaConstrainedWrite] = []
        reasons: list[str] = []
        for write in writes:
            if write.mode != "append" or not any(
                attribute_path in write.target_path
                for attribute_path in ("has_quantitative_attribute", "has_qualitative_attribute")
            ):
                filtered.append(write)
                continue
            parent_path = write.target_path[:-2] if write.target_path.endswith("/-") else write.target_path
            existing_items = cls._value_at_json_pointer(document, parent_path)
            if not isinstance(existing_items, list):
                filtered.append(write)
                continue
            kept_items: list[Any] = []
            comparison_items = [item for item in existing_items if isinstance(item, dict)]
            for item in write.items:
                if not isinstance(item, dict):
                    kept_items.append(item)
                    continue
                duplicate = any(
                    cls._attributes_semantically_compatible(
                        parent_path=parent_path,
                        left=existing,
                        right=item,
                    )
                    for existing in comparison_items
                )
                if duplicate:
                    reasons.append(f"duplicate_semantic_slot: duplicate attribute skipped at {parent_path}.")
                    continue
                kept_items.append(item)
                comparison_items.append(item)
            if kept_items:
                filtered.append(write.model_copy(update={"items": kept_items}))
        return filtered, list(dict.fromkeys(reasons))

    @classmethod
    def _quantitative_attributes_semantically_equal(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        return cls._attributes_semantically_compatible(
            parent_path="",
            left=left,
            right=right,
        )

    @staticmethod
    def _quantitative_attribute_signature(instance: dict[str, Any]) -> tuple[str, str, str] | None:
        quantity = instance.get("has_quantity_type")
        if isinstance(quantity, dict):
            quantity = quantity.get("id") or quantity.get("title")
        quantity_text = re.sub(r"\s+", " ", str(quantity or "")).strip().lower()
        value = ProjectionService._first_number(instance.get("value"))
        if not quantity_text or value is None:
            return None
        unit = instance.get("unit")
        if isinstance(unit, dict):
            unit = unit.get("id") or unit.get("title")
        unit_text = re.sub(r"\s+", " ", str(unit or "")).strip().lower()
        value_text = ("%0.12g" % value).rstrip("0").rstrip(".")
        return quantity_text, value_text, unit_text

    @classmethod
    def _attribute_semantic_slot(
        cls,
        *,
        parent_path: str,
        instance: dict[str, Any],
    ) -> tuple[str, str, str, str, str] | None:
        if not isinstance(instance, dict):
            return None
        kind = (
            "quantitative"
            if "value" in instance and ("has_quantity_type" in instance or cls._first_number(instance.get("value")) is not None)
            else "qualitative"
        )
        label = instance.get("has_quantity_type") if kind == "quantitative" else instance.get("has_attribute_type")
        if isinstance(label, dict):
            label = label.get("id") or label.get("title")
        label_text = " ".join(
            str(part or "")
            for part in (
                label,
                instance.get("title"),
                instance.get("description"),
            )
        )
        normalized_label = cls._normalized_attribute_label(label_text)
        if not normalized_label:
            return None
        value = cls._first_number(instance.get("value"))
        if value is None:
            value_text = cls._normalized_text(instance.get("value"))
            if not value_text:
                return None
        else:
            value_text = ("%0.12g" % value).rstrip("0").rstrip(".")
        unit = instance.get("unit")
        if isinstance(unit, dict):
            unit = unit.get("id") or unit.get("title")
        unit_text = cls._normalized_unit(unit)
        return parent_path.rstrip("/"), kind, normalized_label, value_text, unit_text

    @classmethod
    def _attributes_semantically_compatible(
        cls,
        *,
        parent_path: str,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_slot = cls._attribute_semantic_slot(parent_path=parent_path, instance=left)
        right_slot = cls._attribute_semantic_slot(parent_path=parent_path, instance=right)
        if not left_slot or not right_slot:
            return False
        left_identity = left_slot[:3] + left_slot[4:]
        right_identity = right_slot[:3] + right_slot[4:]
        if left_identity != right_identity:
            return False
        left_number = cls._first_number(left.get("value"))
        right_number = cls._first_number(right.get("value"))
        if left_number is not None or right_number is not None:
            if left_number is None or right_number is None:
                return False
            if left_number == right_number:
                return True
            if left_number == 0 or right_number == 0:
                return False
            return math.isclose(left_number, right_number, rel_tol=1e-6, abs_tol=1e-9)
        return left_slot[3] == right_slot[3]

    @classmethod
    def _attributes_same_semantic_identity(
        cls,
        *,
        parent_path: str,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        """Check if two attributes share the same semantic identity (kind, label, unit)
        regardless of their values.  Used to detect same-label different-value duplicates.

        Only returns True when the *original* (pre-normalization) labels also match or
        share significant words, to avoid merging different quantities that happen to
        normalize to the same semantic label.
        """
        left_slot = cls._attribute_semantic_slot(parent_path=parent_path, instance=left)
        right_slot = cls._attribute_semantic_slot(parent_path=parent_path, instance=right)
        if not left_slot or not right_slot:
            return False
        left_identity = left_slot[:3] + left_slot[4:]
        right_identity = right_slot[:3] + right_slot[4:]
        if left_identity != right_identity:
            return False
        # Require original labels to match or share significant words, not just
        # the normalized label (which may collapse different quantities).
        left_raw = cls._normalized_text(
            " ".join(str(left.get(k) or "") for k in ("has_quantity_type", "has_attribute_type", "title"))
        )
        right_raw = cls._normalized_text(
            " ".join(str(right.get(k) or "") for k in ("has_quantity_type", "has_attribute_type", "title"))
        )
        if left_raw == right_raw:
            return True
        _stop = {"the", "of", "for", "in", "and", "a", "an", "is", "are", "was", "were",
                 "be", "been", "being", "have", "has", "had", "do", "does", "did"}
        left_words = {w for w in left_raw.split() if len(w) > 1 and w not in _stop}
        right_words = {w for w in right_raw.split() if len(w) > 1 and w not in _stop}
        shared = left_words & right_words
        # Require ALL content words to match (not just some), to avoid merging
        # "maximum Y" with "maximum transmittance" just because they share "maximum".
        return bool(left_words and right_words and left_words == right_words)

    @classmethod
    def _attributes_value_equivalent(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        """Check if two quantitative attributes have the same numeric value and compatible units, regardless of label.

        Only returns True when labels share at least one significant word, to avoid merging
        different quantities that happen to share a numeric value.
        """
        left_slot = cls._attribute_semantic_slot(parent_path="", instance=left)
        right_slot = cls._attribute_semantic_slot(parent_path="", instance=right)
        if not left_slot or not right_slot:
            return False
        if left_slot[1] != right_slot[1]:
            return False
        left_number = cls._first_number(left.get("value"))
        right_number = cls._first_number(right.get("value"))
        if left_number is None or right_number is None:
            return False
        if left_number == 0 and right_number == 0:
            return True
        if left_number == 0 or right_number == 0:
            return False
        if not math.isclose(left_number, right_number, rel_tol=1e-6, abs_tol=1e-9):
            return False
        left_unit = left_slot[4]
        right_unit = right_slot[4]
        if left_unit and right_unit and left_unit != right_unit:
            return False
        # Require at least one shared significant word between labels to avoid merging
        # different quantities that happen to share a numeric value.
        left_label = left_slot[2]
        right_label = right_slot[2]
        if left_label == right_label:
            return True
        left_words = {w for w in left_label.split() if len(w) > 1 and w not in {"the", "of", "for", "in", "and"}}
        right_words = {w for w in right_label.split() if len(w) > 1 and w not in {"the", "of", "for", "in", "and"}}
        return bool(left_words and right_words and (left_words & right_words))

    @classmethod
    def _attribute_survivor_score(cls, instance: Any) -> tuple[int, int, int]:
        if not isinstance(instance, dict):
            return 0, 0, 0
        value_text = str(instance.get("value", "")).lower().split("e", maxsplit=1)[0]
        significant_digits = len(re.sub(r"[^0-9]", "", value_text).lstrip("0").rstrip("0"))
        populated_fields = sum(value not in (None, "", [], {}) for value in instance.values())
        semantic_label = cls._normalized_attribute_label(
            " ".join(
                str(instance.get(key) or "")
                for key in ("has_quantity_type", "has_attribute_type", "title")
            )
        )
        return significant_digits, populated_fields, len(semantic_label)

    @classmethod
    def _normalized_attribute_label(cls, value: Any) -> str:
        text = cls._normalized_text(value)
        if not text:
            return ""
        replacements = {
            "npoints": "point count",
            "data points": "point count",
            "number data points": "point count",
            "number of data points": "point count",
            "point count": "point count",
            "points": "point count",
            "maximum": "max",
            "minimum": "min",
            "transmittance value": "transmittance",
            "transmittance spectrum": "transmittance",
            "wavelength value": "wavelength",
            "wavenumber value": "wavenumber",
            "x axis": "x",
            "y axis": "y",
        }
        for source, target in replacements.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        text = re.sub(
            r"\b(the|a|an|value|values|recorded|from|in|of|spectrum|spectra|measurement|measured|data)\b",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        if "point" in tokens and "count" in tokens:
            return "point count"
        token_set = set(tokens)
        has_max = "max" in token_set or "first" in token_set or "upper" in token_set
        has_min = "min" in token_set or "last" in token_set or "lower" in token_set
        has_x_axis = "x" in token_set or "wavenumber" in token_set or "wavelength" in token_set
        has_y_axis = "y" in token_set or "transmittance" in token_set or "intensity" in token_set
        if has_y_axis and has_max:
            return "max transmittance"
        if has_y_axis and has_min:
            return "min transmittance"
        if has_x_axis and has_max:
            return "max wavenumber"
        if has_x_axis and has_min:
            return "min wavenumber"
        if "threshold" in token_set:
            return "threshold"
        if "resolution" in token_set:
            return "resolution"
        return text

    @staticmethod
    def _normalized_unit(value: Any) -> str:
        text = ProjectionService._normalized_text(value)
        aliases = {
            "1 cm": "1/cm",
            "1/cm": "1/cm",
            "cm-1": "1/cm",
            "%": "%",
            "percent": "%",
        }
        if text.endswith("/percent") or "vocab/unit/percent" in text:
            return "%"
        return aliases.get(text, text)

    @staticmethod
    def _normalized_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        text = str(value).lower()
        text = re.sub(r"[^a-z0-9%/+-]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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
        validation_schema: dict[str, Any],
        target_path: str,
        target_class: str,
    ) -> dict[str, Any] | None:
        assert self.ollama_client is not None
        output_schema = SchemaConstrainedPatchRoute.output_schema(
            validation_schema=validation_schema,
            allowed_target_paths=[target_path],
        )
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
                prompt=(
                    prompt
                    + "\nReturn a schema-constrained write envelope. "
                    "Use mode=append for array targets and mode=replace for scalar/object targets."
                ),
                output_type=output_schema,
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
            patch = parse_schema_constrained_patch_result(result.output)
            if not patch.writes:
                return None
            first = patch.writes[0]
            if first.mode == "append":
                return dict(first.items[0]) if isinstance(first.items[0], dict) else None
            return dict(first.value) if isinstance(first.value, dict) else None
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
        validation_schema: dict[str, Any],
        target_path: str,
        target_class: str,
    ) -> dict[str, Any] | None:
        assert self.ollama_client is not None
        output_schema = SchemaConstrainedPatchRoute.output_schema(
            validation_schema=validation_schema,
            allowed_target_paths=[target_path],
        )
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
                prompt=(
                    prompt
                    + "\nReturn a schema-constrained write envelope. "
                    "Use mode=append for array targets and mode=replace for scalar/object targets."
                ),
                output_type=output_schema,
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
            patch = parse_schema_constrained_patch_result(result.output)
            if not patch.writes:
                return None
            first = patch.writes[0]
            if first.mode == "append":
                return dict(first.items[0]) if isinstance(first.items[0], dict) else None
            return dict(first.value) if isinstance(first.value, dict) else None
        except (CompletionError, ValidationError) as exc:
            self._record_llm_call_exception(
                data_package_id=data_package_id,
                exc=exc,
                agent_name="evidence_instance_repair",
            )
            return None

    @classmethod
    def _schema_write_from_instance(
        cls,
        *,
        validation_schema: dict[str, Any],
        target_path: str,
        instance: dict[str, Any],
    ) -> SchemaConstrainedWrite:
        schema_path = target_path[:-2] if target_path.endswith("/-") else target_path
        is_array = target_path.endswith("/-") or cls._schema_is_array(
            schema_for_json_pointer(validation_schema, schema_path),
            validation_schema,
        )
        if is_array:
            return SchemaConstrainedWrite(
                target_path=target_path,
                mode="append",
                items=[instance],
                reason="Schema-constrained evidence instance.",
            )
        return SchemaConstrainedWrite(
            target_path=target_path,
            mode="replace",
            value=instance,
            reason="Schema-constrained evidence instance.",
        )

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
        validation_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], ProjectionLedgerRecord]:
        original = self._clone_json_object(document)
        write = self._schema_write_from_instance(
            validation_schema=validation_schema,
            target_path=target_path,
            instance=instance,
        )
        try:
            updated, changed_paths = apply_schema_constrained_writes(
                document=document,
                writes=[write],
                data_package_id=data_package_id,
                validation_schema=validation_schema,
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
                    projected_paths=changed_paths or [target_path],
                ),
            )
        repaired = await self._repair_evidence_instance(
            data_package_id=data_package_id,
            instance=instance,
            validation_errors=validation.errors,
            schema_branch=schema_branch,
            validation_schema=validation_schema,
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
        repaired_write = self._schema_write_from_instance(
            validation_schema=validation_schema,
            target_path=target_path,
            instance=repaired,
        )
        try:
            updated, changed_paths = apply_schema_constrained_writes(
                document=original,
                writes=[repaired_write],
                data_package_id=data_package_id,
                validation_schema=validation_schema,
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
                    projected_paths=changed_paths or [target_path],
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
        if note.category in {"software_signal", "instrument_signal"} or cls._note_has_device_signal(note):
            return "/was_generated_by/0/carried_out_by/-", "AgenticEntity"
        if note.category == "surrounding_signal" and any(term in text for term in ("origin", "owner", "creator", "author", "team", "laboratory")):
            return "/creator/0", "Agent"
        if note.category == "activity_signal" or any(term in text for term in ("experiment", "acquisition", "generation", "workflow")):
            return "/was_generated_by/0", "DataGeneratingActivity"
        if note.category == "method_signal":
            return "/was_generated_by/0/realized_plan", "Plan"
        if any(term in text for term in ("dataset name", "title", "name")):
            return "/title", None
        if any(term in text for term in ("date", "timestamp", "modified", "modification")):
            return "/modification_date", None
        if any(term in text for term in ("type", "category", "class")):
            return "/type/0", "Concept"
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
                note.category in {"method_signal", "measurement_signal", "instrument_signal"}
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
        for note in notes:
            if note.role != "qualitative_attribute":
                continue
            # Try key=value extraction first
            key, value = cls._assignment_from_note(note)
            if key and value:
                title = key.lower().strip("$")
            else:
                # No key=value pattern — derive title from evidence text
                ev = (note.evidence_text or "").strip()
                claim = (note.claim or "").strip()
                value = claim or ev
                if not value:
                    continue
                # Extract first significant word as title
                title = cls._derive_qualitative_title(ev)
                if not title:
                    continue
            attributes.append({"title": title, "value": value})
        return cls._merge_unique_dicts([], attributes)

    @staticmethod
    def _derive_qualitative_title(evidence_text: str) -> str:
        """Extract a short title from evidence text for qualitative attributes."""
        text = evidence_text.strip()
        # Strip leading ## or ##$
        text = re.sub(r"^#{1,2}\$?", "", text)
        # Try key=value or key: value patterns
        match = re.match(r"([A-Za-z][A-Za-z0-9_]{1,32})\s*[=:]]", text)
        if match:
            return match.group(1).lower().strip("$")
        # Try first significant word
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]{2,32})", text)
        if match:
            return match.group(1).lower()
        return ""

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
        if note.category == "resource_signal":
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
        return note.category in {"software_signal", "activity_signal", "instrument_signal", "surrounding_signal"} and bool(note.claim.strip())

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
            ProjectionService._patch_operation_dict(operation)
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
        source_document = remove_null_values(self._clone_json_object(document))
        grounded_document = self._grounded_profile_document(
            document=source_document,
            normalization=normalization,
            validation_schema=validation_schema,
        )
        pruned_document = self._prune_initial_draft_scaffold(
            grounded_document,
            state.initial_draft_scaffold,
        )
        clean_document = remove_null_values(pruned_document)
        validation = self._validate_profile_document(
            profile_identifier=profile_identifier,
            document=clean_document,
        )
        self._revalidate_requirement_report_against_delivered_document(
            state=state,
            document=clean_document,
            validation_schema=validation_schema,
            schema_valid=validation.status == "valid",
        )
        state.generated_final_draft = clean_document
        if state.generated_reconstructed_draft is None:
            state.generated_reconstructed_draft = self._clone_json_object(source_document)
        state.validation = validation
        if state.curated_document is not None and state.curated_validation is None:
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
        state.curation_ledger = (
            self._build_curation_ledger(
                generated_document=clean_document,
                curated_document=state.curated_document,
                existing_field_ledger=state.field_completion_ledger,
            )
            if state.curated_document is not None
            else []
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
            grounding_policy=state.vocab_query_config,
            grounded_validation=validation,
            grounded_document=clean_document,
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
            generated_core_draft=state.generated_core_draft,
            generated_attribute_draft=state.generated_attribute_draft,
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
            parent_attribute_ledger=state.parent_attribute_ledger,
            field_completion_ledger=state.field_completion_ledger,
            evidence_query_ledger=state.evidence_query_ledger,
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

    @classmethod
    def _revalidate_requirement_report_against_delivered_document(
        cls,
        *,
        state: ExtractionRunState,
        document: dict[str, Any],
        validation_schema: dict[str, Any],
        schema_valid: bool,
    ) -> None:
        report = state.requirement_report
        if report is None:
            return
        configured = {
            requirement.requirement_id: requirement
            for requirement in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS
        }
        semantic_items = [item.model_copy(deep=True) for item in report.semantic_requirements]
        for item in semantic_items:
            requirement = configured.get(item.requirement_id)
            if requirement is None:
                continue
            runtime_requirement = cls._runtime_semantic_requirement(
                requirement=requirement,
                document=document,
            )
            item.target_paths = list(runtime_requirement.target_paths)
            cls._guard_semantic_requirement_assessment(
                requirement=runtime_requirement,
                document=document,
                item=item,
            )
        score_requirement_items(semantic_items)
        state.requirement_report = build_requirement_report(
            schema_valid=schema_valid,
            coverage=compute_coverage_report(document, validation_schema),
            semantic_requirements=semantic_items,
            source_trace=report.source_trace,
            coverage_patches=report.coverage_patches,
            semantic_reconstructions=report.semantic_reconstructions,
        )

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
    ) -> DraftQualityState:
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

    @classmethod
    def _grounded_profile_document(
        cls,
        *,
        document: dict[str, Any],
        normalization: ExtractionNormalization,
        validation_schema: dict[str, Any],
    ) -> dict[str, Any]:
        grounded = cls._apply_deterministic_quantitative_terms(
            cls._clone_json_object(document)
        )
        for item in normalization.profile_fields:
            if item.term is None or not item.term.selected_uri:
                continue
            exists, existing_value = cls._json_pointer_value(grounded, item.json_path)
            field_schema = cls._schema_for_json_pointer(validation_schema, item.json_path)
            selected_value = cls._selected_vocab_value_for_schema(
                field_schema=field_schema,
                root_schema=validation_schema,
                selected_uri=item.term.selected_uri,
                selected_title=item.term.selected_title,
                vocabulary_identifier=item.term.vocabulary_identifier,
                existing_value=existing_value if exists else None,
                rdf_type_term=cls._rdf_type_term_for_grounding_role(item.field_name),
            )
            grounded = cls._set_json_pointer_value(
                grounded,
                item.json_path,
                selected_value,
            )
        return grounded

    @classmethod
    def _apply_deterministic_quantitative_terms(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._apply_deterministic_quantitative_terms(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {
            key: cls._apply_deterministic_quantitative_terms(item)
            for key, item in value.items()
        }
        if "value" in result and "has_quantity_type" in result:
            result.setdefault(
                "rdf_type",
                cls._defined_term(
                    QUDT_QUANTITY_URI,
                    "Quantity",
                    QUDT_SCHEMA_VOCAB,
                ),
            )
        if isinstance(result.get("has_quantity_type"), dict):
            result["has_quantity_type"].setdefault(
                "rdf_type",
                cls._rdf_type_term_for_grounding_role("has_quantity_type"),
            )
        if isinstance(result.get("unit"), dict):
            result["unit"].setdefault(
                "rdf_type",
                cls._rdf_type_term_for_grounding_role("unit"),
            )
        return result

    @staticmethod
    def _rdf_type_term_for_grounding_role(role: str) -> dict[str, Any] | None:
        if role == "has_quantity_type":
            return ProjectionService._defined_term(
                QUDT_QUANTITY_KIND_URI,
                "QuantityKind",
                QUDT_SCHEMA_VOCAB,
            )
        if role == "unit":
            return ProjectionService._defined_term(
                QUDT_UNIT_URI,
                "Unit",
                QUDT_SCHEMA_VOCAB,
            )
        return None

    @staticmethod
    def _defined_term(
        term_id: str,
        title: str | None = None,
        from_cv: str | None = None,
        rdf_type: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {"id": term_id}
        if title:
            value["title"] = title
        if from_cv:
            value["from_CV"] = from_cv
        if rdf_type is not None:
            value["rdf_type"] = rdf_type
        return value

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
            validation_schema=validation_schema,
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

            # Determine selection_kind: vocab_selection, evidence_projection, both, none
            has_vocab = enrichment_status == "grounded"
            has_evidence = bool(source_evidence_by_path.get(path, []))
            if has_vocab and has_evidence:
                selection_kind = "both"
            elif has_vocab:
                selection_kind = "vocab_selection"
            elif has_evidence:
                selection_kind = "evidence_projection"
            else:
                selection_kind = "none"
            ledgers.append(
                FieldCompletionLedgerRecord(
                    json_path=path,
                    field_name=self._field_name_from_pointer(path),
                    generated_value=generated_value if generated_exists else None,
                    curated_value=curated_value if curated_exists else None,
                    source_evidence=source_evidence_by_path.get(path, []),
                    validation_status=validation_status,
                    enrichment_status=enrichment_status,
                    selection_kind=selection_kind,
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
                status = "auto_removed"
            elif not generated_exists or generated_value != curated_value:
                status = "auto_modified"
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
        if state.generated_core_draft is not None:
            self.output_repository.save_generated_core_draft(
                workflow_id=data_package_id,
                document=state.generated_core_draft,
                chat_model=chat_model,
                chunking_strategy=chunking_strategy,
            )
        if state.generated_attribute_draft is not None:
            self.output_repository.save_generated_attribute_draft(
                workflow_id=data_package_id,
                document=state.generated_attribute_draft,
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
        self.output_repository.save_parent_attribute_ledger(
            workflow_id=data_package_id,
            ledger=state.parent_attribute_ledger,
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
                "parent_attribute_ledger": [],
                "field_completion_ledger": [],
                "curation_ledger": [],
            }
        )
        self._save_run_state(data_package_id, cleared)

    def _clear_token_usage_agents(
        self,
        data_package_id: str,
        *,
        agents: set[str],
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> None:
        if self.output_repository is None:
            return
        state = self._load_run_state_or_none(
            data_package_id,
            chunking_strategy=chunking_strategy or "semantic",
            chat_model=chat_model,
        )
        resolved_chat_model = chat_model or (
            state.chat_model
            if state
            else (self.ollama_client.chat_model if self.ollama_client else None)
        )
        resolved_chunking_strategy = chunking_strategy or (
            state.chunking_strategy if state else "semantic"
        )
        totals = self.output_repository.load_token_usage(
            data_package_id,
            chat_model=resolved_chat_model,
            chunking_strategy=resolved_chunking_strategy,
        )
        pruned = {
            agent_name: values
            for agent_name, values in totals.items()
            if agent_name not in agents
        }
        self.output_repository.save_token_usage(
            workflow_id=data_package_id,
            token_usage=pruned,
            chat_model=resolved_chat_model,
            chunking_strategy=resolved_chunking_strategy,
        )

    def _clear_evidence_token_usage(
        self,
        data_package_id: str,
        *,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> None:
        self._clear_token_usage_agents(
            data_package_id,
            agents={
                "chunk_extraction",
                "chunk_extraction_repair",
                "evidence_critic",
                "evidence_critic_repair",
            },
            chunking_strategy=chunking_strategy,
            chat_model=chat_model,
        )

    def _clear_initial_context_token_usage(self, data_package_id: str) -> None:
        self._clear_token_usage_agents(
            data_package_id,
            agents={
                "file_ranking",
                "initial_file_summary",
                "initial_extraction_overview",
                "initial_extraction_overview_fallback",
                "dataset_summary",
            },
        )

    def _clear_profile_projection_token_usage(
        self,
        data_package_id: str,
        *,
        chunking_strategy: str | None = None,
        chat_model: str | None = None,
    ) -> None:
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
        self._clear_token_usage_agents(
            data_package_id,
            agents=projection_agents,
            chunking_strategy=chunking_strategy,
            chat_model=chat_model,
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
        rdf_type_term: dict[str, Any] | None = None,
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
                    rdf_type_term=rdf_type_term,
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
                rdf_type_term=rdf_type_term,
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
        schema_title = schema.get("title")
        if schema_title == "DefinedTerm":
            return True
        properties = schema.get("properties", {})
        return False

    @classmethod
    def _term_object_for_schema(
        cls,
        *,
        schema: dict[str, Any],
        root_schema: dict[str, Any],
        selected_uri: str,
        selected_title: str | None,
        vocabulary_identifier: str | None,
        rdf_type_term: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        schema = cls._resolve_schema_node(schema, root_schema)
        for union_key in ("anyOf", "oneOf"):
            options = schema.get(union_key) if isinstance(schema, dict) else None
            if isinstance(options, list):
                for option in options:
                    if isinstance(option, dict) and cls._schema_accepts_term_object(option, root_schema):
                        schema = cls._resolve_schema_node(option, root_schema)
                        break
                break
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not isinstance(properties, dict):
            return {"id": selected_uri}
        value: dict[str, Any] = {"id": selected_uri}
        if selected_title is not None and (schema.get("title") == "DefinedTerm" or "title" in properties):
            value["title"] = selected_title
        if vocabulary_identifier is not None and (
            schema.get("title") == "DefinedTerm" or "from_CV" in properties
        ):
            value["from_CV"] = vocabulary_identifier
        if "rdf_type" in properties and rdf_type_term is not None:
            value["rdf_type"] = rdf_type_term
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
    def _dataset_only_initial_draft(cls, document: dict[str, Any]) -> dict[str, Any]:
        initial = cls._clone_json_object(document)
        initial.pop("is_about_entity", None)
        initial.pop("is_about_activity", None)
        initial["was_generated_by"] = []
        return initial

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
        if title and ("title" in properties or "title" in document):
            document["title"] = cls._fallback_property_value(properties.get("title"), title)
        if "identifier" in properties:
            document["identifier"] = cls._fallback_property_value(
                properties.get("identifier"),
                data_package_id,
            )
        if "id" in properties:
            document["id"] = data_package_id
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
            "keyword",
            "type",
            "modification_date",
        }
        evidence_slots: set[str] = set()
        if "surrounding_signal" in evidence_categories:
            evidence_slots.add("creator")
            evidence_slots.add("modification_date")

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
            "keyword": [],
            "type": [{"preferred_label": []}],
            "modification_date": "",
        }
        return placeholders.get(slot)

    @staticmethod
    def _placeholder_target_class(slot: str) -> str | None:
        return {
            "creator": "Agent",
            "type": "Concept",
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
            return ["software_signal", "instrument_signal"]
        if path.startswith("/creator"):
            return ["surrounding_signal"]
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
    ) -> str | None:
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
        return None

    @classmethod
    def _profile_vocab_sources(
        cls,
        document: dict[str, Any],
        *,
        enrichable_fields: list[str],
        validation_schema: dict[str, Any] | None = None,
    ) -> list[tuple[str, str, str]]:
        target_fields = ({"has_quantity_type", "unit"} | set(enrichable_fields)) - {"rdf_type"}
        sources: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        def term_source_text(value: Any) -> str:
            if isinstance(value, dict):
                for key in ("title", "preferred_label", "label", "name", "id"):
                    item = value.get(key)
                    if isinstance(item, list):
                        item = next((entry for entry in item if isinstance(entry, str) and entry.strip()), None)
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            return ""

        def object_source_text(value: Any) -> str:
            if not isinstance(value, dict):
                return ""
            parts: list[str] = []
            for key in ("title", "description", "preferred_label", "label", "name", "id"):
                item = value.get(key)
                if isinstance(item, list):
                    item = next((entry for entry in item if isinstance(entry, str) and entry.strip()), None)
                if isinstance(item, str) and item.strip() and item.strip() not in parts:
                    parts.append(item.strip())
            return " ".join(parts)

        def schema_accepts_term_value(field_schema: dict[str, Any]) -> bool:
            if cls._schema_accepts_term_object(field_schema, validation_schema or {}):
                return True
            if validation_schema is None:
                return False
            field_schema = cls._resolve_schema_node(field_schema, validation_schema)
            if cls._schema_is_array(field_schema, validation_schema):
                item_schema = cls._resolve_schema_node(
                    field_schema.get("items", {}),
                    validation_schema,
                )
                return cls._schema_accepts_term_object(item_schema, validation_schema)
            return False

        def term_write_path(field_path: str, field_schema: dict[str, Any]) -> str:
            if validation_schema is None:
                return field_path
            field_schema = cls._resolve_schema_node(field_schema, validation_schema)
            if cls._schema_is_array(field_schema, validation_schema):
                return f"{field_path}/0"
            return field_path

        def collect_term_values(value: Any, path: str) -> list[tuple[str, str]]:
            if isinstance(value, list):
                collected: list[tuple[str, str]] = []
                for index, item in enumerate(value):
                    collected.extend(collect_term_values(item, f"{path}/{index}"))
                return collected
            text = term_source_text(value)
            return [(path, text)] if text else []

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
                if "type" in target_fields and "type" not in value and validation_schema is not None:
                    type_path = f"{path}/{cls._json_pointer_escape('type')}"
                    type_schema = cls._schema_for_json_pointer(
                        validation_schema,
                        type_path,
                    )
                    if schema_accepts_term_value(type_schema):
                        source_text = object_source_text(value)
                        if source_text:
                            record = (term_write_path(type_path, type_schema), "type", source_text)
                            if record not in seen:
                                seen.add(record)
                                sources.append(record)
                for key, item in value.items():
                    if key == "rdf_type":
                        continue
                    item_path = f"{path}/{cls._json_pointer_escape(key)}"
                    schema_accepts_term = False
                    if validation_schema is not None:
                        field_schema = cls._schema_for_json_pointer(
                            validation_schema,
                            item_path,
                        )
                        schema_accepts_term = schema_accepts_term_value(field_schema)
                    include_named_field = key in target_fields and (
                        validation_schema is None
                        or key in {"has_quantity_type", "unit"}
                        or schema_accepts_term
                    )
                    if include_named_field or schema_accepts_term:
                        collector = collect_term_values if schema_accepts_term else collect_scalar_values
                        for scalar_path, scalar in collector(item, item_path):
                            record = (scalar_path, key, scalar)
                            if record not in seen:
                                seen.add(record)
                                sources.append(record)
                    # When has_quantity_type is found but sibling unit key is absent,
                    # emit a synthetic unit source so grounding can infer a unit.
                    if key == "has_quantity_type" and "unit" in target_fields and "unit" not in value:
                        unit_path = f"{item_path.rsplit('/', 1)[0]}/{cls._json_pointer_escape('unit')}"
                        unit_record = (unit_path, "unit", "")
                        if unit_record not in seen:
                            seen.add(unit_record)
                            sources.append(unit_record)
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
