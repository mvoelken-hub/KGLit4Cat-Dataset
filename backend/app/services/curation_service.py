from __future__ import annotations

from app.services.extraction_shared import *


class CurationService:
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
            warnings=self._load_warnings_or_empty(data_package_id),
        )
        self._update_progress(data_package_id, progress)
        return progress

