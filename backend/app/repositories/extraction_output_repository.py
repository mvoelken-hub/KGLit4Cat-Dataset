from typing import Any, Protocol

from app.domain.extraction import (
    CurationLedgerRecord,
    DescriptionMiningArtifact,
    DraftValidationResult,
    EvidenceQueryLedgerEntry,
    ExtractionFileSummary,
    ExtractionNormalization,
    ExtractionOverview,
    ExtractionOverviewStatus,
    ExtractionRunResult,
    ExtractionRunState,
    ExtractionVocabQueryConfig,
    ExtractionVocabQueryRecord,
    FieldCompletionLedgerRecord,
    FilteredEvidenceLedger,
    InitialFileSummaryDiagnostics,
    InitialFileSummaryStatus,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
    ParentAttributeLedgerRecord,
    ProjectionLedgerRecord,
    RequirementReport,
    RoutedEvidenceContext,
)


class ExtractionOutputRepository(Protocol):
    def save_evidence_context(
        self,
        *,
        workflow_id: str,
        evidence_context: RoutedEvidenceContext,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> None: ...

    def save_description_facts(
        self,
        *,
        workflow_id: str,
        artifact: DescriptionMiningArtifact,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_evidence_context(
        self,
        workflow_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> RoutedEvidenceContext: ...

    def save_filtered_evidence_notes(
        self,
        *,
        workflow_id: str,
        ledger: FilteredEvidenceLedger,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> None: ...

    def load_filtered_evidence_notes(
        self,
        workflow_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> FilteredEvidenceLedger: ...

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_extraction_result(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> ExtractionRunResult: ...

    def save_initial_file_summaries(
        self,
        *,
        workflow_id: str,
        summaries: list[ExtractionFileSummary],
        status: InitialFileSummaryStatus | None,
        chat_model: str | None = None,
    ) -> None: ...

    def load_initial_file_summaries(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[list[ExtractionFileSummary], InitialFileSummaryStatus | None]: ...

    def save_initial_file_summary_diagnostics(
        self,
        *,
        workflow_id: str,
        diagnostics: InitialFileSummaryDiagnostics | None,
        chat_model: str | None = None,
    ) -> None: ...

    def save_initial_extraction_overview(
        self,
        *,
        workflow_id: str,
        overview: ExtractionOverview | None,
        status: ExtractionOverviewStatus | None,
        chat_model: str | None = None,
    ) -> None: ...

    def load_initial_extraction_overview(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[ExtractionOverview | None, ExtractionOverviewStatus | None]: ...

    def save_initial_extraction_overview_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None,
        chat_model: str | None = None,
    ) -> None: ...

    def save_generated_initial_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_generated_final_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_generated_core_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_generated_attribute_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_requirement_report(
        self,
        *,
        workflow_id: str,
        report: RequirementReport,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_generated_final_draft(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, Any]: ...

    def save_dataset_summary(
        self,
        *,
        workflow_id: str,
        summary: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_curated_document(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_curated_document(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, Any]: ...

    def save_projection_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[ProjectionLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_projection_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[ProjectionLedgerRecord]: ...

    def save_parent_attribute_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[ParentAttributeLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_parent_attribute_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[ParentAttributeLedgerRecord]: ...

    def save_field_completion_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[FieldCompletionLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_field_completion_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[FieldCompletionLedgerRecord]: ...

    def save_evidence_query_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[EvidenceQueryLedgerEntry],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_evidence_query_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[EvidenceQueryLedgerEntry]: ...

    def save_curation_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[CurationLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_curation_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[CurationLedgerRecord]: ...

    def save_validation(
        self,
        *,
        workflow_id: str,
        validation: DraftValidationResult,
        curated_validation: DraftValidationResult | None = None,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_validation(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> tuple[DraftValidationResult, DraftValidationResult | None]: ...

    def save_grounding_artifacts(
        self,
        *,
        workflow_id: str,
        vocab_queries: list[ExtractionVocabQueryRecord],
        normalization: ExtractionNormalization,
        grounding_policy: ExtractionVocabQueryConfig | None = None,
        grounded_validation: DraftValidationResult | None = None,
        grounded_document: dict[str, Any] | None = None,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def save_extraction_run_state(
        self,
        *,
        workflow_id: str,
        state: ExtractionRunState,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> None: ...

    def load_extraction_run_state(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> ExtractionRunState: ...

    def save_extraction_warnings(self, *, workflow_id: str, warnings: list[str]) -> None: ...

    def load_extraction_warnings(self, workflow_id: str) -> list[str]: ...

    def save_token_usage(
        self,
        *,
        workflow_id: str,
        token_usage: dict[str, dict[str, int]],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def load_token_usage(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, dict[str, int]]: ...

    def append_prompt_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None: ...

    def clear_prompt_diagnostics(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str | None = None,
        stage: str | None = None,
    ) -> None: ...

    def clear_initial_context(self, workflow_id: str) -> None: ...

    def clear_extraction_run(self, workflow_id: str) -> None: ...

    def clear_extraction_downstream(self, workflow_id: str) -> None: ...
