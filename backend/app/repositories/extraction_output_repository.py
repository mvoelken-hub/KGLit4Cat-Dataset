from typing import Any, Protocol

from app.domain.extraction import (
    CurationLedgerRecord,
    DraftValidationResult,
    ExtractionFileSummary,
    InitialFileSummaryDiagnostics,
    InitialFileSummaryStatus,
    ExtractionOverview,
    ExtractionOverviewStatus,
    FilteredEvidenceLedger,
    ExtractionRunResult,
    ExtractionRunState,
    FieldCompletionLedgerRecord,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
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
    ) -> None:
        ...

    def load_evidence_context(self, workflow_id: str) -> RoutedEvidenceContext:
        ...

    def save_filtered_evidence_notes(
        self,
        *,
        workflow_id: str,
        ledger: FilteredEvidenceLedger,
    ) -> None:
        ...

    def load_filtered_evidence_notes(self, workflow_id: str) -> FilteredEvidenceLedger:
        ...

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
    ) -> None:
        ...

    def load_extraction_result(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> ExtractionRunResult:
        ...

    def save_initial_file_summaries(
        self,
        *,
        workflow_id: str,
        summaries: list[ExtractionFileSummary],
        status: InitialFileSummaryStatus | None,
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_initial_file_summaries(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[list[ExtractionFileSummary], InitialFileSummaryStatus | None]:
        ...

    def save_initial_file_summary_diagnostics(
        self,
        *,
        workflow_id: str,
        diagnostics: InitialFileSummaryDiagnostics | None,
        chat_model: str | None = None,
    ) -> None:
        ...

    def save_initial_extraction_overview(
        self,
        *,
        workflow_id: str,
        overview: ExtractionOverview | None,
        status: ExtractionOverviewStatus | None,
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_initial_extraction_overview(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[ExtractionOverview | None, ExtractionOverviewStatus | None]:
        ...

    def save_initial_extraction_overview_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None,
        chat_model: str | None = None,
    ) -> None:
        ...

    def save_generated_initial_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        ...

    def save_generated_final_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        ...

    def save_requirement_report(
        self,
        *,
        workflow_id: str,
        report: RequirementReport,
        chat_model: str | None = None,
    ) -> None:
        ...

    def save_dataset_summary(
        self,
        *,
        workflow_id: str,
        summary: str,
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_generated_final_draft(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> dict[str, Any]:
        ...

    def save_curated_document(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_curated_document(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> dict[str, Any]:
        ...

    def save_projection_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[ProjectionLedgerRecord],
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_projection_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[ProjectionLedgerRecord]:
        ...

    def save_field_completion_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[FieldCompletionLedgerRecord],
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_field_completion_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[FieldCompletionLedgerRecord]:
        ...

    def save_curation_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[CurationLedgerRecord],
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_curation_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[CurationLedgerRecord]:
        ...

    def save_validation(
        self,
        *,
        workflow_id: str,
        validation: DraftValidationResult,
        curated_validation: DraftValidationResult | None = None,
        chat_model: str | None = None,
    ) -> None:
        ...

    def load_validation(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[DraftValidationResult, DraftValidationResult | None]:
        ...

    def save_extraction_run_state(
        self,
        *,
        workflow_id: str,
        state: ExtractionRunState,
    ) -> None:
        ...

    def load_extraction_run_state(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> ExtractionRunState:
        ...

    def save_extraction_warnings(
        self,
        *,
        workflow_id: str,
        warnings: list[str],
    ) -> None:
        ...

    def load_extraction_warnings(self, workflow_id: str) -> list[str]:
        ...

    def save_token_usage(
        self,
        *,
        workflow_id: str,
        token_usage: dict[str, dict[str, int]],
    ) -> None:
        ...

    def load_token_usage(self, workflow_id: str) -> dict[str, dict[str, int]]:
        ...

    def append_prompt_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        ...

    def clear_prompt_diagnostics(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> None:
        ...

    def clear_extraction_run(self, workflow_id: str) -> None:
        ...

    def clear_extraction_downstream(self, workflow_id: str) -> None:
        ...
