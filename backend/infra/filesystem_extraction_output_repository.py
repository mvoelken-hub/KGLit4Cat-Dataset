import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from app.domain.extraction import (
    CurationLedgerRecord,
    DraftValidationResult,
    EvidenceQueryLedgerEntry,
    ExtractionFileSummary,
    ExtractionNormalization,
    ExtractionOverview,
    ExtractionOverviewStatus,
    ExtractionRunResult,
    ExtractionRunState,
    ExtractionVocabQueryRecord,
    FieldCompletionLedgerRecord,
    FilteredEvidenceLedger,
    InitialFileSummaryDiagnostics,
    InitialFileSummaryStatus,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
    ProjectionLedgerRecord,
    RequirementReport,
    RoutedEvidenceContext,
)


EVIDENCE_CONTEXT_FILE = "evidence_context.json"
PORTABLE_EVIDENCE_FILE = "portable_evidence.json"
CONTEXTUAL_EVIDENCE_FILE = "contextual_evidence.json"
REJECTED_EVIDENCE_FILE = "rejected_evidence.json"
EVIDENCE_ASSESSMENTS_FILE = "evidence_assessments.json"
FILTERED_EVIDENCE_NOTES_FILE = "filtered_evidence_notes.json"
EXTRACTION_RESULT_FILE = "extraction_result.json"
EXTRACTION_RUN_STATE_FILE = "extraction_run_state.json"
EXTRACTION_WARNINGS_FILE = "extraction_warnings.json"
TOKEN_USAGE_FILE = "token_usage.json"
PROMPTS_DIR = "prompts"
INITIAL_FILE_SUMMARIES_FILE = "initial_file_summaries.json"
INITIAL_FILE_SUMMARY_DIAGNOSTICS_FILE = "initial_file_summary_diagnostics.json"
INITIAL_EXTRACTION_OVERVIEW_FILE = "initial_extraction_overview.json"
INITIAL_EXTRACTION_OVERVIEW_DIAGNOSTIC_FILE = "initial_extraction_overview_diagnostic.json"
GENERATED_INITIAL_DRAFT_FILE = "generated_initial_draft.json"
GENERATED_FINAL_DRAFT_FILE = "generated_final_draft.json"
REQUIREMENT_REPORT_FILE = "requirement_report.json"
DATASET_SUMMARY_FILE = "dataset_summary.txt"
CURATED_DOCUMENT_FILE = "curated_document.json"
PROJECTION_LEDGER_FILE = "projection_ledger.json"
FIELD_COMPLETION_LEDGER_FILE = "field_completion_ledger.json"
EVIDENCE_QUERY_LEDGER_FILE = "evidence_query_ledger.json"
CURATION_LEDGER_FILE = "curation_ledger.json"
VALIDATION_FILE = "validation.json"
VOCAB_QUERIES_FILE = "vocab_queries.json"
NORMALIZATION_FILE = "normalization.json"
ARTIFACT_INDEX_FILE = "artifact_index.json"


class FileSystemExtractionOutputRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_evidence_context(
        self,
        *,
        workflow_id: str,
        evidence_context: RoutedEvidenceContext,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> None:
        output_dir = self._branch_dir(workflow_id, "evidence_notes", chunking_strategy, chat_model)
        self._write_json_artifact(output_dir / EVIDENCE_CONTEXT_FILE, evidence_context.model_dump(mode="json"))
        self._write_json_artifact(output_dir / PORTABLE_EVIDENCE_FILE, [item.model_dump(mode="json") for item in evidence_context.portable_evidence])
        self._write_json_artifact(output_dir / CONTEXTUAL_EVIDENCE_FILE, [item.model_dump(mode="json") for item in evidence_context.contextual_evidence])
        self._write_json_artifact(output_dir / REJECTED_EVIDENCE_FILE, [item.model_dump(mode="json") for item in evidence_context.rejected_evidence])
        self._write_json_artifact(output_dir / EVIDENCE_ASSESSMENTS_FILE, [item.model_dump(mode="json") for item in evidence_context.assessments])

    def load_evidence_context(
        self,
        workflow_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> RoutedEvidenceContext:
        path = self._branch_dir(workflow_id, "evidence_notes", chunking_strategy, chat_model) / EVIDENCE_CONTEXT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Evidence context output not found for workflow '{workflow_id}'.")
        return RoutedEvidenceContext.model_validate(self._read_json_file(path))

    def save_filtered_evidence_notes(
        self,
        *,
        workflow_id: str,
        ledger: FilteredEvidenceLedger,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "evidence_notes", chunking_strategy, chat_model)
            / FILTERED_EVIDENCE_NOTES_FILE,
            ledger.model_dump(mode="json"),
        )

    def load_filtered_evidence_notes(
        self,
        workflow_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> FilteredEvidenceLedger:
        path = (
            self._branch_dir(workflow_id, "evidence_notes", chunking_strategy, chat_model)
            / FILTERED_EVIDENCE_NOTES_FILE
        )
        if not path.exists():
            return FilteredEvidenceLedger()
        return FilteredEvidenceLedger.model_validate(self._read_json_file(path))

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
        chunking_strategy: str = "semantic",
    ) -> None:
        chat_model = result.chat_model
        self.save_initial_file_summaries(workflow_id=workflow_id, summaries=result.initial_file_summaries, status=result.initial_file_summary_status, chat_model=chat_model)
        self.save_initial_extraction_overview(workflow_id=workflow_id, overview=result.initial_extraction_overview, status=result.initial_extraction_overview_status, chat_model=chat_model)
        if result.generated_initial_draft is not None:
            self.save_generated_initial_draft(workflow_id=workflow_id, document=result.generated_initial_draft, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_generated_final_draft(workflow_id=workflow_id, document=result.generated_final_draft, chat_model=chat_model, chunking_strategy=chunking_strategy)
        if result.requirement_report is not None:
            self.save_requirement_report(workflow_id=workflow_id, report=result.requirement_report, chat_model=chat_model, chunking_strategy=chunking_strategy)
        if result.curated_document is not None:
            self.save_curated_document(workflow_id=workflow_id, document=result.curated_document, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_projection_ledger(workflow_id=workflow_id, ledger=result.projection_ledger, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_field_completion_ledger(workflow_id=workflow_id, ledger=result.field_completion_ledger, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_evidence_query_ledger(workflow_id=workflow_id, ledger=result.evidence_query_ledger, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_curation_ledger(workflow_id=workflow_id, ledger=result.curation_ledger, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self.save_validation(workflow_id=workflow_id, validation=result.validation, curated_validation=result.curated_validation, chat_model=chat_model, chunking_strategy=chunking_strategy)
        self._write_json_artifact(
            self._result_dir(workflow_id, chunking_strategy, chat_model) / EXTRACTION_RESULT_FILE,
            result.model_dump(mode="json"),
        )

    def load_extraction_result(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> ExtractionRunResult:
        path = self._result_dir(workflow_id, chunking_strategy, chat_model) / EXTRACTION_RESULT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Extraction result output not found for workflow '{workflow_id}'.")
        return ExtractionRunResult.model_validate(self._read_json_file(path))

    def save_initial_file_summaries(
        self,
        *,
        workflow_id: str,
        summaries: list[ExtractionFileSummary],
        status: InitialFileSummaryStatus | None,
        chat_model: str | None = None,
    ) -> None:
        self._write_json_artifact(
            self._overview_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARIES_FILE,
            {"status": status, "summaries": [summary.model_dump(mode="json") for summary in summaries]},
        )

    def load_initial_file_summaries(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[list[ExtractionFileSummary], InitialFileSummaryStatus | None]:
        path = self._overview_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARIES_FILE
        if not path.exists():
            raise FileNotFoundError(f"Initial file summaries not found for workflow '{workflow_id}'.")
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Initial file summaries artifact is not a JSON object.")
        status = payload.get("status")
        if status not in {"completed", "partial", "failed", None}:
            status = "failed"
        summaries = payload.get("summaries")
        return (
            [ExtractionFileSummary.model_validate(item) for item in summaries] if isinstance(summaries, list) else [],
            status,
        )

    def save_initial_file_summary_diagnostics(
        self,
        *,
        workflow_id: str,
        diagnostics: InitialFileSummaryDiagnostics | None,
        chat_model: str | None = None,
    ) -> None:
        path = self._overview_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARY_DIAGNOSTICS_FILE
        if diagnostics is None:
            path.unlink(missing_ok=True)
            self._refresh_artifact_index(workflow_id)
            return
        self._write_json_artifact(path, diagnostics.model_dump(mode="json"))

    def save_initial_extraction_overview(
        self,
        *,
        workflow_id: str,
        overview: ExtractionOverview | None,
        status: ExtractionOverviewStatus | None,
        chat_model: str | None = None,
    ) -> None:
        self._write_json_artifact(
            self._overview_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_FILE,
            {"status": status, "overview": overview.model_dump(mode="json") if overview is not None else None},
        )

    def load_initial_extraction_overview(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[ExtractionOverview | None, ExtractionOverviewStatus | None]:
        path = self._overview_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_FILE
        if not path.exists():
            raise FileNotFoundError(f"Initial extraction overview not found for workflow '{workflow_id}'.")
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Initial extraction overview artifact is not a JSON object.")
        status = payload.get("status")
        if status not in {"structured", "unstructured_fallback", "failed", None}:
            status = "failed"
        overview = payload.get("overview")
        return (
            ExtractionOverview.model_validate(overview) if isinstance(overview, dict) else None,
            status,
        )

    def save_initial_extraction_overview_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None,
        chat_model: str | None = None,
    ) -> None:
        path = self._overview_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_DIAGNOSTIC_FILE
        if diagnostic is None:
            path.unlink(missing_ok=True)
            self._refresh_artifact_index(workflow_id)
            return
        self._write_json_artifact(path, diagnostic.model_dump(mode="json"))

    def save_generated_initial_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model)
            / GENERATED_INITIAL_DRAFT_FILE,
            document,
        )

    def save_generated_final_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / GENERATED_FINAL_DRAFT_FILE,
            document,
        )

    def save_requirement_report(
        self,
        *,
        workflow_id: str,
        report: RequirementReport,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / REQUIREMENT_REPORT_FILE,
            report.model_dump(mode="json"),
        )

    def load_generated_final_draft(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, Any]:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / GENERATED_FINAL_DRAFT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Generated final draft not found for workflow '{workflow_id}'.")
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Generated final draft artifact is not a JSON object.")
        return payload

    def save_dataset_summary(
        self,
        *,
        workflow_id: str,
        summary: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / DATASET_SUMMARY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary, encoding="utf-8")
        self._refresh_artifact_index(workflow_id)

    def save_curated_document(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._result_dir(workflow_id, chunking_strategy, chat_model) / CURATED_DOCUMENT_FILE,
            document,
        )

    def load_curated_document(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, Any]:
        path = self._result_dir(workflow_id, chunking_strategy, chat_model) / CURATED_DOCUMENT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Curated document not found for workflow '{workflow_id}'.")
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Curated document artifact is not a JSON object.")
        return payload

    def save_projection_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[ProjectionLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / PROJECTION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_projection_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[ProjectionLedgerRecord]:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / PROJECTION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        return [ProjectionLedgerRecord.model_validate(item) for item in payload] if isinstance(payload, list) else []

    def save_field_completion_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[FieldCompletionLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / FIELD_COMPLETION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_field_completion_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[FieldCompletionLedgerRecord]:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / FIELD_COMPLETION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        return [FieldCompletionLedgerRecord.model_validate(item) for item in payload] if isinstance(payload, list) else []

    def save_evidence_query_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[EvidenceQueryLedgerEntry],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / EVIDENCE_QUERY_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_evidence_query_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[EvidenceQueryLedgerEntry]:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / EVIDENCE_QUERY_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        return [EvidenceQueryLedgerEntry.model_validate(item) for item in payload] if isinstance(payload, list) else []

    def save_curation_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[CurationLedgerRecord],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._result_dir(workflow_id, chunking_strategy, chat_model) / CURATION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_curation_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> list[CurationLedgerRecord]:
        path = self._result_dir(workflow_id, chunking_strategy, chat_model) / CURATION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        return [CurationLedgerRecord.model_validate(item) for item in payload] if isinstance(payload, list) else []

    def save_validation(
        self,
        *,
        workflow_id: str,
        validation: DraftValidationResult,
        curated_validation: DraftValidationResult | None = None,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / VALIDATION_FILE,
            {
                "generated": validation.model_dump(mode="json"),
                "curated": curated_validation.model_dump(mode="json") if curated_validation is not None else None,
            },
        )

    def load_validation(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> tuple[DraftValidationResult, DraftValidationResult | None]:
        path = self._branch_dir(workflow_id, "profile_draft", chunking_strategy, chat_model) / VALIDATION_FILE
        if not path.exists():
            raise FileNotFoundError(f"Validation artifact not found for workflow '{workflow_id}'.")
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Validation artifact is not a JSON object.")
        curated = payload.get("curated")
        return (
            DraftValidationResult.model_validate(payload.get("generated", {})),
            DraftValidationResult.model_validate(curated) if isinstance(curated, dict) else None,
        )

    def save_grounding_artifacts(
        self,
        *,
        workflow_id: str,
        vocab_queries: list[ExtractionVocabQueryRecord],
        normalization: ExtractionNormalization,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        output_dir = self._branch_dir(workflow_id, "grounding", chunking_strategy, chat_model)
        self._write_json_artifact(output_dir / VOCAB_QUERIES_FILE, [item.model_dump(mode="json") for item in vocab_queries])
        self._write_json_artifact(output_dir / NORMALIZATION_FILE, normalization.model_dump(mode="json"))

    def save_extraction_run_state(
        self,
        *,
        workflow_id: str,
        state: ExtractionRunState,
    ) -> None:
        self._write_json_artifact(self._workflow_dir(workflow_id) / EXTRACTION_RUN_STATE_FILE, state.model_dump(mode="json"))

    def load_extraction_run_state(self, workflow_id: str, chat_model: str | None = None) -> ExtractionRunState:
        path = self._workflow_dir(workflow_id) / EXTRACTION_RUN_STATE_FILE
        if not path.exists():
            raise FileNotFoundError(f"Extraction run state not found for workflow '{workflow_id}'.")
        return ExtractionRunState.model_validate(self._read_json_file(path))

    def save_extraction_warnings(self, *, workflow_id: str, warnings: list[str]) -> None:
        self._write_json_artifact(self._workflow_dir(workflow_id) / EXTRACTION_WARNINGS_FILE, warnings)

    def load_extraction_warnings(self, workflow_id: str) -> list[str]:
        path = self._workflow_dir(workflow_id) / EXTRACTION_WARNINGS_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        return [str(item) for item in payload] if isinstance(payload, list) else []

    def save_token_usage(
        self,
        *,
        workflow_id: str,
        token_usage: dict[str, dict[str, int]],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        self._write_json_artifact(
            self._result_dir(workflow_id, chunking_strategy, chat_model) / TOKEN_USAGE_FILE,
            token_usage,
        )

    def load_token_usage(
        self,
        workflow_id: str,
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> dict[str, dict[str, int]]:
        path = self._result_dir(workflow_id, chunking_strategy, chat_model) / TOKEN_USAGE_FILE
        if not path.exists():
            return {}
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            return {}
        return {
            str(agent_name): {key: self._safe_int(values.get(key)) for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "requests",
                "operation_count",
                "prompt_eval_duration_ms",
                "load_duration_ms",
                "response_duration_ms",
                "total_duration_ms",
            )}
            for agent_name, values in payload.items()
            if isinstance(values, dict)
        }

    def append_prompt_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: dict[str, Any],
        chat_model: str | None = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        operation_id = str(diagnostic.get("operation_id") or "structured_completion")
        diagnostics_dir = self._prompt_dir(workflow_id, operation_id, chunking_strategy, chat_model)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        safe_id = self._sanitize_path_component(operation_id)[:96]
        existing = sorted(diagnostics_dir.glob(f"{safe_id}__*.json"))
        self._write_json_artifact(diagnostics_dir / f"{safe_id}__{len(existing) + 1:04d}.json", diagnostic)

    def clear_prompt_diagnostics(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if not workflow_dir.exists():
            return
        for prompts_dir in workflow_dir.rglob(PROMPTS_DIR):
            if prompts_dir.is_dir():
                _remove_tree(prompts_dir)
        self._write_artifact_index(workflow_dir)

    def clear_extraction_run(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if workflow_dir.exists():
            _remove_tree(workflow_dir)

    def clear_extraction_downstream(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if not workflow_dir.exists():
            return
        for name in ("evidence_notes", "profile_draft", "grounding", "result"):
            path = workflow_dir / name
            if path.exists():
                _remove_tree(path)
        for path in (workflow_dir / EXTRACTION_WARNINGS_FILE, workflow_dir / ARTIFACT_INDEX_FILE):
            path.unlink(missing_ok=True)
        self._write_artifact_index(workflow_dir)

    def _overview_dir(self, workflow_id: str, chat_model: str | None = None) -> Path:
        return self._checked_dir(workflow_id, "overview", self._safe_model(chat_model))

    def _branch_dir(
        self,
        workflow_id: str,
        stage: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> Path:
        return self._checked_dir(workflow_id, stage, self._safe_strategy(chunking_strategy), self._safe_model(chat_model))

    def _result_dir(
        self,
        workflow_id: str,
        chunking_strategy: str = "semantic",
        chat_model: str | None = None,
    ) -> Path:
        return self._branch_dir(workflow_id, "result", chunking_strategy, chat_model)

    def _prompt_dir(
        self,
        workflow_id: str,
        operation_id: str,
        chunking_strategy: str,
        chat_model: str | None,
    ) -> Path:
        stage = self._prompt_stage(operation_id)
        if stage == "overview":
            return self._overview_dir(workflow_id, chat_model) / PROMPTS_DIR
        return self._branch_dir(workflow_id, stage, chunking_strategy, chat_model) / PROMPTS_DIR

    def _workflow_dir(self, workflow_id: str) -> Path:
        return self._checked_dir(workflow_id)

    def _checked_dir(self, workflow_id: str, *parts: str | None) -> Path:
        base_path = self.base_path.resolve()
        path = (base_path / self._sanitize_path_component(workflow_id)).resolve()
        for part in parts:
            if part:
                path = (path / part).resolve()
        if not path.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return path

    def _write_json_artifact(self, path: Path, content: Any) -> None:
        self._write_json_file(path, content)
        self._refresh_artifact_index_for_path(path)

    def _refresh_artifact_index_for_path(self, path: Path) -> None:
        base_path = self.base_path.resolve()
        path = path.resolve()
        if not path.is_relative_to(base_path) or path.name == ARTIFACT_INDEX_FILE:
            return
        parts = path.relative_to(base_path).parts
        if parts:
            workflow_dir = base_path / parts[0]
            if workflow_dir.exists():
                self._write_artifact_index(workflow_dir)

    def _refresh_artifact_index(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if workflow_dir.exists():
            self._write_artifact_index(workflow_dir)

    def _write_artifact_index(self, workflow_dir: Path) -> None:
        artifacts = []
        for path in sorted(workflow_dir.rglob("*")):
            if not path.is_file() or path.name == ARTIFACT_INDEX_FILE:
                continue
            relative_path = path.relative_to(workflow_dir).as_posix()
            entry = {
                "path": relative_path,
                "stage": self._stage_from_relative_path(relative_path),
                "strategy": self._strategy_from_relative_path(relative_path),
                "model": self._model_from_relative_path(relative_path),
                "size_bytes": path.stat().st_size,
            }
            if f"/{PROMPTS_DIR}/" in f"/{relative_path}":
                try:
                    payload = self._read_json_file(path)
                    if isinstance(payload, dict):
                        entry["operation_id"] = payload.get("operation_id")
                        entry["agent_name"] = payload.get("agent_name")
                except (OSError, json.JSONDecodeError):
                    pass
            artifacts.append(entry)
        by_stage: dict[str, list[str]] = {}
        for entry in artifacts:
            by_stage.setdefault(str(entry["stage"]), []).append(str(entry["path"]))
        self._write_json_file(
            workflow_dir / ARTIFACT_INDEX_FILE,
            {"workflow_id": workflow_dir.name, "artifacts": artifacts, "by_stage": by_stage},
        )

    @staticmethod
    def _prompt_stage(operation_id: str) -> str:
        operation = operation_id.lower()
        if operation.startswith(("initial_", "file_ranking")):
            return "overview"
        if any(key in operation for key in ("chunk", "evidence_critic", "evidence_extraction")):
            return "evidence_notes"
        if any(key in operation for key in ("vocab", "ground", "normalization")):
            return "grounding"
        return "profile_draft"

    @staticmethod
    def _stage_from_relative_path(path: str) -> str:
        first = path.split("/", 1)[0]
        return first if first in {"chunks", "overview", "evidence_notes", "profile_draft", "grounding", "result"} else "run_state"

    @staticmethod
    def _strategy_from_relative_path(path: str) -> str | None:
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] in {"chunks", "evidence_notes", "profile_draft", "grounding", "result"}:
            return parts[1]
        return None

    @staticmethod
    def _model_from_relative_path(path: str) -> str | None:
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "overview":
            return parts[1]
        if len(parts) >= 4 and parts[0] in {"evidence_notes", "profile_draft", "grounding", "result"}:
            return parts[2]
        return None

    @classmethod
    def _safe_model(cls, chat_model: str | None) -> str:
        return cls._sanitize_path_component(chat_model or "default-model")

    @classmethod
    def _safe_strategy(cls, chunking_strategy: str | None) -> str:
        return cls._sanitize_path_component(chunking_strategy or "semantic")

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", name)

    @staticmethod
    def _read_json_file(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json_file(path: Path, content: Any) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".tmp",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                json.dump(content, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            _atomic_replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


def _atomic_replace(src: Path, dst: Path, *, retries: int = 5, delay: float = 0.15) -> None:
    replace_src = _long_path_for_windows(src)
    replace_dst = _long_path_for_windows(dst)
    for attempt in range(retries):
        try:
            os.replace(replace_src, replace_dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _remove_tree(path: Path) -> None:
    def ignore_missing(_function: Any, _path: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        raise exc

    shutil.rmtree(_long_path_for_windows(path), onexc=ignore_missing)


def _long_path_for_windows(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved
