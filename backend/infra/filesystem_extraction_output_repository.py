import json
import re
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from app.domain.extraction import (
    CurationLedgerRecord,
    DraftValidationResult,
    EvidenceContext,
    FilteredEvidenceLedger,
    ExtractionFileSummary,
    InitialFileSummaryDiagnostics,
    InitialFileSummaryStatus,
    ExtractionOverview,
    ExtractionOverviewStatus,
    ExtractionRunResult,
    ExtractionRunState,
    FieldCompletionLedgerRecord,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
    ProjectionLedgerRecord,
)


EVIDENCE_CONTEXT_FILE = "evidence_context.json"
FILTERED_EVIDENCE_NOTES_FILE = "filtered_evidence_notes.json"
EXTRACTION_RESULT_FILE = "extraction_result.json"
EXTRACTION_RUN_STATE_FILE = "extraction_run_state.json"
EXTRACTION_WARNINGS_FILE = "extraction_warnings.json"
TOKEN_USAGE_FILE = "token_usage.json"
INITIAL_FILE_SUMMARIES_FILE = "initial_file_summaries.json"
INITIAL_FILE_SUMMARY_DIAGNOSTICS_FILE = "initial_file_summary_diagnostics.json"
INITIAL_EXTRACTION_OVERVIEW_FILE = "initial_extraction_overview.json"
INITIAL_EXTRACTION_OVERVIEW_DIAGNOSTIC_FILE = "initial_extraction_overview_diagnostic.json"
GENERATED_FINAL_DRAFT_FILE = "generated_final_draft.json"
CURATED_DOCUMENT_FILE = "curated_document.json"
PROJECTION_LEDGER_FILE = "projection_ledger.json"
FIELD_COMPLETION_LEDGER_FILE = "field_completion_ledger.json"
CURATION_LEDGER_FILE = "curation_ledger.json"
VALIDATION_FILE = "validation.json"


class FileSystemExtractionOutputRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_evidence_context(
        self,
        *,
        workflow_id: str,
        evidence_context: EvidenceContext,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / EVIDENCE_CONTEXT_FILE,
            evidence_context.model_dump(mode="json"),
        )

    def load_evidence_context(self, workflow_id: str) -> EvidenceContext:
        path = self._workflow_dir(workflow_id) / EVIDENCE_CONTEXT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Evidence context output not found for workflow '{workflow_id}'."
            )
        return EvidenceContext.model_validate(self._read_json_file(path))

    def save_filtered_evidence_notes(
        self,
        *,
        workflow_id: str,
        ledger: FilteredEvidenceLedger,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / FILTERED_EVIDENCE_NOTES_FILE,
            ledger.model_dump(mode="json"),
        )

    def load_filtered_evidence_notes(self, workflow_id: str) -> FilteredEvidenceLedger:
        path = self._workflow_dir(workflow_id) / FILTERED_EVIDENCE_NOTES_FILE
        if not path.exists():
            return FilteredEvidenceLedger()
        return FilteredEvidenceLedger.model_validate(self._read_json_file(path))

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
    ) -> None:
        self.save_initial_file_summaries(
            workflow_id=workflow_id,
            summaries=result.initial_file_summaries,
            status=result.initial_file_summary_status,
            chat_model=result.chat_model,
        )
        self.save_initial_extraction_overview(
            workflow_id=workflow_id,
            overview=result.initial_extraction_overview,
            status=result.initial_extraction_overview_status,
            chat_model=result.chat_model,
        )
        self.save_generated_final_draft(
            workflow_id=workflow_id,
            document=result.generated_final_draft,
            chat_model=result.chat_model,
        )
        if result.curated_document is not None:
            self.save_curated_document(
                workflow_id=workflow_id,
                document=result.curated_document,
                chat_model=result.chat_model,
            )
        self.save_projection_ledger(
            workflow_id=workflow_id,
            ledger=result.projection_ledger,
            chat_model=result.chat_model,
        )
        self.save_field_completion_ledger(
            workflow_id=workflow_id,
            ledger=result.field_completion_ledger,
            chat_model=result.chat_model,
        )
        self.save_curation_ledger(
            workflow_id=workflow_id,
            ledger=result.curation_ledger,
            chat_model=result.chat_model,
        )
        self.save_validation(
            workflow_id=workflow_id,
            validation=result.validation,
            curated_validation=result.curated_validation,
            chat_model=result.chat_model,
        )
        self._write_json_file(
            self._workflow_dir(workflow_id, result.chat_model) / EXTRACTION_RESULT_FILE,
            result.model_dump(mode="json"),
        )

    def load_extraction_result(self, workflow_id: str, chat_model: str | None = None) -> ExtractionRunResult:
        path = self._workflow_dir(workflow_id, chat_model) / EXTRACTION_RESULT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Extraction result output not found for workflow '{workflow_id}'."
            )
        return ExtractionRunResult.model_validate(self._read_json_file(path))

    def save_initial_file_summaries(
        self,
        *,
        workflow_id: str,
        summaries: list[ExtractionFileSummary],
        status: InitialFileSummaryStatus | None,
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARIES_FILE,
            {
                "status": status,
                "summaries": [summary.model_dump(mode="json") for summary in summaries],
            },
        )

    def load_initial_file_summaries(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[list[ExtractionFileSummary], InitialFileSummaryStatus | None]:
        path = self._workflow_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARIES_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Initial file summaries not found for workflow '{workflow_id}'."
            )
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Initial file summaries artifact is not a JSON object.")
        summaries_payload = payload.get("summaries")
        summaries = (
            [ExtractionFileSummary.model_validate(item) for item in summaries_payload]
            if isinstance(summaries_payload, list)
            else []
        )
        status = payload.get("status")
        if status not in {"completed", "partial", "failed", None}:
            status = "failed"
        return summaries, status

    def save_initial_file_summary_diagnostics(
        self,
        *,
        workflow_id: str,
        diagnostics: InitialFileSummaryDiagnostics | None,
        chat_model: str | None = None,
    ) -> None:
        path = self._workflow_dir(workflow_id, chat_model) / INITIAL_FILE_SUMMARY_DIAGNOSTICS_FILE
        if diagnostics is None:
            path.unlink(missing_ok=True)
            return
        self._write_json_file(path, diagnostics.model_dump(mode="json"))

    def save_initial_extraction_overview(
        self,
        *,
        workflow_id: str,
        overview: ExtractionOverview | None,
        status: ExtractionOverviewStatus | None,
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_FILE,
            {
                "status": status,
                "overview": (
                    overview.model_dump(mode="json")
                    if overview is not None
                    else None
                ),
            },
        )

    def load_initial_extraction_overview(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[ExtractionOverview | None, ExtractionOverviewStatus | None]:
        path = self._workflow_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Initial extraction overview not found for workflow '{workflow_id}'."
            )
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Initial extraction overview artifact is not a JSON object.")
        overview_payload = payload.get("overview")
        overview = (
            ExtractionOverview.model_validate(overview_payload)
            if isinstance(overview_payload, dict)
            else None
        )
        status = payload.get("status")
        if status not in {"structured", "unstructured_fallback", "failed", None}:
            status = "failed"
        return overview, status

    def save_initial_extraction_overview_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic: InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None,
        chat_model: str | None = None,
    ) -> None:
        path = self._workflow_dir(workflow_id, chat_model) / INITIAL_EXTRACTION_OVERVIEW_DIAGNOSTIC_FILE
        if diagnostic is None:
            path.unlink(missing_ok=True)
            return
        self._write_json_file(path, diagnostic.model_dump(mode="json"))

    def save_generated_final_draft(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / GENERATED_FINAL_DRAFT_FILE,
            document,
        )

    def load_generated_final_draft(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> dict[str, Any]:
        path = self._workflow_dir(workflow_id, chat_model) / GENERATED_FINAL_DRAFT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Generated final draft not found for workflow '{workflow_id}'."
            )
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Generated final draft artifact is not a JSON object.")
        return payload

    def save_curated_document(
        self,
        *,
        workflow_id: str,
        document: dict[str, Any],
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / CURATED_DOCUMENT_FILE,
            document,
        )

    def load_curated_document(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> dict[str, Any]:
        path = self._workflow_dir(workflow_id, chat_model) / CURATED_DOCUMENT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Curated document not found for workflow '{workflow_id}'."
            )
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
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / PROJECTION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_projection_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[ProjectionLedgerRecord]:
        path = self._workflow_dir(workflow_id, chat_model) / PROJECTION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        if not isinstance(payload, list):
            return []
        return [ProjectionLedgerRecord.model_validate(item) for item in payload]

    def save_field_completion_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[FieldCompletionLedgerRecord],
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / FIELD_COMPLETION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_field_completion_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[FieldCompletionLedgerRecord]:
        path = self._workflow_dir(workflow_id, chat_model) / FIELD_COMPLETION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        if not isinstance(payload, list):
            return []
        return [FieldCompletionLedgerRecord.model_validate(item) for item in payload]

    def save_curation_ledger(
        self,
        *,
        workflow_id: str,
        ledger: list[CurationLedgerRecord],
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / CURATION_LEDGER_FILE,
            [item.model_dump(mode="json") for item in ledger],
        )

    def load_curation_ledger(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> list[CurationLedgerRecord]:
        path = self._workflow_dir(workflow_id, chat_model) / CURATION_LEDGER_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        if not isinstance(payload, list):
            return []
        return [CurationLedgerRecord.model_validate(item) for item in payload]

    def save_validation(
        self,
        *,
        workflow_id: str,
        validation: DraftValidationResult,
        curated_validation: DraftValidationResult | None = None,
        chat_model: str | None = None,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, chat_model) / VALIDATION_FILE,
            {
                "generated": validation.model_dump(mode="json"),
                "curated": (
                    curated_validation.model_dump(mode="json")
                    if curated_validation is not None
                    else None
                ),
            },
        )

    def load_validation(
        self,
        workflow_id: str,
        chat_model: str | None = None,
    ) -> tuple[DraftValidationResult, DraftValidationResult | None]:
        path = self._workflow_dir(workflow_id, chat_model) / VALIDATION_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Validation artifact not found for workflow '{workflow_id}'."
            )
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError("Validation artifact is not a JSON object.")
        validation = DraftValidationResult.model_validate(payload.get("generated", {}))
        curated_payload = payload.get("curated")
        curated_validation = (
            DraftValidationResult.model_validate(curated_payload)
            if isinstance(curated_payload, dict)
            else None
        )
        return validation, curated_validation

    def save_extraction_run_state(
        self,
        *,
        workflow_id: str,
        state: ExtractionRunState,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id, state.chat_model) / EXTRACTION_RUN_STATE_FILE,
            state.model_dump(mode="json"),
        )

    def load_extraction_run_state(self, workflow_id: str, chat_model: str | None = None) -> ExtractionRunState:
        path = self._workflow_dir(workflow_id, chat_model) / EXTRACTION_RUN_STATE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Extraction run state not found for workflow '{workflow_id}'."
            )
        return ExtractionRunState.model_validate(self._read_json_file(path))

    def save_extraction_warnings(
        self,
        *,
        workflow_id: str,
        warnings: list[str],
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / EXTRACTION_WARNINGS_FILE,
            warnings,
        )

    def load_extraction_warnings(self, workflow_id: str) -> list[str]:
        path = self._workflow_dir(workflow_id) / EXTRACTION_WARNINGS_FILE
        if not path.exists():
            return []
        payload = self._read_json_file(path)
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload]

    def save_token_usage(
        self,
        *,
        workflow_id: str,
        token_usage: dict[str, dict[str, int]],
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / TOKEN_USAGE_FILE,
            token_usage,
        )

    def load_token_usage(self, workflow_id: str) -> dict[str, dict[str, int]]:
        path = self._workflow_dir(workflow_id) / TOKEN_USAGE_FILE
        if not path.exists():
            return {}
        payload = self._read_json_file(path)
        if not isinstance(payload, dict):
            return {}
        result: dict[str, dict[str, int]] = {}
        for agent_name, values in payload.items():
            if not isinstance(agent_name, str) or not isinstance(values, dict):
                continue
            result[agent_name] = {
                key: self._safe_int(values.get(key))
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "requests",
                    "operation_count",
                    "prompt_eval_duration_ms",
                    "load_duration_ms",
                    "response_duration_ms",
                    "total_duration_ms",
                )
            }
        return result

    def clear_extraction_run(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if workflow_dir.exists():
            shutil.rmtree(workflow_dir)

    def clear_extraction_downstream(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        if not workflow_dir.exists():
            return
        downstream_files = {
            EVIDENCE_CONTEXT_FILE,
            FILTERED_EVIDENCE_NOTES_FILE,
            EXTRACTION_RESULT_FILE,
            GENERATED_FINAL_DRAFT_FILE,
            CURATED_DOCUMENT_FILE,
            PROJECTION_LEDGER_FILE,
            FIELD_COMPLETION_LEDGER_FILE,
            CURATION_LEDGER_FILE,
            VALIDATION_FILE,
        }
        for path in workflow_dir.rglob("*"):
            if path.is_file() and path.name in downstream_files:
                path.unlink()

    def _workflow_dir(self, workflow_id: str, chat_model: str | None = None) -> Path:
        base_path = self.base_path.resolve()
        # Branch output by model so the same dataset can be compared across models.
        # When chat_model is provided, results live under workflow_id/<model>/
        safe_model = self._sanitize_path_component(chat_model) if chat_model else None
        relative = workflow_id if safe_model is None else f"{workflow_id}/{safe_model}"
        workflow_dir = (base_path / relative).resolve()
        if not workflow_dir.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return workflow_dir

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        # Windows forbids < > : " / \ | ? * in file/directory names.
        # Other platforms may also reject colons and slashes, so sanitize universally.
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    @staticmethod
    def _read_json_file(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json_file(path: Path, content: Any) -> None:
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
    """Replace *dst* with *src* atomically, retrying on Windows PermissionError.

    On Windows, ``os.replace`` can fail with ``PermissionError`` when another
    process (antivirus scanner, search indexer, concurrent reader, etc.) still
    holds an open handle on the destination file.  Retrying with a short back-
    off gives the other process time to release the handle.
    """
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
