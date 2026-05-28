import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from app.domain.extraction import ExtractionContext, ExtractionRunResult, ExtractionRunState


EXTRACTION_CONTEXT_FILE = "extraction_context.json"
EXTRACTION_RESULT_FILE = "extraction_result.json"
EXTRACTION_RUN_STATE_FILE = "extraction_run_state.json"
EXTRACTION_WARNINGS_FILE = "extraction_warnings.json"
TOKEN_USAGE_FILE = "token_usage.json"


class FileSystemExtractionOutputRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_extraction_context(
        self,
        *,
        workflow_id: str,
        extraction_context: ExtractionContext,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / EXTRACTION_CONTEXT_FILE,
            extraction_context.model_dump(mode="json"),
        )

    def load_extraction_context(self, workflow_id: str) -> ExtractionContext:
        path = self._workflow_dir(workflow_id) / EXTRACTION_CONTEXT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Extraction context output not found for workflow '{workflow_id}'."
            )
        return ExtractionContext.model_validate(self._read_json_file(path))

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / EXTRACTION_RESULT_FILE,
            result.model_dump(mode="json"),
        )

    def load_extraction_result(self, workflow_id: str) -> ExtractionRunResult:
        path = self._workflow_dir(workflow_id) / EXTRACTION_RESULT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Extraction result output not found for workflow '{workflow_id}'."
            )
        return ExtractionRunResult.model_validate(self._read_json_file(path))

    def save_extraction_run_state(
        self,
        *,
        workflow_id: str,
        state: ExtractionRunState,
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / EXTRACTION_RUN_STATE_FILE,
            state.model_dump(mode="json"),
        )

    def load_extraction_run_state(self, workflow_id: str) -> ExtractionRunState:
        path = self._workflow_dir(workflow_id) / EXTRACTION_RUN_STATE_FILE
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

    def _workflow_dir(self, workflow_id: str) -> Path:
        base_path = self.base_path.resolve()
        workflow_dir = (base_path / workflow_id).resolve()
        if not workflow_dir.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return workflow_dir

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
                prefix=f".{path.name}.",
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
