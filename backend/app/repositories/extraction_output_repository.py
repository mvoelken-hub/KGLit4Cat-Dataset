from typing import Protocol

from app.domain.extraction import ExtractionContext, ExtractionRunResult


class ExtractionOutputRepository(Protocol):
    def save_extraction_context(
        self,
        *,
        workflow_id: str,
        extraction_context: ExtractionContext,
    ) -> None:
        ...

    def load_extraction_context(self, workflow_id: str) -> ExtractionContext:
        ...

    def save_extraction_result(
        self,
        *,
        workflow_id: str,
        result: ExtractionRunResult,
    ) -> None:
        ...

    def load_extraction_result(self, workflow_id: str) -> ExtractionRunResult:
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

    def clear_extraction_run(self, workflow_id: str) -> None:
        ...

