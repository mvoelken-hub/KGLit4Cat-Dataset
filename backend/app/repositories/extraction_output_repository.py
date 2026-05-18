from typing import Any, Protocol

from app.domain.extraction import InitialContext


class ExtractionOutputRepository(Protocol):
    def save_initial_context(
        self,
        *,
        workflow_id: str,
        initial_context: InitialContext,
    ) -> None:
        ...

    def load_initial_context(self, workflow_id: str) -> InitialContext:
        ...

    def save_initial_draft(
        self,
        *,
        workflow_id: str,
        initial_draft: dict[str, Any],
    ) -> None:
        ...

    def load_initial_draft(self, workflow_id: str) -> dict[str, Any]:
        ...

    def save_draft(
        self,
        *,
        workflow_id: str,
        draft: dict[str, Any],
    ) -> None:
        ...

    def load_draft(self, workflow_id: str) -> dict[str, Any]:
        ...

    def save_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        ...
