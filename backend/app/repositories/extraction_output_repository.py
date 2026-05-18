from typing import Protocol

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
