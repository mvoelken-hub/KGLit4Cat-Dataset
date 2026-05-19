from typing import Any, Protocol

from app.domain.extraction import InitialContext
from app.domain.extraction.artifacts import PatchCandidate
from app.domain.extraction.patch_quality import PatchQualityReport, UnmappedFact


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

    def save_raw_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        ...

    def save_accepted_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        ...

    def save_candidates(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        candidates: list[PatchCandidate],
    ) -> None:
        ...

    def save_quality_report(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        quality_report: PatchQualityReport,
    ) -> None:
        ...

    def save_unmapped_facts(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        unmapped_facts: list[UnmappedFact],
    ) -> None:
        ...
