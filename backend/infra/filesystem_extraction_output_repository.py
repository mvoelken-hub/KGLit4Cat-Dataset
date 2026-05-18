import json
from pathlib import Path
from typing import Any

from app.domain.extraction import InitialContext


INITIAL_CONTEXT_FILE = "initial_context.json"
INITIAL_DRAFT_FILE = "initial_draft.json"
DRAFT_FILE = "draft.json"
PATCHES_DIR = "patches"


class FileSystemExtractionOutputRepository:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_initial_context(
        self,
        *,
        workflow_id: str,
        initial_context: InitialContext,
    ) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        workflow_dir.mkdir(parents=True, exist_ok=True)
        with open(workflow_dir / INITIAL_CONTEXT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                initial_context.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_initial_context(self, workflow_id: str) -> InitialContext:
        path = self._workflow_dir(workflow_id) / INITIAL_CONTEXT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Initial context output not found for workflow '{workflow_id}'."
            )
        with open(path, "r", encoding="utf-8") as file:
            return InitialContext.model_validate(json.load(file))

    def save_initial_draft(
        self,
        *,
        workflow_id: str,
        initial_draft: dict[str, Any],
    ) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        workflow_dir.mkdir(parents=True, exist_ok=True)
        with open(workflow_dir / INITIAL_DRAFT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                initial_draft,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_initial_draft(self, workflow_id: str) -> dict[str, Any]:
        path = self._workflow_dir(workflow_id) / INITIAL_DRAFT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Initial draft output not found for workflow '{workflow_id}'."
            )
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_draft(
        self,
        *,
        workflow_id: str,
        draft: dict[str, Any],
    ) -> None:
        self._write_json_file(self._workflow_dir(workflow_id) / DRAFT_FILE, draft)

    def load_draft(self, workflow_id: str) -> dict[str, Any]:
        path = self._workflow_dir(workflow_id) / DRAFT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Draft output not found for workflow '{workflow_id}'."
            )
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        patch_path = (patch_dir / patch_file_name).resolve()
        if not patch_path.is_relative_to(patch_dir.resolve()):
            raise ValueError(f"Patch output path escapes patches directory: {patch_file_name}")
        self._write_json_file(patch_path, patch)

    def _workflow_dir(self, workflow_id: str) -> Path:
        base_path = self.base_path.resolve()
        workflow_dir = (base_path / workflow_id).resolve()
        if not workflow_dir.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return workflow_dir

    @staticmethod
    def _write_json_file(path: Path, content: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
