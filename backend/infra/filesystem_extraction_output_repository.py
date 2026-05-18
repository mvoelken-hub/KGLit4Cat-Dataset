import json
from pathlib import Path

from app.domain.extraction import InitialContext


INITIAL_CONTEXT_FILE = "initial_context.json"


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

    def _workflow_dir(self, workflow_id: str) -> Path:
        base_path = self.base_path.resolve()
        workflow_dir = (base_path / workflow_id).resolve()
        if not workflow_dir.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return workflow_dir
