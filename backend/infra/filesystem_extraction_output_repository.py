import json
from pathlib import Path
from typing import Any

from app.domain.extraction import InitialContext
from app.domain.extraction.artifacts import PatchCandidate
from app.domain.extraction.patch_quality import PatchQualityReport, UnmappedFact


INITIAL_CONTEXT_FILE = "initial_context.json"
INITIAL_DRAFT_FILE = "initial_draft.json"
DRAFT_FILE = "draft.json"
PATCHES_DIR = "patches"
UNMAPPED_FACTS_DIR = "unmapped_facts"
PROTECTED_FIELDS_FILE = "protected_fields.json"
PATCH_REVIEW_STATE_FILE = "patch_review_state.json"


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

    def save_raw_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        raw_file_name = self._with_suffix(patch_file_name, ".raw.json")
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        patch_path = (patch_dir / raw_file_name).resolve()
        if not patch_path.is_relative_to(patch_dir.resolve()):
            raise ValueError(f"Raw patch output path escapes patches directory: {raw_file_name}")
        self._write_json_file(patch_path, patch)

    def save_accepted_patch(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        patch: dict[str, Any],
    ) -> None:
        accepted_file_name = self._with_suffix(patch_file_name, ".accepted.json")
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        patch_path = (patch_dir / accepted_file_name).resolve()
        if not patch_path.is_relative_to(patch_dir.resolve()):
            raise ValueError(f"Accepted patch output path escapes patches directory: {accepted_file_name}")
        self._write_json_file(patch_path, patch)

    def save_candidates(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        candidates: list[PatchCandidate],
    ) -> None:
        candidates_file_name = self._with_suffix(patch_file_name, ".candidates.json")
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        candidates_path = (patch_dir / candidates_file_name).resolve()
        if not candidates_path.is_relative_to(patch_dir.resolve()):
            raise ValueError(f"Candidates output path escapes patches directory: {candidates_file_name}")
        self._write_json_file(
            candidates_path,
            [candidate.model_dump(mode="json") for candidate in candidates],
        )

    def save_quality_report(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        quality_report: PatchQualityReport,
    ) -> None:
        report_file_name = self._with_suffix(patch_file_name, ".quality_report.json")
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        report_path = (patch_dir / report_file_name).resolve()
        if not report_path.is_relative_to(patch_dir.resolve()):
            raise ValueError(f"Quality report output path escapes patches directory: {report_file_name}")
        self._write_json_file(report_path, quality_report.model_dump(mode="json"))

    def save_unmapped_facts(
        self,
        *,
        workflow_id: str,
        patch_file_name: str,
        unmapped_facts: list[UnmappedFact],
    ) -> None:
        facts_file_name = self._with_suffix(patch_file_name, ".unmapped_facts.json")
        facts_dir = self._workflow_dir(workflow_id) / UNMAPPED_FACTS_DIR
        facts_path = (facts_dir / facts_file_name).resolve()
        if not facts_path.is_relative_to(facts_dir.resolve()):
            raise ValueError(f"Unmapped facts output path escapes directory: {facts_file_name}")
        self._write_json_file(
            facts_path,
            [fact.model_dump(mode="json") for fact in unmapped_facts],
        )

    def save_protected_fields(
        self,
        *,
        workflow_id: str,
        protected_fields: list[str],
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / PROTECTED_FIELDS_FILE,
            protected_fields,
        )

    def load_protected_fields(self, workflow_id: str) -> list[str]:
        path = self._workflow_dir(workflow_id) / PROTECTED_FIELDS_FILE
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_patch_review_state(
        self,
        *,
        workflow_id: str,
        review_state: dict[str, Any],
    ) -> None:
        self._write_json_file(
            self._workflow_dir(workflow_id) / PATCH_REVIEW_STATE_FILE,
            review_state,
        )

    def load_patch_review_state(self, workflow_id: str) -> dict[str, Any]:
        path = self._workflow_dir(workflow_id) / PATCH_REVIEW_STATE_FILE
        if not path.exists():
            return {
                "resolved_item_ids": [],
                "unmapped_assignments": {},
                "resolution_notes": {},
                "resolved_at": {},
            }
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return {
            "resolved_item_ids": list(payload.get("resolved_item_ids", [])),
            "unmapped_assignments": dict(payload.get("unmapped_assignments", {})),
            "resolution_notes": dict(payload.get("resolution_notes", {})),
            "resolved_at": dict(payload.get("resolved_at", {})),
        }

    def load_patch_files(self, workflow_id: str) -> list[dict[str, Any]]:
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        if not patch_dir.exists():
            return []

        artifacts: list[dict[str, Any]] = []
        for path in sorted(patch_dir.glob("*.json"), key=lambda item: item.name):
            if path.name.endswith(".quality_report.json"):
                continue
            artifacts.append(
                {
                    "file_name": path.name,
                    "artifact_type": self._patch_artifact_type(path.name),
                    "content": self._read_json_file(path),
                }
            )
        return artifacts

    def load_completed_patch_file_names(self, workflow_id: str) -> set[str]:
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        if not patch_dir.exists():
            return set()

        return {
            path.name
            for path in patch_dir.glob("*.json")
            if self._patch_artifact_type(path.name) == "patch"
        }

    def load_patch_quality_reports(self, workflow_id: str) -> list[dict[str, Any]]:
        patch_dir = self._workflow_dir(workflow_id) / PATCHES_DIR
        if not patch_dir.exists():
            return []

        return [
            {
                "file_name": path.name,
                "content": self._read_json_file(path),
            }
            for path in sorted(
                patch_dir.glob("*.quality_report.json"),
                key=lambda item: item.name,
            )
        ]

    def load_unmapped_facts(self, workflow_id: str) -> list[dict[str, Any]]:
        facts_dir = self._workflow_dir(workflow_id) / UNMAPPED_FACTS_DIR
        if not facts_dir.exists():
            return []

        facts: list[dict[str, Any]] = []
        for path in sorted(
            facts_dir.glob("*.unmapped_facts.json"),
            key=lambda item: item.name,
        ):
            payload = self._read_json_file(path)
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if isinstance(item, dict):
                    facts.append({"file_name": path.name, **item})
                else:
                    facts.append({"file_name": path.name, "fact": item})
        return facts

    def clear_patch_artifacts(self, workflow_id: str) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        draft_path = workflow_dir / DRAFT_FILE
        if draft_path.exists():
            draft_path.unlink()
        patches_dir = workflow_dir / PATCHES_DIR
        if patches_dir.exists():
            import shutil
            shutil.rmtree(patches_dir)
        facts_dir = workflow_dir / UNMAPPED_FACTS_DIR
        if facts_dir.exists():
            import shutil
            shutil.rmtree(facts_dir)
        protected_fields_path = workflow_dir / PROTECTED_FIELDS_FILE
        if protected_fields_path.exists():
            protected_fields_path.unlink()
        review_state_path = workflow_dir / PATCH_REVIEW_STATE_FILE
        if review_state_path.exists():
            review_state_path.unlink()

    @staticmethod
    def _with_suffix(file_name: str, suffix: str) -> str:
        """Replace the .json suffix of *file_name* with *suffix*.

        For example, ``_with_suffix("1_group_patch_1_5.json", ".raw.json")``
        returns ``"1_group_patch_1_5.raw.json"``.
        """
        if file_name.endswith(".json"):
            stem = file_name[:-len(".json")]
            return f"{stem}{suffix}"
        return f"{file_name}{suffix}"

    def _workflow_dir(self, workflow_id: str) -> Path:
        base_path = self.base_path.resolve()
        workflow_dir = (base_path / workflow_id).resolve()
        if not workflow_dir.is_relative_to(base_path):
            raise ValueError(f"Workflow output path escapes output directory: {workflow_id}")
        return workflow_dir

    @staticmethod
    def _patch_artifact_type(file_name: str) -> str:
        if file_name.endswith(".raw.json"):
            return "raw_patch"
        if file_name.endswith(".accepted.json"):
            return "accepted_patch"
        if file_name.endswith(".candidates.json"):
            return "candidates"
        return "patch"

    @staticmethod
    def _read_json_file(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json_file(path: Path, content: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
