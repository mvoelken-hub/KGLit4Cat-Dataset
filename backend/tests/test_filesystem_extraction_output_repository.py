import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, call

from app.domain.extraction import (
    EvidenceContext,
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionRunResult,
    ExtractionRunState,
)
from infra.filesystem_extraction_output_repository import (
    FileSystemExtractionOutputRepository,
    _atomic_replace,
)


class FileSystemExtractionOutputRepositoryTests(unittest.TestCase):
    def test_write_json_replaces_existing_file_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"value": "old"}', encoding="utf-8")

            FileSystemExtractionOutputRepository._write_json_file(path, {"value": "new"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "new"})
            self.assertEqual(list(path.parent.glob(".tmp*.tmp")), [])

    def test_model_name_with_colon_creates_valid_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "lfm2.5-thinking:1.2b-bf16"

            run_state = ExtractionRunState(chat_model=chat_model)
            repo.save_extraction_run_state(workflow_id=workflow_id, state=run_state)

            loaded = repo.load_extraction_run_state(workflow_id, chat_model)
            self.assertEqual(loaded.chat_model, chat_model)
            self.assertTrue(
                (repo._workflow_dir(workflow_id, chat_model) / "extraction_run_state.json").exists()
            )

    def test_extraction_result_writes_and_clears_new_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"
            result = ExtractionRunResult(
                generated_final_draft={"id": "generated", "title": "Generated"},
                machine_evidence_context=EvidenceContext(),
                initial_file_summaries=[
                    ExtractionFileSummary(
                        file_path="dataset_description.txt",
                        rank=1,
                        data_format="plain text",
                    )
                ],
                initial_file_summary_status="completed",
                initial_extraction_overview=ExtractionOverview(
                    observed_signals=["dataset_description.txt mentions NMR."],
                    suggested_interpretations=["Use NMR context as orientation only."],
                    conflicts_or_uncertainties=["Instrument identity is unresolved."],
                ),
                initial_extraction_overview_status="structured",
                curated_document={"id": "curated", "title": "Curated"},
                draft_quality_state="imperfect_final_draft",
                validation={"status": "invalid", "errors": [], "warnings": ["weak"]},
                curated_validation={"status": "valid", "errors": [], "warnings": []},
                projection_ledger=[
                    {
                        "object_identifier": "object-1",
                        "object_kind": "Resource",
                        "status": "projected",
                        "projected_paths": ["/title"],
                        "reason": "projected",
                    }
                ],
                field_completion_ledger=[
                    {
                        "json_path": "/title",
                        "field_name": "title",
                        "generated_value": "Generated",
                        "curated_value": "Curated",
                        "validation_status": "valid",
                        "enrichment_status": "not_grounded",
                    }
                ],
                curation_ledger=[
                    {
                        "json_path": "/title",
                        "field_name": "title",
                        "generated_value": "Generated",
                        "curated_value": "Curated",
                        "status": "user_modified",
                    }
                ],
                chat_model=chat_model,
            )

            repo.save_extraction_result(workflow_id=workflow_id, result=result)
            workflow_dir = repo._workflow_dir(workflow_id, chat_model)

            for name in [
                "initial_extraction_overview.json",
                "initial_file_summaries.json",
                "generated_final_draft.json",
                "curated_document.json",
                "projection_ledger.json",
                "field_completion_ledger.json",
                "curation_ledger.json",
                "validation.json",
                "extraction_result.json",
            ]:
                self.assertTrue((workflow_dir / name).exists(), name)

            overview, overview_status = repo.load_initial_extraction_overview(workflow_id, chat_model)
            summaries, summary_status = repo.load_initial_file_summaries(workflow_id, chat_model)
            self.assertEqual(summary_status, "completed")
            self.assertEqual(summaries[0].file_path, "dataset_description.txt")
            self.assertEqual(overview_status, "structured")
            self.assertEqual(overview.observed_signals, ["dataset_description.txt mentions NMR."])
            self.assertEqual(repo.load_generated_final_draft(workflow_id, chat_model)["id"], "generated")
            self.assertEqual(repo.load_curated_document(workflow_id, chat_model)["id"], "curated")
            self.assertEqual(repo.load_projection_ledger(workflow_id, chat_model)[0].status, "projected")
            self.assertEqual(repo.load_field_completion_ledger(workflow_id, chat_model)[0].json_path, "/title")
            self.assertEqual(repo.load_curation_ledger(workflow_id, chat_model)[0].status, "user_modified")
            generated_validation, curated_validation = repo.load_validation(workflow_id, chat_model)
            self.assertEqual(generated_validation.status, "invalid")
            self.assertEqual(curated_validation.status, "valid")

            repo.clear_extraction_downstream(workflow_id)

            summaries, summary_status = repo.load_initial_file_summaries(workflow_id, chat_model)
            overview, overview_status = repo.load_initial_extraction_overview(workflow_id, chat_model)
            self.assertEqual(summary_status, "completed")
            self.assertEqual(summaries[0].file_path, "dataset_description.txt")
            self.assertEqual(overview_status, "structured")
            self.assertEqual(overview.suggested_interpretations, ["Use NMR context as orientation only."])
            with self.assertRaises(FileNotFoundError):
                repo.load_extraction_result(workflow_id, chat_model)
            with self.assertRaises(FileNotFoundError):
                repo.load_generated_final_draft(workflow_id, chat_model)

            repo.clear_extraction_run(workflow_id)

            self.assertFalse((Path(directory) / workflow_id).exists())

    def test_failed_replace_keeps_previous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"value": "old"}', encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace", side_effect=OSError("locked")):
                with self.assertRaises(OSError):
                    FileSystemExtractionOutputRepository._write_json_file(path, {"value": "new"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "old"})
            self.assertEqual(list(path.parent.glob(".tmp*.tmp")), [])


class AtomicReplaceTests(unittest.TestCase):
    def test_atomic_replace_retries_on_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text("{}", encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace") as mock_replace:
                mock_replace.side_effect = [
                    PermissionError("locked"),
                    None,  # succeeds on 2nd attempt
                ]
                with patch("infra.filesystem_extraction_output_repository.time.sleep") as mock_sleep:
                    _atomic_replace(src, dst)

            self.assertEqual(mock_replace.call_count, 2)
            mock_sleep.assert_called_once_with(0.15)

    def test_atomic_replace_raises_after_exhausted_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text("{}", encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace", side_effect=PermissionError("locked")):
                with patch("infra.filesystem_extraction_output_repository.time.sleep"):
                    with self.assertRaises(PermissionError):
                        _atomic_replace(src, dst, retries=3)

    def test_atomic_replace_succeeds_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text('{"ok": true}', encoding="utf-8")

            _atomic_replace(src, dst)

            self.assertTrue(dst.exists())
            self.assertEqual(json.loads(dst.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()
