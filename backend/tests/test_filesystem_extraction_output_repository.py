import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, call

from app.domain.extraction import (
    EvidenceContext,
    EvidenceCandidate,
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionRunResult,
    ExtractionRunState,
    FilteredEvidenceLedger,
    FilteredEvidenceNote,
    InitialFileSummaryDiagnosticRecord,
    InitialFileSummaryDiagnostics,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
)
from infra.filesystem_extraction_output_repository import (
    FileSystemExtractionOutputRepository,
    _atomic_replace,
)


def overview_for_file(file_path: str, summary: str, uncertainty: str) -> ExtractionOverview:
    return ExtractionOverview(
        source_file_paths=[file_path],
        nodes=[
            {
                "node_id": f"file:{file_path}",
                "label": file_path,
                "kind": "file",
                "file_path": file_path,
                "summary": summary,
            }
        ],
        edges=[],
        uncertainties=[uncertainty],
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
                (
                    repo._workflow_dir(workflow_id)
                    / "run_state"
                    / "semantic"
                    / "lfm2.5-thinking_1.2b-bf16"
                    / "extraction_run_state.json"
                ).exists()
            )

    def test_prompt_diagnostics_append_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"

            repo.append_prompt_diagnostic(
                workflow_id=workflow_id,
                chat_model=chat_model,
                diagnostic={
                    "operation_id": "chunk/a",
                    "agent_name": "chunk_extraction",
                    "attempts": [],
                },
            )
            repo.append_prompt_diagnostic(
                workflow_id=workflow_id,
                chat_model=chat_model,
                diagnostic={
                    "operation_id": "chunk/a",
                    "agent_name": "chunk_extraction",
                    "attempts": [],
                },
            )

            diagnostics_dir = repo._branch_dir(workflow_id, "evidence_notes", "semantic", chat_model) / "prompts"
            files = sorted(diagnostics_dir.glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertTrue(files[0].name.endswith("__0001.json"))
            self.assertTrue(files[1].name.endswith("__0002.json"))
            index = json.loads((repo._workflow_dir(workflow_id) / "artifact_index.json").read_text(encoding="utf-8"))
            prompt_paths = index["by_stage"]["evidence_notes"]
            self.assertEqual(len(prompt_paths), 2)
            self.assertTrue(all("evidence_notes/semantic/model_tag/prompts/" in path for path in prompt_paths))

            repo.clear_prompt_diagnostics(workflow_id, chat_model)

            self.assertFalse(diagnostics_dir.exists())
            index = json.loads((repo._workflow_dir(workflow_id) / "artifact_index.json").read_text(encoding="utf-8"))
            self.assertNotIn("evidence_notes", index["by_stage"])

    def test_prompt_diagnostics_clear_can_target_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"

            repo.append_prompt_diagnostic(
                workflow_id=workflow_id,
                chat_model=chat_model,
                chunking_strategy="fixed_tokens",
                diagnostic={
                    "operation_id": "provenance_core_constructor",
                    "agent_name": "profile_projection",
                    "attempts": [],
                },
            )
            repo.append_prompt_diagnostic(
                workflow_id=workflow_id,
                chat_model=chat_model,
                chunking_strategy="fixed_tokens",
                diagnostic={
                    "operation_id": "chunk/a",
                    "agent_name": "chunk_extraction",
                    "attempts": [],
                },
            )

            profile_prompts = repo._branch_dir(
                workflow_id, "profile_draft", "fixed_tokens", chat_model
            ) / "prompts"
            evidence_prompts = repo._branch_dir(
                workflow_id, "evidence_notes", "fixed_tokens", chat_model
            ) / "prompts"
            self.assertTrue(profile_prompts.exists())
            self.assertTrue(evidence_prompts.exists())

            repo.clear_prompt_diagnostics(
                workflow_id,
                chat_model=chat_model,
                chunking_strategy="fixed_tokens",
                stage="profile_draft",
            )

            self.assertFalse(profile_prompts.exists())
            self.assertTrue(evidence_prompts.exists())
            index = json.loads((repo._workflow_dir(workflow_id) / "artifact_index.json").read_text(encoding="utf-8"))
            self.assertNotIn("profile_draft", index["by_stage"])
            self.assertIn("evidence_notes", index["by_stage"])

    def test_extraction_result_writes_and_clears_new_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"
            result = ExtractionRunResult(
                generated_final_draft={"id": "generated", "title": "Generated"},
                machine_evidence_context=EvidenceContext(),
                generated_core_draft={"id": "core", "title": "Core"},
                generated_attribute_draft={"id": "attribute", "title": "Attribute"},
                generated_reconstructed_draft={"id": "generated", "title": "Generated"},
                initial_file_summaries=[
                    ExtractionFileSummary(
                        file_path="dataset_description.txt",
                        data_format="plain text",
                    )
                ],
                initial_file_summary_status="completed",
                initial_extraction_overview=overview_for_file(
                    "dataset_description.txt",
                    "dataset_description.txt mentions NMR.",
                    "Instrument identity is unresolved.",
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
                parent_attribute_ledger=[
                    {
                        "parent_path": "/was_generated_by/0",
                        "parent_class": "DataGeneratingActivity",
                        "status": "skipped_schema_missing",
                        "reason": "No schema branch was available.",
                        "target_path": "/was_generated_by/0/has_quantitative_attribute/-",
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
                evidence_query_ledger=[
                    {
                        "query_id": "requirement_query:method_plan:test",
                        "requirement_id": "method_plan",
                        "target_path": "/was_generated_by/0/realized_plan",
                        "query": {"text": "pulse sequence"},
                        "result_evidence_ids": ["ev:one", "ev:two"],
                        "selected_evidence_ids": ["ev:one"],
                        "rejected_result_reasons": {"ev:two": "context_window_only"},
                        "ranking_explanation": ["hint match"],
                    }
                ],
                curation_ledger=[
                    {
                        "json_path": "/title",
                        "field_name": "title",
                        "generated_value": "Generated",
                        "curated_value": "Curated",
                        "status": "auto_modified",
                    }
                ],
                chat_model=chat_model,
            )

            repo.save_extraction_result(workflow_id=workflow_id, result=result)
            repo.save_dataset_summary(
                workflow_id=workflow_id,
                summary="evaluated: dataset. generated_by: measurement.",
                chat_model=chat_model,
            )
            overview_dir = repo._overview_dir(workflow_id, chat_model)
            profile_dir = repo._branch_dir(workflow_id, "profile_draft", "semantic", chat_model)
            result_dir = repo._result_dir(workflow_id, "semantic", chat_model)

            for path in [
                overview_dir / "initial_extraction_overview.json",
                overview_dir / "initial_file_summaries.json",
                profile_dir / "generated_core_draft.json",
                profile_dir / "generated_attribute_draft.json",
                profile_dir / "generated_reconstructed_draft.json",
                profile_dir / "dataset_summary.txt",
                result_dir / "curated_document.json",
                profile_dir / "projection_ledger.json",
                profile_dir / "parent_attribute_ledger.json",
                profile_dir / "field_completion_ledger.json",
                profile_dir / "evidence_query_ledger.json",
                result_dir / "curation_ledger.json",
                profile_dir / "validation.json",
                result_dir / "extraction_result.json",
            ]:
                self.assertTrue(path.exists(), str(path))
            safe_model = repo._sanitize_path_component(chat_model)
            index = json.loads((repo._workflow_dir(workflow_id) / "artifact_index.json").read_text(encoding="utf-8"))
            self.assertIn(f"overview/{safe_model}/initial_file_summaries.json", index["by_stage"]["overview"])
            self.assertIn(f"profile_draft/semantic/{safe_model}/generated_reconstructed_draft.json", index["by_stage"]["profile_draft"])
            self.assertIn(f"profile_draft/semantic/{safe_model}/generated_core_draft.json", index["by_stage"]["profile_draft"])
            self.assertIn(f"profile_draft/semantic/{safe_model}/generated_attribute_draft.json", index["by_stage"]["profile_draft"])
            self.assertIn(f"result/semantic/{safe_model}/extraction_result.json", index["by_stage"]["result"])

            overview, overview_status = repo.load_initial_extraction_overview(workflow_id, chat_model)
            summaries, summary_status = repo.load_initial_file_summaries(workflow_id, chat_model)
            self.assertEqual(summary_status, "completed")
            self.assertEqual(summaries[0].file_path, "dataset_description.txt")
            self.assertEqual(overview_status, "structured")
            self.assertEqual(overview.nodes[0].summary, "dataset_description.txt mentions NMR.")
            self.assertEqual(repo.load_generated_final_draft(workflow_id, chat_model)["id"], "generated")
            self.assertEqual(repo.load_curated_document(workflow_id, chat_model)["id"], "curated")
            self.assertEqual(repo.load_projection_ledger(workflow_id, chat_model)[0].status, "projected")
            self.assertEqual(repo.load_parent_attribute_ledger(workflow_id, chat_model)[0].status, "skipped_schema_missing")
            self.assertEqual(repo.load_field_completion_ledger(workflow_id, chat_model)[0].json_path, "/title")
            self.assertEqual(
                repo.load_evidence_query_ledger(workflow_id, chat_model)[0].selected_evidence_ids,
                ["ev:one"],
            )
            self.assertEqual(repo.load_curation_ledger(workflow_id, chat_model)[0].status, "auto_modified")
            generated_validation, curated_validation = repo.load_validation(workflow_id, chat_model)
            self.assertEqual(generated_validation.status, "invalid")
            self.assertEqual(curated_validation.status, "valid")

            repo.clear_extraction_downstream(workflow_id)

            summaries, summary_status = repo.load_initial_file_summaries(workflow_id, chat_model)
            overview, overview_status = repo.load_initial_extraction_overview(workflow_id, chat_model)
            self.assertEqual(summary_status, "completed")
            self.assertEqual(summaries[0].file_path, "dataset_description.txt")
            self.assertEqual(overview_status, "structured")
            self.assertEqual(overview.uncertainties, ["Instrument identity is unresolved."])
            with self.assertRaises(FileNotFoundError):
                repo.load_extraction_result(workflow_id, chat_model)
            with self.assertRaises(FileNotFoundError):
                repo.load_generated_final_draft(workflow_id, chat_model)
            self.assertFalse((profile_dir / "dataset_summary.txt").exists())
            self.assertEqual(repo.load_evidence_query_ledger(workflow_id, chat_model), [])

            repo.clear_extraction_run(workflow_id)

            self.assertFalse((Path(directory) / workflow_id).exists())

    def test_clear_extraction_run_ignores_missing_child_during_tree_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            workflow_dir = repo._workflow_dir(workflow_id)
            diagnostics_dir = workflow_dir / "model" / "prompt_diagnostics"
            diagnostics_dir.mkdir(parents=True)
            (diagnostics_dir / "diagnostic.json").write_text("{}", encoding="utf-8")

            def simulate_missing_child(path, *, onexc):
                onexc(
                    lambda _path: None,
                    str(diagnostics_dir / "diagnostic.json"),
                    FileNotFoundError(),
                )

            with patch(
                "infra.filesystem_extraction_output_repository.shutil.rmtree",
                side_effect=simulate_missing_child,
            ) as rmtree:
                repo.clear_extraction_run(workflow_id)

            rmtree.assert_called_once()

    def test_clear_initial_context_keeps_downstream_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"

            repo.save_initial_file_summaries(
                workflow_id=workflow_id,
                summaries=[ExtractionFileSummary(file_path="README.md")],
                status="completed",
                chat_model=chat_model,
            )
            repo.save_initial_extraction_overview(
                workflow_id=workflow_id,
                overview=overview_for_file("README.md", "README present.", ""),
                status="structured",
                chat_model=chat_model,
            )
            repo.save_dataset_summary(workflow_id=workflow_id, summary="summary", chat_model=chat_model)
            repo.save_generated_core_draft(workflow_id=workflow_id, document={"id": "core"}, chat_model=chat_model)
            repo.save_generated_attribute_draft(workflow_id=workflow_id, document={"id": "attribute"}, chat_model=chat_model)
            repo.save_generated_final_draft(workflow_id=workflow_id, document={"id": "draft"}, chat_model=chat_model)
            repo.save_curated_document(workflow_id=workflow_id, document={"id": "curated"}, chat_model=chat_model)

            repo.clear_initial_context(workflow_id)

            with self.assertRaises(FileNotFoundError):
                repo.load_initial_file_summaries(workflow_id, chat_model)
            with self.assertRaises(FileNotFoundError):
                repo.load_initial_extraction_overview(workflow_id, chat_model)
            self.assertEqual(repo.load_generated_final_draft(workflow_id, chat_model)["id"], "draft")
            self.assertEqual(repo.load_curated_document(workflow_id, chat_model)["id"], "curated")
            self.assertFalse(
                (
                    repo._branch_dir(workflow_id, "profile_draft", "semantic", chat_model)
                    / "dataset_summary.txt"
                ).exists()
            )

    def test_filtered_evidence_notes_artifact_round_trips_and_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            ledger = FilteredEvidenceLedger(
                filtered_notes=[
                    FilteredEvidenceNote(
                        reason="candidate_rejected",
                        note=EvidenceCandidate(
                            candidate_id="low",
                            category="method_signal",
                            role="descriptor",
                            claim="Low-level parameter.",
                            evidence_text="parameter",
                        ),
                        file_path="acqu",
                        chunk_index=3,
                    )
                ],
                summary={"candidate_rejected": 1},
            )

            repo.save_filtered_evidence_notes(workflow_id=workflow_id, ledger=ledger)

            loaded = repo.load_filtered_evidence_notes(workflow_id)
            self.assertEqual(loaded.summary, {"candidate_rejected": 1})
            self.assertEqual(loaded.filtered_notes[0].note.candidate_id, "low")

            repo.clear_extraction_downstream(workflow_id)

            self.assertEqual(repo.load_filtered_evidence_notes(workflow_id).filtered_notes, [])

    def test_initial_overview_diagnostic_writes_and_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"
            diagnostic = InitialOverviewFailureDiagnostic(
                error_type="MaxRetriesExceeded",
                message="Max retries exceeded",
                last_error_type="OutputParsingError",
                last_error="bad json",
                failed_response_excerpt='{"nodes": [',
                prompt_budget={"total_input_tokens": 2000},
                usage={"requests": 2},
            )

            repo.save_initial_extraction_overview_diagnostic(
                workflow_id=workflow_id,
                diagnostic=diagnostic,
                chat_model=chat_model,
            )

            path = (
                repo._overview_dir(workflow_id, chat_model)
                / "initial_extraction_overview_diagnostic.json"
            )
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["error_type"], "MaxRetriesExceeded")
            self.assertEqual(payload["prompt_budget"]["total_input_tokens"], 2000)

            repo.save_initial_extraction_overview_diagnostic(
                workflow_id=workflow_id,
                diagnostic=None,
                chat_model=chat_model,
            )

            self.assertFalse(path.exists())

    def test_initial_overview_success_diagnostic_writes_prompt_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"
            diagnostic = InitialOverviewPromptDiagnostic(
                prompt_budget={
                    "total_input_tokens": 1800,
                    "included_summary_paths": ["dataset_description.txt"],
                    "dropped_summary_paths": ["raw.dat"],
                },
                included_summary_paths=["dataset_description.txt"],
                dropped_summary_paths=["raw.dat"],
                included_ranked_paths=["dataset_description.txt"],
                dropped_ranked_paths=[],
                hard_truncated=False,
            )

            repo.save_initial_extraction_overview_diagnostic(
                workflow_id=workflow_id,
                diagnostic=diagnostic,
                chat_model=chat_model,
            )

            path = (
                repo._overview_dir(workflow_id, chat_model)
                / "initial_extraction_overview_diagnostic.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "structured_success")
            self.assertEqual(payload["included_summary_paths"], ["dataset_description.txt"])
            self.assertEqual(payload["dropped_summary_paths"], ["raw.dat"])
            self.assertEqual(payload["prompt_budget"]["total_input_tokens"], 1800)

    def test_initial_file_summary_diagnostics_write_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            workflow_id = "test_workflow"
            chat_model = "model:tag"
            diagnostics = InitialFileSummaryDiagnostics(
                records=[
                    InitialFileSummaryDiagnosticRecord(
                        file_path="metadata.txt",
                        reason="unsupported_explicit_purpose",
                        message="Purpose was cleared.",
                        details={"explicit_purpose": "Dataset title"},
                    )
                ]
            )

            repo.save_initial_file_summary_diagnostics(
                workflow_id=workflow_id,
                diagnostics=diagnostics,
                chat_model=chat_model,
            )

            path = (
                repo._overview_dir(workflow_id, chat_model)
                / "initial_file_summary_diagnostics.json"
            )
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"][0]["reason"], "unsupported_explicit_purpose")

            repo.save_initial_file_summary_diagnostics(
                workflow_id=workflow_id,
                diagnostics=None,
                chat_model=chat_model,
            )

            self.assertFalse(path.exists())

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

